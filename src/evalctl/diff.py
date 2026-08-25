"""Comparing two runs, or two models within one run.

A selector is ``<run>[:<model>]``:

    evalctl diff latest prev            two runs, matching model ids
    evalctl diff run-a:candidate run-a:baseline    two models in one run
    evalctl diff run-b:new run-a:old               any pairing you like

Comparability is checked, not assumed
-------------------------------------
The most expensive mistake available here is diffing two runs that measured
different things -- someone edited a case, or the suites diverged -- and
reporting the difference as a model change. Both sides carry a suite
fingerprint in their manifest, and per-task fingerprints under it. The diff
compares them, tells you exactly which tasks differ, and joins on the cases the
two runs genuinely share. It reports rather than refuses, because a partial
comparison is often what you want; it just must not be silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .analysis import CaseOutcome, ClusterBy, ErrorPolicy, Metric, collapse_cases, observations
from .errors import RunNotFound
from .stats import DEFAULT_LEVEL, DEFAULT_RESAMPLES, DiffResult, paired_diff
from .store import RunStore, TrialRecord


@dataclass(frozen=True)
class Side:
    """One half of a comparison, already resolved to records."""

    selector: str
    run_id: str
    model_id: str
    store: RunStore
    records: tuple[TrialRecord, ...]
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.run_id[:22]}:{self.model_id}"


def parse_selector(selector: str) -> tuple[str, str | None]:
    """Split ``run[:model]``.

    Windows paths carry a drive colon, so only the last colon counts, and only
    when what follows it is not a path fragment.
    """
    if ":" not in selector:
        return selector, None
    head, _, tail = selector.rpartition(":")
    if not head or not tail or "/" in tail or "\\" in tail:
        return selector, None
    return head, tail


def resolve_side(selector: str, *, runs_dir: str = "runs") -> Side:
    run_selector, model_id = parse_selector(selector)
    store = RunStore.open(run_selector, runs_dir)
    records = store.all_records()
    if not records:
        raise RunNotFound(f"run '{store.run_id}' has no records")

    available = sorted({r.model_id for r in records})
    if model_id is None:
        if len(available) != 1:
            raise RunNotFound(
                f"run '{store.run_id}' contains {len(available)} models ({', '.join(available)}); "
                f"pick one with '{run_selector}:<model>'"
            )
        model_id = available[0]
    elif model_id not in available:
        raise RunNotFound(
            f"model '{model_id}' is not in run '{store.run_id}'. Available: {', '.join(available)}"
        )

    return Side(
        selector=selector,
        run_id=store.run_id,
        model_id=model_id,
        store=store,
        records=tuple(r for r in records if r.model_id == model_id),
        manifest=store.manifest(),
    )


def comparability_warnings(a: Side, b: Side) -> list[str]:
    """Everything that makes this diff less trustworthy than it looks."""
    warnings: list[str] = []
    suite_a = (a.manifest.get("suite") or {})
    suite_b = (b.manifest.get("suite") or {})

    fingerprint_a, fingerprint_b = suite_a.get("fingerprint"), suite_b.get("fingerprint")
    if fingerprint_a and fingerprint_b and fingerprint_a != fingerprint_b:
        warnings.append(
            "the two runs used different suite content "
            f"({fingerprint_a[:12]} vs {fingerprint_b[:12]}); only shared cases are compared"
        )
        changed = _changed_tasks(a.manifest, b.manifest)
        if changed:
            warnings.append(f"task definitions differ for: {', '.join(changed[:8])}")

    policy_a, policy_b = suite_a.get("errors"), suite_b.get("errors")
    if policy_a and policy_b and policy_a != policy_b:
        warnings.append(
            f"the runs used different error policies ('{policy_a}' vs '{policy_b}'), which "
            "changes how failed trials enter each score"
        )

    model_a = _model_entry(a.manifest, a.model_id)
    model_b = _model_entry(b.manifest, b.model_id)
    if model_a and model_b:
        if model_a.get("model") == model_b.get("model") and model_a.get("params") != model_b.get("params"):
            warnings.append(
                f"same model '{model_a.get('model')}' on both sides but different sampling "
                f"params: {model_a.get('params')} vs {model_b.get('params')}"
            )
        if a.run_id == b.run_id and a.model_id == b.model_id:
            warnings.append("both sides refer to the same run and model; the delta will be zero")

    version_a = a.manifest.get("evalctl_version")
    version_b = b.manifest.get("evalctl_version")
    if version_a and version_b and version_a != version_b:
        warnings.append(f"runs produced by different evalctl versions ({version_a} vs {version_b})")

    errors_a = sum(1 for r in a.records if r.status != "ok")
    errors_b = sum(1 for r in b.records if r.status != "ok")
    for label, errors, records in (("A", errors_a, a.records), ("B", errors_b, b.records)):
        if records and errors / len(records) > 0.05:
            warnings.append(
                f"side {label} has a {errors / len(records):.0%} error rate; the comparison is "
                "measuring reliability as much as quality"
            )
    return warnings


def _model_entry(manifest: Mapping[str, Any], model_id: str) -> Mapping[str, Any] | None:
    for entry in manifest.get("models") or []:
        if isinstance(entry, Mapping) and entry.get("id") == model_id:
            return entry
    return None


def _changed_tasks(manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any]) -> list[str]:
    tasks_a = {t.get("id"): t.get("fingerprint") for t in (manifest_a.get("tasks") or [])}
    tasks_b = {t.get("id"): t.get("fingerprint") for t in (manifest_b.get("tasks") or [])}
    changed = [tid for tid, fp in tasks_a.items() if tid in tasks_b and tasks_b[tid] != fp]
    changed += [f"{tid} (only in A)" for tid in tasks_a if tid not in tasks_b]
    changed += [f"{tid} (only in B)" for tid in tasks_b if tid not in tasks_a]
    return sorted(str(c) for c in changed)


@dataclass
class DiffReport:
    a: Side
    b: Side
    metric: str
    cluster_by: str
    result: DiffResult
    warnings: list[str]
    case_deltas: list[tuple[str, float, float, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": {"run_id": self.a.run_id, "model_id": self.a.model_id, "selector": self.a.selector},
            "b": {"run_id": self.b.run_id, "model_id": self.b.model_id, "selector": self.b.selector},
            "metric": self.metric,
            "cluster_by": self.cluster_by,
            "warnings": self.warnings,
            "result": self.result.as_dict(),
            "case_deltas": [
                {"case": unit, "a": round(av, 6), "b": round(bv, 6), "delta": round(d, 6)}
                for unit, av, bv, d in sorted(self.case_deltas, key=lambda x: -abs(x[3]))
            ],
        }


def _outcome_map(outcomes: Sequence[CaseOutcome], metric: Metric) -> dict[str, float]:
    return {o.unit: (o.score if metric == "score" else o.passed) for o in outcomes}


def diff_runs(
    selector_a: str,
    selector_b: str,
    *,
    runs_dir: str = "runs",
    metric: Metric = "score",
    error_policy: ErrorPolicy | None = None,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    cluster_by: ClusterBy = "task",
) -> DiffReport:
    """Compare A against B. A positive delta means A is better."""
    a = resolve_side(selector_a, runs_dir=runs_dir)
    b = resolve_side(selector_b, runs_dir=runs_dir)

    if error_policy is None:
        # Default to whatever the runs themselves declared, and fall back to
        # the stricter reading when they disagree or say nothing.
        policies = {
            (a.manifest.get("suite") or {}).get("errors"),
            (b.manifest.get("suite") or {}).get("errors"),
        } - {None}
        error_policy = policies.pop() if len(policies) == 1 else "zero"  # type: ignore[assignment]

    outcomes_a = collapse_cases(a.records, error_policy=error_policy)
    outcomes_b = collapse_cases(b.records, error_policy=error_policy)

    obs_a = observations(outcomes_a, metric, cluster_by)
    obs_b = observations(outcomes_b, metric, cluster_by)

    result = paired_diff(
        obs_a,
        obs_b,
        level=level,
        resamples=resamples,
        seed=seed,
        binary=(metric == "pass"),
    )

    map_a = _outcome_map(outcomes_a, metric)
    map_b = _outcome_map(outcomes_b, metric)
    case_deltas = [
        (unit, map_a[unit], map_b[unit], map_a[unit] - map_b[unit])
        for unit in sorted(set(map_a) & set(map_b))
        if map_a[unit] != map_b[unit]
    ]

    return DiffReport(
        a=a,
        b=b,
        metric=metric,
        cluster_by=cluster_by,
        result=result,
        warnings=comparability_warnings(a, b),
        case_deltas=case_deltas,
    )
