"""Command line interface.

Exit codes are part of the contract, because the main reason to run this in CI
is to gate on the result:

    0  success
    1  something went wrong (bad spec, unreachable provider, missing run)
    2  usage error
    3  the run completed but failed a gate (``--fail-under``,
       ``--fail-on-regression``)

Separating 3 from 1 is what lets a pipeline tell "the eval says no" apart from
"the eval did not run", which are very different pages to wake someone up for.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from . import providers as provider_registry
from .analysis import analyze_run
from .cache import NullCache, ResponseCache, default_cache_path
from .diff import diff_runs
from .errors import EvalctlError, RunNotFound, SpecError
from .report import CHARS as report_charset
from .report import (
    bullet_list,
    clear_progress,
    color_enabled,
    duration,
    format_case_deltas,
    format_diff_report,
    format_run_report,
    money,
    pct,
    progress_line,
    render_table,
    style,
    to_json,
    write_progress,
)
from .runner import Runner, RunProgress, dry_run, filter_suite
from .scorers import available_scorers
from .spec import Suite, load_suite
from .stats import DEFAULT_LEVEL, DEFAULT_RESAMPLES
from .store import DEFAULT_RUNS_DIR, RunStore, build_manifest, list_runs
from .util import run_id_for

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_GATE = 0, 1, 2, 3

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])?\s*$", re.IGNORECASE)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> float:
    """'30m', '7d', '90' (seconds) -> seconds."""
    match = _DURATION.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"cannot read duration '{text}'; use forms like 90, 30m, 12h, 7d"
        )
    value, unit = match.groups()
    return float(value) * _UNITS[(unit or "s").lower()]


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalctl",
        description="Run LLM benchmark suites, score them, and tell real changes from noise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  evalctl run examples/suites/demo.yaml\n"
            "  evalctl run suite.yaml --model candidate --limit 5 --dry-run\n"
            "  evalctl report latest\n"
            "  evalctl diff latest:candidate latest:baseline\n"
            "  evalctl show latest --failures -n 5\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"evalctl {__version__}")
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="where run artifacts live (default: runs)"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # -- run ---------------------------------------------------------------
    run = sub.add_parser("run", help="execute a suite")
    run.add_argument("suite", help="path to a suite YAML file")
    run.add_argument("--model", action="append", dest="models", metavar="ID",
                     help="only these model ids (repeatable, globs allowed)")
    run.add_argument("--task", action="append", dest="tasks", metavar="ID",
                     help="only these task ids (repeatable, globs allowed)")
    run.add_argument("--tag", action="append", dest="tags", metavar="TAG",
                     help="only tasks/cases carrying this tag")
    run.add_argument("--limit", type=int, metavar="N", help="first N cases per task")
    run.add_argument("--repeats", type=int, metavar="N", help="override suite repeats")
    run.add_argument("--concurrency", type=int, metavar="N", help="override max concurrency")
    run.add_argument("--errors", choices=("zero", "exclude"), help="override the error policy")
    run.add_argument("--no-cache", action="store_true", help="ignore the cache entirely")
    run.add_argument("--refresh-cache", action="store_true",
                     help="re-generate every response and overwrite the cached copies")
    run.add_argument("--cache-ttl", type=parse_duration, metavar="DUR",
                     help="treat cached responses older than this as missing (e.g. 7d)")
    run.add_argument("--cache-path", metavar="PATH", help="cache file (default: .evalctl/cache.sqlite)")
    run.add_argument("--resume", nargs="?", const="latest", metavar="RUN",
                     help="continue an interrupted run, skipping completed trials")
    run.add_argument("--dry-run", action="store_true",
                     help="render prompts and estimate cost without calling anything")
    run.add_argument("--fail-under", type=float, metavar="X",
                     help="exit 3 if any model scores below X (0-1)")
    run.add_argument("--no-progress", action="store_true")
    _add_output_flags(run)
    _add_stats_flags(run)

    # -- report ------------------------------------------------------------
    report = sub.add_parser("report", help="re-render the report for a finished run")
    report.add_argument("run", nargs="?", default="latest", help="run id, prefix, path, or 'latest'")
    report.add_argument("--errors", choices=("zero", "exclude"), help="override the error policy")
    report.add_argument("--no-tasks", action="store_true", help="omit the per-task breakdown")
    _add_output_flags(report)
    _add_stats_flags(report)

    # -- diff --------------------------------------------------------------
    diff = sub.add_parser("diff", help="compare two runs (or two models) with a paired bootstrap")
    diff.add_argument("a", help="candidate selector: <run>[:<model>]")
    diff.add_argument("b", help="baseline selector: <run>[:<model>]")
    diff.add_argument("--metric", choices=("score", "pass"), default="score")
    diff.add_argument("--errors", choices=("zero", "exclude"))
    diff.add_argument("--cases", type=int, default=15, metavar="N",
                      help="how many per-case changes to list (0 to hide)")
    diff.add_argument("--fail-on-regression", action="store_true",
                      help="exit 3 if A is significantly worse than B")
    _add_output_flags(diff)
    _add_stats_flags(diff)

    # -- ls ----------------------------------------------------------------
    ls = sub.add_parser("ls", help="list recorded runs")
    ls.add_argument("-n", "--limit", type=int, default=20)
    _add_output_flags(ls)

    # -- show --------------------------------------------------------------
    show = sub.add_parser("show", help="inspect individual trials")
    show.add_argument("run", nargs="?", default="latest")
    show.add_argument("--model", dest="model_id")
    show.add_argument("--task", dest="task_id")
    show.add_argument("--case", dest="case_id")
    show.add_argument("--failures", action="store_true", help="only trials that did not pass")
    show.add_argument("--errors-only", action="store_true", help="only trials that errored")
    show.add_argument("--disagreement", action="store_true",
                      help="sort by judge disagreement, highest first")
    show.add_argument("-n", "--limit", type=int, default=10)
    show.add_argument("--full", action="store_true", help="do not truncate responses")
    _add_output_flags(show)

    # -- validate ----------------------------------------------------------
    validate = sub.add_parser("validate", help="check specs without calling any model")
    validate.add_argument("specs", nargs="+", help="suite files")
    _add_output_flags(validate)

    # -- cache -------------------------------------------------------------
    cache = sub.add_parser("cache", help="inspect or prune the response cache")
    cache.add_argument("action", choices=("stats", "clear"))
    cache.add_argument("--cache-path", metavar="PATH")
    cache.add_argument("--older-than", type=parse_duration, metavar="DUR",
                       help="with 'clear': only drop entries older than this")
    _add_output_flags(cache)

    # -- info --------------------------------------------------------------
    sub.add_parser("info", help="list available providers, scorers and normalizers")

    return parser


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("text", "json", "md"), default="text")
    parser.add_argument("--out", metavar="PATH", help="also write the output to this file")


def _add_stats_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                        help=f"confidence level (default {DEFAULT_LEVEL})")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                        help=f"bootstrap resamples (default {DEFAULT_RESAMPLES})")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    parser.add_argument(
        "--cluster-by",
        choices=("task", "case"),
        default="task",
        help="what the interval generalises over: whole tasks (default, honest about "
             "correlated cases) or individual cases (narrower, assumes independence)",
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _emit(text: str, args: argparse.Namespace) -> None:
    print(text)
    if getattr(args, "out", None):
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(style(f"wrote {path}", "dim"), file=sys.stderr)


def _open_cache(args: argparse.Namespace, suite: Suite | None = None) -> ResponseCache:
    if getattr(args, "no_cache", False) or (suite is not None and not suite.cache):
        return NullCache()
    path = getattr(args, "cache_path", None) or default_cache_path()
    return ResponseCache(
        path,
        read=not getattr(args, "refresh_cache", False),
        write=True,
        max_age_s=getattr(args, "cache_ttl", None),
    )


def _load_suite(path: str) -> Suite:
    if not Path(path).exists():
        raise SpecError(f"suite file not found: {path}")
    return load_suite(path)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    suite = _load_suite(args.suite)
    suite = filter_suite(
        suite,
        tasks=args.tasks,
        tags=args.tags,
        models=args.models,
        limit=args.limit,
        repeats=args.repeats,
    )
    if args.errors:
        from dataclasses import replace

        suite = replace(suite, errors=args.errors)
    if args.concurrency:
        from dataclasses import replace

        suite = replace(suite, limits=replace(suite.limits, max_concurrency=args.concurrency))

    cache = _open_cache(args, suite)

    if args.dry_run:
        return _dry_run(args, suite, cache)

    if args.resume:
        store = RunStore.open(args.resume, args.runs_dir)
        already = len(store.completed_keys())
        print(
            style(f"resuming {store.run_id}", "bold")
            + style(f"  ({already} trial(s) already recorded)", "dim")
        )
    else:
        store = RunStore.create(args.runs_dir, run_id_for(suite.name))

    store.write_manifest(build_manifest(suite, store.run_id, command=sys.argv))

    show_progress = not args.no_progress and args.format == "text"

    def on_progress(progress: RunProgress) -> None:
        if show_progress:
            write_progress(
                progress_line(
                    progress.done,
                    progress.total,
                    ok=progress.ok,
                    errors=progress.errors,
                    cached=progress.cache_hits,
                    cost=progress.cost_usd,
                    eta_s=progress.eta_s,
                )
            )

    runner = Runner(suite, store, cache=cache, resume=bool(args.resume), progress=on_progress)
    try:
        result = asyncio.run(runner.run())
    except KeyboardInterrupt:
        clear_progress()
        print(
            style(
                f"\ninterrupted. {store.run_id} holds every completed trial -- "
                f"resume with: evalctl run {args.suite} --resume {store.run_id}",
                "yellow",
            ),
            file=sys.stderr,
        )
        return EXIT_ERROR
    finally:
        store.close()
        cache.close()
    clear_progress()

    records = store.all_records()  # includes anything a previous attempt wrote
    analysis = analyze_run(
        records,
        run_id=store.run_id,
        error_policy=args.errors or suite.errors,  # type: ignore[arg-type]
        level=args.level,
        resamples=args.resamples,
        seed=args.seed,
        cluster_by=args.cluster_by,
        duration_s=result.duration_s,
    )
    payload = analysis.as_dict()
    payload["judge"] = runner.judge_costs.as_dict()
    payload["cache"] = dict(result.cache_stats)
    payload["limiters"] = dict(result.limiter_stats)
    store.write_summary(payload)

    if args.format == "json":
        _emit(to_json(payload), args)
    else:
        _emit(
            format_run_report(
                analysis, manifest=store.manifest(), markdown=args.format == "md"
            ),
            args,
        )
        if runner.judge_costs.calls:
            print(
                style(
                    f"judge: {runner.judge_costs.calls} call(s), "
                    f"{runner.judge_costs.cache_hits} from cache, "
                    f"{money(runner.judge_costs.cost_usd)}",
                    "dim",
                )
            )
        print(style(f"run id: {store.run_id}   ({duration(result.duration_s)})", "dim"))

    if args.fail_under is not None:
        below = [m for m in analysis.models if m.score.point < args.fail_under]
        if below:
            names = ", ".join(f"{m.model_id} {pct(m.score.point)}" for m in below)
            print(
                style(f"gate failed: {names} below --fail-under {pct(args.fail_under)}", "red"),
                file=sys.stderr,
            )
            return EXIT_GATE
    return EXIT_OK


def _dry_run(args: argparse.Namespace, suite: Suite, cache: ResponseCache) -> int:
    estimates, rendered = dry_run(suite, cache=cache)
    if args.format == "json":
        _emit(
            to_json(
                {
                    "suite": suite.name,
                    "planned_trials": suite.total_trials,
                    "estimates": [e.as_dict() for e in estimates],
                }
            ),
            args,
        )
        return EXIT_OK

    rows = [
        [
            e.model_id,
            str(e.trials),
            str(e.cached_trials),
            f"{e.input_tokens:,}",
            f"{e.projected_output_tokens:,}",
            money(e.cost_usd),
        ]
        for e in estimates
    ]
    total = sum(e.cost_usd for e in estimates)
    print(style(f"dry run · suite '{suite.name}'", "bold"))
    print(
        render_table(
            ["model", "trials", "cached", "input tok", "output tok (max)", "projected cost"],
            rows,
            align=["l", "r", "r", "r", "r", "r"],
        )
    )
    print(style(f"projected upper bound: {money(total)}", "bold"))
    print(
        style(
            "output tokens are assumed to hit max_tokens, so the real cost lands below this. "
            "Cached trials are already excluded.",
            "dim",
        )
    )
    if rendered:
        trial, request = rendered[0]
        print("\n" + style(f"first prompt · {trial.task.id}/{trial.case.id}", "bold"))
        print(style("-" * 60, "dim"))
        print(request.prompt_text[:1200])
        print(style("-" * 60, "dim"))
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    store = RunStore.open(args.run, args.runs_dir)
    records = store.all_records()
    if not records:
        print(f"run '{store.run_id}' has no records", file=sys.stderr)
        return EXIT_ERROR
    manifest = store.manifest()
    policy = args.errors or (manifest.get("suite") or {}).get("errors") or "zero"
    analysis = analyze_run(
        records,
        run_id=store.run_id,
        error_policy=policy,
        level=args.level,
        resamples=args.resamples,
        seed=args.seed,
        cluster_by=args.cluster_by,
    )
    if args.format == "json":
        _emit(to_json(analysis.as_dict()), args)
    else:
        _emit(
            format_run_report(
                analysis,
                manifest=manifest,
                show_tasks=not args.no_tasks,
                markdown=args.format == "md",
            ),
            args,
        )
    return EXIT_OK


def cmd_diff(args: argparse.Namespace) -> int:
    report = diff_runs(
        args.a,
        args.b,
        runs_dir=args.runs_dir,
        metric=args.metric,
        error_policy=args.errors,
        level=args.level,
        resamples=args.resamples,
        seed=args.seed,
        cluster_by=args.cluster_by,
    )
    if args.format == "json":
        _emit(to_json(report.as_dict()), args)
    else:
        text = format_diff_report(
            report.result,
            label_a=report.a.label,
            label_b=report.b.label,
            metric=args.metric,
            cluster_by=args.cluster_by,
            warnings=report.warnings,
            markdown=args.format == "md",
        )
        if args.cases and report.case_deltas:
            text += "\n\n" + format_case_deltas(
                report.case_deltas, limit=args.cases, markdown=args.format == "md"
            )
        _emit(text, args)

    if args.fail_on_regression and report.result.significant and report.result.delta < 0:
        print(
            style(
                f"gate failed: significant regression of {report.result.delta * 100:.1f}pp",
                "red",
            ),
            file=sys.stderr,
        )
        return EXIT_GATE
    return EXIT_OK


def cmd_ls(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_dir)[: args.limit]
    if not runs:
        print(f"no runs under '{args.runs_dir}'")
        return EXIT_OK
    entries: list[dict[str, Any]] = []
    rows: list[list[str]] = []
    for path in runs:
        store = RunStore(path)
        manifest = store.manifest()
        summary = store.summary()
        models = summary.get("models") or []
        best = max((m.get("score", {}).get("point", 0.0) for m in models), default=None)
        entry = {
            "run_id": store.run_id,
            "created_at": manifest.get("created_at", ""),
            "suite": (manifest.get("suite") or {}).get("name", ""),
            "models": [m.get("model_id") for m in models] or
                      [m.get("id") for m in (manifest.get("models") or [])],
            "trials": summary.get("total_trials", 0),
            "errors": summary.get("total_errors", 0),
            "best_score": best,
            "cost_usd": summary.get("total_cost_usd", 0.0),
        }
        entries.append(entry)
        rows.append(
            [
                store.run_id,
                entry["created_at"][:19].replace("T", " "),
                entry["suite"],
                ",".join(str(m) for m in entry["models"] if m)[:28],
                str(entry["trials"]),
                str(entry["errors"]) if entry["errors"] else "",
                pct(best) if best is not None else "-",
                money(float(entry["cost_usd"] or 0)),
            ]
        )
    if args.format == "json":
        _emit(to_json(entries), args)
    else:
        _emit(
            render_table(
                ["run id", "created (UTC)", "suite", "models", "trials", "err", "best", "cost"],
                rows,
                align=["l", "l", "l", "l", "r", "r", "r", "r"],
            ),
            args,
        )
    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    from .scorers import judge_disagreement

    store = RunStore.open(args.run, args.runs_dir)
    records = store.all_records()
    if not records:
        print(f"run '{store.run_id}' has no records", file=sys.stderr)
        return EXIT_ERROR

    selected = records
    if args.model_id:
        selected = [r for r in selected if r.model_id == args.model_id]
    if args.task_id:
        selected = [r for r in selected if r.task_id == args.task_id]
    if args.case_id:
        selected = [r for r in selected if r.case_id == args.case_id]
    if args.errors_only:
        selected = [r for r in selected if r.status != "ok"]
    elif args.failures:
        selected = [r for r in selected if r.status != "ok" or not r.passed]

    def disagreement_of(record: Any) -> float:
        return max(
            (judge_disagreement(c.get("detail") or {}) for c in record.components
             if c.get("type") == "llm_judge"),
            default=0.0,
        )

    if args.disagreement:
        selected = sorted(selected, key=disagreement_of, reverse=True)
    selected = selected[: args.limit]

    if not selected:
        print("no trials matched those filters")
        return EXIT_OK

    if args.format == "json":
        _emit(to_json([r.as_dict() for r in selected]), args)
        return EXIT_OK

    colored = color_enabled()
    chunks: list[str] = []
    for record in selected:
        head = f"{record.model_id}  {record.task_id}/{record.case_id}#{record.sample_index}"
        if record.status != "ok":
            status = style(f"ERROR ({record.error_type})", "red", enabled=colored)
        else:
            verdict = "PASS" if record.passed else "FAIL"
            status = style(
                f"{verdict}  score {record.score:.3f}",
                "green" if record.passed else "red",
                enabled=colored,
            )
        lines = [style(head, "bold", enabled=colored), f"  {status}"]
        meta = [
            f"tokens {record.input_tokens}->{record.output_tokens}",
            f"{record.latency_ms:.0f}ms" if record.latency_ms else "cached" if record.cached else "",
            f"attempts {record.attempts}" if record.attempts > 1 else "",
            f"finish {record.finish_reason}" if record.finish_reason not in (None, "stop") else "",
        ]
        lines.append(style("  " + "  ·  ".join(m for m in meta if m), "dim", enabled=colored))
        if record.error:
            lines.append(style(f"  {record.error}", "red", enabled=colored))
        if record.response_text:
            body = record.response_text if args.full else record.response_text[:600]
            if len(record.response_text) > len(body):
                body += style(f"  ...[+{len(record.response_text) - len(body)} chars]", "dim",
                              enabled=colored)
            lines.append("  " + body.replace("\n", "\n  "))
        for component in record.components:
            mark = "ok " if component.get("passed") else "no "
            detail = component.get("detail") or {}
            bits: list[str] = []
            for key in ("matched", "normalized_response", "picked", "target", "reason", "rationale"):
                if detail.get(key) not in (None, "", []):
                    bits.append(f"{key}={detail[key]!r}"[:120])
            if component.get("type") == "llm_judge":
                spread = judge_disagreement(detail)
                if spread:
                    bits.append(f"judge_spread={spread:.2f}")
            lines.append(
                style(
                    f"    [{mark}] {component.get('name')} = {component.get('value'):.2f}"
                    + (f"   {'  '.join(bits)}" if bits else ""),
                    "dim",
                    enabled=colored,
                )
            )
        chunks.append("\n".join(lines))
    _emit("\n\n".join(chunks), args)
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    problems = 0
    for spec_path in args.specs:
        try:
            suite = _load_suite(spec_path)
        except EvalctlError as exc:
            print(style(f"FAIL  {spec_path}", "red"))
            print(f"      {exc}")
            problems += 1
            continue
        cases = sum(len(t.cases) for t in suite.tasks)
        print(style(f"OK    {spec_path}", "green"))
        print(
            f"      suite '{suite.name}': {len(suite.tasks)} task(s), {cases} case(s), "
            f"{len(suite.models)} model(s), {suite.total_trials} planned trial(s)"
        )
        print(f"      fingerprint {suite.fingerprint()[:16]}  ·  errors '{suite.errors}'")

        advisories: list[str] = []
        missing_pricing = [m.id for m in suite.models
                           if m.pricing.input_per_mtok == 0 and m.provider != "mock"]
        if missing_pricing:
            advisories.append(
                f"no pricing for {', '.join(missing_pricing)} -- cost will report as $0"
            )
        if len(suite.tasks) < 2:
            advisories.append(
                "only one task: the cluster bootstrap needs at least two to produce an interval"
            )
        if suite.judge and suite.judge.samples == 1:
            advisories.append(
                "judge.samples is 1 -- consider 3 so judge noise is visible rather than silent"
            )
        for task in suite.tasks:
            if len(task.cases) < 3:
                advisories.append(f"task '{task.id}' has only {len(task.cases)} case(s)")
        if advisories:
            print(style("      advisories:", "yellow"))
            print(bullet_list(f"    {a}" for a in advisories))
    return EXIT_ERROR if problems else EXIT_OK


def cmd_cache(args: argparse.Namespace) -> int:
    path = args.cache_path or default_cache_path()
    cache = ResponseCache(path)
    try:
        if args.action == "clear":
            removed = cache.clear(older_than_s=args.older_than)
            print(f"removed {removed} cached response(s) from {path}")
            return EXIT_OK
        summary = cache.summary()
        if args.format == "json":
            _emit(to_json(summary), args)
            return EXIT_OK
        print(style(f"cache {summary['path']}", "bold"))
        print(f"  entries      {summary['entries']:,}")
        print(f"  size         {summary['size_bytes'] / 1e6:.2f} MB")
        print(f"  lifetime hits {summary.get('lifetime_hits', 0):,}")
        if summary.get("by_model"):
            rows = [[m["provider"], m["model"], f"{m['entries']:,}"] for m in summary["by_model"]]
            print(render_table(["provider", "model", "entries"], rows, align=["l", "l", "r"]))
    finally:
        cache.close()
    return EXIT_OK


def cmd_info(args: argparse.Namespace) -> int:
    from .scorers.normalize import NORMALIZERS

    print(style("providers", "bold"))
    print(bullet_list(provider_registry.available_providers()))
    print(style("\nscorers", "bold"))
    print(bullet_list(available_scorers()))
    print(style("\nnormalizers", "bold"))
    print(bullet_list(sorted(NORMALIZERS)))
    return EXIT_OK


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

_COMMANDS = {
    "run": cmd_run,
    "report": cmd_report,
    "diff": cmd_diff,
    "ls": cmd_ls,
    "show": cmd_show,
    "validate": cmd_validate,
    "cache": cmd_cache,
    "info": cmd_info,
}


def _prepare_streams() -> None:
    """Prefer UTF-8 output, then let the renderer see what it actually got.

    Windows still defaults stdout to the ANSI code page, which mangles both the
    table borders and any non-ASCII model output. Upgrading the stream is the
    fix; ``errors="replace"`` keeps a stubborn console from turning a model
    response into a UnicodeEncodeError traceback mid-run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    report_charset.refresh(sys.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    _prepare_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except (SpecError, RunNotFound) as exc:
        print(style(f"error: {exc}", "red"), file=sys.stderr)
        return EXIT_ERROR
    except EvalctlError as exc:
        print(style(f"error: {exc}", "red"), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return EXIT_ERROR
    except BrokenPipeError:  # `evalctl ls | head`
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
