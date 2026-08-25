"""End-to-end runner behaviour and the CLI surface.

These run entirely offline against the mock provider, which is the reason the
mock exists: the whole pipeline -- render, cache, limit, call, score, persist,
analyse -- is exercised with no network and no keys.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evalctl.analysis import analyze_run, collapse_cases
from evalctl.cache import NullCache, ResponseCache
from evalctl.cli import main
from evalctl.runner import Runner, build_trials, expected_answer, filter_suite, render_request
from evalctl.spec import load_suite
from evalctl.store import RunStore, TrialRecord, build_manifest, list_runs
from evalctl.util import run_id_for

SUITE = """
name: t
tasks: [tasks/*.yaml]
repeats: {repeats}
models:
  - id: baseline
    provider: mock
    model: sim-a
    params: {{ability: 0.4, spread: 0.6, seed: 1{extra}}}
  - id: candidate
    provider: mock
    model: sim-b
    params: {{ability: 0.9, spread: 0.6, seed: 1{extra}}}
limits:
  max_concurrency: 4
  requests_per_minute: 0
  tokens_per_minute: 0
  max_retries: {retries}
"""

TASK = """
id: math/{name}
tags: [math]
prompt:
  user: "What is {{{{ q }}}}?"
scoring:
  - type: numeric
    target: "{{{{ a }}}}"
cases:
{cases}
"""


def make_project(tmp_path: Path, *, tasks: int = 4, cases: int = 6, repeats: int = 1,
                 extra_params: str = "", retries: int = 2) -> Path:
    (tmp_path / "tasks").mkdir(exist_ok=True)
    for t in range(tasks):
        body = "\n".join(
            f"  - {{id: c{c}, vars: {{q: '{t}00 plus {c}', a: {t * 100 + c}}}}}"
            for c in range(cases)
        )
        (tmp_path / "tasks" / f"t{t}.yaml").write_text(
            TASK.format(name=f"set{t}", cases=body), encoding="utf-8"
        )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        SUITE.format(repeats=repeats, extra=extra_params, retries=retries), encoding="utf-8"
    )
    return suite


def run_suite(suite_path: Path, runs_dir: Path, *, cache=None, resume=False, run_id=None):
    suite = load_suite(suite_path)
    store = RunStore.create(runs_dir, run_id or run_id_for(suite.name))
    store.write_manifest(build_manifest(suite, store.run_id))
    runner = Runner(suite, store, cache=cache or NullCache(), resume=resume)
    result = asyncio.run(runner.run())
    store.close()
    return result


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_trial_count_is_models_times_cases_times_repeats(tmp_path):
    suite = load_suite(make_project(tmp_path, tasks=3, cases=5, repeats=2))
    assert suite.total_trials == 2 * 3 * 5 * 2
    assert len(build_trials(suite)) == suite.total_trials


def test_filter_by_model_task_tag_and_limit(tmp_path):
    suite = load_suite(make_project(tmp_path, tasks=4, cases=6))
    assert len(filter_suite(suite, models=["candidate"]).models) == 1
    assert len(filter_suite(suite, tasks=["math/set1"]).tasks) == 1
    assert len(filter_suite(suite, tasks=["math/*"]).tasks) == 4
    assert len(filter_suite(suite, tags=["math"]).tasks) == 4
    limited = filter_suite(suite, limit=2)
    assert all(len(t.cases) == 2 for t in limited.tasks), "limit is per task, not global"


def test_filter_that_matches_nothing_raises_with_the_options(tmp_path):
    from evalctl.errors import FatalError

    suite = load_suite(make_project(tmp_path))
    with pytest.raises(FatalError, match="Available"):
        filter_suite(suite, models=["nope"])


def test_expected_answer_is_read_from_the_first_scorer(tmp_path):
    suite = load_suite(make_project(tmp_path, tasks=1, cases=1))
    task = suite.tasks[0]
    assert expected_answer(task, task.cases[0]) == "0"


def test_metadata_is_withheld_from_real_providers(tmp_path):
    """Editing an answer key must not invalidate a cached call whose prompt
    never contained the answer."""
    suite = load_suite(make_project(tmp_path, tasks=1, cases=1))
    task = suite.tasks[0]
    without = render_request(suite, task, task.cases[0], suite.models[0], 0, include_metadata=False)
    with_meta = render_request(suite, task, task.cases[0], suite.models[0], 0, include_metadata=True)
    assert without.metadata == {}
    assert with_meta.metadata == {"expected": "0"}


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def test_run_records_every_trial(tmp_path, tmp_runs):
    result = run_suite(make_project(tmp_path, tasks=3, cases=4), tmp_runs)
    assert len(result.records) == 2 * 3 * 4
    assert all(r.status == "ok" for r in result.records)
    assert {r.model_id for r in result.records} == {"baseline", "candidate"}


def test_a_more_able_model_scores_higher(tmp_path, tmp_runs):
    result = run_suite(make_project(tmp_path, tasks=5, cases=8), tmp_runs)
    analysis = analyze_run(result.records)
    baseline = analysis.model("baseline").score.point
    candidate = analysis.model("candidate").score.point
    assert candidate > baseline + 0.2


def test_records_are_readable_back_from_disk(tmp_path, tmp_runs):
    result = run_suite(make_project(tmp_path, tasks=2, cases=3), tmp_runs)
    reopened = RunStore.open(result.run_id, tmp_runs)
    assert len(reopened.all_records()) == len(result.records)
    assert reopened.manifest()["suite"]["name"] == "t"


def test_cache_makes_the_second_run_free(tmp_path, tmp_runs, tmp_cache_path):
    suite_path = make_project(tmp_path, tasks=2, cases=4)
    with ResponseCache(tmp_cache_path) as cache:
        first = run_suite(suite_path, tmp_runs, cache=cache)
        second = run_suite(suite_path, tmp_runs, cache=cache)
    assert not any(r.cached for r in first.records)
    assert all(r.cached for r in second.records)
    # And the answers are identical, which is what makes a re-score honest.
    assert [r.score for r in first.records] == [r.score for r in second.records]


def test_repeats_produce_distinct_draws(tmp_path, tmp_runs):
    result = run_suite(make_project(tmp_path, tasks=2, cases=4, repeats=3), tmp_runs)
    by_case: dict[tuple, set] = {}
    for record in result.records:
        by_case.setdefault((record.model_id, record.task_id, record.case_id), set()).add(
            record.response_text
        )
    assert any(len(texts) > 1 for texts in by_case.values()), "repeats replayed one answer"


def test_resume_skips_completed_trials_and_reaches_the_same_totals(tmp_path, tmp_runs):
    suite_path = make_project(tmp_path, tasks=3, cases=4)
    full = run_suite(suite_path, tmp_runs)
    complete_scores = sorted(r.score for r in full.records)

    # Simulate a hard interruption: keep only the first half of the records.
    store = RunStore.open(full.run_id, tmp_runs)
    lines = store.records_path.read_text(encoding="utf-8").splitlines()
    store.records_path.write_text("\n".join(lines[:10]) + "\n", encoding="utf-8")

    resumed = run_suite(suite_path, tmp_runs, resume=True, run_id=full.run_id)
    assert len(resumed.records) == len(full.records) - 10
    reopened = RunStore.open(full.run_id, tmp_runs)
    assert sorted(r.score for r in reopened.all_records()) == complete_scores


def test_a_torn_final_line_does_not_break_reading(tmp_path, tmp_runs):
    result = run_suite(make_project(tmp_path, tasks=2, cases=3), tmp_runs)
    store = RunStore.open(result.run_id, tmp_runs)
    with store.records_path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "partial", "model_id": ')  # killed mid-write
    assert len(store.all_records()) == len(result.records)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_failing_trials_are_recorded_and_the_run_continues(tmp_path, tmp_runs):
    suite_path = make_project(tmp_path, tasks=2, cases=5, extra_params=", error_rate: 1.0", retries=1)
    result = run_suite(suite_path, tmp_runs)
    assert len(result.records) == 2 * 2 * 5
    assert all(r.status == "error" for r in result.records)
    assert all(r.error_type == "TransientError" for r in result.records)
    assert all(r.attempts > 1 for r in result.records), "transient errors must be retried"


def test_error_policy_zero_versus_exclude_changes_the_score(tmp_path, tmp_runs):
    suite_path = make_project(tmp_path, tasks=3, cases=6, extra_params=", error_rate: 0.4", retries=0)
    result = run_suite(suite_path, tmp_runs)
    assert any(r.status == "error" for r in result.records)

    as_zero = analyze_run(result.records, error_policy="zero")
    excluded = analyze_run(result.records, error_policy="exclude")
    assert excluded.model("candidate").score.point >= as_zero.model("candidate").score.point
    assert as_zero.notes and "error policy" in as_zero.notes[0]


def test_exclude_drops_a_case_whose_every_repeat_failed():
    records = [
        TrialRecord("r", "m", "t", "c1", 0, status="error"),
        TrialRecord("r", "m", "t", "c2", 0, status="ok", score=1.0, passed=True),
    ]
    assert len(collapse_cases(records, error_policy="zero")) == 2
    assert len(collapse_cases(records, error_policy="exclude")) == 1


def test_repeats_are_averaged_within_a_case_before_anything_else():
    records = [
        TrialRecord("r", "m", "t", "c", i, status="ok", score=float(i % 2), passed=bool(i % 2))
        for i in range(4)
    ]
    outcome = collapse_cases(records)[0]
    assert outcome.score == pytest.approx(0.5)
    assert outcome.passed == pytest.approx(0.5)
    assert outcome.samples == 4


# --------------------------------------------------------------------------
# judge integration
# --------------------------------------------------------------------------


def test_judge_runs_through_the_cache_and_is_costed(tmp_path, tmp_runs, tmp_cache_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "j.yaml").write_text(
        """
id: writing/judged
prompt: {user: "Summarise: {{ passage }}"}
scoring:
  - type: llm_judge
    reference: "{{ reference }}"
    rubric:
      - {id: faithfulness, weight: 2, description: "supported by the passage?"}
      - {id: brevity, description: "one sentence?"}
cases:
  - {id: c1, vars: {passage: "a long story", reference: "a short summary"}}
  - {id: c2, vars: {passage: "another story", reference: "another summary"}}
""",
        encoding="utf-8",
    )
    (tmp_path / "suite.yaml").write_text(
        """
name: j
tasks: [tasks/*.yaml]
models: [{id: m, provider: mock, model: sim}]
judge: {provider: mock, model: sim-judge, samples: 3}
limits: {requests_per_minute: 0, tokens_per_minute: 0}
""",
        encoding="utf-8",
    )
    suite = load_suite(tmp_path / "suite.yaml")
    store = RunStore.create(tmp_runs, run_id_for("j"))
    with ResponseCache(tmp_cache_path) as cache:
        runner = Runner(suite, store, cache=cache)
        result = asyncio.run(runner.run())
        store.close()

    assert len(result.records) == 2
    assert all(r.status == "ok" for r in result.records)
    # 2 cases x 3 judge samples
    assert runner.judge_costs.calls == 6
    component = result.records[0].components[0]
    assert component["type"] == "llm_judge"
    assert component["detail"]["samples"] == 3
    assert set(component["detail"]["criteria"]) == {"faithfulness", "brevity"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_run_report_and_diff(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=4, cases=6)
    runs = str(tmp_runs)

    assert main(["--runs-dir", runs, "run", str(suite_path), "--no-progress"]) == 0
    assert "candidate" in capsys.readouterr().out

    assert main(["--runs-dir", runs, "report", "latest"]) == 0
    assert "95% CI" in capsys.readouterr().out

    assert main(["--runs-dir", runs, "diff", "latest:candidate", "latest:baseline"]) == 0
    diff_output = capsys.readouterr().out
    assert "delta (A - B)" in diff_output
    assert "IMPROVEMENT" in diff_output


def test_cli_diff_json_is_machine_readable(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=4, cases=6)
    runs = str(tmp_runs)
    main(["--runs-dir", runs, "run", str(suite_path), "--no-progress", "--format", "json"])
    capsys.readouterr()
    main(["--runs-dir", runs, "diff", "latest:candidate", "latest:baseline", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["verdict"] in {"improvement", "regression", "no significant difference"}
    assert payload["result"]["paired_ci"]["method"] in {"bca", "percentile", "degenerate"}
    assert payload["cluster_by"] == "task"


def test_cli_fail_under_gates_with_exit_code_3(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=3, cases=5)
    runs = str(tmp_runs)
    assert main(["--runs-dir", runs, "run", str(suite_path), "--no-progress", "--fail-under", "0.99"]) == 3
    assert main(["--runs-dir", runs, "run", str(suite_path), "--no-progress", "--fail-under", "0.01"]) == 0


def test_cli_fail_on_regression(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=5, cases=8)
    runs = str(tmp_runs)
    main(["--runs-dir", runs, "run", str(suite_path), "--no-progress"])
    capsys.readouterr()
    # baseline vs candidate is a real regression in that direction
    code = main([
        "--runs-dir", runs, "diff", "latest:baseline", "latest:candidate",
        "--fail-on-regression", "--cluster-by", "case",
    ])
    assert code == 3


def test_cli_dry_run_makes_no_calls_and_prints_a_projection(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=2, cases=3)
    assert main(["--runs-dir", str(tmp_runs), "run", str(suite_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "projected upper bound" in output
    assert list_runs(tmp_runs) == [], "a dry run must not create a run directory"


def test_cli_validate_reports_ok_and_failure(tmp_path, capsys):
    good = make_project(tmp_path)
    assert main(["validate", str(good)]) == 0
    assert "OK" in capsys.readouterr().out

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nmodels: []\ntasks: []\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_show_filters_to_failures(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=3, cases=6)
    runs = str(tmp_runs)
    main(["--runs-dir", runs, "run", str(suite_path), "--no-progress"])
    capsys.readouterr()
    assert main(["--runs-dir", runs, "show", "latest", "--failures", "--model", "baseline", "-n", "3"]) == 0
    output = capsys.readouterr().out
    assert "FAIL" in output and "PASS" not in output


def test_cli_ls_lists_the_run(tmp_path, tmp_runs, capsys):
    suite_path = make_project(tmp_path, tasks=2, cases=3)
    runs = str(tmp_runs)
    main(["--runs-dir", runs, "run", str(suite_path), "--no-progress"])
    capsys.readouterr()
    assert main(["--runs-dir", runs, "ls"]) == 0
    assert "run id" in capsys.readouterr().out


def test_cli_missing_run_exits_1(tmp_runs, capsys):
    assert main(["--runs-dir", str(tmp_runs), "report", "nope"]) == 1


def test_cli_bad_suite_path_exits_1(capsys):
    assert main(["run", "does/not/exist.yaml"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------
# comparability warnings
# --------------------------------------------------------------------------


def test_diff_warns_when_the_two_runs_measured_different_suites(tmp_path, tmp_runs, capsys):
    from evalctl.diff import diff_runs

    suite_path = make_project(tmp_path, tasks=3, cases=5)
    first = run_suite(suite_path, tmp_runs)

    # Change a case, then run again: the suite fingerprint must diverge.
    task_file = tmp_path / "tasks" / "t0.yaml"
    task_file.write_text(task_file.read_text(encoding="utf-8").replace("a: 0}", "a: 999}"), encoding="utf-8")
    second = run_suite(suite_path, tmp_runs)

    report = diff_runs(
        f"{second.run_id}:candidate", f"{first.run_id}:candidate",
        runs_dir=str(tmp_runs), resamples=400,
    )
    assert any("different suite content" in w for w in report.warnings)
    assert any("task definitions differ" in w for w in report.warnings)


def test_diff_requires_a_model_when_a_run_has_several(tmp_path, tmp_runs):
    from evalctl.diff import resolve_side
    from evalctl.errors import RunNotFound

    result = run_suite(make_project(tmp_path, tasks=2, cases=3), tmp_runs)
    with pytest.raises(RunNotFound, match="pick one"):
        resolve_side(result.run_id, runs_dir=str(tmp_runs))


def test_diff_selector_parsing_tolerates_windows_paths():
    from evalctl.diff import parse_selector

    assert parse_selector("run-1:candidate") == ("run-1", "candidate")
    assert parse_selector("run-1") == ("run-1", None)
    assert parse_selector(r"C:\runs\run-1") == (r"C:\runs\run-1", None)
    assert parse_selector("runs/run-1") == ("runs/run-1", None)
