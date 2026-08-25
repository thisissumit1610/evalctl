"""Spec loading and validation.

The theme: every one of these is a mistake that would otherwise be discovered
*after* paying for a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalctl.errors import SpecError
from evalctl.spec import ScorerSpec, load_suite, load_tasks_from_file, parse_task
from evalctl.templating import render, render_value, variables_in

MINIMAL_TASK = """
id: t/one
prompt:
  user: "What is {{ q }}?"
scoring:
  - type: exact_match
    target: "{{ a }}"
cases:
  - id: c1
    vars: {q: two plus two, a: "4"}
"""

MINIMAL_SUITE = """
name: s
tasks: [tasks/*.yaml]
models:
  - id: m
    provider: mock
    model: fake
"""


def write_suite(tmp_path: Path, suite: str = MINIMAL_SUITE, task: str = MINIMAL_TASK) -> Path:
    (tmp_path / "tasks").mkdir(exist_ok=True)
    (tmp_path / "tasks" / "t.yaml").write_text(task, encoding="utf-8")
    path = tmp_path / "suite.yaml"
    path.write_text(suite, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_loads_a_minimal_suite(tmp_path):
    suite = load_suite(write_suite(tmp_path))
    assert suite.name == "s"
    assert len(suite.tasks) == 1
    assert suite.tasks[0].id == "t/one"
    assert suite.models[0].provider == "mock"
    assert suite.total_trials == 1


def test_task_patterns_resolve_relative_to_the_suite_file(tmp_path, monkeypatch):
    path = write_suite(tmp_path)
    monkeypatch.chdir(tmp_path.parent)  # run from somewhere else entirely
    suite = load_suite(path)
    assert len(suite.tasks) == 1


def test_params_merge_suite_then_task_then_model(tmp_path):
    suite_yaml = """
name: s
params: {temperature: 1.0, max_tokens: 100}
tasks: [tasks/*.yaml]
models:
  - {id: m, provider: mock, model: fake, params: {temperature: 0.0}}
"""
    task_yaml = MINIMAL_TASK + "params: {max_tokens: 50}\n"
    suite = load_suite(write_suite(tmp_path, suite_yaml, task_yaml))
    merged = suite.params_for(suite.tasks[0], suite.models[0])
    assert merged == {"temperature": 0.0, "max_tokens": 50}


def test_example_suites_are_valid(examples_dir):
    for name in ("demo.yaml", "production.yaml"):
        suite = load_suite(examples_dir / "suites" / name)
        assert suite.tasks and suite.models


# --------------------------------------------------------------------------
# the validations that save money
# --------------------------------------------------------------------------


def test_unknown_key_is_an_error_not_a_shrug(tmp_path):
    """`temperture: 0` must not silently run the whole suite at temperature 1."""
    suite_yaml = MINIMAL_SUITE.replace(
        "    model: fake", "    model: fake\n    temperture: 0"
    )
    with pytest.raises(SpecError) as exc:
        load_suite(write_suite(tmp_path, suite_yaml))
    assert "unknown key" in str(exc.value)
    assert "temperture" in str(exc.value)


def test_case_missing_a_template_variable_fails_at_load_time(tmp_path):
    task = MINIMAL_TASK.replace("vars: {q: two plus two, a: \"4\"}", "vars: {question: oops, a: \"4\"}")
    with pytest.raises(SpecError) as exc:
        load_suite(write_suite(tmp_path, task=task))
    assert "missing var" in str(exc.value)
    assert "'q'" in str(exc.value) or " q" in str(exc.value)


def test_scorer_target_variables_are_checked_too(tmp_path):
    task = MINIMAL_TASK.replace('target: "{{ a }}"', 'target: "{{ answer }}"')
    with pytest.raises(SpecError) as exc:
        load_suite(write_suite(tmp_path, task=task))
    assert "answer" in str(exc.value)


def test_duplicate_model_ids_rejected(tmp_path):
    suite_yaml = MINIMAL_SUITE + "  - {id: m, provider: mock, model: other}\n"
    with pytest.raises(SpecError, match="duplicate model id"):
        load_suite(write_suite(tmp_path, suite_yaml))


def test_duplicate_case_ids_rejected(tmp_path):
    task = MINIMAL_TASK + "  - {id: c1, vars: {q: x, a: y}}\n"
    with pytest.raises(SpecError, match="duplicate case id"):
        load_suite(write_suite(tmp_path, task=task))


def test_llm_judge_without_a_judge_endpoint_is_rejected(tmp_path):
    task = """
id: t/judged
prompt: {user: "{{ q }}"}
scoring:
  - type: llm_judge
    rubric: [{id: quality, description: "Is it good?"}]
cases:
  - {id: c1, vars: {q: hello}}
"""
    with pytest.raises(SpecError, match="no 'judge:' endpoint"):
        load_suite(write_suite(tmp_path, task=task))


def test_judge_samples_must_be_odd(tmp_path):
    suite_yaml = MINIMAL_SUITE + """
judge:
  provider: mock
  model: j
  samples: 2
"""
    with pytest.raises(SpecError, match="odd"):
        load_suite(write_suite(tmp_path, suite_yaml))


def test_task_pattern_matching_nothing_is_an_error(tmp_path):
    suite_yaml = MINIMAL_SUITE.replace("tasks: [tasks/*.yaml]", "tasks: [nope/*.yaml]")
    with pytest.raises(SpecError, match="matched no files"):
        load_suite(write_suite(tmp_path, suite_yaml))


def test_prompt_needs_user_or_messages_but_not_both(tmp_path):
    both = MINIMAL_TASK.replace(
        'prompt:\n  user: "What is {{ q }}?"',
        'prompt:\n  user: "{{ q }}"\n  messages: [{role: user, content: "{{ q }}"}]',
    )
    with pytest.raises(SpecError, match="not both"):
        load_suite(write_suite(tmp_path, task=both))


def test_multi_turn_messages_must_end_with_user(tmp_path):
    task = """
id: t/chat
prompt:
  messages:
    - {role: user, content: "{{ q }}"}
    - {role: assistant, content: "thinking"}
scoring: [{type: contains, target: x}]
cases: [{id: c1, vars: {q: hi}}]
"""
    with pytest.raises(SpecError, match="last message must have role 'user'"):
        load_suite(write_suite(tmp_path, task=task))


def test_system_role_in_messages_is_rejected_with_a_pointer(tmp_path):
    task = """
id: t/chat
prompt:
  messages: [{role: system, content: hi}, {role: user, content: "{{ q }}"}]
scoring: [{type: contains, target: x}]
cases: [{id: c1, vars: {q: hi}}]
"""
    with pytest.raises(SpecError, match="prompt.system"):
        load_suite(write_suite(tmp_path, task=task))


def test_error_message_names_the_file_and_the_field(tmp_path):
    suite_yaml = MINIMAL_SUITE.replace("provider: mock", "provider: 12")
    with pytest.raises(SpecError) as exc:
        load_suite(write_suite(tmp_path, suite_yaml))
    message = str(exc.value)
    assert "suite.yaml" in message
    assert "models[0].provider" in message


def test_invalid_yaml_reports_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError, match="invalid YAML"):
        load_suite(path)


def test_missing_cases_is_an_error(tmp_path):
    task = "id: t/empty\nprompt: {user: hi}\nscoring: [{type: contains, target: x}]\n"
    with pytest.raises(SpecError, match="at least one entry under 'cases'"):
        load_suite(write_suite(tmp_path, task=task))


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------


def test_fingerprint_changes_when_a_case_changes(tmp_path):
    first = load_suite(write_suite(tmp_path)).fingerprint()
    changed = MINIMAL_TASK.replace('a: "4"', 'a: "5"')
    second = load_suite(write_suite(tmp_path, task=changed)).fingerprint()
    assert first != second


def test_fingerprint_ignores_models_and_repeats(tmp_path):
    """Swapping the model is the point of a diff -- it must not look like a
    different benchmark."""
    base = load_suite(write_suite(tmp_path)).fingerprint()
    other = MINIMAL_SUITE.replace("model: fake", "model: other") + "repeats: 5\n"
    assert load_suite(write_suite(tmp_path, other)).fingerprint() == base


def test_fingerprint_is_stable_across_key_order(tmp_path):
    reordered = """
id: t/one
cases:
  - {vars: {a: "4", q: two plus two}, id: c1}
scoring: [{target: "{{ a }}", type: exact_match}]
prompt: {user: "What is {{ q }}?"}
"""
    assert load_suite(write_suite(tmp_path)).fingerprint() == load_suite(
        write_suite(tmp_path, task=reordered)
    ).fingerprint()


# --------------------------------------------------------------------------
# task file shapes
# --------------------------------------------------------------------------


def test_task_file_may_hold_a_list_of_tasks(tmp_path):
    path = tmp_path / "many.yaml"
    path.write_text(
        "- " + MINIMAL_TASK.strip().replace("\n", "\n  ") + "\n"
        "- " + MINIMAL_TASK.strip().replace("t/one", "t/two").replace("\n", "\n  ") + "\n",
        encoding="utf-8",
    )
    tasks = load_tasks_from_file(path)
    assert [t.id for t in tasks] == ["t/one", "t/two"]


def test_scorer_shorthand_string_form():
    spec = ScorerSpec.parse("exact_match", "scoring[0]", "x.yaml")
    assert spec.type == "exact_match"
    assert spec.weight == 1.0 and spec.required is True


def test_case_can_override_task_scoring(tmp_path):
    task = """
id: t/mixed
prompt: {user: "{{ q }}"}
scoring: [{type: contains, target: default}]
cases:
  - {id: c1, vars: {q: hi}}
  - {id: c2, vars: {q: hi}, scoring: [{type: contains, target: special}]}
"""
    loaded = parse_task(
        __import__("yaml").safe_load(task), "x.yaml"
    )
    assert loaded.scoring_for(loaded.cases[0])[0].config["target"] == "default"
    assert loaded.scoring_for(loaded.cases[1])[0].config["target"] == "special"


# --------------------------------------------------------------------------
# templating
# --------------------------------------------------------------------------


def test_render_substitutes_and_applies_filters():
    assert render("{{ a }}/{{ b | upper }}", {"a": 1, "b": "x"}) == "1/X"


def test_render_rejects_unknown_variable_rather_than_emitting_blank():
    with pytest.raises(SpecError, match="unknown variable"):
        render("{{ nope }}", {"a": 1})


def test_render_rejects_unknown_filter():
    with pytest.raises(SpecError, match="unknown filter"):
        render("{{ a | shout }}", {"a": 1})


def test_backslash_escapes_a_tag():
    assert render(r"literal \{{ a }}", {"a": 1}) == "literal {{ a }}"


def test_whole_value_substitution_preserves_type():
    assert render_value("{{ xs }}", {"xs": ["a", "b"]}) == ["a", "b"]
    assert render_value("  {{ n }}  ", {"n": 7}) == 7
    assert render_value("n={{ n }}", {"n": 7}) == "n=7"


def test_variables_in_finds_names_including_filtered_ones():
    assert variables_in("{{a}} {{ b | json }} {{c|upper|strip}}") == {"a", "b", "c"}
