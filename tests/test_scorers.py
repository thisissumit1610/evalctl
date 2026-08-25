"""Scorers, normalizers and aggregation.

Most of these are about the gap between "the model was right" and "the harness
noticed" -- which is where benchmark numbers are actually won and lost.
"""

from __future__ import annotations

import asyncio

import pytest

from evalctl.errors import ScoringError
from evalctl.scorers import build, aggregate, resolve
from evalctl.scorers.base import ScoredComponent
from evalctl.scorers.builtin import parse_json_response
from evalctl.scorers.llm_judge import LLMJudgeScorer, judge_disagreement
from evalctl.scorers.normalize import DEFAULT_NORMALIZERS, apply_named, extract_choice
from evalctl.spec import ScorerSpec


def score(scorer_type: str, config: dict, response: str, variables: dict | None = None):
    scorer = build(ScorerSpec(type=scorer_type, config=config), variables or {})
    return asyncio.run(scorer.score(response, None))


# --------------------------------------------------------------------------
# normalizers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  42  ", "42"),
        ("**42**", "42"),
        ("The answer is 42.", "42"),
        ("Answer: 42", "42"),
        ('"42"', "42"),
        ("<think>let me see</think>42", "42"),
        ("```\n42\n```", "42"),
        ("***42***", "42"),
    ],
)
def test_default_normalizers_unwrap_packaging_without_touching_the_answer(raw, expected):
    assert apply_named(raw, list(DEFAULT_NORMALIZERS)) == expected


def test_default_normalizers_do_not_mangle_interior_punctuation():
    """Stripping interior punctuation would turn 3.14 into 314."""
    assert apply_named("3.14", list(DEFAULT_NORMALIZERS)) == "3.14"
    assert apply_named("don't", list(DEFAULT_NORMALIZERS)) == "don't"


def test_extract_choice_takes_the_last_decision():
    assert extract_choice("Maybe A. Actually the answer is C.") == "C"
    assert extract_choice("B") == "B"
    assert extract_choice("I'd say (D)") == "D"


def test_unknown_normalizer_is_rejected_by_name():
    with pytest.raises(ScoringError):
        score("exact_match", {"target": "x", "normalize": ["nope"]}, "x")


# --------------------------------------------------------------------------
# exact match / contains
# --------------------------------------------------------------------------


def test_exact_match_normalizes_both_sides():
    assert score("exact_match", {"target": "  Paris  "}, "**Paris.**").passed


def test_exact_match_any_of_accepts_alternatives():
    result = score("exact_match", {"target": "Bern", "any_of": ["Berne"]}, "Berne")
    assert result.passed and result.detail["matched"] == "Berne"


def test_exact_match_case_sensitivity_is_configurable():
    assert not score("exact_match", {"target": "Paris"}, "paris").passed
    assert score("exact_match", {"target": "Paris", "case_sensitive": False}, "paris").passed


def test_exact_match_requires_a_target():
    with pytest.raises(ScoringError, match="needs 'target' or 'any_of'"):
        score("exact_match", {}, "x")


def test_numeric_target_is_accepted_as_text():
    """`answer: 55` in YAML is an int; it must behave like `answer: "55"`."""
    assert score("exact_match", {"target": 55}, "55").passed


def test_contains_all_of_awards_partial_credit():
    result = score("contains", {"all_of": ["alpha", "beta", "gamma"]}, "alpha and beta")
    assert result.value == pytest.approx(2 / 3)
    assert not result.passed


def test_not_contains_inverts():
    assert score("not_contains", {"any_of": ["sorry"]}, "here you go").passed
    assert not score("not_contains", {"any_of": ["sorry"]}, "sorry, I cannot").passed


# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------


def test_numeric_reads_the_last_number_so_working_is_allowed():
    assert score("numeric", {"target": 42}, "12 x 4 = 48, minus 6, so 42").passed


def test_numeric_handles_thousands_separators_and_boxed_answers():
    assert score("numeric", {"target": 1234}, "the total is 1,234").passed
    assert score("numeric", {"target": 17}, r"therefore \boxed{17}").passed


def test_numeric_tolerances_absolute_and_relative():
    assert score("numeric", {"target": 100, "tolerance": 1}, "100.5").passed
    assert not score("numeric", {"target": 100, "tolerance": 0.1}, "100.5").passed
    assert score("numeric", {"target": 1000, "rel_tolerance": 0.01}, "1005").passed


def test_numeric_strict_extraction_refuses_an_ambiguous_response():
    assert not score("numeric", {"target": 42, "extract": "strict"}, "either 12 or 42").passed
    assert score("numeric", {"target": 42, "extract": "strict"}, "42").passed


def test_numeric_with_no_number_fails_with_a_reason():
    result = score("numeric", {"target": 42}, "I don't know")
    assert not result.passed
    assert "no number" in result.detail["reason"]


def test_numeric_rejects_a_non_numeric_target():
    with pytest.raises(ScoringError, match="not a number"):
        score("numeric", {"target": "abc"}, "1")


# --------------------------------------------------------------------------
# choice / regex / json
# --------------------------------------------------------------------------


def test_choice_extracts_the_letter_from_prose():
    assert score("choice", {"target": "C", "options": ["A", "B", "C", "D"]}, "I think C) is right").passed


def test_choice_target_must_be_among_options():
    with pytest.raises(ScoringError, match="not among options"):
        score("choice", {"target": "Z", "options": ["A", "B"]}, "A")


def test_regex_capture_group_compared_to_target():
    result = score("regex", {"pattern": r"ID-(\d+)", "group": 1, "target": "42"}, "found ID-42 here")
    assert result.passed and result.detail["captured"] == "42"


def test_regex_expect_false_inverts_the_check():
    assert score("regex", {"pattern": "sorry", "expect": False}, "here you go").passed
    assert not score("regex", {"pattern": "sorry", "expect": False}, "sorry!").passed


def test_regex_rejects_an_invalid_pattern_at_construction():
    with pytest.raises(ScoringError, match="invalid regex"):
        score("regex", {"pattern": "([unclosed"}, "x")


def test_json_match_finds_json_wrapped_in_prose():
    result = score("json_match", {"target": {"a": 1}}, 'Sure! {"a": 1} Hope that helps.')
    assert result.passed


def test_json_match_subset_ignores_extra_keys_but_exact_does_not():
    payload = '{"a": 1, "b": 2}'
    assert score("json_match", {"target": {"a": 1}, "mode": "subset"}, payload).passed
    assert not score("json_match", {"target": {"a": 1}, "mode": "exact"}, payload).passed


def test_json_match_partial_credit_and_mismatch_paths():
    result = score("json_match", {"target": {"a": 1, "b": 2, "c": 3}}, '{"a": 1, "b": 9, "c": 3}')
    assert 0 < result.value < 1
    assert any("$.b" in problem for problem in result.detail["mismatches"])


def test_json_match_treats_1_and_1_point_0_as_equal():
    assert score("json_match", {"target": {"a": 1}}, '{"a": 1.0}').passed


def test_json_match_on_unparseable_response():
    result = score("json_match", {"target": {"a": 1}}, "no json here")
    assert result.value == 0.0 and not result.passed


def test_parse_json_response_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_response("absolutely not json")


# --------------------------------------------------------------------------
# unknown options
# --------------------------------------------------------------------------


def test_misspelled_scorer_option_is_rejected():
    """`tolerence` must not silently become tolerance 0."""
    with pytest.raises(ScoringError, match="unknown option"):
        score("numeric", {"target": 1, "tolerence": 0.5}, "1")


def test_unknown_scorer_type_lists_the_available_ones():
    from evalctl.errors import SpecError

    with pytest.raises(SpecError, match="unknown scorer type"):
        resolve("vibes")


def test_scorer_aliases_resolve():
    assert resolve("exact").type == "exact_match"
    assert resolve("judge").type == "llm_judge"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_aggregate_is_a_weighted_mean():
    verdict = aggregate(
        [
            ScoredComponent("a", "exact_match", 1.0, True, 3.0, True),
            ScoredComponent("b", "llm_judge", 0.0, False, 1.0, False),
        ]
    )
    assert verdict.score == pytest.approx(0.75)


def test_only_required_scorers_can_veto_a_pass():
    passing = aggregate([ScoredComponent("j", "llm_judge", 0.2, False, 1.0, required=False),
                         ScoredComponent("e", "exact_match", 1.0, True, 1.0, required=True)])
    failing = aggregate([ScoredComponent("j", "llm_judge", 0.2, False, 1.0, required=True),
                         ScoredComponent("e", "exact_match", 1.0, True, 1.0, required=True)])
    assert passing.passed and not failing.passed


def test_pass_threshold_can_also_veto():
    components = [ScoredComponent("a", "contains", 0.5, True, 1.0, True)]
    assert aggregate(components, threshold=0.4).passed
    assert not aggregate(components, threshold=0.6).passed


def test_all_zero_weights_fall_back_to_an_unweighted_mean():
    verdict = aggregate([ScoredComponent("a", "x", 1.0, True, 0.0, True),
                         ScoredComponent("b", "y", 0.0, True, 0.0, True)])
    assert verdict.score == pytest.approx(0.5)


def test_aggregate_refuses_an_empty_component_list():
    with pytest.raises(ScoringError):
        aggregate([])


# --------------------------------------------------------------------------
# the judge
# --------------------------------------------------------------------------


class FakeJudge:
    """Returns canned judge replies, one per sample."""

    def __init__(self, replies, samples=None):
        self.replies = replies
        self.samples = samples if samples is not None else len(replies)
        self.prompts: list[str] = []

    async def __call__(self, *, system, user, sample_index, cache_tag):
        self.prompts.append(user)
        return self.replies[min(sample_index, len(self.replies) - 1)]


def judge_scorer(**config):
    base = {
        "rubric": [
            {"id": "accuracy", "weight": 2, "description": "correct?"},
            {"id": "brevity", "weight": 1, "description": "short?"},
        ],
        "scale": 4,
    }
    base.update(config)
    return build(ScorerSpec(type="llm_judge", config=base), {})


def run_judge(scorer, judge, response="some answer"):
    from evalctl.scorers.base import ScoringContext
    from evalctl.spec import Case, Endpoint, Task

    task = Task(id="t", cases=(Case(id="c"),), scoring=())
    ctx = ScoringContext(
        task=task,
        case=task.cases[0],
        endpoint=Endpoint(id="m", provider="mock", model="x"),
        sample_index=0,
        prompt_text="",
        judge=judge,
    )
    return asyncio.run(scorer.score(response, ctx))


def test_judge_aggregates_criteria_by_weight():
    judge = FakeJudge(['{"scores": {"accuracy": 4, "brevity": 0}, "rationale": "ok"}'])
    result = run_judge(judge_scorer(), judge)
    # (4*2 + 0*1) / (3 * 4) = 0.667
    assert result.value == pytest.approx(8 / 12)
    assert result.passed  # default threshold 0.6


def test_judge_takes_the_median_across_samples():
    judge = FakeJudge(
        [
            '{"scores": {"accuracy": 0, "brevity": 4}}',
            '{"scores": {"accuracy": 4, "brevity": 4}}',
            '{"scores": {"accuracy": 4, "brevity": 4}}',
        ]
    )
    result = run_judge(judge_scorer(), judge)
    assert result.detail["criteria"]["accuracy"]["score"] == 4
    assert result.detail["criteria"]["accuracy"]["spread"] == 4
    assert judge_disagreement(result.detail) == pytest.approx(1.0)


def test_judge_survives_one_unparseable_sample():
    judge = FakeJudge(
        ["not json at all", '{"scores": {"accuracy": 3, "brevity": 3}}'], samples=2
    )
    result = run_judge(judge_scorer(), judge)
    assert result.detail["samples"] == 1
    assert result.detail["parse_failures"]


def test_judge_raises_rather_than_scoring_zero_when_every_sample_fails():
    """A broken judge is a broken measurement, not a model regression."""
    judge = FakeJudge(["nope", "still nope"], samples=2)
    with pytest.raises(ScoringError, match="no usable scores"):
        run_judge(judge_scorer(), judge)


def test_judge_rejects_output_missing_a_criterion():
    judge = FakeJudge(['{"scores": {"accuracy": 3}}'], samples=1)
    with pytest.raises(ScoringError, match="no usable scores"):
        run_judge(judge_scorer(), judge)


def test_judge_clamps_an_out_of_range_score_instead_of_failing():
    judge = FakeJudge(['{"scores": {"accuracy": 9, "brevity": -2}}'], samples=1)
    result = run_judge(judge_scorer(), judge)
    assert result.detail["criteria"]["accuracy"]["score"] == 4
    assert result.detail["criteria"]["brevity"]["score"] == 0


def test_judge_prompt_contains_rubric_reference_and_the_response():
    judge = FakeJudge(['{"scores": {"accuracy": 3, "brevity": 3}}'], samples=1)
    scorer = judge_scorer(reference="the gold answer", question="what is x?")
    run_judge(scorer, judge, response="the model reply")
    prompt = judge.prompts[0]
    assert "the model reply" in prompt
    assert "the gold answer" in prompt
    assert "accuracy (weight 2)" in prompt
    assert "what is x?" in prompt


def test_judge_requires_a_non_empty_rubric():
    with pytest.raises(ScoringError, match="rubric"):
        build(ScorerSpec(type="llm_judge", config={"rubric": []}), {})


def test_judge_rejects_duplicate_criterion_ids():
    with pytest.raises(ScoringError, match="duplicate rubric"):
        build(
            ScorerSpec(
                type="llm_judge",
                config={
                    "rubric": [
                        {"id": "a", "description": "one"},
                        {"id": "a", "description": "two"},
                    ]
                },
            ),
            {},
        )


def test_judge_without_a_judge_runner_says_so():
    from evalctl.scorers.base import ScoringContext
    from evalctl.spec import Case, Endpoint, Task

    task = Task(id="t", cases=(Case(id="c"),), scoring=())
    ctx = ScoringContext(
        task=task, case=task.cases[0],
        endpoint=Endpoint(id="m", provider="mock", model="x"),
        sample_index=0, prompt_text="", judge=None,
    )
    with pytest.raises(ScoringError, match="needs a 'judge:' endpoint"):
        asyncio.run(judge_scorer().score("x", ctx))
