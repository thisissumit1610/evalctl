"""Statistical correctness.

These are the tests that matter most in this package. A bootstrap that is
subtly wrong still returns a plausible-looking interval on every input, so
nothing short of a coverage simulation actually catches it -- the assertions
here are about calibration, not about the shape of the return value.
"""

from __future__ import annotations

import random

import pytest

from evalctl.stats import (
    Observation,
    bootstrap_ci,
    mcnemar_exact,
    normal_cdf,
    normal_ppf,
    paired_diff,
    required_cases,
)


def make_observations(rng: random.Random, *, tasks: int, cases: int, p: float,
                      task_spread: float = 0.0) -> list[Observation]:
    """Binary outcomes. `task_spread` > 0 makes tasks differ in difficulty."""
    out: list[Observation] = []
    for t in range(tasks):
        task_p = p
        if task_spread:
            task_p = min(0.98, max(0.02, rng.gauss(p, task_spread)))
        for c in range(cases):
            out.append(
                Observation(unit=f"t{t}/c{c}", cluster=f"t{t}", value=1.0 if rng.random() < task_p else 0.0)
            )
    return out


# --------------------------------------------------------------------------
# normal helpers
# --------------------------------------------------------------------------


def test_normal_ppf_matches_known_quantiles():
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert normal_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)
    assert normal_ppf(0.8) == pytest.approx(0.8416212, abs=1e-5)


def test_normal_ppf_is_inverse_of_cdf():
    for p in (0.001, 0.01, 0.2, 0.5, 0.77, 0.99, 0.999):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-8)


def test_normal_ppf_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            normal_ppf(bad)


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_standard_error_matches_analytic_on_independent_data():
    """The one-stage cluster bootstrap must not inflate the standard error.

    Resampling cases *within* each drawn cluster as well as between clusters is
    an easy mistake that double counts within-cluster variance and widens the
    interval by about sqrt(2). On iid data the bootstrap SE should sit close to
    the analytic sqrt(p(1-p)/n), a little above it because there are only 20
    clusters to resample between.
    """
    rng = random.Random(4)
    observations = make_observations(rng, tasks=20, cases=12, p=0.70)
    interval = bootstrap_ci(observations, resamples=3000, seed=1)
    analytic = (0.70 * 0.30 / 240) ** 0.5
    assert interval.std_error == pytest.approx(analytic, rel=0.25)
    assert interval.std_error < analytic * 1.30, "SE inflated -- is the bootstrap two-stage again?"


@pytest.mark.parametrize("method", ["bca", "percentile"])
def test_interval_covers_the_truth_about_95_percent_of_the_time(method):
    """Calibration on independent data."""
    truth, trials, covered = 0.70, 200, 0
    for t in range(trials):
        rng = random.Random(1000 + t)
        observations = make_observations(rng, tasks=20, cases=12, p=truth)
        interval = bootstrap_ci(observations, resamples=600, seed=t, method=method)
        covered += interval.low <= truth <= interval.high
    coverage = covered / trials
    # Nominal 95%. The band is wide enough to be stable across seeds but tight
    # enough to fail a genuinely miscalibrated interval.
    assert 0.90 <= coverage <= 0.995, f"{method} coverage was {coverage:.1%}"


def test_clustering_matters_when_tasks_differ_in_difficulty():
    """The whole argument for clustering, as an executable claim.

    With task-level difficulty variation, pretending each case is independent
    produces intervals that miss the true value far more often than the
    nominal 5%.
    """
    truth, trials = 0.70, 200
    clustered = naive = 0
    for t in range(trials):
        rng = random.Random(7000 + t)
        observations = make_observations(rng, tasks=20, cases=12, p=truth, task_spread=0.18)
        by_task = bootstrap_ci(observations, resamples=600, seed=t)
        by_case = bootstrap_ci(
            [Observation(o.unit, o.unit, o.value) for o in observations], resamples=600, seed=t
        )
        clustered += by_task.low <= truth <= by_task.high
        naive += by_case.low <= truth <= by_case.high
    assert clustered / trials >= 0.90
    assert naive / trials < 0.90
    assert clustered > naive


def test_bootstrap_refuses_to_invent_an_interval_from_one_cluster():
    observations = [Observation(f"t/c{i}", "t", float(i % 2)) for i in range(20)]
    interval = bootstrap_ci(observations, resamples=500, seed=0)
    assert interval.method == "degenerate"
    assert interval.low == interval.high == interval.point


def test_bootstrap_on_empty_input():
    interval = bootstrap_ci([], resamples=100, seed=0)
    assert interval.method == "empty"
    assert interval.point == 0.0


def test_bootstrap_is_reproducible_for_a_given_seed():
    rng = random.Random(11)
    observations = make_observations(rng, tasks=8, cases=10, p=0.6)
    first = bootstrap_ci(observations, resamples=800, seed=42)
    second = bootstrap_ci(observations, resamples=800, seed=42)
    third = bootstrap_ci(observations, resamples=800, seed=43)
    assert (first.low, first.high) == (second.low, second.high)
    assert (first.low, first.high) != (third.low, third.high)


def test_bca_falls_back_to_percentile_when_correction_is_undefined():
    """Every replicate identical -> no bias correction is computable."""
    observations = [Observation(f"t{i}/c0", f"t{i}", 1.0) for i in range(10)]
    interval = bootstrap_ci(observations, resamples=500, seed=0, method="bca")
    assert interval.method == "percentile"
    assert interval.point == 1.0


def test_bootstrap_rejects_unknown_method():
    with pytest.raises(ValueError):
        bootstrap_ci([Observation("a", "t", 1.0)], method="jackknife")


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------


def test_pairing_produces_a_tighter_interval_than_not_pairing():
    """Why `diff` pairs: shared case difficulty cancels."""
    rng = random.Random(3)
    a: list[Observation] = []
    b: list[Observation] = []
    for t in range(15):
        for c in range(10):
            difficulty = rng.random()          # shared between the two sides
            unit = f"t{t}/c{c}"
            a.append(Observation(unit, f"t{t}", 1.0 if difficulty < 0.75 else 0.0))
            b.append(Observation(unit, f"t{t}", 1.0 if difficulty < 0.65 else 0.0))
    result = paired_diff(a, b, resamples=2000, seed=5)
    assert result.delta == pytest.approx(0.10, abs=0.05)
    paired_width = result.paired.high - result.paired.low
    unpaired_width = result.unpaired.high - result.unpaired.low
    assert paired_width < unpaired_width


def test_paired_diff_detects_no_difference_between_identical_sides():
    rng = random.Random(9)
    observations = make_observations(rng, tasks=12, cases=10, p=0.6)
    result = paired_diff(observations, observations, resamples=1500, seed=2)
    assert result.delta == 0.0
    assert not result.significant
    assert result.verdict == "no significant difference"
    assert result.wins == result.losses == 0


def test_paired_diff_reports_a_regression_with_the_right_sign():
    rng = random.Random(17)
    b: list[Observation] = []
    a: list[Observation] = []
    for t in range(20):
        for c in range(12):
            unit = f"t{t}/c{c}"
            difficulty = rng.random()
            b.append(Observation(unit, f"t{t}", 1.0 if difficulty < 0.80 else 0.0))
            a.append(Observation(unit, f"t{t}", 1.0 if difficulty < 0.55 else 0.0))
    result = paired_diff(a, b, resamples=2000, seed=1)
    assert result.delta < 0
    assert result.significant
    assert result.verdict == "regression"


def test_paired_diff_excludes_unshared_cases_and_says_so():
    a = [Observation(f"t0/c{i}", "t0", 1.0) for i in range(6)]
    b = [Observation(f"t0/c{i}", "t0", 0.0) for i in range(4)]
    b.append(Observation("t0/only-b", "t0", 1.0))
    result = paired_diff(a, b, resamples=400, seed=0)
    assert result.n_paired == 4
    assert result.only_in_a == ("t0/c4", "t0/c5")
    assert result.only_in_b == ("t0/only-b",)
    assert any("only one run" in note for note in result.notes)


def test_paired_diff_with_no_shared_cases_is_inconclusive():
    a = [Observation("t0/a", "t0", 1.0)]
    b = [Observation("t0/b", "t0", 0.0)]
    result = paired_diff(a, b, resamples=100, seed=0)
    assert result.n_paired == 0
    assert result.verdict == "inconclusive"
    assert any("share no cases" in note for note in result.notes)


# --------------------------------------------------------------------------
# McNemar and power
# --------------------------------------------------------------------------


def test_mcnemar_is_symmetric_and_bounded():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 2) == mcnemar_exact(2, 10)
    assert 0.0 <= mcnemar_exact(30, 4) <= 1.0


def test_mcnemar_exact_matches_hand_computed_value():
    # b=1, c=5: 2 * P(X <= 1), X ~ Binomial(6, 0.5) = 2 * (1 + 6)/64
    assert mcnemar_exact(1, 5) == pytest.approx(2 * 7 / 64)


def test_mcnemar_flags_a_lopsided_split_and_not_a_balanced_one():
    assert mcnemar_exact(20, 4) < 0.01
    assert mcnemar_exact(12, 11) > 0.5


def test_mcnemar_normal_approximation_agrees_with_exact_at_the_boundary():
    # 1000 is the cutover; the two branches should be close either side of it.
    below = mcnemar_exact(560, 440)
    above = mcnemar_exact(561, 441)
    assert below == pytest.approx(above, abs=0.02)


def test_required_cases_scales_inversely_with_the_square_of_the_effect():
    small = required_cases(0.01, 0.45)
    large = required_cases(0.02, 0.45)
    assert small == pytest.approx(large * 4, rel=0.02)
    assert required_cases(0.0, 0.45) == 0
    assert required_cases(0.05, 0.0) == 0


def test_binary_metric_reports_mcnemar_and_discordant_counts():
    a = [Observation(f"t{i}/c0", f"t{i}", 1.0) for i in range(10)]
    b = [Observation(f"t{i}/c0", f"t{i}", 0.0 if i < 8 else 1.0) for i in range(10)]
    result = paired_diff(a, b, resamples=500, seed=0, binary=True)
    assert result.discordant == (8, 0)
    assert result.mcnemar_p == pytest.approx(2 / 2**8)
