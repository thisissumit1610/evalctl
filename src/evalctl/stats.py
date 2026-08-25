"""Uncertainty for benchmark scores.

The problem this file exists to solve
-------------------------------------
A model change moves your suite from 71.4% to 73.2%. Is that real? On 250 cases
the standard error of a proportion near 0.7 is about 2.9 points, so a 1.8 point
move is comfortably inside the noise -- and yet it is exactly the kind of number
that gets reported as an improvement. Everything here is aimed at making that
mistake hard to make.

Three decisions, each of which changes the answer
-------------------------------------------------
**1. Cases within a task are not independent.** Twenty arithmetic word problems
generated from one template share a prompt, a format and a failure mode. Treating
them as 20 independent samples understates the variance, sometimes badly. So the
bootstrap resamples whole *tasks* with replacement rather than individual cases.
The unit of generalisation becomes "another task like these ones", which is the
claim a benchmark actually wants to make. (Resampling cases *within* the drawn
tasks as well is a tempting extra step and a real bug -- see
:func:`_cluster_replicate`.)

**2. Comparisons must be paired.** Two runs over the same suite see the same
cases, so the per-case difference cancels case difficulty entirely. Pairing
routinely shrinks the interval on a delta by 2-5x versus comparing two
independent intervals -- and the "do the two CIs overlap?" eyeball test is
strictly worse than both, since non-overlapping intervals is a *more*
conservative bar than a significant difference. :func:`paired_diff` reports the
unpaired interval alongside the paired one so the gap is visible.

**3. The interval should be bias-corrected.** Accuracy near a ceiling is
skewed -- there is more room below 0.95 than above it -- and a plain percentile
interval inherits that skew as a coverage error. BCa costs one jackknife pass
and fixes most of it. It degrades to the percentile interval automatically when
the correction is undefined (every replicate identical, one cluster, and so on).

No numpy. Not for purity: a bootstrap whose numbers depend on which BLAS is
installed is a bootstrap you cannot reproduce from a CI log six months later.
Pure Python with a seeded `random.Random` gives byte-identical intervals on
every machine, and 5,000 replicates over a few hundred cases takes about a
second.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

DEFAULT_RESAMPLES = 5000
DEFAULT_LEVEL = 0.95
DEFAULT_POWER = 0.80


# --------------------------------------------------------------------------
# normal distribution helpers (stdlib only)
# --------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation, |relative error| < 1.15e-9 over the open
# interval. Plenty for interval endpoints, and it keeps scipy out of the deps.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf needs 0 < p < 1, got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1
    )


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One case's score, tagged with the cluster it belongs to.

    ``unit`` identifies the case across runs and is what pairing joins on;
    ``cluster`` is the task, and is what the outer bootstrap stage resamples.
    """

    unit: str
    cluster: str
    value: float


def group_by_cluster(observations: Iterable[Observation]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for obs in observations:
        grouped.setdefault(obs.cluster, []).append(obs.value)
    return grouped


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    level: float = DEFAULT_LEVEL
    method: str = "bca"
    n_units: int = 0
    n_clusters: int = 0
    std_error: float = 0.0

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "level": self.level,
            "method": self.method,
            "n_units": self.n_units,
            "n_clusters": self.n_clusters,
            "std_error": round(self.std_error, 6),
        }

    def format(self, *, percent: bool = True, digits: int = 1) -> str:
        scale = 100.0 if percent else 1.0
        suffix = "%" if percent else ""
        return (
            f"{self.point * scale:.{digits}f}{suffix} "
            f"[{self.low * scale:.{digits}f}, {self.high * scale:.{digits}f}]"
        )


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cluster_replicate(cluster_values: Sequence[Sequence[float]], rng: random.Random) -> float:
    """One resample: draw whole clusters with replacement, keep their cases intact.

    Whole clusters, and deliberately **not** a second resampling pass over the
    cases inside them. Adding an inner stage looks more thorough and is simply
    wrong: a cluster's own mean already varies from cluster to cluster because
    of its internal noise, so resampling within a selected cluster counts that
    same within-cluster variance a second time. On independent data it inflates
    the standard error by a factor of sqrt(2) -- an interval that looks
    admirably cautious while being miscalibrated. ``tests/test_stats.py``
    pins the coverage that this distinction buys.

    Clusters may differ in size, so the replicate is the mean over all cases
    drawn (a ratio estimator) rather than an average of cluster means. That
    matches the point estimate, which weights every case equally.
    """
    total = 0.0
    count = 0
    n_clusters = len(cluster_values)
    for _ in range(n_clusters):
        values = cluster_values[rng.randrange(n_clusters)]
        total += sum(values)
        count += len(values)
    return total / count if count else 0.0


def _jackknife_cluster_means(cluster_values: Sequence[Sequence[float]]) -> list[float]:
    """Leave-one-cluster-out means, for the BCa acceleration term."""
    totals = [sum(v) for v in cluster_values]
    sizes = [len(v) for v in cluster_values]
    grand_total, grand_size = sum(totals), sum(sizes)
    out: list[float] = []
    for total, size in zip(totals, sizes):
        remaining = grand_size - size
        out.append((grand_total - total) / remaining if remaining else 0.0)
    return out


def bootstrap_ci(
    observations: Sequence[Observation],
    *,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    method: str = "bca",
) -> Interval:
    """Cluster-bootstrap confidence interval for the mean of `observations`.

    The point estimate is the plain mean over cases (each case weighted
    equally), which is what a reader expects "accuracy" to mean. Only the
    *interval* knows about clustering.
    """
    if method not in {"bca", "percentile"}:
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}")
    if not observations:
        return Interval(0.0, 0.0, 0.0, level=level, method="empty")

    values = [o.value for o in observations]
    point = _mean(values)
    grouped = group_by_cluster(observations)
    cluster_values: list[tuple[float, ...]] = [tuple(v) for v in grouped.values()]
    n_clusters, n_units = len(cluster_values), len(values)

    if n_clusters < 2 or resamples < 2:
        # One cluster gives the bootstrap nothing to resample between, so any
        # interval it produced would be a fabrication. Say so instead.
        return Interval(
            point, point, point, level=level, method="degenerate",
            n_units=n_units, n_clusters=n_clusters,
        )

    rng = random.Random(seed)
    replicates = sorted(_cluster_replicate(cluster_values, rng) for _ in range(resamples))
    std_error = statistics.pstdev(replicates) if len(replicates) > 1 else 0.0

    alpha = 1.0 - level
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    used = "percentile"

    if method == "bca":
        adjusted = _bca_quantiles(replicates, point, cluster_values, alpha)
        if adjusted is not None:
            lo_q, hi_q = adjusted
            used = "bca"

    return Interval(
        point=point,
        low=_quantile(replicates, lo_q),
        high=_quantile(replicates, hi_q),
        level=level,
        method=used,
        n_units=n_units,
        n_clusters=n_clusters,
        std_error=std_error,
    )


def _bca_quantiles(
    replicates: Sequence[float],
    point: float,
    cluster_values: Sequence[Sequence[float]],
    alpha: float,
) -> tuple[float, float] | None:
    """Bias-correction (z0) and acceleration (a). None when undefined."""
    below = sum(1 for r in replicates if r < point)
    fraction = below / len(replicates)
    if fraction <= 0.0 or fraction >= 1.0:
        # Every replicate on one side of the estimate: no usable correction.
        return None
    z0 = normal_ppf(fraction)

    jack = _jackknife_cluster_means(cluster_values)
    jack_mean = _mean(jack)
    deviations = [jack_mean - value for value in jack]
    sum_sq = sum(d * d for d in deviations)
    if sum_sq <= 0:
        return None
    acceleration = sum(d**3 for d in deviations) / (6.0 * sum_sq**1.5)

    z_lo, z_hi = normal_ppf(alpha / 2.0), normal_ppf(1.0 - alpha / 2.0)
    out: list[float] = []
    for z in (z_lo, z_hi):
        denominator = 1.0 - acceleration * (z0 + z)
        if abs(denominator) < 1e-12:
            return None
        out.append(normal_cdf(z0 + (z0 + z) / denominator))
    lo_q, hi_q = out
    if not (0.0 < lo_q < hi_q < 1.0):
        return None
    return lo_q, hi_q


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffResult:
    """A comparison of two runs over the same cases."""

    mean_a: float
    mean_b: float
    delta: float
    paired: Interval
    unpaired: Interval
    n_paired: int
    n_clusters: int
    only_in_a: tuple[str, ...] = ()
    only_in_b: tuple[str, ...] = ()
    mcnemar_p: float | None = None
    discordant: tuple[int, int] = (0, 0)
    mde: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def significant(self) -> bool:
        """Whether the paired interval excludes zero.

        This is the headline claim, and it is deliberately the *paired*
        interval: it is the one that answers "did this change help", rather
        than "are these two numbers far apart".
        """
        return self.paired.excludes_zero

    @property
    def verdict(self) -> str:
        if self.paired.method in {"empty", "degenerate"}:
            return "inconclusive"
        if not self.significant:
            return "no significant difference"
        return "improvement" if self.delta > 0 else "regression"

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "delta": round(self.delta, 6),
            "paired_ci": self.paired.as_dict(),
            "unpaired_ci": self.unpaired.as_dict(),
            "n_paired": self.n_paired,
            "n_clusters": self.n_clusters,
            "only_in_a": list(self.only_in_a),
            "only_in_b": list(self.only_in_b),
            "mcnemar_p": self.mcnemar_p,
            "discordant": {"a_only": self.discordant[0], "b_only": self.discordant[1]},
            "mde": round(self.mde, 6),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "significant": self.significant,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def paired_diff(
    a: Sequence[Observation],
    b: Sequence[Observation],
    *,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    binary: bool = False,
) -> DiffResult:
    """Compare two sets of observations on the cases they share.

    ``a`` is the candidate and ``b`` the baseline, so a positive delta means
    ``a`` is better. Cases present in only one side are excluded from the
    comparison and reported separately -- silently dropping them is how a diff
    ends up comparing different benchmarks.
    """
    index_a = {o.unit: o for o in a}
    index_b = {o.unit: o for o in b}
    shared = sorted(set(index_a) & set(index_b))
    only_a = tuple(sorted(set(index_a) - set(index_b)))
    only_b = tuple(sorted(set(index_b) - set(index_a)))

    notes: list[str] = []
    if only_a or only_b:
        notes.append(
            f"{len(only_a) + len(only_b)} case(s) appear in only one run and were excluded "
            f"from the comparison"
        )

    if not shared:
        empty = Interval(0.0, 0.0, 0.0, level=level, method="empty")
        notes.append("the two runs share no cases; nothing can be compared")
        return DiffResult(
            mean_a=_mean([o.value for o in a]),
            mean_b=_mean([o.value for o in b]),
            delta=0.0,
            paired=empty,
            unpaired=empty,
            n_paired=0,
            n_clusters=0,
            only_in_a=only_a,
            only_in_b=only_b,
            notes=tuple(notes),
        )

    differences = [
        Observation(unit=u, cluster=index_a[u].cluster, value=index_a[u].value - index_b[u].value)
        for u in shared
    ]
    mean_a = _mean([index_a[u].value for u in shared])
    mean_b = _mean([index_b[u].value for u in shared])

    paired = bootstrap_ci(differences, level=level, resamples=resamples, seed=seed)

    # The unpaired interval is computed only to be shown next to the paired
    # one. Its width is the cost of throwing the pairing away.
    unpaired = _unpaired_diff_ci(
        [index_a[u] for u in shared],
        [index_b[u] for u in shared],
        level=level,
        resamples=resamples,
        seed=seed + 1,
    )

    wins = sum(1 for d in differences if d.value > 0)
    losses = sum(1 for d in differences if d.value < 0)
    ties = len(differences) - wins - losses

    mcnemar_p: float | None = None
    discordant = (0, 0)
    if binary:
        b_only = sum(1 for u in shared if index_a[u].value < index_b[u].value)
        a_only = sum(1 for u in shared if index_a[u].value > index_b[u].value)
        discordant = (a_only, b_only)
        mcnemar_p = mcnemar_exact(a_only, b_only)

    z = normal_ppf(1 - (1 - level) / 2) + normal_ppf(DEFAULT_POWER)
    mde = z * paired.std_error

    return DiffResult(
        mean_a=mean_a,
        mean_b=mean_b,
        delta=mean_a - mean_b,
        paired=paired,
        unpaired=unpaired,
        n_paired=len(shared),
        n_clusters=paired.n_clusters,
        only_in_a=only_a,
        only_in_b=only_b,
        mcnemar_p=mcnemar_p,
        discordant=discordant,
        mde=mde,
        wins=wins,
        losses=losses,
        ties=ties,
        notes=tuple(notes),
    )


def _unpaired_diff_ci(
    a: Sequence[Observation],
    b: Sequence[Observation],
    *,
    level: float,
    resamples: int,
    seed: int,
) -> Interval:
    """Bootstrap the difference of two means, resampling each side separately."""
    groups_a = [tuple(v) for v in group_by_cluster(a).values()]
    groups_b = [tuple(v) for v in group_by_cluster(b).values()]
    point = _mean([o.value for o in a]) - _mean([o.value for o in b])
    if len(groups_a) < 2 or len(groups_b) < 2:
        return Interval(point, point, point, level=level, method="degenerate")
    rng = random.Random(seed)
    replicates = sorted(
        _cluster_replicate(groups_a, rng) - _cluster_replicate(groups_b, rng)
        for _ in range(resamples)
    )
    alpha = 1 - level
    return Interval(
        point=point,
        low=_quantile(replicates, alpha / 2),
        high=_quantile(replicates, 1 - alpha / 2),
        level=level,
        method="percentile",
        n_units=len(a) + len(b),
        n_clusters=len(groups_a),
        std_error=statistics.pstdev(replicates) if len(replicates) > 1 else 0.0,
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar test on discordant pairs.

    Only the cases where the two runs disagree carry information about which is
    better; the ones they both get right (or both get wrong) are noise as far as
    the comparison goes. Under the null "a disagreement is equally likely to go
    either way", the count is Binomial(n_discordant, 0.5).

    Exact below 1,000 discordant pairs, normal approximation with a continuity
    correction above -- where the two agree to several decimal places anyway.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    smaller = min(a_only, b_only)
    if n <= 1000:
        tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    else:
        z = (abs(a_only - b_only) - 1) / math.sqrt(n)
        tail = 1.0 - normal_cdf(z)
    return min(1.0, 2.0 * tail)


def required_cases(effect: float, std_dev: float, *, level: float = DEFAULT_LEVEL,
                   power: float = DEFAULT_POWER) -> int:
    """Roughly how many paired cases are needed to detect `effect`.

    Answers the question every eval report should preempt: "you say it is not
    significant -- how much data would settle it?" The answer here assumes
    independent cases, so with clustered tasks treat it as a floor.
    """
    if effect <= 0 or std_dev <= 0:
        return 0
    z = normal_ppf(1 - (1 - level) / 2) + normal_ppf(power)
    return max(1, math.ceil((z * std_dev / effect) ** 2))


def observations_from_scores(
    scores: Mapping[str, Mapping[str, float]],
) -> list[Observation]:
    """Build observations from ``{task_id: {case_id: score}}``."""
    return [
        Observation(unit=f"{task_id}/{case_id}", cluster=task_id, value=float(value))
        for task_id, cases in scores.items()
        for case_id, value in cases.items()
    ]
