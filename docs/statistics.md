# Statistics: is this change real?

Every number a benchmark reports is an estimate from a sample. This document is
about the three choices that decide how wide the error bar is, each of which can
flip a ship/no-ship verdict, and why `evalctl` defaults the way it does.

The implementation is [`src/evalctl/stats.py`](../src/evalctl/stats.py); the
simulations that verify it are in [`tests/test_stats.py`](../tests/test_stats.py).

---

## 1. What the interval generalises over

A benchmark suite is not a flat bag of independent questions. It is a set of
*tasks*, each holding cases built from one template, sharing a prompt, an output
format and a failure mode. If a model does not understand the phrase "at the same
rate", it gets every rate word-problem wrong at once.

Treating those as independent draws understates the variance. How much depends on
the correlation, but it is easy to measure. Simulating a suite of 20 tasks × 12
cases where task difficulty varies (σ = 0.18), a nominal **95%** interval covers
the true value:

| method | coverage |
|:---|---:|
| resample whole tasks (`--cluster-by task`, default) | **96.0%** |
| resample individual cases (`--cluster-by case`) | **83.3%** |

The naive interval is wrong one time in six while claiming to be wrong one time
in twenty. That is not a rounding error — it is the difference between a gate you
can trust and one that greenlights noise.

So the default resamples **whole tasks with replacement**, keeping each drawn
task's cases intact. The claim it supports is "on another benchmark built like
this one", which is what a suite of templated cases can actually justify.

`--cluster-by case` is still offered, for two reasons: sometimes your cases
genuinely are unrelated draws, and the *gap* between the two intervals is
information. If the task-clustered interval is much wider, your effective sample
size is far below your case count, and the fix is more tasks, not more cases per
task.

### The extra stage that looks careful and is a bug

An obvious-seeming refinement: after drawing a task, resample the cases *within*
it too. "Two levels of variation, two levels of resampling."

It is wrong, and it is wrong in the flattering direction. A task's own mean
already varies from draw to draw because of its internal noise — that variation
is captured the moment you resample tasks. Adding an inner pass counts it twice.
On independent data it inflates the standard error by a factor of **√2**:

```
20 tasks × 12 cases, iid, p = 0.70
analytic SE      = sqrt(0.7 × 0.3 / 240)  = 0.0296
one-stage (correct)                        = 0.0319   ← the small excess is the
two-stage  (wrong)                         = 0.0428      finite-cluster penalty
```

An interval 45% too wide looks admirably cautious and is simply miscalibrated.
This project shipped the two-stage version first; the analytic comparison in
`test_bootstrap_standard_error_matches_analytic_on_independent_data` is what
caught it, and is there to stop it coming back.

---

## 2. Pairing

Two runs over the same suite see the same cases. The per-case difference

```
d_i = score_A(i) − score_B(i)
```

cancels case difficulty entirely: a question both models fail contributes 0 to
the delta and nothing to its variance. Bootstrapping `d_i` rather than comparing
two independent means routinely shrinks the interval by 2–5×.

`evalctl diff` reports both, always:

```
95% CI (paired)     [-0.7, +24.1] pp
95% CI (unpaired)   [-4.2, +26.9] pp
```

The paired one is the answer. The unpaired one is printed so the cost of throwing
the pairing away is visible rather than theoretical.

### The overlap test is worse than both

The common shortcut — "the two models' confidence intervals overlap, so the
difference isn't significant" — is not a conservative approximation. It is a
*different and stricter* test: two 95% intervals can overlap while the difference
is significant at the same level. Combined with unpaired data, it is the reason
real improvements get discarded as noise.

Cases present in only one run are excluded from the comparison and reported
separately. Silently dropping them is how a diff ends up comparing two different
benchmarks.

---

## 3. Bias correction

Accuracy near a ceiling is skewed — there is more room below 0.95 than above it —
and a plain percentile interval inherits that skew as a coverage error. `evalctl`
uses **BCa** (bias-corrected and accelerated): a bias term `z₀` from the fraction
of bootstrap replicates below the point estimate, and an acceleration term from a
jackknife over clusters.

It costs one extra pass and degrades gracefully: when the correction is
undefined — every replicate identical, a single cluster — it falls back to the
percentile interval and says so in `method`. Measured coverage on iid binary data
is 97.0% for BCa and 95.3% for percentile at a nominal 95%; BCa is slightly
conservative here because binary outcomes are discrete.

---

## 4. Supporting numbers

**McNemar's exact test** (`--metric pass`) works only on the cases where the two
runs *disagree*. Cases both get right, or both get wrong, carry no information
about which is better. Under the null "a disagreement is equally likely to go
either way", the count is Binomial(n_discordant, ½). Exact below 1,000 discordant
pairs, normal approximation with continuity correction above.

**Minimum detectable effect.** Every diff prints the smallest effect this suite
could have detected at 80% power. When a result is not significant, the immediate
next question is "how much data would settle it?", and a report that does not
preempt it invites the reader to squint at the point estimate instead.

**Reproducibility.** Pure Python with a seeded `random.Random`, no numpy. Not for
purity — a bootstrap whose numbers depend on which BLAS is installed is one you
cannot reproduce from a CI log six months later. 5,000 replicates over a few
hundred cases takes about a quarter of a second.

---

## What this still does not do

- **No multiple-comparison correction.** Comparing one candidate against one
  baseline is fine. Scanning ten prompt variants and reporting the best one at
  p < 0.05 is not, and `evalctl` will not stop you. Pre-register the comparison
  you care about, or apply a correction yourself.
- **Per-task breakdowns have no intervals.** With 8–12 cases each they would be
  too wide to act on, and showing twelve overlapping intervals invites exactly
  the overlap-eyeballing this document argues against. Treat the per-task table
  as a place to look for causes, not as evidence.
- **Repeats do not widen the interval.** They average within a case first, which
  reduces per-case measurement noise. The interval is over which *cases and
  tasks* you happened to write — the population you are generalising to. More
  repeats sharpen each measurement; only more tasks narrow the claim.
