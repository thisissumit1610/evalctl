# Design notes

Why the pieces are shaped the way they are. Statistics has [its own
document](statistics.md); this covers everything else.

```
suite.yaml ──▶ spec ──▶ runner ──▶ providers ──▶ cache ──▶ store ──▶ analysis ──▶ report
                 │         │                                            │
              scorers   limiter                                       stats
                 │
              judge
```

Each arrow is a module boundary that something real crosses: the runner never
learns which provider it is talking to, scorers cannot reach the cache, and the
statistics never see a `TrialRecord`.

---

## Dependency budget

Two runtime dependencies: `httpx` (async HTTP) and `PyYAML`. Spec validation,
templating, table rendering and every statistic are stdlib.

This is a deliberate trade, not minimalism for its own sake. An eval harness gets
installed in CI images, on a colleague's laptop, and inside whatever container
happens to be running a training job. Each dependency is a version to pin, a
transitive tree to audit, and a reason `pip install` fails somewhere you cannot
debug. Against that: pydantic would have saved perhaps 150 lines of validation
whose error messages I would then not control, and `rich` would have saved 90
lines of table rendering.

The validation is the clearer win. `evalctl` needs errors like

```
suite.yaml at 'models[0]': unknown key(s): temperture. Allowed here: api_key_env,
base_url, id, limits, model, params, pricing, provider
```

— naming the file, the dotted path, and the legal alternatives. Hand-written
accessors give that directly.

---

## Caching

The cache key covers everything that determines the model's output: provider,
base URL, model, system prompt, messages, sampling params, `sample_index`, and a
schema version. It deliberately excludes anything about *scoring*.

That exclusion is the whole point. Re-running an eval is usually about the
harness rather than the model — you fixed a normalizer, tightened a rubric,
want markdown output. Regenerating in those cases costs money and, worse,
*changes the answers*, so a scorer fix and a sampling shift land in the same diff
and nothing can tell you which one moved the metric. Re-scoring a cached run is
free and exact.

**`sample_index` is part of the key.** With `repeats: 5` at non-zero temperature
you want five independent draws, not one answer replayed five times. Keying on
the index gives each draw its own slot, so a re-run replays a stochastic
experiment exactly.

**Metadata is withheld from real providers.** The offline mock needs to know the
expected answer; a real provider must not have it in its key, or editing an
answer key would invalidate cache entries for calls whose prompts never contained
the answer. `Provider.uses_metadata` gates it.

**SQLite, not a directory of JSON files.** One file to copy or delete, atomic
writes under concurrency, a real index so `cache stats` does not stat 200k files,
and WAL mode so a second process can read while a run writes.

**Staleness is a real hazard.** Model aliases move: `gpt-4o` and
`claude-sonnet-5` point at different weights over time while the cache key stays
identical, so an unbounded cache will serve you last quarter's model and call it
today's result. Pin exact versioned model ids where you can; use `--cache-ttl 7d`
where you cannot.

---

## Rate limiting

Three mechanisms, because provider limits come in three shapes:

- **requests/minute** — a token bucket at request granularity.
- **tokens/minute** — a second bucket, charged with an estimated prompt size
  before the call. This is the limit that actually bites on long-context suites
  and the one most harnesses ignore until a run dies at 80%.
- **max concurrency** — a semaphore. Buckets bound the *average* rate; they do
  not stop 200 coroutines firing in the same millisecond after a refill.

Two details that matter more than the mechanism:

**The bucket does not start with a minute of burst.** A bucket seeded with 60
tokens at 60/min lets the first 60 requests fire instantly — exactly the burst
that trips server-side limits even though the average rate is legal. Capacity
defaults to one second of refill.

**A 429 pauses every worker on that endpoint, not just the one that got it.**
Backing off a single coroutine leaves the other N−1 hammering the endpoint and
collecting their own 429s; the backoff never converges and you look like an
attacker. The limiter holds a shared deadline, prefers the server's own
`Retry-After` over a guess, and extends rather than stacks when several 429s land
at once. Retries use full jitter, because N workers backing off by the same
amount retry in the same instant and reproduce the original burst.

Limiters are keyed by provider + model + base URL, not by the suite's model id:
two entries for the same model with different sampling params share one
server-side quota and must share one limiter.

---

## The retry taxonomy

Providers raise into three buckets, and getting this classification right is most
of what the provider layer is for:

| | retried? | examples |
|:---|:---|:---|
| `RateLimited` | yes, gated | 429 |
| `TransientError` | yes, backed off | 5xx, timeout, connection reset, non-JSON body |
| `FatalError` | **no** | 401, 400, unknown model, missing API key |

A misclassified 400 burns five retries per case across the whole suite. A
misclassified 503 fails a run that would have succeeded on the next attempt.

Credentials are checked **before** the first trial, for every endpoint including
the judge. Discovering a missing key on trial 400 of 500 is exactly the avoidable
waste this harness exists to prevent.

---

## Scoring

Every scorer returns a value in `[0, 1]` **and** an independent `passed` bool.
Keeping them separate matters: a rubric judge can say 0.72 without that meaning
"pass", and an exact match gives 1.0 or 0.0 where the bool is the only honest
summary.

A task's verdict combines them in exactly one place:

- **score** = weighted mean of component values.
- **passed** = every `required` component passed, *and* the score clears
  `pass_threshold` if the task sets one.

"All required scorers must pass" rather than "the weighted score clears a bar" is
the conservative reading, and it is what makes `required: false` useful: a style
rubric can drag the number down without ever flipping the bit a regression gate
watches. The summarization example uses this shape — a cheap deterministic
`not_contains` guard is required, the judge is not.

### Normalizers

On short-answer tasks, the gap between a "45%" harness and a "72%" harness for
the *same model* is usually not the model. It is whether the harness strips a
trailing period, unwraps `**42**`, or pulls the answer out of "The answer is 42."

Those are research choices that change the headline number, so they are named,
configured per scorer, individually tested, and recorded in the run manifest —
not buried in a scorer's private cleanup pass. The default chain is deliberately
conservative: it removes *packaging* around an answer but never touches the
answer's own characters, so it cannot turn a wrong response into a right one.
`strip_punctuation` trims only leading and trailing punctuation, because
stripping interior punctuation turns `3.14` into `314`.

Normalizers apply to the **target as well as the response**, so a spec cannot
compare a cleaned-up response against a raw answer key and lose points on
formatting it already agreed to ignore.

---

## The judge

Use it only when the task has no answer key. If a key exists, grade against the
key: a judge costs money, adds latency, and introduces a second noise source you
then have to reason about in every interval downstream.

When you do need one, the implementation targets the known failure modes:

**Vague rubrics produce vague scores.** Criteria are structured — id, description,
weight — and every criterion is scored on the same small integer scale with the
anchors spelled out in the prompt. A judge asked for "a score out of 10" returns
7 for everything.

**Judges are noisy near their own boundary.** `samples: 3` polls the judge
several times and takes the **median per criterion**; the spread is kept in the
record. `evalctl show --disagreement` ranks the cases where the judge could not
decide, which are almost always cases where the *rubric* is underspecified. The
spec requires an odd `samples` so the median is a single observed score.

**A reference answer helps more than a better prompt.** Graded against a
reference, a judge is measuring agreement. Graded without one, it is a taste
test. Reference answers go in the prompt when the case supplies one.

**Self-preference is real.** A judge favours output from its own family. The
harness cannot fix that, so it does the next best thing: the judge is a separate
endpoint, it never learns which model produced the answer it is grading, and its
identity is written into the run manifest so a reader can see the pairing and
discount it.

**A broken judge is a broken measurement.** If every poll comes back unparseable,
the scorer raises rather than scoring zero — a zero would look exactly like a
model regression. The trial is recorded as an error and the error policy applies.

The judge runs through the same cache, limiter and cost accounting as any model
under test. An uncached judge is the usual reason a "cheap" eval turns out to
cost more than the generations did.

---

## Run artifacts

```
runs/20260825-121239-demo-3745c8/
  manifest.json   provenance: suite fingerprint, models, limits, git sha, version
  records.jsonl   one line per trial, appended and flushed as it completes
  summary.json    aggregates, written at the end
```

**JSONL, appended and flushed per trial.** Long runs get interrupted — a laptop
sleeps, a token expires, someone hits Ctrl-C watching the cost tick up. One
self-describing line per trial means an interrupted run is still a valid dataset
and `--resume` is just "read the keys already present and skip them". Accumulating
in memory turns every interruption into a total loss of work already paid for.
The flush costs a syscall per trial; the trade is not close.

Errored trials are deliberately **not** counted as complete, so a resume retries
them — the usual reason a run has errors is a transient outage that has since
cleared. A torn final line from a hard kill is skipped rather than fatal.

**The manifest answers "can I trust this comparison?"** It carries a fingerprint
of the tasks and cases, per-task fingerprints under it, exact model params, the
harness version and the git SHA. `evalctl diff` reads both sides, names the tasks
that changed, joins on the cases the runs genuinely share, and warns instead of
quietly averaging two different benchmarks.

The suite fingerprint covers *what was measured* — tasks and cases only. It
deliberately excludes models, limits and repeats: swapping the model is the entire
point of a diff.

---

## Errors

A failed trial is recorded and the run continues. Aborting on one bad case throws
away everything already paid for, and failures are usually a property of a case
(a prompt that trips a filter) rather than of the run.

How errors *count* is a separate decision, made at analysis time rather than in
the execution path, because it materially changes the headline number:

- **`zero`** — an error is a wrong answer. Right when the failure belongs to the
  case: a prompt the provider refuses will fail every time, and hiding it
  flatters the model.
- **`exclude`** — the case is dropped. Right when the failure belongs to the run:
  a rate-limit storm should not be scored as ignorance.

Neither is universally correct, so it is never chosen silently. The policy comes
from the suite, goes into the manifest, prints under every report, and the error
count sits next to the score. Above a 5% error rate the report says the scores
are provisional.

---

## The mock provider

A fake model is a first-class provider here, for two reasons that are not "so the
tests pass".

**The statistics need a ground truth.** Bootstrap intervals are easy to write and
hard to verify against a live API, where you cannot re-run the same experiment
twice. With a generator whose true accuracy you set by hand, you can check that a
95% interval covers the real value about 95% of the time.

**The demo has to run for someone with no API keys.** A portfolio repo whose
first command fails on a missing environment variable is a repo nobody evaluates.

The generative model is chosen so the demo teaches something true. Each case has
a latent difficulty `d` from a hash of its prompt, stable across models; each
model has an ability `a`; the chance of a correct draw is
`clip(a + spread × (0.5 − d))`. So `E[accuracy] = a` exactly, hard cases stay
hard for every model — which is what makes the paired analysis visibly tighter
than the unpaired one — and per-draw noise stays independent so `repeats` behaves
like real sampling noise. Wrong answers are believable near-misses (off-by-one,
factor of ten, truncation), because a harness that only ever sees obviously
broken failures looks healthier than it is.

---

## Things deliberately left out

- **A code-execution scorer.** Running model-authored code needs a real isolation
  story — container, seccomp, network policy, resource caps. A `subprocess.run`
  with a timeout is not that, and shipping one under an innocuous name would be
  worse than not shipping it.
- **Streaming.** Nothing here needs partial output, and it would complicate the
  cache and usage accounting for no benchmark benefit.
- **A web UI.** `--format json` exists so the data goes somewhere that already
  has one.
- **Tool-use and agentic trajectory scoring.** A different measurement problem —
  the unit stops being one response — and bolting it onto this data model would
  compromise the parts that work.
