# evalctl

An LLM evaluation harness. Benchmark tasks are YAML specs; `evalctl` runs them
against several model APIs with rate limiting and response caching, scores them
with exact-match rules and a rubric-based LLM judge, and **diffs two runs with
paired bootstrap confidence intervals** so you can say whether a model change is
real or noise.

Runtime dependencies: `httpx` and `PyYAML`. Everything else — spec validation,
templating, tables, and all of the statistics — is the standard library.

```bash
pip install -e .
evalctl run examples/suites/demo.yaml
```

The demo suite runs offline against a deterministic simulated model, so a fresh
clone produces real reports, a real diff and honest intervals **with no API keys
and no network**, in about a second.

---

## The thing this tool exists to prevent

Here is the demo's own output. Two models, 46 cases across 5 tasks, 3 draws each:

```
┌───────────┬───────┬────────┬───────┬───────┬────────┬─────┬───────┬──────┬─────┐
│   model   │ score │ 95% CI │ pass  │ cases │ trials │ err │ cache │ cost │ p50 │
├───────────┼───────┼────────┼───────┼───────┼────────┼─────┼───────┼──────┼─────┤
│ baseline  │ 57.8% │  ±11.6 │ 56.5% │    46 │    138 │   0 │    0% │   $0 │   - │
│ candidate │ 69.7% │   ±9.7 │ 68.1% │    46 │    138 │   0 │    0% │   $0 │   - │
└───────────┴───────┴────────┴───────┴───────┴────────┴─────┴───────┴──────┴─────┘
```

`57.8%` → `69.7%` is a **+11.9 point** jump. Ship it?

```console
$ evalctl diff latest:candidate latest:baseline
diff · metric 'score' · clustered by task
NO SIGNIFICANT DIFFERENCE  (+11.9pp, interval includes zero)

┌─────────────────────────────────────┬──────────────────────┐
│ A  ...:candidate                    │                69.7% │
│ B  ...:baseline                     │                57.8% │
│ delta (A - B)                       │              +11.9pp │
│ 95% CI (paired)                     │     [-0.7, +24.1] pp │
│ 95% CI (unpaired)                   │     [-4.2, +26.9] pp │
│ cases compared                      │  46 across 5 task(s) │
│ A better / B better / tie           │          20 / 8 / 18 │
│ smallest detectable effect          │ ±18.1pp at 80% power │
└─────────────────────────────────────┴──────────────────────┘
```

Five tasks cannot resolve a 12-point difference. The tool says so, and tells you
the smallest effect this suite *could* have detected (±18.1pp) — so the next
question is "add more tasks", not "write the changelog".

Now watch the verdict flip, twice, on **identical data**:

| what changes | 95% CI on the delta | verdict |
|:---|---:|:---|
| paired, clustered by **task** (default) | `[-0.7, +24.1]` | not significant |
| paired, clustered by **case** (`--cluster-by case`) | `[+2.4, +21.9]` | **significant** |
| unpaired, clustered by case | `[-0.5, +24.5]` | not significant |

Two methodology choices — *are cases from one template independent?* and *do you
compare on the same cases?* — each decide whether this ships. Most harnesses make
both silently, in the direction that produces the narrower interval.

`evalctl` makes both explicit, defaults to the conservative reading, and prints
the alternative next to it so the gap is visible. The reasoning is in
[docs/statistics.md](docs/statistics.md); the coverage simulations that back it
are in [tests/test_stats.py](tests/test_stats.py).

---

## A task spec

```yaml
id: math/word-problems
description: Multi-step arithmetic word problems with a single numeric answer.
tags: [math, reasoning]

prompt:
  system: End your reply with the final answer on its own line.
  user: "{{ question }}"

scoring:
  - type: numeric
    target: "{{ answer }}"
    tolerance: 0.001

cases:
  - id: apples
    vars:
      question: A crate holds 24 apples. Maya buys 3 crates and gives away 17. How many are left?
      answer: 55
```

A **suite** says what to run it against:

```yaml
name: production
tasks: [../tasks/*.yaml]

models:
  - id: sonnet
    provider: anthropic
    model: claude-sonnet-5
    params: {temperature: 0.0, max_tokens: 512}
    pricing: {input_per_mtok: 3.0, output_per_mtok: 15.0}

  - id: gpt
    provider: openai
    model: gpt-4o-mini
    params: {temperature: 0.0, max_tokens: 512}
    pricing: {input_per_mtok: 0.15, output_per_mtok: 0.60}

judge:                       # only needed by the llm_judge scorer
  provider: anthropic
  model: claude-sonnet-5
  samples: 3                 # median of 3, so judge noise is measured not hidden

repeats: 1
errors: zero                 # a provider refusal is a wrong answer, not a missing one
limits:
  max_concurrency: 8
  requests_per_minute: 50
  tokens_per_minute: 80000
```

Full schema: [docs/task-spec.md](docs/task-spec.md).

**Unknown keys are errors.** A spec loader that ignores `temperture: 0` will run
your whole benchmark at temperature 1.0 and report the number with a straight
face. Every mapping is checked against a closed key set, and the error names the
file and the dotted path:

```
error: suite.yaml at 'models[0]': unknown key(s): temperture. Allowed here:
       api_key_env, base_url, id, limits, model, params, pricing, provider
```

Template variables are checked against every case at load time too, so a typo in
`{{ quesiton }}` fails in milliseconds rather than after you have paid for 500
generations.

---

## Commands

```bash
evalctl run suite.yaml                    # run it
evalctl run suite.yaml --dry-run          # render prompts, project the bill, call nothing
evalctl run suite.yaml --limit 3          # 3 cases per task, for a smoke test
evalctl run suite.yaml --resume RUN_ID    # continue an interrupted run
evalctl report latest                     # re-render without re-running
evalctl diff latest prev                  # compare two runs
evalctl diff latest:candidate latest:baseline    # compare two models in one run
evalctl show latest --failures -n 5       # read the actual responses
evalctl show latest --disagreement        # cases where the judge could not decide
evalctl validate suite.yaml               # lint specs, no API calls
evalctl cache stats                       # what the cache is holding
evalctl info                              # available providers, scorers, normalizers
```

Every command takes `--format json|md|text`, so reports drop into a PR comment or
a dashboard without scraping.

### In CI

```bash
evalctl run suite.yaml --fail-under 0.80
evalctl diff latest baseline-run --fail-on-regression
```

Exit codes distinguish the two failures a pipeline cares about:

| code | meaning |
|---:|:---|
| 0 | success |
| 1 | something broke — bad spec, unreachable provider, missing run |
| 2 | usage error |
| 3 | the run completed and **failed a gate** |

"The eval says no" and "the eval did not run" are different pages to wake someone
up for.

---

## Design decisions

Full reasoning in [docs/design.md](docs/design.md). The short version:

**The cache is keyed on the request, never on the scoring.** Re-running an eval is
usually about the *harness* — you fixed a normalizer, tightened a rubric, want a
different output format. Regenerating in those cases costs money and, worse,
changes the answers, so a scorer fix and a sampling shift land in the same diff
and you cannot tell which moved the metric. Re-scoring a cached run is free and
exact. `sample_index` is part of the key, so `repeats: 5` is five independent
draws that replay identically — not one answer replayed five times.

**Rate limiting is three mechanisms, not one.** A requests/minute bucket, a
tokens/minute bucket charged from a pre-flight estimate (the limit that actually
bites on long-context suites), and a concurrency semaphore. On a 429 the limiter
pauses **every** worker on that endpoint until the server's own `Retry-After`,
because backing off only the coroutine that got the 429 leaves the other N-1
hammering the endpoint and the backoff never converges.

**The judge is scaffolding around a known-unreliable instrument.** Use it only
when there is no answer key — a judge costs money, adds latency, and introduces a
second noise source into every downstream interval. When you do: criteria are
structured with weights and a small integer scale (a judge asked for "a score out
of 10" returns 7 for everything); `samples: 3` takes the **median per criterion**
and keeps the spread, so `evalctl show --disagreement` surfaces the cases where
the rubric is underspecified; a reference answer turns a taste test into an
agreement measurement; and a broken judge raises rather than scoring zero,
because a failed measurement must not look like a model regression.

**Deterministic scorers do the pass/fail work.** In the summarization example the
cheap `not_contains` guard is `required` and the judge is not — so a low style
score moves the number without ever flipping the bit a regression gate watches.

**Errors have a policy, and it is printed.** A failed trial can count as zero
(right when the failure belongs to the *case* — a prompt the provider refuses
fails every time) or be excluded (right when it belongs to the *run* — a
rate-limit storm is not ignorance). Neither is universally correct, so `evalctl`
refuses to choose silently: the policy lives in the suite, goes into the
manifest, and prints under every report with the error count beside the score.

**Comparability is checked, not assumed.** Each run records a fingerprint of its
tasks and cases. `evalctl diff` compares them, names the tasks that changed,
joins on the cases the runs genuinely share, and warns rather than quietly
averaging two different benchmarks together.

**Normalizers are named and recorded.** On short-answer tasks the gap between a
"45%" harness and a "72%" harness for the *same model* is usually not the model —
it is whether the harness strips a trailing period or unwraps `**42**`. Those are
research choices, so they are configured per scorer, individually tested, and
written into the run manifest instead of hidden in a private cleanup pass.

---

## What runs it

| providers | `anthropic`, `openai`, `openai-compat`, plus `ollama` / `vllm` / `together` / `groq` / `openrouter` aliases, and `mock` |
| :--- | :--- |
| **scorers** | `exact_match`, `contains`, `not_contains`, `regex`, `numeric`, `choice`, `json_match`, `llm_judge` |
| **normalizers** | 19, including `strip_thinking`, `extract_boxed`, `extract_choice`, `extract_last_number`, `strip_code_fence` |

Adding a provider is one method (`complete`) plus a registry entry; adding a
scorer is one method (`score`) plus a registry entry. Nothing else in the
codebase learns about either.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

184 tests, no network, about 20 seconds. The statistics are verified by
simulation rather than by golden values — `test_stats.py` checks that a nominal
95% interval covers the true value about 95% of the time, and that the
cluster-aware interval keeps its coverage on correlated data where the naive
per-case interval drops to ~83%.

---

## Limitations

- **Text in, text out.** No tool-use, multimodal, or agentic-trajectory scoring.
- **No sandboxed code execution scorer.** Deliberate: running model-authored code
  needs an isolation story this project does not have.
- **The judge shares a family with the models it grades**, if you point it at one
  of them. `evalctl` cannot fix self-preference bias; it records the judge's
  identity in the manifest so a reader can see the pairing and discount it.
- **Bootstrap intervals need enough clusters.** Below ~10 tasks the interval is
  wide and honest rather than precise, as the demo shows. That is the data's
  fault, not the estimator's, but it does mean small suites cannot settle small
  differences.
- **The token estimator is `len/4`, not a real tokenizer.** It only feeds the
  pre-flight rate-limiter budget; billed usage always comes from the API
  response.

## License

MIT
