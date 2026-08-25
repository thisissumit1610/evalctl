# Spec reference

Two document kinds: **task files** (what to ask, how to grade) and **suite files**
(what to run it against). Both are validated against a closed key set — an
unrecognised key is an error, not a shrug.

Check any spec without spending anything:

```bash
evalctl validate examples/suites/demo.yaml
```

---

## Task file

```yaml
id: math/word-problems          # required, unique across the suite
description: Multi-step arithmetic.
tags: [math, reasoning]         # filter with `--tag`

prompt:
  system: End your reply with the final answer on its own line.
  user: "{{ question }}"

params:                         # optional per-task sampling overrides
  max_tokens: 256

pass_threshold: 0.8             # optional: aggregate score also needed to pass

scoring:
  - type: numeric
    target: "{{ answer }}"
    tolerance: 0.001

cases:
  - id: apples
    tags: [arithmetic]
    vars:
      question: A crate holds 24 apples...
      answer: 55
```

A file may hold one task, a list of tasks, or `{tasks: [...]}`.

### Prompts

Either `user` (single turn) or `messages` (multi-turn) — not both. The system
prompt always goes in `prompt.system`, never as a `messages` role, and the last
message must be `user`.

```yaml
prompt:
  system: You are terse.
  messages:
    - {role: user, content: "{{ setup }}"}
    - {role: assistant, content: "Understood."}
    - {role: user, content: "{{ question }}"}
```

### Templating

`{{ name }}` substitutes a case variable. Filters chain left to right:
`{{ items | json }}`, `{{ n | int }}`, `{{ s | upper | strip }}`. Available:
`json`, `json_pretty`, `upper`, `lower`, `strip`/`trim`, `int`, `len`.
`\{{ name }}` emits a literal `{{ name }}`.

A missing variable is a **hard error at load time**, never an empty string — a
silently blank prompt scores as a legitimate wrong answer and poisons the run.
Every task template and every scorer target is checked against every case before
the first API call.

**A template that is exactly one placeholder keeps the value's type.** This is
what lets a list variable reach a list option:

```yaml
scoring:
  - type: exact_match
    target: "{{ answer }}"      # a string
    any_of: "{{ aliases }}"     # the list itself, not "['Berne']"
```

Embed the tag in surrounding text and you get a string, as you must:
`"answer is {{ answer }}"`.

### Cases

`id` (defaults to `case-000`), `vars`, optional `tags`, and an optional `scoring`
list that replaces the task's for that case alone.

---

## Scorers

Common to all: `type`, `weight` (default 1.0), `required` (default true), `name`
(for the report), `normalize` (a list of normalizer names).

**`weight`** sets the share of the task's aggregate score.
**`required`** decides whether the scorer can veto a pass. Set it `false` for a
style rubric you want reflected in the number but not in the pass/fail bit.

### `exact_match`

```yaml
- type: exact_match
  target: "{{ answer }}"
  any_of: ["Berne", "Bern"]     # any one of these counts
  case_sensitive: false          # default true
```

Normalizers apply to the target as well as the response.

### `contains` / `not_contains`

```yaml
- type: contains
  any_of: [alpha, beta]          # any one -> pass
  all_of: [x, y, z]              # partial credit by fraction found
  case_sensitive: false          # default false here
```

`not_contains` inverts both value and verdict — for refusal checks and
leaked-answer guards.

### `numeric`

```yaml
- type: numeric
  target: "{{ answer }}"
  tolerance: 0.001               # absolute
  rel_tolerance: 0.01            # relative; passing either is enough
  extract: last                  # last | first | all | strict
```

Defaults to the **last** number in the response, because models show their
working: "12 × 4 = 48, minus 6, so 42" must score as 42. `strict` requires
exactly one number in the response — use it when a stray figure would be
ambiguous. Handles `1,234`, scientific notation, and `\boxed{17}`.

### `choice`

```yaml
- type: choice
  target: "{{ answer }}"         # "C"
  options: [A, B, C, D]
```

Takes the **last** standalone letter, so "I said A, but actually C" resolves to
C. Separate from `exact_match` because the extraction rule *is* the problem.

### `regex`

```yaml
- type: regex
  pattern: "ID-(\\d+)"
  flags: [i, m]                  # i/ignorecase, m/multiline, s/dotall, x/verbose
  group: 1                       # index or name
  target: "{{ id }}"             # compare the capture; omit to just require a match
  expect: true                   # false = the pattern must NOT match
```

### `json_match`

```yaml
- type: json_match
  target: '{"a": 1, "b": 2}'     # JSON string or inline YAML mapping
  mode: subset                   # subset (ignore extra keys) | exact
  partial_credit: true           # score by fraction of expected leaves matched
```

Finds JSON wrapped in prose, unwraps code fences, and treats `1` and `1.0` as
equal. Mismatches are reported as paths (`$.b: expected 2, got 9`) so a failure
says which field was wrong.

### `llm_judge`

Requires a `judge:` endpoint on the suite.

```yaml
- type: llm_judge
  name: quality
  required: false                # usually: let the judge move the score, not the gate
  threshold: 0.6                 # aggregate needed to pass
  scale: 4                       # integer scale per criterion, 1..10
  question: "{{ question }}"     # what was asked
  reference: "{{ answer }}"      # gold answer, if there is one
  instructions: Ignore British vs American spelling.
  rubric:
    - id: faithfulness
      weight: 2
      description: Is every claim supported by the passage?
    - id: brevity
      description: Is it a single sentence?
```

A bare list of strings works as shorthand: `rubric: ["Is it correct?", "Is it
short?"]`.

Score = weighted mean of criteria, normalised by `scale`. With `judge.samples: 3`
the **median per criterion** is used and the spread is kept — see
`evalctl show --disagreement`.

---

## Normalizers

Applied left to right, to the response and the target alike.

| | |
|:---|:---|
| `strip`, `collapse_whitespace` | whitespace |
| `lower`, `upper`, `normalize_unicode` | case and NFKC folding |
| `strip_punctuation` | leading/trailing only — never interior |
| `strip_articles`, `strip_trailing_period`, `strip_quotes` | small edits |
| `strip_markdown` | unwraps `**42**`, `***42***` |
| `strip_code_fence` | unwraps a fenced block |
| `strip_thinking` | drops `<think>...</think>` |
| `strip_answer_prefix` | drops "The answer is" / "Answer:" |
| `first_line`, `last_line` | line selection |
| `extract_boxed` | last `\boxed{...}` |
| `extract_first_number`, `extract_last_number` | numbers |
| `extract_choice` | multiple-choice letter |

Omitting `normalize` uses a conservative default chain (`strip_thinking`,
`strip_code_fence`, `strip_markdown`, `strip_answer_prefix`, `strip_quotes`,
`collapse_whitespace`, `strip_punctuation`). It removes packaging around an
answer but never touches the answer's own characters, so it cannot turn a wrong
response into a right one. Pass `normalize: []` to compare raw text.

---

## Suite file

```yaml
name: production
description: Five tasks against two live models.

tasks:                          # files, directories, or globs, relative to THIS file
  - ../tasks/*.yaml

models:
  - id: sonnet                  # the label in reports and both sides of a diff
    provider: anthropic
    model: claude-sonnet-5
    params: {temperature: 0.0, max_tokens: 512}
    api_key_env: ANTHROPIC_API_KEY     # optional override
    base_url: https://api.anthropic.com
    pricing: {input_per_mtok: 3.0, output_per_mtok: 15.0}
    limits: {max_concurrency: 4}       # optional per-model override

judge:
  provider: anthropic
  model: claude-sonnet-5
  samples: 3                    # must be odd
  params: {temperature: 0.0, max_tokens: 400}

params: {temperature: 0.0}      # suite-wide defaults
repeats: 1
seed: 11
errors: zero                    # zero | exclude
cache: true
pass_threshold: 0.7             # suite-wide default for tasks

limits:
  max_concurrency: 8
  requests_per_minute: 50       # 0 = unlimited
  tokens_per_minute: 80000
  max_retries: 5
  timeout_s: 120
  initial_backoff_s: 1.0
  max_backoff_s: 60
```

Sampling params merge most-specific-wins: **suite < task < model**.

### Providers

| `provider` | needs | notes |
|:---|:---|:---|
| `anthropic` | `ANTHROPIC_API_KEY` | Messages API; `max_tokens` defaults to 1024 |
| `openai` | `OPENAI_API_KEY` | Chat Completions |
| `openai-compat` | `base_url` | any Chat Completions server |
| `ollama` | — | alias, `http://localhost:11434` |
| `vllm` | — | alias, `http://localhost:8000` |
| `together` / `groq` / `openrouter` | their key env | aliases with base URLs filled in |
| `mock` | — | offline deterministic model |

Params outside `temperature`, `max_tokens`, `top_p`, `stop`, `seed` are forwarded
verbatim, so a provider-specific flag works without a code change.

### Mock provider params

| param | default | meaning |
|:---|---:|:---|
| `ability` | 0.6 | expected accuracy — `E[accuracy] = ability` exactly |
| `spread` | 0.7 | how much per-case difficulty moves it |
| `verbosity` | 0.0 | chance of wrapping a right answer in prose |
| `error_rate` | 0.0 | chance of raising a transient error |
| `latency_ms` | 0.0 | simulated delay |
| `seed` | 0 | shifts every draw |
| `judge_noise` | 0.25 | share of criteria that wobble between judge samples |
