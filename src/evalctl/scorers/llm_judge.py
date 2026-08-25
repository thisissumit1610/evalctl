"""Rubric-based LLM judge.

When to reach for this
----------------------
Only when the task has no answer key. Summarisation quality, tone, "did it
follow the format and stay on topic" -- things a regex genuinely cannot see.
If a key exists, grade against the key: a judge costs money, adds latency, and
introduces a *second* noise source you then have to reason about in every
confidence interval downstream.

What this implementation does about the known failure modes
-----------------------------------------------------------
**Vague rubrics produce vague scores.** Criteria are structured -- id,
description, weight -- and every criterion is scored on the same small integer
scale with the anchors spelled out in the prompt. A judge asked for "a score
out of 10" will cheerfully return 7 for everything.

**Judges are noisy near their own boundary.** ``samples: 3`` polls the judge
several times and takes the **median per criterion**. The spread is kept in the
record, so ``evalctl show --disagreement`` surfaces exactly the cases where the
judge could not make up its mind -- which are usually the cases where the
rubric is underspecified.

**Self-preference is real.** A judge tends to favour output from its own family.
The harness cannot fix that, so it does the next best thing: the judge is
configured as its own endpoint, it never learns which model produced the answer,
and the judge's identity is written into the run manifest so a reader can see
the pairing and discount it.

**Reference answers help more than better prompts do.** If a case supplies a
reference, it goes into the prompt and the judge grades against it. Graded
against a reference, a judge is a comparison; graded without one, it is a taste
test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import ScoringError
from .base import Score, Scorer, ScoringContext
from .builtin import parse_json_response

DEFAULT_SCALE = 4
DEFAULT_THRESHOLD = 0.6

JUDGE_SYSTEM = """You are a strict, consistent grader for an automated evaluation harness.

You grade one RESPONSE against a RUBRIC. Follow these rules exactly:
- Judge only what the rubric asks about. Ignore style, length and formatting unless a criterion mentions them.
- A confident, fluent answer that is wrong scores low. Do not reward tone.
- If a REFERENCE answer is given, treat it as correct and grade agreement with it.
- Use the whole scale. Reserve the top score for a response with no defects on that criterion.
- Output a single JSON object and nothing else. No preamble, no code fence, no commentary."""

SCALE_ANCHORS = {
    0: "fails the criterion entirely",
    1: "major problems",
    2: "partly satisfies it",
    3: "satisfies it with minor issues",
    4: "fully satisfies it",
}


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    weight: float = 1.0


class LLMJudgeScorer(Scorer):
    """Score a response against a weighted rubric using a separate model.

    Config::

        - type: llm_judge
          threshold: 0.6              # aggregate needed to pass
          scale: 4                    # integer scale per criterion
          question: "{{ question }}"  # what was asked (optional but recommended)
          reference: "{{ answer }}"   # gold answer, if there is one
          rubric:
            - id: accuracy
              weight: 2
              description: Are all factual claims correct?
            - id: completeness
              description: Does it address every part of the question?
    """

    type = "llm_judge"
    needs_judge = True

    def validate(self) -> None:
        self.reject_unknown(
            ("rubric", "criteria", "threshold", "scale", "question", "reference", "instructions")
        )
        self.criteria = self._parse_rubric()
        if not self.criteria:
            raise ScoringError(f"{self.where}: llm_judge needs a non-empty 'rubric'")
        ids = [c.id for c in self.criteria]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ScoringError(
                f"{self.where}: duplicate rubric criterion id(s): {', '.join(sorted(duplicates))}"
            )
        scale = self.opt_float("scale", DEFAULT_SCALE) or DEFAULT_SCALE
        if scale != int(scale) or not (1 <= int(scale) <= 10):
            raise ScoringError(f"{self.where}: 'scale' must be a whole number in 1..10, got {scale}")
        self.scale = int(scale)
        threshold = self.opt_float("threshold", DEFAULT_THRESHOLD)
        if threshold is None or not (0.0 <= threshold <= 1.0):
            raise ScoringError(f"{self.where}: 'threshold' must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.question = self.opt_str("question")
        self.reference = self.opt_str("reference")
        self.instructions = self.opt_str("instructions")

    def _parse_rubric(self) -> tuple[Criterion, ...]:
        raw = self.config.get("rubric", self.config.get("criteria"))
        if raw is None:
            raise ScoringError(f"{self.where}: llm_judge needs a 'rubric'")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ScoringError(f"{self.where}: 'rubric' must be a list")
        out: list[Criterion] = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out.append(Criterion(id=f"c{i + 1}", description=item))
                continue
            if not isinstance(item, Mapping):
                raise ScoringError(f"{self.where}: rubric[{i}] must be a string or a mapping")
            unknown = sorted(set(item) - {"id", "description", "weight"})
            if unknown:
                raise ScoringError(f"{self.where}: rubric[{i}] has unknown key(s) {unknown}")
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ScoringError(f"{self.where}: rubric[{i}] needs a non-empty 'description'")
            weight = item.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
                raise ScoringError(f"{self.where}: rubric[{i}].weight must be a number >= 0")
            out.append(
                Criterion(
                    id=str(item.get("id", f"c{i + 1}")),
                    description=description.strip(),
                    weight=float(weight),
                )
            )
        return tuple(out)

    # -- prompt -----------------------------------------------------------

    def build_prompt(self, response_text: str) -> str:
        anchors = ", ".join(
            f"{n} = {SCALE_ANCHORS.get(n, 'intermediate')}"
            for n in range(0, self.scale + 1)
            if n in SCALE_ANCHORS or n == self.scale
        )
        lines: list[str] = []
        if self.question:
            lines.append(f"<question>\n{self.question.strip()}\n</question>\n")
        lines.append(f"<response>\n{response_text.strip()}\n</response>\n")
        if self.reference:
            lines.append(f"<reference_answer>\n{self.reference.strip()}\n</reference_answer>\n")
        if self.instructions:
            lines.append(f"<extra_instructions>\n{self.instructions.strip()}\n</extra_instructions>\n")

        lines.append("<rubric>")
        for criterion in self.criteria:
            lines.append(f"- {criterion.id} (weight {criterion.weight:g}): {criterion.description}")
        lines.append("</rubric>\n")

        lines.append(f"Score every criterion with a whole number from 0 to {self.scale}, where {anchors}.")
        schema = ", ".join(f'"{c.id}": <0-{self.scale}>' for c in self.criteria)
        lines.append(
            "Reply with exactly this JSON object and nothing else:\n"
            '{"scores": {' + schema + '}, "rationale": "<one sentence>"}'
        )
        return "\n".join(lines)

    # -- scoring ----------------------------------------------------------

    def _read_scores(self, raw_text: str) -> tuple[dict[str, float], str]:
        try:
            parsed = parse_json_response(raw_text)
        except ValueError as exc:
            raise ScoringError(f"judge did not return JSON: {raw_text[:300]!r}") from exc
        if not isinstance(parsed, Mapping):
            raise ScoringError(f"judge returned {type(parsed).__name__}, expected an object")
        scores = parsed.get("scores")
        if not isinstance(scores, Mapping):
            raise ScoringError(f"judge output has no 'scores' object: {str(parsed)[:300]}")
        out: dict[str, float] = {}
        for criterion in self.criteria:
            if criterion.id not in scores:
                raise ScoringError(f"judge omitted criterion '{criterion.id}'")
            value = scores[criterion.id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScoringError(
                    f"judge gave a non-numeric score for '{criterion.id}': {value!r}"
                )
            # Clamp rather than reject: a judge that returns 5 on a 0-4 scale
            # meant "top marks", and failing the whole trial over it would
            # throw away a usable grade.
            out[criterion.id] = min(float(self.scale), max(0.0, float(value)))
        rationale = parsed.get("rationale") or parsed.get("reason") or ""
        return out, str(rationale)[:600]

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        if ctx.judge is None:
            raise ScoringError(
                f"{self.where}: llm_judge needs a 'judge:' endpoint configured on the suite"
            )
        prompt = self.build_prompt(response_text)
        samples = max(1, int(getattr(ctx.judge, "samples", 1)))

        collected: list[dict[str, float]] = []
        rationales: list[str] = []
        failures: list[str] = []
        for index in range(samples):
            raw = await ctx.judge(
                system=JUDGE_SYSTEM,
                user=prompt,
                sample_index=index,
                cache_tag=f"{ctx.task.id}/{ctx.case.id}/{self.spec.label}",
            )
            try:
                scores, rationale = self._read_scores(raw)
            except ScoringError as exc:
                failures.append(str(exc))
                continue
            collected.append(scores)
            if rationale:
                rationales.append(rationale)

        if not collected:
            # Every poll came back unusable. Surfacing this as an error rather
            # than as a zero is deliberate: a broken judge is a broken
            # measurement, and scoring it 0 would quietly look like a model
            # regression.
            raise ScoringError(
                f"{self.where}: judge produced no usable scores in {samples} attempt(s). "
                f"Last problem: {failures[-1] if failures else 'unknown'}"
            )

        per_criterion: dict[str, float] = {}
        spread: dict[str, float] = {}
        for criterion in self.criteria:
            values = [c[criterion.id] for c in collected]
            per_criterion[criterion.id] = statistics.median(values)
            spread[criterion.id] = (max(values) - min(values)) if len(values) > 1 else 0.0

        total_weight = sum(c.weight for c in self.criteria)
        if total_weight <= 0:
            normalized = sum(per_criterion.values()) / (len(self.criteria) * self.scale)
        else:
            normalized = sum(
                per_criterion[c.id] * c.weight for c in self.criteria
            ) / (total_weight * self.scale)
        value = min(1.0, max(0.0, normalized))

        return Score(
            value=value,
            passed=value >= self.threshold,
            detail={
                "criteria": {
                    c.id: {
                        "score": per_criterion[c.id],
                        "of": self.scale,
                        "weight": c.weight,
                        "spread": spread[c.id],
                    }
                    for c in self.criteria
                },
                "threshold": self.threshold,
                "samples": len(collected),
                "samples_requested": samples,
                "max_spread": max(spread.values()) if spread else 0.0,
                "rationale": rationales[0] if rationales else "",
                "parse_failures": failures[:3],
            },
        )


def judge_disagreement(detail: Mapping[str, Any]) -> float:
    """Largest inter-sample gap on any criterion, normalised to [0, 1].

    Used by ``evalctl show --disagreement`` to rank the cases whose rubric is
    least well specified. High disagreement is a bug report about the rubric,
    not about the model.
    """
    criteria = detail.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        return 0.0
    worst = 0.0
    for entry in criteria.values():
        if not isinstance(entry, Mapping):
            continue
        scale = float(entry.get("of", DEFAULT_SCALE)) or DEFAULT_SCALE
        worst = max(worst, float(entry.get("spread", 0.0)) / scale)
    return worst
