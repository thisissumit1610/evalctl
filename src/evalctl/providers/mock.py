"""A deterministic offline model, used for tests and for the demo suite.

Why a fake model is a first-class provider here
-----------------------------------------------
Two reasons, and neither is "so the tests pass":

1. **The statistics need a ground truth.** Bootstrap intervals and paired
   diffs are easy to write and hard to verify against a live API, where you
   cannot re-run the same experiment twice. With a generator whose true
   accuracy you set by hand, you can check that a 95% interval covers the
   real value about 95% of the time. ``tests/test_stats.py`` does exactly that.

2. **The demo has to run for someone with no API keys.** ``evalctl run
   examples/suites/demo.yaml`` produces a full report, a real diff and honest
   confidence intervals on a fresh clone, offline, in about a second.

The generative model
--------------------
Each case gets a latent *difficulty* ``d`` from a hash of its prompt, stable
across models. Each model has an *ability* ``a``. The chance of a correct
answer on a given draw is::

    p = clip(a + spread * (0.5 - d), 0, 1)

so ``E[accuracy] = a`` exactly, while hard cases stay hard for every model.
That correlation is the point: it is what makes the *paired* analysis in
``stats.py`` visibly tighter than the unpaired one, which is the whole argument
for pairing. Per-draw noise is independent, so ``repeats`` behaves the way real
sampling noise does.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from ..errors import TransientError
from .base import ChatRequest, ChatResponse, Provider, Usage
from ..util import estimate_tokens

DEFAULT_ABILITY = 0.6
DEFAULT_SPREAD = 0.7


# Patterns for reading a judge prompt back apart. They mirror the shapes that
# scorers/llm_judge.py emits, and nothing else in the package depends on them.
_CRITERION_ID = re.compile(r'"(\w+)": <0-')
_SCALE = re.compile(r"whole number from 0 to (\d+)")
_RESPONSE_BLOCK = re.compile(r"<response>(.*?)</response>", re.DOTALL)


def unit_hash(*parts: Any) -> float:
    """A deterministic float in [0, 1) from arbitrary parts.

    Uses the top 64 bits of SHA-256 rather than :func:`hash`, whose string
    seed is randomised per process -- a fake model that changes its answers
    between runs would defeat the entire purpose.
    """
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _wrong_answer(expected: str, salt: float) -> str:
    """A plausible near-miss, never a crash or an empty string.

    Wrong answers that look obviously broken make a harness feel healthier than
    it is. Real failures are believable, so these are too.
    """
    text = expected.strip()
    try:
        value = float(text)
    except ValueError:
        pass
    else:
        # Off-by-one, sign flip, factor of ten: the arithmetic slips a model
        # actually makes.
        offsets = [1, -1, 2, -2, 10, -10]
        delta = offsets[int(salt * len(offsets)) % len(offsets)]
        shifted = value + delta
        if float(shifted).is_integer() and "." not in text:
            return str(int(shifted))
        return f"{shifted:g}"

    if not text:
        return "unknown"
    words = text.split()
    if len(words) > 2 and salt < 0.5:
        return " ".join(words[:-1])  # truncated answer
    alternatives = ["none of the above", "unknown", "not enough information", "42"]
    return alternatives[int(salt * len(alternatives)) % len(alternatives)]


class MockProvider(Provider):
    """Offline stand-in. Params:

    ``ability``      expected accuracy in [0, 1]           (default 0.6)
    ``spread``       how much case difficulty moves it     (default 0.7)
    ``verbosity``    chance of wrapping a right answer in
                     prose, to exercise the normalizers    (default 0.0)
    ``error_rate``   chance of raising a transient error   (default 0.0)
    ``latency_ms``   simulated per-call delay              (default 0.0)
    ``seed``         shifts every draw                     (default 0)
    """

    name = "mock"
    requires_api_key = False
    default_base_url = None
    uses_metadata = True

    async def complete(self, request: ChatRequest) -> ChatResponse:
        params = dict(request.params)
        if request.role == "judge":
            return self._judge(request, params)
        ability = float(params.get("ability", DEFAULT_ABILITY))
        spread = float(params.get("spread", DEFAULT_SPREAD))
        verbosity = float(params.get("verbosity", 0.0))
        error_rate = float(params.get("error_rate", 0.0))
        latency_ms = float(params.get("latency_ms", 0.0))
        seed = params.get("seed", 0)

        prompt = request.prompt_text
        draw_key = (prompt, self.model, request.sample_index, seed, request.role)

        if error_rate > 0 and unit_hash("error", *draw_key) < error_rate:
            # Exercised by the retry path in tests: a transient error the
            # runner is expected to retry and eventually give up on.
            raise TransientError(f"mock provider: simulated transient failure ({self.model})")

        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)

        expected = str(request.metadata.get("expected", "")) if request.metadata else ""
        difficulty = unit_hash("difficulty", prompt)
        p_correct = min(1.0, max(0.0, ability + spread * (0.5 - difficulty)))
        correct = unit_hash("draw", *draw_key) < p_correct

        if not expected:
            # Nothing to imitate: echo something stable so judge-only tasks and
            # smoke tests still get deterministic text.
            text = f"[mock:{self.model}] {prompt.strip()[-120:]}"
        elif correct:
            text = expected
            if verbosity > 0 and unit_hash("verbose", *draw_key) < verbosity:
                text = f"The answer is {expected}."
        else:
            text = _wrong_answer(expected, unit_hash("wrong", *draw_key))

        return ChatResponse(
            text=text,
            usage=Usage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(text),
            ),
            finish_reason="stop",
            model=self.model,
            response_id=f"mock-{unit_hash(*draw_key):.12f}",
            latency_ms=latency_ms,
        )

    def _judge(self, request: ChatRequest, params: dict[str, Any]) -> ChatResponse:
        """Answer a rubric prompt with well-formed judge JSON.

        Criterion ids are read back out of the prompt the judge scorer built,
        which keeps this honest: the fake judge only knows what a real judge
        would be told. Scores are deterministic per (criterion, response), and
        ``judge_noise`` makes a share of them wobble between samples so the
        median-of-N path and the disagreement report have something real to
        chew on.
        """
        prompt = request.prompt_text
        criteria = _CRITERION_ID.findall(prompt) or ["quality"]
        scale = _SCALE.search(prompt)
        top = int(scale.group(1)) if scale else 4
        graded = _RESPONSE_BLOCK.search(prompt)
        subject = graded.group(1).strip() if graded else prompt
        noise = float(params.get("judge_noise", 0.25))
        severity = float(params.get("judge_severity", 0.0))

        scores: dict[str, int] = {}
        for criterion in criteria:
            base = unit_hash("judge", criterion, subject, params.get("seed", 0))
            value = base
            if unit_hash("wobble", criterion, subject) < noise:
                # This criterion is one the judge is unsure about: let the draw
                # move with the sample index.
                value = unit_hash("judge", criterion, subject, request.sample_index)
            scores[criterion] = max(0, min(top, round(value * top + 0.5 - severity)))

        payload = {
            "scores": scores,
            "rationale": f"Graded {len(criteria)} criteria against the reference.",
        }
        text = json.dumps(payload)
        return ChatResponse(
            text=text,
            usage=Usage(estimate_tokens(prompt), estimate_tokens(text)),
            finish_reason="stop",
            model=self.model,
            response_id=f"mock-judge-{unit_hash(subject, request.sample_index):.12f}",
        )
