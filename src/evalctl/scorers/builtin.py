"""Deterministic scorers: exact match, containment, regex, numeric, choice, JSON.

These are the ones that should carry most of a benchmark. A judge is for
questions with no key; if a task *has* a key, grading it with a model adds cost,
latency and a second source of noise for no gain.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from ..errors import ScoringError
from . import normalize as norm
from .base import Score, Scorer, ScoringContext

_COMMON_KEYS = ("normalize",)


class _NormalizingScorer(Scorer):
    """Shared normalizer handling.

    The same pipeline runs over the response *and* the target, so a spec cannot
    compare a cleaned-up response against a raw answer key and lose points on
    formatting it already agreed to ignore.
    """

    default_normalizers: tuple[str, ...] = norm.DEFAULT_NORMALIZERS

    def validate(self) -> None:
        names = self.config.get("normalize")
        if names is None:
            self.normalizer_names = list(self.default_normalizers)
        else:
            self.normalizer_names = self.str_list("normalize")
        self.normalizers = norm.resolve(self.normalizer_names, where=self.where)

    def clean(self, text: str) -> str:
        return norm.apply(text, self.normalizers)


class ExactMatchScorer(_NormalizingScorer):
    """1.0 when the normalized response equals a normalized target.

    ``any_of`` accepts several correct phrasings, which is how you keep a
    free-text question fair without reaching for a judge.
    """

    type = "exact_match"

    def validate(self) -> None:
        super().validate()
        self.reject_unknown((*_COMMON_KEYS, "target", "any_of", "case_sensitive"))
        targets = self.str_list("any_of") if "any_of" in self.config else []
        if "target" in self.config:
            targets = [self.as_text(self.require("target"), "target"), *targets]
        if not targets:
            raise ScoringError(f"{self.where}: exact_match needs 'target' or 'any_of'")
        self.case_sensitive = self.opt_bool("case_sensitive", True)
        self.targets = targets

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        got = self.clean(response_text)
        candidates = [self.clean(t) for t in self.targets]
        if not self.case_sensitive:
            got_cmp = got.lower()
            hit = next((c for c in candidates if c.lower() == got_cmp), None)
        else:
            hit = next((c for c in candidates if c == got), None)
        passed = hit is not None
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            detail={
                "normalized_response": got,
                "matched": hit,
                "targets": candidates,
                "normalizers": self.normalizer_names,
            },
        )


class ContainsScorer(_NormalizingScorer):
    """Substring checks. ``all_of`` scores partial credit by fraction found."""

    type = "contains"

    def validate(self) -> None:
        super().validate()
        self.reject_unknown((*_COMMON_KEYS, "target", "any_of", "all_of", "case_sensitive"))
        self.any_of = self.str_list("any_of")
        self.all_of = self.str_list("all_of")
        if "target" in self.config:
            self.any_of = [self.as_text(self.require("target"), "target"), *self.any_of]
        if not self.any_of and not self.all_of:
            raise ScoringError(f"{self.where}: contains needs 'target', 'any_of' or 'all_of'")
        self.case_sensitive = self.opt_bool("case_sensitive", False)

    def _find(self, haystack: str, needle: str) -> bool:
        if self.case_sensitive:
            return needle in haystack
        return needle.lower() in haystack.lower()

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        got = self.clean(response_text)
        detail: dict[str, Any] = {"normalized_response": got}
        value = 1.0
        passed = True
        if self.any_of:
            cleaned = [self.clean(t) for t in self.any_of]
            found = [t for t in cleaned if self._find(got, t)]
            detail["any_of_found"] = found
            if not found:
                value, passed = 0.0, False
        if self.all_of and passed:
            cleaned = [self.clean(t) for t in self.all_of]
            found = [t for t in cleaned if self._find(got, t)]
            detail["all_of_found"] = found
            detail["all_of_missing"] = [t for t in cleaned if t not in found]
            value = len(found) / len(cleaned)
            passed = len(found) == len(cleaned)
        return Score(value=value, passed=passed, detail=detail)


class NotContainsScorer(ContainsScorer):
    """Inverted containment -- for refusal checks and leaked-answer guards."""

    type = "not_contains"

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        inner = await super().score(response_text, ctx)
        return Score(
            value=1.0 - inner.value,
            passed=not inner.passed,
            detail={**inner.detail, "inverted": True},
        )


_FLAG_NAMES = {
    "i": re.IGNORECASE,
    "ignorecase": re.IGNORECASE,
    "m": re.MULTILINE,
    "multiline": re.MULTILINE,
    "s": re.DOTALL,
    "dotall": re.DOTALL,
    "x": re.VERBOSE,
    "verbose": re.VERBOSE,
}


class RegexScorer(_NormalizingScorer):
    """Pattern match, optionally comparing a capture group to a target."""

    type = "regex"
    default_normalizers = ("strip_thinking",)

    def validate(self) -> None:
        super().validate()
        self.reject_unknown((*_COMMON_KEYS, "pattern", "flags", "group", "target", "expect"))
        pattern = self.req_str("pattern")
        flags = 0
        for name in self.str_list("flags"):
            flag = _FLAG_NAMES.get(name.lower())
            if flag is None:
                raise ScoringError(
                    f"{self.where}: unknown regex flag '{name}'. "
                    f"Allowed: {', '.join(sorted(_FLAG_NAMES))}"
                )
            flags |= flag
        try:
            self.pattern = re.compile(pattern, flags)
        except re.error as exc:
            raise ScoringError(f"{self.where}: invalid regex {pattern!r}: {exc}") from exc
        self.group: int | str | None = self.config.get("group")
        if self.group is not None and not isinstance(self.group, (int, str)):
            raise ScoringError(f"{self.where}: 'group' must be an int or a name")
        self.target = self.opt_str("target")
        self.expect_match = self.opt_bool("expect", True)

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        got = self.clean(response_text)
        match = self.pattern.search(got)
        detail: dict[str, Any] = {"pattern": self.pattern.pattern, "matched": bool(match)}
        if match is None:
            passed = not self.expect_match
            return Score(value=1.0 if passed else 0.0, passed=passed, detail=detail)
        if not self.expect_match:
            detail["match"] = match.group(0)
            return Score(value=0.0, passed=False, detail=detail)
        if self.group is None and self.target is None:
            detail["match"] = match.group(0)
            return Score(value=1.0, passed=True, detail=detail)
        try:
            captured = match.group(self.group if self.group is not None else 0)
        except (IndexError, re.error) as exc:
            raise ScoringError(
                f"{self.where}: capture group {self.group!r} is not in pattern "
                f"{self.pattern.pattern!r}"
            ) from exc
        captured = (captured or "").strip()
        detail["captured"] = captured
        if self.target is None:
            return Score(value=1.0, passed=True, detail=detail)
        target = self.clean(self.target)
        detail["target"] = target
        passed = captured == target
        return Score(value=1.0 if passed else 0.0, passed=passed, detail=detail)


class NumericScorer(Scorer):
    """Compare numbers with a tolerance.

    Defaults to reading the *last* number in the response, because models
    routinely show their working: "12 x 4 = 48, minus 6, so 42" must score as
    42, not 12. Absolute and relative tolerances are both supported and a value
    passes if it clears either, which is what you want across answers spanning
    many orders of magnitude.
    """

    type = "numeric"
    _NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def validate(self) -> None:
        self.reject_unknown(("target", "tolerance", "rel_tolerance", "extract", "strip_units"))
        raw_target = self.require("target")
        self.target = self._to_float(raw_target, "target")
        self.tolerance = self.opt_float("tolerance", 0.0) or 0.0
        self.rel_tolerance = self.opt_float("rel_tolerance", 0.0) or 0.0
        if self.tolerance < 0 or self.rel_tolerance < 0:
            raise ScoringError(f"{self.where}: tolerances must be >= 0")
        self.extract = self.opt_str("extract", "last") or "last"
        if self.extract not in {"last", "first", "all", "strict"}:
            raise ScoringError(
                f"{self.where}: 'extract' must be last|first|all|strict, got '{self.extract}'"
            )

    def _to_float(self, value: Any, key: str) -> float:
        if isinstance(value, bool):
            raise ScoringError(f"{self.where}: '{key}' must be a number, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            try:
                return float(text)
            except ValueError as exc:
                raise ScoringError(f"{self.where}: '{key}' is not a number: {value!r}") from exc
        raise ScoringError(f"{self.where}: '{key}' must be a number, got {type(value).__name__}")

    def _candidates(self, text: str) -> list[float]:
        cleaned = norm.strip_thinking(text)
        cleaned = norm.extract_boxed(cleaned)
        found = [m.replace(",", "") for m in self._NUM.findall(cleaned)]
        values: list[float] = []
        for item in found:
            try:
                values.append(float(item))
            except ValueError:
                continue
        if not values:
            return []
        if self.extract == "first":
            return values[:1]
        if self.extract == "last":
            return values[-1:]
        if self.extract == "strict":
            return values if len(values) == 1 else []
        return values

    def _close(self, got: float) -> bool:
        if math.isnan(got) or math.isnan(self.target):
            return False
        if got == self.target:
            return True
        delta = abs(got - self.target)
        if self.tolerance and delta <= self.tolerance:
            return True
        if self.rel_tolerance and abs(self.target) > 0:
            return delta / abs(self.target) <= self.rel_tolerance
        return False

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        candidates = self._candidates(response_text)
        detail: dict[str, Any] = {
            "target": self.target,
            "candidates": candidates[:10],
            "extract": self.extract,
        }
        if not candidates:
            detail["reason"] = "no number found in response"
            return Score(value=0.0, passed=False, detail=detail)
        hit = next((c for c in candidates if self._close(c)), None)
        detail["matched"] = hit
        passed = hit is not None
        return Score(value=1.0 if passed else 0.0, passed=passed, detail=detail)


class ChoiceScorer(Scorer):
    """Multiple choice. Pulls the answer letter out of prose.

    Kept separate from ``exact_match`` because the extraction rule is the whole
    problem: "I think it's B, no wait, D" has to resolve to D, and a plain
    exact match on the raw text scores it wrong for the wrong reason.
    """

    type = "choice"

    def validate(self) -> None:
        self.reject_unknown(("target", "options"))
        self.target = self.as_text(self.require("target"), "target").strip().upper()
        options = self.str_list("options")
        self.options = [o.strip().upper() for o in options] if options else []
        if self.options and self.target not in self.options:
            raise ScoringError(
                f"{self.where}: target '{self.target}' is not among options {self.options}"
            )

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        picked = norm.extract_choice(response_text).strip().upper()
        if self.options and picked not in self.options:
            picked_clean = picked[:1]
            picked = picked_clean if picked_clean in self.options else picked
        passed = picked == self.target
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            detail={"picked": picked, "target": self.target, "options": self.options},
        )


def parse_json_response(text: str) -> Any:
    """Best-effort JSON out of a model response.

    Tries the whole string, then the widest balanced object or array inside it.
    Models wrap JSON in prose ("Here is the result: {...} Let me know!") far
    more often than they do not, and refusing to look past that would score a
    correct answer as malformed. Shared with the judge, which has the same
    problem for the same reason.
    """
    cleaned = norm.strip_code_fence(norm.strip_thinking(text)).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("response does not contain JSON")


def _json_subset(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Structural containment: every key/value in `expected` appears in `actual`.

    Lists compare element-wise and in order. Returns human-readable mismatch
    paths so a failure says *which field* was wrong, not just "no match".
    """
    problems: list[str] = []
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected an object, got {type(actual).__name__}"]
        for key, want in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems.extend(_json_subset(want, actual[key], f"{path}.{key}"))
        return problems
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return [f"{path}: expected an array, got {type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{path}: expected {len(expected)} items, got {len(actual)}"]
        for i, (want, got) in enumerate(zip(expected, actual)):
            problems.extend(_json_subset(want, got, f"{path}[{i}]"))
        return problems
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not isinstance(expected, bool) and not isinstance(actual, bool):
            if float(expected) != float(actual):  # 1 == 1.0 should not fail
                problems.append(f"{path}: expected {expected!r}, got {actual!r}")
            return problems
    if expected != actual:
        problems.append(f"{path}: expected {expected!r}, got {actual!r}")
    return problems


class JSONMatchScorer(Scorer):
    """Parse the response as JSON and compare it structurally.

    Scores partial credit in ``subset`` mode by the fraction of expected leaves
    that matched, so a response that gets three of four fields right is not
    indistinguishable from one that returned nothing.
    """

    type = "json_match"

    def validate(self) -> None:
        self.reject_unknown(("target", "mode", "partial_credit"))
        raw = self.require("target")
        if isinstance(raw, str):
            try:
                self.target = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"{self.where}: 'target' is not valid JSON: {exc}") from exc
        else:
            self.target = raw
        self.mode = self.opt_str("mode", "subset") or "subset"
        if self.mode not in {"subset", "exact"}:
            raise ScoringError(f"{self.where}: 'mode' must be subset|exact, got '{self.mode}'")
        self.partial_credit = self.opt_bool("partial_credit", True)

    @staticmethod
    def _leaf_count(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(JSONMatchScorer._leaf_count(v) for v in value.values()) or 1
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return sum(JSONMatchScorer._leaf_count(v) for v in value) or 1
        return 1

    async def score(self, response_text: str, ctx: ScoringContext) -> Score:
        try:
            parsed = parse_json_response(response_text)
        except ValueError as exc:
            return Score(value=0.0, passed=False, detail={"reason": str(exc)})
        if self.mode == "exact":
            problems = _json_subset(self.target, parsed)
            extra = []
            if isinstance(self.target, Mapping) and isinstance(parsed, Mapping):
                extra = sorted(set(parsed) - set(self.target))
                problems += [f"$.{k}: unexpected key" for k in extra]
        else:
            problems = _json_subset(self.target, parsed)
        passed = not problems
        if passed:
            value = 1.0
        elif self.partial_credit:
            total = max(1, self._leaf_count(self.target))
            value = max(0.0, (total - len(problems)) / total)
        else:
            value = 0.0
        return Score(
            value=value,
            passed=passed,
            detail={"mismatches": problems[:20], "mode": self.mode, "parsed": parsed},
        )
