r"""Answer normalizers.

Where benchmark numbers actually come from
------------------------------------------
On short-answer tasks, the gap between a "45%" harness and a "72%" harness for
the *same model* is usually not the model at all -- it is whether the harness
strips a trailing period, unwraps ``**42**``, or pulls the answer out of "The
answer is 42." Every one of those decisions is a research choice that changes
the headline number, so they are named, individually testable, and recorded in
the run manifest instead of buried in a scorer's private cleanup pass.

Normalizers apply left to right and are applied to the *target* as well as the
response, so a spec cannot accidentally compare a normalized answer against an
unnormalized key.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Sequence

from ..errors import ScoringError

Normalizer = Callable[[str], str]

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\s\"'`([{<.,;:!?*_-]+|[\s\"'`)\]}>.,;:!?*_-]+$")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_THINK = re.compile(r"<(think|thinking|scratchpad|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*",
    re.IGNORECASE,
)
_CHOICE = re.compile(r"\b([A-Ja-j])\b(?=[).:,\s]|$)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1", re.DOTALL)


def strip_whitespace(text: str) -> str:
    return text.strip()


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def lower(text: str) -> str:
    return text.lower()


def upper(text: str) -> str:
    return text.upper()


def strip_punctuation(text: str) -> str:
    """Trim leading/trailing punctuation only.

    Interior punctuation is left alone deliberately: stripping it would turn
    ``3.14`` into ``314`` and ``don't`` into ``dont``, silently changing what
    counts as correct.
    """
    return _PUNCT_EDGE.sub("", text)


def strip_articles(text: str) -> str:
    return collapse_whitespace(_ARTICLES.sub(" ", text))


def strip_code_fence(text: str) -> str:
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models emit inline."""
    return _THINK.sub("", text).strip()


def strip_markdown(text: str) -> str:
    """Unwrap **bold** / *italic* / __underline__ around an answer."""
    previous = None
    out = text
    while previous != out:  # nested emphasis: ***42***
        previous = out
        out = _MD_EMPHASIS.sub(r"\2", out)
    return out


def strip_quotes(text: str) -> str:
    stripped = text.strip()
    pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    for open_q, close_q in pairs:
        if len(stripped) >= 2 and stripped.startswith(open_q) and stripped.endswith(close_q):
            return stripped[1:-1].strip()
    return stripped


def strip_answer_prefix(text: str) -> str:
    """Remove a leading 'The answer is' / 'Answer:' from each line's start."""
    return _ANSWER_LEAD.sub("", text.strip()).strip()


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def last_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def extract_boxed(text: str) -> str:
    """Pull the last \\boxed{...}, the LaTeX convention in math benchmarks."""
    matches = _BOXED.findall(text)
    return matches[-1].strip() if matches else text


def extract_last_number(text: str) -> str:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else text


def extract_first_number(text: str) -> str:
    matches = _NUMBER.findall(text)
    return matches[0].replace(",", "") if matches else text


def extract_choice(text: str) -> str:
    """Pull a multiple-choice letter, preferring the last standalone one.

    'Let me consider A... actually the answer is C' should score as C, so the
    last match wins rather than the first.
    """
    candidate = strip_answer_prefix(last_line(text)) or text
    matches = _CHOICE.findall(candidate) or _CHOICE.findall(text)
    return matches[-1].upper() if matches else text.strip()


def normalize_unicode(text: str) -> str:
    """NFKC fold, so a full-width digit or a smart quote compares equal."""
    return unicodedata.normalize("NFKC", text)


def strip_trailing_period(text: str) -> str:
    return text[:-1] if text.endswith(".") else text


NORMALIZERS: dict[str, Normalizer] = {
    "strip": strip_whitespace,
    "collapse_whitespace": collapse_whitespace,
    "lower": lower,
    "upper": upper,
    "strip_punctuation": strip_punctuation,
    "strip_articles": strip_articles,
    "strip_code_fence": strip_code_fence,
    "strip_thinking": strip_thinking,
    "strip_markdown": strip_markdown,
    "strip_quotes": strip_quotes,
    "strip_answer_prefix": strip_answer_prefix,
    "strip_trailing_period": strip_trailing_period,
    "first_line": first_line,
    "last_line": last_line,
    "extract_boxed": extract_boxed,
    "extract_last_number": extract_last_number,
    "extract_first_number": extract_first_number,
    "extract_choice": extract_choice,
    "normalize_unicode": normalize_unicode,
}

# A pragmatic default for short free-text answers. Chosen to be *conservative*:
# it removes packaging around the answer but never touches the answer's own
# characters, so it cannot turn a wrong response into a right one.
DEFAULT_NORMALIZERS: tuple[str, ...] = (
    "strip_thinking",
    "strip_code_fence",
    "strip_markdown",
    "strip_answer_prefix",
    "strip_quotes",
    "collapse_whitespace",
    "strip_punctuation",
)


def resolve(names: Sequence[str] | None, *, where: str = "normalize") -> tuple[Normalizer, ...]:
    if names is None:
        return ()
    resolved: list[Normalizer] = []
    for name in names:
        fn = NORMALIZERS.get(name)
        if fn is None:
            raise ScoringError(
                f"{where}: unknown normalizer '{name}'. "
                f"Available: {', '.join(sorted(NORMALIZERS))}"
            )
        resolved.append(fn)
    return tuple(resolved)


def apply(text: str, normalizers: Iterable[Normalizer]) -> str:
    for fn in normalizers:
        text = fn(text)
    return text


def apply_named(text: str, names: Sequence[str] | None, *, where: str = "normalize") -> str:
    return apply(text, resolve(names, where=where))
