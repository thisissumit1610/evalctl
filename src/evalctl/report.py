"""Terminal, Markdown and JSON output.

No rich, no tabulate: the whole renderer is under 100 lines, and a table is not
worth a dependency that has to be pinned, audited and installed in every CI
image that wants to read a benchmark result.

The reports are opinionated about one thing -- a bare score never appears
without its interval and its error count next to it. Making the uncertainty
harder to omit than to include is the only reliable way to stop a point
estimate from being quoted on its own.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

from .analysis import ModelStats, RunAnalysis
from .stats import DiffResult, Interval

# --------------------------------------------------------------------------
# styling
# --------------------------------------------------------------------------

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def unicode_enabled(stream=None) -> bool:
    """Whether the output stream can actually render box drawing.

    A legacy Windows console at cp1252 turns every table border into a question
    mark. Rather than shipping output that only looks right on a Mac, the
    renderer probes the stream encoding once and falls back to ASCII. Set
    EVALCTL_ASCII=1 to force the plain form anywhere.
    """
    if os.environ.get("EVALCTL_ASCII"):
        return False
    encoding = getattr(stream or sys.stdout, "encoding", None) or ""
    try:
        "─·█".encode(encoding)
    except (LookupError, UnicodeEncodeError, TypeError):
        return False
    return True


class _Charset:
    """Box-drawing glyphs, resolved once per process."""

    def __init__(self) -> None:
        self.refresh()

    def refresh(self, stream=None) -> None:
        if unicode_enabled(stream):
            self.h, self.v = "─", "│"
            self.tl, self.tm, self.tr = "┌", "┬", "┐"
            self.ml, self.mm, self.mr = "├", "┼", "┤"
            self.bl, self.bm, self.br = "└", "┴", "┘"
            self.bullet = "·"
            self.full, self.empty = "█", "░"
        else:
            self.h, self.v = "-", "|"
            self.tl = self.tm = self.tr = "+"
            self.ml = self.mm = self.mr = "+"
            self.bl = self.bm = self.br = "+"
            self.bullet = "*"
            self.full, self.empty = "#", "."


CHARS = _Charset()


def color_enabled(stream: Any = None) -> bool:
    """Respect NO_COLOR, FORCE_COLOR and whether we are actually on a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def style(text: str, *names: str, enabled: bool | None = None) -> str:
    if enabled is None:
        enabled = color_enabled()
    if not enabled or not names:
        return text
    prefix = "".join(_ANSI.get(n, "") for n in names)
    return f"{prefix}{text}{_ANSI['reset']}" if prefix else text


def _visible_width(text: str) -> int:
    """Character count ignoring ANSI escapes, so padding stays aligned."""
    out, in_escape = 0, False
    for char in text:
        if in_escape:
            if char.isalpha():
                in_escape = False
            continue
        if char == "\033":
            in_escape = True
            continue
        out += 1
    return out


def _pad(text: str, width: int, align: str) -> str:
    padding = max(0, width - _visible_width(text))
    if align == "r":
        return " " * padding + text
    if align == "c":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: Sequence[str] | None = None,
    indent: str = "",
) -> str:
    """A plain box-drawn table."""
    if not headers:
        return ""
    align = list(align or ["l"] * len(headers))
    align += ["l"] * (len(headers) - len(align))
    widths = [_visible_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[: len(headers)]):
            widths[i] = max(widths[i], _visible_width(str(cell)))

    def line(left: str, mid: str, right: str) -> str:
        return indent + left + mid.join(CHARS.h * (w + 2) for w in widths) + right

    out = [line(CHARS.tl, CHARS.tm, CHARS.tr)]
    out.append(
        indent + CHARS.v + " " + (" " + CHARS.v + " ").join(_pad(h, widths[i], "c") for i, h in enumerate(headers)) + " " + CHARS.v
    )
    out.append(line(CHARS.ml, CHARS.mm, CHARS.mr))
    for row in rows:
        cells = [str(c) for c in row[: len(headers)]]
        cells += [""] * (len(headers) - len(cells))
        out.append(
            indent + CHARS.v + " " + (" " + CHARS.v + " ").join(_pad(c, widths[i], align[i]) for i, c in enumerate(cells)) + " " + CHARS.v
        )
    out.append(line(CHARS.bl, CHARS.bm, CHARS.br))
    return "\n".join(out)


def render_markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, align: Sequence[str] | None = None
) -> str:
    align = list(align or ["l"] * len(headers))
    align += ["l"] * (len(headers) - len(align))
    separators = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(separators.get(a, ":---") for a in align) + " |")
    for row in rows:
        cells = [str(c).replace("|", "\\|") for c in row[: len(headers)]]
        cells += [""] * (len(headers) - len(cells))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def ci_cell(interval: Interval, digits: int = 1) -> str:
    if interval.method in {"empty", "degenerate"}:
        return "n/a"
    return f"±{interval.half_width * 100:.{digits}f}"


def money(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# --------------------------------------------------------------------------
# run report
# --------------------------------------------------------------------------


def _model_rows(models: Sequence[ModelStats], colored: bool) -> list[list[str]]:
    rows: list[list[str]] = []
    best = max((m.score.point for m in models), default=0.0)
    for model in models:
        label = model.model_id
        if len(models) > 1 and model.score.point >= best and best > 0:
            label = style(label, "bold", "green", enabled=colored)
        errors = str(model.errors)
        if model.errors:
            errors = style(errors, "yellow" if model.error_rate < 0.05 else "red", enabled=colored)
        rows.append(
            [
                label,
                pct(model.score.point),
                ci_cell(model.score),
                pct(model.pass_rate.point),
                str(model.cases),
                str(model.trials),
                errors,
                f"{model.cache_hit_rate * 100:.0f}%",
                money(model.cost_usd),
                f"{model.latency_p50_ms / 1000:.2f}s" if model.latency_p50_ms else "-",
            ]
        )
    return rows


def format_run_report(
    analysis: RunAnalysis,
    *,
    manifest: Mapping[str, Any] | None = None,
    show_tasks: bool = True,
    markdown: bool = False,
    colored: bool | None = None,
) -> str:
    colored = color_enabled() if colored is None else colored
    manifest = manifest or {}
    suite = manifest.get("suite") or {}
    headers = ["model", "score", "95% CI", "pass", "cases", "trials", "err", "cache", "cost", "p50"]
    align = ["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
    rows = _model_rows(analysis.models, colored and not markdown)

    out: list[str] = []
    title = f"run {analysis.run_id}" if analysis.run_id else "run report"
    if suite.get("name"):
        title += f"  ·  suite '{suite['name']}'"
    if markdown:
        out.append(f"## {title}\n")
        out.append(render_markdown_table(headers, rows, align=align))
    else:
        out.append(style(title, "bold", enabled=colored))
        if manifest.get("created_at"):
            meta = f"  {manifest['created_at']}"
            if analysis.duration_s:
                meta += f"  ·  {duration(analysis.duration_s)}"
            if suite.get("fingerprint"):
                meta += f"  ·  suite {suite['fingerprint'][:12]}"
            out.append(style(meta, "dim", enabled=colored))
        out.append("")
        out.append(render_table(headers, rows, align=align))

    footer = (
        f"score = weighted mean of scorer values; pass = all required scorers passed. "
        f"CI = {int(analysis.level * 100)}% bootstrap clustered by "
        f"{analysis.cluster_by} ({analysis.resamples} resamples, seed {analysis.seed}). "
        f"errors counted as '{analysis.error_policy}'."
    )
    out.append("")
    out.append(style(footer, "dim", enabled=colored and not markdown) if not markdown else f"_{footer}_")

    for note in analysis.notes:
        out.append(style(f"note: {note}", "yellow", enabled=colored and not markdown))

    if show_tasks:
        for model in analysis.models:
            if not model.tasks:
                continue
            task_rows = [
                [
                    t.task_id,
                    str(t.cases),
                    pct(t.score),
                    pct(t.pass_rate),
                    str(t.errors) if t.errors else "",
                ]
                for t in model.tasks
            ]
            heading = f"by task · {model.model_id}"
            task_headers = ["task", "cases", "score", "pass", "err"]
            task_align = ["l", "r", "r", "r", "r"]
            out.append("")
            if markdown:
                out.append(f"### {heading}\n")
                out.append(render_markdown_table(task_headers, task_rows, align=task_align))
            else:
                out.append(style(heading, "bold", enabled=colored))
                out.append(render_table(task_headers, task_rows, align=task_align))

    return "\n".join(out)


# --------------------------------------------------------------------------
# diff report
# --------------------------------------------------------------------------


def format_diff_report(
    result: DiffResult,
    *,
    label_a: str,
    label_b: str,
    metric: str = "score",
    cluster_by: str = "task",
    warnings: Sequence[str] = (),
    markdown: bool = False,
    colored: bool | None = None,
) -> str:
    colored = color_enabled() if colored is None else colored
    delta_pp = result.delta * 100
    arrow = "+" if delta_pp >= 0 else ""
    verdict_color = (
        "green" if result.verdict == "improvement"
        else "red" if result.verdict == "regression"
        else "yellow"
    )

    rows = [
        [f"A  {label_a}", pct(result.mean_a)],
        [f"B  {label_b}", pct(result.mean_b)],
        ["delta (A - B)", f"{arrow}{delta_pp:.1f}pp"],
        [
            f"{int(result.paired.level * 100)}% CI (paired)",
            f"[{result.paired.low * 100:+.1f}, {result.paired.high * 100:+.1f}] pp",
        ],
        [
            f"{int(result.unpaired.level * 100)}% CI (unpaired)",
            f"[{result.unpaired.low * 100:+.1f}, {result.unpaired.high * 100:+.1f}] pp",
        ],
        [
            "cases compared",
            f"{result.n_paired} across {result.n_clusters} task(s)"
            if cluster_by == "task"
            else f"{result.n_paired}, treated as independent",
        ],
        ["A better / B better / tie", f"{result.wins} / {result.losses} / {result.ties}"],
    ]
    if result.mcnemar_p is not None:
        rows.append(["McNemar exact p", f"{result.mcnemar_p:.4f}"])
    if result.mde:
        rows.append(["smallest detectable effect", f"±{result.mde * 100:.1f}pp at 80% power"])

    out: list[str] = []
    heading = f"diff · metric '{metric}' · clustered by {cluster_by}"
    verdict_line = f"{result.verdict.upper()}"
    if result.significant:
        verdict_line += f"  ({arrow}{delta_pp:.1f}pp, interval excludes zero)"
    else:
        verdict_line += f"  ({arrow}{delta_pp:.1f}pp, interval includes zero)"

    if markdown:
        out.append(f"## {heading}\n")
        out.append(f"**{verdict_line}**\n")
        out.append(render_markdown_table(["", ""], rows, align=["l", "r"]))
    else:
        out.append(style(heading, "bold", enabled=colored))
        out.append(style(verdict_line, "bold", verdict_color, enabled=colored))
        out.append("")
        out.append(render_table(["", ""], rows, align=["l", "r"]))

    explanation = (
        "The paired interval is the one to read: it compares A and B on the same cases, so "
        "case difficulty cancels. The unpaired interval is shown only to make the cost of "
        "throwing that pairing away visible."
    )
    if cluster_by == "case":
        explanation += (
            " Clustering by case assumes cases are independent draws; if they share a template, "
            "re-check with --cluster-by task before believing this interval."
        )
    out.append("")
    out.append(style(explanation, "dim", enabled=colored and not markdown) if not markdown else f"_{explanation}_")

    for note in list(result.notes) + list(warnings):
        out.append(style(f"warning: {note}", "yellow", enabled=colored and not markdown))
    return "\n".join(out)


def format_case_deltas(
    deltas: Sequence[tuple[str, float, float, float]],
    *,
    limit: int = 15,
    colored: bool | None = None,
    markdown: bool = False,
) -> str:
    """The cases that moved most -- where you actually go to look for the cause."""
    colored = color_enabled() if colored is None else colored
    if not deltas:
        return ""
    ordered = sorted(deltas, key=lambda d: -abs(d[3]))[:limit]
    rows = []
    for unit, a_val, b_val, delta in ordered:
        marker = "+" if delta > 0 else ""
        cell = f"{marker}{delta * 100:.0f}pp"
        if not markdown:
            cell = style(cell, "green" if delta > 0 else "red", enabled=colored)
        rows.append([unit, pct(a_val, 0), pct(b_val, 0), cell])
    headers = ["case", "A", "B", "delta"]
    align = ["l", "r", "r", "r"]
    if markdown:
        return "### biggest per-case changes\n\n" + render_markdown_table(headers, rows, align=align)
    return (
        style("biggest per-case changes", "bold", enabled=colored)
        + "\n"
        + render_table(headers, rows, align=align)
    )


def to_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def progress_line(done: int, total: int, *, ok: int, errors: int, cached: int,
                  cost: float, eta_s: float, width: int = 24) -> str:
    """A single rewritable status line -- no dependency, no scrollback spam."""
    fraction = done / total if total else 1.0
    filled = int(fraction * width)
    bar = CHARS.full * filled + CHARS.empty * (width - filled)
    parts = [
        f"{bar} {done}/{total}",
        f"ok {ok}",
    ]
    if errors:
        parts.append(f"err {errors}")
    if cached:
        parts.append(f"cached {cached}")
    if cost > 0:
        parts.append(money(cost))
    if eta_s > 1 and done < total:
        parts.append(f"eta {duration(eta_s)}")
    return "  ".join(parts)


def write_progress(text: str, stream: Any = None) -> None:
    stream = stream or sys.stderr
    if getattr(stream, "isatty", lambda: False)():
        stream.write("\r\033[K" + text)
        stream.flush()


def clear_progress(stream: Any = None) -> None:
    stream = stream or sys.stderr
    if getattr(stream, "isatty", lambda: False)():
        stream.write("\r\033[K")
        stream.flush()


def bullet_list(items: Iterable[str], colored: bool | None = None) -> str:
    return "\n".join(f"  {CHARS.bullet} {item}" for item in items)
