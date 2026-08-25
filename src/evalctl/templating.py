r"""A deliberately tiny {{var}} template engine.

Why not Jinja2: prompt templates in this harness only ever substitute case
variables. Jinja brings loops, conditionals, and arbitrary attribute access,
which turn a "data file" into "code that can do anything" -- a real problem when
you accept task specs from other people. This renders values and nothing else.

Syntax:
    {{ name }}            -- substitute
    {{ name | json }}     -- json.dumps the value
    {{ name | upper }}    -- filters chain left to right: {{ x | strip | lower }}
    \{{ name }}           -- a backslash escapes the tag, emitting {{ name }}

A missing variable is a hard error, never an empty string: a silently blank
prompt scores as a legitimate wrong answer and quietly poisons the whole run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Mapping

from .errors import SpecError

_TAG = re.compile(r"(?<!\\)\{\{\s*([^{}]*?)\s*\}\}")
_WHOLE_TAG = re.compile(r"^\s*\{\{\s*([^{}]*?)\s*\}\}\s*$")
_ESCAPED = re.compile(r"\\(\{\{)")

Filter = Callable[[Any], Any]

FILTERS: dict[str, Filter] = {
    "json": lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True),
    "json_pretty": lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2),
    "upper": lambda v: str(v).upper(),
    "lower": lambda v: str(v).lower(),
    "strip": lambda v: str(v).strip(),
    "trim": lambda v: str(v).strip(),
    "int": lambda v: str(int(float(v))),
    "len": lambda v: str(len(v)),
}


def render(template: str, variables: Mapping[str, Any], *, where: str = "template") -> str:
    """Substitute {{var}} in `template`. Raises SpecError on unknown var/filter."""
    if not isinstance(template, str):
        raise SpecError(f"expected a string template, got {type(template).__name__}", source=where)

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if not expr:
            raise SpecError("empty '{{ }}' placeholder", source=where)
        name, *filter_names = [part.strip() for part in expr.split("|")]
        if name not in variables:
            known = ", ".join(sorted(map(str, variables))) or "(none)"
            raise SpecError(
                f"unknown variable '{name}' in template; case defines: {known}", source=where
            )
        value: Any = variables[name]
        for filter_name in filter_names:
            fn = FILTERS.get(filter_name)
            if fn is None:
                raise SpecError(
                    f"unknown filter '{filter_name}' (have: {', '.join(sorted(FILTERS))})",
                    source=where,
                )
            try:
                value = fn(value)
            except (TypeError, ValueError) as exc:
                raise SpecError(
                    f"filter '{filter_name}' failed on variable '{name}': {exc}", source=where
                ) from exc
        return value if isinstance(value, str) else str(value)

    return _ESCAPED.sub(r"\1", _TAG.sub(replace, template))


def render_value(template: str, variables: Mapping[str, Any], *, where: str = "template") -> Any:
    """Render, preserving type when the template is exactly one placeholder.

    ``any_of: "{{ aliases }}"`` has to produce the *list* the case defines, not
    the string ``"['Berne']"``. YAML has no way to say "substitute this whole
    value", so the rule is positional: a template that is nothing but a single
    tag yields the raw value; a tag embedded in surrounding text yields a
    string, as it must.
    """
    match = _WHOLE_TAG.match(template)
    if match is None:
        return render(template, variables, where=where)
    expr = match.group(1).strip()
    name, *filter_names = [part.strip() for part in expr.split("|")]
    if name not in variables:
        known = ", ".join(sorted(map(str, variables))) or "(none)"
        raise SpecError(f"unknown variable '{name}' in template; case defines: {known}", source=where)
    value: Any = variables[name]
    for filter_name in filter_names:
        fn = FILTERS.get(filter_name)
        if fn is None:
            raise SpecError(
                f"unknown filter '{filter_name}' (have: {', '.join(sorted(FILTERS))})", source=where
            )
        value = fn(value)
    return value


def variables_in(template: str) -> set[str]:
    """Names referenced by a template -- used by `evalctl validate` to catch
    typos in case vars before you spend money on a run."""
    if not isinstance(template, str):
        return set()
    found: set[str] = set()
    for match in _TAG.finditer(template):
        expr = match.group(1).strip()
        if expr:
            found.add(expr.split("|")[0].strip())
    return found


def render_deep(value: Any, variables: Mapping[str, Any], *, where: str = "template") -> Any:
    """Render every string inside a nested structure. Used for scorer configs so
    a target can be written as `target: "{{ answer }}"`."""
    if isinstance(value, str):
        return render_value(value, variables, where=where)
    if isinstance(value, Mapping):
        return {k: render_deep(v, variables, where=f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return [render_deep(v, variables, where=f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def variables_in_deep(value: Any) -> set[str]:
    if isinstance(value, str):
        return variables_in(value)
    found: set[str] = set()
    if isinstance(value, Mapping):
        for v in value.values():
            found |= variables_in_deep(v)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for v in value:
            found |= variables_in_deep(v)
    return found
