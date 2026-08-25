"""Exception hierarchy.

The split that matters at runtime is retryable vs. not. `RateLimited` and
`TransientError` are retried by the runner; `FatalError` is not, because
retrying a 401 or a malformed request just burns the budget.
"""

from __future__ import annotations


class EvalctlError(Exception):
    """Base class for everything this package raises deliberately."""


class SpecError(EvalctlError):
    """A YAML spec is invalid.

    Carries the file and the dotted path to the offending field so the CLI can
    print something you can act on without opening the file.
    """

    def __init__(self, message: str, *, path: str | None = None, source: str | None = None):
        self.path = path
        self.source = source
        location = " ".join(p for p in (source, f"at '{path}'" if path else None) if p)
        super().__init__(f"{location}: {message}" if location else message)


class ProviderError(EvalctlError):
    """Base for provider transport failures."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        self.status = status
        self.body = body
        super().__init__(message)


class RateLimited(ProviderError):
    """429 or an explicit quota signal. `retry_after` is seconds, if the API said."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw):
        self.retry_after = retry_after
        super().__init__(message, **kw)


class TransientError(ProviderError):
    """5xx, timeout, connection reset -- worth another attempt."""


class FatalError(ProviderError):
    """Auth failure, bad request, unknown model. Retrying will not help."""


class ScoringError(EvalctlError):
    """A scorer could not run (bad config, unusable judge output)."""


class RunNotFound(EvalctlError):
    """No run directory matched the given id or prefix."""
