"""evalctl -- an LLM evaluation harness.

Pipeline:  YAML specs -> async runner (rate limited, cached) -> scorers
           -> run store (JSONL) -> report / paired-bootstrap diff.
"""

__version__ = "0.1.0"

# Bumping this invalidates every cached response. Bump it when the shape of a
# provider request changes in a way that would alter the model's output for an
# otherwise-identical spec.
REQUEST_SCHEMA_VERSION = 1

__all__ = ["__version__", "REQUEST_SCHEMA_VERSION"]
