"""No-op trace shim.

The LLM pool was lifted from a project that emitted structured pipeline
events to ``<run_dir>/.trace/events.jsonl``. Saddle has no pipeline, so
tracing is a no-op: ``get_trace()`` returns ``None`` (every call site
guards on truthiness) and ``trace_llm`` is an async context manager that
yields a throwaway dict.

If saddle ever grows its own observability, replace this module — the
pool's call sites already speak this interface.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator


def get_trace() -> None:
    """No active trace. Pool call sites guard on ``if trace:``."""
    return None


@asynccontextmanager
async def trace_llm(**_kwargs: Any) -> AsyncIterator[dict]:
    """No-op async trace context. Yields a throwaway dict that callers
    may write into (e.g. ``trace_out["output_chars"] = ...``)."""
    yield {}
