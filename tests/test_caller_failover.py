"""FallbackCaller robustness — a lead-provider subprocess crash must DEGRADE,
not abort the run.

saddle's lead provider is the local ``claude`` CLI (via the Agent SDK). When
that subprocess crashes the SDK raises an opaque error whose real cause is
swallowed into stderr, so the message carries no recognizable "transient"
marker. The old FallbackCaller classified by message substring only, so it
mistook a momentary lead-provider crash for a permanent error and aborted the
whole orchestration instead of failing over to minimax.

The fix gives the crash a TYPED signal — :class:`ClaudeAgentUnavailable` —
that FallbackCaller always routes past, regardless of message content, while a
genuine error (bad schema, programming bug) still propagates so real defects
aren't masked. These tests pin that contract with no real LLM and no real CLI.
"""
from __future__ import annotations

import asyncio
import importlib.util

from saddle.llm.callers import FallbackCaller, _count_thinking_chunks
from saddle.llm.claude_agent import ClaudeAgentUnavailable


class _Raises:
    """A caller that always fails with a fixed exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False,
                       label: str = "") -> str:
        self.calls += 1
        raise self.exc


class _Returns:
    """A caller that always succeeds with a fixed string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False,
                       label: str = "") -> str:
        self.calls += 1
        return self.text


def test_fallback_routes_past_claude_agent_unavailable_without_marker():
    # The message deliberately contains NO known transient marker ("exit code
    # 1" is not in the substring set). Routing must therefore come from the
    # TYPED signal — this is the exact regression that crashed the dogfood.
    down = _Raises(ClaudeAgentUnavailable(
        "claude CLI failed for orchestrate/fold: Command failed with exit code 1"
    ))
    backup = _Returns("backup-result")
    fc = FallbackCaller([down, backup])

    out = asyncio.run(fc("sys", "prompt", json_mode=True, label="orchestrate/fold"))

    assert out == "backup-result"
    assert down.calls == 1      # lead was tried
    assert backup.calls == 1    # and it degraded to the backup


def test_fallback_reraises_genuine_error_without_routing():
    # A real defect (no transient marker, not a provider-down signal) must
    # propagate so it isn't silently masked by failover.
    boom = _Raises(ValueError("schema mismatch: required field 'goals' missing"))
    backup = _Returns("never reached")
    fc = FallbackCaller([boom, backup])

    raised = False
    try:
        asyncio.run(fc("sys", "prompt"))
    except ValueError:
        raised = True

    assert raised, "a genuine error must propagate, not fail over"
    assert backup.calls == 0, "must NOT route past a genuine (non-transient) error"


def test_fallback_still_routes_known_transient_message():
    # Guard the pre-existing substring path: an HTTP-provider rate-limit string
    # still routes even though it isn't a ClaudeAgentUnavailable.
    down = _Raises(RuntimeError("429 too many requests"))
    backup = _Returns("ok")
    fc = FallbackCaller([down, backup])

    assert asyncio.run(fc("sys", "prompt")) == "ok"
    assert backup.calls == 1


def test_fallback_all_down_raises_with_last_cause():
    down1 = _Raises(ClaudeAgentUnavailable("claude CLI failed: exit code 1"))
    down2 = _Raises(RuntimeError("503 service unavailable"))
    fc = FallbackCaller([down1, down2])

    raised = False
    try:
        asyncio.run(fc("sys", "prompt"))
    except RuntimeError as exc:
        raised = "All callers failed" in str(exc)

    assert raised, "with every provider down, the chain must raise a clear summary"
    assert down1.calls == 1 and down2.calls == 1


def test_claude_agent_options_wire_stderr_sink():
    # Verifies ClaudeAgentCaller threads the SDK stderr callback into the sink
    # so a CLI crash's real cause is captured. SDK-guarded: skip where the
    # Agent SDK (and thus ClaudeAgentOptions) isn't installed.
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return
    from saddle.llm.claude_agent import ClaudeAgentCaller

    sink: list[str] = []
    opts = ClaudeAgentCaller({})._options("sys", stderr_sink=sink)
    assert opts.stderr is not None
    opts.stderr("boom-line")
    assert sink == ["boom-line"]

    # No sink supplied -> no stderr callback installed.
    opts_none = ClaudeAgentCaller({})._options("sys")
    assert opts_none.stderr is None


def test_thinking_chunks_do_not_bleed_into_the_answer():
    # THE regression: the old inline counter never reset `in_thinking`, so once
    # a <think> opened it mislabeled the entire post-</think> answer as thinking
    # and reported all 6 chunks. The span here is <think>, reasoning, more,
    # </think> — the closer-bearing chunk is the LAST thinking chunk (it carries
    # the delimiter), so the count is 4; the 2 answer chunks strictly AFTER the
    # close are not thinking.
    pieces = ["<think>", "reasoning", "more", "</think>", "answer one", "answer two"]
    assert _count_thinking_chunks(pieces) == 4


def test_open_and_close_in_one_chunk_counts_once():
    # MiniMax often packs the whole CoT into a single SSE delta. The opener and
    # closer share that one chunk -> exactly one thinking chunk, and the trailing
    # answer chunks stay uncounted.
    assert _count_thinking_chunks(["<think>cot</think>", "ans", "tail"]) == 1


def test_no_thinking_span_counts_zero():
    # A plain answer with no CoT must report zero — not silently accrue once the
    # state flag is set, which is exactly what the old `... and "</think>" not in`
    # expression did.
    assert _count_thinking_chunks(["just", "an", "answer"]) == 0


def test_tags_split_across_content_bearing_chunks():
    # Tags can be glued to content: "<think>start" opens, "end</think>tail"
    # closes. Every chunk from opener THROUGH the closer counts (3); the pure
    # answer chunk after does not.
    pieces = ["<think>start", "mid", "end</think>tail", "pure answer"]
    assert _count_thinking_chunks(pieces) == 3


def test_empty_stream_counts_zero():
    assert _count_thinking_chunks([]) == 0
