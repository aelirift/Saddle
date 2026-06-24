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
import time

from saddle import supervise
from saddle.llm.callers import (
    FallbackCaller,
    MiniMaxCaller,
    _count_thinking_chunks,
    _is_retryable_status,
    _retry_same_provider,
)
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


def test_fallback_reraises_genuine_error_whose_text_contains_a_bare_5xx_number():
    # THE substring regression: the old classifier routed past anything whose
    # message merely CONTAINED "500"/"403"/"1234". A real schema defect reading
    # "expected 500 rows" is not a provider 500 — it must propagate so it isn't
    # masked behind failover, exactly like the ValueError case above.
    boom = _Raises(ValueError("expected 500 rows for field 'goals', got 3"))
    backup = _Returns("never reached")
    fc = FallbackCaller([boom, backup])

    raised = False
    try:
        asyncio.run(fc("sys", "prompt"))
    except ValueError:
        raised = True

    assert raised, "a bare number in the message must NOT be read as a provider 5xx"
    assert backup.calls == 0


def test_fallback_routes_structured_http_5xx_even_without_a_reason_phrase():
    # A real provider outage arrives as an HTTP error carrying a structured
    # status. Routing must come from that status (503), not from finding "503"
    # in the text — so a terse "server error" with .response.status_code routes.
    class _Httpish(RuntimeError):
        def __init__(self, msg: str, status: int) -> None:
            super().__init__(msg)
            self.response = type("R", (), {"status_code": status})()

    down = _Raises(_Httpish("server error", 503))
    backup = _Returns("ok")
    fc = FallbackCaller([down, backup])

    assert asyncio.run(fc("sys", "prompt")) == "ok"
    assert backup.calls == 1


def test_fallback_routes_content_filter_false_positive():
    # A provider safety classifier rejecting a benign prompt is a routing
    # failure — a different provider may accept it. Preserved as an explicit
    # phrase match (not a bare code), so it still routes.
    down = _Raises(RuntimeError("request was rejected: content_filter flagged high risk"))
    backup = _Returns("ok")
    fc = FallbackCaller([down, backup])

    assert asyncio.run(fc("sys", "prompt")) == "ok"
    assert backup.calls == 1


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


def test_gateway_5xx_are_retryable_statuses():
    # THE regression: MiniMax's hand-rolled `== 529 or == 500` covered only
    # those two, so a 502/503/504 gateway outage — every bit as transient —
    # raised with NO in-provider backoff-retry. The canonical set covers them.
    for status in (500, 502, 503, 504, 529):
        assert _is_retryable_status(status), status


def test_client_errors_are_not_retryable_statuses():
    # A 2xx/4xx is not a transient outage; never burn a retry on it.
    for status in (200, 400, 401, 403, 404, 422):
        assert not _is_retryable_status(status), status


def test_rate_limit_is_a_retryable_status_but_routes_to_failover():
    # 429 is in the retryable set, but its category sends the call to a DIFFERENT
    # provider rather than a same-provider retry that can't clear in the backoff
    # window — so retryable-status and retry-here are two separate decisions.
    assert _is_retryable_status(429)
    assert _retry_same_provider("external_rate_limit") is False
    assert _retry_same_provider("provider_outage") is True
    assert _retry_same_provider("timeout") is True


# --- MiniMax streaming heartbeat ----------------------------------------------
# A silent-but-OPEN SSE stream is the exact "I never see you return" failure: the
# connection is alive so httpx's read timeout never fires, yet no tokens arrive.
# The idle heartbeat turns that silence into a fast typed error (Stalled) the
# retry loop fails over on, instead of the run sitting wedged up to the 600s
# read ceiling. These pin that contract with NO network and NO real stream.

class _SilentLines:
    """An async line-iterator that yields NOTHING then blocks far past any test
    idle budget — a stream that is OPEN (HTTP 200) but has gone silent."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(60)        # cancelled by the heartbeat's idle wait_for
        raise StopAsyncIteration


class _SilentResp:
    status_code = 200

    def aiter_lines(self):
        return _SilentLines()

    async def aread(self):
        return b""


class _SilentStream:
    async def __aenter__(self):
        return _SilentResp()

    async def __aexit__(self, *exc):
        return False


class _SilentClient:
    """httpx-client stand-in whose stream opens with 200 then says nothing."""

    def stream(self, *args, **kwargs):
        return _SilentStream()


def _bare_minimax() -> MiniMaxCaller:
    # Bypass __init__ (it resolves real config + keys); we exercise only the
    # streaming/retry plumbing, so a minimal _cfg is all that's required.
    caller = MiniMaxCaller.__new__(MiniMaxCaller)
    caller._cfg = {
        "api_base_url": "http://example.invalid/v1/chat",
        "api_key": "test-key",
        "model": "test-model",
        "stream": True,
    }
    return caller


def test_minimax_silent_stream_raises_stalled_not_hang(monkeypatch):
    # Tiny idle budget so the test trips in a fraction of a second instead of the
    # 600s production ceiling; max kept above idle so it is the IDLE clock
    # (Stalled), not the wall-clock (DeadlineExceeded), that fires.
    monkeypatch.setenv("RAYXI_MINIMAX_STREAM_IDLE_SECONDS", "0.2")
    monkeypatch.setenv("RAYXI_MINIMAX_STREAM_MAX_SECONDS", "5")
    caller = _bare_minimax()

    raised = None
    try:
        asyncio.run(caller._call_streaming(
            _SilentClient(), {"stream": True}, label="probe", json_mode=False,
        ))
    except supervise.Stalled as exc:
        raised = exc

    assert raised is not None, "a silent stream must raise Stalled, not hang"
    assert "minimax stream probe" in str(raised)


def test_minimax_persistent_stall_retries_once_then_raises(monkeypatch):
    # The retry loop must treat a wedged stream exactly like a transport timeout:
    # try once more, then RAISE so FallbackCaller routes to the next provider —
    # never swallow it, never sit silent.
    async def _instant(*_a, **_k):           # collapse the backoff so the test is fast
        return None
    monkeypatch.setattr(asyncio, "sleep", _instant)
    caller = _bare_minimax()

    calls = {"n": 0}

    async def _always_stall(client, body, *, label, json_mode):
        calls["n"] += 1
        raise supervise.Stalled("minimax stream wedged")

    monkeypatch.setattr(caller, "_call_streaming", _always_stall)

    raised = None
    try:
        asyncio.run(caller("sys", "prompt", json_mode=True, label="probe"))
    except supervise.Stalled as exc:
        raised = exc

    assert raised is not None, "a persistent stall must propagate so failover triggers"
    assert calls["n"] == 2, "must retry exactly once (2 attempts) before raising"


# --- FallbackCaller circuit breaker -------------------------------------------
# A provider that fails on EVERY call must not be re-spawned on every call: after
# a threshold of consecutive failures it trips OPEN and is skipped for a cooldown,
# with a single half-open re-probe to detect recovery. This is the wasted-work
# drag seen auditing rayxi behind a wedged `claude` CLI — 100+ doomed lead spawns.

class _FailsThenRecovers:
    """Fails (routably) until ``.fail`` is flipped off, then succeeds."""

    def __init__(self, ok: str) -> None:
        self.ok = ok
        self.fail = True
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        if self.fail:
            raise ClaudeAgentUnavailable("claude CLI failed: exit code 1")
        return self.ok


def test_circuit_breaker_skips_a_repeatedly_failing_lead_after_threshold():
    down = _Raises(ClaudeAgentUnavailable("claude CLI failed: exit code 1"))
    backup = _Returns("ok")
    fc = FallbackCaller([down, backup], failure_threshold=3, cooldown_s=999)

    # Three calls: the lead fails each time and the backup serves each time.
    for _ in range(3):
        assert asyncio.run(fc("s", "p")) == "ok"
    assert down.calls == 3, "lead tried on each of the first three calls"

    # Fourth call: the breaker is now OPEN — the lead is SKIPPED entirely and the
    # call goes straight to the backup. The dead lead is not re-spawned.
    assert asyncio.run(fc("s", "p")) == "ok"
    assert down.calls == 3, "tripped lead must NOT be re-attempted while open"
    assert backup.calls == 4


def test_circuit_breaker_reprobes_after_cooldown_and_resets_on_recovery():
    lead = _FailsThenRecovers("lead-ok")
    backup = _Returns("backup-ok")
    fc = FallbackCaller([lead, backup], failure_threshold=2, cooldown_s=0.2)

    assert asyncio.run(fc("s", "p")) == "backup-ok"     # lead fail #1
    assert asyncio.run(fc("s", "p")) == "backup-ok"     # lead fail #2 → trips open
    assert lead.calls == 2

    # Open: lead skipped, backup serves, lead not re-spawned.
    assert asyncio.run(fc("s", "p")) == "backup-ok"
    assert lead.calls == 2

    # Cooldown lapses and the lead recovers → half-open re-probe succeeds and the
    # breaker resets, so the lead is primary again on the next call.
    time.sleep(0.25)
    lead.fail = False
    assert asyncio.run(fc("s", "p")) == "lead-ok"
    assert lead.calls == 3
    assert asyncio.run(fc("s", "p")) == "lead-ok"
    assert lead.calls == 4


def test_circuit_breaker_still_attempts_when_every_circuit_is_open():
    # With both providers tripped open, the chain must STILL try them (last-resort
    # pass) rather than raise without attempting anything — a breaker must never
    # leave the run with zero attempts.
    a = _Raises(ClaudeAgentUnavailable("claude CLI failed: exit code 1"))
    b = _Raises(RuntimeError("503 service unavailable"))
    fc = FallbackCaller([a, b], failure_threshold=1, cooldown_s=999)

    for expected_calls in (1, 2):
        raised = False
        try:
            asyncio.run(fc("s", "p"))
        except RuntimeError:
            raised = True
        assert raised, "every provider down must raise a clear summary"
        assert a.calls == expected_calls and b.calls == expected_calls, (
            "both providers must be attempted even when their circuits are open"
        )


def test_circuit_breaker_disabled_with_zero_threshold_retries_every_call():
    down = _Raises(ClaudeAgentUnavailable("claude CLI failed: exit code 1"))
    backup = _Returns("ok")
    fc = FallbackCaller([down, backup], failure_threshold=0, cooldown_s=999)

    for _ in range(5):
        assert asyncio.run(fc("s", "p")) == "ok"
    # threshold 0 disables the breaker — the lead is re-tried on every call.
    assert down.calls == 5
