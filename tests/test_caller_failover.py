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
    _call_with_provider_deadline,
    _count_thinking_chunks,
    _is_retryable_status,
    _provider_gate,
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


# --- Per-provider concurrency gate + the silent-hang cure ---------------------
# The supervisory chain (FallbackCaller) does NOT go through the LLMPool, so its
# per-provider _DynamicLimiter never caps it. audit/run.py fans out asyncio.gather
# over every target sharing ONE FallbackCaller, so on fail-over N targets can hit
# ONE provider at once — the burst that maxes the rate-limit-prone Opus CLI
# ("3 calls will max it out ... run 2 at a time"). _call_with_provider_deadline
# now gates each provider attempt by the provider's tested concurrent_request_cap.
# These pin: the cap is enforced, the burst still COMPLETES (queued not dropped),
# no cross-provider slot is double-booked on fail-over, and — the real cure — a
# SILENT hang becomes a deadline -> fail-over within budget.

class _RecordingProvider:
    """Records peak concurrent in-flight; configurable cap + behavior.

    behavior: "ok" returns after work_s; "hang" sleeps effectively forever with
    NO exception (the exact claude-CLI-on-overload failure mode)."""

    def __init__(self, cap, name, behavior="ok", work_s=0.02):
        self.concurrent_request_cap = cap
        self.name = name
        self.behavior = behavior
        self.work_s = work_s
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.behavior == "hang":
                await asyncio.sleep(3600)
            await asyncio.sleep(self.work_s)
            return f"{self.name}:ok"
        finally:
            self.in_flight -= 1


def test_provider_gate_caps_in_flight_at_cap_yet_runs_every_call(monkeypatch):
    # An 8-wide burst against a cap-2 provider must never exceed 2 in-flight, AND
    # all 8 must still complete — excess calls QUEUE (await the slot), never drop.
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "5")
    p = _RecordingProvider(cap=2, name="claude")

    async def _burst():
        await asyncio.gather(*(
            _call_with_provider_deadline(p, "s", "h", json_mode=False, label="t")
            for _ in range(8)
        ))

    asyncio.run(_burst())
    assert p.peak == 2, f"gate must cap in-flight at the cap, saw peak={p.peak}"
    assert p.calls == 8, "every queued call must still run (none dropped)"


def test_provider_gate_is_none_when_no_positive_cap_declared():
    # A caller without concurrent_request_cap (legacy/stub) degrades to UNBOUNDED
    # rather than crashing — same graceful contract as llm_pool._get_cap.
    class _NoCap:
        async def __call__(self, *a, **k):
            return "x"

    class _ZeroCap:
        concurrent_request_cap = 0
        async def __call__(self, *a, **k):
            return "x"

    async def _check():
        return _provider_gate(_NoCap()), _provider_gate(_ZeroCap())

    g_none, g_zero = asyncio.run(_check())
    assert g_none is None and g_zero is None


def test_provider_gate_does_not_double_book_across_failover(monkeypatch):
    # The gate is per-PROVIDER-ATTEMPT, not per FallbackCaller.__call__. A lead
    # that hangs (holding its OWN slot until the deadline) must NOT also hold the
    # backup's slot: the backup must run at its own full cap. Here lead cap=1; a
    # 3-wide burst would deadlock the backup IF the lead's gate leaked across the
    # fail-over hop. It doesn't — the lead slot releases on deadline before the
    # backup slot is acquired, so all 3 complete on the backup.
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "0.3")
    lead = _RecordingProvider(cap=1, name="claude", behavior="hang")
    backup = _RecordingProvider(cap=36, name="minimax")
    fc = FallbackCaller([lead, backup], failure_threshold=0)

    async def _burst():
        return await asyncio.gather(*(
            fc("s", "h", json_mode=False, label="t") for _ in range(3)
        ))

    res = asyncio.run(_burst())
    assert all(r == "minimax:ok" for r in res), f"all must complete on backup: {res}"
    assert lead.peak <= 1, f"lead gate breached: peak={lead.peak}"
    assert backup.calls == 3, "backup must serve all three despite lead's held slots"


def test_provider_deadline_turns_silent_hang_into_failover(monkeypatch):
    # THE cure (separate from the gate): a lead that SILENTLY HANGS — no
    # exception, no tokens — must not wedge the run. The per-provider deadline
    # raises asyncio.TimeoutError -> categorized "timeout" -> FallbackCaller
    # routes to the backup, all within the deadline. This is the regression that
    # left the converge run hung with minimax never tried.
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "0.3")
    lead = _RecordingProvider(cap=2, name="claude", behavior="hang")
    backup = _RecordingProvider(cap=36, name="minimax")
    fc = FallbackCaller([lead, backup], failure_threshold=0)

    t0 = time.monotonic()
    out = asyncio.run(fc("s", "h", json_mode=False, label="t"))
    dt = time.monotonic() - t0

    assert out == "minimax:ok", "a silent hang must fail over to the backup"
    assert dt < 1.0, f"failover must happen within the deadline, took {dt:.2f}s"
    assert lead.calls == 1 and backup.calls == 1


def test_provider_deadline_zero_disables_the_clock(monkeypatch):
    # Explicit opt-out: RAYXI_LLM_PROVIDER_DEADLINE_SECONDS=0 awaits unbounded
    # (no asyncio.wait_for wrap). A fast call still returns normally — proving 0
    # is read as "disabled", not coerced back to the 20s default by _env_float.
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "0")
    p = _RecordingProvider(cap=4, name="claude")
    out = asyncio.run(
        _call_with_provider_deadline(p, "s", "h", json_mode=False, label="t")
    )
    assert out == "claude:ok"
