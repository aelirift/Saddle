"""LLMPool — intent-based dispatch pool for all LLM-shaped calls.

Single API surface for every LLM call in rayxiv4:
    pool.call(intent="content_pack", system=..., prompt=..., ...)
    pool.call(intent="voice_s2t", audio=...)
    pool.call(intent="image_gen", prompt=...)
    pool.call(provider="deepseek_pro", system=..., prompt=...)  # explicit-only

The pool owns provider instances + an IntentRouter that selects providers
by health (success rate + EMA latency, with degradation cooldowns).
Provider-side failures auto-retry with an alternate.

The pool delegates intent → provider selection to IntentRouter and the
actual HTTP call goes through the existing per-caller code path (which
already wraps in PoolSlot for memory pacing + per-label timing). So the
LLMPool sits on TOP of the existing ProcessPool/PoolSlot machinery, not
in place of it.

Backward compat for existing call_structured() callsites: caller_for(intent)
returns an _IntentCaller adapter that conforms to the LLMCaller protocol,
so callers can swap from a specific provider caller to a pool-routed
adapter without touching call signatures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

from saddle.llm.protocol import LLMCaller
from saddle.llm.retry_category import is_rate_limit_text
from saddle.llm.router import IntentRouter
from saddle.trace import trace_llm


class _DynamicLimiter:
    """AIMD per-provider concurrency limiter — the queue AND the adaptive cap.

    Floors at ``floor`` (the provider's configured cap, e.g. 36 for MiniMax) and
    self-tunes toward the provider's real rate-limit boundary:
      - +1 on a clean SATURATED completion (the limit was the binding
        constraint, so there's pressure to grow),
      - -1 on a 429 / rate-limit,
    clamped to ``[floor, hard_max]``. ``acquire()`` QUEUES a call when in-flight
    reaches the current limit ("stop the queue when the pool is full");
    ``release()`` frees the slot, adjusts the limit, and wakes waiters. One
    asyncio loop → the Condition is correct without extra locks. Increasing only
    when saturated keeps the limit tracking actual pressure instead of drifting
    above it (which would make it slow to react to a 429).
    """

    def __init__(self, floor: int, hard_max: int) -> None:
        self.floor = max(1, int(floor))
        self.hard_max = max(self.floor, int(hard_max))
        self.limit = self.floor
        self.in_flight = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> bool:
        async with self._cond:
            waited = False
            while self.in_flight >= self.limit:
                waited = True
                await self._cond.wait()
            self.in_flight += 1
            # "Saturated" = this call filled the pool (or had to wait for a
            # slot) → the limit is the binding constraint, so a clean completion
            # is evidence we could use more concurrency (additive-increase).
            return waited or self.in_flight >= self.limit

    async def release(self, *, rate_limited: bool, was_saturated: bool) -> None:
        async with self._cond:
            self.in_flight = max(0, self.in_flight - 1)
            if rate_limited:
                self.limit = max(self.floor, self.limit - 1)
            elif was_saturated and self.limit < self.hard_max:
                self.limit += 1
            self._cond.notify_all()

    def snapshot(self) -> dict:
        return {"limit": self.limit, "in_flight": self.in_flight,
                "floor": self.floor, "hard_max": self.hard_max}

# When every provider for an intent is transiently cooling at once (a
# content_pack burst rate-limited them all), a routed call WAITS for the
# soonest provider to re-admit rather than hard-failing the build — bounded by
# this wall-clock budget so a genuine total outage still surfaces. Pairs with
# the short transient-failure cooldowns in router.py. Env-overridable.
_ROUTE_WAIT_BUDGET_S = float(
    os.environ.get("RAYXI_LLM_ROUTE_WAIT_BUDGET_S", "180") or 180
)

_log = logging.getLogger("saddle.llm.llm_pool")


def _get_cap(caller: Any) -> int | None:
    """Return the caller's `concurrent_request_cap` if declared, else None
    (no published admission ceiling — router treats it as unbounded).

    Reads the INSTANCE attribute first, falling back to the class. Most
    callers set it on the class (caller class in callers.py is the natural
    home for "this vendor can do N req/sec safely"), but some providers share
    one caller class with different caps per instance — e.g. DeepSeek flash
    (cap 20) vs pro (cap 10) are both DeepSeekCaller, distinguished only by the
    per-instance cap the factory stamps on.

    Returns None for legacy callers without the attribute so we don't break
    test harnesses that stub custom caller classes — admission control degrades
    gracefully to unbounded.
    """
    cap = getattr(caller, "concurrent_request_cap", None)
    if cap is None:
        return None
    try:
        n = int(cap)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# Initial intent → provider capability table.
# Text-generation intents — any general text LLM can serve.
TEXT_INTENTS: set[str] = {
    "default",
    "chat",
    "prompt_analysis",
    "design_intent",
    "doctrine_fill",
    "mechanic_graph",
    "content_pack",
    "creative_board",
    "design",
    # Creator-mode helper daemon — fast intent + target_system
    # classification of inbox transcripts. Routes to the same text
    # providers (MiniMax / Kimi / GLM) but tagged as its own intent
    # for telemetry + future provider-tuning.
    "intent_classify",
}
# Coding intents — kimi gets a preference seed (kimi-for-coding model).
CODING_INTENTS: set[str] = {"coding"}

# Reasoning_heavy is OPT-IN ONLY — never picked unless the caller
# explicitly requests intent="reasoning_heavy" or provider="deepseek_pro".
REASONING_HEAVY_INTENTS: set[str] = {"reasoning_heavy"}


class LLMPool:
    """Routes every LLM call through one place. Picks providers by health,
    records observations back to the router, retries on provider-side error.
    """

    def __init__(self) -> None:
        # Lazy import — callers.py imports from pool.py for PoolSlot, so
        # importing it at module-load time would create a circular dep.
        from saddle.llm.callers import ACTIVE_PRIORITY, build_callers

        self._callers: dict[str, LLMCaller] = build_callers()
        # Cache the priority list so _register_default_text_providers
        # uses the same source-of-truth as build_callers' fallback chain.
        # Both registry membership (which providers get text capabilities)
        # AND priority order are driven by callers.ACTIVE_PRIORITY — one
        # edit to that list rotates the LLMPool too.
        self._text_priority: list[str] = list(ACTIVE_PRIORITY)
        self._router = IntentRouter()
        # ABSOLUTE + ADAPTIVE per-provider gate (one _DynamicLimiter per
        # provider). The router's in-flight count is only a soft ROUTING
        # preference — select() falls through and overflows a provider when all
        # candidates are saturated, which let bursts blow past the cap and 429.
        # The limiter is the hard ceiling: excess calls QUEUE, never exceed the
        # current limit; and the limit self-tunes (AIMD) toward the provider's
        # 429 boundary, floored at its configured cap. Lazily created in-loop
        # (see _provider_limiter) and reused.
        self._provider_limiters: dict[str, _DynamicLimiter] = {}
        # Non-text providers (voice, image) attach via register_external().
        self._external_callers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._register_default_text_providers()
        self._register_image_provider()
        self._register_voice_provider()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_default_text_providers(self) -> None:
        """Register the text-generation providers built by callers.build_callers().

        Both registry membership and priority order come from
        callers.ACTIVE_PRIORITY (cached as self._text_priority). One
        edit to that list rotates LLMPool routing too — no parallel
        list to keep in sync.
        """
        text_caps = TEXT_INTENTS | CODING_INTENTS
        for name in self._text_priority:
            if name in self._callers:
                self._router.register_provider(
                    name, text_caps,
                    concurrent_request_cap=_get_cap(self._callers[name]),
                )

        # Pin a strict-priority chain for every text intent — saddle's
        # policy order (config/llm_policy.json -> priority, default
        # claude_agent -> minimax -> deepseek_flash). Providers whose
        # factory returned None (no creds) drop out silently.
        priority_chain = [
            name for name in self._text_priority
            if name in self._callers
        ]
        for intent in TEXT_INTENTS:
            self._router.set_preferred_order(intent, priority_chain)

        # Saddle routes EVERY text intent through the single policy chain
        # set above (config/llm_policy.json -> priority). The sibling
        # project's per-intent minimax-first override (for its pipeline
        # phases: content_pack / mechanic_graph / creative_board / ...)
        # was removed — saddle has no pipeline, and pinning minimax-first
        # would fight the policy's claude_agent lead.

        # Optional: spread mode. RAYXI_LLM_MODE=spread switches every
        # text intent from "first healthy in priority order" to
        # round-robin over healthy providers. Use case: fanout testing
        # — fire 100 parallel calls and watch them split ~34/33/33
        # across the 3-provider chain so per-provider latency, error
        # rate, and JSON quality are directly comparable. Default
        # ("priority") stays in place for production after the test
        # numbers come back.
        import os
        mode = os.environ.get("RAYXI_LLM_MODE", "priority").lower()
        if mode not in ("priority", "spread"):
            _log.warning(
                "RAYXI_LLM_MODE=%r unknown; falling back to 'priority'", mode,
            )
            mode = "priority"
        if mode == "spread":
            for intent in TEXT_INTENTS:
                self._router.set_mode(intent, "spread")
            _log.info(
                "LLMPool: SPREAD mode active — text intents round-robin "
                "across %s",
                priority_chain,
            )

        # Seed kimi as the preferred provider for coding-style intents.
        # The credentials file lives at ~/.kimi/credentials/kimi-code.json
        # — Kimi is explicitly tuned for code, so bias the router toward
        # it for coding regardless of single-call probe latency.
        # Coding intents intentionally do NOT get the demotion above:
        # if Kimi is reachable, it's the right tool for code.
        if "kimi" in self._callers:
            self._router.seed_health("kimi", "coding", ema_latency_s=0.5)

        # deepseek_pro is opt-in only — register with reasoning_heavy
        # ONLY so it never gets picked by routine intent rotation.
        if "deepseek_pro" in self._callers:
            self._router.register_provider(
                "deepseek_pro", REASONING_HEAVY_INTENTS,
                concurrent_request_cap=_get_cap(self._callers["deepseek_pro"]),
            )

    def _register_image_provider(self) -> None:
        """Register MiniMaxImageCaller for the image_gen intent.

        Skips quietly if no MiniMax API key is configured (image gen is
        optional infrastructure, not all environments have it).
        """
        try:
            from saddle.llm.image_gen import MiniMaxImageCaller
            image_caller = MiniMaxImageCaller()
        except Exception as e:
            _log.info(
                "LLMPool: skipping image provider registration: %s", e,
            )
            return

        async def _image_call(
            *,
            system: str = "",
            prompt: str = "",
            label: str = "",
            aspect_ratio: str = "1:1",
            n: int = 1,
            response_format: str = "base64",
            **_unused: Any,
        ) -> bytes:
            return await image_caller.generate(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=n,
                response_format=response_format,
                label=label,
            )

        self.register_external(
            "minimax_image", _image_call, capabilities={"image_gen"},
        )

    def _register_voice_provider(self) -> None:
        """Register the whisper voice service for the voice_s2t intent.

        Skips quietly if the voice subsystem isn't importable (server
        not built into this environment, or transcribe endpoint unset).
        """
        try:
            from saddle.voice.pool_caller import whisper_pool_call
        except Exception as e:
            _log.info(
                "LLMPool: skipping voice provider registration: %s", e,
            )
            return

        self.register_external(
            "whisper", whisper_pool_call, capabilities={"voice_s2t"},
        )

    def register_external(
        self,
        name: str,
        async_callable: Callable[..., Awaitable[Any]],
        capabilities: set[str],
    ) -> None:
        """Register an external provider — voice whisper, image gen, etc.

        async_callable receives (**kwargs) from pool.call() and returns
        the response. The pool routes by intent, then invokes
        async_callable(**kwargs). The caller is responsible for its own
        argument shape (e.g., voice expects audio=, image expects prompt=).
        """
        self._external_callers[name] = async_callable
        self._router.register_provider(name, capabilities)
        _log.info("LLMPool: registered external provider %r intents=%r",
                  name, sorted(capabilities))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def call(
        self,
        *,
        intent: str | None = None,
        provider: str | None = None,
        system: str = "",
        prompt: str = "",
        json_mode: bool = False,
        label: str = "",
        max_attempts: int = 3,
        **extra: Any,
    ) -> str:
        """Dispatch a call. Returns the response.

        Either intent OR provider must be set.
        - intent: pool picks the best provider via router; on failure,
          retries with an alternate up to max_attempts.
        - provider: bypass router; use this provider directly (escape
          hatch — only for cases that genuinely need a specific model).

        For external providers (voice, image), pass intent= and any
        extra kwargs (audio=, prompt=, etc.) — they're forwarded to the
        registered async callable. system/prompt are ignored unless the
        external callable expects them.
        """
        if not intent and not provider:
            raise ValueError("LLMPool.call() requires intent= or provider=")

        if provider is not None:
            return await self._call_specific(
                provider, system=system, prompt=prompt,
                json_mode=json_mode, label=label, **extra,
            )

        return await self._call_routed(
            intent=intent, system=system, prompt=prompt,
            json_mode=json_mode, label=label,
            max_attempts=max_attempts, **extra,
        )

    async def _call_specific(
        self,
        provider: str,
        *,
        system: str,
        prompt: str,
        json_mode: bool,
        label: str,
        **extra: Any,
    ) -> str:
        """Execute a call against a named provider with no router fallback."""
        full_label = f"{provider}:{label}" if label else provider

        # External provider (voice / image)
        if provider in self._external_callers:
            t0 = time.monotonic()
            try:
                resp = await self._external_callers[provider](
                    system=system, prompt=prompt, label=full_label, **extra,
                )
                self._router.record_success(provider, intent="explicit",
                                            latency_s=time.monotonic() - t0)
                return resp
            except Exception as e:
                self._router.record_failure(provider, intent="explicit",
                                            error=f"{type(e).__name__}: {e}")
                raise

        # Text provider (LLMCaller)
        if provider not in self._callers:
            raise LookupError(
                f"LLMPool: provider={provider!r} not registered"
            )
        caller = self._callers[provider]
        t0 = time.monotonic()
        # trace_llm wraps the call in llm_start/llm_end events on the
        # active trace (no-op if no trace active). Build Inspector +
        # assistant subscribe to those events for live visibility.
        async with trace_llm(
            phase="llm_pool", label=full_label,
            caller=provider, input_chars=len(prompt) + len(system),
        ) as trace_out:
            try:
                resp = await caller(system, prompt, json_mode=json_mode, label=full_label)
                self._router.record_success(provider, intent="explicit",
                                            latency_s=time.monotonic() - t0)
                trace_out["output_chars"] = len(resp)
                return resp
            except Exception as e:
                self._router.record_failure(provider, intent="explicit",
                                            error=f"{type(e).__name__}: {e}")
                raise

    async def _call_routed(
        self,
        *,
        intent: str,
        system: str,
        prompt: str,
        json_mode: bool,
        label: str,
        max_attempts: int,
        **extra: Any,
    ) -> str:
        excluded: set[str] = set()
        last_error: Exception | None = None
        # Resilience contract ("minimax shouldn't FAIL, just be busy"): under a
        # content_pack burst every provider for an intent can be transiently
        # cooling at the same instant. Rather than hard-fail the whole build,
        # wait for the soonest provider to re-admit (cooldowns are short — see
        # router.py), bounded by _ROUTE_WAIT_BUDGET_S so a genuine total outage
        # still surfaces after a fair wait. max_attempts is the floor on DISTINCT
        # provider tries before the budget can end the call.
        # Read per-call so ops / tests can tune without re-import.
        budget_s = float(
            os.environ.get("RAYXI_LLM_ROUTE_WAIT_BUDGET_S", "") or _ROUTE_WAIT_BUDGET_S
        )
        deadline = time.monotonic() + budget_s
        backoff = 1.0
        tries = 0
        while True:
            provider = self._router.select(intent, exclude=excluded)
            if provider is None:
                # No provider pickable this instant. If the intent has any
                # registered provider and we still have budget (or haven't yet
                # met the max_attempts floor), wait for one to re-admit and
                # clear the per-call exclusions so a provider that failed
                # earlier gets another shot once its short cooldown elapses.
                have_providers = bool(self._router.candidates(intent))
                within_budget = time.monotonic() < deadline
                if have_providers and (within_budget or tries < max_attempts):
                    await asyncio.sleep(min(backoff, 8.0))
                    backoff = min(backoff * 1.5, 8.0)
                    excluded.clear()
                    continue
                if last_error:
                    raise RuntimeError(
                        f"LLMPool: no live providers left for intent={intent!r} "
                        f"after {tries} attempt(s) + waiting; last error: {last_error}"
                    )
                raise LookupError(
                    f"LLMPool: no providers registered for intent={intent!r}"
                )

            tries += 1
            full_label = (
                f"{provider}:{intent}:{label}" if label
                else f"{provider}:{intent}"
            )
            t0 = time.monotonic()
            # Live-trace each provider attempt. Build Inspector renders
            # one card per llm_start with provider+intent+attempt; that
            # card resolves on llm_end with output_chars + duration.
            async with trace_llm(
                phase=intent, label=full_label,
                caller=provider, input_chars=len(prompt) + len(system),
            ) as trace_out:
                try:
                    resp = await self._invoke(provider,
                                              system=system, prompt=prompt,
                                              json_mode=json_mode,
                                              label=full_label, **extra)
                    self._router.record_success(provider, intent,
                                                latency_s=time.monotonic() - t0)
                    trace_out["output_chars"] = len(resp) if isinstance(resp, str) else 0
                    return resp
                except Exception as e:
                    latency = time.monotonic() - t0
                    self._router.record_failure(provider, intent,
                                                error=f"{type(e).__name__}: {e}")
                    _log.warning(
                        "LLMPool: provider=%r intent=%r try=%d failed in %.1fs (%s); "
                        "trying alternate",
                        provider, intent, tries, latency, type(e).__name__,
                    )
                    excluded.add(provider)
                    last_error = e
                    # Surface the error in the llm_end trace event so the
                    # Build Inspector shows WHY this attempt failed (not
                    # just that it ended). We can't re-raise here — that
                    # would skip the retry loop's fallthrough to next
                    # provider — so we record the error in trace_out
                    # and let the `async with` exit normally.
                    trace_out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            # Loop back: select() picks the next live provider, or — when all
            # are momentarily cooling — the None-branch above waits within the
            # budget instead of failing the build.

    # Fallback per-provider floor when a provider declares no cap, + the
    # additive-increase headroom above the floor the dynamic cap may explore.
    _DEFAULT_PROVIDER_CAP = 8
    _DYNAMIC_HEADROOM = 24

    def _provider_limiter(self, provider: str) -> _DynamicLimiter:
        """The ABSOLUTE, ADAPTIVE per-provider gate (AIMD). At most `limit` calls
        run at once — excess QUEUE; `limit` floors at the provider's configured
        cap and self-tunes toward its 429 boundary (see _DynamicLimiter). Lazily
        created in the running loop and reused; the get-or-create is race-free
        (single asyncio loop, no await between read and write)."""
        lim = self._provider_limiters.get(provider)
        if lim is None:
            cap = self._router.cap_for(provider)
            if cap is None:
                cap = getattr(self._callers.get(provider),
                              "concurrent_request_cap", None)
            floor = max(1, int(cap or self._DEFAULT_PROVIDER_CAP))
            lim = _DynamicLimiter(floor=floor, hard_max=floor + self._DYNAMIC_HEADROOM)
            self._provider_limiters[provider] = lim
        return lim

    async def _invoke(
        self,
        provider: str,
        *,
        system: str,
        prompt: str,
        json_mode: bool,
        label: str,
        **extra: Any,
    ) -> str:
        """Fire the call against the named provider's underlying callable.

        Concurrency contract: the per-provider _DynamicLimiter is the ABSOLUTE,
        ADAPTIVE cap — at most `limit` calls run at once (excess queue), and the
        limit floors at the provider's cap and self-tunes via AIMD (+1 on a
        saturated clean completion, -1 on a 429). The router's inc/dec_inflight
        is the SOFT signal select() uses to PREFER a non-saturated provider;
        it brackets the wait so a queued call still steers routing.
        """
        self._router.inc_inflight(provider)
        limiter = self._provider_limiter(provider)
        was_saturated = await limiter.acquire()
        rate_limited = False
        try:
            if provider in self._external_callers:
                return await self._external_callers[provider](
                    system=system, prompt=prompt, label=label, **extra,
                )
            return await self._callers[provider](
                system, prompt, json_mode=json_mode, label=label,
            )
        except Exception as e:  # noqa: BLE001 — classify then re-raise
            rate_limited = is_rate_limit_text(str(e))
            raise
        finally:
            await limiter.release(rate_limited=rate_limited,
                                  was_saturated=was_saturated)
            self._router.dec_inflight(provider)

    # ------------------------------------------------------------------
    # Adapter for backward-compat with call_structured() callsites
    # ------------------------------------------------------------------

    def caller_for(self, intent: str) -> LLMCaller:
        """Return an LLMCaller-compatible adapter pinned to one intent.

        Lets existing call_structured(caller, ...) callsites swap from a
        specific provider caller to a pool-routed caller without changing
        their signature.
        """
        return _IntentCaller(self, intent)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def router(self) -> IntentRouter:
        return self._router

    def list_providers(self) -> list[str]:
        return sorted(set(self._callers) | set(self._external_callers))

    def print_stats(self) -> None:
        _log.info("\n%s", self._router.format_stats())


class _IntentCaller:
    """LLMCaller-compatible adapter — pool-routed by a fixed intent."""

    def __init__(self, pool: LLMPool, intent: str) -> None:
        self._pool = pool
        self._intent = intent

    async def __call__(
        self,
        system: str,
        prompt: str,
        *,
        json_mode: bool = False,
        label: str = "",
    ) -> str:
        return await self._pool.call(
            intent=self._intent,
            system=system,
            prompt=prompt,
            json_mode=json_mode,
            label=label,
        )


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_llm_pool: LLMPool | None = None


def get_llm_pool() -> LLMPool:
    """Return the global LLMPool singleton.

    First call lazily constructs the pool (and triggers build_callers()).
    Subsequent calls return the same instance — provider state and router
    health observations persist for the process lifetime.
    """
    global _llm_pool
    if _llm_pool is None:
        _llm_pool = LLMPool()
    return _llm_pool


def reset_llm_pool() -> None:
    """Reset the singleton — for tests only."""
    global _llm_pool
    _llm_pool = None
