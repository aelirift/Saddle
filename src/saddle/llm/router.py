"""Intent-based provider router for LLM calls.

Routes calls by functional intent (e.g., "voice_s2t", "content_pack",
"creative_board") to the best-suited provider, using per-(provider,
intent) health observations: success rate + EMA latency. Providers
that fail repeatedly enter a cooldown; canary calls re-admit them.

Selection algorithm:
  1. Filter providers that support the requested intent
  2. Remove any in `exclude` (to avoid retrying a just-failed provider)
  3. Remove degraded providers whose cooldown hasn't elapsed
  4. If all candidates are degraded but at least one cooldown has
     elapsed, mark it as canary and return it
  5. Otherwise weighted-random pick by health:
       weight = max(MIN_WEIGHT, success_rate^ALPHA / latency^BETA)

Stateful but not a singleton — LLMPool owns one instance. Health is
in-process only; restarts forget observations.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass

from saddle.llm.retry_category import is_rate_limit_text

_log = logging.getLogger("saddle.llm.router")

# Tuning constants — exposed for test override.
DEFAULT_EMA_ALPHA = 0.2          # new observation weight vs prior EMA
DEGRADE_AFTER_FAILURES = 3       # consecutive failures → cooldown
# Cooldowns are for "this provider is BUSY right now", not "this provider is
# down for the day". A content_pack burst transiently rate-limits a provider
# (429 / incomplete stream) that recovers within seconds; the old 300s→1800s
# backoff turned every transient 429 into a 5-30 min provider outage, and when
# all providers transiently degraded together the whole build hard-failed with
# "no providers". The user's contract is "minimax shouldn't FAIL, just be
# busy" — so a degraded provider must re-admit (via canary) in tens of seconds,
# not tens of minutes. Env-overridable for ops tuning without a code change.
COOLDOWN_BASE_S = float(os.environ.get("RAYXI_LLM_COOLDOWN_BASE_S", "20") or 20)
COOLDOWN_MAX_S = float(os.environ.get("RAYXI_LLM_COOLDOWN_MAX_S", "90") or 90)
SELECTION_SUCCESS_ALPHA = 2.0    # success_rate exponent in weight
SELECTION_LATENCY_BETA = 1.0     # latency exponent in weight
MIN_SELECTION_WEIGHT = 0.05      # floor so untried providers get tried


@dataclass
class _Health:
    """Per-(provider, intent) observed health state."""
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    ema_latency_s: float = 1.0       # neutral seed
    has_data: bool = False           # any observation landed yet?
    degraded_until_s: float = 0.0    # 0 = not degraded
    cooldown_count: int = 0          # successive degradations (exp backoff)
    canary_pending: bool = False     # next call after cooldown is a canary

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # optimistic for untried provider/intent pairs
        return self.success_count / total


class IntentRouter:
    """Routes LLM calls by functional intent with health-aware selection."""

    def __init__(self) -> None:
        # provider_name → set of intent strings it can serve
        self._capabilities: dict[str, set[str]] = {}
        # (provider_name, intent) → _Health
        self._health: dict[tuple[str, str], _Health] = {}
        # Optional strict-priority order per intent. When set, select()
        # picks the first healthy provider in this list rather than the
        # weighted-random tie-break. Used when we WANT a deterministic
        # fallback chain (e.g., demote kimi for text intents while it's
        # in a billing-limit window: minimax → deepseek_flash → kimi).
        self._preferred_order: dict[str, list[str]] = {}
        # Per-intent selection mode. "priority" (default) returns the
        # first healthy provider in _preferred_order. "spread" rotates
        # round-robin through healthy providers in _preferred_order so
        # high-fanout phases (e.g. 100 parallel skill-gen calls) hit
        # all providers proportionally — useful for measurement +
        # rate-limit avoidance during testing. Once we have stats and
        # know which provider is best, flip back to "priority".
        self._modes: dict[str, str] = {}
        # Round-robin index per intent (only consulted under "spread").
        # Increments on every spread-mode select() — synchronously,
        # serialized through asyncio's single-threaded event loop, so
        # 100 concurrent select() calls each get a distinct index
        # without a lock.
        self._spread_counter: dict[str, int] = {}
        # ---- Admission control (proactive rate-limit avoidance) ----
        # Per-provider concurrent-request cap + live in-flight count.
        # Closes the 2026-05-13 cascade where 20 parallel content_pack
        # phase2b calls all hit Kimi within 0.16s and 429'd —
        # reactive health tracking (3-failure-then-cooldown) can't
        # keep up with burst fan-out. The cap is PROACTIVE: select()
        # treats a saturated provider as unpickable, routing new
        # calls to the next healthy provider with capacity. The
        # dispatch path calls inc_inflight()/dec_inflight() around
        # each call so the counter stays accurate. Cap values come
        # from each caller class's `concurrent_request_cap` attribute
        # (see callers.py) — kept on the caller so rate-limit
        # knowledge lives with the provider it describes.
        self._cap: dict[str, int] = {}      # provider → max in-flight
        self._inflight: dict[str, int] = {}  # provider → current in-flight

    def register_provider(
        self,
        name: str,
        capabilities: set[str],
        *,
        concurrent_request_cap: int | None = None,
    ) -> None:
        """Register a provider and the set of intents it can serve.

        Re-registering a provider replaces its capability set. If
        `concurrent_request_cap` is supplied (the caller's published
        per-provider rate-limit ceiling), select() will skip this
        provider once the in-flight count reaches the cap, routing
        new calls to the next healthy provider with capacity. Pass
        None to disable admission control for this provider
        (treated as unbounded — should only apply to legacy /
        unknown-rate-limit providers).
        """
        self._capabilities[name] = set(capabilities)
        if concurrent_request_cap is not None and concurrent_request_cap > 0:
            self._cap[name] = int(concurrent_request_cap)
            self._inflight.setdefault(name, 0)
            _log.debug(
                "Router: registered %r intents=%r cap=%d",
                name, sorted(capabilities), concurrent_request_cap,
            )
        else:
            _log.debug(
                "Router: registered %r intents=%r (no admission cap)",
                name, sorted(capabilities),
            )

    def cap_for(self, name: str) -> int | None:
        """The registered concurrent-request cap for a provider (None = uncapped).
        Used by LLMPool to size the ABSOLUTE per-provider semaphore — the
        in-flight counter below is only a soft ROUTING preference."""
        return self._cap.get(name)

    def is_at_capacity(self, name: str) -> bool:
        """True when the provider's in-flight count meets/exceeds its
        admission cap. Providers without a registered cap are never
        at capacity (treated as unbounded — same semantics as a
        legacy untyped provider).
        """
        cap = self._cap.get(name)
        if cap is None:
            return False
        return self._inflight.get(name, 0) >= cap

    def inc_inflight(self, name: str) -> None:
        """Increment the provider's in-flight counter. The dispatch
        path calls this BEFORE issuing the LLM request so select()
        on a concurrent caller sees the saturated state and routes
        elsewhere. Safe to call for providers without a cap (no-op
        beyond bookkeeping)."""
        self._inflight[name] = self._inflight.get(name, 0) + 1

    def dec_inflight(self, name: str) -> None:
        """Decrement the provider's in-flight counter. Always called
        AFTER the LLM call completes (success or failure) so the
        slot frees up for the next caller. Floors at 0 in case of
        accounting drift — the cap is the ceiling that matters."""
        self._inflight[name] = max(0, self._inflight.get(name, 0) - 1)

    def inflight_snapshot(self) -> dict[str, int]:
        """Read-only view of current in-flight counts. For diag /
        observability — the snapshot is point-in-time and may be
        stale by the time the caller reads it."""
        return dict(self._inflight)

    def set_mode(self, intent: str, mode: str) -> None:
        """Set the selection mode for an intent.

        - "priority" (default): first healthy provider in
          _preferred_order wins. Strict failover to the next on health
          degradation. Right for production once we know which provider
          is best.
        - "spread": round-robin over healthy providers in
          _preferred_order. Right for fanout-testing — fire 100 calls
          and observe per-provider latency / error rate / JSON quality
          before committing to a single primary.
        """
        if mode not in ("priority", "spread"):
            raise ValueError(
                f"IntentRouter mode must be 'priority' or 'spread', got {mode!r}"
            )
        self._modes[intent] = mode
        # Reset the round-robin counter when entering spread mode so
        # the rotation starts at index 0 for predictable distribution.
        if mode == "spread":
            self._spread_counter[intent] = 0
        _log.info("Router: mode for intent=%r → %s", intent, mode)

    def set_preferred_order(self, intent: str, providers: list[str]) -> None:
        """Pin a strict-priority fallback chain for one intent.

        select() will pick the first healthy + non-excluded provider in
        this list. Providers not in the list are never picked while
        any listed provider is healthy. When every listed provider is
        degraded/excluded, falls back to weighted-random over remaining
        candidates (so we still recover if the whole priority chain is
        sick).

        Pass `[]` to clear.
        """
        self._preferred_order[intent] = list(providers)
        _log.info(
            "Router: preferred order for intent=%r → %s",
            intent, " > ".join(providers) if providers else "(cleared)",
        )

    def seed_health(
        self,
        provider: str,
        intent: str,
        *,
        ema_latency_s: float,
        success_rate: float = 1.0,
    ) -> None:
        """Seed prior knowledge so the router has a non-neutral starting bias.

        Use sparingly — only when there's a strong external reason to prefer
        one provider for an intent (e.g., 'kimi-for-coding' wraps a model
        explicitly tuned for code, so seed kimi/coding with a low latency).
        """
        h = self._health_for(provider, intent)
        h.ema_latency_s = ema_latency_s
        if success_rate < 1.0:
            # Encode prior success rate as if there had been some history
            h.success_count = int(success_rate * 10)
            h.failure_count = 10 - h.success_count
        h.has_data = True

    def candidates(self, intent: str) -> list[str]:
        """Return providers that declare support for this intent."""
        return [name for name, caps in self._capabilities.items() if intent in caps]

    def _health_for(self, provider: str, intent: str) -> _Health:
        key = (provider, intent)
        if key not in self._health:
            self._health[key] = _Health()
        return self._health[key]

    def is_in_cooldown(self, provider: str, intent: str, now_s: float | None = None) -> bool:
        """True if this (provider, intent) is in active cooldown — not pickable."""
        h = self._health_for(provider, intent)
        if h.degraded_until_s == 0.0:
            return False
        now = now_s if now_s is not None else time.monotonic()
        return now < h.degraded_until_s

    def is_degraded(self, provider: str, intent: str, now_s: float | None = None) -> bool:
        """True if this (provider, intent) has been degraded and not yet
        canary-verified back to clean state. Even after cooldown elapses,
        a degraded provider stays 'needs canary' until one success clears
        it. Healthy providers are always preferred over canary-due ones.
        """
        return self._health_for(provider, intent).degraded_until_s != 0.0

    def select(self, intent: str, exclude: set[str] | None = None) -> str | None:
        """Pick a provider for the intent.

        Strategy:
          1. If a strict-priority order is set for this intent
             (set_preferred_order), pick the FIRST healthy + non-excluded
             provider in that list. This is deterministic and gives a
             stable fallback chain.
          2. Otherwise prefer 'healthy' providers (no prior degradation,
             or previous degradation cleared by a canary success).
             Weighted-random pick by health.
          3. If no healthy candidates exist (under either strategy),
             canary-admit the longest-cooldown-elapsed provider (gives
             stale-down providers a probe chance, longest-waiting first).
          4. If neither, return None — every candidate is still cooling.
        """
        exclude = exclude or set()
        cands = [c for c in self.candidates(intent) if c not in exclude]
        if not cands:
            return None

        # A preferred order, when set, is AUTHORITATIVE for candidacy —
        # not just a sort hint. Every recovery path below (weighted-
        # random over healthy, canary re-admission, the all-saturated
        # fall-through) must stay INSIDE the chain. Pre-fix those paths
        # consulted the capability registry, so a provider deliberately
        # excluded from an intent's chain (chat_service dropped from
        # content_pack/mechanic_graph in d8e6a551 — it provably can't
        # stream large outputs) was still selected the moment every
        # listed provider was momentarily benched or admission-
        # saturated: the 2026-06-11 fresh WoW build routed 150
        # content_pack calls onto it that way, two of which hung
        # 881s/1200s and killed the phase. When every listed provider
        # is excluded for this call, return None — the dispatch loop's
        # wait-and-clear path re-tries the chain; off-chain providers
        # are never a fallback for an intent that excluded them.
        order = self._preferred_order.get(intent)
        if order:
            listed = set(order)
            cands = [c for c in cands if c in listed]
            if not cands:
                return None

        # Admission gate — treat saturated providers (in-flight count
        # >= cap) as unpickable for THIS select() call, exactly the
        # same way we treat excluded / failed providers. Proactive
        # control: prevents burst fan-out from piling onto one
        # provider faster than the reactive health gate can degrade
        # it. The 2026-05-13 cascade (20 parallel content_pack phase
        # 2b calls → all to Kimi → all 429) is the canonical case
        # this closes. After the cap routes calls 6+ to the next
        # healthy provider, those calls finish, in-flight counts
        # drop, and subsequent calls naturally redistribute.
        #
        # NOTE: when ALL candidates are saturated, fall through to
        # the unfiltered cands list so the call doesn't unnecessarily
        # fail. The reactive backoff/retry path still handles "every
        # provider is genuinely overloaded" — admission is a
        # preference, not a hard veto.
        admissible = [c for c in cands if not self.is_at_capacity(c)]
        if admissible:
            cands = admissible

        now = time.monotonic()

        # Healthy: degraded_until_s == 0 means either never failed or
        # canary-verified clean. These are always preferred.
        healthy = [c for c in cands if not self.is_degraded(c, intent)]
        if healthy:
            order = self._preferred_order.get(intent)
            if order:
                healthy_set = set(healthy)
                healthy_in_order = [name for name in order if name in healthy_set]
                if healthy_in_order:
                    mode = self._modes.get(intent, "priority")
                    # Spread mode applies ONLY to the INITIAL selection
                    # (exclude is empty). Retries — `exclude` carries
                    # the providers that already failed for this call —
                    # always fall back to priority semantics so a failed
                    # call deterministically lands on the first healthy
                    # provider in priority order, not on the next
                    # round-robin index. This way fanout testing still
                    # gives ~33/33/33 on initial calls but each call's
                    # 1-5 failures all retry on the priority head (the
                    # "first available healthy service") regardless of
                    # which spot in the rotation they originated from.
                    if mode == "spread" and not exclude:
                        # Round-robin over healthy providers in priority
                        # order. Synchronous increment is race-free in
                        # asyncio's single-threaded event loop — 100
                        # concurrent select() calls each get a distinct
                        # index without a lock.
                        n = self._spread_counter.get(intent, 0)
                        self._spread_counter[intent] = n + 1
                        return healthy_in_order[n % len(healthy_in_order)]
                    # priority (or spread retry): first healthy in
                    # priority order. exclude is already filtered out
                    # at the cands level, so the head is "first
                    # available, healthy, not previously failed".
                    return healthy_in_order[0]
                # No listed provider is healthy — drop to weighted-random
                # over remaining healthy candidates so we still recover.
            weights = []
            for c in healthy:
                h = self._health_for(c, intent)
                sr = h.success_rate ** SELECTION_SUCCESS_ALPHA
                inv_lat = 1.0 / max(0.1, h.ema_latency_s) ** SELECTION_LATENCY_BETA
                weights.append(max(MIN_SELECTION_WEIGHT, sr * inv_lat))
            return random.choices(healthy, weights=weights, k=1)[0]

        # No healthy left — try canary admission of the longest-elapsed
        # cooldown (favours the provider that's been waiting longest).
        best = None
        best_overshoot = -1.0
        for c in cands:
            h = self._health_for(c, intent)
            # Skip those still cooling
            if self.is_in_cooldown(c, intent, now):
                continue
            # Cooldown elapsed — measure overshoot to pick the longest-waiting
            overshoot = now - h.degraded_until_s
            if overshoot > best_overshoot:
                best_overshoot = overshoot
                best = c
        if best is not None:
            self._health_for(best, intent).canary_pending = True
            _log.info(
                "Router: canary-admitting %r for intent=%r (cooled down %.0fs ago)",
                best, intent, best_overshoot,
            )
            return best
        return None

    def record_success(self, provider: str, intent: str, latency_s: float) -> None:
        h = self._health_for(provider, intent)
        h.success_count += 1
        h.consecutive_failures = 0
        if h.has_data:
            h.ema_latency_s = (
                (1 - DEFAULT_EMA_ALPHA) * h.ema_latency_s
                + DEFAULT_EMA_ALPHA * latency_s
            )
        else:
            h.ema_latency_s = latency_s
        h.has_data = True
        if h.canary_pending:
            h.canary_pending = False
            h.degraded_until_s = 0.0
            h.cooldown_count = 0
            _log.info(
                "Router: canary success for %r intent=%r — re-admitted clean",
                provider, intent,
            )

    def record_failure(self, provider: str, intent: str, error: str = "") -> None:
        h = self._health_for(provider, intent)
        h.failure_count += 1
        h.consecutive_failures += 1
        h.has_data = True

        if h.canary_pending:
            h.canary_pending = False
            h.cooldown_count += 1
            cooldown_s = min(COOLDOWN_MAX_S, COOLDOWN_BASE_S * (2 ** h.cooldown_count))
            h.degraded_until_s = time.monotonic() + cooldown_s
            _log.warning(
                "Router: canary failed for %r intent=%r (%s) — cooldown %.0fs",
                provider, intent, error, cooldown_s,
            )
            return

        # A rate-limit (429) is an EXPLICIT "I'm at capacity right now" signal,
        # so bench the provider on the FIRST occurrence rather than waiting for
        # DEGRADE_AFTER_FAILURES *consecutive* misses. During a content_pack
        # burst the throttled provider's 429s are interleaved with the odd
        # success, which kept resetting consecutive_failures so the breaker
        # never latched: every call wasted a provider slot + ~10s on a
        # guaranteed 429 before failing over, pinning the pool at its cap and
        # OOM-rebooting the host (run 171926, 2026-06-20). Latching on the
        # first 429 routes the burst straight to the healthy fallback and
        # canary-reprobes the throttled provider each cooldown window.
        rate_limited = is_rate_limit_text(error)
        now = time.monotonic()
        already_cooling = h.degraded_until_s > now
        if (rate_limited or h.consecutive_failures >= DEGRADE_AFTER_FAILURES) \
                and not already_cooling:
            # Don't re-escalate while already cooling — a burst of concurrent
            # in-flight 429s landing at the trip instant would otherwise slam
            # cooldown straight to COOLDOWN_MAX_S. One trip per window; the
            # canary path owns escalation on a failed re-probe.
            h.cooldown_count += 1
            cooldown_s = min(COOLDOWN_MAX_S, COOLDOWN_BASE_S * (2 ** h.cooldown_count))
            h.degraded_until_s = now + cooldown_s
            _log.warning(
                "Router: %r intent=%r degraded (%s) — cooldown %.0fs",
                provider, intent,
                "rate-limit 429" if rate_limited
                else f"{h.consecutive_failures} consecutive failures",
                cooldown_s,
            )

    def health_snapshot(self) -> dict[tuple[str, str], dict]:
        """Return a snapshot of all per-(provider, intent) health entries."""
        snap: dict[tuple[str, str], dict] = {}
        now = time.monotonic()
        for key, h in self._health.items():
            snap[key] = {
                "success_count": h.success_count,
                "failure_count": h.failure_count,
                "success_rate": round(h.success_rate, 3),
                "ema_latency_s": round(h.ema_latency_s, 2),
                "consecutive_failures": h.consecutive_failures,
                "degraded": h.degraded_until_s > now,
                "degraded_remaining_s": max(0.0, h.degraded_until_s - now),
                "cooldown_count": h.cooldown_count,
                "canary_pending": h.canary_pending,
            }
        return snap

    def format_stats(self) -> str:
        """Human-readable per-(provider, intent) health summary."""
        lines = ["Router health by (provider, intent):"]
        lines.append(
            f"  {'provider':<18} {'intent':<22} {'ok':>4} {'fail':>4} "
            f"{'rate':>5} {'lat_s':>7} {'state':>10}"
        )
        lines.append(
            f"  {'-' * 18} {'-' * 22} {'-' * 4} {'-' * 4} "
            f"{'-' * 5} {'-' * 7} {'-' * 10}"
        )
        for (p, i), s in sorted(self.health_snapshot().items()):
            state = "DEGRADED" if s["degraded"] else "canary" if s["canary_pending"] else "ok"
            lines.append(
                f"  {p:<18} {i:<22} {s['success_count']:>4} {s['failure_count']:>4} "
                f"{s['success_rate']:>5.2f} {s['ema_latency_s']:>6.2f}s {state:>10}"
            )
        return "\n".join(lines)
