"""LLM callers for saddle.

Per-provider caller classes (one per LLM endpoint family) plus
factory + registry plumbing consumed by `LLMPool`. The active provider
order lives in `ACTIVE_PRIORITY` — leftmost wins.

The OLD `CallerRouter` / `build_router` abstraction (primary/fast/hlr
caller split keyed off call_type strings) was retired 2026-05-04 when
the pipeline finished migrating to `LLMPool`. Per-call provider
selection now happens at the pool layer via intent-based routing
(see `saddle.llm.llm_pool`).

Saddle's lead provider is `ClaudeAgentCaller` (saddle.llm.claude_agent):
Claude via the Agent SDK, driving the local `claude` CLI directly
(subscription auth — no api key). The same SDK backs the interactive
chat (`python -m saddle.chat`), so the user-facing surface and the pool's
lead provider are one engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import weakref
from pathlib import Path

import httpx

from .json_tools import extract_json_text, strip_think
from .pool import PoolSlot, get_pool
from .protocol import LLMCaller
# Module-level safe: claude_agent.py imports the Agent SDK lazily (inside
# methods), so importing just the exception type pulls no SDK on SDK-less hosts.
from .claude_agent import ClaudeAgentUnavailable
from .retry_category import categorize_retry
from .policy import active_priority, merged_config
from saddle.context import Context
from saddle import supervise

_KIMI_CRED_PATH = Path.home() / ".kimi/credentials/kimi-code.json"
_KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
_KIMI_AUTH_URL = "https://auth.kimi.com/api/oauth/token"
_KIMI_DEFAULT_API_BASE_URL = "https://api.kimi.com/coding/v1"
_KIMI_DEFAULT_MODEL = "kimi-for-coding"
_DEEPSEEK_DEFAULT_API_BASE_URL = "https://api.deepseek.com/anthropic"
_DEEPSEEK_PRO_DEFAULT_MODEL = "deepseek-v4-pro"
_DEEPSEEK_FLASH_DEFAULT_MODEL = "deepseek-v4-flash"
_MINIMAX_DEFAULT_API_BASE_URL = "https://api.minimax.io/v1/chat/completions"
_MINIMAX_DEFAULT_MODEL = "MiniMax-M3"

_kimi_token_cache: dict = {}
_kimi_token_cache_time: float = 0
_kimi_refresh_lock = threading.Lock()
_log = logging.getLogger("saddle.llm.callers")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _provider_deadline_seconds() -> float:
    """Per-provider wall-clock deadline (seconds). A single provider call that
    runs past this raises ``asyncio.TimeoutError`` -> categorized ``timeout`` ->
    :class:`FallbackCaller` routes to the NEXT provider. This is the lever that
    turns a SILENT provider HANG (the local ``claude`` CLI riding out an Opus
    rate-limit / overload with no exception, producing no tokens) into a
    fail-over WITHIN budget, instead of wedging the whole inline-hook deadline
    (~60s) with no alternate ever tried.

    Default 20s: comfortably above the measured happy path (claude 2-4s, minimax
    2.2s isolated / ~13s under a 3-way burst) yet well under the 60s design-hook
    budget, leaving room for the fail-over provider's own latency. Set
    ``RAYXI_LLM_PROVIDER_DEADLINE_SECONDS=0`` to DISABLE (await unbounded) — an
    EXPLICIT opt-out, never the silent default that caused the converge hang.
    Read directly (not via ``_env_float``, which coerces 0 -> default and would
    make the opt-out unreachable)."""
    raw = os.environ.get("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS")
    if raw is None or str(raw).strip() == "":
        return 20.0
    try:
        return float(raw)
    except ValueError:
        return 20.0


def _minimax_stream_idle_seconds() -> float:
    """Idle-heartbeat budget for a MiniMax SSE stream: silence longer than this
    means the stream is wedged, not merely slow. Sized above a reasoning model's
    time-to-first-token under load (TTFT can be tens of seconds) so real CoT
    streaming never trips it, yet far under httpx's 600s read ceiling so a dead
    connection is caught in minutes, not ten of them."""
    return _env_float("RAYXI_MINIMAX_STREAM_IDLE_SECONDS", 180.0)


def _minimax_stream_max_seconds() -> float:
    """Wall-clock ceiling across a whole MiniMax stream — mirrors the httpx
    client timeout default so the two agree on 'this turn has gone on too long'."""
    return _env_float("RAYXI_MINIMAX_STREAM_MAX_SECONDS", 600.0)


def _circuit_failure_threshold() -> int:
    """Consecutive failover-routing failures from one provider before the
    FallbackCaller trips its circuit and stops re-attempting it on every call.
    0 disables the breaker (always re-try every provider, every call)."""
    raw = os.environ.get("SADDLE_CIRCUIT_FAILURE_THRESHOLD")
    if raw is None or not raw.strip():
        return 3
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _circuit_cooldown_s() -> float:
    """How long a tripped provider is skipped before a single half-open re-probe.
    Sized so a reliably-down lead (e.g. a crashing local CLI) is probed about
    once per minute instead of on all N calls of a run, yet recovers promptly
    once it comes back."""
    return _env_float("SADDLE_CIRCUIT_COOLDOWN_S", 60.0)


# ---------------------------------------------------------------------------
# Per-provider concurrency gate for the SUPERVISORY fail-over path
# ---------------------------------------------------------------------------
# The LLMPool already caps every *routed* call with a per-provider limiter sized
# to the provider's tested ``concurrent_request_cap`` (llm_pool.py). But the
# supervisory chain — ``build_callers()["default"]`` -> :class:`FallbackCaller`
# — does NOT go through the pool. ``audit/run.py`` fans out ``asyncio.gather``
# over every target, all sharing ONE FallbackCaller (verified: run.py builds the
# caller once at line ~148 and the per-target closure captures that single
# instance), so on fail-over N targets can hit ONE provider at once with nothing
# capping them. That burst is what maxes the rate-limit-prone Opus CLI
# (operator: "3 calls will max it out ... run 2 at a time").
#
# Cap that burst HERE, at the single chokepoint every FallbackCaller / RaceCaller
# provider call already passes through. Sized to the SAME tested
# ``concurrent_request_cap`` the pool uses — one shared source of truth in
# config, not a parallel number.
#
# Why a plain Semaphore and NOT the pool's adaptive ``_DynamicLimiter``: the
# limiter's AIMD feedback GROWS the cap on saturation and SHRINKS it on a 429.
# The lead provider here (the local ``claude`` CLI) fails by SILENTLY HANGING on
# Opus overload — it emits no 429 — so the shrink signal never fires and an
# adaptive cap would only ever creep UP, straight into the hang zone. A fixed
# Semaphore is exactly "run N at a time", stable regardless of provider silence.
# (The pool keeps ``_DynamicLimiter`` because its routed traffic DOES carry 429
# telemetry to drive the AIMD.)
#
# IMPORTANT — this gate is PREVENTION, not the cure. The cure for a hang is the
# per-provider DEADLINE (below) turning it into a fail-over; the LOUD REPORT
# (FallbackCaller failover capture) is what SURFACES that it happened. The cap
# only lowers how often we provoke the overload; it must never become the thing
# that "makes runs finish" by hiding a hang.
_provider_gates: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _provider_gate(caller: LLMCaller):
    """Per-caller, per-loop concurrency gate sized to the caller's tested
    ``concurrent_request_cap``. Returns ``None`` when the caller declares no
    positive cap — the gate then degrades to unbounded, the same graceful
    fallback ``llm_pool._get_cap`` gives stub callers in tests.

    Keyed by caller IDENTITY (provider instances are process-lifetime singletons)
    AND by running loop, so the Semaphore is always bound to the loop that awaits
    it (tests spin up fresh loops). WeakKeyDictionary so a discarded caller's gate
    is collected with it — no leak."""
    cap = getattr(caller, "concurrent_request_cap", None)
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return None
    if cap <= 0:
        return None
    loop = asyncio.get_running_loop()
    entry = _provider_gates.get(caller)
    if entry is None or entry[0] is not loop:
        sem = asyncio.Semaphore(cap)
        _provider_gates[caller] = (loop, sem)
        return sem
    return entry[1]


async def _call_under_deadline(
    caller: LLMCaller,
    system: str,
    prompt: str,
    *,
    json_mode: bool,
    label: str,
) -> str:
    """Run one provider call under the per-provider wall-clock DEADLINE — the
    cure that turns a silent HANG into ``asyncio.TimeoutError`` -> fail-over
    (see :func:`_provider_deadline_seconds`)."""
    deadline = _provider_deadline_seconds()
    coro = caller(system, prompt, json_mode=json_mode, label=label)
    if deadline <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=deadline)


async def _call_with_provider_deadline(
    caller: LLMCaller,
    system: str,
    prompt: str,
    *,
    json_mode: bool,
    label: str,
) -> str:
    """Gate a single provider call by per-provider CONCURRENCY, then run it under
    the per-provider DEADLINE.

    The gate is acquired OUTSIDE the deadline: a call that politely WAITS its turn
    must not have that queue time counted against its own wall-clock budget (it
    would spuriously "time out" while idle). The queue wait DOES count against the
    outer inline-hook budget, which is the real ceiling.

    The gate is per-PROVIDER-ATTEMPT, not per FallbackCaller ``__call__``: this
    helper is invoked once per provider in the fail-over loop, so a minimax call
    that fails over to claude releases the minimax slot (``async with`` exit)
    BEFORE claude's slot is acquired — no cross-provider slot is double-booked."""
    gate = _provider_gate(caller)
    if gate is None:
        return await _call_under_deadline(
            caller, system, prompt, json_mode=json_mode, label=label,
        )
    async with gate:
        return await _call_under_deadline(
            caller, system, prompt, json_mode=json_mode, label=label,
        )


def _config_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("RAYXI_LLM_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "config" / "llm_config.json",
            Path.cwd() / "config" / "llm_config.json",
            Path.home() / ".config" / "rayxi" / "llm_config.json",
            # Removed: hardcoded cross-project path leaked local filesystem.
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def _resolve_config_path() -> Path | None:
    for candidate in _config_candidates():
        if candidate.exists():
            return candidate
    return None


def _safe_response_payload(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {"data": data}
    except Exception:
        text = (resp.text or "").strip()
        return {"raw_text": text[:1000]}


def _http_error(provider: str, resp: httpx.Response) -> RuntimeError:
    payload = _safe_response_payload(resp)
    return RuntimeError(f"{provider} error {resp.status_code}: {json.dumps(payload)[:200]}")


def _is_retryable_status(status_code: int) -> bool:
    # 529 = Anthropic-style "overloaded": a transient outage, retryable like the
    # other 5xx. Its omission left both the Kimi path and (once unified) MiniMax
    # treating an overload as a hard failure instead of a quick backoff-retry.
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _retry_same_provider(category: str) -> bool:
    """Whether an in-provider retry is worth attempting for this category.

    A rate-limit (429 / quota) won't clear in the 1-2s an in-provider
    retry sleeps — retrying the SAME provider just burns the call's time
    budget before the fallback/race chain can route to a different
    provider. So `external_rate_limit` gets NO in-provider retry: raise
    at once and let FallbackCaller / RaceCaller pick another provider.
    Other retryable categories (5xx outage, timeout, empty / incomplete
    stream) DO clear quickly, so they keep their in-provider retry.
    """
    return category != "external_rate_limit"


def _normalize_content(text: str, *, json_mode: bool) -> str:
    if json_mode:
        return extract_json_text(text).strip()
    # Prose: think-strip only — never fence-unwrap, so a code block inside the
    # body survives instead of being mistaken for the whole payload.
    return strip_think(text).strip()


def _count_thinking_chunks(pieces: list[str]) -> int:
    """How many streamed chunks fell inside a ``<think>…</think>`` reasoning span
    — a CoT-latency diagnostic for the MiniMax log, nothing more. A chunk counts
    while the span is open (opener and closer may share one chunk or split across
    chunks); answer chunks AFTER the close do not count. The old inline test
    (``"<think" in p or thinking_count and "</think>" not in p``) never reset its
    state, so once thinking began it mislabeled the entire answer as thinking."""
    n = 0
    in_thinking = False
    for piece in pieces:
        low = piece.lower()
        if "<think" in low:
            in_thinking = True
        if in_thinking:
            n += 1
        if "</think>" in low:
            in_thinking = False
    return n


def _load_config(ctx: "Context | None" = None) -> dict:
    # Saddle's secrets/policy split: api keys come from a SHARED file, routing
    # (priority + caps) from saddle's OWN config/llm_policy.json. merged_config
    # overlays the two into the {"providers": {...}} shape the factories expect.
    # The resolved set is tenant/project-specific (saddle.llm.policy +
    # saddle.context); ctx=None resolves the ambient (env + cwd) context.
    return merged_config(ctx)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _get_kimi_token() -> str:
    global _kimi_token_cache, _kimi_token_cache_time
    now = time.time()
    if _kimi_token_cache and now - _kimi_token_cache_time < 30:
        return _kimi_token_cache["access_token"]
    # Serialize refresh across threads (in-process) to prevent stampede.
    with _kimi_refresh_lock:
        # Re-check after acquiring lock — another thread may have refreshed.
        now = time.time()
        if _kimi_token_cache and now - _kimi_token_cache_time < 30:
            return _kimi_token_cache["access_token"]
        creds = json.loads(_KIMI_CRED_PATH.read_text(encoding="utf-8"))
        if creds.get("expires_at", 0) - now < 300:
            body = urllib.parse.urlencode({
                "grant_type": "refresh_token",
                "refresh_token": creds["refresh_token"],
                "client_id": _KIMI_CLIENT_ID,
            }).encode()
            req = urllib.request.Request(
                _KIMI_AUTH_URL, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                new_creds = json.loads(resp.read())
            new_creds["expires_at"] = time.time() + new_creds.get("expires_in", 900)
            # Atomic write: tmp file + rename prevents concurrent readers
            # from seeing a half-written credential file.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(_KIMI_CRED_PATH.parent), suffix=".tmp",
            )
            try:
                os.write(tmp_fd, json.dumps(new_creds).encode("utf-8"))
                os.close(tmp_fd)
                os.chmod(tmp_path, 0o600)
                os.replace(tmp_path, str(_KIMI_CRED_PATH))
            except BaseException:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            creds = new_creds
        _kimi_token_cache = creds
        _kimi_token_cache_time = now
        return creds["access_token"]


class GlmCaller:
    """GLM caller — uses Anthropic-compatible endpoint at z.ai by default.

    The z.ai anthropic endpoint works reliably. The paas/v4 OpenAI-compat
    endpoint throws billing errors even with balance remaining — avoid it.

    For web search: uses the paas/v4 endpoint with tools parameter since
    the anthropic endpoint doesn't support tools. Falls back gracefully.
    """

    # Concurrent-request admission cap — see MiniMaxCaller for the
    # full design comment. GLM is rarely picked (not in ACTIVE_PRIORITY
    # by default); 4 is a conservative tested ceiling.
    concurrent_request_cap: int = 4


    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        base = cfg.get("api_base_url", "https://api.z.ai/api/anthropic/v1")
        # Keep the anthropic endpoint as-is — it works
        if "/anthropic/" in base:
            self._anthropic_url = base.rstrip("/") + "/messages"
            self._openai_url = "https://api.z.ai/api/paas/v4/chat/completions"
            self._use_anthropic = True
        elif base.endswith("/chat/completions"):
            self._anthropic_url = None
            self._openai_url = base
            self._use_anthropic = False
        else:
            self._anthropic_url = None
            self._openai_url = f"{base}/chat/completions"
            self._use_anthropic = False

    async def _call_anthropic(
        self, system: str, prompt: str, *, json_mode: bool, label: str,
    ) -> str:
        """Call via Anthropic-compatible endpoint (reliable, no web search)."""
        body: dict = {
            "model": self._cfg.get("model", "GLM-5.1"),
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
        }
        slot_label = f"GLM-anthropic/{label}" if label else "GLM-anthropic"
        async with PoolSlot(get_pool(), slot_label):
            async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 600)) as client:
                resp = await client.post(
                    self._anthropic_url,
                    headers={
                        "x-api-key": self._cfg["api_key"],
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        data = _safe_response_payload(resp)
        if resp.status_code != 200:
            raise _http_error("GLM-anthropic", resp)
        # Anthropic format: data.content[0].text
        content = data.get("content", [])
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
        else:
            text = str(content)
        text = _normalize_content(text, json_mode=json_mode)
        if not text:
            raise RuntimeError(f"GLM-anthropic empty response: {json.dumps(data)[:300]}")
        return text

    async def _call_openai(
        self, system: str, prompt: str, *,
        json_mode: bool, label: str, web_search: bool = False,
    ) -> str:
        """Call via OpenAI-compatible endpoint (needed for web search tools)."""
        body: dict = {
            "model": self._cfg.get("model", "GLM-5.1"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if web_search:
            body["tools"] = [{"type": "web_search", "web_search": {"enable": True}}]
        slot_label = f"GLM-openai/{label}" if label else "GLM-openai"
        async with PoolSlot(get_pool(), slot_label):
            async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 600)) as client:
                resp = await client.post(
                    self._openai_url,
                    headers={"Authorization": f"Bearer {self._cfg['api_key']}", "Content-Type": "application/json"},
                    json=body,
                )
        data = _safe_response_payload(resp)
        if resp.status_code != 200:
            raise _http_error("GLM-openai", resp)
        text = _normalize_content(data["choices"][0]["message"]["content"], json_mode=json_mode)
        if not text:
            raise RuntimeError(f"GLM-openai empty response: {json.dumps(data)[:300]}")
        return text

    async def __call__(
        self, system: str, prompt: str, *,
        json_mode: bool = False, label: str = "", web_search: bool = False,
    ) -> str:
        # Web search requires OpenAI-compat endpoint (anthropic doesn't support tools)
        if web_search:
            return await self._call_openai(
                system, prompt, json_mode=json_mode, label=label, web_search=True,
            )
        # Normal calls: prefer anthropic endpoint (reliable)
        if self._use_anthropic:
            return await self._call_anthropic(
                system, prompt, json_mode=json_mode, label=label,
            )
        return await self._call_openai(
            system, prompt, json_mode=json_mode, label=label,
        )


# ChatServiceCaller (the sibling project's gallery /api/agent/query HTTP
# relay) was removed for saddle. The Claude surface is now
# ClaudeAgentCaller, which drives the Agent SDK directly
# (saddle.llm.claude_agent) — there is no gallery process to POST to.
# See _build_claude_agent below.


class MiniMaxCaller:
    # Concurrent-request admission cap — max in-flight calls to this
    # provider AT ONCE before the IntentRouter routes new calls
    # elsewhere. Proactive admission control: prevents the 20-parallel-
    # call burst that 429-cascaded Kimi on 2026-05-13.
    #
    # MiniMax: the dynamic per-provider limiter (llm_pool._DynamicLimiter) uses
    # this as the FLOOR and self-tunes UP toward the empirical 429 onset
    # (~45 concurrent — measured from the crash runs, 2026-06-20). 36 = ~80% of
    # the lowest observed onset (45), a strong safe floor; config override
    # (providers.minimax.concurrent_request_cap) / RAYXI_MINIMAX_CONCURRENT_CAP
    # still win. The 429 limit is really tokens/minute, so the dynamic cap +
    # 429-latch handle sustained-rate throttling above this floor.
    concurrent_request_cap: int = 36

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        # Per-instance admission cap override. MiniMax's token PLAN carries a
        # per-minute token rate limit (429 "Token Plan rate limit reached
        # (2062)") that a 20-concurrent burst of large content_pack calls
        # blows through — and with minimax the sole content_pack provider
        # (deepseek out of credit), there's nothing to fail over to, so the
        # build hard-fails. Pace the burst by lowering the cap. Config
        # (`providers.minimax.concurrent_request_cap`) or env
        # (RAYXI_MINIMAX_CONCURRENT_CAP) overrides the class default; a
        # lower cap trades build wall-clock for staying under the plan.
        _cap_override = cfg.get("concurrent_request_cap")
        if _cap_override is None:
            _env_cap = _first_env("RAYXI_MINIMAX_CONCURRENT_CAP")
            _cap_override = _env_cap if _env_cap else None
        if _cap_override is not None:
            try:
                self.concurrent_request_cap = max(1, int(_cap_override))
            except (TypeError, ValueError):
                pass

    async def _call_streaming(
        self,
        client: httpx.AsyncClient,
        body: dict,
        *,
        label: str,
        json_mode: bool,
    ) -> str:
        content_parts: list[str] = []
        chunk_count = 0
        saw_done = False

        async with client.stream(
            "POST",
            self._cfg["api_base_url"],
            headers={"Authorization": f"Bearer {self._cfg['api_key']}", "Content-Type": "application/json"},
            json=body,
        ) as resp:
            if resp.status_code != 200:
                body_text = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"MiniMax error {resp.status_code}: {body_text[:500]}")

            # Wrap the SSE line stream in an idle heartbeat: httpx's read timeout
            # only fires if a single read blocks, so a connection that stays OPEN
            # but goes SILENT mid-turn would hang up to the full 600s read ceiling
            # with nothing to distinguish "still thinking" from "wedged". The
            # heartbeat resets on every line, so real CoT streaming sails through
            # while true silence past the idle window raises Stalled — the same
            # liveness discipline the claude path already has on its stream.
            async for raw_line in supervise.heartbeat(
                resp.aiter_lines(),
                idle_seconds=_minimax_stream_idle_seconds(),
                max_seconds=_minimax_stream_max_seconds(),
                what=f"minimax stream {label or 'request'}",
            ):
                line = (raw_line or "").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    saw_done = True
                    break
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    _log.debug("MiniMax stream non-JSON frame for %s: %s", label or "request", line[:160])
                    continue

                choices = frame.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                piece = (
                    delta.get("content")
                    or message.get("content")
                    or delta.get("reasoning_content")
                    or message.get("reasoning_content")
                    or ""
                )
                if not isinstance(piece, str) or not piece:
                    continue
                chunk_count += 1
                content_parts.append(piece)
                if chunk_count == 1:
                    _log.info("MiniMax stream started for %s", label or "request")

        text = "".join(content_parts)
        _log.info(
            "MiniMax stream finished for %s: chunks=%d thinking_chunks=%d done=%s",
            label or "request", chunk_count, _count_thinking_chunks(content_parts), saw_done,
        )
        return _normalize_content(text, json_mode=json_mode)

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False, label: str = "") -> str:
        body: dict = {
            "model": self._cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
        }
        # Thinking OFF by default (operator directive 2026-06-27). MiniMax-M3 is
        # a reasoning model: it emits a <think>…</think> trace before the answer
        # (stripped downstream). For saddle's use — supervisory verdicts plus the
        # default structured path — that trace is HIDDEN from the user and only
        # adds latency; saddle consumes the produced result, never the CoT. So
        # disable it unless a call explicitly opts back in via
        # `providers.minimax.thinking` (the rare phase that genuinely wants the
        # deeper reasoning pass). Only `thinking: {type: disabled}` actually
        # stops generation — other vendor flags were ignored (tested live).
        _thinking = self._cfg.get("thinking")
        if _thinking is None:
            _thinking = {"type": "disabled"}
        body["thinking"] = _thinking
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        use_stream = bool(self._cfg.get("stream", False))
        if use_stream:
            body["stream"] = True

        last_exc: Exception | None = None
        # 2 retries max — per project rule "every LLM retry is a bug",
        # empty/malformed responses should surface the contract gap fast.
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 600)) as client:
                    if use_stream:
                        text = await self._call_streaming(client, body, label=label, json_mode=json_mode)
                        if not text:
                            last_exc = RuntimeError("MiniMax empty streaming response after wrapper-strip")
                            if attempt < 1:
                                wait = 2 ** attempt
                                _category = categorize_retry(raw_response="")
                                _log.warning(
                                    "MiniMax empty streaming response for %s "
                                    "— retry %d in %ds [retry_category=%s]",
                                    label or "request", attempt + 1, wait,
                                    _category,
                                )
                                await asyncio.sleep(wait)
                                continue
                            raise last_exc
                        return text
                    resp = await client.post(
                        self._cfg["api_base_url"],
                        headers={"Authorization": f"Bearer {self._cfg['api_key']}", "Content-Type": "application/json"},
                        json=body,
                    )
                if _is_retryable_status(resp.status_code):
                    # Use the canonical retryable-status set (like KimiCaller)
                    # rather than a hand-rolled `== 529 or == 500`, which silently
                    # excluded 502/503/504 — gateway outages every bit as transient
                    # as 500/529 — so they raised with no in-provider backoff-retry.
                    # First detect permanent token-plan errors and bail
                    # immediately — retrying a "your current token plan not
                    # support model X" 4 times with 1+2+4+8s backoff is 15s
                    # of pure waste that compounds across ~70 calls per
                    # pipeline run (~17 min lost per attempt).
                    body_text = (resp.text or "")[:500]
                    permanent_markers = (
                        "token plan not support",
                        "model not support",
                        "model not found",
                        "invalid model",
                    )
                    if any(marker in body_text for marker in permanent_markers):
                        _log.warning(
                            "MiniMax %d for %s — permanent error, no retry: %s",
                            resp.status_code, label or "request", body_text[:200],
                        )
                        raise _http_error("MiniMax", resp)
                    wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                    # 5xx → provider_outage; 429 → external_rate_limit (which
                    # _retry_same_provider sends straight to failover). categorize
                    # reads status + body so the surfacing stays accurate.
                    _category = categorize_retry(
                        status_code=resp.status_code,
                        raw_response=body_text,
                    )
                    if not _retry_same_provider(_category):
                        # Throttle dressed as a 5xx — a quota won't reset
                        # in the backoff window. Fail over to another
                        # provider instead of retrying this one.
                        _log.warning(
                            "MiniMax %d for %s — rate-limited, no in-provider "
                            "retry, failing over [retry_category=%s]",
                            resp.status_code, label or "request", _category,
                        )
                        raise _http_error("MiniMax", resp)
                    _log.warning(
                        "MiniMax %d for %s — retry %d in %ds "
                        "[retry_category=%s]",
                        resp.status_code, label or "request",
                        attempt + 1, wait, _category,
                    )
                    last_exc = _http_error("MiniMax", resp)
                    await asyncio.sleep(wait)
                    continue
                data = _safe_response_payload(resp)
                if resp.status_code != 200:
                    raise _http_error("MiniMax", resp)
                choices = data.get("choices")
                if not choices or not choices[0].get("message", {}).get("content"):
                    last_exc = RuntimeError(f"MiniMax empty response: {json.dumps(data)[:300]}")
                    if attempt < 1:
                        wait = 2 ** attempt
                        _category = categorize_retry(raw_response="")
                        _log.warning(
                            "MiniMax empty response for %s — retry %d "
                            "in %ds [retry_category=%s]",
                            label or "request", attempt + 1, wait,
                            _category,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise last_exc
                text = _normalize_content(choices[0]["message"]["content"], json_mode=json_mode)
                if not text:
                    last_exc = RuntimeError("MiniMax empty response after think-strip")
                    if attempt < 1:
                        wait = 2 ** attempt
                        _category = categorize_retry(raw_response="")
                        _log.warning(
                            "MiniMax empty response after wrapper-strip "
                            "for %s — retry %d in %ds [retry_category=%s]",
                            label or "request", attempt + 1, wait,
                            _category,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise last_exc
                return text
            except (httpx.TimeoutException, httpx.TransportError,
                    supervise.Stalled, supervise.DeadlineExceeded) as exc:
                # A wedged-but-open stream (supervise.Stalled / DeadlineExceeded)
                # is treated exactly like an httpx transport timeout: retry once
                # in case it was a momentary provider hiccup, then RAISE so the
                # FallbackCaller routes the call to the next provider instead of
                # the run sitting silent — the user's "I never see you return"
                # failure mode, now turned into a fast, typed failover.
                last_exc = exc
                if attempt < 1:
                    wait = 2 ** attempt
                    _category = categorize_retry(exception=exc)
                    _log.warning(
                        "MiniMax transport error for %s — retry %d in %ds "
                        "[retry_category=%s]",
                        label or "request", attempt + 1, wait, _category,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"MiniMax failed after 2 attempts: {last_exc}")


def _resolve_kimi_config(providers: dict) -> dict | None:
    for name in ("kimi", "kimik"):
        cfg = providers.get(name)
        if isinstance(cfg, dict) and cfg.get("api_key"):
            fixed = dict(cfg)
            if not fixed.get("api_base_url"):
                fixed["api_base_url"] = _KIMI_DEFAULT_API_BASE_URL
            if not fixed.get("model"):
                fixed["model"] = _KIMI_DEFAULT_MODEL
            return fixed

    env_key = _first_env(
        "RAYXI_KIMIK_API_KEY",
        "KIMIK_API_KEY",
        "RAYXI_KIMI_API_KEY",
        "KIMI_API_KEY",
        "MOONSHOT_API_KEY",
    )
    if env_key:
        return {
            "api_key": env_key,
            "api_base_url": _KIMI_DEFAULT_API_BASE_URL,
            "model": _KIMI_DEFAULT_MODEL,
        }

    # Kimi Code keys are sometimes stored under older "moonshot" config
    # names. They authenticate against api.kimi.com, not api.moonshot.ai.
    cfg = providers.get("moonshot")
    if isinstance(cfg, dict) and str(cfg.get("api_key", "")).startswith("sk-kimi"):
        fixed = dict(cfg)
        base = str(fixed.get("api_base_url", "")).lower()
        if "moonshot.ai" in base or not base:
            fixed["api_base_url"] = _KIMI_DEFAULT_API_BASE_URL
        if not fixed.get("model"):
            fixed["model"] = _KIMI_DEFAULT_MODEL
        return fixed
    return None


class KimiCaller:
    # Concurrent-request admission cap — see MiniMaxCaller for the
    # full design comment. Kimi's measured ceiling is ~5 req/sec
    # safe burst before 429s land (inferred from the 2026-05-13
    # cascade where 20 parallel calls all 429'd within 0.16s).
    # Reserved for JSON-gen / coding / design intents where its
    # quality edge is worth the lower throughput. Bulk-burst phases
    # fall through to MiniMax (cap=20) automatically when this cap
    # saturates.
    concurrent_request_cap: int = 5

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        self._api_key = str(self._cfg.get("api_key", "")).strip() or None
        base = str(self._cfg.get("api_base_url", _KIMI_DEFAULT_API_BASE_URL)).rstrip("/")
        self._url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False, label: str = "") -> str:
        token = self._api_key or _get_kimi_token()
        body: dict = {
            "model": self._cfg.get("model") or _KIMI_DEFAULT_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
        }
        # Thinking-mode toggle. kimi-for-coding ships with extended
        # thinking ON by default — single calls land at 25-130s because
        # the model dumps a long internal monologue before producing
        # the output the pipeline actually parses. The thinking is rarely
        # load-bearing for our structured-JSON contracts; turning it off
        # gets us back to deepseek-flash latency territory while
        # keeping kimi's qualitative advantage on the structural
        # validators (kit_size enums, role_primary_stat lookups, etc.).
        # Override via cfg["enable_thinking"]=true for the (rare) case
        # where a phase actually wants the deeper reasoning pass.
        enable_thinking = self._cfg.get("enable_thinking", False)
        body["enable_thinking"] = bool(enable_thinking)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        slot_label = f"Kimi/{label}" if label else "Kimi"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "claude-code/2.0",
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 1800)) as client:
                    resp = await client.post(
                        self._url,
                        headers=headers,
                        json=body,
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == 2:
                    raise
                # Categorize: timeout vs transport error (could be
                # rate limit body inside the exception text).
                _category = categorize_retry(exception=exc)
                _log.warning(
                    "KimiCaller transport failure for %s on attempt %d — "
                    "retrying [retry_category=%s]",
                    label or "request",
                    attempt + 1,
                    _category,
                )
                await asyncio.sleep(1.5)
                continue

            if resp.status_code != 200:
                last_exc = _http_error("Kimi", resp)
                # Categorize by status: 429 → rate_limit, 5xx →
                # outage, 408/413/414 → timeout/input_too_large.
                _category = categorize_retry(
                    status_code=resp.status_code,
                    raw_response=getattr(resp, "text", "") or "",
                )
                if (
                    attempt < 2
                    and _is_retryable_status(resp.status_code)
                    and _retry_same_provider(_category)
                ):
                    _log.warning(
                        "KimiCaller HTTP %d for %s on attempt %d — "
                        "retrying [retry_category=%s]",
                        resp.status_code,
                        label or "request",
                        attempt + 1,
                        _category,
                    )
                    await asyncio.sleep(1.5)
                    continue
                # Rate-limited (or non-retryable): raise now so the
                # fallback/race chain routes to a different provider — a
                # quota won't reset in the 1.5s an in-provider retry waits.
                if _category == "external_rate_limit":
                    _log.warning(
                        "KimiCaller HTTP %d for %s — rate-limited, no "
                        "in-provider retry, failing over [retry_category=%s]",
                        resp.status_code, label or "request", _category,
                    )
                raise last_exc

            data = _safe_response_payload(resp)
            choices = data.get("choices")
            if not choices or not choices[0].get("message", {}).get("content"):
                last_exc = RuntimeError(f"Kimi empty response: {json.dumps(data)[:300]}")
                if attempt < 2:
                    # Provider returned 200 but no content → empty_response.
                    _category = categorize_retry(raw_response="")
                    _log.warning(
                        "KimiCaller empty response for %s on attempt %d "
                        "— retrying [retry_category=%s]",
                        label or "request",
                        attempt + 1,
                        _category,
                    )
                    await asyncio.sleep(1.5)
                    continue
                raise last_exc
            text = _normalize_content(choices[0]["message"]["content"], json_mode=json_mode)
            if not text:
                last_exc = RuntimeError(f"Kimi empty response: {json.dumps(data)[:300]}")
                if attempt < 2:
                    _category = categorize_retry(raw_response="")
                    _log.warning(
                        "KimiCaller blank normalized response for %s on "
                        "attempt %d — retrying [retry_category=%s]",
                        label or "request",
                        attempt + 1,
                        _category,
                    )
                    await asyncio.sleep(1.5)
                    continue
                raise last_exc
            return text

        raise RuntimeError(f"Kimi request failed after retries: {last_exc}")


class DeepSeekCaller:
    """DeepSeek-V4 caller — uses Anthropic-compatible endpoint at
    api.deepseek.com/anthropic.

    Same Messages-API shape as Anthropic / GLM-anthropic / KimiCaller:
    POST /v1/messages with x-api-key header.  When Anthropic-format
    is unavailable, falls back to OpenAI-compat at /chat/completions
    (same pattern as GlmCaller).
    """

    # Concurrent-request admission cap — see MiniMaxCaller for the
    # full design comment. DeepSeek Flash has moderate published rpm;
    # tuned to 10 in-flight, well below typical paid-tier quotas.
    # Pro mode (reasoning_heavy intent only) uses the same class so
    # the cap covers both — pro calls are rare enough this cap
    # functions effectively as a flash-burst cap.
    concurrent_request_cap: int = 10

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        base = cfg.get("api_base_url", "https://api.deepseek.com/anthropic").rstrip("/")
        if base.endswith("/anthropic"):
            self._anthropic_url = f"{base}/v1/messages"
            self._openai_url = "https://api.deepseek.com/v1/chat/completions"
            self._use_anthropic = True
        elif base.endswith("/v1/messages"):
            self._anthropic_url = base
            self._openai_url = "https://api.deepseek.com/v1/chat/completions"
            self._use_anthropic = True
        elif base.endswith("/chat/completions"):
            self._anthropic_url = None
            self._openai_url = base
            self._use_anthropic = False
        else:
            self._anthropic_url = None
            self._openai_url = f"{base}/chat/completions"
            self._use_anthropic = False

    async def _call_anthropic(
        self, system: str, prompt: str, *, json_mode: bool, label: str,
    ) -> str:
        body: dict = {
            "model": self._cfg.get("model", "DeepSeek-V4-Pro"),
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
            # Thinking off. DeepSeek rejects the combo of
            # thinking={"type":"disabled"} + output_config.effort on
            # both Anthropic-compat AND OpenAI-compat endpoints
            # (effort is a thinking-mode-only parameter per provider
            # design — internally the Anthropic endpoint maps
            # output_config.effort to reasoning_effort and validates
            # the same constraint). Keeping thinking off so the
            # streaming response is the structured-output payload
            # only; effort knob omitted because it can't coexist with
            # disabled thinking.
            "thinking": {"type": "disabled"},
        }
        slot_label = f"DeepSeek-anthropic/{label}" if label else "DeepSeek-anthropic"
        async with PoolSlot(get_pool(), slot_label):
            async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 600)) as client:
                resp = await client.post(
                    self._anthropic_url,
                    headers={
                        "x-api-key": self._cfg["api_key"],
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        data = _safe_response_payload(resp)
        if resp.status_code != 200:
            raise _http_error("DeepSeek-anthropic", resp)
        # DeepSeek's Anthropic endpoint returns interleaved thinking +
        # text blocks: [{"type":"thinking","thinking":"..."}, {"type":"text","text":"..."}].
        # Find the first text block (or concatenate all of them).
        content = data.get("content", [])
        text = ""
        if isinstance(content, list):
            text_parts = [
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            text = "".join(text_parts)
            if not text:
                # Fallback: some providers put the answer in "thinking"
                # when no text block was emitted.
                think_parts = [
                    blk.get("thinking", "")
                    for blk in content
                    if isinstance(blk, dict) and blk.get("type") == "thinking"
                ]
                text = "".join(think_parts)
        elif content:
            text = str(content)
        text = _normalize_content(text, json_mode=json_mode)
        if not text:
            raise RuntimeError(f"DeepSeek-anthropic empty response: {json.dumps(data)[:300]}")
        return text

    async def _call_openai(
        self, system: str, prompt: str, *, json_mode: bool, label: str,
    ) -> str:
        body: dict = {
            "model": self._cfg.get("model", "DeepSeek-V4-Pro"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": self._cfg.get("max_tokens", 32768),
            "temperature": self._cfg.get("temperature", 0.7),
            # Thinking off — see _call_anthropic for the rationale.
            # reasoning_effort omitted because DeepSeek rejects it
            # alongside thinking={"type":"disabled"}.
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        slot_label = f"DeepSeek-openai/{label}" if label else "DeepSeek-openai"
        async with PoolSlot(get_pool(), slot_label):
            async with httpx.AsyncClient(timeout=self._cfg.get("timeout_seconds", 600)) as client:
                resp = await client.post(
                    self._openai_url,
                    headers={"Authorization": f"Bearer {self._cfg['api_key']}", "Content-Type": "application/json"},
                    json=body,
                )
        data = _safe_response_payload(resp)
        if resp.status_code != 200:
            raise _http_error("DeepSeek-openai", resp)
        text = _normalize_content(data["choices"][0]["message"]["content"], json_mode=json_mode)
        if not text:
            raise RuntimeError(f"DeepSeek-openai empty response: {json.dumps(data)[:300]}")
        return text

    async def __call__(
        self, system: str, prompt: str, *, json_mode: bool = False, label: str = "",
    ) -> str:
        # Always use the Anthropic-compatible endpoint. DeepSeek's
        # OpenAI-compat endpoint rejects the (thinking={"type":"disabled"}
        # + reasoning_effort="max") combo we want for max-quality
        # structured output, while the Anthropic endpoint accepts both
        # together. OpenAI-compat is kept as a fallback only when the
        # Anthropic endpoint is unavailable.
        if self._use_anthropic and self._anthropic_url:
            try:
                return await self._call_anthropic(system, prompt, json_mode=json_mode, label=label)
            except Exception as exc:
                _log.warning(
                    "DeepSeek-anthropic failed (%s), falling back to OpenAI endpoint",
                    str(exc)[:120],
                )
        return await self._call_openai(system, prompt, json_mode=json_mode, label=label)


# Failure categories (from the canonical `categorize_retry`) that mean "this
# PROVIDER is the problem; a DIFFERENT provider may succeed" — FallbackCaller
# routes past them. The complement (empty_response / parse_error /
# validation_failed / input_too_large / output_too_large / other) is OUR
# contract gap: the next provider would hit the same defect, so we re-raise and
# surface it instead of silently masking it behind failover.
_FAILOVER_ROUTE_CATEGORIES = frozenset({
    "external_rate_limit",
    "provider_outage",
    "timeout",
    "provider_blocked",
})

# Provider-side safety-classifier rejections. A benign prompt one provider
# flags, another may accept — so route past. Kept as explicit PHRASES (never
# bare codes) so they cannot over-match a genuine error the way "500" or "403"
# substrings did.
_CONTENT_FILTER_MARKERS = (
    "content_filter", "content filter", "high risk", "request was rejected",
)


def _is_content_filter_rejection(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _CONTENT_FILTER_MARKERS)


def _failover_routes_past(exc: Exception) -> bool:
    """Should FallbackCaller degrade to the next provider for this failure?

    Classify by the SHAPE of the failure (typed signal, structured HTTP status,
    exception type, then text) via the canonical categorizer the retry loop
    already uses — NOT by hunting bare numeric substrings like "500"/"403"/
    "1234" in the message. Those over-match: a schema error reading "expected
    500 rows" must not be mistaken for a provider 500 and routed past, masking
    the real defect (the same false-classification family that crashed the
    dogfood, just in the opposite direction)."""
    # A lead-provider subprocess crash (the local `claude` CLI exiting non-zero)
    # is a routing failure regardless of what its captured stderr says.
    if isinstance(exc, ClaudeAgentUnavailable):
        return True
    if _is_content_filter_rejection(str(exc)):
        return True
    # Prefer the structured HTTP status when the exception carries one, so a
    # real provider 5xx routes even when its message has no reason phrase.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    category = categorize_retry(exception=exc, status_code=status)
    return category in _FAILOVER_ROUTE_CATEGORIES


class FallbackCaller:
    """Try providers in order; route past a transient/unavailable failure to the
    next; re-raise a genuine error so real defects aren't masked.

    A CIRCUIT BREAKER keeps a provider that fails on EVERY call from being
    re-attempted on every call: after ``failure_threshold`` consecutive
    failover-routing failures the provider is tripped OPEN and skipped for
    ``cooldown_s`` — then a single half-open re-probe lets it back in the moment
    it recovers (a success resets the breaker). Without this, a reliably-down
    lead (e.g. a local CLI that crashes on every spawn) makes every call in a run
    pay its full failure latency before falling over — the exact wasted-work
    drag observed auditing rayxi with a wedged ``claude`` CLI. The breaker NEVER
    leaves the chain with zero attempts: if every provider is open, a last-resort
    pass tries the open ones anyway.
    """

    def __init__(
        self,
        callers: list,
        *,
        failure_threshold: int | None = None,
        cooldown_s: float | None = None,
    ) -> None:
        self._callers = callers
        self._failure_threshold = (
            _circuit_failure_threshold() if failure_threshold is None
            else max(0, failure_threshold)
        )
        self._cooldown_s = (
            _circuit_cooldown_s() if cooldown_s is None else cooldown_s
        )
        # keyed by chain index: consecutive failures, and the monotonic time the
        # provider's circuit re-closes (0.0 == closed).
        self._fail_counts: dict[int, int] = {}
        self._open_until: dict[int, float] = {}

    def _breaker_open(self, i: int, now: float) -> bool:
        return self._open_until.get(i, 0.0) > now

    def _record_success(self, i: int) -> None:
        if self._fail_counts.get(i) or self._open_until.get(i):
            _log.info(
                "FallbackCaller: %s recovered — circuit reset",
                type(self._callers[i]).__name__,
            )
        self._fail_counts[i] = 0
        self._open_until[i] = 0.0

    def _record_failure(self, i: int, now: float) -> None:
        if self._failure_threshold <= 0:
            return
        count = self._fail_counts.get(i, 0) + 1
        self._fail_counts[i] = count
        if count >= self._failure_threshold and not self._breaker_open(i, now):
            self._open_until[i] = now + self._cooldown_s
            _log.warning(
                "FallbackCaller: %s tripped circuit after %d consecutive "
                "failures — skipping for %.0fs",
                type(self._callers[i]).__name__, count, self._cooldown_s,
            )

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False, label: str = "") -> str:
        last_exc: Exception | None = None
        threshold = self._failure_threshold
        # Pass 1 tries providers whose circuit is CLOSED (or whose cooldown has
        # lapsed → a half-open re-probe). Pass 2 only runs if pass 1 SKIPPED an
        # open provider and nothing has yet succeeded — it tries the still-open
        # ones as a last resort so a tripped breaker never yields zero attempts.
        skipped_open = False
        for pass_no in (1, 2):
            if pass_no == 2 and not skipped_open:
                break
            for i, caller in enumerate(self._callers):
                now = time.monotonic()
                is_open = self._breaker_open(i, now)
                if pass_no == 1 and threshold > 0 and is_open:
                    skipped_open = True
                    _log.info(
                        "FallbackCaller: %s circuit open — skipping, trying next…",
                        type(caller).__name__,
                    )
                    continue
                if pass_no == 2 and not is_open:
                    continue  # already attempted in pass 1
                try:
                    result = await _call_with_provider_deadline(
                        caller,
                        system,
                        prompt,
                        json_mode=json_mode,
                        label=label,
                    )
                    self._record_success(i)
                    return result
                except Exception as exc:
                    if _failover_routes_past(exc):
                        self._record_failure(i, time.monotonic())
                        _log.warning(
                            "FallbackCaller: %s failed (%s), trying next…",
                            type(caller).__name__, str(exc)[:120],
                        )
                        last_exc = exc
                    else:
                        raise
        raise RuntimeError(f"All callers failed. Last error: {last_exc}")


class RaceCaller:
    """Fires every racer concurrently and returns the FIRST successful response.
    Loser requests are cancelled the moment the winner resolves. Failures from
    an early-returning racer don't abort — the caller keeps waiting on the
    other racers until one succeeds or all fail.

    Context: sequential FallbackCaller makes every call pay the primary's
    full latency + retry window before trying the secondary. With MiniMax
    running 200-600s for a 30KB content_pack prompt and occasional
    transport stalls, that single-provider latency dominates wall time.
    Racing MiniMax + Kimi in parallel halves expected latency (whichever
    is faster wins) AND hides per-provider transient failures (the slow/
    failing one is ignored if the other succeeds first).
    """

    def __init__(self, racers: list) -> None:
        if not racers:
            raise ValueError("RaceCaller requires at least one racer")
        self._racers = list(racers)

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False, label: str = "") -> str:
        tasks: list[asyncio.Task] = [
            asyncio.create_task(_call_with_provider_deadline(
                racer,
                system,
                prompt,
                json_mode=json_mode,
                label=label,
            ))
            for racer in self._racers
        ]
        errors: list[Exception] = []
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception()
                    if exc is None:
                        result = task.result()
                        if json_mode:
                            try:
                                parsed = json.loads(result)
                            except Exception as parse_exc:
                                errors.append(RuntimeError(
                                    "RaceCaller json_mode rejected non-JSON racer response: "
                                    f"{parse_exc}"
                                ))
                                _log.warning(
                                    "RaceCaller: racer returned non-JSON in json_mode, "
                                    "waiting on %d other(s)…",
                                    len(pending),
                                )
                                continue
                            if not isinstance(parsed, dict):
                                errors.append(RuntimeError(
                                    "RaceCaller json_mode expected top-level JSON object, got "
                                    f"{type(parsed).__name__}"
                                ))
                                _log.warning(
                                    "RaceCaller: racer returned top-level %s in json_mode, "
                                    "waiting on %d other(s)…",
                                    type(parsed).__name__,
                                    len(pending),
                                )
                                continue
                        # Winner — cancel every loser and return the result.
                        for loser in pending:
                            loser.cancel()
                        return result
                    errors.append(exc)
                    _log.warning(
                        "RaceCaller: racer failed (%s), waiting on %d other(s)…",
                        str(exc)[:120], len(pending),
                    )
            # Every racer failed.
            last = errors[-1] if errors else RuntimeError("no racers returned")
            raise RuntimeError(
                f"RaceCaller: all {len(self._racers)} racer(s) failed. Last error: {last}"
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()


# ---------------------------------------------------------------------------
# Caller routing — maps call types to the right LLM
# ---------------------------------------------------------------------------

# Named priority list — single source of truth for both
# `build_callers` (FallbackCaller chain assembly) and `LLMPool`
# (intent-based provider rotation).  iter 13 (2026-05-02): MiniMax came
# back online (plan re-bought) and DeepSeek added a flash variant
# (deepseek-v4-flash, ~4x faster than -pro).  New active chain orders
# providers by observed single-call latency from the post-restore
# probes: minimax 1.3s < deepseek_flash 1.5s < kimi 2.3s.  DeepSeek-Pro
# stays registered but is opt-in only (intent="reasoning_heavy" or
# explicit provider="deepseek_pro" — not in the rotation chain).
# GLM remains dropped at user request.
# ---------------------------------------------------------------------------
# Caller registry + active priority
#
# Two structures, separated by purpose:
#
#   ALL_CALLERS — the full registry. Every caller this codebase knows
#   about gets a factory entry here. Add a new provider by adding a
#   factory; the registry stays the source-of-truth list.
#
#   ACTIVE_PRIORITY — the active rotation for the default fallback +
#   race chains. Edit this list to change the order without touching
#   any factory. Names must exist in ALL_CALLERS (or be skipped at
#   build time if the caller couldn't be constructed). LEFTMOST WINS.
#
# Cron history: provider order has changed multiple times as plans
# came/went online (MiniMax billing limit → DeepSeek primary →
# MiniMax restored). Splitting registry from order means each rotation
# is a one-line edit to ACTIVE_PRIORITY.
# ---------------------------------------------------------------------------

# Factory registry — name → builder fn that returns an LLMCaller or
# None if the caller can't be constructed (no creds, no config). The
# build_callers loop calls each factory; failures are silent (the
# caller just doesn't get registered, ACTIVE_PRIORITY skips it).
def _build_claude_agent(providers: dict) -> LLMCaller | None:
    # claude_agent: Claude via the Agent SDK (local `claude` CLI,
    # subscription auth — no api key). Saddle's LEAD provider and the
    # same engine as the interactive chat. Registered ONLY when the SDK
    # is importable: on a host without `claude_agent_sdk` installed the
    # caller would raise ModuleNotFoundError at call time and abort the
    # whole fallback chain. Gate it at construction instead — the same
    # "factory returns None when the caller can't be built" contract the
    # cred-less providers use — so the chain falls through to the next
    # priority entry (e.g. minimax) on SDK-less hosts.
    import importlib.util
    if importlib.util.find_spec("claude_agent_sdk") is None:
        _log.info(
            "claude_agent: claude_agent_sdk not installed — provider skipped"
        )
        return None
    from .claude_agent import ClaudeAgentCaller
    return ClaudeAgentCaller(providers.get("claude_agent") or {})


def _build_kimi(providers: dict) -> LLMCaller | None:
    cfg = _resolve_kimi_config(providers)
    if cfg is None and not _KIMI_CRED_PATH.exists():
        return None
    return KimiCaller(cfg)


def _deepseek_config(providers: dict, *names: str) -> dict | None:
    for name in names:
        cfg = providers.get(name)
        if cfg:
            return dict(cfg)
    env_names: list[str] = []
    if "deepseek_flash" in names:
        env_names.extend(["RAYXI_DEEPSEEK_FLASH_API_KEY", "DEEPSEEK_FLASH_API_KEY"])
    if "deepseek_pro" in names:
        env_names.extend(["RAYXI_DEEPSEEK_PRO_API_KEY", "DEEPSEEK_PRO_API_KEY"])
    env_names.extend(["RAYXI_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"])
    api_key = _first_env(*env_names)
    if api_key:
        return {"api_key": api_key}
    return None


def _build_deepseek_pro(providers: dict) -> LLMCaller | None:
    cfg = _deepseek_config(providers, "deepseek_pro", "deepseek")
    if cfg is None:
        return None
    if not cfg.get("api_base_url"):
        cfg["api_base_url"] = _DEEPSEEK_DEFAULT_API_BASE_URL
    if not cfg.get("model"):
        cfg["model"] = _DEEPSEEK_PRO_DEFAULT_MODEL
    caller = DeepSeekCaller(cfg)
    caller.concurrent_request_cap = 10  # pro: lower cap (operator dir)
    return caller


def _build_deepseek_flash(providers: dict) -> LLMCaller | None:
    cfg = _deepseek_config(providers, "deepseek_flash")
    if cfg is None:
        cfg = _deepseek_config(providers, "deepseek", "deepseek_pro")
        if cfg is None:
            return None
        cfg["model"] = cfg.get("flash_model") or _DEEPSEEK_FLASH_DEFAULT_MODEL
    if not cfg.get("api_base_url"):
        cfg["api_base_url"] = _DEEPSEEK_DEFAULT_API_BASE_URL
    if not cfg.get("model"):
        cfg["model"] = _DEEPSEEK_FLASH_DEFAULT_MODEL
    caller = DeepSeekCaller(cfg)
    caller.concurrent_request_cap = 20  # flash: higher cap (operator dir)
    return caller


def _resolve_minimax_config(providers: dict) -> dict | None:
    cfg = providers.get("minimax")
    if cfg:
        resolved = dict(cfg)
    else:
        api_key = os.environ.get("RAYXI_MINIMAX_API_KEY") or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            return None
        resolved = {"api_key": api_key}

    if not resolved.get("api_base_url"):
        resolved["api_base_url"] = (
            os.environ.get("RAYXI_MINIMAX_API_BASE_URL")
            or _MINIMAX_DEFAULT_API_BASE_URL
        )
    if not resolved.get("model"):
        resolved["model"] = os.environ.get("RAYXI_MINIMAX_MODEL") or _MINIMAX_DEFAULT_MODEL
    resolved.setdefault("stream", True)
    return resolved


def _build_minimax(providers: dict) -> LLMCaller | None:
    cfg = _resolve_minimax_config(providers)
    if cfg is None:
        return None
    return MiniMaxCaller(cfg)


def _build_glm(providers: dict) -> LLMCaller | None:
    if "glm" not in providers:
        return None
    return GlmCaller(providers["glm"])


ALL_CALLERS: dict[str, "callable"] = {
    "claude_agent":   _build_claude_agent,
    "deepseek":       _build_deepseek_pro,    # alias for deepseek_pro
    "deepseek_pro":   _build_deepseek_pro,
    "deepseek_flash": _build_deepseek_flash,
    "minimax":        _build_minimax,
    "kimi":           _build_kimi,
    "glm":            _build_glm,
}

# Saddle's rotation is POLICY-DRIVEN (config/llm_policy.json -> priority),
# not hardcoded here. Default when unset: claude_agent -> minimax ->
# deepseek_flash (see saddle.llm.policy.active_priority). Leftmost wins.
ACTIVE_PRIORITY: list[str] = active_priority()

def build_callers(ctx: "Context | None" = None) -> dict[str, LLMCaller]:
    """Construct the LLM caller set for a tenant/project context.

    Returns a dict keyed by provider name plus a "default" fallback
    chain and a "race" parallel chain built from the context's resolved
    priority. Callers registered in ALL_CALLERS but absent from that
    priority stay accessible by name (for explicit provider= overrides)
    but don't appear in the rotation. ctx=None resolves the ambient
    (env + cwd) context.
    """
    cfg = _load_config(ctx)
    providers = cfg.get("providers", {})
    priority = active_priority(ctx)
    callers: dict[str, LLMCaller] = {}

    # Construct every caller in the registry — silent skip on
    # build-failure (no creds, no config). The registry order doesn't
    # matter; ACTIVE_PRIORITY drives the chain.
    for name, factory in ALL_CALLERS.items():
        try:
            caller = factory(providers)
        except Exception as exc:
            _log.warning("caller factory %r failed: %s", name, exc)
            continue
        if caller is not None:
            callers[name] = caller

    # Build the fallback chain in active-priority order; missing
    # callers (factory returned None) drop silently.
    #
    # RAYXI_LLM_SKIP_PROVIDERS=name1,name2 — comma-separated provider
    # names to EXCLUDE from the default chain. Useful on hosts where
    # one provider's endpoint is unreachable (e.g. chat_service on
    # aelimain) — without the skip, every LLM call pays the
    # provider's full retry budget (~5-10s) before falling through
    # to the next provider, ballooning pipeline wall-time. Skipped
    # providers stay accessible by name for explicit override use.
    _skip_raw = os.environ.get("RAYXI_LLM_SKIP_PROVIDERS", "")
    _skip_set = {s.strip() for s in _skip_raw.split(",") if s.strip()}
    chain = [
        callers[name] for name in priority
        if name in callers and name not in _skip_set
    ]
    if not chain:
        raise RuntimeError(
            "No LLM providers available. Configure at least one of: "
            + ", ".join(priority)
            + (f". Skipped via RAYXI_LLM_SKIP_PROVIDERS: {sorted(_skip_set)!r}" if _skip_set else "")
        )
    callers["default"] = FallbackCaller(chain)

    # Z20: race caller — fires the active chain in parallel, returns
    # first valid response. Used by content_pack generation where the
    # failure rate is high enough that paying N× compute for non-
    # determinism is worth it (cron data showed different failure
    # modes per provider — racing gives the validator the best of N
    # attempts).
    callers["race"] = RaceCaller(list(chain))
    return callers
