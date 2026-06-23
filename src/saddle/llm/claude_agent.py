"""Claude Agent SDK callers for saddle.

Two surfaces, both backed by the local ``claude`` CLI via
``claude_agent_sdk`` (subscription auth — no API key):

  - :class:`ClaudeAgentCaller` — a one-shot :class:`LLMCaller` for the
    provider pool. Single turn, tools disabled, returns text. This is
    saddle's lead provider, replacing the gallery-HTTP ``chat_service``
    relay the pool was originally lifted with.

  - :class:`ChatSession` — the interactive, multi-turn, tool-capable agent
    that powers saddle's chat REPL: an agentic Claude like Claude Code
    itself, scoped to the saddle project.

The SDK spawns the ``claude`` binary as a subprocess, so the one-shot
caller is paced through the same memory-aware ProcessPool as the HTTP
providers (``PoolSlot``). ``claude_agent_sdk`` is imported lazily inside
methods so ``import saddle.llm.callers`` works even where the SDK isn't
installed.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, AsyncIterator

from .json_tools import extract_json_text, strip_think
from .pool import PoolSlot, get_pool

if TYPE_CHECKING:  # pragma: no cover
    from claude_agent_sdk import ClaudeAgentOptions

_log = logging.getLogger("saddle.llm.claude_agent")


# saddle's standard Agent-SDK model + effort. Opus 4.8 at xhigh effort.
# The local `claude` CLI's --effort accepts low/medium/high/xhigh/max; the
# Python SDK's Literal type lags (lists only up to "max") but the transport
# passes the string straight through to the CLI, and we build options from an
# untyped kwargs dict, so "xhigh" rides through unflagged. Both are
# overridable per-provider (providers.claude_agent.{model,effort}) or via the
# SADDLE_AGENT_MODEL / SADDLE_AGENT_EFFORT env vars; empty falls back here.
_DEFAULT_MODEL = "claude-opus-4-8"
_DEFAULT_EFFORT = "xhigh"


def _env_model() -> str:
    return (os.environ.get("SADDLE_AGENT_MODEL", "") or "").strip()


def _env_effort() -> str:
    return (os.environ.get("SADDLE_AGENT_EFFORT", "") or "").strip()


class ClaudeAgentUnavailable(RuntimeError):
    """The local ``claude`` CLI subprocess failed for one call.

    The Agent SDK surfaces a CLI crash as an opaque ``Exception`` —
    *"Command failed with exit code 1 — Check stderr output for details"* —
    folding the real cause (rate-limit, overload, transport, auth) into the
    subprocess stderr it then drops. :class:`ClaudeAgentCaller` captures that
    stderr (via the SDK's ``stderr`` callback) and re-raises it as this typed
    error so two things hold:

      1. The real reason is visible in the message + logs, not swallowed.
      2. A subprocess-level crash means the LEAD provider is unavailable for
         THIS call — a routing failure, not a bad prompt — so
         :class:`~saddle.llm.callers.FallbackCaller` always degrades to the
         next provider (e.g. minimax) instead of aborting the whole run.
    """


class ClaudeAgentCaller:
    """One-shot ``LLMCaller`` backed by the Claude Agent SDK.

    Single user turn, tools disabled, no project setting bleed — pure
    text/JSON generation for the provider pool. Subscription auth via the
    local ``claude`` CLI; no API key required.
    """

    # Admission cap. Conservative default — the SDK spawns a `claude`
    # subprocess per call and subscription tiers rate-limit concurrency
    # sooner than metered HTTP APIs. Override via the policy
    # (providers.claude_agent.concurrent_request_cap) or
    # SADDLE_AGENT_CONCURRENT_CAP.
    concurrent_request_cap: int = 4

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        cap = self._cfg.get("concurrent_request_cap")
        if cap is None:
            env_cap = os.environ.get("SADDLE_AGENT_CONCURRENT_CAP")
            cap = env_cap if env_cap and env_cap.strip() else None
        if cap is not None:
            try:
                self.concurrent_request_cap = max(1, int(cap))
            except (TypeError, ValueError):
                pass
        # Resolve model + effort: explicit cfg > env var > saddle default
        # (opus 4.8 / xhigh). Never empty — saddle pins the model rather than
        # deferring to the CLI's configured default.
        self._model: str = (
            self._cfg.get("model") or _env_model() or _DEFAULT_MODEL
        ).strip()
        self._effort: str = (
            self._cfg.get("effort") or _env_effort() or _DEFAULT_EFFORT
        ).strip()

    def _options(
        self, system: str, *, stderr_sink: list[str] | None = None
    ) -> "ClaudeAgentOptions":
        from claude_agent_sdk import ClaudeAgentOptions

        kwargs: dict = {
            "max_turns": 1,
            "allowed_tools": [],          # no tools — pure generation
            "setting_sources": [],        # hermetic: no CLAUDE.md / settings
            "permission_mode": "default",
        }
        if system:
            kwargs["system_prompt"] = system
        if self._model:
            kwargs["model"] = self._model
        if self._effort:
            kwargs["effort"] = self._effort
        if stderr_sink is not None:
            # Capture the `claude` CLI subprocess stderr. The SDK otherwise
            # drops it on a crash, leaving only the opaque "Check stderr
            # output for details" — see ClaudeAgentUnavailable.
            kwargs["stderr"] = stderr_sink.append
        return ClaudeAgentOptions(**kwargs)

    async def __call__(
        self,
        system: str,
        prompt: str,
        *,
        json_mode: bool = False,
        label: str = "",
    ) -> str:
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        if json_mode:
            prompt = prompt + (
                "\n\n(Respond with ONLY the JSON object that satisfies the "
                "schema. No prose, no markdown fences, no commentary.)"
            )

        stderr_lines: list[str] = []
        options = self._options(system, stderr_sink=stderr_lines)
        slot_label = f"claude_agent/{label}" if label else "claude_agent"
        parts: list[str] = []
        try:
            async with PoolSlot(get_pool(), slot_label):
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                parts.append(block.text)
        except ClaudeAgentUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — re-typed as ClaudeAgentUnavailable
            # A `claude` CLI subprocess crash. Re-raise as the typed error
            # carrying the captured stderr so the real cause is visible AND the
            # fallback chain routes past it to the next provider (see
            # ClaudeAgentUnavailable) rather than aborting the run.
            detail = "\n".join(s.rstrip() for s in stderr_lines if s and s.strip())
            msg = f"claude CLI failed for {label or 'request'}: {exc}"
            if detail:
                msg += f"\n--- claude stderr ---\n{detail}"
            raise ClaudeAgentUnavailable(msg) from exc

        joined = "".join(parts)
        # JSON mode: scan for the committed JSON span. Prose mode: only strip
        # <think> blocks — keep the body verbatim so an embedded code fence in a
        # design write-up isn't truncated to its first fenced span.
        text = (extract_json_text(joined) if json_mode else strip_think(joined)).strip()
        if not text:
            raise RuntimeError(
                f"claude_agent empty response for {label or 'request'}"
            )
        return text


class ChatSession:
    """Interactive, multi-turn, tool-capable Claude — saddle's chat agent.

    Wraps :class:`claude_agent_sdk.ClaudeSDKClient`: keeps a live session
    across turns and runs with the full Claude Code toolset so it behaves
    like Claude Code itself, scoped to ``cwd``. Stream a turn's text with
    :meth:`ask`.

        async with ChatSession(cwd="/home/aeli/projects/saddle") as chat:
            async for chunk in chat.ask("what's in this repo?"):
                print(chunk, end="")
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        permission_mode: str = "bypassPermissions",
        setting_sources: list[str] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = (model or _env_model() or _DEFAULT_MODEL).strip()
        self._effort = (effort or _env_effort() or _DEFAULT_EFFORT).strip()
        self._cwd = cwd
        self._permission_mode = permission_mode
        # Behave like Claude Code: honor the user's + project's settings,
        # CLAUDE.md, and permissions. saddle has no `.mcp.json`, and its
        # settings.json disables `backpack`, so no sibling-project MCP
        # server leaks into this session.
        self._setting_sources = (
            setting_sources if setting_sources is not None
            else ["user", "project", "local"]
        )
        self._client = None

    def _build_options(self) -> "ClaudeAgentOptions":
        from claude_agent_sdk import ClaudeAgentOptions

        kwargs: dict = {
            "permission_mode": self._permission_mode,
            "setting_sources": self._setting_sources,
        }
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        if self._model:
            kwargs["model"] = self._model
        if self._effort:
            kwargs["effort"] = self._effort
        if self._cwd:
            kwargs["cwd"] = self._cwd
        return ClaudeAgentOptions(**kwargs)

    async def connect(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        self._client = ClaudeSDKClient(options=self._build_options())
        await self._client.connect()

    async def ask(self, prompt: str) -> AsyncIterator[str]:
        """Send one turn; yield assistant text blocks as they arrive.

        Tool use happens transparently inside the turn (the SDK runs tools
        and continues); only the assistant's text is yielded. The async
        iterator completes when the turn's ``ResultMessage`` arrives.
        """
        from claude_agent_sdk import AssistantMessage, TextBlock

        if self._client is None:
            await self.connect()
        assert self._client is not None

        await self._client.query(prompt)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text

    async def close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def __aenter__(self) -> "ChatSession":
        await self.connect()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()
