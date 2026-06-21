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

from .json_tools import extract_json_text, strip_llm_wrappers
from .pool import PoolSlot, get_pool

if TYPE_CHECKING:  # pragma: no cover
    from claude_agent_sdk import ClaudeAgentOptions

_log = logging.getLogger("saddle.llm.claude_agent")


def _env_model() -> str:
    return (os.environ.get("SADDLE_AGENT_MODEL", "") or "").strip()


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
        # Empty string == "let the SDK/CLI pick its configured default".
        self._model: str = (self._cfg.get("model") or _env_model()).strip()

    def _options(self, system: str) -> "ClaudeAgentOptions":
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

        options = self._options(system)
        slot_label = f"claude_agent/{label}" if label else "claude_agent"
        parts: list[str] = []
        async with PoolSlot(get_pool(), slot_label):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)

        text = strip_llm_wrappers("".join(parts))
        if json_mode:
            text = extract_json_text(text)
        text = text.strip()
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
        cwd: str | None = None,
        permission_mode: str = "bypassPermissions",
        setting_sources: list[str] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = (model or _env_model()).strip()
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
