from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMCaller(Protocol):
    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False, label: str = "") -> str: ...
