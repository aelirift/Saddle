"""saddle chat — an interactive, agentic Claude REPL.

Backed by the Claude Agent SDK (the local ``claude`` CLI, subscription
auth). Multi-turn and tool-capable — the same kind of agent as Claude Code
itself, scoped to the saddle project.

Run::

    python -m saddle.chat        # or the `saddle` console script
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from saddle.llm.claude_agent import ChatSession

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXIT_WORDS = {"/exit", "/quit", "/q"}


async def _run() -> int:
    print("saddle chat — agentic Claude. Ctrl-D or /exit to quit.\n")
    async with ChatSession(cwd=str(_REPO_ROOT)) as chat:
        while True:
            try:
                line = await asyncio.to_thread(input, "you ▸ ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            prompt = line.strip()
            if not prompt:
                continue
            if prompt in _EXIT_WORDS:
                break
            print("claude ▸ ", end="", flush=True)
            try:
                async for chunk in chat.ask(prompt):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            except Exception as exc:  # noqa: BLE001 — surface, keep REPL alive
                print(f"\n[error] {exc}")
                continue
            print("\n")
    print("bye.")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
