# saddle

A small **LLM harness** — a project-specific provider pool plus an agentic
Claude chat service. It rides on top of a sibling project's shared API keys
without duplicating any secret.

## What's here

- **`saddle.llm`** — a memory-paced, intent-routed LLM provider pool lifted
  from a sibling project. Saddle owns its own **priority order**, **per-provider
  caps**, and **pool sizing** via `config/llm_policy.json` — decoupled from the
  sibling's hardcoded values.
- **`saddle.llm.claude_agent`** — the Claude Agent SDK integration:
  - `ClaudeAgentCaller`: the one-shot pool provider (single turn, no tools).
  - `ChatSession`: the interactive, multi-turn, tool-capable chat agent.
- **`saddle.chat`** — an interactive REPL (`python -m saddle.chat`) — an
  agentic Claude like Claude Code itself, scoped to this project.

## Secrets vs. policy

| | Lives in | Owned by |
|---|---|---|
| **Secrets** (API keys) | sibling `config/llm_config.json` | shared, referenced via `keys_from` |
| **Policy** (priority, caps, pool) | `config/llm_policy.json` | saddle |

`claude_agent` is **keyless** — it authenticates through the local `claude`
CLI's existing subscription login, so no API key is needed for the lead
provider.

## Config — `config/llm_policy.json`

```json
{
  "keys_from": "/home/aeli/projects/rayxiv4/config/llm_config.json",
  "priority": ["claude_agent", "minimax"],
  "providers": {
    "claude_agent": {"concurrent_request_cap": 4},
    "minimax": {"concurrent_request_cap": 36}
  },
  "pool": {"max_procs": 24, "mem_ceiling_pct": 80, "mem_resume_pct": 70}
}
```

Env overrides: `SADDLE_LLM_POLICY`, `SADDLE_LLM_CONFIG`, `SADDLE_AGENT_MODEL`,
`SADDLE_AGENT_CONCURRENT_CAP`, `SADDLE_MAX_PROCS`.

## Run

```sh
pip install -e .
saddle              # or:  python -m saddle.chat
```

## Notes

- The `backpack` MCP server is disabled for this project
  (`.claude/settings.json` → `disabledMcpjsonServers`).
- Pool usage:

  ```python
  from saddle.llm import get_pool
  from saddle.llm.llm_pool import get_llm_pool
  ```
