"""saddle — a small LLM harness.

Two surfaces:
  - a project-specific provider pool (lifted from a sibling project, but
    owning saddle's own priority order, per-provider caps, and pool sizing
    via ``config/llm_policy.json``); and
  - an agentic Claude chat service backed by the Claude Agent SDK.

Secrets are shared, not duplicated: API keys come from a sibling project's
key file via the policy's ``keys_from`` pointer.
"""

__all__: list[str] = []
