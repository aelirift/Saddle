"""PreToolUse doctrine hook — the enforced passthrough in front of a tool call.

This is the backpack-style interception the README gestures at, realized as a
Claude Code ``PreToolUse`` hook: before the harness runs an Edit / Write /
MultiEdit / NotebookEdit / Bash, the tool call is piped here as JSON, mapped to
the doctrine :class:`~saddle.doctrine.Action`\\ (s) it represents, and evaluated
by the SAME gate the CLI uses. On BLOCK the hook DENIES the tool — so a drifting
model is stopped *before* the mutation lands, with no LLM in the enforcement
loop and no voluntary call to skip.

Why a hook (and not a tool the model chooses to call): saddle's lesson is that
a rule pasted into a prompt decays. A PreToolUse hook runs in saddle's control
flow, not the model's, so it cannot be reasoned away. And because saddle's
converge coder is a :class:`~saddle.llm.claude_agent.ChatSession` whose
``setting_sources`` include ``"project"``, the SDK loads this hook from the
project's ``.claude/settings.json`` automatically — the in-product coder is
gated by the very same mechanism that gates any agent operating in the project.

Wire it (``.claude/settings.json``)::

    "hooks": {"PreToolUse": [{
      "matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash",
      "hooks": [{"type": "command",
                 "command": "PYTHONPATH=${CLAUDE_PROJECT_DIR}/src python3 -m saddle.doctrine_hook"}]}]}

The focus project (the scope-fence's "inside") is resolved via
:func:`saddle.context.code_root` — ``$SADDLE_CODE_ROOT`` first, then the git
root, then cwd — so pinning ``SADDLE_CODE_ROOT`` makes saddle the focus no
matter which directory the agent was launched from.

Protocol: emits the structured PreToolUse decision JSON on stdout and the human
reason on stderr; exit 0 always (the decision rides in the JSON, not the exit
code). Fails OPEN on an unreadable payload — a hook crash must never wedge the
agent — but says so loudly on stderr.
"""

from __future__ import annotations

import json
import sys


def _deny_doc(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0  # nothing to gate
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Can't read the call -> fail OPEN (don't wedge the agent), but be loud.
        print("doctrine_hook: unparseable hook payload; allowing", file=sys.stderr)
        return 0

    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}

    # Import lazily so a hook-payload read failure above never needs saddle's
    # full import graph, and so import errors surface as fail-open + a log line.
    try:
        from saddle.context import code_root
        from saddle.doctrine import gate_tool_call

        root = str(code_root())
        verdict = gate_tool_call(tool_name, tool_input, project_root=root)
    except Exception as exc:  # noqa: BLE001 — gate failure must not wedge the agent
        print(f"doctrine_hook: gate error ({exc!r}); allowing", file=sys.stderr)
        return 0

    if verdict.allowed:
        return 0

    # The scope-fence is the one rule with a legitimate cross-project override.
    # actions_from_tool attaches no evidence, so the tool path cannot say "this
    # is cross-project" inline -- instead we consult persisted authorization
    # grants. Any other rule (no-unwired-delete, disposition-coherent) is a
    # non-scope concern and is never overridden here.
    if getattr(verdict, "rule_id", None) == "stay-in-project-focus":
        try:
            from saddle.crossproject import authorize_tool

            note = authorize_tool(tool_name, tool_input, focus=root)
        except Exception as exc:  # noqa: BLE001 -- grant-check failure keeps the block
            note = None
            print(f"doctrine_hook: grant check error ({exc!r}); keeping block",
                  file=sys.stderr)
        if note is not None:
            print(f"[doctrine] cross-project ALLOW: {note}", file=sys.stderr)
            return 0

    print(json.dumps(_deny_doc(verdict.render())))
    print(f"[doctrine] BLOCKED {tool_name}: {verdict.render()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
