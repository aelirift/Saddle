"""UserPromptSubmit bubble-up hook — saddle's per-prompt voice, agent-independent.

This is the OBSERVATION counterpart to :mod:`saddle.doctrine_hook` (which is
ENFORCEMENT). Where the doctrine hook is a ``PreToolUse`` gate that DENIES a
drifting mutation, this is a ``UserPromptSubmit`` hook that *speaks*: on every
user prompt, before the agent starts, saddle itemizes the request, catches drift
the agent committed since the last turn, and surfaces the standing commitment —
bubbling the whole lot back as context the agent (and the human) can see.

Why a hook, and why it must do the WHOLE job itself: saddle is used across many
projects and tenants by many different agents. It cannot depend on the agent
choosing to call saddle's MCP tools — a drifting or compacting model is exactly
the one that won't. So this hook runs in saddle's control flow, reads the agent's
own transcript, and produces its verdict with no LLM-of-the-agent in the loop and
no voluntary call to skip. The agent is never asked to do saddle's work.

Two latency classes, handled deliberately:

* **Fast / deterministic — always.** Replay the transcript since a persisted
  per-session cursor (:func:`saddle.transcript.replay`): record any fork the
  agent offered, bind prior user picks, and surface any action that contradicts
  the live commitment (the "picked a) then did b)" drift). Then bind THIS prompt
  if it's a pick, and read back the standing commitment. No LLM, milliseconds —
  so drift is surfaced immediately and never blocks the turn.
* **Slow / LLM — substantive prompts only, bounded, fail-open.** Run Layer 1
  :func:`saddle.intake.decompose` (itemize -> coverage-audit -> focus-scope) and
  render the typed know/do list plus any out-of-focus scope warning. A bare
  continuation ("go", "proceed", "a then b") has no braided asks to untangle, so
  it skips the LLM and gets only the fast surface — the iteration loop stays
  instant instead of paying ~20s per "proceed". Itemization is bounded by a
  timeout and fails OPEN: a slow or failed call degrades to the fast surface, it
  never wedges the prompt.

Isolation: the (tenant, project) this hook speaks for is fixed by
``SADDLE_TENANT`` / ``SADDLE_PROJECT`` and the code root by ``SADDLE_CODE_ROOT``,
exactly as the CLI and MCP server resolve it — that pair is the hard fence, so a
prompt in one project can never read or bind another's ledger.

Wire it (``.claude/settings.json``)::

    "hooks": {"UserPromptSubmit": [{
      "hooks": [{"type": "command",
                 "command": "PYTHONPATH=${CLAUDE_PROJECT_DIR}/src python3 -m saddle.intake_hook"}]}]}

Knobs (env, all optional):

* ``SADDLE_HOOK_ITEMIZE``        "0" disables the LLM itemize (fast-only mode); default on.
* ``SADDLE_HOOK_INTAKE_TIMEOUT`` seconds ceiling on decompose; default 150.
* ``SADDLE_HOOK_MAX_AUDITS``     coverage-audit passes in the hook path; default 1.
* ``SADDLE_HOOK_MIN_WORDS``      below this word count a prompt is a trivial
                                 continuation (skip itemize); default 4.

Protocol: emits the ``UserPromptSubmit`` ``additionalContext`` decision JSON on
stdout (injected into the agent's context) and a human-readable copy on stderr
(shown on screen); **exit 0 always, never exit 2** — this hook observes, it does
not block (blocking is the doctrine hook's job). Fails OPEN on any error: a hook
crash must never wedge the agent, but it says so loudly on stderr.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context
    from saddle.dialog import IntentTracker
    from saddle.models import Binding, DriftVerdict

# Cap how many surfaced drift/confirm lines the bubble-up shows. A first wire-up
# against a long existing transcript can surface a backlog; the ledger keeps all
# of them, the bubble-up shows the most recent so it stays readable.
_MAX_SURFACED = 6


# -- emit --------------------------------------------------------------------

def _emit(ctx_key: str, sections: list[str]) -> None:
    """Print the bubble-up: ``additionalContext`` JSON on stdout (the agent's
    context channel) and a human-readable copy on stderr (the on-screen view)."""
    body = f"━━ saddle [{ctx_key}] ━━\n" + "\n\n".join(sections)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": body,
        }
    }))
    print(body, file=sys.stderr)


# -- per-session replay cursor (idempotent transcript tailing) ---------------

def _cursor_path(session: str) -> Path:
    """Where this session's last-seen transcript uuid is parked — alongside the
    saddle DB so it shares the install's data dir and isolation."""
    from saddle.store import default_db_path

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session) or "default"
    return default_db_path().parent / "cursors" / f"{safe}.json"


def _read_cursor(session: str) -> str | None:
    try:
        raw = _cursor_path(session).read_text(encoding="utf-8")
        val = json.loads(raw).get("last_uuid")
        return str(val) if val else None
    except (OSError, ValueError):
        return None  # absent or unreadable -> no cursor (replay from the top)


def _write_cursor(session: str, last_uuid: str) -> None:
    """Atomically park the cursor (temp + os.replace) so a crash mid-write can
    never leave a half-written file that re-floods the next replay."""
    if not last_uuid:
        return
    p = _cursor_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"last_uuid": last_uuid}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        print(f"intake_hook: cursor write failed ({exc!r})", file=sys.stderr)


# -- rendering ---------------------------------------------------------------

def _render_verdict(v: "DriftVerdict") -> str:
    if v.is_drift:
        bound = v.bound_choice or v.bound_label or "?"
        acted = v.action_choice or v.action_label or "?"
        return f"⚠ DRIFT — you committed {bound}, the agent acted on {acted}: {v.reason}"
    # a surfaced must-confirm UNKNOWN (ambiguous / wrong-fork / uncited bare label)
    return f"⚠ MUST CONFIRM — {v.reason}"


def _render_binding(b: "Binding") -> str:
    choice = b.choice_id or b.label or "?"
    return (
        f"standing commitment: {choice} — {b.user_text!r} "
        f"(via {b.method}, confidence {b.confidence:.2f})"
    )


# -- fast section: replay drift + catch the ledger up ------------------------

def _replay_section(
    ctx: "Context", tracker: "IntentTracker", transcript_path: str, session: str
) -> list[str]:
    """Deterministic, fast. Tail the transcript since our cursor into the tracker,
    surfacing any drift / must-confirm the agent committed since the last prompt.
    Advances the cursor so the next prompt only sees what's new."""
    from saddle.transcript import replay

    if not transcript_path:
        return []
    res = replay(ctx, transcript_path, tracker=tracker, after_uuid=_read_cursor(session))
    if res.last_uuid:
        _write_cursor(session, res.last_uuid)
    surfaced = res.surfaced
    if not surfaced:
        return []
    shown = surfaced[-_MAX_SURFACED:]
    lines = [_render_verdict(v) for v in shown]
    if len(surfaced) > len(shown):
        lines.insert(0, f"(saddle surfaced {len(surfaced)} items since last turn; newest {len(shown)} shown)")
    return ["\n".join(lines)]


# -- slow section: the itemized know/do list ---------------------------------

def _is_trivial(prompt: str) -> bool:
    """A bare continuation/pick ('go', 'proceed', 'a then b') with no braided
    asks for the itemizer to untangle. Below the word threshold a prompt cannot
    carry multiple distinct asks, so skipping the LLM pass is correct, not lazy."""
    try:
        min_words = int(os.environ.get("SADDLE_HOOK_MIN_WORDS", "4") or 4)
    except ValueError:
        min_words = 4
    return len(prompt.split()) < min_words


async def _itemize(ctx: "Context", prompt: str) -> str:
    from saddle.intake import decompose, format_intake

    try:
        max_audits = int(os.environ.get("SADDLE_HOOK_MAX_AUDITS", "1") or 1)
    except ValueError:
        max_audits = 1
    intake = await decompose(prompt, ctx, max_audits=max_audits)
    return format_intake(intake)


def _itemize_section(ctx: "Context", prompt: str) -> list[str]:
    """The LLM know/do list — substantive prompts only, bounded, fail-open."""
    if os.environ.get("SADDLE_HOOK_ITEMIZE", "1") == "0" or _is_trivial(prompt):
        return []
    try:
        timeout = float(os.environ.get("SADDLE_HOOK_INTAKE_TIMEOUT", "150") or 150)
    except ValueError:
        timeout = 150.0
    try:
        return [asyncio.run(asyncio.wait_for(_itemize(ctx, prompt), timeout))]
    except (Exception, asyncio.TimeoutError) as exc:  # noqa: BLE001 — fail OPEN
        # The itemize is the costly part; a slow/failed call degrades to the fast
        # surface rather than wedging the prompt. The drift/commitment sections
        # already ran, so saddle still spoke.
        return [f"(saddle: itemization unavailable this turn — {exc!r})"]


# -- entry point -------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0  # nothing submitted
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print("intake_hook: unparseable hook payload; allowing", file=sys.stderr)
        return 0

    prompt = str(payload.get("prompt") or "").strip()
    session = str(payload.get("session_id") or "")
    transcript_path = str(payload.get("transcript_path") or "")
    if not prompt:
        return 0  # nothing to itemize (e.g. a slash-command with no text)

    # Import lazily so an unreadable payload above never needs saddle's full
    # import graph, and so any import/ctx error surfaces as fail-open + a log.
    try:
        from saddle.context import resolve
        from saddle.dialog import get_tracker

        ctx = resolve(os.environ.get("SADDLE_TENANT"), os.environ.get("SADDLE_PROJECT"))
        tracker = get_tracker()
    except Exception as exc:  # noqa: BLE001 — a hook crash must not wedge the agent
        print(f"intake_hook: import/ctx error ({exc!r}); allowing", file=sys.stderr)
        return 0

    sections: list[str] = []

    # 1) fast: catch the ledger up + surface drift the agent committed since last.
    try:
        sections += _replay_section(ctx, tracker, transcript_path, session)
    except Exception as exc:  # noqa: BLE001
        print(f"intake_hook: replay error ({exc!r}); continuing", file=sys.stderr)

    # 2) fast: bind THIS prompt if it's a pick, so the commitment surface below
    #    reflects the choice the user just made (immediate, not a turn late).
    try:
        tracker.observe_user_message(ctx, prompt, session=session)
    except Exception as exc:  # noqa: BLE001
        print(f"intake_hook: observe error ({exc!r}); continuing", file=sys.stderr)

    # 3) slow: the itemized know/do list (+ scope warning), fail-open.
    sections += _itemize_section(ctx, prompt)

    # 4) fast: the standing commitment the agent must honor.
    try:
        b = tracker.active_binding(ctx, session=session)
        if b is not None:
            sections.append(_render_binding(b))
    except Exception as exc:  # noqa: BLE001
        print(f"intake_hook: commitment error ({exc!r}); continuing", file=sys.stderr)

    if sections:
        _emit(ctx.key, sections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
