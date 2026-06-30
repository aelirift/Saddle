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

Stage 3 — the pre-code design review — rides this same hook (the guard is
ENFORCEMENT; Stage 3 is OBSERVATION). On the FIRST code-mutating edit of a turn,
once the deterministic guard has ALLOWED it, saddle reads the agent's pre-edit
reasoning from the transcript (:func:`saddle.transcript.latest_turn`) and audits
it via :func:`saddle.design.audit_proposal` — the EXACT ``_SYS_AUDIT`` the design
pipeline runs on its own output: did the approach reach the root cause, honor the
binding directives, avoid a band-aid, cover the whole goal? Jumping straight to an
edit with no recorded approach is itself the finding. A flaw is a LOUD ALERT
(durable bubble + agent ``additionalContext``); it NEVER blocks the edit (blocking
stays the guard's job). It fires once per turn, anchored on the user prompt's uuid.

Knobs (env, all optional):

* ``SADDLE_HOOK_DESIGN``         "0" disables the Stage-3 design review; default on.
* ``SADDLE_HOOK_DESIGN_TIMEOUT`` seconds ceiling on the audit; default 60.

Protocol: on a BLOCK (a cross-project DELETE, or an in-focus code delete with no
disposition), emits the structured PreToolUse deny decision JSON on stdout and
the reason on stderr; on a cross-project EDIT / WRITE the scope-fence ALLOWS with
a WARNING, emits NO stdout decision but surfaces it on stderr + a durable bubble
(a USER-granted pair -> NOTICE, an ungranted wander -> ALERT — it is allowed yet
never silent); on an ALLOW with a Stage-3 finding, emits a PreToolUse
``additionalContext`` blob (NO ``permissionDecision`` — the review observes, it
never auto-approves); otherwise nothing. Exit 0 always (the decision rides in the
JSON, not the exit code). Two failure regimes, deliberately
different: a HOOK-level infra error (unreadable payload, ctx/transcript read)
fails OPEN — a hook crash must never wedge the agent — but says so loudly on
stderr; a Stage-3 AUDIT that cannot run fails LOUD via
:func:`saddle.supervisor.run_stage` (a classified ALERT bubble), never a swallowed
pass.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context
    from saddle.supervisor import StageOutcome


def _deny_doc(reason: str, *, system_msg: str = "") -> dict:
    doc: dict = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    # The reason rides in permissionDecisionReason (agent-facing); systemMessage is
    # the ONE field Claude Code renders ON SCREEN to the watching human, so a BLOCK
    # — saddle's loudest act — is never invisible under a non-TTY / SDK host.
    if system_msg:
        doc["systemMessage"] = system_msg
    return doc


def _bubble(level: str, stage: str, text: str, session: str) -> None:
    """Best-effort durable copy of a gate decision to the client-agnostic outbox
    (:mod:`saddle.bubble`), so an AFK / non-TTY human SEES a block — or a
    cross-project allow — that stderr alone would hide under an SDK host. The
    bubble's (tenant, project) is the agent's ambient context (where it is
    operating), resolved exactly like the intake hook. Never affects the
    deny/allow decision: a failure here is logged and swallowed."""
    try:
        from saddle.bubble import emit_bubble
        from saddle.context import resolve

        emit_bubble(resolve(), text, level=level, stage=stage, session=session)
    except Exception as exc:  # noqa: BLE001 — the outbox is a convenience, not a gate
        print(f"doctrine_hook: bubble emit failed ({exc!r})", file=sys.stderr)


def _surface_allow_warn(
    tool_name: str, tool_input, root: str, session: str, verdict
) -> None:
    """Surface a cross-project EDIT / WRITE the scope-fence ALLOWED with a WARNING.

    Option A: a cross-project edit/write no longer BLOCKS (only a cross-project
    DELETE does), but it must never be SILENT — a wander into the wrong project
    has to be SEEN. This is the warn path's concrete user-visible surface:

    * a covering USER-issued grant -> the move is a sanctioned pair: a stderr
      ``cross-project ALLOW`` line + a durable NOTICE bubble.
    * no covering grant -> a genuine out-of-focus wander: a loud stderr ``WARN``
      line + a durable ALERT bubble naming the off-focus target.

    The alert-vs-notice demotion keys on :func:`saddle.crossproject.authorize_tool`,
    which honors USER-issued grants ONLY — so the constrained session cannot
    self-issue a grant to quiet its own wander (the gate review's #1/#3). Emits
    NOTHING on stdout: the caller still runs Stage 3, which owns the stdout
    PreToolUse JSON, and a second object here would corrupt it."""
    try:
        from saddle.crossproject import authorize_tool

        note = authorize_tool(tool_name, tool_input, focus=root)
    except Exception as exc:  # noqa: BLE001 — surfacing must never wedge the edit
        note = None
        print(f"doctrine_hook: grant check error ({exc!r})", file=sys.stderr)
    if note is not None:
        print(f"[doctrine] cross-project ALLOW: {note}", file=sys.stderr)
        _bubble("notice", "guard", f"cross-project ALLOW: {note}", session)
        return
    reason = verdict.render()
    print(f"[doctrine] WARN out-of-focus {tool_name}: {reason}", file=sys.stderr)
    _bubble(
        "alert", "guard",
        f"OUT-OF-FOCUS — allowed with WARNING (not blocked): {tool_name}\n{reason}",
        session,
    )


# -- Stage 3 (design) — the anti-band-aid pre-code review --------------------

# The unambiguous code-WRITES that trip the pre-code design review. A Bash
# code-write (`cat > f`) is rarer and the deterministic guard already fences it;
# keying Stage 3 on the edit family alone avoids an LLM audit on every `Bash`
# (builds, greps, tests) while still catching the common case — the agent edits a
# file. Deliberately narrow, not an oversight.
_CODE_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


def _design_marker_path(session: str):
    """Where this session's last design-gate turn-anchor is parked — alongside the
    saddle DB, in its own ``design_gate`` namespace so it never collides with the
    intake hook's replay cursor."""
    from saddle.store import default_db_path

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session) or "default"
    return default_db_path().parent / "design_gate" / f"{safe}.json"


def _design_already_fired(session: str, anchor: str) -> bool:
    """Has the design gate already fired for THIS turn (this user-prompt anchor)?
    The per-session marker makes the gate fire on the FIRST code edit of a turn and
    stay silent on the rest. An absent / unreadable marker ⇒ not yet fired."""
    if not anchor:
        return False
    try:
        raw = _design_marker_path(session).read_text(encoding="utf-8")
        return json.loads(raw).get("anchor") == anchor
    except (OSError, ValueError):
        return False


def _mark_design_fired(session: str, anchor: str) -> None:
    """Record that the design gate fired for ``anchor`` (atomic temp + replace,
    like the intake cursor) so the rest of the turn's edits skip it. Best-effort:
    an IO failure is logged, never raised — at worst it re-audits, never wedges."""
    if not anchor:
        return
    p = _design_marker_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"anchor": anchor}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        print(f"doctrine_hook: design marker write failed ({exc!r})", file=sys.stderr)


def _emit_pretool_context(
    ctx: "Context", sections: list[str], *, system_msg: str = ""
) -> None:
    """Surface Stage 3's finding on every channel it has:

    * ``additionalContext`` on stdout — the agent's mid-turn read, so it can
      self-correct before its NEXT edit;
    * ``systemMessage`` on the same stdout JSON — the ONE channel Claude Code
      renders ON SCREEN to the watching HUMAN, so a caught band-aid / missing-design
      is visible even under a non-TTY / SDK host where stderr is swallowed (the
      "I don't see any outputs" gap);
    * a stderr copy for an interactive TTY.

    Emits NO ``permissionDecision`` — the design review OBSERVES, it never
    auto-approves or blocks the edit (blocking is the deterministic guard's job).
    The durable per-stage bubble already went out via :func:`run_stage`. Empty
    sections AND an empty ``system_msg`` render to nothing, so a clean review is
    silent on every channel."""
    from saddle.supervisor import render_sections

    body = render_sections(ctx, sections)
    if not body and not system_msg:
        return
    out: dict = {}
    if body:
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "additionalContext": body,
        }
    if system_msg:
        out["systemMessage"] = system_msg
    print(json.dumps(out))
    if body:
        print(body, file=sys.stderr)


def _design_outcome(ctx: "Context", goal: str, approach: str) -> "StageOutcome | None":
    """Stage 3 body — audit the agent's pre-edit approach, fail LOUD.

    No recorded approach ⇒ the agent jumped straight to code, which is itself the
    finding (a deterministic ALERT, no LLM call). Otherwise the approach is audited
    under a deadline via :func:`saddle.supervisor.run_bounded`, which PROPAGATES a
    timeout / provider outage / contract gap to :func:`run_stage` to classify and
    bubble — never a swallowed 'looked fine'. A clean audit returns ``None``
    (silent); issues return an ALERT naming each one."""
    from saddle.models import BUBBLE_ALERT
    from saddle.supervisor import StageOutcome

    approach = (approach or "").strip()
    if not approach:
        return StageOutcome(
            sections=[
                "⚠ NO RECORDED DESIGN — the agent went straight to a code edit "
                "without first discussing the problem or its approach. Discuss the "
                "approach before coding so it can be reviewed for drift."
            ],
            level=BUBBLE_ALERT,
        )

    from saddle.design import audit_proposal
    from saddle.supervisor import run_bounded

    try:
        timeout = float(os.environ.get("SADDLE_HOOK_DESIGN_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    verdict = run_bounded(
        audit_proposal(goal, approach, ctx),
        seconds=timeout,
        what="the design review (root-cause / band-aid / coverage)",
    )
    if not verdict.has_issues:
        return None  # the approach audited clean -> silent
    body = "\n".join(f"  • {i}" for i in verdict.issues)
    return StageOutcome(
        sections=[
            "this turn's approach has design issues that should be resolved "
            f"before code is written:\n{body}"
        ],
        level=BUBBLE_ALERT,
        # The caught flaws ride on the bubble (like Stage 4's code drift) so the
        # turn-end lesson harvest (Stage 5) can learn from them without re-parsing
        # the rendered prose.
        meta={"issues": list(verdict.issues)},
    )


def _run_design_stage(tool_name: str, transcript_path: str, session: str) -> None:
    """Run Stage 3 on the FIRST code-mutating edit of a turn (after the guard
    allowed it). Reads the agent's pre-edit reasoning from the transcript and
    audits it, bubbling a LOUD ALERT for a band-aid / misread cause / no-design and
    injecting the finding into the agent's context so it can self-correct.

    Infra (no transcript, anchor IO, ctx resolution) fails OPEN — observation must
    never wedge an edit — but the AUDIT itself fails LOUD via :func:`run_stage`."""
    if os.environ.get("SADDLE_HOOK_DESIGN", "1") == "0":
        return
    if tool_name not in _CODE_EDIT_TOOLS or not transcript_path:
        return
    try:
        from saddle.context import resolve
        from saddle.transcript import latest_turn

        ctx = resolve(os.environ.get("SADDLE_TENANT"), os.environ.get("SADDLE_PROJECT"))
        turn = latest_turn(transcript_path)
    except Exception as exc:  # noqa: BLE001 — a setup failure must not wedge the edit
        print(f"doctrine_hook: design-stage setup error ({exc!r}); skipping",
              file=sys.stderr)
        return

    if not turn.anchor or _design_already_fired(session, turn.anchor):
        return  # no turn to anchor on, or already reviewed this turn

    from saddle.models import STAGE_DESIGN
    from saddle.supervisor import run_stage, system_message

    result = run_stage(
        ctx, STAGE_DESIGN,
        lambda: _design_outcome(ctx, turn.goal, turn.approach),
        session=session,
        what="the agent's approach before its first code edit",
    )
    # Once per turn regardless of outcome: a clean run, a found drift, AND a
    # classified failure each count as "reviewed" — re-auditing every later edit of
    # the same turn would spam. The failure already bubbled its own loud ALERT.
    _mark_design_fired(session, turn.anchor)
    if result.sections:
        # The agent reads additionalContext; the human reads systemMessage — a
        # caught band-aid / no-design (or a could-not-run ALERT) is heralded on
        # screen, not just injected into the model's context.
        _emit_pretool_context(
            ctx, result.sections, system_msg=system_message(ctx, [result])
        )


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
    session = str(payload.get("session_id") or "")
    transcript_path = str(payload.get("transcript_path") or "")

    # Import lazily so a hook-payload read failure above never needs saddle's
    # full import graph, and so import errors surface as fail-open + a log line.
    try:
        from saddle.context import code_root
        from saddle.doctrine import SCOPE_FENCE_RULE_IDS, gate_tool_call

        root = str(code_root())
        verdict = gate_tool_call(tool_name, tool_input, project_root=root)
    except Exception as exc:  # noqa: BLE001 — gate failure must not wedge the agent
        print(f"doctrine_hook: gate error ({exc!r}); allowing", file=sys.stderr)
        return 0

    if verdict.allowed:
        # A cross-project EDIT / WRITE is ALLOWED with a WARNING (Option A — only a
        # cross-project DELETE still hard-blocks), but it must never be silent:
        # surface it (USER-grant -> notice, ungranted wander -> alert) BEFORE the
        # observation companion runs, so a missed warn can never be a false negative.
        if getattr(verdict, "severity", "") == "warn":
            _surface_allow_warn(tool_name, tool_input, root, session, verdict)
        # The deterministic guard passed; now the OBSERVATION companion — Stage 3,
        # the anti-band-aid design review — runs on the first code edit of the turn
        # (a no-op for a non-edit tool or a later edit). It surfaces (bubble + agent
        # context) and never blocks; any stdout it emits is additionalContext, not a
        # permission decision, so the normal approval flow is untouched.
        _run_design_stage(tool_name, transcript_path, session)
        return 0

    # The scope-fence rules are the ones with a legitimate cross-project override.
    # actions_from_tool attaches no evidence, so the tool path cannot say "this is
    # cross-project" inline -- instead we consult persisted USER-issued grants. The
    # only block reachable here from a scope rule is a cross-project DELETE
    # (no-cross-project-delete); a covering user-granted pair unblocks it. Any
    # non-scope rule (no-unwired-delete, disposition-coherent) is a code-safety
    # invariant and is NEVER overridden by a grant.
    if getattr(verdict, "rule_id", None) in SCOPE_FENCE_RULE_IDS:
        try:
            from saddle.crossproject import authorize_tool

            note = authorize_tool(tool_name, tool_input, focus=root)
        except Exception as exc:  # noqa: BLE001 -- grant-check failure keeps the block
            note = None
            print(f"doctrine_hook: grant check error ({exc!r}); keeping block",
                  file=sys.stderr)
        if note is not None:
            print(f"[doctrine] cross-project ALLOW: {note}", file=sys.stderr)
            _bubble("notice", "guard", f"cross-project ALLOW: {note}", session)
            # A grant changes WHERE the agent may operate, not WHETHER the work
            # deserves review. This override path is reached for a cross-project
            # DELETE (Bash rm of a granted sibling); _run_design_stage is the same
            # first-edit audit the allow path runs and is a no-op for a non-edit
            # tool, so calling it keeps the two paths symmetric without re-auditing.
            _run_design_stage(tool_name, transcript_path, session)
            return 0

    reason = verdict.render()
    print(json.dumps(_deny_doc(reason, system_msg=f"⛔ saddle BLOCKED {tool_name}: {reason}")))
    print(f"[doctrine] BLOCKED {tool_name}: {reason}", file=sys.stderr)
    _bubble("alert", "guard", f"BLOCKED {tool_name}: {reason}", session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
