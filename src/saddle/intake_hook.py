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
* **Slow / LLM — substantive prompts only, bounded, fail-LOUD.** Run Layer 1
  :func:`saddle.intake.decompose` (itemize -> coverage-audit -> focus-scope) and
  render the typed know/do list plus any out-of-focus scope warning. A bare
  continuation ("go", "proceed", "a then b") has no braided asks to untangle, so
  it skips the LLM and gets only the fast surface — the iteration loop stays
  instant instead of paying ~20s per "proceed". Itemization is the supervisory
  **intake stage** (Stage 1), driven through :func:`saddle.supervisor.run_stage`
  under a deadline: a slow or failed call no longer degrades to a swallowed
  "(unavailable)" string — that silent fail-open is exactly what
  ``knowledge_seed_fail_loud`` forbids. Instead the failure is CLASSIFIED
  (wall-clock vs provider outage vs oversized-input contract gap) and surfaced as
  a LOUD ALERT naming what saddle did NOT verify this turn, in both the durable
  bubble and the agent's ``additionalContext`` — so the human AND the agent learn
  supervision was incomplete. It still never *blocks* the turn (observation, not
  enforcement); it just refuses to pretend it ran. A SECOND bounded, fail-loud LLM
  check rides the same path — the **intent stage** (Stage 2) project/design/history
  axis (:func:`saddle.intent.history_drift`): it retrieves the project's settled
  designs + closed decisions from the DKB and flags a prompt that contradicts one,
  re-opens a closed decision, or creeps past the focus. Same discipline — a clean
  compare is silent, a divergence bubbles (ALERT for a hard contradiction, NOTICE
  for scope-creep), a failed check is a classified ALERT, never a swallowed pass —
  and a brand-new project with nothing settled skips the LLM call entirely.

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
* ``SADDLE_HOOK_INTENT``         "0" disables the LLM project/design/history check; default on.
* ``SADDLE_HOOK_INTENT_TIMEOUT`` seconds ceiling on the history check; default 60.
* ``SADDLE_HOOK_MIN_WORDS``      below this word count a prompt is a trivial
                                 continuation (skip the LLM checks); default 4.

Protocol: emits the ``UserPromptSubmit`` ``additionalContext`` decision JSON on
stdout (injected into the agent's context — the COMBINED agent channel over every
stage's sections) and a human-readable copy on stderr (shown on screen in an
interactive TTY). The durable copy to the client-agnostic bubble outbox
(:mod:`saddle.bubble`) is emitted PER-STAGE by :func:`saddle.supervisor.run_stage`
— because under an SDK / service host stderr is swallowed, so the outbox is the
only channel an AFK or non-TTY human can still read saddle's voice from, and a
per-stage bubble lets a stage failure be its OWN alert instead of buried in a
batch. **Exit 0 always, never exit 2** — this hook observes, it does not block
(blocking is the doctrine hook's job). Two failure regimes, deliberately
different: a HOOK-LEVEL infra error (unparseable payload, ctx resolution) fails
**open** — exit 0, a stderr log, no false verdict — because saddle cannot speak
for a turn it never parsed; but a STAGE that cannot run fails **loud** — a
classified ALERT bubble naming what saddle did NOT verify — because there the
turn IS understood and a swallowed check would be a silent fail-open.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context
    from saddle.dialog import IntentTracker
    from saddle.models import Binding, DriftVerdict, Fork
    from saddle.supervisor import StageOutcome

# Cap how many surfaced drift/confirm lines the bubble-up shows. A first wire-up
# against a long existing transcript can surface a backlog; the ledger keeps all
# of them, the bubble-up shows the most recent so it stays readable.
_MAX_SURFACED = 6


# -- emit --------------------------------------------------------------------

def _emit_context(ctx: "Context", sections: list[str], *, system_msg: str = "") -> None:
    """Emit saddle's voice on all THREE channels for the turn:

    * ``additionalContext`` JSON on stdout — the MODEL reads every stage's sections
      once (the combined agent channel);
    * ``systemMessage`` on the same stdout JSON — the ONE channel Claude Code
      renders to the watching HUMAN, so saddle is no longer invisible on screen
      under a non-TTY / SDK host (the "I don't see any outputs" gap);
    * a human-readable copy on stderr — the on-screen view in an interactive TTY.

    The DURABLE per-stage bubbles already went out via
    :func:`saddle.supervisor.run_stage`; this is the agent's single read + the
    human heralds. Empty sections AND an empty ``system_msg`` render to nothing, so
    a silent turn stays silent on stdout."""
    from saddle.supervisor import render_sections

    body = render_sections(ctx, sections)
    if not body and not system_msg:
        return
    out: dict = {}
    if body:
        out["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": body,
        }
    if system_msg:
        out["systemMessage"] = system_msg
    print(json.dumps(out))
    if body:
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


def _clip(text: str, limit: int = 160) -> str:
    """One-line, length-bounded view of possibly-long fork prose: collapse
    whitespace so multi-line option text stays a single readout line, and cap the
    length so a verbose proposal can't flood the agent's context (a real risk — a
    fork's prompt/option text can run to a full paragraph)."""
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _render_binding(b: "Binding", fork: "Fork | None" = None) -> str:
    """The carry-forward readout for the agent channel.

    NOT a reminder that a decision happened ("the user said a)") — that nags
    without helping. Instead it restates WHAT the chosen option actually was, as a
    directive the agent can act on: the proposal it answered, the substance of the
    committed option foregrounded as the thing to do now, the unchosen options
    demoted to a drift-guard aside, and the qualified choice-id last as a citation
    hint. The fork row is saddle's durable store, so this survives a context-window
    wipe; rendering is the only gap, and this closes it."""
    choice = b.choice_id or b.label or "?"
    label = b.label or (choice.rsplit(".", 1)[-1] if "." in choice else choice)
    lead = "↳ carry-forward context (already decided — act on it, don't re-ask):"
    if fork is None or not fork.options:
        tail = f" — {_clip(b.user_text)!r}" if b.user_text else ""
        return f"{lead} the user chose {choice}{tail}"
    chosen = next((o for o in fork.options if b.label and o.label == b.label), None)
    lines = [lead]
    if fork.prompt:
        lines.append(f"  on your proposal {_clip(fork.prompt)!r}, the user chose {label}):")
    else:
        lines.append(f"  the user chose {label}):")
    if chosen is not None:
        lines.append(f"  → {_clip(chosen.text)}   ← this is what to do now")
    else:
        lines.append(f"  → (chosen option {label} is not among the fork's options)")
    others = [o for o in fork.options if o is not chosen]
    if others:
        alt = "; ".join(f"{o.label}) {_clip(o.text, 80)}" for o in others)
        lines.append(f"  not chosen (so you don't drift to them): {alt}")
    lines.append(f"  [ref {choice} — cite it when you act]")
    return "\n".join(lines)


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


def _replay_outcome(
    ctx: "Context", tracker: "IntentTracker", transcript_path: str, session: str
) -> "StageOutcome | None":
    """Stage 2 (intent), pick-drift axis — the fast transcript replay as a
    supervisory stage. Surfaced "committed a) then acted on b)" drift is always
    correction-worthy, so a non-empty replay is an ALERT; nothing surfaced is a
    silent clean run. Wrapping it in a stage (vs the old swallowed try/except)
    means a replay that itself THROWS becomes a classified ALERT, not a lost log.

    This is only the pick-drift axis; the cross-project axis (out-of-focus scope)
    rides in the intake stage's ``format_intake`` today, and the
    project/design/history axis is the gap Stage 2 proper (item 14) adds here."""
    from saddle.models import BUBBLE_ALERT
    from saddle.supervisor import StageOutcome

    lines = _replay_section(ctx, tracker, transcript_path, session)
    return StageOutcome(sections=lines, level=BUBBLE_ALERT) if lines else None


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


# -- the turn-end DRAIN (mediator design: no more dead letters) ---------------
#
# Stop-hook findings (a prose-proposal review, code-vs-design drift, the lesson
# harvest, the voice check) are produced AFTER the agent's reply — the agent
# never saw them, and under a non-TTY host neither did the human. This drain
# reads them back at the START of the next turn and injects them into the
# agent's context, converting the turn-end channel's dead letters into
# delivered feedback. Cursor per session (atomic replace, like the replay
# cursor); origin-tagged bubbles only, so mid-turn injections the agent already
# read are never duplicated.

def _drain_cursor_path(session: str) -> Path:
    from saddle.store import default_db_path

    safe = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in session
    ) or "default"
    return default_db_path().parent / "drain" / f"{safe}.json"


def _drain_since(session: str) -> float:
    try:
        raw = _drain_cursor_path(session).read_text(encoding="utf-8")
        return float(json.loads(raw).get("ts", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0


def _set_drain_cursor(session: str, ts: float) -> None:
    p = _drain_cursor_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"ts": float(ts)}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        print(f"intake_hook: drain cursor write failed ({exc!r})", file=sys.stderr)


def _drained_findings(ctx: "Context", session: str) -> str:
    """The rendered drain section, or ``""`` when nothing unseen accumulated.
    Reads across project ledgers (a mixed-scope turn's findings live in each
    project's own ledger) but only THIS session's bubbles, only those tagged
    ``origin: turn-end``. Advances the cursor after reading — always, so a
    quiet turn still bounds the next scan."""
    import time as _time

    from saddle.bubble import recent_bubbles
    from saddle.voice import turn_end_findings_head

    since = _drain_since(session)
    now = _time.time()
    events = recent_bubbles(
        ctx, session=session, since_ts=since or None, limit=50, any_project=True
    )
    _set_drain_cursor(session, now)
    unseen = [
        e for e in reversed(events)  # oldest first for reading order
        if (e.meta or {}).get("origin") == "turn-end" and e.text.strip()
    ]
    if not unseen:
        return ""
    blocks = []
    for e in unseen[-_MAX_SURFACED:]:
        text = e.text.strip()
        if text.startswith("━━") and "\n" in text:
            text = text.split("\n", 1)[1].strip()  # drop the banner — we re-frame
        tag = f"[{e.project}] " if e.project != ctx.project else ""
        blocks.append(f"• {tag}{text}")
    return turn_end_findings_head() + "\n" + "\n".join(blocks)


async def _itemize(ctx: "Context", prompt: str, session: str = "") -> str:
    from saddle.intake import decompose, format_intake

    try:
        max_audits = int(os.environ.get("SADDLE_HOOK_MAX_AUDITS", "1") or 1)
    except ValueError:
        max_audits = 1
    intake = await decompose(prompt, ctx, max_audits=max_audits)
    _record_turn_scopes(ctx, intake, session)
    return format_intake(intake)


def _record_turn_scopes(ctx: "Context", intake, session: str) -> None:
    """Persist the turn's ACTIVE SCOPE SET (mediator design §4): the ambient
    project plus every sibling the scope router stamped on an item. The
    doctrine fence and the turn-end stages read it back, so a two-project turn
    is in scope everywhere at once. Also LEARNS the ambient root into the
    project registry (idempotent) — the registry is how the router knows
    sibling names next time. Best-effort: a failure narrows scope back to
    single-focus (stricter, never looser), and is logged, never raised."""
    try:
        import time as _time

        from saddle import registry
        from saddle.context import code_root
        from saddle.focus import record_active_scopes

        registry.register_root(ctx.tenant, code_root(), ts=_time.time())
        routed: dict[str, list[str]] = {}
        for it in intake.items:
            slug = getattr(it, "project", "")
            if slug:
                routed.setdefault(slug, []).append(it.ask)
        if session:
            record_active_scopes(
                session, ctx.tenant, [ctx.project, *sorted(routed)],
                asks=routed,
            )
    except Exception as exc:  # noqa: BLE001 — scope recording must not fail intake
        print(f"intake_hook: scope-set record error ({exc!r}); "
              "single-focus fence this turn", file=sys.stderr)


def _itemize_outcome(ctx: "Context", prompt: str, session: str = "") -> "StageOutcome | None":
    """Stage 1 (intake) — the itemized know/do list, fail LOUD.

    Returns the rendered list as a NOTICE :class:`StageOutcome`, or ``None`` when
    there is nothing to itemize (the LLM pass is disabled, or the prompt is a
    trivial continuation with no braided asks). It does **not** catch the itemize
    failure: a slow or failed ``decompose`` is the recurring drift-blind hang, so
    it is driven under a deadline by :func:`saddle.supervisor.run_bounded` and
    allowed to PROPAGATE to :func:`run_stage`, which classifies it (a wall-clock
    :class:`~saddle.supervise.DeadlineExceeded`, a provider outage, or an
    oversized-input contract gap) and bubbles a LOUD ALERT naming what saddle did
    NOT verify. That replaces the old swallow-to-"(unavailable)" fail-open — the
    silent degrade ``knowledge_seed_fail_loud`` forbids. A bounded RETRY is the
    caller's policy for an external rate-limit alone (``knowledge_seed_blind_retry``);
    every other failure surfaces its root cause here instead of being papered over."""
    if os.environ.get("SADDLE_HOOK_ITEMIZE", "1") == "0" or _is_trivial(prompt):
        return None
    from saddle.supervisor import StageOutcome, run_bounded

    try:
        timeout = float(os.environ.get("SADDLE_HOOK_INTAKE_TIMEOUT", "150") or 150)
    except ValueError:
        timeout = 150.0
    text = run_bounded(_itemize(ctx, prompt, session), seconds=timeout,
                       what="the breakdown of your message into its requests")
    return StageOutcome(sections=[text]) if text and text.strip() else None


def _history_outcome(ctx: "Context", prompt: str) -> "StageOutcome | None":
    """Stage 2 (intent), project/design/history axis — the gap the other two
    intent axes don't cover. Does THIS prompt pull against what the project has
    ALREADY settled: contradict a committed design, re-open a closed decision, or
    creep past the stated focus? (The cross-project axis rides in the intake
    stage's scope warning; the pick-drift axis is :func:`_replay_outcome`.)

    Returns the divergence section(s) at the report's level — an ALERT for a hard
    contradiction / re-opened decision, a NOTICE for scope-creep — or ``None`` when
    the check is disabled, the prompt is a trivial continuation (no new intent to
    weigh against the settled state), or nothing drifted. Like the itemize stage it
    does **not** catch a failure: the single classify call runs under a deadline via
    :func:`saddle.supervisor.run_bounded` and PROPAGATES to :func:`run_stage`, which
    classifies and bubbles a LOUD ALERT naming what saddle did NOT verify — never a
    swallowed "looked fine". A brand-new project with no settled state short-circuits
    inside :func:`saddle.intent.history_drift` with no LLM call at all."""
    if os.environ.get("SADDLE_HOOK_INTENT", "1") == "0" or _is_trivial(prompt):
        return None
    from saddle.intent import history_drift
    from saddle.supervisor import StageOutcome, run_bounded

    try:
        timeout = float(os.environ.get("SADDLE_HOOK_INTENT_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    report = run_bounded(history_drift(prompt, ctx), seconds=timeout,
                         what="the project/design/history check")
    sections = report.sections()
    return StageOutcome(sections=sections, level=report.level) if sections else None


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

    from saddle.models import STAGE_INTAKE, STAGE_INTENT
    from saddle.supervisor import run_stage, system_message

    results = []

    # The turn-end DRAIN — deliver what the Stop-hook checks found after the
    # agent's last reply (the agent has not seen those). A read, not a check:
    # best-effort like the commitment readout, never a classified stage.
    drained = ""
    try:
        drained = _drained_findings(ctx, session) if session else ""
    except Exception as exc:  # noqa: BLE001 — the drain must not wedge the turn
        print(f"intake_hook: drain error ({exc!r}); continuing", file=sys.stderr)

    # Stage 2 (intent), pick-drift axis — fast, deterministic: surface any
    # "committed a) then acted on b)" drift the agent made since the last prompt.
    # Through run_stage it gets its OWN bubble (an ALERT when drift surfaces) and
    # the fail-classification the old swallowed try/except lacked — a replay that
    # itself throws now becomes a loud classified ALERT, not a lost stderr line.
    results.append(run_stage(
        ctx, STAGE_INTENT,
        lambda: _replay_outcome(ctx, tracker, transcript_path, session),
        session=session, what="this prompt against the standing commitment",
    ))

    # Stage 2 (intent), project/design/history axis — the gap: does THIS prompt
    # pull against a settled design / closed decision / the stated focus? Its OWN
    # STAGE_INTENT bubble (ALERT for a hard contradiction, NOTICE for scope-creep),
    # and a failed check (timeout / provider / malformed) is a classified ALERT,
    # never a swallowed "looked fine". Skipped (silent) on a trivial continuation.
    results.append(run_stage(
        ctx, STAGE_INTENT,
        lambda: _history_outcome(ctx, prompt),
        session=session,
        what="this prompt against the project's settled designs & decisions",
    ))

    # bind THIS prompt if it's a pick, so the commitment surface below reflects the
    # choice the user just made (immediate, not a turn late). A pure state mutation
    # with no section of its own; a failure here is logged, never a false verdict.
    try:
        tracker.observe_user_message(ctx, prompt, session=session)
    except Exception as exc:  # noqa: BLE001
        print(f"intake_hook: observe error ({exc!r}); continuing", file=sys.stderr)

    # Stage 1 (intake) — the itemized know/do list (+ scope warning), fail LOUD.
    # run_stage owns the discipline: a clean itemize is a NOTICE bubble; a slow /
    # failed one is a CLASSIFIED ALERT naming what saddle did not verify — never
    # the old silent degrade.
    results.append(run_stage(
        ctx, STAGE_INTAKE,
        lambda: _itemize_outcome(ctx, prompt, session),
        session=session, what="the breakdown of your message into its requests",
    ))

    # Stage 2 (intent), sibling-project axis (mediator design §4): a prompt that
    # routes asks to ANOTHER known project gets a history check against THAT
    # project's settled state, under THAT project's context — its bubbles, drift
    # findings, and eventual lessons land in the sibling's ledger, never blended
    # into the ambient project's. One extra bounded LLM call per involved
    # sibling (usually zero). Fail-loud like every stage; a failure names the
    # sibling it could not check.
    try:
        from saddle.context import Context as _Ctx
        from saddle.focus import active_scope_asks

        sibling_asks = active_scope_asks(session, ctx.tenant) if session else {}
    except Exception as exc:  # noqa: BLE001 — routing read must not wedge the turn
        sibling_asks = {}
        print(f"intake_hook: sibling-scope read error ({exc!r})", file=sys.stderr)
    for slug, asks in sorted(sibling_asks.items()):
        if slug == ctx.project or not asks:
            continue
        sib_ctx = _Ctx(tenant=ctx.tenant, project=slug)
        joined = "\n".join(f"- {a}" for a in asks)
        results.append(run_stage(
            sib_ctx, STAGE_INTENT,
            lambda c=sib_ctx, j=joined: _history_outcome(c, j),
            session=session,
            what=f"the asks routed to {slug} against {slug}'s settled designs",
        ))

    # the standing commitment the agent must honor — a dialog readout for the
    # agent channel, not a fail-classified stage (it is a reminder, not drift).
    commitment = ""
    try:
        b, fork = tracker.committed_fork(ctx, session=session)
        if b is not None:
            commitment = _render_binding(b, fork)
    except Exception as exc:  # noqa: BLE001
        print(f"intake_hook: commitment error ({exc!r}); continuing", file=sys.stderr)

    # Two channels, deliberately different audiences:
    #  * agent-context (additionalContext): EVERY stage's sections + the standing
    #    commitment — the model's single read, including the routine reminders.
    #  * user-screen (systemMessage): ONLY what a stage SPOKE — drift caught, an
    #    itemization, or a could-not-run ALERT — so a clean turn stays quiet on
    #    screen but saddle is never invisible when it has something to say. The
    #    commitment is an agent reminder, not a screen herald, so it is NOT passed.
    sections = [s for r in results for s in r.sections]
    if drained:
        sections.insert(0, drained)  # last turn's unseen findings lead
    if commitment:
        sections.append(commitment)
    sm = system_message(ctx, results)
    _emit_context(ctx, sections, system_msg=sm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
