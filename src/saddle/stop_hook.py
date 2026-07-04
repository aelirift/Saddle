"""Stop (turn-end) bubble-up hook — saddle's retrospective drift checks.

This is the TURN-END counterpart to :mod:`saddle.intake_hook` (per-prompt, Stages
1+2) and :mod:`saddle.doctrine_hook` (per-tool guard + Stage 3). When the agent
finishes responding, the work it actually DID must be checked against what it
committed to — retrospectively, over the whole turn's edits:

* **Stage 3 (turn-end) — the prose-proposal review.** The pre-edit design gate
  (:mod:`saddle.doctrine_hook`) is EDIT-gated — it fires on the first code-mutating
  edit of a turn. A turn that proposes an approach in TEXT and writes no file (a
  'here's my plan / which option?' turn, a pure discussion turn) triggers no edit,
  so the edit-gated gate never sees it: the prose-proposal BLIND SPOT. This hook
  always fires at turn-end, so it closes the spot — when the design gate did NOT
  already fire this turn, it audits the turn's proposed approach through the SAME
  :func:`saddle.design.audit_proposal` engine. The two SHARE the gate's per-turn
  anchor marker, so exactly one reviews a turn (the pre-edit one if any edit fired,
  this one otherwise). Its caught issues feed Stage 5's harvest like the pre-edit
  gate's do.
* **Stage 4 — code conformance.** Did the code the turn wrote DRIFT from a settled
  design's committed completeness surface? :func:`saddle.design.conformance_scan`
  re-parses the target tree FRESH and re-runs each settled design's own gate
  (:func:`saddle.design.intent_drift`); a design whose surface the code no longer
  satisfies is the "committed to a structural fix in Stage 3, then quietly wrote
  the swallow anyway" miss — caught here even though Stage 3 blessed the prose.
  Because the parse is fresh every turn, an UNcorrected drift is re-caught next
  turn-end (the "not once and done" guarantee), not blessed by a cached map.
* **Stage 5 — lesson harvest.** Did this turn TEACH anything? The design/code
  drift caught this turn (Stages 3 + 4, read back from the durable outbox since
  the last turn-end) is distilled by :func:`saddle.design.harvest_turn` into
  durable, deduped DKB lessons (``source=audit``) the NEXT turn's design/intent
  retrieval stands on — saddle's cumulative, not "caught once and done",
  guarantee. A clean turn teaches nothing and stays silent.

Like the other two hooks this is agent-independent: saddle reads the project's own
ledger and the code on disk, with no voluntary call from the agent to skip. The
(tenant, project) it speaks for is fixed by ``SADDLE_TENANT`` / ``SADDLE_PROJECT``
and the tree by ``SADDLE_CODE_ROOT`` — exactly as the CLI, MCP server, and the
other hooks resolve it.

Wire it (``.claude/settings.json``)::

    "hooks": {"Stop": [{
      "hooks": [{"type": "command",
                 "command": "PYTHONPATH=${CLAUDE_PROJECT_DIR}/src python3 -m saddle.stop_hook"}]}]}

Knobs (env, all optional):

* ``SADDLE_HOOK_DESIGN``        "0" disables the Stage-3 turn-end proposal review;
                                default on. (The pre-edit gate honours the same knob.)
* ``SADDLE_HOOK_DESIGN_TIMEOUT`` deadline (s) for the turn-end design-review LLM
                                call; default 60.
* ``SADDLE_HOOK_CODE``          "0" disables the Stage-4 conformance scan; default on.
* ``SADDLE_HOOK_CODE_LIMIT``    how many of the most-recent settled designs to gate
                                against the current code; default 8.
* ``SADDLE_HOOK_LESSON``        "0" disables the Stage-5 lesson harvest; default on.
* ``SADDLE_HOOK_LESSON_TIMEOUT`` deadline (s) for the harvest LLM call; default 60.

Protocol: exit 0 always. Observe-only EXCEPT the goal-keeper (user-granted
2026-07-03): an active, unmet goal with an agent not blocked on the user
emits a Stop ``decision: block`` that drives the agent back to work — see
the goal-keeper section + the recorded policy directive. Everything else: The five stages observe and surface —
they do NOT block the turn (blocking is the doctrine guard's job), and a ``Stop``
hook that "blocked" would force the agent to keep going, which is enforcement, not
observation. So this hook emits NO stop decision: the durable per-stage bubble
(emitted by :func:`saddle.supervisor.run_stage`) is the channel an AFK / non-TTY
human reads saddle's turn-end voice from, with a human-readable copy on stderr for
an interactive TTY. Two failure regimes, deliberately different (mirroring
``intake_hook``): a HOOK-LEVEL infra error (unparseable payload, ctx resolution,
no code root) fails **open** — exit 0, a stderr log, no false verdict — because
saddle cannot speak for a turn it never resolved; but a STAGE that cannot run
(a parse blow-up) fails **loud** through ``run_stage`` — a classified ALERT naming
what saddle did NOT verify — because a swallowed conformance check is the silent
fail-open ``knowledge_seed_fail_loud`` forbids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saddle.codemap import Finding
    from saddle.context import Context
    from saddle.design import ConformanceResult
    from saddle.supervisor import StageOutcome

# Cap how many unsatisfied touchpoints a single drifting design shows in the
# bubble — the ledger/result keeps them all; the surface stays readable.
_MAX_FINDINGS = 8


# -- per-session harvest watermark (Stage 5) ---------------------------------

def _harvest_watermark_path(session: str):
    """Where this session's last-harvest timestamp is parked — alongside the
    saddle DB, in its own ``harvest`` namespace, so it never collides with the
    design gate's turn anchor or the intake replay cursor."""
    from saddle.store import default_db_path

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session) or "default"
    return default_db_path().parent / "harvest" / f"{safe}.json"


def _harvest_watermark(session: str) -> float:
    """The ts of this session's last turn-end harvest (``0.0`` if none yet). Stage
    5 learns only from drift bubbled SINCE this mark, so each turn's lessons are
    harvested once. An absent / unreadable mark ⇒ ``0.0`` (re-consider all — the
    DKB title-dedup still prevents re-filing, so it is safe, never double-files)."""
    try:
        raw = _harvest_watermark_path(session).read_text(encoding="utf-8")
        return float(json.loads(raw).get("ts", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0


def _set_harvest_watermark(session: str, ts: float) -> None:
    """Advance the harvest watermark to ``ts`` (atomic temp + replace, like the
    design marker). Best-effort: an IO failure is logged, never raised — at worst
    the next turn re-considers the same drift (deduped at the DKB), never wedges."""
    p = _harvest_watermark_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"ts": float(ts)}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        print(f"stop_hook: harvest watermark write failed ({exc!r})", file=sys.stderr)


# -- emit --------------------------------------------------------------------

def _emit(ctx: "Context", sections: list[str], *, system_msg: str = "") -> None:
    """Emit saddle's turn-end voice on the two channels a ``Stop`` hook has:

    * ``systemMessage`` on a stdout JSON — the ONE channel Claude Code renders to
      the watching HUMAN, so saddle's turn-end findings (code drift, a lesson
      harvest, a could-not-run ALERT) are visible ON SCREEN even under a non-TTY /
      SDK host where stderr is swallowed — the "I don't see any outputs" gap. A
      ``Stop`` hook has no ``additionalContext`` channel (it runs after the agent
      replied), so this is saddle's only LIVE surface to the human; ``suppressOutput``
      keeps the raw JSON out of transcript mode while the systemMessage still shows.
    * a human-readable copy on stderr — the on-screen view in an interactive TTY.

    The DURABLE per-stage bubbles already went out via :func:`run_stage`. We emit
    NO stop ``decision`` — observe-only, the turn always completes (a ``Stop`` block
    would force the agent to keep going, which is enforcement, not observation).
    Empty sections AND an empty ``system_msg`` render to nothing, so a clean turn
    stays silent on every channel."""
    from saddle.supervisor import render_sections

    if system_msg:
        print(json.dumps({"systemMessage": system_msg, "suppressOutput": True}))
    body = render_sections(ctx, sections)
    if body:
        print(body, file=sys.stderr)


# -- rendering ---------------------------------------------------------------

def _finding_line(f: "Finding") -> str:
    """Render one unsatisfied touchpoint, prefixing the enclosing function when the
    finding carries it (``value_propagation`` / liveness checks set
    ``detail['func']``). Naming the SITE — ``in sweep(): ...`` — makes the drift
    actionable: the agent fixes a function, not a bare line number."""
    func = (f.detail or {}).get("func")
    return f"  in {func}(): {f}" if func else f"  {f}"


def _render_drift(result: "ConformanceResult") -> list[str]:
    """One section per settled design the current code does not satisfy. Each
    lists the design and its unsatisfied touchpoints (capped for readability)."""
    sections: list[str] = []
    for d in result.drifts:
        shown = d.findings[:_MAX_FINDINGS]
        lines = [
            f"⚠ CODE DRIFT — the code does not satisfy settled design "
            f"{d.design_id} ({d.summary}):"
        ]
        lines.extend(_finding_line(f) for f in shown)
        if len(d.findings) > len(shown):
            lines.append(f"  (+{len(d.findings) - len(shown)} more touchpoint(s))")
        sections.append("\n".join(lines))
    return sections


# -- Stage 3 (turn-end) — the prose-proposal review -------------------------

def _proposal_outcome(
    ctx: "Context", session: str, transcript_path: str
) -> "StageOutcome | None":
    """Stage 3 at TURN-END — audit a design the agent proposed in PROSE but never
    triggered the pre-edit review for.

    Stage 3 in :mod:`saddle.doctrine_hook` is EDIT-gated: it fires on the first
    code-mutating edit of a turn. A turn that proposes an approach in TEXT and stops
    there (the 'here's my plan / which option do you want?' turn, or a pure
    discussion turn) triggers no edit, so the edit-gated gate never sees it — the
    prose-proposal BLIND SPOT. This closes it from the one hook that ALWAYS fires at
    turn-end: if the design gate did NOT already fire this turn AND the agent
    actually wrote an approach, audit that approach through the EXACT same engine
    (:func:`saddle.design.audit_proposal`) the pre-edit gate uses — so a band-aid /
    misread cause / uncovered ask in a prose plan is caught before the user acts on
    it, not silently blessed because no file happened to be edited.

    Complementary to the pre-edit gate, never duplicative: it SHARES that gate's
    per-turn anchor marker (:func:`saddle.doctrine_hook._design_already_fired` /
    ``_mark_design_fired`` — a persistent file in the ``design_gate`` namespace), so
    EXACTLY ONE of the two reviews a turn — the pre-edit one if any edit fired (it
    marked the turn), this one otherwise. It marks the turn after running so a repeat
    ``Stop`` in the same turn stays silent.

    Returns ``None`` (silent) when: the gate is disabled, the pre-edit gate already
    fired, there is no turn to anchor on, no approach prose was written (a pure
    tool-run / no-op turn is not a blind spot — nothing to review), or the audit is
    clean. Issues return an ALERT under ``STAGE_DESIGN`` (the same shape + ``meta``
    as the pre-edit gate) so Stage 5's harvest learns from them THIS turn.

    Setup (no transcript, anchor IO) is best-effort and returns silent — turn-end
    observation must never wedge the turn — but the AUDIT itself fails LOUD via
    :func:`run_stage` (the caller): a timeout / outage / contract gap PROPAGATES to
    a classified ALERT, never a swallowed 'looked fine'."""
    if os.environ.get("SADDLE_HOOK_DESIGN", "1") == "0":
        return None
    if not transcript_path:
        return None
    from saddle.doctrine_hook import _design_already_fired, _mark_design_fired
    from saddle.transcript import latest_turn

    turn = latest_turn(transcript_path)
    if not turn.anchor or _design_already_fired(session, turn.anchor):
        return None  # no turn to anchor on, or the pre-edit gate already reviewed it
    approach = (turn.approach or "").strip()
    if not approach:
        # No approach prose AND no edit this turn -> there is no design to review
        # (a pure tool-run / no-op turn), not a blind spot. Mark it so a repeat Stop
        # doesn't reconsider, and stay silent.
        _mark_design_fired(session, turn.anchor)
        return None

    from saddle.design import audit_proposal
    from saddle.models import BUBBLE_ALERT
    from saddle.supervisor import StageOutcome, run_bounded

    try:
        timeout = float(os.environ.get("SADDLE_HOOK_DESIGN_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    verdict = run_bounded(
        audit_proposal(turn.goal, approach, ctx),
        seconds=timeout,
        what="the turn-end design review (root-cause / band-aid / coverage)",
    )
    # One review per turn, even on a repeat Stop. Reached only when the audit RAN
    # (clean or issues); a RAISED audit propagated above to run_stage's loud ALERT
    # and left the turn unmarked, so a retry can still complete the review.
    _mark_design_fired(session, turn.anchor)
    if not verdict.has_issues:
        return None  # the prose approach audited clean -> silent
    from saddle.voice import design_issues_turn_end

    body = "\n".join(f"  • {i}" for i in verdict.issues)
    return StageOutcome(
        sections=[design_issues_turn_end(body)],
        level=BUBBLE_ALERT,
        meta={"issues": list(verdict.issues), "origin": "turn-end"},
    )


# -- Stage 4 — code conformance ----------------------------------------------

def _conformance_outcome(ctx: "Context") -> "StageOutcome | None":
    """Stage 4 (code) — re-verify the project's settled designs against the code
    as it stands at turn-end. Returns the drift section(s) as a NOTICE, escalated
    to an ALERT when any finding is error-grade (a hard conformance break the
    design promised against), or ``None`` when the scan is disabled or nothing
    drifted. It does **not** catch a scan failure: a parse blow-up PROPAGATES to
    :func:`run_stage`, which classifies it and bubbles a LOUD ALERT naming what
    saddle could not verify — never a swallowed 'looked fine'. No settled design
    with a surface, or no code root, returns a clean (empty) result inside the
    engine, so the stage stays silent (nothing can drift)."""
    if os.environ.get("SADDLE_HOOK_CODE", "1") == "0":
        return None
    from saddle.design import conformance_scan
    from saddle.models import BUBBLE_ALERT, BUBBLE_NOTICE
    from saddle.supervisor import StageOutcome

    try:
        limit = int(os.environ.get("SADDLE_HOOK_CODE_LIMIT", "8") or 8)
    except ValueError:
        limit = 8
    result = conformance_scan(ctx, limit=limit)
    if not result.has_drift:
        return None
    sections = _render_drift(result)
    level = BUBBLE_ALERT if result.has_error else BUBBLE_NOTICE
    return StageOutcome(
        sections=sections,
        level=level,
        title="code drifted from a settled design",
        meta={
            "origin": "turn-end",
            "designs_checked": result.designs_checked,
            "drifts": [
                {
                    "design_id": d.design_id,
                    "summary": d.summary,
                    "findings": [str(f) for f in d.findings],
                    "has_error": d.has_error,
                }
                for d in result.drifts
            ],
        },
    )


# -- Stage 5 — lesson harvest ------------------------------------------------

def _turn_issues(
    ctx: "Context", session: str, since_ts: float
) -> dict[str, list[str]]:
    """The design/code flaws CAUGHT this turn, read back from the durable outbox
    since the last harvest, GROUPED BY the project whose ledger each landed in
    (mediator design §4 — a mixed-scope turn teaches each project its OWN
    lessons, never a blended set): Stage 3's ``meta['issues']`` (band-aids /
    misread causes) and Stage 4's ``meta['drifts'][*]['findings']``
    (code-conformance breaks). An intake infra-failure or an intent pull is NOT
    a generalizable design lesson, so those stages are deliberately excluded —
    the harvest engine is tuned for design/code flaws."""
    from saddle.bubble import recent_bubbles
    from saddle.models import STAGE_CODE, STAGE_DESIGN

    issues: dict[str, list[str]] = {}
    for b in recent_bubbles(ctx, session=session, since_ts=since_ts, limit=200,
                            any_project=True):
        meta = b.meta or {}
        found: list[str] = []
        if b.stage == STAGE_DESIGN:
            found = [str(i).strip() for i in meta.get("issues", []) if str(i).strip()]
        elif b.stage == STAGE_CODE:
            found = [
                str(f).strip()
                for d in meta.get("drifts", [])
                for f in d.get("findings", [])
                if str(f).strip()
            ]
        if found:
            issues.setdefault(b.project or ctx.project, []).extend(found)
    return issues


def _harvest_outcome(
    ctx: "Context", session: str, transcript_path: str, since_ts: float
) -> "StageOutcome | None":
    """Stage 5 (lesson) — distil this turn's caught drift into durable DKB lessons.
    Returns a NOTICE naming the lessons filed, or ``None`` when the stage is
    disabled, nothing was caught this turn, or nothing cleared the harvest bar
    (all silent — a clean turn teaches nothing). A harvest that THROWS is NOT
    caught here: it PROPAGATES to :func:`run_stage` to be classified and bubbled as
    a LOUD ALERT (fail-loud), never a swallowed 'learned nothing'."""
    if os.environ.get("SADDLE_HOOK_LESSON", "1") == "0":
        return None
    by_project = _turn_issues(ctx, session, since_ts)
    if not by_project:
        return None  # a clean turn -> no LLM call, nothing to learn

    from saddle.context import Context
    from saddle.design import harvest_turn
    from saddle.models import BUBBLE_NOTICE
    from saddle.supervisor import StageOutcome, run_bounded
    from saddle.transcript import latest_turn

    goal = latest_turn(transcript_path).goal if transcript_path else ""
    try:
        timeout = float(os.environ.get("SADDLE_HOOK_LESSON_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    # One harvest per project the turn's caught drift landed in — each files
    # into ITS project's ledger, so a mixed-scope turn never blends lessons
    # (mediator design §4). The ambient project goes first for readability.
    ordered = sorted(by_project.items(), key=lambda kv: kv[0] != ctx.project)
    total, sections, titles_all = 0, [], []
    considered = 0
    for project, issues in ordered:
        p_ctx = ctx if project == ctx.project else Context(ctx.tenant, project)
        result = run_bounded(
            harvest_turn(goal, issues, p_ctx),
            seconds=timeout,
            what=f"the turn-end lesson harvest for {project}",
        )
        considered += result.considered
        if result.harvested <= 0:
            continue
        total += result.harvested
        titles_all.extend(result.titles)
        titles = "\n".join(f"  • {t}" for t in result.titles)
        sections.append(
            f"\U0001f4d3 Lessons saved for {project} — {result.harvested} from "
            f"this turn's caught problems (future checks on {project} will "
            f"use them):\n{titles}"
        )
    if total <= 0:
        return None  # considered the flaws; nothing new/general enough to file
    return StageOutcome(
        sections=sections,
        level=BUBBLE_NOTICE,
        title="lessons learned this turn",
        meta={
            "origin": "turn-end",
            "harvested": total,
            "considered": considered,
            "titles": titles_all,
            "projects": [p for p, _ in ordered],
        },
    )


# -- Voice — the plain-language check on saddle's OWN messages ----------------

def _voice_outcome(
    ctx: "Context", session: str, since_ts: float
) -> "StageOutcome | None":
    """Judge the prose saddle itself emitted this turn against the voice
    contract (:mod:`saddle.voice`) — enforcement, not aspiration: a render a
    non-technical reader can't follow is a finding, bubbled as an ALERT and
    harvestable like any other caught drift. Skips saddle's OWN voice-stage
    bubbles (they quote the offending jargon, so re-auditing them would
    re-flag the quote every turn — a feedback loop, not a finding). Silent
    when saddle said nothing this turn (no LLM call). Runs at turn-end where
    latency is off the critical path."""
    if os.environ.get("SADDLE_HOOK_VOICE", "1") == "0":
        return None
    from saddle.bubble import recent_bubbles
    from saddle.models import STAGE_VOICE

    texts = [
        b.text.strip()
        for b in recent_bubbles(ctx, session=session, since_ts=since_ts, limit=200)
        if b.stage != STAGE_VOICE and b.text.strip()
    ]
    if not texts:
        return None

    from saddle.models import BUBBLE_ALERT
    from saddle.supervisor import StageOutcome, run_bounded
    from saddle.voice import audit_plainness

    try:
        timeout = float(os.environ.get("SADDLE_HOOK_VOICE_TIMEOUT", "45") or 45)
    except ValueError:
        timeout = 45.0
    issues = run_bounded(
        audit_plainness("\n\n---\n\n".join(texts), ctx),
        seconds=timeout,
        what="the plain-language check of saddle's own messages this turn",
    )
    if not issues:
        return None
    body = "\n".join(f"  • {i}" for i in issues)
    return StageOutcome(
        sections=[
            "Saddle's own messages this turn were not plain enough for a "
            f"non-technical reader — each needs simpler wording:\n{body}"
        ],
        level=BUBBLE_ALERT,
        title="saddle's wording needs simplifying",
        meta={"issues": list(issues), "origin": "turn-end"},
    )


# -- Completion — did the USER'S ACTUAL GOAL get finished? --------------------

# The last completion verdict of THIS hook invocation — written by the
# completion stage, read by the goal-keeper decision below. A holder (not a
# return) because run_stage's contract returns a StageOutcome, and the keeper
# needs the verdict even when the stage says nothing (no overclaim).
_LAST_COMPLETION_VERDICT: list = []


def _completion_outcome(
    ctx: "Context", session: str, transcript_path: str
) -> "StageOutcome | None":
    """Judge this turn's completion claim against the goal AS THE USER MEANT
    IT (saddle.completion): a reply that reads as "finished" while the goal's
    broader clauses (a quality bar, an "everything"/"all" scope, still-open
    recorded asks) remain is an OVERCLAIM — alerted to the human now and
    delivered to the agent on its next turn via the drain, so a premature
    "done" is corrected instead of compounding into a cleared goal. A reply
    that makes no completion claim is silent. Fails LOUD via run_stage."""
    if os.environ.get("SADDLE_HOOK_COMPLETION", "1") == "0" or not transcript_path:
        return None
    from saddle.completion import audit_completion
    from saddle.models import BUBBLE_ALERT
    from saddle.supervisor import StageOutcome, run_bounded
    from saddle.transcript import latest_turn

    turn = latest_turn(transcript_path)
    if not turn.approach.strip():
        return None  # nothing was said this turn — nothing to judge
    try:
        timeout = float(os.environ.get("SADDLE_HOOK_COMPLETION_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    verdict = run_bounded(
        audit_completion(turn.goal, turn.approach, ctx),
        seconds=timeout,
        what="whether the goal was truly finished before the reply said so",
    )
    _LAST_COMPLETION_VERDICT.clear()
    _LAST_COMPLETION_VERDICT.append(verdict)
    if not verdict.overclaim:
        return None
    body = "\n".join(f"  • {m}" for m in verdict.missing) or "  • (unspecified)"
    return StageOutcome(
        sections=[
            "The reply reads as \"finished\", but the goal as the user meant "
            "it is NOT complete. Read the goal at its full breadth — these "
            f"parts are still open:\n{body}"
        ],
        level=BUBBLE_ALERT,
        title="finished was claimed too early",
        meta={"missing": list(verdict.missing), "origin": "turn-end"},
    )


# -- The GOAL-KEEPER — an unmet goal drives the agent back to work -------------
#
# User directive (2026-07-03): "if saddle catches it, it should drive you back
# into working." This supersedes the original observe-only decision for the
# Stop hook: when the completion audit says a driving goal is ACTIVE, NOT
# complete, and the agent is NOT genuinely blocked on the user, the stop is
# BLOCKED and the agent resumes with the missing list as its marching orders.
# Runaway guard: at most _KEEPER_MAX_CONSECUTIVE blocks in a row per session —
# past that, the keeper alerts loudly instead of blocking, so a truly stuck
# agent surfaces to the human rather than spinning.

_KEEPER_MAX_CONSECUTIVE = 3


def _keeper_marker_path(session: str):
    from saddle.store import default_db_path

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session) or "default"
    return default_db_path().parent / "keeper" / f"{safe}.json"


def _keeper_count(session: str) -> int:
    try:
        return int(json.loads(_keeper_marker_path(session).read_text(
            encoding="utf-8")).get("consecutive", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _set_keeper_count(session: str, n: int) -> None:
    p = _keeper_marker_path(session)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"consecutive": int(n)}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        print(f"stop_hook: keeper marker write failed ({exc!r})", file=sys.stderr)


def _keeper_decision(ctx: "Context", session: str) -> str:
    """The goal-keeper's verdict for this turn: a non-empty BLOCK REASON when
    the stop must be refused, else "". Reads the completion verdict the
    completion stage just computed; no verdict (stage off / failed / silent
    turn) means no block — enforcement never guesses."""
    if os.environ.get("SADDLE_GOAL_KEEPER", "1") == "0":
        return ""
    if not _LAST_COMPLETION_VERDICT:
        return ""
    verdict = _LAST_COMPLETION_VERDICT[0]
    if not verdict.should_keep_working:
        _set_keeper_count(session, 0)  # progress or a legit stop — reset
        return ""
    blocks = _keeper_count(session)
    if blocks >= _KEEPER_MAX_CONSECUTIVE:
        # The cap: alert the human instead of a fourth forced lap.
        from saddle.bubble import emit_bubble
        from saddle.models import BUBBLE_ALERT, STAGE_COMPLETION

        emit_bubble(
            ctx,
            "Saddle's goal-keeper stopped pushing after "
            f"{blocks} forced continuations in a row — the agent may be "
            "stuck. The goal is still not finished; it needs a human look.",
            level=BUBBLE_ALERT, stage=STAGE_COMPLETION, session=session,
            meta={"origin": "turn-end", "keeper_capped": True},
        )
        _set_keeper_count(session, 0)
        return ""
    _set_keeper_count(session, blocks + 1)
    from saddle.voice import goal_keeper_reason

    return goal_keeper_reason(list(verdict.missing))


# -- entry point -------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0  # nothing submitted
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print("stop_hook: unparseable hook payload; allowing", file=sys.stderr)
        return 0

    session = str(payload.get("session_id") or "")
    transcript_path = str(payload.get("transcript_path") or "")

    # Import lazily so an unreadable payload above never needs saddle's full import
    # graph, and so any import/ctx error surfaces as fail-open + a log.
    try:
        from saddle.context import resolve

        ctx = resolve(os.environ.get("SADDLE_TENANT"), os.environ.get("SADDLE_PROJECT"))
    except Exception as exc:  # noqa: BLE001 — a hook crash must not wedge the agent
        print(f"stop_hook: import/ctx error ({exc!r}); allowing", file=sys.stderr)
        return 0

    from saddle.models import STAGE_CODE, STAGE_DESIGN, STAGE_LESSON
    from saddle.supervisor import run_stage, system_message

    results = []

    # Stage 3 (turn-end) — the prose-proposal review. The pre-edit design gate
    # (doctrine_hook) is EDIT-gated, so a turn that proposes an approach in TEXT and
    # writes no file is never reviewed — the prose-proposal blind spot. This closes
    # it: if the gate did NOT fire this turn and the agent wrote an approach, audit
    # it now through the SAME engine. It shares the gate's per-turn marker, so
    # exactly one review runs per turn. Runs FIRST so its STAGE_DESIGN ALERT is in
    # the outbox before Stage 5's harvest reads this turn's caught drift below.
    results.append(run_stage(
        ctx, STAGE_DESIGN,
        lambda: _proposal_outcome(ctx, session, transcript_path),
        session=session,
        what="the plan the agent wrote this turn (it changed no code yet)",
    ))

    # Stage 4 (code) — turn-end code-vs-design conformance. Its OWN STAGE_CODE
    # bubble (ALERT for an error-grade break, NOTICE for a soft gap); a scan that
    # itself throws (a parse blow-up) becomes a classified ALERT, never a swallowed
    # pass. Silent when no settled design declared a surface (nothing can drift).
    results.append(run_stage(
        ctx, STAGE_CODE,
        lambda: _conformance_outcome(ctx),
        session=session,
        what="the turn's code against the project's settled designs",
    ))

    # Stage 5 (lesson) — harvest this turn's caught drift (Stages 3 + 4, read from
    # the durable outbox since the last turn-end) into durable DKB lessons. Its own
    # STAGE_LESSON NOTICE names what was learned; a harvest that throws is a
    # classified ALERT (fail-loud). The watermark advances every turn so the next
    # turn only sees NEW drift — captured AFTER Stage 4 emitted (so its just-bubbled
    # code drift is harvested this turn) and BEFORE Stage 5 reads.
    since = _harvest_watermark(session)
    mark = time.time()
    results.append(run_stage(
        ctx, STAGE_LESSON,
        lambda: _harvest_outcome(ctx, session, transcript_path, since),
        session=session,
        what="this turn's caught drift into durable lessons",
    ))

    # Voice — did saddle itself speak plainly this turn? Reads every bubble in
    # the same window Stage 5 used PLUS the ones the stages above just emitted
    # (they precede this line), so a jargon-heavy alert is caught the very turn
    # it went out. Runs before the watermark advances; its own bubble is
    # excluded from future windows by stage, not by timestamp.
    from saddle.models import STAGE_VOICE

    results.append(run_stage(
        ctx, STAGE_VOICE,
        lambda: _voice_outcome(ctx, session, since),
        session=session,
        what="whether saddle's own messages this turn read plainly",
    ))

    # Completion gate — a reply that says "finished" is audited against the
    # goal AS THE USER MEANT IT (broad clauses + still-open ledger asks),
    # never against the sub-list the reply enumerates. The founding incident:
    # a goal auto-cleared on a confident summary while its "everything"/"AAA"
    # clauses were open (2026-07-03).
    from saddle.models import STAGE_COMPLETION

    results.append(run_stage(
        ctx, STAGE_COMPLETION,
        lambda: _completion_outcome(ctx, session, transcript_path),
        session=session,
        what="whether the goal was truly finished before the reply said so",
    ))
    _set_harvest_watermark(session, mark)

    # User-screen herald (systemMessage) for what a stage SPOKE this turn — code
    # drift, a lesson harvest, or a could-not-run ALERT — plus the stderr copy for
    # a TTY. The durable per-stage bubbles already went out via run_stage.
    sections = [s for r in results for s in r.sections]
    sm = system_message(ctx, results)

    # The goal-keeper (enforcement, user-directed 2026-07-03): an active,
    # unmet goal with an agent that is not blocked on the user REFUSES the
    # stop — the agent resumes with the missing list. Emitted as the Stop
    # decision JSON; everything above stays observation.
    reason = ""
    try:
        reason = _keeper_decision(ctx, session)
    except Exception as exc:  # noqa: BLE001 — keeper failure must not wedge the turn
        print(f"stop_hook: goal-keeper error ({exc!r}); not blocking", file=sys.stderr)
    if reason:
        print(json.dumps({
            "decision": "block",
            "reason": reason,
            "systemMessage": "🔁 saddle's goal-keeper: the goal is not "
                             "finished — sending the agent back to work.",
        }))
        print("[goal-keeper] BLOCKED the stop — goal still open", file=sys.stderr)
        return 0

    _emit(ctx, sections, system_msg=sm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
