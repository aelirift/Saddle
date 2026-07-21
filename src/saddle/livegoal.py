"""The live-goal layer — keeps the commitment LIVE instead of frozen (#76).

Saddle's commitment (``IntentTracker.active_binding``) is a deterministic
fork-pick ledger: the last confidently-resolved fork whose fork was not
explicitly retired. It survives a context-window wipe, but it FREEZES — when the
user naturally pivots to a new task with no "pick a)", the old pick keeps being
re-injected every turn as "already decided — act on it" (audit Gap 1, proven
live on ``p5458`` / ``p39``). The deterministic ledger has no way to notice the
topic moved; ``dialog.py``'s ``Matcher`` seam left the semantic judgment to "a
later layer" that was never built. This is that layer.

The design (creator ruling 2026-07-20 — BUBBLE-TO-CONFIRM, not auto-supersede):

* Each turn, if there is a live commitment, an LLM reads the committed goal + the
  user's new message and judges whether the user has DEMONSTRABLY moved to a
  different task. ASYMMETRIC CAUTION: a wrong supersede silently drops the real
  goal (the exact failure being fixed), so it only flags a CLEAR move; when in
  doubt it keeps the commitment.
* On a clear move it does NOT retire — it BUBBLES to the user ("you seem to have
  moved from X to Y — retire the X commitment?") and records a PENDING proposal.
  The commitment stays LIVE until the user says so.
* The next turn, a confirm-classifier reads the reply: a clear YES retires the
  fork (``FORK_SUPERSEDED``) so the commitment re-derives; a clear NO keeps it;
  anything else leaves the proposal pending (no nagging, no action).

Retirement is therefore ALWAYS user-closed — the same contract as the raised-item
ledger (#77) and consistent with the "never automatic" fork-retirement invariant.
The deterministic pick-ledger stays the strong signal; this LLM layer only keeps
it honest about whether it is still what the user is working on.

This mirrors :mod:`saddle.hold` (per-session marker + stateful classify + apply);
both are fail-LOUD (a classify failure PROPAGATES to :func:`run_stage`, and
because no marker is written on failure the prior state carries forward unchanged).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.llm.json_tools import call_json
from saddle.llm.pool import tenant_gate
from saddle.models import BUBBLE_NOTICE
from saddle.voice import VOICE_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    from saddle.dialog import IntentTracker
    from saddle.llm.protocol import LLMCaller
    from saddle.models import Binding, Fork


# -- the pending supersede proposal (per session) ----------------------------

@dataclass
class LiveGoalProposal:
    """A supersede awaiting the user's word. ``fork_id`` is the committed fork the
    proposal would retire; ``committed_gist`` / ``moved_to_gist`` are short plain
    summaries for the surfaces + the confirm classifier; ``anchor`` is the turn the
    proposal was raised on; ``updated_ts`` the last write time. An EMPTY ``fork_id``
    means 'no proposal pending'."""

    fork_id: str = ""
    committed_gist: str = ""
    moved_to_gist: str = ""
    anchor: str = ""
    updated_ts: float = 0.0

    @property
    def pending(self) -> bool:
        return bool(self.fork_id)

    def to_dict(self) -> dict:
        return {
            "fork_id": self.fork_id,
            "committed_gist": self.committed_gist,
            "moved_to_gist": self.moved_to_gist,
            "anchor": self.anchor,
            "updated_ts": self.updated_ts,
        }


def _proposal_path(session: str):
    from saddle.store import default_db_path

    safe = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in session
    ) or "default"
    return default_db_path().parent / "live_goal" / f"{safe}.json"


def read_proposal(session: str) -> LiveGoalProposal:
    """The session's pending supersede proposal, or an empty one (nothing pending).
    Absent -> empty (silent); corrupt / unreadable -> empty + a LOUD stderr line (a
    garbled proposal must never be trusted into retiring a commitment)."""
    path = _proposal_path(session)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LiveGoalProposal()
    except OSError as exc:
        print(f"livegoal: proposal unreadable ({exc!r}); treating as none",
              file=sys.stderr)
        return LiveGoalProposal()
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"livegoal: proposal corrupt ({exc!r}); treating as none",
              file=sys.stderr)
        return LiveGoalProposal()
    if not isinstance(doc, dict):
        print("livegoal: proposal is not an object; treating as none",
              file=sys.stderr)
        return LiveGoalProposal()
    return LiveGoalProposal(
        fork_id=str(doc.get("fork_id") or ""),
        committed_gist=str(doc.get("committed_gist") or ""),
        moved_to_gist=str(doc.get("moved_to_gist") or ""),
        anchor=str(doc.get("anchor") or ""),
        updated_ts=float(doc.get("updated_ts") or 0.0),
    )


def write_proposal(session: str, proposal: LiveGoalProposal) -> None:
    """Atomically persist ``proposal`` (temp file + ``os.replace``) so a crash
    mid-write can never leave a half-written proposal. Best-effort: an IO failure is
    logged, never raised."""
    proposal.updated_ts = time.time()
    path = _proposal_path(session)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(proposal.to_dict(), fh)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"livegoal: proposal write failed ({exc!r})", file=sys.stderr)


def clear_proposal(session: str) -> None:
    """Drop any pending proposal (best-effort; a missing file is already clear)."""
    try:
        _proposal_path(session).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"livegoal: proposal clear failed ({exc!r})", file=sys.stderr)


# -- rendering the committed goal --------------------------------------------

def _clip(text: str, limit: int = 200) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def commitment_gist(binding: "Binding", fork: "Fork | None") -> str:
    """A short plain-words summary of what the live commitment IS — the proposal it
    answered plus the chosen option's substance — so the classifier and the bubble
    can name it. Falls back to the bare choice id when the fork row is gone."""
    if fork is None or not fork.options:
        choice = getattr(binding, "choice_id", "") or getattr(binding, "label", "") or "?"
        ut = getattr(binding, "user_text", "")
        return f"{choice}{(' — ' + _clip(ut, 120)) if ut else ''}"
    label = getattr(binding, "label", "")
    chosen = next((o for o in fork.options if label and o.label == label), None)
    prompt = _clip(fork.prompt, 140) if fork.prompt else ""
    pick = _clip(chosen.text, 140) if chosen is not None else (label or "?")
    if prompt:
        return f"{prompt} → chose {label}) {pick}"
    return f"chose {label}) {pick}"


# -- the topic-move classifier ------------------------------------------------

@dataclass
class TopicVerdict:
    """Whether the user has moved off the committed goal. ``moved`` is the flag;
    ``moved_to`` is a short plain summary of what they seem to be on now (for the
    bubble); ``confidence`` is the classifier's own [0, 1]."""

    moved: bool = False
    moved_to: str = ""
    confidence: float = 0.0


_SYS_TOPIC_MOVE = (
    "You are the live-goal gate of a supervisory harness that sits between a user "
    "and a coding agent. Saddle holds a COMMITTED GOAL — a decision the user made "
    "earlier that the agent is still meant to be acting on. Read the user's newest "
    "message and decide ONE thing: has the user DEMONSTRABLY moved on to a "
    "DIFFERENT task, so that the committed goal is no longer what they are working "
    "on?\n"
    "Judge by INTENT and EFFECT, not surface wording. A message that CONTINUES, "
    "refines, questions, or reports on the committed goal has NOT moved "
    "(moved=false). A message that starts a clearly different piece of work — a "
    "new feature, a different bug, a different subsystem, 'let's switch to…', "
    "'now do…', 'forget that, …' — HAS moved (moved=true).\n"
    "ASYMMETRIC CAUTION — this is the crux. A WRONG 'moved' proposes dropping the "
    "real goal the user still cares about, which is the exact failure this gate "
    "exists to prevent. So flag moved=true ONLY on a CLEAR change of task; when the "
    "message could plausibly still be part of the committed goal, choose "
    "moved=false. A short continuation ('ok', 'go on', 'what about the tests') is "
    "NOT a move. Never guess a move from ambiguity.\n"
    "When moved=true, also return moved_to: a short plain-words summary (≤12 words) "
    "of what the user now seems to be working on. confidence: your own certainty "
    "from 0 to 1.\n"
    "Respond with ONLY JSON: "
    '{"moved": true|false, "moved_to": "...", "confidence": 0.0}'
    + VOICE_CONTRACT
)


def _move_prompt(committed_gist: str, prompt: str) -> str:
    return (
        f"THE COMMITTED GOAL (what the agent is meant to be working on):\n"
        f"{committed_gist}\n\n"
        f"THE USER'S NEWEST MESSAGE:\n{prompt}\n\n"
        "Has the user demonstrably moved on to a different task?"
    )


async def classify_topic_move(
    committed_gist: str,
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
) -> TopicVerdict:
    """Classify whether ``prompt`` moves off ``committed_gist``. One bounded
    ``call_json`` inside the tenant fairness gate. FAIL-LOUD: a caller/parse failure
    PROPAGATES (never a false 'moved'). Pass ``caller`` to bypass provider
    resolution (tests)."""
    text = (prompt or "").strip()
    if not text or not (committed_gist or "").strip():
        return TopicVerdict(moved=False, confidence=1.0)
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers

        caller = build_callers(ctx)["default"]
    async with tenant_gate(ctx):
        payload = await call_json(
            caller, _SYS_TOPIC_MOVE, _move_prompt(committed_gist, text),
            label="livegoal/topic-move",
        )
    moved = bool(payload.get("moved"))
    try:
        conf = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return TopicVerdict(
        moved=moved,
        moved_to=str(payload.get("moved_to") or "").strip(),
        confidence=conf,
    )


# -- the supersede-confirm classifier ----------------------------------------

CONFIRM = "confirm"      # "yes, retire it" — the user closes the old commitment
DECLINE = "decline"      # "no, keep it" — the commitment stands
CONFIRM_NONE = "none"    # neither — leave the proposal pending, do nothing
CONFIRM_KINDS: frozenset[str] = frozenset({CONFIRM, DECLINE, CONFIRM_NONE})


_SYS_SUPERSEDE_CONFIRM = (
    "You are the live-goal gate of a supervisory harness. Saddle asked the user "
    "whether to RETIRE an old committed goal because their work appeared to move "
    "to something new. Read the user's reply and decide what it means for that "
    "retirement.\n"
    "Choose exactly one kind:\n"
    "- confirm: the user agrees the old goal is done/abandoned and the new one "
    "takes over — 'yes', 'retire it', 'drop it', 'yes I've moved on', 'correct', "
    "'right, do the new thing'.\n"
    "- decline: the user says the old goal still stands — 'no', 'keep it', 'I'm "
    "still on that', 'not yet', 'no, come back to it', 'both'.\n"
    "- none: the reply does not answer the retire question — an unrelated "
    "instruction, a question, or anything ambiguous.\n"
    "ASYMMETRIC CAUTION: retiring wrongly DROPS a goal the user still wanted, so "
    "choose confirm ONLY on a clear yes to the retirement. When the reply is "
    "ambiguous or unrelated, choose none (the proposal simply stays pending — "
    "nothing is lost). A clear 'no' is decline.\n"
    "confidence: your own certainty from 0 to 1.\n"
    "Respond with ONLY JSON: "
    '{"kind": "confirm|decline|none", "confidence": 0.0}'
    + VOICE_CONTRACT
)


def _confirm_prompt(committed_gist: str, moved_to_gist: str, prompt: str) -> str:
    move = f" (they appeared to move to: {moved_to_gist})" if moved_to_gist else ""
    return (
        f"THE OLD COMMITTED GOAL saddle proposed to retire:\n{committed_gist}{move}\n\n"
        f"THE USER'S REPLY:\n{prompt}\n\n"
        "Does this reply confirm retiring the old goal, decline, or neither?"
    )


async def classify_supersede_confirm(
    committed_gist: str,
    moved_to_gist: str,
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
) -> str:
    """Classify the user's reply to a pending supersede proposal into one of
    :data:`CONFIRM_KINDS`. Bounded ``call_json``; FAIL-LOUD on caller/parse error."""
    text = (prompt or "").strip()
    if not text:
        return CONFIRM_NONE
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers

        caller = build_callers(ctx)["default"]
    async with tenant_gate(ctx):
        payload = await call_json(
            caller, _SYS_SUPERSEDE_CONFIRM,
            _confirm_prompt(committed_gist, moved_to_gist, text),
            label="livegoal/supersede-confirm",
        )
    kind = str(payload.get("kind") or "").strip().lower()
    return kind if kind in CONFIRM_KINDS else CONFIRM_NONE


# -- the composed turn (read commitment -> pending? confirm : detect move) ----

MOVE_PROPOSED = "move_proposed"   # bubbled a supersede proposal; commitment kept
RETIRED = "retired"               # user confirmed; the fork was superseded
KEPT = "kept"                     # user declined; the commitment stands
NOOP = "noop"                     # nothing to do this turn


@dataclass
class LiveGoalOutcome:
    """What the live-goal step did this turn. ``action`` is one of the module
    constants; ``herald`` is the plain on-screen line when the user should see a
    change (a proposal raised, a commitment retired or kept), else ""; ``level`` is
    its bubble level; ``fork_id`` the commitment involved."""

    action: str = NOOP
    herald: str = ""
    level: str = BUBBLE_NOTICE
    fork_id: str = ""


async def livegoal_turn(
    prompt: str,
    session: str,
    *,
    ctx: Context | None = None,
    tracker: "IntentTracker | None" = None,
    caller: "LLMCaller | None" = None,
) -> LiveGoalOutcome:
    """The whole live-goal step for one prompt: read the live commitment; if a
    supersede proposal is pending, read the reply and retire / keep / wait; else
    detect whether the user moved off the commitment and, on a clear move, bubble a
    proposal (never auto-retiring). The single coroutine the intake hook drives
    under a deadline via :func:`saddle.supervisor.run_bounded`."""
    ctx = ctx or _default_ctx()
    if tracker is None:
        from saddle.dialog import get_tracker

        tracker = get_tracker()

    binding, fork = tracker.committed_fork(ctx, session=session)
    if binding is None:
        # No live commitment -> nothing to track; drop any stale proposal.
        if read_proposal(session).pending:
            clear_proposal(session)
        return LiveGoalOutcome(action=NOOP)

    live_fork_id = binding.fork_id
    proposal = read_proposal(session)

    # -- a proposal is pending: is THIS message the user's answer? --------------
    if proposal.pending:
        if proposal.fork_id != live_fork_id:
            # The commitment already moved on its own (a fresh pick / retirement).
            # The stale proposal no longer applies.
            clear_proposal(session)
        else:
            kind = await classify_supersede_confirm(
                proposal.committed_gist, proposal.moved_to_gist, prompt, ctx,
                caller=caller,
            )
            if kind == CONFIRM:
                tracker.supersede(ctx, live_fork_id, session=session)
                clear_proposal(session)
                from saddle.voice import livegoal_retired

                return LiveGoalOutcome(
                    action=RETIRED, fork_id=live_fork_id,
                    herald=livegoal_retired(proposal.committed_gist),
                    level=BUBBLE_NOTICE,
                )
            if kind == DECLINE:
                clear_proposal(session)
                from saddle.voice import livegoal_kept

                return LiveGoalOutcome(
                    action=KEPT, fork_id=live_fork_id,
                    herald=livegoal_kept(proposal.committed_gist),
                    level=BUBBLE_NOTICE,
                )
            # none -> leave the proposal pending; don't nag, don't act.
            return LiveGoalOutcome(action=NOOP, fork_id=live_fork_id)

    # -- no proposal pending: has the user moved off the committed goal? --------
    gist = commitment_gist(binding, fork)
    verdict = await classify_topic_move(gist, prompt, ctx, caller=caller)
    if not verdict.moved:
        return LiveGoalOutcome(action=NOOP, fork_id=live_fork_id)

    write_proposal(session, LiveGoalProposal(
        fork_id=live_fork_id,
        committed_gist=gist,
        moved_to_gist=verdict.moved_to,
    ))
    from saddle.voice import livegoal_move_suspected

    return LiveGoalOutcome(
        action=MOVE_PROPOSED, fork_id=live_fork_id,
        herald=livegoal_move_suspected(gist, verdict.moved_to),
        level=BUBBLE_NOTICE,
    )
