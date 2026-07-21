"""The design-hold posture — an ORTHOGONAL per-session axis: did the USER ask to
be in the loop before code is written?

WHY THIS EXISTS
---------------
The pre-code design gate (:mod:`saddle.doctrine_hook`) audits the agent's plan
and, on a CLEAN audit, AUTO-SETTLES it as an agreed design
(``approved_by="converged"``). That is right when saddle and the agent are meant
to converge on their own — but it is WRONG whenever the USER said "show me the
plan first" / "don't just start coding". The auto-settle fired regardless of
whether the user asked to hold, so a clean-looking plan sailed past the user who
explicitly wanted to approve it.

This module adds the missing axis. It is:

* **Orthogonal** — a separate per-session ``posture`` (default / hold /
  autonomous) that lives alongside the design gate, not inside it. The design
  gate still judges the plan's QUALITY; this judges whether the user wants a say.
* **Checked FIRST and DETERMINISTICALLY** — the deny path (holding a code edit
  under an active hold) uses NO language model. Reading the posture is a file
  read; the decision is a plain state comparison. The classifier that MOVES the
  posture is a language model, but it never sits in the deny path.
* **Asymmetric by design** — the classifier never fabricates an APPROVE or an
  AUTONOMY grant out of an ambiguous message (those default to NONE); it leans
  toward REQUEST_REVIEW on a plausible hold. Getting "hold" wrong costs a
  redirect; getting "approved" wrong lets unreviewed code through.

THE POSTURE
-----------
Three states, one per session (never shared across sessions — a fresh session
always starts at :data:`HOLD_DEFAULT`):

* :data:`HOLD_DEFAULT` — the original behaviour: the design gate may auto-settle
  a clean plan.
* :data:`HOLD_HELD_STATE` — the user asked to approve the plan first. A clean
  audit is HELD (not settled); an unapproved code edit is DENIED and the agent
  is redirected to present the plan.
* :data:`HOLD_AUTONOMOUS` — the user handed over the wheel ("go with your
  recommendations"). Sticky: it stays until the user intervenes or asks to
  review again.

FAIL POSTURE
------------
* posture absent            -> :data:`HOLD_DEFAULT` (fail-open, silent).
* posture corrupt           -> :data:`HOLD_DEFAULT` + a LOUD stderr line.
* classifier failure        -> PROPAGATES (fail-loud, for ``run_stage`` to
  classify), and the posture is left UNCHANGED (sticky carry-forward).
* hold deny decision        -> fail-CLOSED (:func:`design_in_play` defaults True
  under an active hold), which is safe because the deny REDIRECTS.

This SUPERSEDES the brittle keyword matcher
:func:`saddle.stop_hook._user_directed_halt` (a hardcoded word list that reads
"stop"/"halt" literally): that judged only the STOP axis by wording, while this
judges the review/approve/autonomy axes by INTENT. ``_user_directed_halt`` is
left in place for now (it still guards the goal-keeper's stop path); this module
is the semantic replacement to migrate that to.
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
    from saddle.llm.protocol import LLMCaller


# -- the posture states + scopes ---------------------------------------------

HOLD_DEFAULT = "default"        # the design gate's original auto-settle behaviour
HOLD_HELD_STATE = "hold"        # the user asked to approve the plan before code
HOLD_AUTONOMOUS = "autonomous"  # the user handed over the wheel (sticky)
HOLD_STATES: frozenset[str] = frozenset(
    {HOLD_DEFAULT, HOLD_HELD_STATE, HOLD_AUTONOMOUS}
)

# How long a hold lasts. ``once`` releases after a single approval; ``standing``
# keeps the user in the loop until they explicitly hand over or intervene.
SCOPE_ONCE = "once"
SCOPE_STANDING = "standing"
HOLD_SCOPES: frozenset[str] = frozenset({SCOPE_ONCE, SCOPE_STANDING})


@dataclass
class HoldPosture:
    """The per-session review posture.

    ``state`` is one of :data:`HOLD_STATES`. ``hold_scope`` (only meaningful in
    :data:`HOLD_HELD_STATE`) is whether the hold releases after one approval or
    stands. ``held`` carries the held ask — ``{anchor, goal, approach_digest,
    condition}`` — so the classifier and the surfaces can name WHAT is held.
    ``approved_design_id`` is set once the user approves (the v1 "approved"
    signal). ``autonomy_since_anchor`` records the turn autonomous mode began,
    for the on-screen reminder. ``updated_ts`` is the last write time.
    """

    state: str = HOLD_DEFAULT
    hold_scope: str = SCOPE_ONCE
    held: dict = field(default_factory=dict)
    approved_design_id: str = ""
    autonomy_since_anchor: str = ""
    updated_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "hold_scope": self.hold_scope,
            "held": dict(self.held),
            "approved_design_id": self.approved_design_id,
            "autonomy_since_anchor": self.autonomy_since_anchor,
            "updated_ts": self.updated_ts,
        }


# -- per-session persistence (same discipline as completion / focus markers) --

def _posture_path(session: str):
    from saddle.store import default_db_path

    safe = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in session
    ) or "default"
    return default_db_path().parent / "design_hold" / f"{safe}.json"


def read_posture(session: str) -> HoldPosture:
    """The session's current posture. Absent -> :data:`HOLD_DEFAULT` (fail-open,
    silent — a session that never held starts at default); corrupt / unreadable
    -> :data:`HOLD_DEFAULT` + a LOUD stderr line (a garbled posture must never be
    trusted as a hold OR a grant, and the operator should know it was reset)."""
    path = _posture_path(session)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HoldPosture()  # never held -> default, silently
    except OSError as exc:
        print(f"hold: posture unreadable ({exc!r}); using DEFAULT", file=sys.stderr)
        return HoldPosture()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        print(f"hold: posture corrupt ({exc!r}); using DEFAULT", file=sys.stderr)
        return HoldPosture()
    if not isinstance(doc, dict):
        print("hold: posture is not an object; using DEFAULT", file=sys.stderr)
        return HoldPosture()
    state = str(doc.get("state") or HOLD_DEFAULT)
    if state not in HOLD_STATES:
        print(f"hold: unknown posture state {state!r}; using DEFAULT", file=sys.stderr)
        state = HOLD_DEFAULT
    scope = str(doc.get("hold_scope") or SCOPE_ONCE)
    if scope not in HOLD_SCOPES:
        scope = SCOPE_ONCE
    held = doc.get("held")
    return HoldPosture(
        state=state,
        hold_scope=scope,
        held=dict(held) if isinstance(held, dict) else {},
        approved_design_id=str(doc.get("approved_design_id") or ""),
        autonomy_since_anchor=str(doc.get("autonomy_since_anchor") or ""),
        updated_ts=float(doc.get("updated_ts") or 0.0),
    )


def write_posture(session: str, posture: HoldPosture) -> None:
    """Atomically persist ``posture`` for ``session`` (temp file + ``os.replace``,
    like the focus / cursor markers) so a crash mid-write can never leave a
    half-written posture that reads as a phantom hold. Best-effort: an IO failure
    is logged, never raised — at worst the next read sees the prior posture."""
    posture.updated_ts = time.time()
    path = _posture_path(session)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(posture.to_dict(), fh)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"hold: posture write failed ({exc!r})", file=sys.stderr)


# -- the review-intent taxonomy ----------------------------------------------

REQUEST_REVIEW = "request_review"    # "show me the plan first" / "don't just code"
APPROVE = "approve"                  # "yes, go ahead" — approve the held plan
AUTONOMY_GRANT = "autonomy_grant"    # "go with your recommendations" — take the wheel
INTERVENE = "intervene"              # "stop" / "hold on" — take the wheel back
NONE = "none"                        # nothing that moves the posture
REVIEW_INTENT_KINDS: frozenset[str] = frozenset(
    {REQUEST_REVIEW, APPROVE, AUTONOMY_GRANT, INTERVENE, NONE}
)


@dataclass
class ReviewIntent:
    """What a user message means for the review posture. ``kind`` is one of
    :data:`REVIEW_INTENT_KINDS`; ``scope`` is a hint on a REQUEST_REVIEW (once /
    standing); ``condition`` captures a conditional gate ("do not go forward
    UNLESS the tests pass"); ``confidence`` is the classifier's own [0, 1]."""

    kind: str = NONE
    scope: str = SCOPE_ONCE
    condition: str = ""
    confidence: float = 0.0


_SYS_REVIEW_INTENT = (
    "You are the review-posture gate of a supervisory harness that sits between "
    "a user and a coding agent. Read ONE user message and decide what it means "
    "for whether the user wants to APPROVE the agent's plan before any code is "
    "written. Judge by INTENT and EFFECT, never by surface wording.\n"
    "You are given the CURRENT posture (default / hold / autonomous) and, if a "
    "plan is being held, a short summary of it — use them: the same short reply "
    "means different things depending on the state (a bare 'go' while a plan is "
    "HELD is an approval; a bare 'go' from default is nothing).\n"
    "Choose exactly one kind:\n"
    "- request_review: the user wants to see and approve the plan/approach "
    "BEFORE code — 'show me the plan first', 'discuss before you build', 'don't "
    "just start coding', 'run it by me'. RESOLVE NEGATION: 'do NOT go forward' / "
    "'do not start yet' is request_review, NOT approve. A CONDITIONAL — 'do not "
    "go forward UNLESS the tests pass', 'only proceed once I say so' — is "
    "request_review WITH the condition captured; it is NEVER an approval, even "
    "though it contains 'proceed'/'go forward'.\n"
    "- approve: the user is approving the plan that is being held so code can "
    "start — 'yes', 'go ahead', 'looks good, build it', 'approved', 'that plan "
    "works', and bare affirmatives / continuations WHEN A PLAN IS HELD: "
    "'proceed', 'go', 'continue', 'do it', 'ship it', 'lgtm', 'ok', 'sounds "
    "good'. Only choose this when a plan is actually being held AND the message "
    "assents — an affirmative that is NOT a negation, NOT a conditional, and NOT "
    "a fresh review request. The held plan IS the thing being assented to: a "
    "short 'proceed' right after saddle heralded 'plan held — awaiting your "
    "approval' is that approval, not a new instruction.\n"
    "- autonomy_grant: the user hands over the wheel — proceed on your own "
    "judgment without stopping for each plan approval: 'go with your "
    "recommendations', 'your call', 'use your best judgment', 'don't ask me, "
    "just do it', 'proceed on your own from here'.\n"
    "- intervene: the user takes the wheel back or halts — 'stop', 'wait', "
    "'hold on', 'pause', 'let me look first' (when NOT currently holding).\n"
    "- none: an ordinary instruction, question, or continuation that does not "
    "change who approves plans.\n"
    "ASYMMETRIC CAUTION — this is the crux, and it is STATE-DEPENDENT. In the "
    "DEFAULT posture (no plan held) a false 'approve' / 'autonomy_grant' lets "
    "unreviewed code through, so NEVER invent them on an uncertain or ambiguous "
    "message: when in doubt, choose none. But when a plan is HELD the user has "
    "been explicitly asked to approve or reject it, so a plain affirmative "
    "continuation IS the expected assent — read a bare 'proceed' / 'go' / 'yes' "
    "/ 'do it' as approve; do NOT demote a clear affirmative to none just "
    "because it is short. The negation and conditional carve-outs STILL hold in "
    "BOTH states ('do not proceed', 'only once the tests pass', 'not yet' => "
    "request_review, never approve). Getting 'request_review' wrong only costs a "
    "harmless pause, so LEAN toward request_review whenever the message "
    "plausibly asks to be in the loop. Double-negatives resolve to their real "
    "effect: 'I don't want you to not check with me' means request_review.\n"
    "Also return: scope — for request_review, 'once' (approve this one plan) or "
    "'standing' (keep me in the loop from now on); default 'once'. condition — "
    "the plain-words gate on a conditional request, else ''. confidence — your "
    "own certainty from 0 to 1.\n"
    "Respond with ONLY JSON: "
    '{"kind": "request_review|approve|autonomy_grant|intervene|none", '
    '"scope": "once|standing", "condition": "...", "confidence": 0.0}'
    + VOICE_CONTRACT
)


def held_summary(posture: HoldPosture) -> str:
    """A short, plain summary of what (if anything) is being held — fed to the
    stateful classifier so a bare 'go' can be read as approving THIS plan."""
    if posture.state == HOLD_HELD_STATE and posture.held:
        goal = str(posture.held.get("goal") or "").strip()
        cond = str(posture.held.get("condition") or "").strip()
        parts = [f"a plan is HELD awaiting the user's approval"]
        if goal:
            parts.append(f"for: {goal}")
        if cond:
            parts.append(f"(only proceed when: {cond})")
        return " ".join(parts)
    if posture.state == HOLD_AUTONOMOUS:
        return "autonomous mode is active (the user handed over the wheel)"
    return "no plan is held; the assistant is in its default posture"


def _intent_prompt(prompt: str, posture: HoldPosture, summary: str) -> str:
    return (
        f"CURRENT POSTURE: {posture.state}\n"
        f"WHAT IS HELD: {summary}\n\n"
        f"THE USER'S MESSAGE:\n{prompt}\n\n"
        "What does this message mean for whether the user wants to approve the "
        "plan before code?"
    )


async def classify_review_intent(
    prompt: str,
    ctx: Context | None = None,
    *,
    posture: HoldPosture,
    held_summary: str = "",
    caller: "LLMCaller | None" = None,
) -> ReviewIntent:
    """Classify what ``prompt`` means for the review posture — STATEFUL (fed the
    current posture + the held-plan summary so a state-dependent reply reads
    correctly).

    One bounded ``call_json`` inside the tenant fairness gate. FAIL-LOUD: a
    caller/parse failure PROPAGATES (never swallowed into a false NONE) so
    :func:`saddle.supervisor.run_stage` classifies it as an ALERT — and because
    the caller writes no posture, a failed classify leaves the prior posture
    UNCHANGED (the sticky carry-forward the design demands). Pass ``caller`` to
    bypass provider resolution (tests)."""
    text = (prompt or "").strip()
    if not text:
        return ReviewIntent(kind=NONE, confidence=1.0)
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers

        caller = build_callers(ctx)["default"]
    async with tenant_gate(ctx):
        payload = await call_json(
            caller,
            _SYS_REVIEW_INTENT,
            _intent_prompt(text, posture, held_summary or ""),
            label="hold/review-intent",
        )
    kind = str(payload.get("kind") or "").strip().lower()
    if kind not in REVIEW_INTENT_KINDS:
        kind = NONE  # a hallucinated kind can never move the posture
    scope = str(payload.get("scope") or SCOPE_ONCE).strip().lower()
    if scope not in HOLD_SCOPES:
        scope = SCOPE_ONCE
    try:
        conf = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return ReviewIntent(
        kind=kind,
        scope=scope,
        condition=str(payload.get("condition") or "").strip(),
        confidence=conf,
    )


# -- the transition table ----------------------------------------------------

@dataclass
class PostureTransition:
    """What one review-intent did to the posture. ``posture`` is the NEW posture
    (already persisted); ``prev_state`` is where it came from; ``herald`` is the
    plain on-screen message when the posture FLIPPED in a way the user should
    see (approval recorded, autonomy engaged, held-awaiting, released), else "";
    ``level`` is that herald's bubble level; ``design_id`` is a settled design's
    id when an APPROVE settled one."""

    posture: HoldPosture
    prev_state: str = HOLD_DEFAULT
    herald: str = ""
    level: str = BUBBLE_NOTICE
    design_id: str = ""


def _digest(text: str, limit: int = 280) -> str:
    """A short single-line digest of the agent's approach prose, stored with a
    held plan so the surfaces can name WHAT is held without carrying kilobytes."""
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def apply_review_intent(
    session: str,
    intent: ReviewIntent,
    *,
    goal: str = "",
    approach: str = "",
    anchor: str = "",
    posture: HoldPosture | None = None,
    ctx: Context | None = None,
    caller: "LLMCaller | None" = None,
) -> PostureTransition:
    """Apply one :class:`ReviewIntent` to the session's posture and persist it.

    The transition table (the design's contract):

    * DEFAULT + request_review        -> HOLD (record the held goal + scope)
    * AUTONOMOUS + request_review     -> HOLD (autonomy auto-cleared)
    * HOLD + request_review           -> HOLD (refresh the held goal; re-arm gate)
    * HOLD + approve                  -> settle (approved_by="user"), record the
      approved design; release to DEFAULT if the scope was 'once', stay HOLD
      (gate now open) if 'standing'
    * any + autonomy_grant            -> AUTONOMOUS
    * HOLD / AUTONOMOUS + intervene   -> DEFAULT (cleared)
    * AUTONOMOUS + none               -> AUTONOMOUS (STICKY)
    * everything else                 -> unchanged

    ``goal`` is this turn's ask (the goal to hold on a request, the goal to
    settle on an approve — the held goal is preferred when present). ``approach``
    is the agent's plan prose lifted from the transcript, settled on approval.
    Pass ``posture`` to avoid a re-read (tests / a single-read caller)."""
    if posture is None:
        posture = read_posture(session)
    prev = posture.state
    kind = intent.kind

    # --- autonomy grant: take the wheel, from any state -----------------------
    if kind == AUTONOMY_GRANT:
        new = HoldPosture(
            state=HOLD_AUTONOMOUS,
            autonomy_since_anchor=anchor or posture.autonomy_since_anchor,
        )
        write_posture(session, new)
        from saddle.voice import autonomy_engaged

        return PostureTransition(
            posture=new, prev_state=prev,
            herald=autonomy_engaged(intent.condition), level=BUBBLE_NOTICE,
        )

    # --- intervene: take the wheel back / halt --------------------------------
    if kind == INTERVENE:
        new = HoldPosture(state=HOLD_DEFAULT)
        write_posture(session, new)
        if prev in (HOLD_HELD_STATE, HOLD_AUTONOMOUS):
            from saddle.voice import hold_released

            return PostureTransition(
                posture=new, prev_state=prev,
                herald=hold_released(), level=BUBBLE_NOTICE,
            )
        return PostureTransition(posture=new, prev_state=prev)

    # --- request review: the user wants to approve the plan first -------------
    if kind == REQUEST_REVIEW:
        held = {
            "anchor": anchor,
            "goal": (goal or "").strip(),
            "approach_digest": _digest(approach),
            "condition": intent.condition,
        }
        new = HoldPosture(
            state=HOLD_HELD_STATE,
            hold_scope=intent.scope if intent.scope in HOLD_SCOPES else SCOPE_ONCE,
            held=held,
            # A fresh review request RE-ARMS the gate: any prior approval is void.
            approved_design_id="",
        )
        write_posture(session, new)
        # No herald here: the design gate itself heralds "held, awaiting
        # approval" when the agent next reaches for code — that is the moment
        # the user needs to see it, and it avoids a double announcement.
        return PostureTransition(posture=new, prev_state=prev)

    # --- approve: only meaningful while a plan is held ------------------------
    if kind == APPROVE and prev == HOLD_HELD_STATE:
        settle_goal = str(posture.held.get("goal") or "").strip() or (goal or "").strip()
        settle_approach = (approach or "").strip()
        design_id = ""
        if settle_goal and settle_approach:
            try:
                from saddle.design import settle_approach as _settle

                design = await _settle(
                    settle_goal, settle_approach, ctx or _default_ctx(),
                    approved_by="user", caller=caller,
                )
                design_id = design.id
            except Exception as exc:  # noqa: BLE001 — settle failure must not lose the approval
                print(f"hold: could not record approved design ({exc!r}); "
                      "approval still opens the gate", file=sys.stderr)
        # The approval must open the gate even when there was no plan prose to
        # settle (the user approved a verbal discussion) — use a sentinel so
        # user_approved_current() reads True in the standing case.
        approved_id = design_id or f"user-approved-{anchor or int(time.time())}"
        release = posture.hold_scope != SCOPE_STANDING
        new = HoldPosture(
            state=HOLD_DEFAULT if release else HOLD_HELD_STATE,
            hold_scope=posture.hold_scope,
            held={} if release else dict(posture.held),
            approved_design_id=approved_id,
        )
        write_posture(session, new)
        from saddle.voice import approval_recorded

        return PostureTransition(
            posture=new, prev_state=prev,
            herald=approval_recorded(design_id), level=BUBBLE_NOTICE,
            design_id=design_id,
        )

    # --- none / approve-with-nothing-held / anything else: unchanged ----------
    # AUTONOMOUS + none is the STICKY case: the posture is left as-is (still
    # autonomous), and the per-turn on-screen reminder is emitted by the caller.
    return PostureTransition(posture=posture, prev_state=prev)


# -- the composed turn step (read -> classify -> apply) ----------------------

async def review_turn(
    prompt: str,
    session: str,
    *,
    ctx: Context | None = None,
    transcript_path: str = "",
    posture: HoldPosture | None = None,
    caller: "LLMCaller | None" = None,
) -> PostureTransition:
    """The whole review step for one prompt: read the posture, classify the
    message against it, and apply the transition — the single coroutine the
    intake hook drives under a deadline via :func:`saddle.supervisor.run_bounded`.

    The agent's plan prose (for a possible settle on approval) and the turn
    anchor are lifted from the transcript. A transcript read failure is
    non-fatal — the classify + transition still run, just without an approach to
    settle (the approval still opens the gate)."""
    ctx = ctx or _default_ctx()
    if posture is None:
        posture = read_posture(session)
    approach = ""
    anchor = ""
    if transcript_path:
        try:
            from saddle.transcript import latest_turn

            turn = latest_turn(transcript_path)
            approach = turn.approach
            anchor = turn.anchor
        except Exception as exc:  # noqa: BLE001 — transcript read must not wedge the step
            print(f"hold: transcript read error ({exc!r}); "
                  "classifying without the plan prose", file=sys.stderr)
    intent = await classify_review_intent(
        prompt, ctx, posture=posture, held_summary=held_summary(posture),
        caller=caller,
    )
    return await apply_review_intent(
        session, intent, goal=prompt, approach=approach, anchor=anchor,
        posture=posture, ctx=ctx, caller=caller,
    )


# -- the pure decision helpers the doctrine gate reads -----------------------

def design_in_play(turn, posture: HoldPosture) -> bool:
    """Is a design 'in play' for this code edit — i.e. is there a plan the user's
    hold should gate on?

    True when the turn carries approach prose, OR a plan is already held, OR the
    posture is an active HOLD. FAIL-CLOSED: under an active hold this returns True
    unconditionally, so a no-plan straight-to-code edit cannot dodge the gate by
    simply never stating a plan (stating and getting the plan approved IS the way
    through)."""
    if posture.state == HOLD_HELD_STATE:
        return True
    if posture.held:
        return True
    approach = getattr(turn, "approach", "") if turn is not None else ""
    return bool(approach and approach.strip())


def user_approved_current(posture: HoldPosture, turn=None, ctx: Context | None = None) -> bool:
    """Has the user approved the plan the current hold is gating?

    v1 (the 'once' model): true when an approval has been recorded
    (``approved_design_id`` is set). A fresh REQUEST_REVIEW clears it, so each
    explicit "review my plan" re-arms the gate. ``turn`` / ``ctx`` are accepted
    for a future per-plan match (approving plan A must not open the gate for a
    different plan B) but are unused in v1."""
    return bool(posture.approved_design_id)
