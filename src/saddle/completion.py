"""The completion gate — did the USER'S ACTUAL GOAL get finished?

Founding incident (user, 2026-07-03): a goal auto-cleared as "achieved"
because the agent's closing summary matched the goal's ENUMERATED items,
while the goal's broader clauses ("play test every feature", "AAA quality")
were still open. The user's directive: when a goal is set, saddle must judge
completion against the goal AS THE USER MEANT IT — interpreted at a high
level (the end result the user wants), never narrowed to the sub-list the
agent echoed back — and against best practice with edge cases considered.

So, at turn end, when the agent's reply READS as "finished", this stage
audits that claim against:

* the user prompt that opened the turn (the immediate ask),
* the still-open recorded asks from recent intakes (the ledger's memory of
  what the user requested and never withdrew),
* an explicit instruction to interpret the goal broadly.

An unjustified claim is a loud alert naming exactly what is still missing —
delivered to the human via the turn-end bubble and to the agent via the next
turn's drain, so the overclaim is corrected instead of compounding. A reply
that makes no completion claim is silent (no second LLM call, no noise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.llm.json_tools import extract_json_text
from saddle.voice import VOICE_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller

# How many open asks / recent intakes ground the audit. Recency-bounded so a
# long-lived project ledger doesn't drown the judgment in stale items.
_MAX_OPEN_ASKS = 30
_MAX_INTAKES = 3

_SYS_COMPLETION = (
    "You are the completion gate of a supervisory harness. You are given the "
    "user's current request, the still-open recorded asks from the project's "
    "ledger, and the agent's reply for this turn. Decide two things:\n"
    "1. claims_done — does the reply state or clearly imply that the user's "
    "goal (not merely one sub-task) is finished? ALSO judge "
    "opening_declares_finished: does the reply's FIRST sentence or headline "
    "alone (ignore everything after it) declare a finished/verified state "
    "('all N issues fixed and verified', 'everything requested is done')?\n"
    "2. If it does: is that claim JUSTIFIED? Interpret the goal AS THE USER "
    "MEANT IT — at a high level, the end result they want. Never narrow the "
    "goal to the sub-list the reply enumerates: a broad clause like 'test "
    "everything', 'fix all', or a quality bar ('production ready', 'AAA') is "
    "part of the goal even when the reply's list is complete. Hold the claim "
    "to best practice: work is complete only when verified, edge cases "
    "considered, and nothing the user asked for is left as 'next steps'.\n"
    "A reply that reports progress and plainly says what remains makes "
    "claims_done false — do not flag honest status reports. But judge "
    "opening_declares_finished on the opening ALONE: an automated goal "
    "checker reads the opening, so a finished-sounding opening counts even "
    "when later paragraphs list further work.\n\n"
    "Also judge two more things:\n"
    "- goal_active: is there an unfinished multi-step goal the agent is "
    "responsible for DRIVING (an explicit goal, a fix-everything directive, "
    "recorded open asks)? A one-off question or a purely conversational "
    "exchange is NOT an active goal.\n"
    "- awaiting_user: does the reply end genuinely blocked on the USER — a "
    "direct question, a decision only the user can make, or an explicit "
    "hand-back? Merely listing remaining work, waiting on background jobs, "
    "or narrating status is NOT awaiting the user.\n\n"
    "Respond with ONLY JSON: {\"claims_done\": true|false, "
    '"opening_declares_finished": true|false, '
    '"complete": true|false, "goal_active": true|false, '
    '"awaiting_user": true|false, '
    '"missing": ["<each thing the goal still needs, '
    "one line each, plain words>\"]}. missing lists what the FULL goal still "
    "needs whenever it is not complete, even for an honest status report."
    + VOICE_CONTRACT
)


@dataclass
class CompletionVerdict:
    """The gate's judgment of this turn's completion claim."""

    claims_done: bool = False
    opening_declares_finished: bool = False
    complete: bool = True
    goal_active: bool = False
    awaiting_user: bool = False
    missing: list[str] = field(default_factory=list)

    @property
    def should_keep_working(self) -> bool:
        """True when the goal-keeper should push the agent back to work: a
        driving goal is active, it is not complete, and the agent is not
        genuinely blocked on the user (user directive 2026-07-03: 'if saddle
        catches it, it should drive you back into working')."""
        return (
            self.goal_active
            and (not self.complete or bool(self.missing))
            and not self.awaiting_user
        )

    @property
    def overclaim(self) -> bool:
        """True when the reply — or its OPENING alone (what an automated goal
        checker reads) — says 'finished' while the goal, read broadly, is
        not. The founding incident was a finished-sounding opening above an
        honest remainder list; the deterministic OR here means the verdict
        cannot hinge on how a judge model weighs that nuance."""
        unfinished = (not self.complete) or bool(self.missing)
        return (self.claims_done or self.opening_declares_finished) and unfinished


def _ledger_context(ctx: Context) -> tuple[list[str], list[str]]:
    """(open asks, recent intake summaries) — the ledger's memory of what the
    user asked for. Best-effort: an unreadable store yields empty context and
    the audit judges from the reply + prompt alone (weaker, never wrong-er)."""
    from saddle.store import get_store

    asks: list[str] = []
    summaries: list[str] = []
    try:
        store = get_store()
        for item in store.todos(ctx)[:_MAX_OPEN_ASKS]:
            asks.append(f"[{item.kind}] {item.ask}")
        for intake in store.list_intakes(ctx, limit=_MAX_INTAKES):
            if intake.summary:
                summaries.append(intake.summary)
    except Exception:  # noqa: BLE001 — context is a grounding aid, not a gate
        pass
    return asks, summaries


def _prompt(goal: str, reply: str, asks: list[str], summaries: list[str]) -> str:
    asks_block = "\n".join(f"- {a}" for a in asks) or "(none recorded)"
    sum_block = "\n".join(f"- {s}" for s in summaries) or "(none)"
    return (
        f"THE USER'S CURRENT REQUEST (this turn):\n{goal or '(none)'}\n\n"
        f"STILL-OPEN RECORDED ASKS (project ledger):\n{asks_block}\n\n"
        f"RECENT REQUEST SUMMARIES:\n{sum_block}\n\n"
        f"THE AGENT'S REPLY THIS TURN:\n{reply}\n\n"
        "Judge the reply's completion claim against the goal as the user "
        "meant it."
    )


async def audit_completion(
    goal: str,
    reply: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
) -> CompletionVerdict:
    """Judge whether ``reply`` overclaims completion of the user's goal.

    One bounded LLM call; a reply with no text is trivially no-claim. Raises
    on caller failure so :func:`saddle.supervisor.run_stage` classifies it
    loudly — a completion gate that cannot run must never read as a pass."""
    text = (reply or "").strip()
    if not text:
        return CompletionVerdict()
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers

        caller = build_callers(ctx)["default"]
    asks, summaries = _ledger_context(ctx)
    raw = await caller(
        _SYS_COMPLETION, _prompt(goal, text, asks, summaries),
        json_mode=True, label="completion/gate",
    )
    doc = json.loads(extract_json_text(raw))
    if not isinstance(doc, dict):
        raise ValueError("completion gate returned no JSON object")
    return CompletionVerdict(
        claims_done=bool(doc.get("claims_done")),
        opening_declares_finished=bool(doc.get("opening_declares_finished")),
        complete=bool(doc.get("complete", True)),
        goal_active=bool(doc.get("goal_active")),
        awaiting_user=bool(doc.get("awaiting_user")),
        missing=[str(m).strip() for m in doc.get("missing") or [] if str(m).strip()],
    )
