"""Stage 2 — intent drift: this prompt vs the project's own settled state.

WHY THIS EXISTS
---------------
Stage 1 (intake) asks "is the prompt understood?"; Stage 3 (design) asks "is the
proposed design sound against UNIVERSAL best practice?". Between them sits a
distinct drift axis this module owns: does THIS prompt pull against what THIS
project has ALREADY settled — a design it committed to, a decision it closed, the
focus it declared?

Stage 2 is three axes; two already exist elsewhere and one is the gap this module
closes:

* **cross-project** — already built: :func:`saddle.intake._classify_scope` flags
  a TASK that would modify a DIFFERENT project than the focus.
* **pick-drift** — already built: transcript replay + the
  :class:`saddle.dialog.IntentTracker` surface "you committed a), the agent acted
  on b)".
* **project / design / history** — THIS module: retrieve the project's prior
  :class:`~saddle.models.Design` s and harvested decisions/lessons from the DKB
  and compare the new ask against them. Does it CONTRADICT a settled design,
  RE-OPEN a closed decision, or CREEP past the project's stated focus?

The verdict is ANNOTATION, never refusal (refusing is the guard's job): a NOTICE
describing any divergence, escalated to an ALERT for a HARD pull — a direct
contradiction of a settled design or the re-opening of a closed decision. The
human confirms or corrects through the dialog channel.

WHY IT DROPS THE GLOBAL SEED CORPUS
-----------------------------------
Retrieval is scope-laddered (global ∪ tenant ∪ project), but Stage 2 keeps only
THIS tenant's own entries (:func:`_project_scoped`). Divergence from UNIVERSAL
wisdom — "no band-aids", "fail loud" — is Stage 3's design audit, not Stage 2's.
Stage 2 is strictly "this project's own settled state", so feeding it the global
seed corpus would blur the two stages into one.

FAIL-LOUD
---------
A brand-new project with no designs and no decisions has nothing to pull against,
so :func:`history_drift` returns a checked, empty report WITHOUT an LLM call — the
fast path. Otherwise the single classify call runs inside the tenant gate, and any
failure (timeout, provider outage, a malformed reply) PROPAGATES to
:func:`saddle.supervisor.run_stage`, which classifies and bubbles it as an ALERT.
This module never swallows a failed check into a silent "looked fine".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.llm.json_tools import call_json
from saddle.llm.pool import tenant_gate
from saddle.models import BUBBLE_ALERT, BUBBLE_NOTICE

if TYPE_CHECKING:  # pragma: no cover
    from saddle.dkb import DKB
    from saddle.llm.protocol import LLMCaller
    from saddle.models import Design, Knowledge

_log = logging.getLogger("saddle.intent")


# -- the divergence taxonomy -------------------------------------------------

CONTRADICTS_DESIGN = "contradicts_design"  # pulls against a design already settled
REOPENS_DECISION = "reopens_decision"      # re-opens a decision already closed
SCOPE_CREEP = "scope_creep"                # creeps past the project's stated focus
DIVERGENCE_KINDS: frozenset[str] = frozenset(
    {CONTRADICTS_DESIGN, REOPENS_DECISION, SCOPE_CREEP}
)

# The HARD pulls: a direct contradiction or a re-opened decision is an ALERT (it
# breaches a standing commitment); scope-creep is a NOTICE (worth a look, not a
# breach). The split is what makes ``IntentReport.level`` per-finding, not flat.
_HARD_KINDS: frozenset[str] = frozenset({CONTRADICTS_DESIGN, REOPENS_DECISION})

# Compact, human-readable headline per kind for the rendered section.
_KIND_LABEL: dict[str, str] = {
    CONTRADICTS_DESIGN: "CONTRADICTS A SETTLED DESIGN",
    REOPENS_DECISION: "RE-OPENS A CLOSED DECISION",
    SCOPE_CREEP: "CREEPS PAST THE PROJECT'S FOCUS",
}


# -- what a single divergence is ---------------------------------------------

@dataclass(frozen=True)
class IntentDivergence:
    """One way this prompt pulls against the project's settled state.

    ``kind`` is one of :data:`DIVERGENCE_KINDS`; ``what`` is the pull in one line
    ("re-opens the in-memory-vs-redis cache decision"); ``why`` is the evidence
    (which settled design / decision it collides with and how); ``ref`` is an
    optional locator (a design id, a decision's title) the human can pull up.
    """

    kind: str
    what: str
    why: str = ""
    ref: str = ""

    @property
    def hard(self) -> bool:
        """A hard pull (contradiction / re-opened decision) -> the report alerts."""
        return self.kind in _HARD_KINDS

    def render(self) -> str:
        """One readable block: a tagged headline, the pull, and the evidence."""
        head = f"• {_KIND_LABEL.get(self.kind, self.kind.upper())}"
        if self.ref:
            head += f" [{self.ref}]"
        lines = [f"{head}: {self.what}"]
        if self.why:
            lines.append(f"    ↳ {self.why}")
        return "\n".join(lines)


# -- the Stage 2 history-axis verdict for one prompt -------------------------

@dataclass
class IntentReport:
    """The Stage 2 history-axis verdict for one prompt.

    ``divergences`` is empty when the prompt builds cleanly on the project's
    settled state. ``checked`` records that the comparison actually RAN — a
    brand-new project with nothing to compare is still ``checked=True`` (it ran
    and found nothing, distinct from could-not-run, which is a raised failure
    :func:`saddle.supervisor.run_stage` turns into an ALERT). ``considered`` is
    how many settled designs + decisions were weighed, for the audit trail.
    """

    divergences: list[IntentDivergence] = field(default_factory=list)
    checked: bool = False
    considered: int = 0

    @property
    def has_drift(self) -> bool:
        return bool(self.divergences)

    @property
    def level(self) -> str:
        """ALERT if ANY divergence is hard (a breached commitment); else NOTICE."""
        return BUBBLE_ALERT if any(d.hard for d in self.divergences) else BUBBLE_NOTICE

    def sections(self) -> list[str]:
        """The human-readable section(s) for the bubble / agent-context — empty
        when nothing drifted, so the stage stays silent on a clean compare."""
        if not self.divergences:
            return []
        head = "this prompt pulls against what the project has already settled:"
        body = "\n".join(d.render() for d in self.divergences)
        return [f"{head}\n{body}"]


# -- the classify prompt + its context blocks --------------------------------

_SYS_HISTORY = (
    "You are the intent-continuity gate of an assistant DEDICATED to one "
    "project. You are given the project's SETTLED state — designs it has already "
    "committed to and decisions/lessons it has already closed — and a NEW user "
    "request. Decide whether CARRYING OUT the new request would pull AGAINST that "
    "settled state. Judge by INTENT and EFFECT, not by wording.\n"
    "Flag exactly these pulls:\n"
    "- contradicts_design: doing it would undo, bypass, or directly conflict with "
    "a design the project already settled.\n"
    "- reopens_decision: it re-litigates a decision already closed — picks the "
    "option that was rejected, or re-asks a settled question.\n"
    "- scope_creep: it pushes the project past its stated focus / boundary.\n"
    "Building ON, refining, fixing, or extending the settled state is NOT a "
    "divergence — neither is brand-new work that does not conflict. Only flag a "
    "GENUINE pull. When unsure, prefer ALIGNED and flag nothing: a false alarm "
    "every turn trains the human to ignore the channel.\n"
    "SELF-DIRECTED WORK IS NOT A DIVERGENCE. This assistant develops and maintains "
    "its OWN code, gates, prompts, behavior, and configuration; a request to add, "
    "refine, tune, fix, or configure the assistant ITSELF is the project's own "
    "legitimate work — never scope_creep, never a contradiction. New feature work "
    "on the project is fine even when no existing settled design covers it: the "
    "settled designs are a SUBSET of the project's work, not its outer boundary, so "
    "'not covered by a settled design' is NOT scope_creep. And the USER holds "
    "authority over the settled state — a user EXPLICITLY directing a change to a "
    "prior decision is the user re-deciding, which is legitimate; flag "
    "reopens_decision ONLY when the request appears to contradict a closed decision "
    "UNAWARES (silently, as if it had been forgotten), never when the user is "
    "deliberately directing the change.\n"
    "For each genuine pull give: kind (one of the three above), what (the pull in "
    "one clear line), why (which settled design/decision it collides with and "
    "how), ref (a short locator — the design's id or the decision's title — or "
    '"").\n\n'
    "Respond with ONLY a JSON object: "
    '{"divergences": [{"kind": "...", "what": "...", "why": "...", '
    '"ref": "..."}]}. '
    "Return an empty list when the request aligns."
)


def _format_designs(designs: list["Design"]) -> str:
    """One line per settled design (id, status, the gist) + its approach."""
    if not designs:
        return "(none yet)"
    lines: list[str] = []
    for d in designs:
        head = (d.summary or d.ask or d.body[:120]).strip()
        lines.append(f"- [{d.id or 'design'}] ({d.status}) {head}")
        if d.approach.strip():
            lines.append(f"    approach: {d.approach.strip()}")
    return "\n".join(lines)


def _format_decisions(items: list["Knowledge"]) -> str:
    """One line per closed decision / lesson (title, kind, the body)."""
    if not items:
        return "(none yet)"
    return "\n".join(
        f"- [{k.title.strip()}] ({k.kind}) {k.body.strip()}" for k in items
    )


def _history_prompt(prompt: str, designs_block: str, decisions_block: str) -> str:
    return (
        f"SETTLED DESIGNS:\n{designs_block}\n\n"
        f"CLOSED DECISIONS / LESSONS:\n{decisions_block}\n\n"
        f"NEW REQUEST:\n{prompt}\n\n"
        "Does carrying out the new request pull against the settled state above?"
    )


# -- retrieval scoping + tolerant parse --------------------------------------

def _project_scoped(
    hits: list[tuple["Knowledge", float]], ctx: Context
) -> list["Knowledge"]:
    """Keep only THIS tenant's own settled entries — drop the global seed corpus.

    Stage 2 is "this project's own settled state"; divergence from universal seed
    wisdom is Stage 3's design audit. A global seed has an empty ``scope_tenant``,
    so matching the tenant drops it while keeping both tenant-wide and
    project-specific lessons/decisions (both ARE this project's own history)."""
    return [k for k, _ in hits if k.scope_tenant and k.scope_tenant == ctx.tenant]


def _divergences_from(raw: object) -> list[IntentDivergence]:
    """Build divergences from the model's array, dropping malformed / unknown-kind
    rows. This is a tolerant PARSE of a present reply — distinct from a failed
    CALL, which propagates. Only the three known kinds with a non-empty ``what``
    survive, so a hallucinated kind can never reach the human."""
    out: list[IntentDivergence] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind", "")).strip().lower()
        what = str(entry.get("what", "")).strip()
        if kind not in DIVERGENCE_KINDS or not what:
            continue
        out.append(
            IntentDivergence(
                kind=kind,
                what=what,
                why=str(entry.get("why", "")).strip(),
                ref=str(entry.get("ref", "")).strip(),
            )
        )
    return out


# -- the entrypoint ----------------------------------------------------------

async def history_drift(
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    dkb: "DKB | None" = None,
    design_limit: int = 12,
    retrieve_k: int = 6,
) -> IntentReport:
    """Does ``prompt`` pull against what this project has ALREADY settled?

    Retrieves the project's prior designs and its most relevant closed
    decisions/lessons from the DKB, then asks the model whether carrying out the
    new request would contradict a settled design, re-open a closed decision, or
    creep past the project's focus. Returns an :class:`IntentReport` — annotation
    only; Stage 2 never refuses.

    Fast path: a project with NO settled designs and NO decisions has nothing to
    pull against, so it returns a checked, empty report WITHOUT an LLM call — no
    false drift on a clean slate, and no needless provider round-trip. Otherwise
    the single classify call runs inside ``ctx``'s tenant gate, and any failure
    PROPAGATES (fail-loud) for :func:`saddle.supervisor.run_stage` to classify and
    bubble. Pass ``caller`` / ``dkb`` to bypass resolution (tests).
    """
    text = (prompt or "").strip()
    ctx = ctx or _default_ctx()
    if not text:
        return IntentReport(checked=True)

    if dkb is None:
        from saddle.dkb import get_dkb
        dkb = get_dkb()

    # Retrieval is sync (sqlite + embed). Run it OFF the event loop and OUTSIDE
    # the tenant gate — it is not an LLM generation, so it must not hold a
    # provider fairness slot while it touches the db.
    designs = await asyncio.to_thread(dkb.list_designs, ctx, limit=design_limit)
    hits = await asyncio.to_thread(dkb.search_knowledge, ctx, text, k=retrieve_k)
    decisions = _project_scoped(hits, ctx)

    considered = len(designs) + len(decisions)
    if considered == 0:
        # Brand-new project: nothing settled to pull against. It RAN (checked) and
        # found nothing — the clean-slate fast path, no LLM call.
        return IntentReport(checked=True)

    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    async with tenant_gate(ctx):
        payload = await call_json(
            caller,
            _SYS_HISTORY,
            _history_prompt(
                text, _format_designs(designs), _format_decisions(decisions)
            ),
            label="intent/history",
        )
    return IntentReport(
        divergences=_divergences_from(payload.get("divergences")),
        checked=True,
        considered=considered,
    )
