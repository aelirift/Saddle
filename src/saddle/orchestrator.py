"""Layer 1 ⇄ Layer 2 orchestration — the whole prompt-to-designs path.

One user message comes in; this module drives it end to end, tying intake
(Layer 1) to best-practice design (Layer 2):

  1. decompose  — Layer 1 (:func:`saddle.intake.decompose`) turns the prompt
                  into discrete, classified items, exhaustively, nothing dropped.
  2. promote    — every DIRECTIVE-kind item becomes a standing rule persisted at
                  project scope (:func:`saddle.llm.policy.promote_directive`), so
                  the user's binding preferences ("no band-aids", "no hard-coding")
                  are honored on every FUTURE design without being restated.
  3. fold       — the actionable asks (task / decision / question) are folded
                  HIERARCHICALLY into the smallest set of coherent design GOALS:
                  related asks become ONE higher-level design instead of many
                  small disconnected pieces — but EVERY ask stays covered.
                  Coverage is enforced in code: an ask the fold model forgets
                  becomes its own standalone goal rather than vanishing. This is
                  the same "miss nothing" guarantee intake's coverage audit gives,
                  applied to folding.
  4. design     — each folded goal runs through Layer 2 (:func:`saddle.design.
                  design_for`): diagnose → retrieve → design → audit → harvest,
                  under the now-promoted binding directives.

The result is an :class:`Orchestration`: the recorded intake, the directives
promoted on this run, and the folded goals, each carrying the design it produced.

Every LLM call here resolves its provider/policy from the :class:`~saddle.context.Context`
and runs inside that tenant's fairness gate. The orchestrator itself holds no
gate — decompose, the fold call, and each design_for each take and release the
gate around their own work, so there is never a re-entrant acquire (which would
deadlock a cap-1 tenant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.design import design_for, format_design
from saddle.dkb import DKB, get_dkb
from saddle.intake import classify_directive_durability, decompose, format_intake
from saddle.llm import policy
from saddle.llm.json_tools import call_json
from saddle.llm.pool import tenant_gate
from saddle.models import (
    CONTEXT,
    DECISION,
    DIRECTIVE,
    QUESTION,
    TASK,
    Design,
    Intake,
    Item,
)

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller
    from saddle.store import Store

_log = logging.getLogger("saddle.orchestrator")

# The item kinds that are actual asks to design FOR. Directives are promoted to
# standing rules and applied as constraints (not designed); context is folded in
# as background. Question is design-worthy here: in a design harness a question
# is a design fork ("FTS or vector?"), answered AS a design.
_DESIGN_KINDS: frozenset[str] = frozenset({TASK, DECISION, QUESTION})

_SYS_FOLD = (
    "You are the fold stage of a design harness. You are given a numbered list "
    "of ASKS pulled from ONE user's message — each an action to perform, a "
    "decision to make, or a question to resolve — plus any BACKGROUND. Your job "
    "is to fold related asks together into the SMALLEST set of coherent design "
    "GOALS: asks that belong to a single design MUST become ONE higher-level "
    "goal rather than many small disconnected pieces. Think structurally — group "
    "by the design they share, not by surface wording.\n\n"
    "Hard rule: cover EVERYTHING. Every ask must belong to exactly one goal; "
    "never drop an ask, never leave one out. An ask that genuinely stands on its "
    "own is its own goal.\n\n"
    "For each goal give:\n"
    "- title: a short name for the goal.\n"
    "- statement: a clear, self-contained statement of the WHOLE goal, written "
    "so a designer can design all of it from this text alone — fold the covered "
    "asks (and any relevant background) into one coherent brief.\n"
    "- covers: the list of ask NUMBERS (from the ASKS list) this goal folds in.\n\n"
    "Respond with ONLY a JSON object: "
    '{"goals": [{"title": "...", "statement": "...", "covers": [1, 2]}]}'
)


@dataclass
class Goal:
    """One folded design goal — a coherent unit covering one or more asks.

    ``statement`` is the self-contained brief handed to Layer 2; ``item_ids``
    traces it back to the intake items it folds; ``design`` is filled once
    :func:`saddle.design.design_for` has run.
    """

    title: str
    statement: str
    item_ids: list[str] = field(default_factory=list)
    design: Design | None = None


@dataclass
class Orchestration:
    """The full Layer 1 → Layer 2 result for one prompt."""

    intake: Intake
    promoted: list[str] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)

    @property
    def designs(self) -> list[Design]:
        return [g.design for g in self.goals if g.design is not None]


def _short(text: str, n: int = 60) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _background_block(intake: Intake) -> str:
    """Summary + context items — the non-ask material a goal brief can draw on."""
    parts: list[str] = []
    if intake.summary:
        parts.append(f"SUMMARY: {intake.summary}")
    for it in intake.items:
        if it.kind == CONTEXT and it.ask.strip():
            parts.append(f"- {it.ask}")
    return "\n".join(parts) if parts else "(none)"


def _fold_prompt(asks: list[Item], background: str) -> str:
    ask_lines = "\n".join(f"{i + 1}. [{it.kind}] {it.ask}" for i, it in enumerate(asks))
    return (
        f"ASKS:\n{ask_lines}\n\n"
        f"BACKGROUND:\n{background}\n\n"
        "Fold these asks into the smallest set of coherent design goals. Cover "
        "every ask exactly once."
    )


def _valid_indices(raw: object, n: int) -> list[int]:
    """Coerce a model's 1-based ``covers`` list into valid 0-based indices."""
    out: list[int] = []
    seen: set[int] = set()
    if isinstance(raw, list):
        for x in raw:
            try:
                idx = int(x) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def _goal_from_items(statement: str, items: list[Item], *, title: str = "") -> Goal:
    return Goal(
        title=title or _short(statement),
        statement=statement,
        item_ids=[it.id for it in items if it.id],
    )


async def _fold(
    caller: "LLMCaller", ctx: Context, asks: list[Item], background: str
) -> list[Goal]:
    """Fold asks into coherent goals; guarantee every ask is covered.

    The LLM proposes the grouping; code enforces the contract. Any ask the model
    leaves uncovered is recovered as its own standalone goal, so folding can
    consolidate but can never drop.
    """
    if not asks:
        return []
    if len(asks) == 1:
        return [_goal_from_items(asks[0].ask, asks)]

    async with tenant_gate(ctx):
        payload = await call_json(
            caller, _SYS_FOLD, _fold_prompt(asks, background), label="orchestrate/fold"
        )

    goals: list[Goal] = []
    covered: set[int] = set()
    raw_goals = payload.get("goals")
    if isinstance(raw_goals, list):
        for g in raw_goals:
            if not isinstance(g, dict):
                continue
            statement = str(g.get("statement", "")).strip()
            if not statement:
                continue
            idxs = _valid_indices(g.get("covers"), len(asks))
            members = [asks[i] for i in idxs]
            goals.append(
                _goal_from_items(
                    statement, members, title=str(g.get("title", "")).strip()
                )
            )
            covered.update(idxs)

    # Coverage guarantee — recover any ask the fold dropped as its own goal.
    missed = [i for i in range(len(asks)) if i not in covered]
    for i in missed:
        goals.append(_goal_from_items(asks[i].ask, [asks[i]]))
    if missed:
        _log.warning(
            "fold left %d ask(s) uncovered for %s — recovered as standalone goals",
            len(missed), ctx.key,
        )
    if not goals:
        # Fold returned nothing usable at all — fall back to one goal per ask so
        # the whole prompt is still designed (never silently empty).
        goals = [_goal_from_items(it.ask, [it]) for it in asks]
    return goals


async def _promote_directives(
    ctx: Context, intake: Intake, scope: str, caller: "LLMCaller"
) -> list[str]:
    """Persist DURABLE directive items as standing rules; return the new ones.

    Closes design_issues Gap 4: not every ``directive``-kind ask is a standing
    rule — many are task-local instructions ("respond with only JSON", "make THIS
    design game-agnostic") that, persisted as policy, would be enforced on every
    future task out of context. So directive items are first classified for
    durability (:func:`saddle.intake.classify_directive_durability`); only the
    STANDING ones are promoted, and the task-local ones are logged (never silently
    dropped) so the skip is visible. The classifier fails safe toward not
    promoting, because over-promotion is the harm this gate exists to prevent."""
    directive_items = [
        it for it in intake.items if it.kind == DIRECTIVE and it.ask.strip()
    ]
    if not directive_items:
        return []
    verdict = await classify_directive_durability(caller, directive_items, ctx)
    standing, task = verdict["standing"], verdict["task"]
    if task:
        _log.info(
            "directive durability: %d task-local instruction(s) NOT promoted to "
            "%s policy (would pollute future tasks): %s",
            len(task), scope, "; ".join(it.ask[:60] for it in task),
        )
    promoted: list[str] = []
    for it in standing:
        text = it.ask.strip()
        try:
            if policy.promote_directive(ctx, text, scope=scope):
                promoted.append(text)
        except Exception as exc:  # noqa: BLE001 — one bad rule must not abort the run
            _log.warning("could not promote directive %r: %s", text, exc)
    if promoted:
        _log.info(
            "promoted %d standing directive(s) to %s scope for %s",
            len(promoted), scope, ctx.key,
        )
    return promoted


async def orchestrate(
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    store: "Store | None" = None,
    dkb: DKB | None = None,
    persist: bool = True,
    run_designs: bool = True,
    promote_scope: str = "project",
    max_audits: int = 2,
    retrieve_k: int = 8,
) -> Orchestration:
    """Drive ``prompt`` through Layer 1 + Layer 2 for ``ctx``.

    decompose → promote directives → fold the actionable asks into coherent
    design goals (nothing dropped) → run Layer 2 on each goal under the binding
    directives. Persists the intake, the promoted rules, and every design unless
    ``persist=False``. Set ``run_designs=False`` to stop after folding (inspect
    the goals without paying for the design pipeline). Pass ``caller`` to bypass
    provider resolution (tests).
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("cannot orchestrate an empty prompt")
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    # 1. Layer 1 — decompose (manages its own gate + persistence).
    intake = await decompose(text, ctx, caller=caller, store=store, persist=persist)

    # 2. Promote DURABLE directive-kind items to standing rules BEFORE designing,
    #    so the design audit enforces them on this very run (promote busts the
    #    cache). Task-local instructions are filtered out (Gap 4) so they never
    #    become forever-rules.
    promoted = await _promote_directives(ctx, intake, promote_scope, caller)

    # 3. Fold the actionable asks into coherent design goals (coverage enforced).
    asks = [it for it in intake.items if it.kind in _DESIGN_KINDS]
    goals = await _fold(caller, ctx, asks, _background_block(intake))
    _log.info(
        "folded %d ask(s) into %d design goal(s) for %s",
        len(asks), len(goals), ctx.key,
    )

    # 4. Design each goal through Layer 2 under one shared directive snapshot.
    if run_designs and goals:
        dkb = dkb or get_dkb()
        binding = policy.directives(ctx)
        for i, g in enumerate(goals, 1):
            _log.info(
                "designing goal %d/%d for %s: %s", i, len(goals), ctx.key, g.title
            )
            g.design = await design_for(
                g.statement, ctx,
                caller=caller, dkb=dkb, directives=binding,
                persist=persist, max_audits=max_audits, retrieve_k=retrieve_k,
                intake_id=intake.id,
            )

    return Orchestration(intake=intake, promoted=promoted, goals=goals)


def format_orchestration(orc: Orchestration) -> str:
    """Human-readable rendering of a full orchestration for the CLI."""
    lines: list[str] = [format_intake(orc.intake)]
    if orc.promoted:
        lines.append(f"\npromoted {len(orc.promoted)} standing directive(s):")
        lines.extend(f"  - {d}" for d in orc.promoted)
    lines.append(f"\nfolded into {len(orc.goals)} design goal(s):")
    for i, g in enumerate(orc.goals, 1):
        lines.append(f"\n{'=' * 70}\ngoal {i}/{len(orc.goals)}: {g.title}\n{'=' * 70}")
        if g.design is not None:
            lines.append(format_design(g.design))
        else:
            lines.append(f"(not designed)\n{g.statement}")
    return "\n".join(lines)
