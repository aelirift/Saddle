"""Layer 1 — prompt intake & understanding.

The first layer of saddle's harness. A user message — which may be long and
braid many distinct asks together — comes in; :func:`decompose` turns it into
a recorded :class:`~saddle.models.Intake`: a clear summary plus a flat,
exhaustive list of discrete :class:`~saddle.models.Item` s, each classified
(question / task / directive / context / decision) and restated so it stands
on its own. The actionable subset (task + directive) becomes the persistent
todo list via the store.

The contract that matters most: **no ask may be dropped.** A single
itemizing pass can silently miss an item buried in a wall of text, so
decompose runs two-pass — an itemize pass, then a dedicated coverage-audit
pass whose only job is to scan the original against what was extracted and
surface anything missed. The audit repeats (bounded) until it reports
complete. Coverage stats land in ``intake.meta``.

Every LLM call resolves its provider/policy from the :class:`~saddle.context.Context`
and runs inside that tenant's fairness gate, so one tenant's intake burst
can't starve another's.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from saddle.context import Context, default as _default_ctx
from saddle.focus import focus_descriptor, focus_project
from saddle.llm.json_tools import call_json as _call_json
from saddle.llm.pool import tenant_gate
from saddle.models import (
    CONTEXT,
    DECISION,
    DIRECTIVE,
    ITEM_KINDS,
    QUESTION,
    TASK,
    Intake,
    Item,
)
from saddle.store import get_store

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller
    from saddle.store import Store

_log = logging.getLogger("saddle.intake")

# Map sloppy / synonym kind labels onto the canonical taxonomy so a model
# that says "todo" or "rule" instead of "task"/"directive" still classifies
# correctly instead of silently falling through to context.
_KIND_SYNONYMS: dict[str, str] = {
    "todo": TASK, "action": TASK, "do": TASK, "request": TASK,
    "rule": DIRECTIVE, "preference": DIRECTIVE, "constraint": DIRECTIVE,
    "standing": DIRECTIVE, "policy": DIRECTIVE,
    "ask": QUESTION, "q": QUESTION, "query": QUESTION,
    "info": CONTEXT, "background": CONTEXT, "fyi": CONTEXT, "note": CONTEXT,
    "choice": DECISION, "fork": DECISION, "option": DECISION,
}

_SYSTEM_ITEMIZE = (
    "You are the intake layer of an assistant. Read the user's message — it "
    "may be long and tangle many distinct asks together — and decompose it "
    "EXHAUSTIVELY into a flat list of discrete items. Do not miss anything: "
    "every span that carries a question, an action to perform, a standing "
    "rule or preference, a piece of background, or a decision to make MUST "
    "become its own item. When in doubt, emit an item rather than drop it. "
    "Split compound asks — 'do X and also tell me Y' is two items.\n\n"
    "The message may be preceded by a RELEVANT KNOWLEDGE block of facts already "
    "established about this project (its identity, conventions, prior lessons). "
    "Use it to interpret the message correctly — a term defined there means what "
    "it says there — but do NOT itemize the background; decompose only the USER "
    "MESSAGE.\n\n"
    "Classify each item's kind as exactly one of:\n"
    "- question: the user wants an answer or information back.\n"
    "- task: a concrete action to perform.\n"
    "- directive: a standing rule, preference, or constraint to honor.\n"
    "- context: background the user is providing; no action needed.\n"
    "- decision: a fork the user is leaving for you to choose.\n\n"
    "For each item give:\n"
    "- kind: one of the five above.\n"
    "- ask: a clear, standalone restatement of this single ask, fully "
    "unambiguous on its own without the surrounding message.\n"
    "- source_text: the verbatim span(s) from the message this item came "
    "from (join non-contiguous spans with ' … ').\n"
    "- detail: any sub-points or specifics for this item, else \"\".\n\n"
    "Also write a one- or two-sentence summary of the whole message.\n\n"
    "Respond with ONLY a JSON object: "
    '{"summary": "...", "items": [{"kind": "...", "ask": "...", '
    '"source_text": "...", "detail": "..."}]}'
)

_SYSTEM_AUDIT = (
    "You are the coverage auditor of an assistant's intake layer. You are "
    "given a user's ORIGINAL message and the items already extracted from "
    "it. Your only job is to find asks that were MISSED — anything in the "
    "original not yet represented: an unanswered question, an un-listed "
    "task, a preference or constraint not captured, context skipped, a "
    "decision not surfaced. Scan the original span by span and check each "
    "against the extracted items. Do NOT restate items already covered; "
    "return only genuinely missed asks, in the same schema.\n\n"
    "Respond with ONLY a JSON object: "
    '{"complete": true, "missed": [{"kind": "...", "ask": "...", '
    '"source_text": "...", "detail": "..."}]}. '
    "Set complete to true only when nothing was missed (missed is empty)."
)

_SYSTEM_SCOPE = (
    "You are the scope ROUTER of an assistant that supervises work across a "
    "person's projects. You are given the FOCUS project (where the agent is "
    "currently working), the person's OTHER KNOWN projects, and a numbered "
    "list of ACTION items — tasks the user wants performed. For each item, "
    "decide WHICH project carrying it out would change, build, or operate on. "
    "Judge by what the action would TOUCH, not by wording.\n"
    "Route each item to exactly one of:\n"
    "- focus: it operates on the focus project itself; OR on THIS assistant "
    "or the current session/configuration (wiring, connecting, or configuring "
    "this assistant counts as focus); OR it only researches, explains, "
    "references, or plans around another project WITHOUT changing that "
    "project's code.\n"
    "- <a known project's exact name>: doing it would modify THAT project's "
    "code, files, or state (build, fix, edit, create, delete, refactor, "
    "audit-with-changes). Use the name exactly as listed.\n"
    "- outside: it would modify a project or system that is NEITHER the focus "
    "NOR any known project in the list.\n"
    "Merely naming or describing another system is NOT a route to it — only "
    "acting on its code or files is. When a task genuinely WOULD modify "
    "another project and you are unsure which, mark outside so it is surfaced "
    "rather than waved through.\n\n"
    "Respond with ONLY a JSON object: "
    '{"verdicts": [{"index": <the item\'s number>, "scope": "focus" | '
    '"<known project name>" | "outside", "reason": "<short why>"}]}. '
    "Include EVERY item exactly once."
)


def _coerce_kind(raw: object) -> str:
    k = str(raw or "").strip().lower()
    if k in ITEM_KINDS:
        return k
    return _KIND_SYNONYMS.get(k, CONTEXT)


def _items_from_list(raw_items: object) -> list[Item]:
    """Build Items from a model's item array, skipping empty/no-ask rows."""
    out: list[Item] = []
    if not isinstance(raw_items, list):
        return out
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        ask = str(entry.get("ask", "")).strip()
        if not ask:
            continue
        out.append(
            Item(
                kind=_coerce_kind(entry.get("kind")),
                ask=ask,
                source_text=str(entry.get("source_text", "")).strip(),
                detail=str(entry.get("detail", "")).strip(),
            )
        )
    return out


def _norm(ask: str) -> str:
    return " ".join(ask.lower().split())


def _itemize_prompt(prompt: str, grounding: str = "") -> str:
    """The itemizer's user message, optionally prefixed with a bounded RELEVANT
    KNOWLEDGE block (from :mod:`saddle.recall`) so the model interprets the prompt
    against what the project already knows — e.g. that 'saddle' names THIS harness,
    not some same-named external product."""
    if grounding:
        return f"{grounding}\n\nUSER MESSAGE:\n{prompt}"
    return f"USER MESSAGE:\n{prompt}"


def _audit_prompt(prompt: str, items: list[Item]) -> str:
    lines = [f"{i + 1}. [{it.kind}] {it.ask}" for i, it in enumerate(items)]
    extracted = "\n".join(lines) if lines else "(none)"
    return (
        f"ORIGINAL MESSAGE:\n{prompt}\n\n"
        f"ITEMS ALREADY EXTRACTED:\n{extracted}\n\n"
        "Return any asks in the ORIGINAL that are not represented above."
    )


def _scope_prompt(focus_desc: str, known_block: str, items: list[Item]) -> str:
    lines = [f"{i + 1}. [{it.kind}] {it.ask}" for i, it in enumerate(items)]
    listing = "\n".join(lines) if lines else "(none)"
    known = known_block or "(none known)"
    return (
        f"FOCUS PROJECT: {focus_desc}\n\n"
        f"OTHER KNOWN PROJECTS:\n{known}\n\n"
        f"WORK ITEMS:\n{listing}\n\n"
        "Route each item to the project it would act on."
    )


async def _classify_scope(
    caller: "LLMCaller", items: list[Item], ctx: Context
) -> dict[str, Any]:
    """Route each ACTION (task) item to the project it would act on.

    Only tasks are weighed. A question, a piece of context, a decision, or a
    standing directive that *mentions* another project is discussion or
    planning — referencing another codebase is not drift; only a task that
    would MODIFY another project's code or files crosses the boundary. So
    non-task items are never routed, and a prompt with no tasks at all skips
    the LLM call entirely: nothing to act on, nothing to route.

    Three routes per task (mediator design §4):

    * the FOCUS project (or this assistant/session) — the default, no marking;
    * a KNOWN sibling project from the registry — the item is STAMPED
      (``item.project``) so every downstream stage reads/writes THAT project's
      ledger; both projects are simply active this turn, not a warning;
    * OUTSIDE every known project — surfaced as a warning exactly as before
      (an unknown target should be seen, not silently routed).

    Annotates — never filters. If the model fails to route every task, that is
    recorded (``scope_checked=False``) and surfaced rather than silently read
    as "all in focus" — the same never-imply-coverage stance as the audit pass.
    """
    # Only actions can cross the project boundary. Filter to tasks, keeping each
    # one's ORIGINAL position so the surfaced warning numbers line up with the
    # full item list the user sees. No tasks -> nothing to check, no LLM call.
    actionable = [(i, it) for i, it in enumerate(items) if it.kind == TASK]
    if not actionable:
        return {"out_of_focus": [], "scope_warning": "", "scope_checked": True,
                "item_projects": {}}

    from saddle import registry

    known = registry.known_projects(ctx.tenant)
    known_block = registry.descriptor_block(ctx.tenant, exclude=ctx.project)
    raw = await _call_json(
        caller, _SYSTEM_SCOPE,
        _scope_prompt(focus_descriptor(ctx), known_block,
                      [it for _, it in actionable]),
        label="intake/scope",
    )
    verdicts = raw.get("verdicts")
    out_of_focus: list[dict[str, Any]] = []
    item_projects: dict[int, str] = {}
    classified: set[int] = set()
    if isinstance(verdicts, list):
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            try:
                sub = int(v.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= sub < len(actionable)) or sub in classified:
                continue
            classified.add(sub)
            scope = str(v.get("scope", "")).strip().lower()
            orig_idx, item = actionable[sub]
            if scope in ("focus", "in_focus", "", ctx.project):
                continue  # ambient — no stamp needed
            if scope in known:
                # Routed to a known sibling: stamp the item so drift checks and
                # lessons land in THAT project's ledger. Not a warning.
                item.project = scope
                item_projects[orig_idx + 1] = scope
            else:
                # "outside" or an unrecognized name — surfaced, never guessed.
                out_of_focus.append({
                    "n": orig_idx + 1,
                    "ask": item.ask,
                    "reason": str(v.get("reason", "")).strip(),
                })
    scope_checked = len(classified) >= len(actionable)
    warning = ""
    if out_of_focus:
        proj = focus_project(ctx)
        warning = (
            f"{len(out_of_focus)} of {len(actionable)} action(s) target work "
            f"outside {proj} and outside every project saddle knows. That work "
            f"changes something unrecognized and should not be acted on here "
            f"without an explicit decision."
        )
    elif not scope_checked:
        warning = (
            "the scope router did not classify every action — review scope "
            "manually before acting."
        )
    return {
        "out_of_focus": out_of_focus,
        "scope_warning": warning,
        "scope_checked": scope_checked,
        "item_projects": item_projects,
    }


# -- directive durability gate (closes design_issues Gap 4) ------------------
#
# A `directive` item is a candidate STANDING rule. But not every directive a user
# states is durable: many are TASK-LOCAL instructions ("respond with only JSON",
# "make THIS design game-agnostic", "limit THIS audit to X"). Persisting those as
# standing policy pollutes every FUTURE task — they get enforced forever, out of
# context (the live failure that buried saddle-self work under rayxiv4-audit
# directives). This gate classifies durability so only genuinely-standing rules
# are promoted; task-local ones still guide the current task but are never stored.

_SYSTEM_DURABILITY = (
    "You are the directive-durability gate of an assistant. You are given "
    "candidate STANDING RULES — directive-type asks pulled from a user's message. "
    "For each, decide whether it is a DURABLE standing rule or a TASK-LOCAL "
    "instruction.\n"
    "- standing: a preference, constraint, or principle the user wants honored on "
    "EVERY FUTURE task, indefinitely — it is about HOW work should ALWAYS be done "
    "('always fail loud', 'no band-aids', 'never delete code without classifying "
    "it', 'prefer structural fixes over patches').\n"
    "- task: an instruction tied to the CURRENT request — an output format for "
    "this one answer, a constraint on this one artifact/design, one-off scope or "
    "framing ('respond with only JSON', 'make THIS design game-agnostic', 'limit "
    "THIS audit to drifts', 'slice THIS work'). It should guide the current task "
    "but must NOT become a forever-rule.\n"
    "When unsure, choose 'task'. Wrongly persisting a one-off instruction as a "
    "standing rule pollutes every future task (it is enforced forever, out of "
    "context) — far worse than declining to persist a borderline rule the user can "
    "simply restate or add explicitly.\n\n"
    "Respond with ONLY a JSON object: "
    '{"verdicts": [{"index": <the item\'s number>, "durability": "standing" | '
    '"task", "reason": "<short why>"}]}. '
    "Include EVERY item exactly once."
)


def _durability_prompt(items: list[Item]) -> str:
    lines = [f"{i + 1}. {it.ask}" for i, it in enumerate(items)]
    listing = "\n".join(lines) if lines else "(none)"
    return (
        f"CANDIDATE STANDING RULES:\n{listing}\n\n"
        "Classify each as a durable standing rule or a task-local instruction."
    )


async def classify_directive_durability(
    caller: "LLMCaller", directive_items: list[Item], ctx: Context
) -> dict[str, Any]:
    """Split directive items into durable STANDING rules vs TASK-LOCAL instructions.

    Returns ``{"standing": [Item], "task": [Item], "checked": bool}``. Only
    ``standing`` items should be persisted to policy; ``task`` ones still ride with
    the current request but are never stored. Fails SAFE toward Gap 4: an item the
    model does not classify, and the whole set if the call fails, default to
    ``task`` (NOT promoted), because the harm this gate exists to stop is
    over-promotion — a real standing rule left un-persisted is recoverable (the
    user restates or runs ``saddle directives --add``); a one-off persisted forever
    is not, short of curation. Runs inside ``ctx``'s tenant fairness gate."""
    items = [it for it in directive_items if it.ask.strip()]
    if not items:
        return {"standing": [], "task": [], "checked": True}

    try:
        async with tenant_gate(ctx):
            raw = await _call_json(
                caller, _SYSTEM_DURABILITY, _durability_prompt(items),
                label="intake/directive-durability",
            )
    except Exception as exc:  # noqa: BLE001 — fail safe: promote nothing, surface why
        _log.warning(
            "directive-durability classify failed (%s); promoting no directives "
            "this run (fail-safe against policy pollution)", exc,
        )
        return {"standing": [], "task": list(items), "checked": False}

    verdicts = raw.get("verdicts")
    standing_idx: set[int] = set()
    classified: set[int] = set()
    if isinstance(verdicts, list):
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            try:
                sub = int(v.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= sub < len(items)) or sub in classified:
                continue
            classified.add(sub)
            if str(v.get("durability", "")).strip().lower() == "standing":
                standing_idx.add(sub)
    # Unclassified items default to task (conservative against pollution).
    standing = [it for i, it in enumerate(items) if i in standing_idx]
    task = [it for i, it in enumerate(items) if i not in standing_idx]
    return {
        "standing": standing,
        "task": task,
        "checked": len(classified) >= len(items),
    }


async def decompose(
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    store: "Store | None" = None,
    persist: bool = True,
    max_audits: int = 2,
    check_focus: bool = True,
    ground: bool = True,
) -> Intake:
    """Decompose ``prompt`` into a recorded :class:`Intake` for ``ctx``.

    Two-pass: itemize, then audit coverage (bounded by ``max_audits``)
    appending any missed items until the auditor reports complete. A final
    focus pass (unless ``check_focus=False``) tags any TASK that targets work
    OUTSIDE the focus project and records a ``scope_warning`` in ``meta`` —
    discussing or referencing another project (questions, context, decisions)
    is not drift, only acting on its code is. It annotates, never filters, so
    off-focus asks are surfaced, not dropped. The
    whole decomposition runs inside ``ctx``'s tenant fairness gate. Persists to
    the store unless ``persist=False``. Pass ``caller`` to bypass provider
    resolution (tests); otherwise the context's default fallback chain is used.

    ``ground`` (default on) grounds the itemizer in the project's memory: a
    bounded :func:`saddle.recall.recall_block` is retrieved for ``prompt`` and
    prefixed to the itemize call, so a term the project has already defined (its
    own identity, a convention) is read correctly instead of guessed. It is
    retrieval, not prepend — top-k and length-bounded (see :mod:`saddle.recall`)
    — and best-effort: a down embed server or empty DKB simply yields no block.
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("cannot decompose an empty prompt")
    ctx = ctx or _default_ctx()

    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    # Ground the itemizer in project memory (bounded, off the event loop). Done
    # before the fairness gate: recall is a local retrieval, not a provider call,
    # so it does not belong inside the tenant's LLM quota.
    grounding = ""
    if ground:
        from saddle.recall import recall_block
        grounding = await asyncio.to_thread(recall_block, ctx, text)

    audits_run = 0
    audit_complete = False
    # Scope-router result — populated below unless check_focus is off / no items.
    scope_info: dict[str, Any] = {
        "out_of_focus": [], "scope_warning": "", "scope_checked": False,
        "item_projects": {},
    }
    async with tenant_gate(ctx):
        first = await _call_json(
            caller, _SYSTEM_ITEMIZE, _itemize_prompt(text, grounding),
            label="intake/itemize",
        )
        summary = str(first.get("summary", "")).strip()
        items = _items_from_list(first.get("items"))
        pass1_count = len(items)

        # Coverage audit loop — the "miss nothing" guarantee. Each pass
        # surfaces asks the prior passes dropped; merge novel ones and
        # re-audit until the auditor says complete or the bound is hit.
        for _ in range(max(0, max_audits)):
            audit = await _call_json(
                caller, _SYSTEM_AUDIT, _audit_prompt(text, items), label="intake/audit"
            )
            audits_run += 1
            missed = _items_from_list(audit.get("missed"))
            seen = {_norm(it.ask) for it in items}
            novel = [m for m in missed if _norm(m.ask) not in seen]
            items.extend(novel)
            if bool(audit.get("complete")) and not novel:
                audit_complete = True
                break
            if not novel:
                # Auditor flagged incomplete but named nothing new — no
                # forward progress is possible; stop rather than spin.
                break

        # Focus gate — flag TASKS that target work OUTSIDE this project (only
        # actions can drift; discussing or referencing another project is fine).
        # Annotate, never filter: out-of-focus tasks stay in the list, but the
        # drift ("you pulled me onto another project") is surfaced. Refusing
        # the work is Layer 0's job, not intake's.
        if check_focus and items:
            scope_info = await _classify_scope(caller, items, ctx)

    intake = Intake(
        raw_prompt=text,
        summary=summary,
        items=items,
        meta={
            "pass1_items": pass1_count,
            "audits_run": audits_run,
            "audit_complete": audit_complete,
            "final_items": len(items),
            "todo_items": sum(1 for it in items if it.is_todo()),
            "out_of_focus": scope_info["out_of_focus"],
            "scope_warning": scope_info["scope_warning"],
            "scope_checked": scope_info["scope_checked"],
            # item number -> sibling project slug, for renders and the turn's
            # active-scope set. The stamp itself rides on each Item.project.
            "item_projects": {
                str(k): v for k, v in scope_info.get("item_projects", {}).items()
            },
        },
    )

    if persist:
        (store or get_store()).save_intake(ctx, intake)
        _log.info(
            "intake %s recorded for %s: %d items (%d todo)",
            intake.id, ctx.key, len(items), intake.meta["todo_items"],
        )
    return intake


def format_intake(intake: Intake) -> str:
    """Human-readable rendering of an intake for the CLI."""
    lines: list[str] = []
    if intake.id:
        lines.append(f"intake {intake.id}  [{intake.tenant}/{intake.project}]")
    if intake.summary:
        lines.append(f"summary: {intake.summary}")
    warning = intake.meta.get("scope_warning", "")
    if warning:
        lines.append(f"⚠ SCOPE: {warning}")
        for o in intake.meta.get("out_of_focus", []):
            reason = f" — {o['reason']}" if o.get("reason") else ""
            lines.append(f"    • item {o['n']} is out of focus: {o['ask']}{reason}")
    routed = intake.meta.get("item_projects", {})
    if routed:
        names = sorted(set(routed.values()))
        ambient = intake.project or "the current project"
        lines.append(
            f"this message spans more than one project: items marked with a "
            f"→ act on {', '.join(names)}; the rest act on {ambient}."
        )
    todo = intake.meta.get("todo_items", sum(1 for i in intake.items if i.is_todo()))
    lines.append(f"{len(intake.items)} items ({todo} todo):")
    for i, it in enumerate(intake.items, 1):
        # Tag only items routed AWAY from the ambient project (after persist,
        # ambient items carry the intake's own project — not a route).
        proj = getattr(it, "project", "")
        route = f" → {proj}" if proj and proj != intake.project else ""
        lines.append(f"  {i:>2}. [{it.kind:<9}]{route} {it.ask}")
        if it.detail:
            lines.append(f"        ↳ {it.detail}")
    # Default False, not True: an intake whose meta carries no convergence verdict
    # has NOT proven coverage, so surface the gap rather than silently imply it
    # converged. Matches the `audit_complete = False` init in itemize().
    if not intake.meta.get("audit_complete", False):
        lines.append("(note: coverage audit did not converge — review for gaps)")
    return "\n".join(lines)
