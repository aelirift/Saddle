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

import logging
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
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


def _itemize_prompt(prompt: str) -> str:
    return f"USER MESSAGE:\n{prompt}"


def _audit_prompt(prompt: str, items: list[Item]) -> str:
    lines = [f"{i + 1}. [{it.kind}] {it.ask}" for i, it in enumerate(items)]
    extracted = "\n".join(lines) if lines else "(none)"
    return (
        f"ORIGINAL MESSAGE:\n{prompt}\n\n"
        f"ITEMS ALREADY EXTRACTED:\n{extracted}\n\n"
        "Return any asks in the ORIGINAL that are not represented above."
    )


async def decompose(
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
    store: "Store | None" = None,
    persist: bool = True,
    max_audits: int = 2,
) -> Intake:
    """Decompose ``prompt`` into a recorded :class:`Intake` for ``ctx``.

    Two-pass: itemize, then audit coverage (bounded by ``max_audits``)
    appending any missed items until the auditor reports complete. The whole
    decomposition runs inside ``ctx``'s tenant fairness gate. Persists to the
    store unless ``persist=False``. Pass ``caller`` to bypass provider
    resolution (tests); otherwise the context's default fallback chain is used.
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("cannot decompose an empty prompt")
    ctx = ctx or _default_ctx()

    if caller is None:
        from saddle.llm.callers import build_callers
        caller = build_callers(ctx)["default"]

    audits_run = 0
    audit_complete = False
    async with tenant_gate(ctx):
        first = await _call_json(
            caller, _SYSTEM_ITEMIZE, _itemize_prompt(text), label="intake/itemize"
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
    todo = intake.meta.get("todo_items", sum(1 for i in intake.items if i.is_todo()))
    lines.append(f"{len(intake.items)} items ({todo} todo):")
    for i, it in enumerate(intake.items, 1):
        lines.append(f"  {i:>2}. [{it.kind:<9}] {it.ask}")
        if it.detail:
            lines.append(f"        ↳ {it.detail}")
    if not intake.meta.get("audit_complete", True):
        lines.append("(note: coverage audit did not converge — review for gaps)")
    return "\n".join(lines)
