"""The raised-item ledger — user-closed-only (#77).

Saddle itemizes every prompt into typed asks and persists the task/directive ones
as an OPEN todo ledger (``store.save_intake`` -> ``store.todos``). But the schema's
``set_item_status`` had NO caller anywhere, so nothing ever moved an item out of
OPEN: the backlog grew forever (audit Gap 3 — 500 open for one project). The "6
asks -> all 6 tracked to done" guarantee had no runtime.

This is that runtime — and it closes items the ONE way the creator allows: by the
USER's word (ruling 2026-07-20). An agent may NEVER self-close (a coder claiming
"done" is exactly the over-claim the completion gate distrusts). The three
user-driven closures, plus the reverse:

* "happy with it"  -> DONE
* "don't want it"  -> DROPPED
* "changed it to X" -> SUPERSEDED (the new ask X is itemized as its own OPEN item
  by Stage 1 in the same turn — this only retires the old one)
* "no, that's not done / reopen it" -> back to OPEN (the escape hatch for a
  direct-close that matched the wrong item)

Model (creator ruling 2026-07-20 — DIRECT-CLOSE + herald): each turn an LLM reads
the USER's message against the recent todo backlog and, on a CLEAR closure signal
that clearly names an item, closes it immediately and HERALDS what changed (a wrong
match is one "no, reopen that" away, and the audit's real failure is UNDER-closing).
ASYMMETRIC CAUTION: it closes ONLY on a clear signal tied to a specific item; an
ordinary instruction, a question, or an ambiguous reference closes nothing.

Fail-LOUD like the other stages: a classify failure PROPAGATES to
:func:`saddle.supervisor.run_stage`; nothing is written on failure, so the ledger
is left unchanged.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from saddle.context import Context, default as _default_ctx
from saddle.llm.json_tools import call_json
from saddle.llm.pool import tenant_gate
from saddle.models import (
    DONE,
    DROPPED,
    OPEN,
    SUPERSEDED,
    TODO_KINDS,
)
from saddle.voice import VOICE_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller
    from saddle.models import Item
    from saddle.store import Store


# How many of the most-recent todo items the classifier weighs — bounds the LLM
# input so a huge backlog (500+) never floods the call; a user closes recent work.
CANDIDATE_CAP = 30

# The dispositions the classifier may return, and how each maps to a status. A
# reopen returns an OPEN item to the backlog (the direct-close escape hatch).
CLOSE_DONE = "done"
CLOSE_DROPPED = "dropped"
CLOSE_SUPERSEDED = "superseded"
REOPEN = "reopen"
_DISPOSITION_STATUS = {
    CLOSE_DONE: DONE,
    CLOSE_DROPPED: DROPPED,
    CLOSE_SUPERSEDED: SUPERSEDED,
    REOPEN: OPEN,
}
DISPOSITIONS: frozenset[str] = frozenset(_DISPOSITION_STATUS)


@dataclass
class ItemClosure:
    """One user-driven ledger change: ``item`` is the affected todo, ``disposition``
    one of :data:`DISPOSITIONS`, ``replacement`` the plain text of what a supersede
    changed it to (else "")."""

    item: "Item"
    disposition: str
    replacement: str = ""


def _candidate_items(ctx: Context, store: "Store") -> list["Item"]:
    """The recent todo items the closure classifier weighs — the most recent
    :data:`CANDIDATE_CAP` task/directive items across statuses (OPEN ones are
    closeable, recently-closed ones are reopenable), newest first."""
    items = store.list_items(ctx, kinds=TODO_KINDS)
    items.sort(key=lambda i: i.ts, reverse=True)
    return items[:CANDIDATE_CAP]


# -- the closure classifier ---------------------------------------------------

_SYS_ITEM_CLOSE = (
    "You are the raised-item ledger of a supervisory harness. Saddle tracks a "
    "backlog of the user's still-open tasks/directives. Read the user's newest "
    "message and decide whether it CLOSES (or reopens) any of the listed items — "
    "and ONLY by the user's own word, never because an agent thinks it finished "
    "the work.\n"
    "You are given a numbered list of RECENT backlog items, each with its current "
    "status and text. For each item the user's message clearly acts on, return an "
    "entry with the item's number and one disposition:\n"
    "- done: the user is satisfied it is handled — 'that's done', 'happy with the "
    "X', 'looks good, ship it', 'that works now'.\n"
    "- dropped: the user no longer wants it — 'forget X', 'drop that', 'never "
    "mind the X', 'we don't need that anymore'.\n"
    "- superseded: the user changed it to something else — 'actually make X do Y "
    "instead', 'replace the X plan with …', 'scratch that, do Z'. Put the new "
    "direction in 'replacement'.\n"
    "- reopen: the user says a previously-closed item is NOT actually done / bring "
    "it back — 'no, that's not done', 'reopen the X', 'X is still broken'.\n"
    "ASYMMETRIC CAUTION — this is the crux. Closing an item the user did NOT "
    "close loses real tracked work, so return an entry ONLY when the message "
    "clearly acts on a SPECIFIC listed item. An ordinary instruction, a question, "
    "a report of progress by the assistant, or an ambiguous reference closes "
    "NOTHING — return an empty list. Never guess which item from a vague 'done'; "
    "if it does not clearly name one of the listed items, close nothing. Match by "
    "MEANING, not shared words.\n"
    "Return ONLY JSON: {\"closures\": [{\"n\": <item number>, \"disposition\": "
    "\"done|dropped|superseded|reopen\", \"replacement\": \"…\", \"confidence\": "
    "0.0}]}. An empty closures list is the common, correct answer."
    + VOICE_CONTRACT
)


def _close_prompt(candidates: list["Item"], prompt: str) -> str:
    lines = ["RECENT BACKLOG ITEMS:"]
    for n, it in enumerate(candidates, 1):
        ask = " ".join((it.ask or "").split())
        if len(ask) > 200:
            ask = ask[:199] + "…"
        lines.append(f"  {n}. [{it.status}] {ask}")
    return (
        "\n".join(lines)
        + f"\n\nTHE USER'S NEWEST MESSAGE:\n{prompt}\n\n"
        "Which listed items, if any, does this message close or reopen?"
    )


async def classify_item_closures(
    candidates: list["Item"],
    prompt: str,
    ctx: Context | None = None,
    *,
    caller: "LLMCaller | None" = None,
) -> list[ItemClosure]:
    """Classify which of ``candidates`` the user's ``prompt`` closes/reopens. One
    bounded ``call_json`` inside the tenant fairness gate. FAIL-LOUD on caller/parse
    error (never a false empty). Returns only well-formed, in-range entries."""
    text = (prompt or "").strip()
    if not text or not candidates:
        return []
    ctx = ctx or _default_ctx()
    if caller is None:
        from saddle.llm.callers import build_callers

        caller = build_callers(ctx)["default"]
    async with tenant_gate(ctx):
        payload = await call_json(
            caller, _SYS_ITEM_CLOSE, _close_prompt(candidates, text),
            label="ledger/item-close",
        )
    out: list[ItemClosure] = []
    seen: set[str] = set()
    for entry in payload.get("closures") or []:
        if not isinstance(entry, dict):
            continue
        disp = str(entry.get("disposition") or "").strip().lower()
        if disp not in DISPOSITIONS:
            continue
        try:
            n = int(entry.get("n"))
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= len(candidates)):
            continue          # a hallucinated index can never touch a real item
        item = candidates[n - 1]
        if item.id in seen:
            continue          # one disposition per item per turn
        seen.add(item.id)
        out.append(ItemClosure(
            item=item, disposition=disp,
            replacement=str(entry.get("replacement") or "").strip(),
        ))
    return out


def apply_item_closures(
    ctx: Context, closures: list[ItemClosure], store: "Store"
) -> list[ItemClosure]:
    """Persist each closure via ``set_item_status`` (a reopen -> OPEN). A closure
    whose item was already at the target status, or whose write returns no row, is
    dropped from the returned list — so the herald names only REAL changes. This is
    the ONLY writer of a user-driven item status (an agent never self-closes)."""
    applied: list[ItemClosure] = []
    for c in closures:
        target = _DISPOSITION_STATUS.get(c.disposition)
        if target is None or c.item.status == target:
            continue
        try:
            if store.set_item_status(ctx, c.item.id, target):
                applied.append(c)
        except Exception as exc:  # noqa: BLE001 — one bad write must not sink the rest
            print(f"item_ledger: set_item_status failed for {c.item.id} ({exc!r})",
                  file=sys.stderr)
    return applied


# -- the composed turn --------------------------------------------------------

@dataclass
class LedgerOutcome:
    """What the ledger step did this turn. ``closed`` are the applied closures (for
    the herald); ``herald`` is the plain on-screen line when items changed, else ""."""

    closed: list[ItemClosure] = field(default_factory=list)
    herald: str = ""


async def ledger_turn(
    prompt: str,
    *,
    ctx: Context | None = None,
    store: "Store | None" = None,
    caller: "LLMCaller | None" = None,
) -> LedgerOutcome:
    """The whole ledger step for one prompt: weigh the recent backlog, classify
    which items the user's message closes/reopens, apply them, and herald the real
    changes. The single coroutine the intake hook drives under a deadline via
    :func:`saddle.supervisor.run_bounded`."""
    ctx = ctx or _default_ctx()
    if store is None:
        from saddle.store import get_store

        store = get_store()

    candidates = _candidate_items(ctx, store)
    if not candidates:
        return LedgerOutcome()
    closures = await classify_item_closures(candidates, prompt, ctx, caller=caller)
    if not closures:
        return LedgerOutcome()
    applied = apply_item_closures(ctx, closures, store)
    if not applied:
        return LedgerOutcome()
    from saddle.voice import ledger_items_closed

    return LedgerOutcome(closed=applied, herald=ledger_items_closed(applied))
