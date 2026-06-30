"""Recall — bounded, on-demand retrieval from the DKB for a single turn.

This is the READ face of saddle's memory (the DKB write face is
:meth:`saddle.dkb.DKB.add_knowledge` / ``saddle remember``). A layer that is
about to reason — Layer 1 itemizing a prompt, the agent serving a request —
asks recall for the few entries most relevant to what's in front of it, and
gets back the top matches and nothing else.

**Retrieval, never prepend — the whole point.** The naive way to give a model
its memory is to paste the standing facts in front of every prompt. That grows
without bound: every new fact lengthens every future turn, and the context
balloons exactly as it accumulates. Recall is the opposite contract:

* **query-scoped** — only entries relevant to *this* prompt come back, found by
  the DKB's hybrid search (semantic ∪ keyword), so an unrelated fact costs
  nothing this turn;
* **top-k bounded** — at most ``k`` entries (default :data:`_DEFAULT_K`), so the
  injected block has a fixed ceiling no matter how large the DKB grows;
* **length-clipped** — each entry's body is capped (:data:`_CLIP`), so one
  verbose entry can't flood the turn;
* **fresh, not cumulative** — recomputed per turn and discarded, never appended
  to a growing preamble.

So memory can grow forever while what any single turn carries stays small and
on-topic. Recall degrades softly: if the embed server is down the DKB falls back
to keyword-only, and if retrieval fails entirely recall returns nothing rather
than sinking the turn (memory is enrichment, not a gate).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Iterable

from saddle.context import Context

if TYPE_CHECKING:  # pragma: no cover
    from saddle.dkb import DKB
    from saddle.models import Knowledge

_log = logging.getLogger("saddle.recall")

_DEFAULT_K = 5      # entries returned per recall — the fixed per-turn ceiling
_CLIP = 280         # per-entry body length cap in the rendered block


def _recall_k(k: int | None) -> int:
    """Resolve the recall ceiling: explicit arg wins, else ``$SADDLE_RECALL_K``,
    else :data:`_DEFAULT_K`. A value <= 0 disables recall (returns nothing) — an
    explicit opt-out, never a silent one."""
    if k is not None:
        return k
    raw = os.environ.get("SADDLE_RECALL_K", "")
    if not raw.strip():
        return _DEFAULT_K
    try:
        return int(raw)
    except ValueError:
        _log.warning("SADDLE_RECALL_K=%r is not an int; using %d", raw, _DEFAULT_K)
        return _DEFAULT_K


def recall(
    ctx: Context,
    query: str,
    *,
    k: int | None = None,
    kinds: Iterable[str] | None = None,
    dkb: "DKB | None" = None,
) -> list["Knowledge"]:
    """Up to ``k`` DKB entries most relevant to ``query``, best first, scope-
    laddered to ``ctx``. ``kinds`` optionally restricts the families searched
    (e.g. only ``fact``/``reference`` for "what do we know", excluding design
    wisdom). Returns ``[]`` on an empty query, a disabled ceiling, or a retrieval
    failure — never raises, because memory is enrichment and must not break the
    turn it was meant to help."""
    n = _recall_k(k)
    q = (query or "").strip()
    if n <= 0 or not q:
        return []
    from saddle.dkb import get_dkb

    d = dkb if dkb is not None else get_dkb()
    try:
        hits = d.search_knowledge(ctx, q, k=n, kinds=kinds)
    except Exception as exc:  # noqa: BLE001 — recall must never sink its caller
        _log.warning("recall failed for %r in [%s]: %s", q[:60], ctx.key, exc)
        return []
    return [kn for kn, _ in hits]


def format_recall(entries: list["Knowledge"], *, clip: int = _CLIP) -> str:
    """Render recalled entries as a compact, length-bounded grounding block for
    injection into ONE LLM call (or the agent's per-turn context). Empty in →
    empty string out, so a no-hit recall renders to nothing. The framing tells
    the reader these are already-established facts to USE, not re-derive — which
    is what stops the "figure out what saddle is again" loop."""
    if not entries:
        return ""
    lines = [
        "RELEVANT KNOWLEDGE — saddle's memory of established facts and lessons "
        "for this project. Treat these as already known; use them instead of "
        "re-deriving, re-reading files, or re-searching the web:"
    ]
    for kn in entries:
        body = " ".join((kn.body or "").split())
        if len(body) > clip:
            body = body[: clip - 1] + "…"
        lines.append(f"- [{kn.kind}] {kn.title}: {body}")
    return "\n".join(lines)


def recall_block(
    ctx: Context,
    query: str,
    *,
    k: int | None = None,
    kinds: Iterable[str] | None = None,
    dkb: "DKB | None" = None,
    clip: int = _CLIP,
) -> str:
    """:func:`recall` + :func:`format_recall` in one call — the bounded grounding
    string a caller injects, or ``""`` when nothing relevant is known."""
    return format_recall(
        recall(ctx, query, k=k, kinds=kinds, dkb=dkb), clip=clip
    )
