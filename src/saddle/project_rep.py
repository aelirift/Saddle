"""The project rep — saddle's compact, live "what this project is about" model.

The backbone of the live-brain (owner vision 2026-07-20): instead of re-reading raw
transcripts, saddle holds ONE compact, per-project model it can feed the agent and
reason over. This is that model — assembled as a LIVE VIEW over the durable
substrate saddle already keeps, so it is always current without a second store to
drift:

* ``intent``      — the live commitment (the fork/pick ledger, kept honest by the
  #76 live-goal layer);
* ``designs``     — the recent SETTLED designs (the design/concept ledger);
* ``knowledge``   — the top DKB entries (facts / lessons / anti-patterns /
  best-practices) relevant to the current intent;
* ``map_status``  — the freshness of the project's structural maps, via a PLUGGABLE
  probe (saddle must not couple to any one project's map importer — the concrete
  wiring + the freshness guarantee is a separate slice);
* ``rolling_chat``— a bounded digest of the recent conversation (owner budget: 10k
  tokens/project). v1 carries whatever the caller supplies; the incremental
  compactor that keeps it live is the next slice.

Rendering is token-bounded (``token_budget``) so the fed block never floods the
agent. Assembly is fail-soft like :mod:`saddle.recall` — the rep is enrichment and
must never wedge the turn it was meant to help.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from saddle.recall import recall

if TYPE_CHECKING:  # pragma: no cover
    from saddle.context import Context
    from saddle.dialog import IntentTracker
    from saddle.dkb import DKB

_log = logging.getLogger("saddle.project_rep")

# Owner ruling 2026-07-20: 10k tokens/project (against ~1M input windows, growing).
DEFAULT_TOKEN_BUDGET = 10_000
_CHARS_PER_TOKEN = 4          # crude budget accounting without a tokenizer in the loop
_CLIP = 240                   # per-entry body cap in the rendered block

# A map-freshness probe: given a ctx, return a short human status ("fresh",
# "stale: 12 files", "no map") or "" if this project has no registered map.
MapProbe = Callable[["Context"], str]


@dataclass
class ProjectRep:
    """The assembled compact model for one (tenant, project). Every field is a live
    view over durable state, so a fresh assemble is always current."""

    project_key: str = ""
    intent: str = ""
    designs: list[str] = field(default_factory=list)
    knowledge: list[tuple[str, str, str]] = field(default_factory=list)  # (kind,title,body)
    map_status: str = ""
    rolling_chat: str = ""
    token_budget: int = DEFAULT_TOKEN_BUDGET


def assemble(
    ctx: "Context",
    *,
    query: str = "",
    dkb: "DKB | None" = None,
    tracker: "IntentTracker | None" = None,
    map_probe: "MapProbe | None" = None,
    rolling_chat: str = "",
    max_designs: int = 5,
    recall_k: int = 8,
) -> ProjectRep:
    """Assemble the project rep for ``ctx`` from the live substrate. ``query`` biases
    the knowledge pull (defaults to the current intent). ``map_probe`` supplies map
    freshness without coupling saddle to a project's importer. Each source is
    isolated in its own try/except — a single failing source degrades that field to
    empty, never sinks the whole rep (enrichment must not wedge the turn)."""
    intent = ""
    try:
        tr = tracker
        if tr is None:
            from saddle.dialog import get_tracker

            tr = get_tracker()
        binding, fork = tr.committed_fork(ctx)
        if binding is not None:
            from saddle.livegoal import commitment_gist

            intent = commitment_gist(binding, fork)
    except Exception as exc:  # noqa: BLE001 — a source failure must not sink the rep
        _log.warning("project_rep: intent source failed for [%s]: %s", ctx.key, exc)

    designs: list[str] = []
    try:
        d = dkb
        if d is None:
            from saddle.dkb import get_dkb

            d = get_dkb()
        for dsn in d.list_designs(ctx, status="final", limit=max_designs):
            summary = (dsn.summary or dsn.ask or "").strip()
            if summary:
                designs.append(summary)
    except Exception as exc:  # noqa: BLE001
        _log.warning("project_rep: designs source failed for [%s]: %s", ctx.key, exc)

    knowledge: list[tuple[str, str, str]] = []
    try:
        q = (query or intent or ctx.project or "").strip()
        for kn in recall(ctx, q, k=recall_k, dkb=dkb):
            knowledge.append((kn.kind, kn.title, kn.body or ""))
    except Exception as exc:  # noqa: BLE001
        _log.warning("project_rep: knowledge source failed for [%s]: %s", ctx.key, exc)

    map_status = ""
    if map_probe is not None:
        try:
            map_status = (map_probe(ctx) or "").strip()
        except Exception as exc:  # noqa: BLE001 — a probe crash is a "status unknown", not a wedge
            _log.warning("project_rep: map probe failed for [%s]: %s", ctx.key, exc)
            map_status = ""

    return ProjectRep(
        project_key=getattr(ctx, "key", ""),
        intent=intent,
        designs=designs,
        knowledge=knowledge,
        map_status=map_status,
        rolling_chat=(rolling_chat or "").strip(),
        token_budget=DEFAULT_TOKEN_BUDGET,
    )


def _clip(text: str, limit: int = _CLIP) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def render(rep: ProjectRep, *, budget: int | None = None) -> str:
    """Render the rep as a compact, token-bounded 'what this project is about' block
    for the agent. Intent is foregrounded (it's the thing to act on); sections are
    added in priority order (intent → map status → designs → knowledge → recent
    conversation) and truncated when the char budget (``budget`` tokens ×
    ~4 chars/token) is reached, so the block never floods the agent. An empty rep
    renders to ''."""
    budget = budget or rep.token_budget
    char_budget = max(0, int(budget) * _CHARS_PER_TOKEN)

    header = f"PROJECT REP — {rep.project_key}: what saddle knows this project is about."
    sections: list[str] = []
    if rep.intent:
        sections.append(f"CURRENT INTENT (act on this): {_clip(rep.intent)}")
    if rep.map_status:
        sections.append(f"STRUCTURAL MAPS: {_clip(rep.map_status, 120)}")
    if rep.designs:
        lines = "\n".join(f"  - {_clip(d)}" for d in rep.designs)
        sections.append(f"SETTLED DESIGNS (the standing decisions):\n{lines}")
    if rep.knowledge:
        lines = "\n".join(f"  - [{k}] {t}: {_clip(b)}" for k, t, b in rep.knowledge)
        sections.append(f"KNOWN (facts / lessons / anti-patterns / best-practices):\n{lines}")
    if rep.rolling_chat:
        sections.append(f"RECENT CONVERSATION:\n{_clip(rep.rolling_chat, 4000)}")

    if not sections:
        return ""

    out = header
    for sec in sections:
        candidate = f"{out}\n\n{sec}"
        if len(candidate) > char_budget:
            out += "\n\n… (project rep truncated to the token budget)"
            break
        out = candidate
    return out


def rep_block(
    ctx: "Context",
    *,
    query: str = "",
    dkb: "DKB | None" = None,
    tracker: "IntentTracker | None" = None,
    map_probe: "MapProbe | None" = None,
    rolling_chat: str = "",
) -> str:
    """:func:`assemble` + :func:`render` in one call — the bounded project-rep block
    to feed the agent. Returns '' when there is nothing to say."""
    return render(assemble(
        ctx, query=query, dkb=dkb, tracker=tracker, map_probe=map_probe,
        rolling_chat=rolling_chat,
    ))
