"""Recall — bounded, on-demand retrieval (the read face of saddle's memory).

Pins the contract that keeps memory from ballooning context: top-k bounded,
query-scoped, kinds-filterable, length-clipped, and best-effort (a retrieval
failure yields nothing rather than sinking the caller). A stub DKB keeps these
offline and deterministic — the DKB's own hybrid ranking is tested elsewhere.
"""
from __future__ import annotations

from saddle.context import Context
from saddle.models import FACT, REFERENCE, Knowledge
from saddle.recall import format_recall, recall, recall_block

CTX = Context(tenant="acme", project="game")


class _StubDKB:
    """Records the search args; returns canned (Knowledge, score) hits, honoring
    ``k`` so the bounding assertion is real. ``boom=True`` raises, to prove recall
    swallows a retrieval failure."""

    def __init__(self, hits=None, *, boom=False):
        self._hits = hits or []
        self.boom = boom
        self.calls: list[dict] = []

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        if self.boom:
            raise RuntimeError("retrieval down")
        self.calls.append({"query": query, "k": k, "kinds": kinds})
        return self._hits[:k]


def _kn(title, kind=FACT, body="body"):
    return Knowledge(kind=kind, title=title, body=body)


def test_recall_returns_entries_best_first_and_bounded():
    hits = [(_kn("a"), 0.9), (_kn("b"), 0.8), (_kn("c"), 0.7)]
    d = _StubDKB(hits)
    out = recall(CTX, "query", k=2, dkb=d)
    assert [k.title for k in out] == ["a", "b"]             # top-k, order preserved
    assert d.calls[0]["k"] == 2


def test_recall_passes_kinds_filter_through():
    d = _StubDKB([(_kn("f"), 1.0)])
    recall(CTX, "q", kinds=[FACT, REFERENCE], dkb=d)
    assert d.calls[0]["kinds"] == [FACT, REFERENCE]


def test_recall_empty_query_skips_retrieval():
    d = _StubDKB([(_kn("x"), 1.0)])
    assert recall(CTX, "   ", dkb=d) == []
    assert d.calls == []                                    # no search at all


def test_recall_disabled_when_k_non_positive():
    d = _StubDKB([(_kn("x"), 1.0)])
    assert recall(CTX, "q", k=0, dkb=d) == []
    assert d.calls == []


def test_recall_env_k_default(monkeypatch):
    monkeypatch.setenv("SADDLE_RECALL_K", "1")
    d = _StubDKB([(_kn("a"), 1.0), (_kn("b"), 0.5)])
    recall(CTX, "q", dkb=d)                                 # no explicit k -> env
    assert d.calls[0]["k"] == 1


def test_recall_is_best_effort_on_failure():
    # Memory is enrichment, never a gate: a thrown retrieval yields [], not an error.
    assert recall(CTX, "q", dkb=_StubDKB(boom=True)) == []


# -- formatting --------------------------------------------------------------

def test_format_recall_empty_is_empty_string():
    assert format_recall([]) == ""


def test_format_recall_renders_kind_title_body():
    block = format_recall([_kn("Identity", body="saddle is the harness")])
    assert "RELEVANT KNOWLEDGE" in block
    assert "[fact] Identity: saddle is the harness" in block


def test_format_recall_clips_long_bodies():
    long_body = "word " * 200
    block = format_recall([_kn("T", body=long_body)], clip=50)
    # the entry line is bounded near the clip (+ ellipsis + label), never the full body
    entry_line = [ln for ln in block.splitlines() if ln.startswith("- [")][0]
    assert "…" in entry_line
    assert len(entry_line) < 120


def test_recall_block_combines_recall_and_format():
    d = _StubDKB([(_kn("Identity", body="saddle is the harness"), 1.0)])
    block = recall_block(CTX, "what is saddle", dkb=d)
    assert "Identity" in block and "RELEVANT KNOWLEDGE" in block
