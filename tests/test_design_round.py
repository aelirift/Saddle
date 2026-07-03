"""The design round-trip (mediator design slice 3): settlement of an
agent-authored approach, the MCP design_propose round, and the turn-end drain
that delivers Stop-hook findings to the next turn's agent context.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from saddle.context import Context
from saddle.models import DESIGN_FINAL

_CTX = Context(tenant="acme", project="game")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.delenv("SADDLE_CODE_ROOT", raising=False)
    return tmp_path


class _IndexCaller:
    """Answers the settle-index call; the surface stage no-ops (no code root)."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.labels.append(label)
        assert label == "design/settle-index"
        return json.dumps({
            "summary": "bound the cache with an LRU",
            "satisfies": ["cache stays bounded"],
            "avoids": ["unbounded growth"],
            "heeds": ["no band-aids"],
        })


class _StubDKB:
    def __init__(self) -> None:
        self.designs: list = []

    def add_design(self, ctx, design):
        self.designs.append((ctx, design))
        return design


def test_settle_approach_persists_final_design():
    from saddle.design import settle_approach

    dkb = _StubDKB()
    design = asyncio.run(settle_approach(
        "bound the cache", "Root cause: unbounded map. Add an LRU with eviction.",
        _CTX, approved_by="user", caller=_IndexCaller(), dkb=dkb,
    ))
    assert design.status == DESIGN_FINAL
    assert design.summary == "bound the cache with an LRU"
    assert design.meta["approved_by"] == "user"
    assert design.meta["settled_from"] == "live-session"
    assert "surface" in design.meta
    assert len(dkb.designs) == 1 and dkb.designs[0][0] is _CTX


def test_settle_approach_rejects_empty():
    from saddle.design import settle_approach

    with pytest.raises(ValueError):
        asyncio.run(settle_approach("goal", "  ", _CTX, caller=_IndexCaller()))


def test_design_propose_round_critique_then_settle(monkeypatch):
    """The MCP round: issues -> critique text (NOT settled); clean -> SETTLED."""
    from saddle import mcp_server
    from saddle.design import AuditVerdict
    from saddle.models import Design

    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    rounds = {"n": 0}

    async def _audit(goal, approach, ctx=None, **kw):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return AuditVerdict(ok=False, issues=["hand-waving: no mechanism"])
        return AuditVerdict(ok=True, issues=[])

    settled = {}

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        settled["approved_by"] = approved_by
        return Design(ask=goal, summary="s", approach=approach, body=approach,
                      status=DESIGN_FINAL, id="design_x1")

    monkeypatch.setattr("saddle.design.audit_proposal", _audit)
    monkeypatch.setattr("saddle.design.settle_approach", _settle)

    first = asyncio.run(mcp_server.design_propose(
        "fix the cache", "I'll just make it better."))
    assert first.startswith("NOT SETTLED") and "hand-waving" in first
    assert "approved_by" not in settled

    second = asyncio.run(mcp_server.design_propose(
        "fix the cache", "Root cause: unbounded map; add LRU, evict at N."))
    assert second.startswith("SETTLED") and "design_x1" in second
    assert settled["approved_by"] == "converged"


def test_drain_delivers_turn_end_findings_once(monkeypatch):
    """Origin-tagged Stop-hook findings reach the NEXT turn's context exactly
    once; mid-turn bubbles (no origin tag) are never duplicated."""
    from saddle.bubble import emit_bubble
    from saddle.intake_hook import _drained_findings

    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    emit_bubble(_CTX, "mid-turn design alert the agent already saw",
                stage="design", session="s1", meta={"issues": ["x"]})
    emit_bubble(_CTX, "━━ saddle [acme/game] ━━\ncode drifted from design d1",
                stage="code", session="s1",
                meta={"origin": "turn-end", "drifts": []})
    other = Context(tenant="acme", project="sibling")
    emit_bubble(other, "sibling lesson filed", stage="lesson", session="s1",
                meta={"origin": "turn-end"})

    out = _drained_findings(_CTX, "s1")
    assert "code drifted from design d1" in out
    assert "[sibling] sibling lesson filed" in out
    assert "mid-turn design alert" not in out
    assert "━━" not in out  # banner stripped — the drain re-frames

    # cursor advanced: a second drain has nothing new
    assert _drained_findings(_CTX, "s1") == ""
