"""The design council (Part 2): blast-radius triage (skip on a tiny surface),
distinct-family membership dedup, drop-and-quorum fallback, and the full offline
convene path — synthesis, the audit floor that consensus cannot launder past,
user forks on an unresolved conflict, and settlement on a clean synthesis.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from saddle import council
from saddle.codemap import SurfaceManifest
from saddle.context import Context
from saddle.council import CouncilResult, convene, model_family, select_members

_CTX = Context(tenant="acme", project="game")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.delenv("SADDLE_CODE_ROOT", raising=False)
    return tmp_path


# -- membership: distinct model families, never two clones -------------------

def test_model_family_collapses_variants():
    assert model_family("claude_agent") == "anthropic"
    assert model_family("deepseek_pro") == model_family("deepseek_flash") == "deepseek"
    assert model_family("minimax") == "minimax"
    assert model_family("kimi") == "moonshot"


def test_select_members_dedups_by_family():
    # Two DeepSeek tiers are ONE family -> only one critic seat, not two clones.
    chair, critics = select_members(
        {"deepseek_pro", "deepseek_flash"},
        {"members": ["deepseek_pro", "deepseek_flash"]},
        ["deepseek_pro", "deepseek_flash"],
    )
    assert critics == ["deepseek_pro"]


def test_select_members_seats_two_distinct_families():
    chair, critics = select_members(
        {"claude_agent", "deepseek_pro", "deepseek_flash", "minimax"},
        {"members": ["claude_agent", "minimax"], "synthesis": "claude_agent"},
        ["claude_agent", "minimax"],
    )
    assert chair == "claude_agent"
    assert critics == ["claude_agent", "minimax"]           # two families
    assert model_family(critics[0]) != model_family(critics[1])


# -- triage: a tiny surface skips the council entirely (no LLM) ---------------

class _Boom:
    async def __call__(self, *a, **k):
        raise AssertionError("no LLM call expected on a triaged-out surface")


def test_tiny_surface_skips_the_council():
    result = asyncio.run(convene(
        "add a log line", "print it", _CTX,
        surface=SurfaceManifest(),                          # empty -> 0 sites
        callers={"claude_agent": _Boom(), "minimax": _Boom(), "default": _Boom()},
    ))
    assert result.convened is False
    assert result.fell_back is False                        # trivial, not a failure


def _wide_surface() -> SurfaceManifest:
    return SurfaceManifest.from_dict({
        "lifecycle": [{"name": "a", "symbol": "cfg_a"},
                      {"name": "b", "symbol": "cfg_b"}],
    })


# -- a fake caller that answers critiques (JSON) and synthesis (prose) --------

class _Caller:
    def __init__(self, *, critique=None, synth=None, boom=False):
        self._critique = critique or {
            "verdict": "pass", "concern": "holds up", "severity": "advisory",
            "recommendation": "",
        }
        self._synth = synth or "The reconciled design: add a resolver and route every consumer through it."
        self._boom = boom
        self.labels: list[str] = []

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.labels.append(label)
        if self._boom:
            raise RuntimeError("critic down")
        if json_mode:
            return json.dumps(self._critique)
        return self._synth


@pytest.fixture
def _stub_floor(monkeypatch):
    """Stub the audit floor (clean by default) + settle, so convene runs offline.
    Returns a dict the test can flip to make the floor return issues."""
    from saddle.design import AuditVerdict
    from saddle.models import Design

    state = {"issues": []}

    async def _audit(goal, approach, ctx=None, **kw):
        return AuditVerdict(ok=not state["issues"], issues=list(state["issues"]))

    settled: list[str] = []

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        assert approved_by == "council"
        settled.append(approach)
        return Design(ask=goal, summary="reconciled", id="design_c", status="final")

    monkeypatch.setattr("saddle.design.audit_proposal", _audit)
    monkeypatch.setattr("saddle.design.settle_approach", _settle)
    state["settled"] = settled
    return state


def _callers(**over):
    base = {
        "claude_agent": _Caller(critique={"verdict": "flaw", "concern": "coupling",
                                           "severity": "advisory", "recommendation": "invert"}),
        "minimax": _Caller(critique={"verdict": "pass", "concern": "covers the goal",
                                     "severity": "advisory", "recommendation": ""}),
        "default": _Caller(),
    }
    base.update(over)
    return base


# -- the happy path: two critiques -> synthesis -> floor -> settle ------------

def test_convene_settles_a_clean_synthesis(_stub_floor):
    callers = _callers()
    result = asyncio.run(convene(
        "add a cooldown modifier", "wire it through the resolver", _CTX,
        surface=_wide_surface(), callers=callers,
    ))
    assert result.convened and not result.fell_back
    assert result.design_id == "design_c"
    assert result.settled is True
    assert result.members == ["claude_agent", "minimax"]     # longevity, root-cause
    assert len(result.critiques) == 2 and all(c.ok for c in result.critiques)
    assert _stub_floor["settled"]                            # settle_approach ran
    # the longevity lens went to the strongest (chair) model
    assert "council/longevity" in callers["claude_agent"].labels
    assert "council/root-cause" in callers["minimax"].labels
    assert "council/synthesis" in callers["claude_agent"].labels


# -- the audit floor: consensus cannot launder a band-aid --------------------

def test_audit_floor_blocks_settle(_stub_floor):
    _stub_floor["issues"] = ["band-aid: swallow-and-log instead of a fix"]
    result = asyncio.run(convene(
        "fix the crash", "wrap it in try/except", _CTX,
        surface=_wide_surface(), callers=_callers(),
    ))
    assert result.convened is True
    assert result.design_id == ""                           # NOT settled
    assert result.audit_issues == ["band-aid: swallow-and-log instead of a fix"]
    assert not _stub_floor["settled"]


# -- an unresolved conflict is surfaced as a user fork, never settled --------

def test_unresolved_conflict_becomes_a_user_fork(_stub_floor):
    forked = _Caller(synth="UNRESOLVED FORK: in-memory cache vs redis?\nOption A ...\nOption B ...")
    result = asyncio.run(convene(
        "choose the cache backend", "use redis", _CTX,
        surface=_wide_surface(),
        callers=_callers(claude_agent=forked),
    ))
    assert result.convened is True
    assert result.forks == ["in-memory cache vs redis?"]
    assert result.design_id == ""                           # not settled by fiat
    assert not _stub_floor["settled"]


# -- drop-and-quorum: both critics fail -> fall back loudly, never wedge ------

def test_quorum_failure_falls_back(_stub_floor):
    result = asyncio.run(convene(
        "big change", "do it", _CTX,
        surface=_wide_surface(),
        callers=_callers(claude_agent=_Caller(boom=True), minimax=_Caller(boom=True)),
    ))
    assert result.convened is True
    assert result.fell_back is True                         # caller runs the single audit
    assert result.design_id == ""
    assert "critics returned" in result.dissent


def test_one_critic_survives_quorum(_stub_floor):
    # min_quorum defaults to 1: one dropped critic still yields a verdict.
    result = asyncio.run(convene(
        "big change", "do it", _CTX,
        surface=_wide_surface(),
        callers=_callers(minimax=_Caller(boom=True)),
    ))
    assert result.convened is True
    assert result.fell_back is False
    assert result.design_id == "design_c"                   # synthesised on the survivor


# -- an unformable panel (one family only) falls back ------------------------

def test_single_family_panel_falls_back(_stub_floor, monkeypatch):
    # Only one family available -> cannot seat two distinct critics.
    result = asyncio.run(convene(
        "big change", "do it", _CTX,
        surface=_wide_surface(),
        callers={"claude_agent": _Caller(), "default": _Caller()},
    ))
    assert result.convened is False
    assert result.fell_back is True
    assert "distinct-family" in result.dissent
