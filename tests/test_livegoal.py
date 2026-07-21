"""The live-goal layer (#76): proposal persistence, the committed-goal render, and
the bubble-to-confirm turn logic (topic-move -> propose; user's yes/no -> retire /
keep) — the LLM classifiers are stubbed, so this is offline and deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

from saddle.context import Context
from saddle.livegoal import (
    CONFIRM,
    CONFIRM_NONE,
    DECLINE,
    KEPT,
    MOVE_PROPOSED,
    NOOP,
    RETIRED,
    LiveGoalProposal,
    TopicVerdict,
    clear_proposal,
    commitment_gist,
    livegoal_turn,
    read_proposal,
    write_proposal,
)
from saddle.models import Binding, Fork, ForkOption

_CTX = Context(tenant="acme", project="game")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    return tmp_path


# -- persistence -------------------------------------------------------------

def test_proposal_round_trips_and_clears():
    p = LiveGoalProposal(fork_id="p1.f1", committed_gist="build X",
                         moved_to_gist="the ledger", anchor="u2")
    assert not read_proposal("s1").pending
    write_proposal("s1", p)
    got = read_proposal("s1")
    assert got.pending and got.fork_id == "p1.f1"
    assert got.committed_gist == "build X" and got.moved_to_gist == "the ledger"
    assert got.updated_ts > 0
    clear_proposal("s1")
    assert not read_proposal("s1").pending


def test_proposal_write_is_atomic_no_temp_left(tmp_path):
    write_proposal("s1", LiveGoalProposal(fork_id="p1.f1"))
    files = sorted(p.name for p in (tmp_path / "live_goal").iterdir())
    assert files == ["s1.json"]  # temp replaced, none left behind


def test_corrupt_proposal_reads_as_none(tmp_path):
    d = tmp_path / "live_goal"
    d.mkdir(parents=True)
    (d / "s1.json").write_text("{not json", encoding="utf-8")
    assert not read_proposal("s1").pending  # garbled -> none, never a false retire


# -- commitment_gist ---------------------------------------------------------

def _fork(prompt, opts):
    return Fork(options=[ForkOption(label=l, text=t) for l, t in opts], prompt=prompt,
                id="p1.f1")


def test_commitment_gist_renders_prompt_and_chosen_option():
    fork = _fork("how to retire a fork?", [("a", "bubble to confirm"),
                                           ("b", "auto-supersede")])
    b = Binding(fork_id="p1.f1", label="a", resolved=True)
    g = commitment_gist(b, fork)
    assert "how to retire a fork" in g and "bubble to confirm" in g and "a)" in g


def test_commitment_gist_falls_back_when_fork_missing():
    b = Binding(fork_id="p1.f1", label="a", choice_id="p1.f1.a", user_text="do a")
    g = commitment_gist(b, None)
    assert "p1.f1.a" in g


# -- the turn logic (classifiers stubbed) ------------------------------------

class _StubTracker:
    def __init__(self, binding=None, fork=None):
        self._binding, self._fork = binding, fork
        self.superseded: list[str] = []

    def committed_fork(self, ctx, *, session=""):
        return self._binding, self._fork

    def supersede(self, ctx, fork_id, *, session=""):
        self.superseded.append(fork_id)
        return True


def _committed():
    fork = _fork("gate the council?", [("a", "non-trivial only")])
    return Binding(fork_id="p1.f1", label="a", resolved=True), fork


def _run(prompt, session, tracker):
    return asyncio.run(livegoal_turn(prompt, session, ctx=_CTX, tracker=tracker))


@pytest.fixture
def _stub_move(monkeypatch):
    box = {"moved": False, "moved_to": ""}

    async def _fake(gist, prompt, ctx=None, **kw):
        return TopicVerdict(moved=box["moved"], moved_to=box["moved_to"], confidence=0.9)

    monkeypatch.setattr("saddle.livegoal.classify_topic_move", _fake)
    return box


@pytest.fixture
def _stub_confirm(monkeypatch):
    box = {"kind": CONFIRM_NONE}

    async def _fake(committed, moved_to, prompt, ctx=None, **kw):
        return box["kind"]

    monkeypatch.setattr("saddle.livegoal.classify_supersede_confirm", _fake)
    return box


def test_no_commitment_is_noop_and_clears_stale_proposal(_stub_move, _stub_confirm):
    write_proposal("s", LiveGoalProposal(fork_id="p9.f9"))  # stale
    tr = _StubTracker(binding=None, fork=None)
    out = _run("anything", "s", tr)
    assert out.action == NOOP
    assert not read_proposal("s").pending          # stale proposal dropped


def test_move_detected_bubbles_a_proposal_without_retiring(_stub_move, _stub_confirm):
    _stub_move["moved"] = True
    _stub_move["moved_to"] = "the item ledger"
    b, f = _committed()
    tr = _StubTracker(binding=b, fork=f)
    out = _run("now let's build the item ledger", "s", tr)
    assert out.action == MOVE_PROPOSED
    assert tr.superseded == []                      # NOT retired
    p = read_proposal("s")
    assert p.pending and p.fork_id == "p1.f1" and p.moved_to_gist == "the item ledger"
    assert "retire" in out.herald.lower()


def test_no_move_is_a_silent_noop(_stub_move, _stub_confirm):
    _stub_move["moved"] = False
    b, f = _committed()
    tr = _StubTracker(binding=b, fork=f)
    out = _run("ok what about the tests for that", "s", tr)
    assert out.action == NOOP and out.herald == ""
    assert not read_proposal("s").pending


def test_pending_confirm_retires_the_fork(_stub_move, _stub_confirm):
    write_proposal("s", LiveGoalProposal(fork_id="p1.f1", committed_gist="gate council"))
    _stub_confirm["kind"] = CONFIRM
    b, f = _committed()
    tr = _StubTracker(binding=b, fork=f)
    out = _run("yes, retire it", "s", tr)
    assert out.action == RETIRED
    assert tr.superseded == ["p1.f1"]               # user-confirmed retirement
    assert not read_proposal("s").pending


def test_pending_decline_keeps_the_commitment(_stub_move, _stub_confirm):
    write_proposal("s", LiveGoalProposal(fork_id="p1.f1", committed_gist="gate council"))
    _stub_confirm["kind"] = DECLINE
    b, f = _committed()
    tr = _StubTracker(binding=b, fork=f)
    out = _run("no, I'm still on that", "s", tr)
    assert out.action == KEPT
    assert tr.superseded == []                      # commitment stands
    assert not read_proposal("s").pending


def test_pending_none_leaves_the_proposal_pending(_stub_move, _stub_confirm):
    write_proposal("s", LiveGoalProposal(fork_id="p1.f1", committed_gist="gate council"))
    _stub_confirm["kind"] = CONFIRM_NONE
    b, f = _committed()
    tr = _StubTracker(binding=b, fork=f)
    out = _run("what does this function do?", "s", tr)
    assert out.action == NOOP
    assert tr.superseded == []
    assert read_proposal("s").pending               # still awaiting a yes/no


def test_pending_proposal_for_a_stale_fork_is_dropped(_stub_move, _stub_confirm):
    # A proposal was raised for p1.f1, but the commitment has since advanced to
    # p2.f2 (a fresh pick). The stale proposal no longer applies.
    write_proposal("s", LiveGoalProposal(fork_id="p1.f1"))
    b = Binding(fork_id="p2.f2", label="a", resolved=True)
    tr = _StubTracker(binding=b, fork=_fork("newer?", [("a", "x")]))
    _stub_confirm["kind"] = CONFIRM  # even a "yes" must not retire the wrong fork
    out = _run("yes", "s", tr)
    assert tr.superseded == []
    assert not read_proposal("s").pending
