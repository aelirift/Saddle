"""The design-hold posture (Part 1): the ORTHOGONAL per-session review axis that
gates the design gate's auto-settle on the USER's own request to be in the loop.

Covers: posture read/write round-trip + atomicity, the fail postures (absent =>
DEFAULT silent, corrupt => DEFAULT + loud), the full transition table (each
cell), the fail-closed ``design_in_play`` / ``user_approved_current`` decision
helpers, and the classifier's schema + the asymmetric mapping (negation /
conditional => REQUEST_REVIEW, never APPROVE) through a stubbed caller.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from saddle import hold
from saddle.context import Context
from saddle.hold import (
    APPROVE,
    AUTONOMY_GRANT,
    HOLD_AUTONOMOUS,
    HOLD_DEFAULT,
    HOLD_HELD_STATE,
    INTERVENE,
    NONE,
    REQUEST_REVIEW,
    SCOPE_ONCE,
    SCOPE_STANDING,
    HoldPosture,
    ReviewIntent,
    apply_review_intent,
    classify_review_intent,
    design_in_play,
    read_posture,
    user_approved_current,
    write_posture,
)

_CTX = Context(tenant="acme", project="game")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    return tmp_path


def _apply(session, intent, **kw):
    return asyncio.run(apply_review_intent(session, intent, **kw))


# -- persistence: round-trip + atomicity + fail postures ---------------------

def test_posture_round_trips():
    p = HoldPosture(state=HOLD_HELD_STATE, hold_scope=SCOPE_STANDING,
                    held={"anchor": "u1", "goal": "build X", "approach_digest": "plan"},
                    approved_design_id="design_x", autonomy_since_anchor="")
    write_posture("s1", p)
    got = read_posture("s1")
    assert got.state == HOLD_HELD_STATE
    assert got.hold_scope == SCOPE_STANDING
    assert got.held["goal"] == "build X"
    assert got.approved_design_id == "design_x"
    assert got.updated_ts > 0


def test_posture_write_is_atomic_no_temp_left(tmp_path):
    write_posture("s1", HoldPosture(state=HOLD_AUTONOMOUS))
    d = tmp_path / "design_hold"
    files = sorted(p.name for p in d.iterdir())
    assert files == ["s1.json"]  # temp file replaced, none left behind


def test_absent_posture_is_default_and_silent(capsys):
    got = read_posture("never-written")
    assert got.state == HOLD_DEFAULT
    assert capsys.readouterr().err == ""  # fail-open, no noise


def test_corrupt_posture_is_default_and_loud(tmp_path, capsys):
    p = hold._posture_path("s1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    got = read_posture("s1")
    assert got.state == HOLD_DEFAULT
    assert "corrupt" in capsys.readouterr().err.lower()


def test_unknown_state_falls_back_to_default(tmp_path):
    p = hold._posture_path("s1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"state": "banana"}), encoding="utf-8")
    assert read_posture("s1").state == HOLD_DEFAULT


# -- the transition table: every cell ----------------------------------------

def test_default_plus_request_review_holds():
    t = _apply("s", ReviewIntent(kind=REQUEST_REVIEW, scope=SCOPE_ONCE),
               posture=HoldPosture(state=HOLD_DEFAULT),
               goal="build X", approach="my plan", anchor="u1")
    assert t.posture.state == HOLD_HELD_STATE
    assert t.posture.held["goal"] == "build X"
    assert t.posture.approved_design_id == ""       # gate armed
    assert read_posture("s").state == HOLD_HELD_STATE  # persisted


def test_hold_plus_approve_once_settles_and_releases(monkeypatch):
    from saddle.models import Design

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        assert approved_by == "user"
        assert approach == "the agreed plan"
        return Design(ask=goal, id="design_z", status="final")

    monkeypatch.setattr("saddle.design.settle_approach", _settle)
    posture = HoldPosture(state=HOLD_HELD_STATE, hold_scope=SCOPE_ONCE,
                          held={"goal": "build X"})
    t = _apply("s", ReviewIntent(kind=APPROVE), posture=posture,
               goal="looks good", approach="the agreed plan", anchor="u2", ctx=_CTX)
    assert t.posture.state == HOLD_DEFAULT            # once -> released
    assert t.posture.approved_design_id == "design_z"
    assert t.design_id == "design_z"
    assert "approval" in t.herald.lower() or "approved" in t.herald.lower()


def test_hold_plus_approve_standing_stays_held_but_opens_gate(monkeypatch):
    from saddle.models import Design

    async def _settle(goal, approach, ctx=None, *, approved_by="converged", **kw):
        return Design(ask=goal, id="design_s", status="final")

    monkeypatch.setattr("saddle.design.settle_approach", _settle)
    posture = HoldPosture(state=HOLD_HELD_STATE, hold_scope=SCOPE_STANDING,
                          held={"goal": "build X"})
    t = _apply("s", ReviewIntent(kind=APPROVE), posture=posture,
               goal="go", approach="plan", anchor="u2", ctx=_CTX)
    assert t.posture.state == HOLD_HELD_STATE         # standing -> stays held
    assert t.posture.approved_design_id == "design_s"  # but the gate is now open


def test_hold_plus_approve_without_plan_still_opens_gate():
    # No approach prose to settle -> the approval still records a sentinel so the
    # gate opens (the user approved a verbal discussion).
    posture = HoldPosture(state=HOLD_HELD_STATE, hold_scope=SCOPE_STANDING,
                          held={"goal": "g"})
    t = _apply("s", ReviewIntent(kind=APPROVE), posture=posture,
               goal="yes", approach="", anchor="u3", ctx=_CTX)
    assert t.posture.approved_design_id != ""
    assert t.design_id == ""                          # nothing settled


def test_any_plus_autonomy_grant_goes_autonomous():
    for start in (HOLD_DEFAULT, HOLD_HELD_STATE, HOLD_AUTONOMOUS):
        t = _apply("s", ReviewIntent(kind=AUTONOMY_GRANT),
                   posture=HoldPosture(state=start), anchor="u1")
        assert t.posture.state == HOLD_AUTONOMOUS
        assert t.posture.autonomy_since_anchor == "u1"
        assert "autonomous" in t.herald.lower()


def test_hold_plus_intervene_returns_to_default():
    t = _apply("s", ReviewIntent(kind=INTERVENE),
               posture=HoldPosture(state=HOLD_HELD_STATE, held={"goal": "g"}))
    assert t.posture.state == HOLD_DEFAULT
    assert t.posture.held == {}
    assert t.herald  # released -> heralded


def test_autonomous_plus_intervene_returns_to_default():
    t = _apply("s", ReviewIntent(kind=INTERVENE),
               posture=HoldPosture(state=HOLD_AUTONOMOUS))
    assert t.posture.state == HOLD_DEFAULT


def test_autonomous_plus_none_is_sticky():
    t = _apply("s", ReviewIntent(kind=NONE),
               posture=HoldPosture(state=HOLD_AUTONOMOUS, autonomy_since_anchor="u1"))
    assert t.posture.state == HOLD_AUTONOMOUS          # STICKY
    assert t.herald == ""                              # no flip to herald


def test_autonomous_plus_request_review_holds():
    t = _apply("s", ReviewIntent(kind=REQUEST_REVIEW),
               posture=HoldPosture(state=HOLD_AUTONOMOUS, autonomy_since_anchor="u1"),
               goal="new plan please", anchor="u2")
    assert t.posture.state == HOLD_HELD_STATE          # autonomy auto-cleared
    assert t.posture.autonomy_since_anchor == ""


def test_default_plus_none_stays_default():
    t = _apply("s", ReviewIntent(kind=NONE), posture=HoldPosture(state=HOLD_DEFAULT))
    assert t.posture.state == HOLD_DEFAULT


def test_default_plus_approve_is_a_noop():
    # Nothing is held, so an "approve" cannot open a gate that isn't there.
    t = _apply("s", ReviewIntent(kind=APPROVE), posture=HoldPosture(state=HOLD_DEFAULT))
    assert t.posture.state == HOLD_DEFAULT
    assert t.posture.approved_design_id == ""


def test_request_review_rearms_the_gate_over_a_prior_approval():
    posture = HoldPosture(state=HOLD_HELD_STATE, held={"goal": "old"},
                          approved_design_id="design_old")
    t = _apply("s", ReviewIntent(kind=REQUEST_REVIEW), posture=posture,
               goal="new plan", anchor="u9")
    assert t.posture.state == HOLD_HELD_STATE
    assert t.posture.approved_design_id == ""          # prior approval voided
    assert t.posture.held["goal"] == "new plan"


# -- the pure decision helpers -----------------------------------------------

class _Turn:
    def __init__(self, approach=""):
        self.approach = approach


def test_design_in_play_is_fail_closed_under_hold():
    # No plan prose, nothing held on the object — but an active HOLD forces True.
    assert design_in_play(_Turn(approach=""), HoldPosture(state=HOLD_HELD_STATE)) is True


def test_design_in_play_default_needs_a_plan_or_held():
    assert design_in_play(_Turn(approach=""), HoldPosture(state=HOLD_DEFAULT)) is False
    assert design_in_play(_Turn(approach="a plan"), HoldPosture(state=HOLD_DEFAULT)) is True
    assert design_in_play(_Turn(approach=""),
                          HoldPosture(state=HOLD_DEFAULT, held={"goal": "g"})) is True


def test_user_approved_current_keys_on_approved_design_id():
    assert user_approved_current(HoldPosture(state=HOLD_HELD_STATE)) is False
    assert user_approved_current(
        HoldPosture(state=HOLD_HELD_STATE, approved_design_id="design_x")) is True


# -- the classifier: schema + the asymmetric negation/conditional mapping ----

class _CannedCaller:
    """Returns a fixed JSON reply and records the system prompt it saw."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.system = ""

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        assert json_mode and label == "hold/review-intent"
        self.system = system
        return json.dumps(self.doc)


def _classify(prompt, doc, *, posture=None):
    caller = _CannedCaller(doc)
    intent = asyncio.run(classify_review_intent(
        prompt, _CTX, posture=posture or HoldPosture(), held_summary="", caller=caller,
    ))
    return intent, caller


def test_negation_maps_to_request_review_not_approve():
    intent, caller = _classify("do NOT go forward with the edits",
                               {"kind": "request_review", "scope": "once",
                                "condition": "", "confidence": 0.9})
    assert intent.kind == REQUEST_REVIEW
    # the binding negation/asymmetric instructions ride every call
    assert "RESOLVE NEGATION" in caller.system
    assert "ASYMMETRIC CAUTION" in caller.system


def test_conditional_maps_to_request_review_with_condition():
    intent, _ = _classify("do not go forward unless the tests pass",
                          {"kind": "request_review", "scope": "standing",
                           "condition": "the tests pass", "confidence": 0.8})
    assert intent.kind == REQUEST_REVIEW
    assert intent.scope == SCOPE_STANDING
    assert intent.condition == "the tests pass"


def test_hallucinated_kind_defaults_to_none():
    intent, _ = _classify("carry on", {"kind": "banana", "confidence": 1.0})
    assert intent.kind == NONE


def test_empty_prompt_skips_the_model():
    class _Boom:
        async def __call__(self, *a, **k):
            raise AssertionError("no model call on an empty prompt")

    intent = asyncio.run(classify_review_intent(
        "   ", _CTX, posture=HoldPosture(), caller=_Boom()))
    assert intent.kind == NONE


def test_classifier_failure_propagates_fail_loud():
    class _Boom:
        async def __call__(self, *a, **k):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(classify_review_intent(
            "show me the plan first", _CTX, posture=HoldPosture(), caller=_Boom()))
