"""intent — Stage 2's project/design/history axis (gap-2's genuine new check).

The two OTHER intent axes already exist (cross-project scope in intake, pick-drift
in dialog/replay); THIS module is the gap: does this prompt pull against what the
project has ALREADY settled — contradict a committed design, re-open a closed
decision, or creep past the focus? These tests pin that seam against a faked DKB +
LLM so the wiring (retrieve → scope-filter → classify → level) stays deterministic
and offline:

* the empty / clean-slate fast paths take NO LLM call — a brand-new project has
  nothing settled to pull against;
* only THIS tenant's settled entries reach the model — the global seed corpus is
  Stage 3's job and is dropped before the prompt is built;
* a hard contradiction / re-opened decision is an ALERT, scope-creep a NOTICE,
  and ANY hard pull escalates a mixed report to ALERT;
* a malformed reply is parsed tolerantly (junk rows dropped) but a FAILED call
  PROPAGATES (fail-loud) for ``run_stage`` to classify — it is never swallowed.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from saddle import intent
from saddle.context import Context
from saddle.models import BUBBLE_ALERT, BUBBLE_NOTICE, Design, Knowledge

CTX = Context(tenant="acme", project="game")


class FakeCaller:
    """Records each call; returns a canned JSON reply. ``forbidden=True`` turns
    any call into a failure — the assertion that the fast paths consult NO model.
    ``boom=True`` raises, to prove a failed check PROPAGATES (fail-loud)."""

    def __init__(self, response: str = "{}", *, forbidden: bool = False, boom: bool = False):
        self.response = response
        self.forbidden = forbidden
        self.boom = boom
        self.calls: list[dict] = []

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        if self.forbidden:
            raise AssertionError("history LLM was called on a fast path")
        if self.boom:
            raise RuntimeError("provider down")
        self.calls.append({"system": system, "prompt": prompt, "label": label})
        return self.response


class FakeDKB:
    """A DKB stub returning canned designs + knowledge hits, matching the two
    methods ``history_drift`` touches with their real keyword-only signatures."""

    def __init__(self, designs=None, hits=None):
        self._designs = list(designs or [])
        self._hits = list(hits or [])

    def list_designs(self, ctx, *, status=None, limit=50):
        return list(self._designs)

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        return list(self._hits)


def _run(prompt, *, caller, dkb) -> intent.IntentReport:
    return asyncio.run(intent.history_drift(prompt, CTX, caller=caller, dkb=dkb))


def _design(**kw) -> Design:
    kw.setdefault("ask", "the cache")
    return Design(**kw)


def _kn(title, body, *, tenant="acme", project="game", kind="lesson") -> Knowledge:
    return Knowledge(kind=kind, title=title, body=body,
                     scope_tenant=tenant, scope_project=project)


def _verdict(divs) -> str:
    return json.dumps({"divergences": divs})


# --- fast paths: nothing submitted / nothing settled -> no model call --------

def test_empty_prompt_is_checked_without_a_model_call():
    caller = FakeCaller(forbidden=True)
    rep = _run("   ", caller=caller, dkb=FakeDKB())
    assert rep.checked is True and rep.has_drift is False
    assert caller.calls == []                            # never consulted


def test_clean_slate_project_skips_the_llm():
    """A brand-new project with no designs and no decisions has nothing to pull
    against, so the model is never consulted — checked, considered 0, silent."""
    caller = FakeCaller(forbidden=True)
    rep = _run("rip out the LRU cache and use redis", caller=caller, dkb=FakeDKB())
    assert rep.checked is True
    assert rep.considered == 0
    assert rep.sections() == []
    assert caller.calls == []


# --- the global seed is Stage 3's job: dropped before the prompt is built -----

def test_global_seed_is_dropped_only_tenant_state_reaches_the_model():
    seed = Knowledge(kind="principle", title="no band-aids", body="keep it strict",
                     scope_tenant="", scope_project="")                  # global
    mine = _kn("cache decision", "the project chose an in-memory LRU cache")  # tenant
    caller = FakeCaller(_verdict([]))
    rep = _run("add a metrics endpoint", caller=caller,
               dkb=FakeDKB(hits=[(seed, 0.9), (mine, 0.8)]))
    assert rep.checked is True and rep.considered == 1   # only the tenant entry counts
    sent = caller.calls[0]["prompt"]
    assert "cache decision" in sent                      # the tenant decision is in scope
    assert "no band-aids" not in sent                    # the global seed is NOT


# --- aligned: settled designs present, model finds no pull -> silent ----------

def test_aligned_prompt_runs_but_stays_silent():
    designs = [_design(id="design_1", summary="in-memory LRU cache",
                       approach="bound the map, evict LRU", status="final")]
    caller = FakeCaller(_verdict([]))
    rep = _run("add a TTL knob to the LRU cache", caller=caller, dkb=FakeDKB(designs))
    assert rep.checked is True and rep.has_drift is False
    assert rep.sections() == []
    assert len(caller.calls) == 1
    sent = caller.calls[0]["prompt"]
    assert "in-memory LRU cache" in sent and "[design_1]" in sent  # the design reached the model
    assert "bound the map, evict LRU" in sent                     # incl. its approach


# --- a hard contradiction of a settled design -> ALERT ------------------------

def test_contradiction_of_a_settled_design_is_an_alert():
    designs = [_design(id="design_1", summary="in-memory LRU cache", status="final")]
    div = {"kind": "contradicts_design", "what": "switches the cache to redis",
           "why": "design_1 settled on an in-memory LRU", "ref": "design_1"}
    rep = _run("replace the cache with redis", caller=FakeCaller(_verdict([div])),
               dkb=FakeDKB(designs))
    assert rep.has_drift is True
    assert rep.level == BUBBLE_ALERT                     # a hard pull -> loud
    (section,) = rep.sections()
    assert "Conflicts with an earlier design" in section
    assert "switches the cache to redis" in section
    assert "(recorded as: design_1)" in section


# --- a re-opened decision is also hard -> ALERT -------------------------------

def test_reopened_decision_is_an_alert():
    decisions = [(_kn("cache backend", "closed: in-memory, not redis"), 0.9)]
    div = {"kind": "reopens_decision", "what": "re-asks the cache-backend choice",
           "why": "that decision was closed in favor of in-memory"}
    rep = _run("should we use redis for the cache?",
               caller=FakeCaller(_verdict([div])), dkb=FakeDKB(hits=decisions))
    assert rep.level == BUBBLE_ALERT
    assert "Goes against an earlier decision" in rep.sections()[0]


# --- scope-creep alone is a NOTICE; any hard pull escalates a mix -------------

def test_scope_creep_alone_is_a_notice():
    designs = [_design(id="design_1", summary="the 2D camera system")]
    div = {"kind": "scope_creep", "what": "builds a separate launcher app",
           "why": "the project's focus is the camera engine"}
    rep = _run("also build a standalone launcher",
               caller=FakeCaller(_verdict([div])), dkb=FakeDKB(designs))
    assert rep.has_drift is True
    assert rep.level == BUBBLE_NOTICE                    # a soft pull -> notice
    assert "Outside this project's current focus" in rep.sections()[0]


def test_mixed_findings_escalate_to_alert():
    designs = [_design(id="design_1", summary="LRU cache")]
    divs = [
        {"kind": "scope_creep", "what": "adds a launcher"},
        {"kind": "contradicts_design", "what": "rips out the LRU"},
    ]
    rep = _run("rebuild it", caller=FakeCaller(_verdict(divs)), dkb=FakeDKB(designs))
    assert len(rep.divergences) == 2
    assert rep.level == BUBBLE_ALERT                     # ANY hard pull -> alert


# --- tolerant parse of a present reply vs fail-loud on a FAILED call ----------

def test_malformed_rows_are_dropped_not_raised():
    designs = [_design(id="design_1", summary="LRU cache")]
    divs = [
        {"kind": "contradicts_design", "what": "real pull"},   # kept
        {"kind": "made_up_kind", "what": "dropped"},            # unknown kind
        {"kind": "scope_creep", "what": ""},                   # empty what
        "not even a dict",                                     # junk
    ]
    rep = _run("change things", caller=FakeCaller(_verdict(divs)), dkb=FakeDKB(designs))
    assert [d.kind for d in rep.divergences] == ["contradicts_design"]
    assert rep.checked is True


def test_a_failed_call_propagates_for_run_stage_to_classify():
    """The fail-loud seam: history_drift does NOT swallow a failed classify. A
    raising caller propagates OUT so run_stage classifies + bubbles an ALERT —
    never a silent 'looked fine'."""
    designs = [_design(id="design_1", summary="LRU cache")]
    with pytest.raises(RuntimeError, match="provider down"):
        _run("change things", caller=FakeCaller(boom=True), dkb=FakeDKB(designs))


# --- the result dataclasses' contract ----------------------------------------

def test_intent_divergence_hardness_and_render():
    hard = intent.IntentDivergence(kind=intent.REOPENS_DECISION, what="x", ref="r")
    soft = intent.IntentDivergence(kind=intent.SCOPE_CREEP, what="y")
    assert hard.hard is True and soft.hard is False
    rendered = hard.render()
    assert "Goes against an earlier decision" in rendered and "(recorded as: r)" in rendered


def test_project_scoped_keeps_tenant_drops_global_and_other_tenants():
    seed = Knowledge(kind="principle", title="seed", body="b",
                     scope_tenant="", scope_project="")               # global
    mine = _kn("mine", "b")                                           # acme/game
    other = Knowledge(kind="lesson", title="other", body="b",
                      scope_tenant="other", scope_project="x")        # different tenant
    kept = intent._project_scoped([(seed, 1.0), (mine, 0.9), (other, 0.8)], CTX)
    assert [k.title for k in kept] == ["mine"]
