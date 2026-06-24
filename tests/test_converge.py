"""The convergence controller — drive the coder until the design's surface holds.

These tests pin the LOOP, not the LLM. The coder is faked (it performs a scripted
file edit, or crashes) and the gate is either injected (scripted gap sequences) or
the REAL :func:`saddle.design.intent_drift` run against a tiny tree the fake coder
fixes. That proves the load-bearing guarantees deterministically: gate after every
turn, re-prompt with the actual findings, never trust a self-claim, and the three
self-terminating exits (converge / stall / exhaust) plus halt-on-coder-crash.
"""
from __future__ import annotations

import asyncio

from saddle.codemap import Finding, SurfaceManifest, ValueSpec
from saddle.converge import (
    ALREADY,
    CODER_FAILED,
    CONVERGED,
    EXHAUSTED,
    NO_SURFACE,
    STALLED,
    converge_design,
)
from saddle.models import Design

# --- the real-gate fixture (mirrors test_design_intent_drift) ----------------
_DRIFTED = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    return resolve_cd(inst)

def sweep(inst):
    return inst["cooldown_s"] - 1   # DRIFT: raw base read, modifier misses
'''

_CLEAN = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    return resolve_cd(inst)

def sweep(inst):
    cd = resolve_cd(inst)        # routed through the resolver -> covered
    return cd - 1
'''


def _design(surface: bool = True) -> Design:
    meta = {}
    if surface:
        meta["surface"] = SurfaceManifest(
            values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
        ).to_dict()
    return Design(ask="route every cooldown read through the resolver",
                  body="Make sweep() read the cooldown through resolve_cd().",
                  summary="cooldown-resolver routing", meta=meta, id="dsg_test")


def _finding(thing: str) -> Finding:
    return Finding(check="value_propagation", severity="error", node_kind="value",
                   thing=thing, message="raw base read; modifier misses",
                   location=f"abil.py:{len(thing)}", detail={"func": thing})


# --- fakes -------------------------------------------------------------------
class _ScriptedCoder:
    """Each turn runs the next scripted action (or None) and returns canned text.
    Records every prompt it was briefed with, and its open/close lifecycle."""

    def __init__(self, actions=()):
        self._actions = list(actions)
        self.prompts: list[str] = []
        self.turns = 0
        self.opened = False
        self.closed = False

    async def __aenter__(self):
        self.opened = True
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def turn(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.turns += 1
        if self._actions:
            act = self._actions.pop(0)
            if act is not None:
                act()
        return "made some edits"


class _CrashingCoder:
    def __init__(self):
        self.turns = 0
        self.opened = False

    async def __aenter__(self):
        self.opened = True
        return self

    async def __aexit__(self, *exc):
        pass

    async def turn(self, prompt: str) -> str:
        self.turns += 1
        raise RuntimeError("claude CLI failed for converge/turn: exit code 1")


def _gate_seq(*returns):
    """Gate callable returning each list in turn; repeats the last once exhausted.
    Call 1 is the pre-loop gate; call n+1 is the re-gate after round n."""
    seq = [list(r) for r in returns]
    box = {"i": 0}

    def gate():
        i = min(box["i"], len(seq) - 1)
        box["i"] += 1
        return list(seq[i])

    return gate


def _run(coro):
    return asyncio.run(coro)


# --- the loop ----------------------------------------------------------------
def test_already_satisfied_never_opens_the_coder():
    coder = _ScriptedCoder()
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq([]), directives=[], persist=False,
    ))
    assert res.status == ALREADY and res.ok
    assert coder.opened is False and coder.turns == 0
    assert res.rounds == []


def test_no_surface_short_circuits_before_gate_or_coder():
    coder = _ScriptedCoder()

    def _boom():
        raise AssertionError("gate must not run when there is no surface")

    res = _run(converge_design(
        _design(surface=False), code_root="/nonexistent", coder=coder,
        gate=_boom, directives=[], persist=False,
    ))
    assert res.status == NO_SURFACE and not res.ok
    assert coder.opened is False


def test_converges_in_one_round_when_the_turn_closes_the_gap():
    g = _finding("sweep")
    coder = _ScriptedCoder(actions=[None])  # one turn; the gate flips to clean
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq([g], []), directives=[], persist=False,
    ))
    assert res.status == CONVERGED and res.ok
    assert coder.turns == 1 and coder.opened and coder.closed
    assert len(res.rounds) == 1
    assert res.rounds[0].gaps_before == 1 and res.rounds[0].gaps_after == 0


def test_round_closed_counts_the_set_difference_not_the_count_delta():
    """When a round CLOSES one gap but OPENS another, the gap count is unchanged —
    a raw before-minus-after delta records 0 closed, which lies. ``Round.closed``
    must be the true set-difference (the actual finding that went away)."""
    a, b, c = _finding("alpha"), _finding("beta"), _finding("gamma")
    coder = _ScriptedCoder(actions=[None, None])
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        # pre-gate {a,b}; round1 -> {b,c} (a closed, c opened); round2 -> {} (clean)
        gate=_gate_seq([a, b], [b, c], []), directives=[], persist=False,
        max_rounds=4, stall_repeat=3,
    ))
    assert res.status == CONVERGED and res.ok
    r1 = res.rounds[0]
    assert r1.gaps_before == 2 and r1.gaps_after == 2   # count is unchanged...
    assert r1.closed == 1                               # ...but one gap (alpha) truly closed
    assert res.rounds[1].closed == 2                    # round 2 closed both remaining


def test_real_gate_end_to_end_drives_a_real_edit_to_clean(tmp_path):
    """The crown jewel: a REAL intent_drift gate over a real tree, driven to clean
    by a fake coder that writes the fix. No injected gate — the actual codemap."""
    abil = tmp_path / "abil.py"
    abil.write_text(_DRIFTED)
    design = _design()

    coder = _ScriptedCoder(actions=[lambda: abil.write_text(_CLEAN)])
    res = _run(converge_design(
        design, code_root=tmp_path, coder=coder, directives=[], persist=False,
        max_rounds=4,
    ))
    assert res.status == CONVERGED and res.ok
    assert coder.turns == 1
    assert abil.read_text() == _CLEAN          # the coder's edit landed
    assert res.final_gaps == []


def test_real_gate_rejects_a_coder_that_only_claims_done(tmp_path):
    """The coder returns 'done' text but never edits the file — the gate sees the
    unchanged drift and the loop refuses to converge (self-claim is never trusted)."""
    abil = tmp_path / "abil.py"
    abil.write_text(_DRIFTED)
    coder = _ScriptedCoder(actions=[None, None])  # talks, never edits
    res = _run(converge_design(
        _design(), code_root=tmp_path, coder=coder, directives=[], persist=False,
        max_rounds=3, stall_repeat=3,
    ))
    assert res.status == STALLED and not res.ok       # never accepted as done
    assert res.final_gaps and res.final_gaps[0].detail["func"] == "sweep"


def test_stalls_on_a_recurring_gap_set_before_the_cap():
    g = _finding("sweep")
    coder = _ScriptedCoder(actions=[None] * 9)
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq([g]),  # always the same gap
        directives=[], persist=False, max_rounds=8, stall_repeat=3,
    ))
    assert res.status == STALLED and not res.ok
    assert coder.turns == 3            # stalled at the 3rd recurrence, not the cap
    assert {f.thing for f in res.final_gaps} == {"sweep"}


def test_exhausts_the_round_cap_when_each_round_is_distinct():
    seq = [[_finding(f"g{i}")] for i in range(5)]  # all distinct -> never stalls
    coder = _ScriptedCoder(actions=[None] * 5)
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq(*seq), directives=[], persist=False,
        max_rounds=3, stall_repeat=3,
    ))
    assert res.status == EXHAUSTED and not res.ok
    assert coder.turns == 3 and len(res.rounds) == 3


def test_coder_crash_halts_after_bounded_retries():
    coder = _CrashingCoder()
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq([_finding("sweep")]), directives=[], persist=False,
        turn_retries=2,
    ))
    assert res.status == CODER_FAILED and not res.ok
    assert coder.turns == 3            # initial try + 2 retries
    assert "claude CLI failed" in res.error
    assert res.final_gaps and res.final_gaps[0].thing == "sweep"


def test_first_brief_and_reprompt_are_effect_grounded():
    fa, fb = _finding("alpha"), _finding("beta")
    coder = _ScriptedCoder(actions=[None, None])
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=coder,
        gate=_gate_seq([fa, fb], [fb], []),  # close alpha, then beta
        directives=["no band-aids"], persist=False,
    ))
    assert res.status == CONVERGED
    first, reprompt = coder.prompts[0], coder.prompts[1]
    # First brief: the design body, the binding rule, the gaps, the anti-self-claim.
    assert "read the cooldown through resolve_cd" in first
    assert "no band-aids" in first
    assert "CURRENT GAPS" in first and str(fa) in first and str(fb) in first
    assert "do not declare" in first.lower()
    # Re-prompt: the progress delta (alpha closed) + the still-unsatisfied (beta).
    assert "CLOSED since last turn" in reprompt and str(fa) in reprompt
    assert "STILL UNSATISFIED" in reprompt and str(fb) in reprompt
    assert "do not declare" in reprompt.lower()


def test_persists_the_convergence_trail_via_the_dkb():
    from saddle.context import resolve

    recorded: list[tuple] = []

    class _FakeDKB:
        def update_design_meta(self, ctx, did, patch):
            recorded.append((did, patch))
            return True

    ctx = resolve("aeli", "convtest")
    res = _run(converge_design(
        _design(), code_root="/nonexistent", coder=_ScriptedCoder(actions=[None]),
        gate=_gate_seq([_finding("sweep")], []), directives=[],
        ctx=ctx, dkb=_FakeDKB(), persist=True,
    ))
    assert res.status == CONVERGED
    assert len(recorded) == 1
    did, patch = recorded[0]
    assert did == "dsg_test"
    assert patch["convergence"]["outcome"] == CONVERGED
    assert patch["convergence"]["rounds"][0]["closed"] == 1


# --- live coder streaming ----------------------------------------------------
class _FakeAskSession:
    """Stand-in for ChatSession: ``ask`` yields the canned chunks one at a time."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def ask(self, prompt: str):
        for c in self._chunks:
            yield c


def test_chatsessioncoder_streams_each_chunk_to_the_sink_live():
    from saddle.converge import ChatSessionCoder

    seen: list[str] = []
    coder = ChatSessionCoder(cwd="/x", on_chunk=seen.append)
    coder._session = _FakeAskSession(["analyzing ", "editing ", "done"])
    out = _run(coder.turn("go"))
    # The full text is still returned for the loop's accounting...
    assert out == "analyzing editing done"
    # ...AND every chunk was surfaced live, terminated by a newline so the next
    # narration line starts clean.
    assert seen == ["analyzing ", "editing ", "done", "\n"]


def test_chatsessioncoder_without_sink_just_returns_text():
    from saddle.converge import ChatSessionCoder

    coder = ChatSessionCoder(cwd="/x")  # no on_chunk
    coder._session = _FakeAskSession(["a", "b"])
    assert _run(coder.turn("go")) == "ab"
