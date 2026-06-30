"""stop_hook — the Stop (turn-end) bubble-up hook (Stage 4: code conformance).

saddle's retrospective channel: when the agent finishes a turn, the code it wrote
is re-verified against the project's settled designs. A design whose committed
surface the code no longer satisfies bubbles as a ``stage=code`` event — and a
scan that itself fails is surfaced LOUD, never swallowed. The hook observes only:
it never blocks (exit 0 always) and emits no stop decision. Unit tests stub the
scan to script each outcome; one end-to-end test drives the REAL engine against a
real tmp tree to prove the whole wiring (hook -> run_stage -> conformance_scan ->
intent_drift -> finding -> bubble) holds.
"""
from __future__ import annotations

import io
import json

from saddle.codemap import Finding, SurfaceManifest, ValueSpec
from saddle.context import Context
from saddle.design import AuditVerdict, ConformanceDrift, ConformanceResult
from saddle.models import DESIGN_FINAL, Design

_CTX = Context(tenant="acme", project="game")

# A raw base read that bypasses the resolver -> the modifier misses -> real DRIFT.
_DRIFTED = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def sweep(inst):
    return inst["cooldown_s"] - 1   # DRIFT: raw base read, modifier misses
'''


def _finding(severity: str = "error", func: str = "sweep") -> Finding:
    return Finding(
        check="value_propagation", severity=severity, node_kind="value",
        thing="cooldown", message="raw base read bypasses the resolver",
        location="abil.py:10", detail={"func": func},
    )


class _StubDKB:
    """Serves designs like the real DKB's ``list_designs`` (status + limit filter)."""

    def __init__(self, designs: list[Design]) -> None:
        self._designs = list(designs)

    def list_designs(self, ctx, *, status=None, limit=50):
        rows = [d for d in self._designs if status is None or d.status == status]
        return rows[: int(limit)]


def _run_stop(payload, monkeypatch, capsys, tmp_path, *, env=None):
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import stop_hook
    rc = stop_hook.main()
    return rc, capsys.readouterr()


def _bubbles(level=None):
    from saddle.bubble import recent_bubbles
    return recent_bubbles(_CTX, level=level)


# --- fail-open infra edges --------------------------------------------------

def test_empty_stdin_is_silent(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    from saddle import stop_hook
    assert stop_hook.main() == 0
    assert capsys.readouterr().err.strip() == ""


def test_unparseable_payload_fails_open(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    from saddle import stop_hook
    assert stop_hook.main() == 0
    assert "unparseable" in capsys.readouterr().err


def test_disabled_by_env_skips_scan(monkeypatch, capsys, tmp_path):
    """SADDLE_HOOK_CODE=0 turns Stage 4 off entirely — the scan is never reached
    (sentinel, since run_stage would otherwise classify a raised assertion into a
    masking ALERT) and the turn stays silent."""
    called = {"v": False}

    def sentinel(*a, **k):
        called["v"] = True
        return ConformanceResult()

    monkeypatch.setattr("saddle.design.conformance_scan", sentinel)
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path,
                        env={"SADDLE_HOOK_CODE": "0"})
    assert rc == 0
    assert called["v"] is False
    assert out.err.strip() == "" and _bubbles() == []


# --- the three run_stage outcomes -------------------------------------------

def test_clean_turn_is_silent(monkeypatch, capsys, tmp_path):
    """Settled designs gated, code satisfies them -> the stage RAN and stayed
    silent: no stderr, no bubble (silent success, not a skipped check)."""
    monkeypatch.setattr("saddle.design.conformance_scan",
                        lambda *a, **k: ConformanceResult(designs_checked=2))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert out.err.strip() == ""
    assert _bubbles() == []


def test_error_grade_drift_bubbles_an_alert(monkeypatch, capsys, tmp_path):
    """A hard (error-grade) conformance break -> an ALERT bubble under stage=code,
    plus the on-screen copy. The drifting design and its touchpoint are named."""
    drift = ConformanceDrift(design_id="d1", summary="cooldown resolver design",
                             findings=[_finding("error")])
    monkeypatch.setattr("saddle.design.conformance_scan",
                        lambda *a, **k: ConformanceResult(drifts=[drift], designs_checked=1))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert "CODE DRIFT" in out.err and "d1" in out.err
    alerts = _bubbles(level="alert")
    assert any(b.stage == "code" and "CODE DRIFT" in b.text and "d1" in b.text
               for b in alerts)
    # the structured payload rides on the bubble for a richer client.
    b = next(b for b in alerts if b.stage == "code")
    assert b.meta["designs_checked"] == 1
    assert b.meta["drifts"][0]["design_id"] == "d1" and b.meta["drifts"][0]["has_error"]


def test_warn_only_drift_is_a_notice_not_an_alert(monkeypatch, capsys, tmp_path):
    """A soft (warn-grade) gap is surfaced as a NOTICE, not escalated to an ALERT —
    the level tracks the finding severity the codemap already classified."""
    drift = ConformanceDrift(design_id="d2", summary="x", findings=[_finding("warn")])
    monkeypatch.setattr("saddle.design.conformance_scan",
                        lambda *a, **k: ConformanceResult(drifts=[drift], designs_checked=1))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert any(b.stage == "code" for b in _bubbles(level="notice"))
    assert _bubbles(level="alert") == []


def test_scan_failure_fails_loud(monkeypatch, capsys, tmp_path):
    """fail-loud: a scan that throws (a parse blow-up) is classified and bubbled as
    a stage=code ALERT naming what saddle did NOT verify + the root cause — never a
    swallowed pass. The hook still never blocks (rc 0 — observation)."""
    def boom(*a, **k):
        raise RuntimeError("parse exploded")
    monkeypatch.setattr("saddle.design.conformance_scan", boom)
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    alerts = _bubbles(level="alert")
    code_alert = next((b for b in alerts if b.stage == "code"), None)
    assert code_alert is not None
    assert "could not run" in code_alert.text and "did NOT verify" in code_alert.text
    assert "RuntimeError" in code_alert.text
    assert "could not run" in out.err


# --- the user-screen channel (ask #3: saddle visible on screen) -------------

def test_drift_heralds_on_screen_via_systemmessage(monkeypatch, capsys, tmp_path):
    """saddle is no longer invisible: a turn-end code drift is heralded to the
    HUMAN on the ``systemMessage`` channel (the one Claude Code renders on screen),
    not only injected as agent context or buried in the durable outbox. A ``Stop``
    hook has no additionalContext channel, so this stdout JSON is saddle's only live
    surface to the watching person; ``suppressOutput`` keeps the raw JSON out of
    transcript mode while the message still shows."""
    drift = ConformanceDrift(design_id="d1", summary="cooldown resolver design",
                             findings=[_finding("error")])
    monkeypatch.setattr("saddle.design.conformance_scan",
                        lambda *a, **k: ConformanceResult(drifts=[drift], designs_checked=1))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    payload = json.loads(out.out)
    assert "CODE DRIFT" in payload["systemMessage"] and "d1" in payload["systemMessage"]
    assert payload["suppressOutput"] is True
    assert "decision" not in payload  # observe-only: a Stop block would force a re-run


def test_scan_failure_heralds_could_not_run_on_screen(monkeypatch, capsys, tmp_path):
    """The cardinal case: when a stage COULD-NOT-RUN, the human must SEE that saddle
    did not verify the code this turn — the classified ALERT reaches the screen, not
    just the outbox."""
    def boom(*a, **k):
        raise RuntimeError("parse exploded")
    monkeypatch.setattr("saddle.design.conformance_scan", boom)
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    payload = json.loads(out.out)
    assert "could not run" in payload["systemMessage"]
    assert "did NOT verify" in payload["systemMessage"]


def test_clean_turn_emits_no_stdout(monkeypatch, capsys, tmp_path):
    """A clean turn stays quiet on screen: no systemMessage, no stdout JSON at all,
    so saddle never announces it had nothing to say."""
    monkeypatch.setattr("saddle.design.conformance_scan",
                        lambda *a, **k: ConformanceResult(designs_checked=2))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert out.out.strip() == ""


# --- end-to-end: the real engine against a real tree ------------------------

def test_end_to_end_real_conformance_drift_bubbles(monkeypatch, capsys, tmp_path):
    """The whole wiring, unstubbed except the design source: a real settled design
    with a surface + a real drifted tree -> the REAL conformance_scan/intent_drift
    parse finds the bypass and the hook bubbles it as a stage=code ALERT naming the
    design and the offending function."""
    (tmp_path / "abil.py").write_text(_DRIFTED)
    design = Design(
        id="dreal", ask="route every cooldown read through the resolver",
        summary="cooldown resolver design", status=DESIGN_FINAL,
        meta={"surface": SurfaceManifest(
            values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
        ).to_dict()},
    )
    monkeypatch.setattr("saddle.design.get_dkb", lambda: _StubDKB([design]))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    alerts = _bubbles(level="alert")
    code_alert = next((b for b in alerts if b.stage == "code"), None)
    assert code_alert is not None
    assert "CODE DRIFT" in code_alert.text and "dreal" in code_alert.text
    assert "sweep" in code_alert.text          # the real touchpoint the parse found
    assert code_alert.meta["drifts"][0]["design_id"] == "dreal"


def test_re_catches_uncorrected_drift_next_turn(monkeypatch, capsys, tmp_path):
    """'Not once and done': the SAME uncorrected drift is re-caught on a second
    turn-end (fresh parse every time), and disappears only once the code is fixed —
    proving the gate is not a one-shot that blesses stale code."""
    abil = tmp_path / "abil.py"
    abil.write_text(_DRIFTED)
    design = Design(
        id="dreal", ask="route every cooldown read through the resolver",
        summary="cooldown resolver design", status=DESIGN_FINAL,
        meta={"surface": SurfaceManifest(
            values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
        ).to_dict()},
    )
    monkeypatch.setattr("saddle.design.get_dkb", lambda: _StubDKB([design]))

    rc, _ = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert any(b.stage == "code" for b in _bubbles(level="alert"))   # turn 1: caught

    # the human did NOT correct it -> turn 2 catches it again (re-parsed fresh).
    rc, _ = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert sum(b.stage == "code" for b in _bubbles(level="alert")) == 2

    # now fix the code; the next turn-end is clean (no new code alert).
    abil.write_text(_DRIFTED.replace('inst["cooldown_s"] - 1', "resolve_cd(inst) - 1"))
    rc, out = _run_stop({"session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert sum(b.stage == "code" for b in _bubbles(level="alert")) == 2   # unchanged
    assert out.err.strip() == ""


# --- Stage 3 (turn-end): the prose-proposal blind-spot review ----------------
#
# The pre-edit design gate (doctrine_hook) is EDIT-gated, so a turn that proposes
# an approach in PROSE and writes no file is never reviewed. This hook closes that
# blind spot at turn-end, SHARING the pre-edit gate's per-turn anchor marker so
# EXACTLY ONE of the two reviews a turn. These stub the audit LLM (monkeypatch
# audit_proposal) and drive the whole hook (main()), asserting the wiring: a
# prose-only proposal with a flaw -> a STAGE_DESIGN ALERT carrying meta['issues'];
# a turn the pre-edit gate already reviewed -> silent (no double review); a turn
# with no approach prose -> silent + marked (a no-op turn is not a blind spot); a
# repeat Stop in the same turn -> silent (once per turn); the knob disables it; and
# an audit that itself cannot run fails LOUD (never a swallowed 'looked fine').


def _t_user(text, uuid):
    return {"type": "user", "sessionId": "s1", "uuid": uuid, "isSidechain": False,
            "message": {"role": "user", "content": text}}


def _t_asst(text, uuid):
    return {"type": "assistant", "sessionId": "s1", "uuid": uuid, "isSidechain": False,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def _transcript(tmp_path, objs):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return str(p)


def _stop_payload(tp, session="s1"):
    return {"session_id": session, "transcript_path": tp}


def _canned_audit(verdict):
    async def _audit(goal, approach, ctx=None, **kw):
        return verdict
    return _audit


def test_proposal_review_fires_on_prose_only_blind_spot(monkeypatch, capsys, tmp_path):
    """The blind spot, CLOSED: a turn that proposed an approach in prose and wrote
    NO file (so the edit-gated pre-edit gate never saw it) is reviewed at turn-end
    through the SAME audit engine; a flaw surfaces as a STAGE_DESIGN ALERT carrying
    meta['issues'] (so Stage 5 can harvest it) and is heralded on screen."""
    monkeypatch.setattr("saddle.design.audit_proposal", _canned_audit(
        AuditVerdict(ok=False, issues=["band-aid: clamps the tooltip, not the cause"])))
    tp = _transcript(tmp_path, [
        _t_user("the cooldown ignores the talent — what should we do?", "u1"),
        _t_asst("I'll just clamp the displayed cooldown in the tooltip.", "u2"),
    ])
    rc, out = _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    design_alert = next((b for b in _bubbles(level="alert") if b.stage == "design"), None)
    assert design_alert is not None
    assert "wrote no code" in design_alert.text
    assert "band-aid: clamps the tooltip, not the cause" in design_alert.text
    # the structured issues ride on the bubble so Stage 5's harvest can read them.
    assert design_alert.meta["issues"] == ["band-aid: clamps the tooltip, not the cause"]
    # and the human SEES it on screen via the systemMessage herald.
    assert "band-aid: clamps the tooltip, not the cause" in json.loads(out.out)["systemMessage"]


def test_proposal_review_silent_when_pre_edit_gate_already_fired(
    monkeypatch, capsys, tmp_path
):
    """Complementary, never duplicative: when the pre-edit gate already reviewed THIS
    turn (its per-turn anchor marker is set on the user-prompt uuid), the turn-end
    review stands down — exactly one of the two audits a turn. audit_proposal is
    never called a second time and no turn-end design bubble is emitted."""
    called = {"n": 0}

    async def _counting(goal, approach, ctx=None, **kw):
        called["n"] += 1
        return AuditVerdict(ok=False, issues=["should not run"])

    monkeypatch.setattr("saddle.design.audit_proposal", _counting)
    tp = _transcript(tmp_path, [
        _t_user("do the thing", "u1"),
        _t_asst("here's my approach", "u2"),
    ])
    # simulate the pre-edit gate having fired on this turn's anchor (the user uuid).
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    from saddle.doctrine_hook import _mark_design_fired
    _mark_design_fired("s1", "u1")

    rc, out = _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    assert called["n"] == 0                                       # no second audit
    assert not any(b.stage == "design" for b in _bubbles())       # no turn-end design bubble


def test_proposal_review_silent_when_no_approach_prose(monkeypatch, capsys, tmp_path):
    """A turn that wrote NO approach prose (a pure tool-run / no-op turn) is not a
    blind spot — there is nothing to review. The review stays silent AND marks the
    turn (so a later edit or repeat Stop in this turn doesn't reconsider), without
    ever calling the audit."""
    called = {"n": 0}

    async def _forbidden(*a, **k):
        called["n"] += 1
        return AuditVerdict(ok=False, issues=["x"])

    monkeypatch.setattr("saddle.design.audit_proposal", _forbidden)
    tp = _transcript(tmp_path, [_t_user("just run the tests", "u1")])  # no assistant prose

    rc, out = _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    assert called["n"] == 0                                       # no approach -> no audit
    assert not any(b.stage == "design" for b in _bubbles())
    # the turn was MARKED, so the pre-edit gate / a repeat Stop now skips it.
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    from saddle.doctrine_hook import _design_already_fired
    assert _design_already_fired("s1", "u1") is True


def test_proposal_review_marks_turn_so_repeat_stop_is_silent(
    monkeypatch, capsys, tmp_path
):
    """Once per turn, end to end: after the turn-end review RUNS, it marks the turn
    through the shared anchor marker, so a second Stop in the same turn does not
    re-review (no duplicate STAGE_DESIGN bubble, no second audit call)."""
    calls = {"n": 0}

    async def _counting(goal, approach, ctx=None, **kw):
        calls["n"] += 1
        return AuditVerdict(ok=False, issues=["band-aid: swallow-and-log"])

    monkeypatch.setattr("saddle.design.audit_proposal", _counting)
    tp = _transcript(tmp_path, [
        _t_user("fix it", "u1"),
        _t_asst("I'll wrap it in try/except and log.", "u2"),
    ])
    _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)   # turn-end: reviews
    _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)   # same turn: skipped

    assert calls["n"] == 1                                         # anchored on the user uuid
    assert sum(b.stage == "design" for b in _bubbles(level="alert")) == 1


def test_proposal_review_disabled_by_env(monkeypatch, capsys, tmp_path):
    """SADDLE_HOOK_DESIGN=0 turns the turn-end proposal review off (the same knob
    the pre-edit gate honours) — the audit is never reached and nothing is said."""
    called = {"n": 0}

    async def _counting(*a, **k):
        called["n"] += 1
        return AuditVerdict()

    monkeypatch.setattr("saddle.design.audit_proposal", _counting)
    tp = _transcript(tmp_path, [_t_user("do it", "u1"), _t_asst("approach", "u2")])
    rc, out = _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path,
                        env={"SADDLE_HOOK_DESIGN": "0"})

    assert rc == 0
    assert called["n"] == 0
    assert not any(b.stage == "design" for b in _bubbles())


def test_proposal_review_audit_failure_fails_loud(monkeypatch, capsys, tmp_path):
    """Cardinal-sin guard: if the turn-end audit itself cannot run (provider outage,
    a contract gap), it is NOT swallowed into a silent 'looked fine' — it propagates
    to run_stage and bubbles a classified STAGE_DESIGN ALERT naming what saddle did
    NOT verify this turn, on screen too. Observation never blocks (rc 0)."""
    async def _boom(goal, approach, ctx=None, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr("saddle.design.audit_proposal", _boom)
    tp = _transcript(tmp_path, [
        _t_user("fix the timeout", "u1"),
        _t_asst("I'll wrap the failing call in try/except and log it.", "u2"),
    ])
    rc, out = _run_stop(_stop_payload(tp), monkeypatch, capsys, tmp_path)

    assert rc == 0
    assert any(b.stage == "design" and "could not run" in b.text
               for b in _bubbles(level="alert"))
    assert "could not run" in json.loads(out.out)["systemMessage"]
