"""Stage 5 — turn-end lesson harvest (the cumulative, not 'caught once and done',
guarantee).

Two seams, both offline:

* the ENGINE ``design.harvest_turn`` — distils the flaws CAUGHT this turn into
  durable, deduped DKB lessons (``source=audit``) via the EXACT ``_harvest`` the
  design pipeline runs on its own caught flaws; no caught flaws ⇒ no LLM call;
* the WIRING in ``stop_hook`` — at turn-end it reads this turn's design (Stage 3)
  and code (Stage 4) drift back from the durable outbox (since a per-session
  watermark), harvests it, bubbles a ``stage=lesson`` NOTICE naming what it
  learned, and advances the watermark so an UNcorrected-but-already-learned drift
  is not re-harvested next turn.

The LLM and the DKB are faked: a caller pops a canned harvest payload and a stub
DKB records what gets filed (and serves existing titles for the dedup check).
"""
from __future__ import annotations

import asyncio
import io
import json

import pytest

from saddle.context import Context
from saddle.design import HarvestResult, harvest_turn
from saddle.models import (
    ANTI_PATTERN,
    AUDIT,
    BUBBLE_ALERT,
    BUBBLE_NOTICE,
    LESSON,
    STAGE_CODE,
    STAGE_DESIGN,
    STAGE_INTAKE,
    STAGE_INTENT,
    STAGE_LESSON,
    Knowledge,
)

_CTX = Context(tenant="acme", project="game")

_ENTRY = {
    "kind": "lesson",
    "title": "Bump is not balance",
    "body": "A flat multiplier never fixes a kit that lacks a rotation.",
    "tags": ["balance", "design"],
}


# === fakes ===================================================================

class _HarvestCaller:
    """Pops a canned harvest payload; records the prompt and call count. ``boom``
    raises, to prove a failed classify PROPAGATES (fail-loud)."""

    def __init__(self, entries, *, boom: bool = False) -> None:
        self.entries = entries
        self.boom = boom
        self.prompt = ""
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        if self.boom:
            raise RuntimeError("harvest provider down")
        self.prompt = prompt
        return json.dumps({"entries": self.entries})


class _StubDKB:
    """Records harvested lessons and serves existing titles for the dedup gate."""

    def __init__(self, existing=None) -> None:
        self.added: list[Knowledge] = list(existing or [])

    def list_knowledge(self, ctx=None, *, limit=200, **kw):
        return list(self.added)

    def add_knowledge(self, kn: Knowledge) -> Knowledge:
        self.added.append(kn)
        return kn


def _harvest(issues, *, caller, dkb, goal="make the mage hit harder"):
    return asyncio.run(harvest_turn(goal, issues, _CTX, caller=caller, dkb=dkb))


# === engine: harvest_turn ====================================================

def test_harvest_turn_no_issues_makes_no_llm_call():
    """A clean turn taught nothing: no issues ⇒ the engine short-circuits with an
    empty result and never touches the provider (output is the cost we protect)."""
    caller = _HarvestCaller([_ENTRY])
    res = _harvest(["   ", ""], caller=caller, dkb=_StubDKB())
    assert isinstance(res, HarvestResult)
    assert res.harvested == 0 and res.considered == 0 and res.titles == []
    assert caller.calls == 0


def test_harvest_turn_files_scoped_audit_lessons():
    """Caught flaws become durable, project-scoped, audit-sourced lessons; the
    engine returns the titles filed (so the bubble can name them) and counts the
    flaws it considered."""
    caller = _HarvestCaller([_ENTRY])
    dkb = _StubDKB()
    res = _harvest(["band-aid: a flat damage multiplier", "no rotation in the kit"],
                   caller=caller, dkb=dkb)
    assert res.harvested == 1 and res.considered == 2
    assert res.titles == ["Bump is not balance"]
    assert caller.calls == 1
    assert len(dkb.added) == 1
    lesson = dkb.added[0]
    assert lesson.kind == LESSON and lesson.source == AUDIT
    assert lesson.scope_tenant == "acme" and lesson.scope_project == "game"
    # the caught flaws were actually fed to the harvest prompt.
    assert "flat damage multiplier" in caller.prompt


def test_harvest_turn_dedupes_against_existing_titles():
    """A lesson already on file is NOT re-filed — the harvest is cumulative, never
    duplicative, so re-catching the same drift each turn cannot spam the DKB."""
    existing = Knowledge(kind=LESSON, title="Bump is not balance",
                         body="already known", scope_tenant="acme",
                         scope_project="game", source=AUDIT)
    dkb = _StubDKB(existing=[existing])
    res = _harvest(["band-aid: a flat multiplier"],
                   caller=_HarvestCaller([_ENTRY]), dkb=dkb)
    assert res.harvested == 0          # nothing new filed
    assert len(dkb.added) == 1         # only the pre-seeded one remains


def test_harvest_turn_failed_classify_propagates():
    """fail-loud: a raising caller propagates OUT for run_stage to classify and
    bubble an ALERT — a harvest that could not run is never a silent 'learned'."""
    with pytest.raises(RuntimeError, match="harvest provider down"):
        _harvest(["some flaw"], caller=_HarvestCaller([], boom=True), dkb=_StubDKB())


# === wiring: stop_hook Stage 5 ===============================================

def _prep(monkeypatch, tmp_path, **extra):
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("SADDLE_HOOK_CODE", "0")  # isolate Stage 5 from Stage 4
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def _emit(stage, level, text, meta):
    from saddle.bubble import emit_bubble
    return emit_bubble(_CTX, text, level=level, stage=stage, session="s1", meta=meta)


def _design_bubble(issues):
    return _emit(STAGE_DESIGN, BUBBLE_ALERT, "approach has design issues",
                 {"issues": list(issues)})


def _code_bubble(findings, *, design_id="d1"):
    return _emit(STAGE_CODE, BUBBLE_ALERT, "code drift",
                 {"drifts": [{"design_id": design_id, "summary": "x",
                              "findings": list(findings), "has_error": True}]})


def _run(monkeypatch, capsys, payload=None):
    monkeypatch.setattr("sys.stdin",
                        io.StringIO(json.dumps(payload or {"session_id": "s1"})))
    from saddle import stop_hook
    return stop_hook.main(), capsys.readouterr()


def _bubbles(level=None, stage=None):
    from saddle.bubble import recent_bubbles
    return [b for b in recent_bubbles(_CTX, level=level)
            if stage is None or b.stage == stage]


class _CannedHarvest:
    """Stands in for ``design.harvest_turn`` in the hook; records the goal + issues
    it was handed so the wiring (outbox -> _turn_issues -> harvest) is provable."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0
        self.last_goal = None
        self.last_issues = None

    async def __call__(self, goal, issues, ctx=None, **kw):
        self.calls += 1
        self.last_goal = goal
        self.last_issues = list(issues)
        return self.result


def _t_user(text, uuid="u1"):
    return {"type": "user", "sessionId": "s1", "uuid": uuid, "isSidechain": False,
            "message": {"role": "user", "content": text}}


def _transcript(tmp_path, objs):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return str(p)


def test_turn_issues_gathers_design_and_code_only(monkeypatch, tmp_path):
    """``_turn_issues`` reads Stage-3 design issues and Stage-4 code findings from
    the outbox — and DELIBERATELY ignores intake/intent/lesson, which are not
    generalizable design lessons the harvest engine is tuned for."""
    _prep(monkeypatch, tmp_path)
    _design_bubble(["band-aid: swallow-and-log"])
    _code_bubble(["raw base read bypasses the resolver"])
    _emit(STAGE_INTAKE, BUBBLE_ALERT, "itemize failed", {"failure": "timeout"})
    _emit(STAGE_INTENT, BUBBLE_ALERT, "contradicts a settled design", {})
    _emit(STAGE_LESSON, BUBBLE_NOTICE, "a prior lesson", {"titles": ["old"]})

    from saddle import stop_hook
    issues = stop_hook._turn_issues(_CTX, "s1", 0.0)
    assert "band-aid: swallow-and-log" in issues
    assert "raw base read bypasses the resolver" in issues
    assert all("itemize" not in i and "contradicts" not in i and "prior" not in i
               for i in issues)
    assert len(issues) == 2


def test_clean_turn_harvests_nothing(monkeypatch, capsys, tmp_path):
    """No design/code drift bubbled this turn ⇒ Stage 5 stays silent: no LLM call,
    no lesson bubble (a clean turn teaches nothing)."""
    _prep(monkeypatch, tmp_path)
    canned = _CannedHarvest(HarvestResult(titles=["x"], considered=1))
    monkeypatch.setattr("saddle.design.harvest_turn", canned)
    rc, out = _run(monkeypatch, capsys)
    assert rc == 0
    assert canned.calls == 0
    assert _bubbles(stage=STAGE_LESSON) == []
    assert out.err.strip() == ""


def test_caught_drift_is_harvested_and_named(monkeypatch, capsys, tmp_path):
    """This turn caught a band-aid (Stage 3) ⇒ Stage 5 feeds it to the harvest and
    bubbles a stage=lesson NOTICE naming the durable lesson filed, with the goal
    lifted from the transcript and the structured payload on the bubble."""
    _prep(monkeypatch, tmp_path)
    _design_bubble(["band-aid: swallow-and-log the error"])
    canned = _CannedHarvest(HarvestResult(titles=["Bump is not balance"], considered=1))
    monkeypatch.setattr("saddle.design.harvest_turn", canned)
    tp = _transcript(tmp_path, [_t_user("fix the timeout")])

    rc, out = _run(monkeypatch, capsys, {"session_id": "s1", "transcript_path": tp})
    assert rc == 0
    assert canned.calls == 1
    assert canned.last_goal == "fix the timeout"
    assert "band-aid: swallow-and-log the error" in canned.last_issues
    notices = _bubbles(level="notice", stage=STAGE_LESSON)
    assert len(notices) == 1
    b = notices[0]
    assert "LESSON HARVEST" in b.text and "Bump is not balance" in b.text
    assert b.meta["harvested"] == 1 and b.meta["titles"] == ["Bump is not balance"]
    assert "LESSON HARVEST" in out.err          # on-screen copy too


def test_lesson_stage_disabled_by_env(monkeypatch, capsys, tmp_path):
    """SADDLE_HOOK_LESSON=0 turns Stage 5 off: even with caught drift on the outbox
    the harvest is never reached and the turn stays silent."""
    _prep(monkeypatch, tmp_path, SADDLE_HOOK_LESSON="0")
    _design_bubble(["band-aid: swallow-and-log"])
    canned = _CannedHarvest(HarvestResult(titles=["x"], considered=1))
    monkeypatch.setattr("saddle.design.harvest_turn", canned)
    rc, out = _run(monkeypatch, capsys)
    assert rc == 0 and canned.calls == 0
    assert _bubbles(stage=STAGE_LESSON) == []


def test_uncorrected_drift_not_reharvested_next_turn(monkeypatch, capsys, tmp_path):
    """The watermark makes the harvest fire ONCE per drift: turn 1 harvests the
    caught band-aid; turn 2 (no NEW drift bubbled) reads nothing past the watermark
    and stays silent — the same lesson is not re-filed every turn."""
    _prep(monkeypatch, tmp_path)
    _design_bubble(["band-aid: swallow-and-log"])
    canned = _CannedHarvest(HarvestResult(titles=["Bump is not balance"], considered=1))
    monkeypatch.setattr("saddle.design.harvest_turn", canned)

    _run(monkeypatch, capsys)                       # turn 1: harvested
    assert canned.calls == 1
    from saddle import stop_hook
    assert stop_hook._harvest_watermark("s1") > 0.0  # watermark advanced

    _run(monkeypatch, capsys)                       # turn 2: nothing new
    assert canned.calls == 1                         # NOT re-harvested
    assert len(_bubbles(stage=STAGE_LESSON)) == 1


def test_harvest_failure_fails_loud(monkeypatch, capsys, tmp_path):
    """fail-loud through the hook: a harvest that THROWS is classified and bubbled
    as a stage=lesson ALERT naming what saddle did NOT do — never swallowed. The
    hook still never blocks (rc 0)."""
    _prep(monkeypatch, tmp_path)
    _design_bubble(["band-aid: swallow-and-log"])

    async def _boom(goal, issues, ctx=None, **kw):
        raise RuntimeError("harvest exploded")
    monkeypatch.setattr("saddle.design.harvest_turn", _boom)

    rc, out = _run(monkeypatch, capsys)
    assert rc == 0
    alerts = _bubbles(level="alert", stage=STAGE_LESSON)
    assert len(alerts) == 1
    assert "RuntimeError" in alerts[0].text


def test_end_to_end_real_harvest_files_lesson(monkeypatch, capsys, tmp_path):
    """The whole Stage-5 chain unstubbed except the provider + DKB: a real design
    bubble on the outbox -> _turn_issues -> the REAL harvest_turn/_harvest ->
    a lesson filed into the DKB (source=audit) AND a stage=lesson NOTICE naming
    it. Proves a flaw the live supervisor caught teaches the DKB the same way
    design_for's own caught flaws do."""
    _prep(monkeypatch, tmp_path)
    _design_bubble(["band-aid: a flat damage multiplier instead of a rotation"])
    caller = _HarvestCaller([_ENTRY])
    dkb = _StubDKB()
    monkeypatch.setattr("saddle.llm.callers.build_callers", lambda ctx: {"default": caller})
    monkeypatch.setattr("saddle.design.get_dkb", lambda: dkb)
    tp = _transcript(tmp_path, [_t_user("make the mage hit harder")])

    rc, out = _run(monkeypatch, capsys, {"session_id": "s1", "transcript_path": tp})
    assert rc == 0
    assert caller.calls == 1
    assert len(dkb.added) == 1
    lesson = dkb.added[0]
    assert lesson.title == "Bump is not balance"
    assert lesson.source == AUDIT and lesson.scope_project == "game"
    notices = _bubbles(level="notice", stage=STAGE_LESSON)
    assert len(notices) == 1 and "Bump is not balance" in notices[0].text
