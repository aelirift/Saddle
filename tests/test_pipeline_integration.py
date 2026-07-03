"""End-to-end supervisory pipeline — the WHOLE thing, across real turns.

The per-stage units are covered elsewhere (``test_intake_hook`` /
``test_doctrine_hook`` / ``test_stop_hook`` / ``test_conformance`` /
``test_lesson_harvest``). This file proves the property those isolated tests
cannot: that the FIVE stages, wired through the THREE real hook entry points
(``intake_hook`` -> ``doctrine_hook`` -> ``stop_hook``), behave as one live
supervisor over a multi-turn session —

  * it **bubbles** every caught drift to the durable outbox (the "panel" an AFK /
    non-TTY human reads), AND tells the agent on its own channel (the discuss);
  * it catches **multiple KINDS** of drift — a pick-drift (Stage 2), a band-aid
    design (Stage 3), a code-conformance break (Stage 4) — and harvests the
    lesson (Stage 5);
  * it catches them **multiple TIMES** — an uncorrected code drift is re-caught
    every turn (fresh parse, "not once and done"), and a NEW pick-drift on a
    later turn is caught on its own;
  * and the noise **clears** the moment the human corrects it — a fixed tree goes
    silent, an aligned action stops drifting.

Nothing here mocks a stage's verdict: the guard, the replay drift check, the
conformance parse, and the harvest dedup all run for real. Only the two seams
that would otherwise reach the network / a real provider are faked — the LLM
behind Stage 3's ``audit_proposal`` and Stage 5's harvest caller — and the DKB,
which serves a settled design to Stage 4 and records lessons for Stage 5. The
two intake LLM stages (itemize + the intent history axis) are switched off; they
are not drift channels under test here and have their own coverage.
"""
from __future__ import annotations

import io
import json

from saddle.context import Context
from saddle.models import (
    AUDIT,
    LESSON,
    STAGE_CODE,
    STAGE_DESIGN,
    STAGE_INTENT,
    STAGE_LESSON,
    Knowledge,
)

_CTX = Context(tenant="acme", project="game")

# A real drifted tree: ``sweep`` reads the base ``cooldown_s`` raw, bypassing the
# ``resolve_cd`` resolver the settled design committed every read must route
# through. The codemap's value_propagation check finds it for real (no stub).
_DRIFTED = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def sweep(inst):
    return inst["cooldown_s"] - 1   # DRIFT: raw base read, modifier misses
'''
_FIXED = _DRIFTED.replace('inst["cooldown_s"] - 1', "resolve_cd(inst) - 1")

# Three forks the agent offers over a session. a/b/c is safe to reuse: each fork
# is RESOLVED (bound) before the next opens, and a bare-label carrier is scoped
# to the open + committed forks only, so there is never a cross-fork ambiguity.
_FORK = "Here's the call on caching:\na) in-memory LRU\nb) redis\nc) sqlite cache"
_FORK2 = "Pick the store backend:\na) flat file\nb) postgres\nc) sqlite"
_FORK3 = "And the transport:\na) rest\nb) grpc\nc) websocket"

# The canned lesson the faked harvest provider returns for any caught flaw.
_ENTRY = {
    "kind": "lesson",
    "title": "Bump is not balance",
    "body": "A flat tweak never fixes a value that must route through its resolver.",
    "tags": ["design", "conformance"],
}


# === fakes (only the two network seams + the DKB) ===========================

class _HarvestCaller:
    """Stands in for the harvest LLM: pops the canned lesson, counts calls."""

    def __init__(self, entries) -> None:
        self.entries = entries
        self.calls = 0

    async def __call__(self, system, prompt, *, json_mode=False, label=""):
        self.calls += 1
        return json.dumps({"entries": self.entries})


class _IntegDKB:
    """One DKB for the whole turn-end: serves the settled design to Stage 4's
    conformance scan, and records/serves lessons for Stage 5's harvest + dedup."""

    def __init__(self, designs, existing_lessons=None) -> None:
        self._designs = list(designs)
        self.added: list[Knowledge] = list(existing_lessons or [])

    # -- Stage 4 (conformance) --
    def list_designs(self, ctx, *, status=None, limit=50):
        rows = [d for d in self._designs if status is None or d.status == status]
        return rows[: int(limit)]

    # -- Stage 3 settlement (a clean pre-edit approach is recorded) --
    def add_design(self, ctx, design):
        self._designs.append(design)
        return design

    # -- Stage 5 (harvest) --
    def list_knowledge(self, ctx=None, *, limit=200, **kw):
        return list(self.added)

    def add_knowledge(self, kn: Knowledge) -> Knowledge:
        self.added.append(kn)
        return kn


# === the harness: drive the three real hooks over a growing transcript ======

class _Pipeline:
    """A single Claude Code session: a shared tmp tree, a growing transcript, a
    persisted ledger + bubble outbox, and the three real hook ``main()``s fired
    exactly as CC fires them across a turn (UserPromptSubmit -> PreToolUse ->
    Stop). State persists across fires within a test, like the real install."""

    def __init__(self, tmp_path, monkeypatch, capsys, *, session="s1") -> None:
        self.mp = monkeypatch
        self.capsys = capsys
        self.session = session
        self.code = tmp_path / "code"
        self.code.mkdir(parents=True, exist_ok=True)
        self.tpath = tmp_path / "t.jsonl"
        self.events: list[dict] = []
        self._uid = 0
        monkeypatch.setenv("SADDLE_TENANT", "acme")
        monkeypatch.setenv("SADDLE_PROJECT", "game")
        monkeypatch.setenv("SADDLE_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("SADDLE_CODE_ROOT", str(self.code))
        # The intake LLM stages are not drift channels under test -> off (the
        # deterministic pick-drift replay still runs; it is NOT gated by these).
        monkeypatch.setenv("SADDLE_HOOK_ITEMIZE", "0")
        monkeypatch.setenv("SADDLE_HOOK_INTENT", "0")
        # One persisted ledger across every intake fire (a fresh process IRL).
        from saddle.dialog import IntentTracker, set_tracker
        from saddle.dialog_store import InMemoryForkStore

        self.tracker = IntentTracker(store=InMemoryForkStore())
        set_tracker(self.tracker)
        self.dkb: _IntegDKB | None = None
        self.harvest_caller = _HarvestCaller([_ENTRY])

    # -- transcript construction ------------------------------------------
    def _uuid(self) -> str:
        self._uid += 1
        return f"u{self._uid}"

    def _flush(self) -> None:
        self.tpath.write_text("\n".join(json.dumps(o) for o in self.events) + "\n")

    def _append_user(self, text: str) -> None:
        self.events.append({
            "type": "user", "sessionId": self.session, "uuid": self._uuid(),
            "isSidechain": False, "message": {"role": "user", "content": text},
        })
        self._flush()

    def asst(self, text: str) -> None:
        """The agent speaks (an offered fork, a stated approach, a declared
        action) — appended to the transcript the hooks read, like a real turn."""
        self.events.append({
            "type": "assistant", "sessionId": self.session, "uuid": self._uuid(),
            "isSidechain": False,
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": text}]},
        })
        self._flush()

    def write_code(self, src: str) -> None:
        (self.code / "abil.py").write_text(src)

    # -- install the engine stubs (per the test's needs) ------------------
    def surface_design(self, *, design_id="dreal") -> None:
        """Install one settled design whose cooldown surface Stage 4 enforces,
        plus the faked harvest seam Stage 5 files into."""
        from saddle.codemap import SurfaceManifest, ValueSpec
        from saddle.models import DESIGN_FINAL, Design

        self.dkb = _IntegDKB([Design(
            id=design_id, ask="route every cooldown read through the resolver",
            summary="cooldown resolver design", status=DESIGN_FINAL,
            meta={"surface": SurfaceManifest(
                values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd",
                                  ("build_def",))]).to_dict()},
        )])
        self.mp.setattr("saddle.design.get_dkb", lambda: self.dkb)
        self.mp.setattr("saddle.llm.callers.build_callers",
                        lambda ctx: {"default": self.harvest_caller})

    def stub_audit(self) -> None:
        """Stage 3's LLM: a band-aid approach (try/except / swallow / flat tweak)
        -> issues; a root-cause approach -> clean. Deterministic, no network."""
        from saddle.design import AuditVerdict

        async def _audit(goal, approach, ctx=None, **kw):
            a = (approach or "").lower()
            if any(w in a for w in ("try/except", "swallow", "band-aid",
                                    "just subtract", "quick hack", "flat")):
                return AuditVerdict(ok=False,
                                    issues=[f"band-aid: {approach.strip()[:70]}"])
            return AuditVerdict(ok=True, issues=[])

        self.mp.setattr("saddle.design.audit_proposal", _audit)

    # -- fire the real hooks ----------------------------------------------
    def _run(self, mod_name: str, payload: dict):
        self.mp.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        import importlib

        mod = importlib.import_module(f"saddle.{mod_name}")
        rc = mod.main()
        return rc, self.capsys.readouterr()

    def intake(self, prompt: str):
        """UserPromptSubmit: append the submitted prompt to the transcript (as CC
        does) then fire the intake hook (Stages 1 + 2)."""
        self._append_user(prompt)
        return self._run("intake_hook", {
            "prompt": prompt, "session_id": self.session,
            "transcript_path": str(self.tpath),
        })

    def edit(self, *, path=None):
        """PreToolUse on an Edit: the guard + Stage 3 design review."""
        path = path or str(self.code / "abil.py")
        return self._run("doctrine_hook", {
            "tool_name": "Edit", "tool_input": {"file_path": path},
            "session_id": self.session, "transcript_path": str(self.tpath),
        })

    def stop(self):
        """Turn-end: Stage 4 conformance + Stage 5 lesson harvest."""
        return self._run("stop_hook", {
            "session_id": self.session, "transcript_path": str(self.tpath),
        })


# === bubble (panel) + agent-channel helpers =================================

def _bubbles(level=None, stage=None):
    from saddle.bubble import recent_bubbles

    return [b for b in recent_bubbles(_CTX, level=level)
            if stage is None or b.stage == stage]


def _agent_ctx(out) -> str:
    """What the hook injected into the AGENT's context this fire (stdout JSON), or
    '' if the turn was silent — the 'discuss it with the agent' channel."""
    if not out.out.strip():
        return ""
    return json.loads(out.out)["hookSpecificOutput"].get("additionalContext", "")


# === 1. multiple KINDS in one turn, bubbled AND discussed ===================

def test_turn_catches_design_and_code_then_harvests_the_lesson(
    monkeypatch, capsys, tmp_path
):
    """One turn drives all three hooks and trips three stages at once: the agent
    states a band-aid (Stage 3), the tree it leaves drifts from the settled design
    (Stage 4), and the turn-end harvests the lesson (Stage 5). Each lands on the
    durable panel AND is surfaced to the agent — proving saddle both *bubbles* and
    *discusses* every kind it catches, in a single realistic turn."""
    p = _Pipeline(tmp_path, monkeypatch, capsys)
    p.surface_design()
    p.stub_audit()
    p.write_code(_DRIFTED)

    # the prompt opens clean — nothing has drifted yet.
    rc, out = p.intake("make every cooldown read go through the resolver")
    assert rc == 0
    assert _bubbles(stage=STAGE_INTENT) == []          # clean start, no pick-drift

    # the agent states a band-aid approach, then edits.
    p.asst("I'll just subtract 1 from the base cooldown_s with a quick try/except.")
    rc, out = p.edit()
    assert rc == 0
    # discussed with the agent (PreToolUse additionalContext, never a block)...
    assert "band-aid" in _agent_ctx(out)
    # ...and on the durable panel as a stage=design ALERT.
    design_alerts = _bubbles(level="alert", stage=STAGE_DESIGN)
    assert len(design_alerts) == 1 and "band-aid" in design_alerts[0].text

    # turn-end: Stage 4 catches the real conformance break, Stage 5 harvests it.
    rc, out = p.stop()
    assert rc == 0
    code_alerts = _bubbles(level="alert", stage=STAGE_CODE)
    assert len(code_alerts) == 1
    assert "CODE DRIFT" in code_alerts[0].text and "dreal" in code_alerts[0].text
    assert "in sweep():" in code_alerts[0].text          # the real touchpoint
    assert "CODE DRIFT" in out.err                        # discussed on-screen too

    lessons = _bubbles(level="notice", stage=STAGE_LESSON)
    assert len(lessons) == 1 and "Bump is not balance" in lessons[0].text
    # the lesson was actually filed into the DKB, source=audit, project-scoped.
    assert p.dkb is not None and len(p.dkb.added) == 1
    filed = p.dkb.added[0]
    assert filed.kind == LESSON and filed.source == AUDIT
    assert filed.scope_project == "game"


# === 2. the same code drift, caught EVERY turn until corrected ==============

def test_uncorrected_code_drift_is_recaught_until_corrected(
    monkeypatch, capsys, tmp_path
):
    """'Not once and done', through the full per-turn cycle (intake -> edit ->
    stop). The agent records a SOUND approach each turn (so Stage 3 stays silent
    and only the code channel is exercised) but does not actually fix the tree:
    turn 1 catches the conformance break, turn 2 RE-catches the identical drift
    (fresh parse, no cached pass), and only when the human fixes the code on turn 3
    does the panel go quiet — and stays quiet."""
    p = _Pipeline(tmp_path, monkeypatch, capsys)
    p.surface_design()
    p.stub_audit()
    p.write_code(_DRIFTED)

    def a_turn(prompt, approach):
        p.intake(prompt)
        p.asst(approach)
        p.edit()
        return p.stop()

    # turn 1 — caught.
    a_turn("route cooldowns through resolve_cd",
           "Root cause: sweep reads cooldown_s raw; I'll route it through resolve_cd.")
    assert len(_bubbles(level="alert", stage=STAGE_CODE)) == 1
    assert _bubbles(level="alert", stage=STAGE_DESIGN) == []   # sound approach -> silent

    # turn 2 — the human did NOT correct it -> re-caught fresh.
    a_turn("did you fix the cooldown path?",
           "Confirming the resolver routing for sweep is in place.")
    assert len(_bubbles(level="alert", stage=STAGE_CODE)) == 2  # re-caught, not blessed

    # turn 3 — NOW the code is fixed -> the drift clears.
    p.write_code(_FIXED)
    rc, out = a_turn("apply the resolver fix to sweep",
                     "Replacing the raw read in sweep with resolve_cd(inst).")
    assert rc == 0
    assert len(_bubbles(level="alert", stage=STAGE_CODE)) == 2  # no NEW code alert
    assert "CODE DRIFT" not in out.err                          # silent on-screen

    # the harvest was cumulative, never duplicative: one lesson filed across the
    # whole run despite the drift being re-caught (DKB title-dedup held).
    assert p.dkb is not None and len(p.dkb.added) == 1
    assert len(_bubbles(level="notice", stage=STAGE_LESSON)) == 1


# === 3. distinct pick-drifts, caught across turns, cleared on alignment ======

def test_pick_drift_recurs_across_turns_then_clears_on_alignment(
    monkeypatch, capsys, tmp_path
):
    """Stage 2 through the real intake hook, over a session: the agent commits to
    one option then acts on another (the '22-hour drift'), and saddle catches it
    from the transcript it does not control. A SECOND, distinct pick-drift on a
    later fork is caught on its own (multiple times, not a one-shot). When the
    agent finally acts on the option actually committed, nothing drifts — the
    catch tracks the live commitment, it does not cry wolf."""
    p = _Pipeline(tmp_path, monkeypatch, capsys)

    # --- drift #1: offer a/b/c, commit a, then act on b ---
    p.intake("let's decide the cache")
    p.asst(_FORK)
    p.intake("a")                                   # binds a (no action yet -> quiet)
    assert _bubbles(stage=STAGE_INTENT) == []
    p.asst("Going with option b — it's faster.")    # the drift, not yet replayed
    rc, out = p.intake("what's the status?")        # replay surfaces it here
    assert rc == 0
    c = _agent_ctx(out)
    assert "DRIFT" in c and "you committed" in c     # discussed with the agent
    assert len(_bubbles(level="alert", stage=STAGE_INTENT)) == 1

    # --- drift #2: a brand-new fork, same shape -> caught independently ---
    p.asst(_FORK2)
    p.intake("a")                                   # binds the new fork's a
    p.asst("Going with option b.")                  # drift on the new commitment
    rc, out = p.intake("and now?")
    assert "DRIFT" in _agent_ctx(out)
    assert len(_bubbles(level="alert", stage=STAGE_INTENT)) == 2   # caught again

    # --- alignment: commit b, act on b -> NO drift (the noise clears) ---
    p.asst(_FORK3)
    p.intake("b")                                   # binds b
    p.asst("Going with option b.")                  # acts on what was committed
    rc, out = p.intake("done?")
    assert "DRIFT" not in _agent_ctx(out)
    assert len(_bubbles(level="alert", stage=STAGE_INTENT)) == 2   # unchanged


# === 4. capstone: the panel shows every KIND across one session =============

def test_panel_accumulates_every_stage_across_a_session(
    monkeypatch, capsys, tmp_path
):
    """The direct answer to 'ensure it bubbles up texts to this panel': a single
    session that, over three turns, exercises a pick-drift (intent), a band-aid
    (design), a conformance break (code), and a harvest (lesson) — and asserts the
    durable outbox ends with at least one bubble for EACH stage. This is the view
    the AFK / non-TTY human reads; every kind of drift reaches it."""
    p = _Pipeline(tmp_path, monkeypatch, capsys)
    p.surface_design()
    p.stub_audit()
    p.write_code(_DRIFTED)

    # turn 1: a clean prompt; the agent offers a fork; turn-end catches the code
    # drift already on disk + harvests the lesson.
    p.intake("decide the cooldown approach")
    p.asst(_FORK)
    p.stop()
    assert len(_bubbles(level="alert", stage=STAGE_CODE)) >= 1
    assert len(_bubbles(level="notice", stage=STAGE_LESSON)) >= 1

    # turn 2: the user commits a; the agent drifts to b AND states a band-aid then
    # edits -> Stage 3 fires; turn-end re-catches the code.
    p.intake("a")
    p.asst("Going with option b.")                  # pick-drift (surfaces next fire)
    p.asst("I'll just swallow the error in sweep and move on.")  # band-aid
    p.edit()
    p.stop()
    assert len(_bubbles(level="alert", stage=STAGE_DESIGN)) >= 1

    # turn 3: the next prompt replays turn 2's pick-drift -> intent bubble.
    p.intake("status?")
    assert len(_bubbles(level="alert", stage=STAGE_INTENT)) >= 1

    # the panel carries every kind saddle caught this session.
    present = {b.stage for b in _bubbles()}
    assert {STAGE_INTENT, STAGE_DESIGN, STAGE_CODE, STAGE_LESSON} <= present
