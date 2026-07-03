"""intake_hook — the UserPromptSubmit bubble-up hook.

Verifies saddle's per-prompt *observation* channel (the counterpart to the
PreToolUse enforcement hook): that a submitted prompt is replayed for drift,
itemized, and that the commitment + scope surface ride back as
``additionalContext`` — and that EVERY failure path fails OPEN (exit 0, never a
block). The LLM itemize is stubbed so the suite stays deterministic and offline;
the drift/commitment/scope paths are exercised for real against the in-memory
ledger.
"""
from __future__ import annotations

import io
import json

import pytest

from saddle.context import Context
from saddle.dialog import IntentTracker, set_tracker
from saddle.dialog_store import InMemoryForkStore
from saddle.models import Intake, Item


# --- transcript builders (same on-disk shape the adapter parses) -----------

def _asst(text="", *, session="s1", uuid="", sidechain=False):
    content = [{"type": "text", "text": text}] if text else []
    return {
        "type": "assistant", "sessionId": session, "uuid": uuid,
        "isSidechain": sidechain, "message": {"role": "assistant", "content": content},
    }


def _user(text, *, session="s1", uuid=""):
    return {
        "type": "user", "sessionId": session, "uuid": uuid, "isSidechain": False,
        "message": {"role": "user", "content": text},
    }


_FORK = "Here's the call on caching:\na) in-memory LRU\nb) redis\nc) sqlite cache"


def _write_transcript(tmp_path, objs):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return p


# --- harness ---------------------------------------------------------------

def _run_hook(payload, monkeypatch, capsys, tmp_path, *, tracker=None, env=None):
    monkeypatch.setenv("SADDLE_TENANT", "acme")
    monkeypatch.setenv("SADDLE_PROJECT", "game")
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    monkeypatch.setenv("SADDLE_CODE_ROOT", str(tmp_path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    set_tracker(tracker or IntentTracker(store=InMemoryForkStore()))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    from saddle import intake_hook
    rc = intake_hook.main()
    return rc, capsys.readouterr()


def _context(out) -> str:
    """The additionalContext saddle injected (stdout JSON), or '' if it stayed silent."""
    if not out.out.strip():
        return ""
    return json.loads(out.out)["hookSpecificOutput"]["additionalContext"]


def _system_msg(out) -> str:
    """The on-screen ``systemMessage`` saddle emitted (stdout JSON), or '' if it had
    nothing screen-worthy to herald (the user-screen channel — ask #3)."""
    if not out.out.strip():
        return ""
    return json.loads(out.out).get("systemMessage", "")


# --- fail-open / no-op edges -----------------------------------------------

def test_empty_stdin_is_silent(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    from saddle import intake_hook
    assert intake_hook.main() == 0
    assert capsys.readouterr().out.strip() == ""


def test_unparseable_payload_fails_open(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    from saddle import intake_hook
    assert intake_hook.main() == 0
    out = capsys.readouterr()
    assert out.out.strip() == ""
    assert "unparseable" in out.err


def test_blank_prompt_is_silent(monkeypatch, capsys, tmp_path):
    rc, out = _run_hook({"prompt": "   ", "session_id": "s1"}, monkeypatch, capsys, tmp_path,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert rc == 0 and out.out.strip() == ""


def test_itemize_failure_fails_loud_not_open(monkeypatch, capsys, tmp_path):
    """Stage 1's whole point: a failed itemize is no longer swallowed to a quiet
    "(unavailable)" degrade — it is CLASSIFIED and surfaced LOUD. The hook still
    never blocks (rc 0 — observation, not enforcement), but the agent context now
    names what saddle did NOT verify and the root-cause exception, AND a durable
    ALERT bubble lands so an AFK / non-TTY human sees the gap."""
    async def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr("saddle.intake.decompose", boom)
    rc, out = _run_hook(
        {"prompt": "audit the whole pipeline end to end please", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    assert rc == 0                                  # observation never blocks
    c = _context(out)
    assert "did not run" in c                     # loud, not swallowed
    assert "went unchecked" in c                    # names the unverified subject
    assert "RuntimeError" in c                      # the root cause is surfaced
    assert "itemization unavailable" not in c       # the old fail-open string is gone
    # the durable ALERT bubble landed under the intake stage (the AFK channel).
    from saddle.bubble import recent_bubbles
    from saddle.context import Context
    alerts = recent_bubbles(Context(tenant="acme", project="game"), level="alert")
    assert any(b.stage == "intake" and "did not run" in b.text for b in alerts)


# --- the slow path: itemized know/do list + scope warning ------------------

def _fake_intake(prompt):
    return Intake(
        raw_prompt=prompt,
        summary="Fix the camera and (out of scope) build a separate launcher.",
        items=[
            Item(kind="task", ask="Fix the camera-follow bug in the launcher"),
            Item(kind="task", ask="Build the matrix terminal launcher in the other repo"),
        ],
        meta={
            "scope_warning": "1 of 2 item(s) target work OUTSIDE the focus project (game).",
            "out_of_focus": [{"n": 2, "ask": "Build the matrix terminal launcher in the other repo",
                              "reason": "a different repository/product"}],
            "todo_items": 2, "audit_complete": True,
        },
    )


def test_substantive_prompt_itemizes_and_surfaces_scope(monkeypatch, capsys, tmp_path):
    async def fake(prompt, ctx=None, **k):
        return _fake_intake(prompt)
    monkeypatch.setattr("saddle.intake.decompose", fake)
    rc, out = _run_hook(
        {"prompt": "fix the camera follow bug and build the matrix launcher", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    ctx = _context(out)
    assert rc == 0
    assert "Fix the camera-follow bug" in ctx                 # the know/do list rendered
    assert "SCOPE" in ctx and "out of focus" in ctx           # the cross-project drift surfaced
    assert "saddle [acme/game]" in ctx                        # spoken under the isolation key


def test_trivial_prompt_skips_both_llm_checks(monkeypatch, capsys, tmp_path):
    """A bare continuation has no braided asks AND no new intent to weigh, so BOTH
    LLM stages — the intake itemize (Stage 1) and the intent history axis (Stage 2)
    — are skipped, and skipped is SILENT, not a failure. (run_stage would otherwise
    CLASSIFY a wrongly-called engine into a loud ALERT and mask the regression, so
    we assert NEITHER engine was reached via sentinels AND that the turn stayed
    fully silent — no spurious section or bubble.)"""
    reached = {"itemize": False, "history": False}
    async def no_itemize(*a, **k):                             # must NOT be reached
        reached["itemize"] = True
        raise AssertionError("decompose called for a trivial prompt")
    async def no_history(*a, **k):                             # must NOT be reached
        reached["history"] = True
        raise AssertionError("history_drift called for a trivial prompt")
    monkeypatch.setattr("saddle.intake.decompose", no_itemize)
    monkeypatch.setattr("saddle.intent.history_drift", no_history)
    rc, out = _run_hook({"prompt": "proceed", "session_id": "s1"}, monkeypatch, capsys, tmp_path)
    assert rc == 0
    assert reached == {"itemize": False, "history": False}     # neither engine reached
    assert _context(out) == ""                                 # trivial turn is silent


# --- Stage 2 history axis: this prompt vs the project's settled state -------

def test_history_axis_surfaces_design_drift(monkeypatch, capsys, tmp_path):
    """The genuine new intent axis wired into the hook: a prompt that contradicts
    a settled design surfaces a hard divergence in the agent context AND as a
    durable stage=intent ALERT. Itemize is disabled so only the history finding
    shows — proving the history stage is what bubbled it."""
    from saddle import intent
    def history(*a, **k):                                      # async -> awaitable report
        async def _r():
            return intent.IntentReport(
                divergences=[intent.IntentDivergence(
                    kind=intent.CONTRADICTS_DESIGN,
                    what="switches the cache to redis", ref="design_1")],
                checked=True, considered=2,
            )
        return _r()
    monkeypatch.setattr("saddle.intent.history_drift", history)
    rc, out = _run_hook(
        {"prompt": "rip out the LRU cache and switch to redis", "session_id": "s1"},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "Conflicts with an earlier design" in ctx        # the history finding rendered
    assert "switches the cache to redis" in ctx
    from saddle.bubble import recent_bubbles
    from saddle.context import Context
    alerts = recent_bubbles(Context(tenant="acme", project="game"), level="alert")
    assert any(b.stage == "intent" and "Conflicts with an earlier design" in b.text for b in alerts)


def test_history_axis_failure_fails_loud(monkeypatch, capsys, tmp_path):
    """A FAILED history check is not swallowed: like the intake stage it classifies
    and bubbles a loud stage=intent ALERT naming what saddle did NOT verify, and the
    hook still never blocks (rc 0 — observation, not enforcement)."""
    async def boom(*a, **k):
        raise RuntimeError("dkb down")
    monkeypatch.setattr("saddle.intent.history_drift", boom)
    rc, out = _run_hook(
        {"prompt": "does this contradict the cache design we settled on?", "session_id": "s1"},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    assert rc == 0
    c = _context(out)
    assert "did not run" in c and "went unchecked" in c and "RuntimeError" in c
    from saddle.bubble import recent_bubbles
    from saddle.context import Context
    alerts = recent_bubbles(Context(tenant="acme", project="game"), level="alert")
    assert any(b.stage == "intent" and "did not run" in b.text for b in alerts)


# --- the user-screen channel (ask #3: saddle visible on screen) -------------

def test_itemization_heralds_on_screen(monkeypatch, capsys, tmp_path):
    """ask #3 — a successful intake itemization is visible to the HUMAN on the
    ``systemMessage`` channel (the one Claude Code renders on screen), not only
    injected into the agent's ``additionalContext``. So the user SEES that saddle
    read and decomposed the prompt."""
    async def fake(prompt, ctx=None, **k):
        return _fake_intake(prompt)
    monkeypatch.setattr("saddle.intake.decompose", fake)
    rc, out = _run_hook(
        {"prompt": "fix the camera follow bug and build the matrix launcher", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    assert rc == 0
    sm = _system_msg(out)
    assert "Fix the camera-follow bug" in sm
    assert "saddle [acme/game]" in sm


def test_itemize_failure_heralds_on_screen(monkeypatch, capsys, tmp_path):
    """The cardinal case: a could-not-run intake stage heralds 'did NOT verify' to
    the HUMAN on screen, not just to the agent — so an AFK/non-TTY user can never
    miss that saddle's check was incomplete this turn."""
    async def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr("saddle.intake.decompose", boom)
    rc, out = _run_hook(
        {"prompt": "audit the whole pipeline end to end please", "session_id": "s1"},
        monkeypatch, capsys, tmp_path,
    )
    assert rc == 0
    sm = _system_msg(out)
    assert "did not run" in sm and "went unchecked" in sm


def test_replay_drift_heralds_on_screen(monkeypatch, capsys, tmp_path):
    """A pick-drift caught from the transcript ('committed a) then acted on b)') —
    saddle's signature catch — reaches the screen via systemMessage, not only the
    agent's context."""
    tp = _write_transcript(tmp_path, [
        _user("let's decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
        _user("a", uuid="u3"),
        _asst(text="Going with option b — it's faster.", uuid="u4"),
    ])
    rc, out = _run_hook(
        {"prompt": "what's the status?", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    assert rc == 0
    assert "DRIFT" in _system_msg(out) and "you committed" in _system_msg(out)


def test_commitment_is_agent_only_not_an_on_screen_herald(monkeypatch, capsys, tmp_path):
    """The standing commitment is an AGENT reminder, not a screen herald: when the
    only thing to surface is the commitment (a clean pick, itemize off), the agent
    sees it via additionalContext but the systemMessage stays empty — so the user's
    screen isn't spammed every turn there's an active binding."""
    tp = _write_transcript(tmp_path, [
        _user("decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
    ])
    rc, out = _run_hook(
        {"prompt": "a", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    assert rc == 0
    assert "carry-forward" in _context(out)          # the agent sees the carry-forward
    assert _system_msg(out) == ""                    # but the screen stays quiet


# --- the fast path: drift caught from the transcript -----------------------

def test_replay_surfaces_a_then_b_drift(monkeypatch, capsys, tmp_path):
    tp = _write_transcript(tmp_path, [
        _user("let's decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),                          # offers a/b/c
        _user("a", uuid="u3"),                                 # binds a
        _asst(text="Going with option b — it's faster.", uuid="u4"),  # DRIFT
    ])
    rc, out = _run_hook(
        {"prompt": "what's the status?", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "DRIFT" in ctx and "you committed" in ctx


def test_current_prompt_pick_binds_commitment(monkeypatch, capsys, tmp_path):
    # the agent offered a fork (in the transcript); the user's CURRENT prompt is
    # the pick — the commitment must surface THIS turn, not a turn late.
    tp = _write_transcript(tmp_path, [
        _user("decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
    ])
    rc, out = _run_hook(
        {"prompt": "a", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "carry-forward" in ctx and "p1.f1.a" in ctx


def test_commitment_readout_surfaces_the_proposal_and_every_option(monkeypatch, capsys, tmp_path):
    # The readout must carry the DECISION, not a bare locator: the proposal it
    # answered and EVERY option (the committed one marked) so the agent re-reads
    # what p1.f1.a actually meant after the transcript compacts — the fork row is
    # saddle's durable store, so only the rendering was missing.
    tp = _write_transcript(tmp_path, [
        _user("decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
    ])
    rc, out = _run_hook(
        {"prompt": "a", "session_id": "s1", "transcript_path": str(tp)},
        monkeypatch, capsys, tmp_path, env={"SADDLE_HOOK_ITEMIZE": "0"},
    )
    ctx = _context(out)
    assert rc == 0
    assert "carry-forward" in ctx and "p1.f1.a" in ctx         # choice id rides as a [ref]
    assert "proposal" in ctx and "caching" in ctx              # the question it answered
    # every option surfaces — the chosen one as the directive, the rest as a guard ...
    assert "in-memory LRU" in ctx and "redis" in ctx and "sqlite cache" in ctx
    assert "this is what to do now" in ctx                # ... the chosen one foregrounded


def test_cursor_makes_replay_idempotent(monkeypatch, capsys, tmp_path):
    # one durable ledger across two hook fires (a fresh process each time IRL);
    # the persisted cursor means the same drift is NOT re-surfaced next prompt.
    tracker = IntentTracker(store=InMemoryForkStore())
    tp = _write_transcript(tmp_path, [
        _asst(text=_FORK, uuid="u1"),
        _user("a", uuid="u2"),
        _asst(text="Going with option b.", uuid="u3"),         # DRIFT
    ])
    payload = {"prompt": "status?", "session_id": "s1", "transcript_path": str(tp)}
    _, out1 = _run_hook(payload, monkeypatch, capsys, tmp_path, tracker=tracker,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert "DRIFT" in _context(out1)
    _, out2 = _run_hook(payload, monkeypatch, capsys, tmp_path, tracker=tracker,
                        env={"SADDLE_HOOK_ITEMIZE": "0"})
    assert "DRIFT" not in _context(out2)                       # cursor advanced past it
