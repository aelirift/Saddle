"""Transcript adapter — parse the real Claude Code JSONL shapes and replay a
conversation into the tracker, catching the "picked a) then did b)" drift by
reading the log saddle does not control.
"""
from __future__ import annotations

import json

from saddle.context import Context
from saddle.dialog import IntentTracker
from saddle.dialog_store import InMemoryForkStore
from saddle.transcript import (
    latest_turn,
    parse_transcript_line,
    read_transcript,
    replay,
)


def _asst(text="", thinking="", *, session="s1", uuid="", sidechain=False):
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    if text:
        content.append({"type": "text", "text": text})
    return {
        "type": "assistant", "sessionId": session, "uuid": uuid,
        "cwd": "/home/u/proj", "gitBranch": "main", "isSidechain": sidechain,
        "timestamp": "2026-06-25T00:00:00Z",
        "message": {"role": "assistant", "content": content},
    }


def _user(text, *, session="s1", uuid="", as_blocks=False):
    content = [{"type": "text", "text": text}] if as_blocks else text
    return {
        "type": "user", "sessionId": session, "uuid": uuid,
        "cwd": "/home/u/proj", "gitBranch": "main", "isSidechain": False,
        "userType": "external", "timestamp": "2026-06-25T00:00:00Z",
        "message": {"role": "user", "content": content},
    }


def _tool_result(uuid="", session="s1"):
    return {
        "type": "user", "sessionId": session, "uuid": uuid, "isSidechain": False,
        "message": {"role": "user",
                    "content": [{"type": "tool_result", "content": "ok"}]},
    }


_FORK = "Here's the call on caching:\na) in-memory LRU\nb) redis\nc) sqlite cache"


def _write(tmp_path, objs):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    return p


# === parsing discrimination ==============================================

def test_tool_result_user_line_is_not_a_prompt():
    assert parse_transcript_line(_tool_result(uuid="x")) is None


def test_sidechain_assistant_is_skipped():
    assert parse_transcript_line(_asst(text=_FORK, sidechain=True)) is None


def test_non_dialog_types_are_skipped():
    for o in [{"type": "system", "content": "x"}, {"type": "ai-title"},
              {"type": "attachment"}, {"type": "last-prompt"}, "garbage", 5]:
        assert parse_transcript_line(o) is None


def test_user_prompt_as_text_blocks_is_parsed():
    ev = parse_transcript_line(_user("go with a", as_blocks=True))
    assert ev is not None and ev.role == "user" and ev.text == "go with a"


def test_assistant_thinking_is_carried_not_spoken():
    ev = parse_transcript_line(_asst(text="hi", thinking="secret reasoning"))
    assert ev.text == "hi" and ev.thinking == "secret reasoning"


# === the headline: replay catches pick a) then do b) =====================

def test_replay_catches_a_then_b_drift(tmp_path):
    objs = [
        _user("let's decide the cache", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),                       # offers a/b/c
        _tool_result(uuid="u3"),                            # ignored
        _user("a", uuid="u4"),                              # binds a
        _asst(thinking="a is fine but b is faster",
              text="Going with option b — it's faster.", uuid="u5"),  # DRIFT
        _asst(text="Now implementing approach a as you chose.", uuid="u6"),  # aligned
        _asst(text="Sub split:\na) foo\nb) bar", uuid="u7", sidechain=True),  # ignored
        {"type": "system", "content": "compacted", "uuid": "u8"},   # ignored
    ]
    p = _write(tmp_path, objs)
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())

    res = replay(ctx, p, tracker=tracker)
    assert res.forks == 1                      # the sidechain fork was NOT counted
    assert res.bindings == 1
    assert len(res.drifts) == 1
    d = res.drifts[0]
    assert d.bound_label == "a" and d.action_label == "b" and d.is_drift
    assert res.last_uuid == "u6"               # last DIALOG event (u7/u8 skipped)


def test_after_uuid_cursor_resumes_past_seen_lines(tmp_path):
    objs = [
        _user("let's decide", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),
        _tool_result(uuid="u3"),
        _user("a", uuid="u4"),
    ]
    p = _write(tmp_path, objs)
    # resume after the fork line -> first event seen is the user pick "a"
    evs = list(read_transcript(p, after_uuid="u2"))
    assert [e.role for e in evs] == ["user"]
    assert evs[0].text == "a"


def test_replay_is_idempotent_with_cursor(tmp_path):
    objs = [_asst(text=_FORK, uuid="u1"), _user("a", uuid="u2")]
    p = _write(tmp_path, objs)
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    first = replay(ctx, p, tracker=tracker)
    assert first.events == 2 and first.last_uuid == "u2"
    # tailing forward from the cursor sees nothing new
    again = replay(ctx, p, tracker=tracker, after_uuid=first.last_uuid)
    assert again.events == 0 and again.drifts == []


def test_replay_session_isolation(tmp_path):
    # a fork in session s1; a pick in session s2 must not bind it.
    objs = [_asst(text=_FORK, uuid="u1", session="s1"),
            _user("a", uuid="u2", session="s2")]
    p = _write(tmp_path, objs)
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    res = replay(ctx, p, tracker=tracker)
    assert res.forks == 1 and res.bindings == 0   # cross-session pick did not bind


def test_missing_transcript_is_empty_not_an_error(tmp_path):
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    res = replay(ctx, tmp_path / "nope.jsonl", tracker=tracker)
    assert res.events == 0 and res.drifts == []


# === fork-identity drift through the transcript ==========================

_FORK2 = "Pick the store backend:\na) flat file\nb) postgres\nc) sqlite"


def test_replay_catches_wrong_fork_citation_as_drift(tmp_path):
    """The agent cites the "a" of a DIFFERENT, real fork (p2.f1) than the committed
    one (p1.f1). Same letter, wrong fork -> the 22-hour drift, caught from the log.
    The cited node is one saddle minted, so this is a real drift, not a stray
    number masquerading as a citation."""
    objs = [
        _user("decide caching", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),                      # fork node p1.f1
        _user("a", uuid="u3"),                             # binds p1.f1.a
        _asst(text=_FORK2, uuid="u4"),                     # a real DIFFERENT fork, node p2.f1
        _asst(text="Acting on p2.f1.a now.", uuid="u5"),   # the other fork's a -> DRIFT
    ]
    p = _write(tmp_path, objs)
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    res = replay(ctx, p, tracker=tracker)
    assert len(res.drifts) == 1
    d = res.drifts[0]
    assert d.bound_choice == "p1.f1.a" and d.action_choice == "p2.f1.a"
    assert d.is_drift and d.announce


def test_replay_surfaces_ambiguous_bare_label_as_must_confirm(tmp_path):
    """A bare "option a" when TWO open forks both offer "a" is not a deterministic
    drift — but it must not go silent either. It lands in ``confirms`` (surfaced),
    so saddle forces the agent to cite the fork-choice id."""
    objs = [
        _user("decide caching", uuid="u1"),
        _asst(text=_FORK, uuid="u2"),                       # fork p1.f1 (a/b/c)
        _user("a", uuid="u3"),                              # binds p1.f1.a
        _asst(text=_FORK2, uuid="u4"),                      # fork p2.f1 (a/b/c)
        _asst(text="Going with option a.", uuid="u5"),      # ambiguous across forks
    ]
    p = _write(tmp_path, objs)
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    res = replay(ctx, p, tracker=tracker)
    assert res.forks == 2
    assert res.drifts == []                  # not a deterministic drift...
    assert len(res.confirms) == 1            # ...but surfaced, never swallowed
    assert len(res.surfaced) == 1
    c = res.confirms[0]
    assert not c.is_drift and c.announce


# === latest_turn — the pre-code design gate's transcript view ============

def test_latest_turn_lifts_goal_approach_and_anchor(tmp_path):
    objs = [
        _user("fix the cache bug", uuid="u1"),
        _asst(text="The root cause is an unbounded map.",
              thinking="secret messy reasoning", uuid="u2"),
        _asst(text="I'll bound it with an LRU and evict.", uuid="u3"),
    ]
    p = _write(tmp_path, objs)
    turn = latest_turn(p)
    assert turn.goal == "fix the cache bug"
    assert turn.anchor == "u1"
    assert "root cause is an unbounded map" in turn.approach
    assert "bound it with an LRU" in turn.approach
    assert "secret" not in turn.approach        # thinking is NOT the recorded design


def test_latest_turn_uses_only_the_last_turns_prose(tmp_path):
    """Each user prompt opens a new turn, resetting the approach — the gate audits
    THIS turn's plan, not prose carried over from an earlier ask."""
    objs = [
        _user("first ask", uuid="u1"),
        _asst(text="approach for the first ask", uuid="u2"),
        _user("second ask", uuid="u3"),
        _asst(text="approach for the second ask", uuid="u4"),
    ]
    p = _write(tmp_path, objs)
    turn = latest_turn(p)
    assert turn.goal == "second ask" and turn.anchor == "u3"
    assert turn.approach == "approach for the second ask"
    assert "first ask" not in turn.approach


def test_latest_turn_empty_approach_when_agent_jumps_to_edit(tmp_path):
    """A user prompt with NO spoken approach before the edit -> empty approach but a
    real anchor; the 'no recorded design' finding is the hook's to make from this."""
    p = _write(tmp_path, [_user("just do it", uuid="u1")])
    turn = latest_turn(p)
    assert turn.goal == "just do it" and turn.anchor == "u1"
    assert turn.approach == ""


def test_latest_turn_no_user_turn_has_empty_anchor(tmp_path):
    p = _write(tmp_path, [_asst(text="musing with no prompt", uuid="u1")])
    turn = latest_turn(p)
    assert turn.anchor == "" and turn.goal == ""


def test_latest_turn_missing_transcript_is_empty(tmp_path):
    turn = latest_turn(tmp_path / "nope.jsonl")
    assert turn.anchor == "" and turn.approach == ""


def test_latest_turn_ignores_tool_results_and_sidechains(tmp_path):
    objs = [
        _user("the ask", uuid="u1"),
        _tool_result(uuid="u2"),                                   # not the user
        _asst(text="real approach", uuid="u3"),
        _asst(text="sidechain noise", uuid="u4", sidechain=True),  # sub-agent, skipped
    ]
    p = _write(tmp_path, objs)
    turn = latest_turn(p)
    assert turn.goal == "the ask" and turn.anchor == "u1"
    assert turn.approach == "real approach"
    assert "sidechain" not in turn.approach
