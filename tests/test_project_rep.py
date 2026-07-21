"""The project rep (#project-rep, the live-brain backbone): assembly from the live
substrate, token-bounded render, and the two safety properties saddle's own review
flagged — no map probe renders ABSENT (never a false 'healthy'), and every source
is fail-soft (a failing source empties its field, never sinks the rep).
"""

from __future__ import annotations

import pytest

from saddle.context import Context
from saddle.models import (
    Binding,
    Design,
    Fork,
    ForkOption,
    Knowledge,
    TN_TOPIC,
    TN_WORK,
    TS_BUILT_UNTESTED,
    TopicNode,
)
from saddle.project_rep import (
    DEFAULT_TOKEN_BUDGET,
    ProjectRep,
    assemble,
    render,
    rep_block,
)
from saddle.topic_store import reset_topic_store, set_topic_store

_CTX = Context(tenant="acme", project="game")


# -- stubs -------------------------------------------------------------------

class _StubTracker:
    def __init__(self, binding=None, fork=None):
        self._b, self._f = binding, fork

    def committed_fork(self, ctx, *, session=""):
        return self._b, self._f


class _StubTopicStore:
    def __init__(self, nodes=None):
        self._nodes = nodes or []

    def list_nodes(self, ctx, *, status=None, type=None, topic_key=None, limit=200):
        out = [n for n in self._nodes if status is None or n.status == status]
        return out[:limit]

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _empty_topic_store():
    """Default every rep test to an EMPTY topic store, so no test touches the real
    ~/.saddle db when assemble lazily resolves the singleton. A test that wants
    threads passes its own ``topic_store=`` explicitly."""
    set_topic_store(_StubTopicStore([]))
    yield
    reset_topic_store()


class _StubDKB:
    def __init__(self, designs=None, knowledge=None, raise_on=None):
        self._designs = designs or []
        self._knowledge = knowledge or []
        self._raise_on = raise_on or set()

    def list_designs(self, ctx, *, status=None, limit=50):
        if "designs" in self._raise_on:
            raise RuntimeError("designs boom")
        return [d for d in self._designs if status is None or d.status == status][:limit]

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        if "knowledge" in self._raise_on:
            raise RuntimeError("knowledge boom")
        return [(kn, 1.0) for kn in self._knowledge[:k]]


def _committed():
    fork = Fork(options=[ForkOption(label="a", text="build the live brain")],
                prompt="how to structure it?", id="p1.f1")
    return Binding(fork_id="p1.f1", label="a", resolved=True), fork


def _dkb(**kw):
    return _StubDKB(
        designs=[Design(ask="gate the council", summary="council on non-trivial only",
                        status="final")],
        knowledge=[Knowledge(kind="lesson", title="No band-aids", body="fix at source"),
                   Knowledge(kind="fact", title="Identity", body="saddle is the harness")],
        **kw,
    )


# -- assembly ----------------------------------------------------------------

def test_assemble_pulls_intent_designs_knowledge_and_map():
    b, f = _committed()
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f),
                   map_probe=lambda ctx: "fresh")
    assert "build the live brain" in rep.intent
    assert rep.designs == ["council on non-trivial only"]
    assert ("lesson", "No band-aids", "fix at source") in rep.knowledge
    assert rep.map_status == "fresh"
    assert rep.token_budget == DEFAULT_TOKEN_BUDGET


def test_no_map_probe_leaves_status_empty_not_false_healthy():
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(), map_probe=None)
    assert rep.map_status == ""                         # ABSENT, never a fake "fresh"
    assert "STRUCTURAL MAPS" not in render(rep)         # the section is simply omitted


def test_map_probe_crash_is_unknown_not_a_wedge():
    def _boom(ctx):
        raise RuntimeError("probe down")
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(), map_probe=_boom)
    assert rep.map_status == ""                         # crash -> unknown, rep still built
    assert rep.knowledge                                # the rest of the rep survived


def test_assemble_is_fail_soft_per_source():
    b, f = _committed()
    # designs source raises -> that field empties, intent + knowledge still land.
    rep = assemble(_CTX, dkb=_dkb(raise_on={"designs"}), tracker=_StubTracker(b, f))
    assert rep.designs == []
    assert "build the live brain" in rep.intent
    assert rep.knowledge


def test_assemble_no_commitment_gives_empty_intent():
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(None, None))
    assert rep.intent == ""


# -- render ------------------------------------------------------------------

def test_render_foregrounds_intent_and_lists_sections():
    b, f = _committed()
    block = render(assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f),
                            map_probe=lambda ctx: "stale: 3 files"))
    assert "CURRENT INTENT (act on this):" in block
    assert block.index("CURRENT INTENT") < block.index("SETTLED DESIGNS")  # intent first
    assert "STRUCTURAL MAPS: stale: 3 files" in block
    assert "No band-aids" in block


def test_render_is_token_bounded():
    # A tiny budget must truncate — the fed block can never flood the agent.
    b, f = _committed()
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f))
    tiny = render(rep, budget=8)                        # 8 tokens ≈ 32 chars
    assert "truncated to the token budget" in tiny
    assert len(tiny) < 400


def test_empty_rep_renders_empty():
    assert render(ProjectRep(project_key="acme/game")) == ""


def test_rep_block_assembles_and_renders():
    b, f = _committed()
    block = rep_block(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f))
    assert "PROJECT REP — acme/game" in block and "build the live brain" in block


# -- OPEN THREADS (the topic mind-map, Slice #3) -----------------------------

def test_open_threads_surface_with_work_state():
    nodes = [
        TopicNode(title="B1b instance server", type=TN_WORK,
                  testing_state=TS_BUILT_UNTESTED, status="open"),
        TopicNode(title="dungeon quality", type=TN_TOPIC, status="open"),
    ]
    b, f = _committed()
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f),
                   topic_store=_StubTopicStore(nodes))
    assert rep.topics == ["B1b instance server [work: built_untested]",
                          "dungeon quality [topic]"]
    block = render(rep)
    assert "OPEN THREADS" in block and "[work: built_untested]" in block
    # threads surface right after the intent, before the settled designs.
    assert (block.index("CURRENT INTENT") < block.index("OPEN THREADS")
            < block.index("SETTLED DESIGNS"))


def test_topics_source_is_fail_soft():
    class _BoomStore:
        def list_nodes(self, *a, **k):
            raise RuntimeError("topics boom")
        def close(self):
            pass
    b, f = _committed()
    rep = assemble(_CTX, dkb=_dkb(), tracker=_StubTracker(b, f), topic_store=_BoomStore())
    assert rep.topics == []                 # a failing source empties its field…
    assert "build the live brain" in rep.intent   # …and never sinks the rep
