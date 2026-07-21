"""The topic mind-map store (Slice #3): typed nodes with per-type closure, the
``work`` testing-state machine + evidence-gated close, soft drop/reopen (there is
NO hard delete), conditional back-edges, and (tenant, project) isolation.

Offline + deterministic — a fresh in-memory SQLite store per test.
"""

from __future__ import annotations

import pytest

from saddle.context import Context
from saddle.models import (
    CLOSED,
    DROPPED,
    EDGE_BACK,
    EDGE_CONTAINS,
    OPEN,
    TN_DECISION,
    TN_FINDING,
    TN_QUESTION,
    TN_ROOT,
    TN_TOPIC,
    TN_WORK,
    TS_BUILT_UNTESTED,
    TS_CLOSED,
    TS_NONE,
    TS_TESTED_UNCLOSED,
    TopicEdge,
    TopicNode,
)
from saddle.topic_store import SqliteTopicStore

CTX = Context(tenant="acme", project="game")
OTHER_PROJECT = Context(tenant="acme", project="other")
OTHER_TENANT = Context(tenant="beta", project="game")


@pytest.fixture
def store():
    s = SqliteTopicStore(":memory:")
    yield s
    s.close()


def _node(store, ctx=CTX, **kw) -> TopicNode:
    kw.setdefault("title", "a thread")
    return store.add_node(ctx, TopicNode(**kw))


# -- add / get / list --------------------------------------------------------

def test_add_node_stamps_ids_and_round_trips(store):
    n = _node(store, type=TN_TOPIC, topic_key="stage-1")
    assert n.id.startswith("topic_")
    assert n.tenant == "acme" and n.project == "game"
    assert n.ts > 0 and n.updated_ts >= n.ts
    got = store.get_node(CTX, n.id)
    assert got is not None and got.title == "a thread" and got.topic_key == "stage-1"


def test_get_missing_node_is_none(store):
    assert store.get_node(CTX, "topic_deadbeef") is None


@pytest.mark.parametrize("field,bad", [
    ("type", "not_a_type"),
    ("status", "not_a_status"),
    ("testing_state", "not_a_state"),
])
def test_add_node_rejects_invalid_enums(store, field, bad):
    with pytest.raises(ValueError):
        store.add_node(CTX, TopicNode(title="x", **{field: bad}))


def test_list_nodes_filters_and_orders_newest_first(store):
    a = _node(store, type=TN_TOPIC, topic_key="k1")
    b = _node(store, type=TN_WORK, topic_key="k2")
    # newest-updated first: touch a AFTER b so it leads.
    store.set_reopen_flag(CTX, a.id, True)
    ids = [n.id for n in store.list_nodes(CTX)]
    assert ids[0] == a.id and set(ids) == {a.id, b.id}
    assert [n.id for n in store.list_nodes(CTX, type=TN_WORK)] == [b.id]
    assert [n.id for n in store.list_nodes(CTX, topic_key="k1")] == [a.id]


# -- the work testing-state machine ------------------------------------------

def test_work_advances_forward_only(store):
    w = _node(store, type=TN_WORK)
    assert w.testing_state == TS_NONE
    assert store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED).testing_state == TS_BUILT_UNTESTED
    assert store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED).testing_state == TS_TESTED_UNCLOSED
    # backwards is refused — a build never un-happens.
    with pytest.raises(ValueError):
        store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)


def test_advance_testing_cannot_reach_closed(store):
    w = _node(store, type=TN_WORK)
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    with pytest.raises(ValueError):
        store.advance_testing(CTX, w.id, TS_CLOSED)   # closing is evidence-gated


def test_testing_state_only_on_work_nodes(store):
    t = _node(store, type=TN_TOPIC)
    with pytest.raises(ValueError):
        store.advance_testing(CTX, t.id, TS_BUILT_UNTESTED)


def test_is_reminder_tracks_build_but_unclosed(store):
    w = _node(store, type=TN_WORK)
    assert not store.get_node(CTX, w.id).is_reminder()          # none -> not yet
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    assert store.get_node(CTX, w.id).is_reminder()              # built, not closed -> nag
    reminders = store.reminders(CTX)
    assert [n.id for n in reminders] == [w.id]


# -- evidence-gated close (the crux: no evidence, no close) -------------------

def test_work_close_requires_evidence(store):
    w = _node(store, type=TN_WORK)
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED)
    with pytest.raises(ValueError):
        store.close_node(CTX, w.id)                             # no evidence -> refuse
    with pytest.raises(ValueError):
        store.close_node(CTX, w.id, evidence="   ")            # blank isn't evidence


def test_work_close_requires_tested_state(store):
    w = _node(store, type=TN_WORK)
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    with pytest.raises(ValueError):
        store.close_node(CTX, w.id, evidence="ran pytest")     # not tested yet


def test_work_close_with_evidence_succeeds_and_records_it(store):
    w = _node(store, type=TN_WORK)
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED)
    closed = store.close_node(CTX, w.id, evidence="pytest 42 passed")
    assert closed.status == CLOSED and closed.testing_state == TS_CLOSED
    assert "pytest 42 passed" in closed.recorded_context     # evidence folded in


# -- per-type closure semantics ----------------------------------------------

def test_root_and_topic_refuse_close_over_open_children(store):
    root = _node(store, type=TN_ROOT, title="the ask")
    child = _node(store, type=TN_TOPIC, title="a stage")
    store.add_edge(CTX, TopicEdge(parent_id=root.id, child_id=child.id))
    with pytest.raises(ValueError):
        store.close_node(CTX, root.id)                         # child still open
    # close the child, then the parent closes.
    store.close_node(CTX, child.id)
    assert store.close_node(CTX, root.id).status == CLOSED


def test_decision_and_question_close_directly(store):
    d = _node(store, type=TN_DECISION, title="which port pool?")
    q = _node(store, type=TN_QUESTION, title="which host?")
    assert store.close_node(CTX, d.id).status == CLOSED
    assert store.close_node(CTX, q.id).status == CLOSED


# -- soft drop / reopen (no hard delete) -------------------------------------

def test_drop_is_soft_and_keeps_why_summary(store):
    n = _node(store, type=TN_TOPIC)
    dropped = store.drop_node(CTX, n.id, summary="tried genetic-algo; failed on X; skip unless Y")
    assert dropped.status == DROPPED
    # STILL queryable — a dropped node is never hard-deleted.
    again = store.get_node(CTX, n.id)
    assert again is not None and "failed on X" in again.recorded_context


def test_reopen_restores_open_and_clears_flag(store):
    n = _node(store, type=TN_TOPIC)
    store.set_reopen_flag(CTX, n.id, True)
    store.drop_node(CTX, n.id)
    reopened = store.reopen_node(CTX, n.id, reason="Y changed")
    assert reopened.status == OPEN and reopened.reopen_flag is False
    assert "Y changed" in reopened.recorded_context


def test_reopen_work_returns_to_a_reminder_state(store):
    w = _node(store, type=TN_WORK)
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED)
    store.close_node(CTX, w.id, evidence="tested")
    reopened = store.reopen_node(CTX, w.id)
    assert reopened.status == OPEN and reopened.testing_state == TS_TESTED_UNCLOSED
    assert reopened.is_reminder()                              # re-nags for sign-off


def test_reopen_via_same_topic_key_coexists(store):
    old = _node(store, type=TN_TOPIC, topic_key="dungeon-instancing")
    store.drop_node(CTX, old.id, summary="deferred for context-fatigue")
    fresh = _node(store, type=TN_TOPIC, topic_key="dungeon-instancing", title="B1b resumed")
    by_key = store.list_nodes(CTX, topic_key="dungeon-instancing")
    assert {n.id for n in by_key} == {old.id, fresh.id}       # same thread, two nodes
    assert store.get_node(CTX, fresh.id).status == OPEN
    assert store.get_node(CTX, old.id).status == DROPPED


def test_there_is_no_hard_delete(store):
    # The store exposes no delete — the retirement gate is structural.
    assert not hasattr(store, "delete_node")
    assert not hasattr(store, "remove_node")


# -- edges -------------------------------------------------------------------

def test_contains_edge_and_children(store):
    parent = _node(store, type=TN_TOPIC, title="parent")
    c1 = _node(store, type=TN_WORK, title="c1")
    c2 = _node(store, type=TN_WORK, title="c2")
    store.add_edge(CTX, TopicEdge(parent_id=parent.id, child_id=c1.id))
    store.add_edge(CTX, TopicEdge(parent_id=parent.id, child_id=c2.id))
    kids = {n.id for n in store.children(CTX, parent.id)}
    assert kids == {c1.id, c2.id}
    assert {e.child_id for e in store.edges_from(CTX, parent.id)} == {c1.id, c2.id}
    assert [e.parent_id for e in store.edges_to(CTX, c1.id)] == [parent.id]


def test_back_edge_requires_condition(store):
    a = _node(store, type=TN_TOPIC)
    b = _node(store, type=TN_TOPIC)
    with pytest.raises(ValueError):
        store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id=b.id, kind=EDGE_BACK))
    ok = store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id=b.id,
                                       kind=EDGE_BACK, condition="design X changes"))
    assert ok.kind == EDGE_BACK and ok.condition == "design X changes"
    # a back-edge is not a contains child.
    assert store.children(CTX, a.id) == []
    assert [n.id for n in store.children(CTX, a.id, kind=EDGE_BACK)] == [b.id]


def test_edge_rejects_self_loop_and_dangling_endpoints(store):
    a = _node(store, type=TN_TOPIC)
    with pytest.raises(ValueError):
        store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id=a.id))
    with pytest.raises(ValueError):
        store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id="topic_missing"))


def test_edge_is_idempotent_on_key(store):
    a = _node(store, type=TN_TOPIC)
    b = _node(store, type=TN_TOPIC)
    store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id=b.id))
    store.add_edge(CTX, TopicEdge(parent_id=a.id, child_id=b.id, condition="note"))
    assert len(store.edges_from(CTX, a.id)) == 1              # same (parent,child,kind)


# -- isolation ---------------------------------------------------------------

def test_nodes_are_tenant_project_scoped(store):
    n = _node(store, CTX, type=TN_TOPIC)
    assert store.get_node(OTHER_PROJECT, n.id) is None
    assert store.get_node(OTHER_TENANT, n.id) is None
    assert store.list_nodes(OTHER_PROJECT) == []


def test_edge_cannot_bridge_two_scopes(store):
    here = _node(store, CTX, type=TN_TOPIC)
    there = _node(store, OTHER_PROJECT, type=TN_TOPIC)
    # the child exists, but not in CTX's scope -> refused.
    with pytest.raises(ValueError):
        store.add_edge(CTX, TopicEdge(parent_id=here.id, child_id=there.id))


# -- bounding ----------------------------------------------------------------

def test_recorded_context_is_bounded(store):
    n = store.add_node(CTX, TopicNode(title="x", type=TN_TOPIC,
                                      recorded_context="y" * 5000))
    assert len(store.get_node(CTX, n.id).recorded_context) <= 2000


# -- defect fixes surfaced by the verification pass --------------------------

def test_reminders_not_dropped_when_budget_filled_by_non_reminders(store):
    """reminders() must filter to reminder states BEFORE the limit — else newer
    non-reminder work nodes fill the budget and real reminders vanish (the
    'loud, never silent' nag guarantee)."""
    r1 = _node(store, type=TN_WORK)
    r2 = _node(store, type=TN_WORK)
    store.advance_testing(CTX, r1.id, TS_BUILT_UNTESTED)   # real reminders
    store.advance_testing(CTX, r2.id, TS_BUILT_UNTESTED)
    for _ in range(5):                                     # newer, NOT reminders
        _node(store, type=TN_WORK)                          # testing_state = none
    got = {n.id for n in store.reminders(CTX, limit=5)}
    assert got == {r1.id, r2.id}                            # the 2 real ones, not []


def test_advance_testing_rejects_skips(store):
    """A work node cannot jump none -> tested_unclosed (skipping built) — the
    build->tested reminder ladder must not be bypassable."""
    w = _node(store, type=TN_WORK)
    with pytest.raises(ValueError):
        store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED)   # skips built_untested
    # the single-step path still works.
    store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED)
    # holding at the same state is allowed.
    assert store.advance_testing(CTX, w.id, TS_TESTED_UNCLOSED).testing_state == TS_TESTED_UNCLOSED


def test_finding_is_never_open(store):
    """A finding is informational: created non-open, absent from OPEN THREADS, and
    never blocks its parent's closure."""
    f = _node(store, type=TN_FINDING, title="camp mob level can't be clamped")
    assert store.get_node(CTX, f.id).status != OPEN         # never 'open'
    assert f.id not in {n.id for n in store.list_nodes(CTX, status=OPEN)}
    # a topic with only a finding child closes (the finding doesn't block it).
    parent = _node(store, type=TN_TOPIC, title="dungeon quality")
    store.add_edge(CTX, TopicEdge(parent_id=parent.id, child_id=f.id))
    assert store.open_children(CTX, parent.id) == []
    assert store.close_node(CTX, parent.id).status == CLOSED


def test_update_reads_back_the_state_it_wrote(store):
    """_update returns the node with exactly the fields this call set (atomic
    write+read-back under one lock)."""
    w = _node(store, type=TN_WORK)
    out = store.advance_testing(CTX, w.id, TS_BUILT_UNTESTED)
    assert out.testing_state == TS_BUILT_UNTESTED and out.id == w.id
