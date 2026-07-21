"""The raised-item ledger (#77): candidate selection, the closure classifier's
index-mapping + validation, the set_item_status writer, and the composed turn —
LLM stubbed, real SqliteStore, so it is deterministic and exercises the ledger for
real.
"""

from __future__ import annotations

import asyncio

import pytest

from saddle.context import Context
from saddle.item_ledger import (
    CANDIDATE_CAP,
    CLOSE_DONE,
    CLOSE_DROPPED,
    REOPEN,
    ItemClosure,
    _candidate_items,
    apply_item_closures,
    classify_item_closures,
    ledger_turn,
)
from saddle.models import DONE, DROPPED, OPEN, TODO_KINDS, Intake, Item

_CTX = Context(tenant="acme", project="game")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    from saddle.store import get_store, reset_store

    reset_store()
    s = get_store()
    yield s
    reset_store()


def _mk(store, *asks, ts0=1000.0):
    """Create task items with strictly increasing ts (newest last)."""
    items = [Item(kind="task", ask=a, ts=ts0 + i) for i, a in enumerate(asks)]
    store.save_intake(_CTX, Intake(raw_prompt="p", items=items))
    return {it.ask: it for it in store.list_items(_CTX, kinds=TODO_KINDS)}


# -- candidate selection -----------------------------------------------------

def test_candidate_items_orders_newest_first_and_caps(store):
    _mk(store, *[f"task {i}" for i in range(CANDIDATE_CAP + 5)])
    cands = _candidate_items(_CTX, store)
    assert len(cands) == CANDIDATE_CAP                       # capped
    assert cands[0].ts >= cands[-1].ts                       # newest first
    asks = {c.ask for c in cands}
    assert "task 0" not in asks                              # oldest dropped
    assert f"task {CANDIDATE_CAP + 4}" in asks               # newest kept


# -- the closure classifier (call_json stubbed) ------------------------------

@pytest.fixture
def _stub_call_json(monkeypatch):
    box: dict = {"payload": {"closures": []}}

    async def _fake(caller, sys, prompt, **kw):
        return box["payload"]

    monkeypatch.setattr("saddle.item_ledger.call_json", _fake)
    monkeypatch.setattr("saddle.item_ledger.build_callers",
                        lambda ctx: {"default": object()}, raising=False)
    return box


def _cands(*asks):
    return [Item(kind="task", ask=a, id=f"item_{i}", status=OPEN)
            for i, a in enumerate(asks)]


def test_classify_maps_index_to_item_and_disposition(_stub_call_json):
    _stub_call_json["payload"] = {"closures": [
        {"n": 2, "disposition": "done", "confidence": 0.9},
        {"n": 1, "disposition": "dropped", "confidence": 0.8},
    ]}
    cands = _cands("gate the council", "wire the ledger", "build the brain")
    out = asyncio.run(classify_item_closures(cands, "happy with 2, forget 1", _CTX,
                                             caller=object()))
    got = {c.item.ask: c.disposition for c in out}
    assert got == {"wire the ledger": CLOSE_DONE, "gate the council": CLOSE_DROPPED}


def test_classify_drops_bad_index_and_unknown_disposition(_stub_call_json):
    _stub_call_json["payload"] = {"closures": [
        {"n": 99, "disposition": "done"},          # out of range
        {"n": 1, "disposition": "frobnicate"},     # not a disposition
        {"n": 0, "disposition": "done"},           # index 0 invalid (1-based)
    ]}
    cands = _cands("only one")
    out = asyncio.run(classify_item_closures(cands, "x", _CTX, caller=object()))
    assert out == []                                # nothing hallucinated through


def test_classify_dedupes_repeated_item(_stub_call_json):
    _stub_call_json["payload"] = {"closures": [
        {"n": 1, "disposition": "done"},
        {"n": 1, "disposition": "dropped"},         # same item again -> ignored
    ]}
    out = asyncio.run(classify_item_closures(_cands("a"), "x", _CTX, caller=object()))
    assert len(out) == 1 and out[0].disposition == CLOSE_DONE


def test_classify_empty_when_no_candidates(_stub_call_json):
    assert asyncio.run(classify_item_closures([], "anything", _CTX, caller=object())) == []


# -- the writer --------------------------------------------------------------

def test_apply_sets_status_and_skips_noops(store):
    items = _mk(store, "gate", "wire")
    closures = [
        ItemClosure(item=items["gate"], disposition=CLOSE_DONE),
        ItemClosure(item=items["wire"], disposition=CLOSE_DROPPED),
    ]
    applied = apply_item_closures(_CTX, closures, store)
    assert len(applied) == 2
    by_ask = {i.ask: i.status for i in store.list_items(_CTX, kinds=TODO_KINDS)}
    assert by_ask["gate"] == DONE and by_ask["wire"] == DROPPED
    # todos() now returns only the still-open backlog
    assert {i.ask for i in store.todos(_CTX)} == set()


def test_apply_skips_already_at_target(store):
    items = _mk(store, "gate")
    store.set_item_status(_CTX, items["gate"].id, DONE)      # already done
    items["gate"].status = DONE
    applied = apply_item_closures(
        _CTX, [ItemClosure(item=items["gate"], disposition=CLOSE_DONE)], store)
    assert applied == []                                     # no-op, not heralded


def test_apply_reopen_returns_item_to_open(store):
    items = _mk(store, "gate")
    store.set_item_status(_CTX, items["gate"].id, DONE)
    items["gate"].status = DONE
    applied = apply_item_closures(
        _CTX, [ItemClosure(item=items["gate"], disposition=REOPEN)], store)
    assert len(applied) == 1
    assert store.list_items(_CTX, kinds=TODO_KINDS)[0].status == OPEN


# -- the composed turn (classifier stubbed, real store) ----------------------

@pytest.fixture
def _stub_classify(monkeypatch):
    box: dict = {"closures": []}

    async def _fake(candidates, prompt, ctx=None, **kw):
        # candidates are the REAL items; map scripted (ask, disposition) onto them
        by_ask = {c.ask: c for c in candidates}
        return [ItemClosure(item=by_ask[a], disposition=d)
                for a, d in box["closures"] if a in by_ask]

    monkeypatch.setattr("saddle.item_ledger.classify_item_closures", _fake)
    return box


def _turn(prompt, store):
    return asyncio.run(ledger_turn(prompt, ctx=_CTX, store=store))


def test_ledger_turn_closes_and_heralds(_stub_classify, store):
    _mk(store, "gate the council", "wire the ledger")
    _stub_classify["closures"] = [("gate the council", CLOSE_DONE)]
    out = _turn("the council gate is done, happy with it", store)
    assert len(out.closed) == 1
    assert "gate the council"[:10] in out.herald and "done" in out.herald
    by_ask = {i.ask: i.status for i in store.list_items(_CTX, kinds=TODO_KINDS)}
    assert by_ask["gate the council"] == DONE
    assert by_ask["wire the ledger"] == OPEN                 # untouched


def test_ledger_turn_no_closure_leaves_backlog_open(_stub_classify, store):
    _mk(store, "gate the council")
    _stub_classify["closures"] = []                          # user didn't close anything
    out = _turn("how does the council gate work?", store)
    assert out.closed == [] and out.herald == ""
    assert store.list_items(_CTX, kinds=TODO_KINDS)[0].status == OPEN  # nothing self-closed


def test_ledger_turn_no_candidates_is_a_silent_noop(_stub_classify, store):
    out = _turn("anything", store)                           # empty backlog
    assert out.closed == [] and out.herald == ""
