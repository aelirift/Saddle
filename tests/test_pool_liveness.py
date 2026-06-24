"""ProcessPool admission liveness — the memory gate WAITS, but never forever.

The admission memory ceiling can only pause new acquires; it cannot reclaim
in-flight calls. So if something else on the host pins RAM above the ceiling and
never frees it, the old wait loop spun forever with only a periodic warning — a
silent hang, the exact "I never see saddle return" failure. These pin the bound:
sustained pressure raises a typed :class:`supervise.DeadlineExceeded`, and the
hard-cap slot is handed back (not leaked) on that raise so the pool doesn't
permanently shrink. Memory is fully mocked — no real host pressure, no network.
"""
from __future__ import annotations

import asyncio

import pytest

from saddle import supervise
import saddle.llm.pool as pool


def test_acquire_bounds_sustained_memory_pressure_and_returns_the_slot(monkeypatch):
    # Tiny budget + poll so the test trips in a fraction of a second instead of
    # the 600s production ceiling.
    monkeypatch.setattr(pool, "_MEM_WAIT_DEADLINE_S", 0.3)
    monkeypatch.setattr(pool, "_MEM_POLL_S", 0.05)

    async def _scenario():
        p = pool.ProcessPool()

        # Host pegged ABOVE the ceiling, never recovering.
        monkeypatch.setattr(pool, "_memory_usage_percent", lambda: 99.0)
        with pytest.raises(supervise.DeadlineExceeded):
            await p.acquire("probe")

        # The hard-cap slot must have been returned, not leaked: with memory now
        # healthy a fresh acquire resolves immediately. A leaked slot would make
        # this block until wait_for fires TimeoutError.
        monkeypatch.setattr(pool, "_memory_usage_percent", lambda: 10.0)
        start = await asyncio.wait_for(p.acquire("after"), timeout=1.0)
        assert start > 0
        p.release("after", start)
        # Full capacity restored — no permanent shrink from the failed acquire.
        assert p._semaphore._value == pool._MAX_PROCS

    asyncio.run(_scenario())


def test_acquire_admits_promptly_when_memory_is_healthy(monkeypatch):
    # The bound must not change the happy path: under the ceiling, admission is
    # immediate and returns a real start time.
    monkeypatch.setattr(pool, "_memory_usage_percent", lambda: 12.0)

    async def _scenario():
        p = pool.ProcessPool()
        start = await asyncio.wait_for(p.acquire("ok"), timeout=1.0)
        assert start > 0
        p.release("ok", start)

    asyncio.run(_scenario())


def test_zero_deadline_is_an_explicit_unbounded_opt_out(monkeypatch):
    # _MEM_WAIT_DEADLINE_S <= 0 means "wait forever" (opt-out). Prove the loop
    # does NOT raise on its own: flip memory healthy after a couple of polls and
    # the acquire still succeeds — i.e. the deadline branch is what would have
    # raised, and disabling it restores the patient wait.
    monkeypatch.setattr(pool, "_MEM_WAIT_DEADLINE_S", 0.0)
    monkeypatch.setattr(pool, "_MEM_POLL_S", 0.02)

    state = {"polls": 0}

    def _usage() -> float:
        state["polls"] += 1
        return 99.0 if state["polls"] <= 3 else 10.0   # pressured, then clears

    async def _scenario():
        p = pool.ProcessPool()
        monkeypatch.setattr(pool, "_memory_usage_percent", _usage)
        start = await asyncio.wait_for(p.acquire("patient"), timeout=2.0)
        assert start > 0
        p.release("patient", start)

    asyncio.run(_scenario())
