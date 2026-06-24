"""saddle.supervise — the liveness discipline: deadlines + heartbeat + poll.

Pins the contract the 150s converge hang taught us: an ``await`` on something
saddle does not control must time out (typed) rather than block forever; a stream
must tell "actively working" (every item resets the heartbeat) apart from
"wedged" (silence past the idle window); and a marker-based wait must wake on the
condition and bail on the budget — never sleep blindly in between.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from saddle.supervise import (
    DeadlineExceeded,
    Stalled,
    bounded,
    heartbeat,
    poll_until,
)


def _run(coro):
    return asyncio.run(coro)


async def _collect(aiter):
    return [x async for x in aiter]


async def _agen(items, *, gap: float = 0.0, tail: float = 0.0):
    for x in items:
        if gap:
            await asyncio.sleep(gap)
        yield x
    if tail:
        await asyncio.sleep(tail)


# --- bounded -----------------------------------------------------------------
def test_bounded_returns_value_under_budget():
    async def fast():
        return 42

    assert _run(bounded(fast(), seconds=5, what="fast")) == 42


def test_bounded_raises_deadline_and_does_not_wait_out_the_full_awaitable():
    async def main():
        t = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            await bounded(asyncio.sleep(5), seconds=0.2, what="slow")
        return time.monotonic() - t

    assert _run(main()) < 1.0  # fired at ~0.2s, not after the 5s sleep


def test_bounded_zero_seconds_is_an_explicit_unbounded_opt_out():
    async def fast():
        return "ok"

    assert _run(bounded(fast(), seconds=0, what="unbounded")) == "ok"


# --- heartbeat ---------------------------------------------------------------
def test_heartbeat_passes_a_healthy_stream_through_in_order():
    got = _run(_collect(heartbeat(
        _agen([1, 2, 3], gap=0.02), idle_seconds=1.0, max_seconds=10, what="h",
    )))
    assert got == [1, 2, 3]


def test_heartbeat_raises_stalled_on_idle_silence():
    async def main():
        t = time.monotonic()
        with pytest.raises(Stalled):
            async for _ in heartbeat(
                _agen([1], tail=5.0), idle_seconds=0.3, max_seconds=30, what="h",
            ):
                pass
        return time.monotonic() - t

    assert _run(main()) < 1.5  # caught the silence promptly, didn't wait out tail


def test_heartbeat_raises_deadline_on_absolute_wall_clock_cap():
    async def main():
        with pytest.raises(DeadlineExceeded):
            async for _ in heartbeat(
                _agen(range(1000), gap=0.02), idle_seconds=5.0, max_seconds=0.3,
                what="h",
            ):
                pass

    _run(main())


# --- poll_until --------------------------------------------------------------
def test_poll_until_returns_the_resolved_value_when_the_condition_flips():
    async def main():
        flip = {"v": False}

        async def arm():
            await asyncio.sleep(0.3)
            flip["v"] = True

        asyncio.create_task(arm())
        return await poll_until(
            lambda: flip["v"], interval_seconds=0.02, timeout_seconds=5, what="p",
        )

    assert _run(main()) is True


def test_poll_until_times_out_when_the_condition_never_holds():
    async def main():
        with pytest.raises(DeadlineExceeded):
            await poll_until(
                lambda: False, interval_seconds=0.02, timeout_seconds=0.3, what="p",
            )

    _run(main())


def test_poll_until_awaits_an_async_check():
    async def main():
        async def check():
            return "done"

        return await poll_until(
            check, interval_seconds=0.02, timeout_seconds=2, what="p",
        )

    assert _run(main()) == "done"
