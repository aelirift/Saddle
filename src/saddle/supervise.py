"""Liveness discipline for work saddle drives but does not control the clock of.

WHY THIS EXISTS
---------------
saddle hands work to processes whose timing it does not own: the local ``claude``
CLI subprocess (the coder + the lead provider), an HTTP LLM stream, a future
build / godot run. Any of them can STALL rather than fail — a subprocess that
never execs (binary off PATH), an auth handshake that never completes, a stream
that goes silent mid-turn. A controller that simply ``await``\\ s such a thing
blocks FOREVER; the only thing that frees it is an external ``timeout`` / SIGTERM,
which looks identical to a crash and teaches the harness nothing. That is exactly
the 150s converge hang: ``ChatSession.connect()`` waited on a ``claude`` that was
never found, with no deadline to notice it would never arrive.

THE RULE
--------
Every ``await`` on something saddle does not control gets a DEADLINE, and every
STREAM gets a HEARTBEAT (an idle watchdog) so "actively working" — tokens or
tool-calls still arriving — is told apart from "wedged" — total silence. Crossing
either bound raises a TYPED error (:class:`DeadlineExceeded` / :class:`Stalled`)
the caller can ACT on: fail the provider over, mark the coder unavailable, halt
the round. It never hangs, and it never silently swallows the difference between
"slow" and "stuck".

This is the single home for that policy so it is one import, not re-rolled
``asyncio.wait_for`` calls scattered across every caller. Both errors subclass the
builtin :class:`TimeoutError`, so a caller that only wants "did this time out?"
can catch the base type; one that wants to distinguish a dead connection from a
wedged stream catches the specific subclass.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from typing import AsyncIterator, Awaitable, TypeVar

T = TypeVar("T")

# Ambient budget — the absolute monotonic deadline of the innermost ``bounded()``
# currently driving this task's await chain. A consumer that fans work across
# ALTERNATIVES (e.g. the provider fail-over chain) reads the REMAINING budget to
# apportion per-alternative deadlines from the time the stage actually has,
# instead of guessing with a flat constant sized for the small-call happy path
# (the flat 20s cut every provider on long-prompt hook stages -> "All callers
# failed"). Innermost bounded() wins; reset restores the enclosing budget.
_BUDGET_ENDS_AT: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "saddle_budget_ends_at", default=None
)


def budget_remaining() -> float | None:
    """Seconds left in the innermost active :func:`bounded` budget, or ``None``
    when no bounded deadline is ambient. Never negative — an exhausted budget
    reads ``0.0`` (callers treat it as "fail fast"; the outer bound is about to
    fire anyway)."""
    ends = _BUDGET_ENDS_AT.get()
    if ends is None:
        return None
    return max(0.0, ends - time.monotonic())


class DeadlineExceeded(TimeoutError):
    """A bounded await did not complete within its deadline — treat as wedged."""


class Stalled(TimeoutError):
    """A stream produced nothing for longer than its idle heartbeat.

    Distinct from :class:`DeadlineExceeded`: the work may not be over-budget
    overall, it has simply gone *silent* — a healthy stream resets the heartbeat
    on every item, so silence past the idle window means stuck, not merely slow.
    """


async def bounded(awaitable: Awaitable[T], *, seconds: float, what: str) -> T:
    """Await ``awaitable``, raising :class:`DeadlineExceeded` if it runs past
    ``seconds``.

    ``seconds <= 0`` disables the deadline (await unbounded) — an EXPLICIT
    opt-out a caller has to choose, never the silent default that caused the hang.
    """
    if seconds <= 0:
        return await awaitable
    # Publish the budget BEFORE wait_for wraps the awaitable in a Task: the task
    # snapshots the current context at creation, so consumers down the await
    # chain (budget_remaining) see this deadline. Reset restores any enclosing
    # bounded()'s budget on the way out.
    token = _BUDGET_ENDS_AT.set(time.monotonic() + seconds)
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise DeadlineExceeded(
            f"{what}: no completion within {seconds:.0f}s — treating as wedged"
        ) from exc
    finally:
        _BUDGET_ENDS_AT.reset(token)


async def heartbeat(
    stream: AsyncIterator[T],
    *,
    idle_seconds: float,
    max_seconds: float,
    what: str,
) -> AsyncIterator[T]:
    """Re-yield ``stream`` while watching two clocks:

      idle  — reset on EVERY item; if nothing arrives for ``idle_seconds`` the
              stream is wedged -> :class:`Stalled`.
      total — a wall-clock cap across the whole stream -> :class:`DeadlineExceeded`.

    A non-positive bound disables that clock. Items pass through untouched and in
    order, so wrapping a stream is transparent to its consumer — the only added
    behaviour is that a wedged or runaway stream RAISES instead of hanging.
    """
    it = stream.__aiter__()
    start = time.monotonic()
    while True:
        if max_seconds > 0 and (time.monotonic() - start) >= max_seconds:
            raise DeadlineExceeded(
                f"{what}: exceeded {max_seconds:.0f}s wall-clock — halting"
            )
        timeout: float | None = idle_seconds if idle_seconds > 0 else None
        try:
            if timeout is None:
                item = await it.__anext__()
            else:
                item = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            # A whole turn that blows the wall-clock cap surfaces as that, not as
            # a momentary idle — pick the message by which bound we actually hit.
            if max_seconds > 0 and (time.monotonic() - start) >= max_seconds:
                raise DeadlineExceeded(
                    f"{what}: exceeded {max_seconds:.0f}s wall-clock — halting"
                ) from exc
            raise Stalled(
                f"{what}: no output for {idle_seconds:.0f}s — treating as wedged"
            ) from exc
        yield item


async def poll_until(
    check,
    *,
    interval_seconds: float,
    timeout_seconds: float,
    what: str,
):
    """Poll ``check`` every ``interval_seconds`` until it returns truthy, then
    return that value; raise :class:`DeadlineExceeded` if ``timeout_seconds``
    passes first.

    The detect-when-done primitive for work that exposes a *marker* rather than a
    stream — a file that appears, a subprocess that exits, a status that flips.
    ``check`` may be sync or async; its truthy return is handed back so the
    caller gets the resolved value, not just "it happened". This is the polite
    inverse of a blind ``sleep``: it wakes the moment the condition holds and
    bails the moment the budget is spent, never in between.
    """
    start = time.monotonic()
    while True:
        result = check()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        if timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds:
            raise DeadlineExceeded(
                f"{what}: condition not met within {timeout_seconds:.0f}s"
            )
        await asyncio.sleep(max(0.0, interval_seconds))
