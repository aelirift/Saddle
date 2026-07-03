"""Budget-aware provider deadlines — the fix for the flat-20s "All callers
failed" class.

Contract under test:

* ``supervise.bounded`` publishes its deadline as an ambient budget that
  ``supervise.budget_remaining`` reads from anywhere down the await chain, and
  restores the enclosing budget on exit (nesting: innermost wins).
* ``callers._effective_provider_deadline`` resolves, in order: explicit env
  override verbatim (including 0 = unbounded) → equal share of the remaining
  ambient budget across untried providers → the flat happy-path default.
"""

from __future__ import annotations

import asyncio

import pytest

from saddle import supervise
from saddle.llm import callers as callers_mod


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", raising=False)


def test_no_ambient_budget_reads_none():
    assert supervise.budget_remaining() is None


def test_bounded_publishes_and_restores_budget():
    async def probe() -> float | None:
        return supervise.budget_remaining()

    async def scenario():
        outer = await supervise.bounded(probe(), seconds=30.0, what="outer")
        after = supervise.budget_remaining()
        return outer, after

    outer, after = asyncio.run(scenario())
    assert outer is not None and 0.0 < outer <= 30.0
    assert after is None  # reset on exit


def test_nested_bounded_innermost_wins():
    async def inner_probe() -> float | None:
        return supervise.budget_remaining()

    async def mid():
        # Inner budget (5s) is tighter than outer (60s): the probe must see it.
        seen = await supervise.bounded(inner_probe(), seconds=5.0, what="inner")
        # After the inner bounded exits, the OUTER budget is ambient again.
        restored = supervise.budget_remaining()
        return seen, restored

    seen, restored = asyncio.run(supervise.bounded(mid(), seconds=60.0, what="outer"))
    assert seen is not None and seen <= 5.0
    assert restored is not None and 5.0 < restored <= 60.0


def test_unbounded_optout_publishes_nothing():
    async def probe() -> float | None:
        return supervise.budget_remaining()

    assert asyncio.run(supervise.bounded(probe(), seconds=0, what="optout")) is None


def test_effective_deadline_env_override_wins(monkeypatch):
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "7")

    async def scenario():
        # Ambient budget present, but the explicit override must win verbatim.
        async def probe() -> float:
            return callers_mod._effective_provider_deadline(2)

        return await supervise.bounded(probe(), seconds=100.0, what="env")

    assert asyncio.run(scenario()) == 7.0


def test_effective_deadline_shares_ambient_budget():
    async def probe() -> float:
        return callers_mod._effective_provider_deadline(2)

    share = asyncio.run(supervise.bounded(probe(), seconds=60.0, what="share"))
    # Two untried providers split the ~60s budget; loop overhead shaves a hair.
    assert 25.0 < share <= 30.0


def test_effective_deadline_last_provider_gets_full_remainder():
    async def probe() -> float:
        return callers_mod._effective_provider_deadline(1)

    share = asyncio.run(supervise.bounded(probe(), seconds=40.0, what="last"))
    assert 35.0 < share <= 40.0


def test_effective_deadline_default_without_budget():
    assert callers_mod._effective_provider_deadline(3) == 20.0


def test_fallback_long_call_survives_under_budget():
    """A provider slower than its would-be flat share must SUCCEED when the
    ambient budget affords it — the exact long-prompt hook-path failure."""

    class SlowOK:
        async def __call__(self, system, prompt, *, json_mode=False, label=""):
            await asyncio.sleep(0.3)
            return "ok"

    fb = callers_mod.FallbackCaller([SlowOK()], failure_threshold=0)

    async def scenario():
        # Budget 1.0s, one provider -> share ~1.0s > 0.3s sleep. With a flat
        # deadline of 0.05s (env) this same call would die; prove budget wins
        # only when env is absent.
        return await supervise.bounded(
            fb("s", "p"), seconds=1.0, what="slow-ok",
        )

    assert asyncio.run(scenario()) == "ok"


def test_fallback_flat_env_still_cuts(monkeypatch):
    monkeypatch.setenv("RAYXI_LLM_PROVIDER_DEADLINE_SECONDS", "0.05")

    class SlowOK:
        async def __call__(self, system, prompt, *, json_mode=False, label=""):
            await asyncio.sleep(0.3)
            return "ok"

    fb = callers_mod.FallbackCaller([SlowOK()], failure_threshold=0)

    async def scenario():
        return await supervise.bounded(fb("s", "p"), seconds=1.0, what="cut")

    with pytest.raises(RuntimeError, match="All callers failed"):
        asyncio.run(scenario())
