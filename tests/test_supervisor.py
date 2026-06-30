"""supervisor — the staged supervisory runner (gap-2's five live drift checks).

These tests pin the ONE discipline ``run_stage`` exists to enforce: a stage
either RAN (and stayed silent, or surfaced a finding) or COULD-NOT-RUN (and said
so loudly). The third case — the classified ALERT — is the whole point: it is the
generalization of the intake fail-loud fix, and the property the old fail-open
violated. So the tests cover, against an in-memory outbox that captures every
emitted bubble:

* **silent success** — a stage that ran and found nothing emits NO bubble and is
  still ``ok`` (ran ≠ found something),
* **a notice finding** — sections at the default level emit one ``notice`` bubble
  tagged with the stage,
* **an alert finding** — a stage that returns ``level=alert`` (a caught drift)
  bubbles at alert,
* **a failure** — a stage that RAISES is caught, classified, and bubbled as an
  ALERT whose text names what saddle did NOT verify; ``ok`` is False and the
  classified category rides on the result,
* **control-flow is not a finding** — ``KeyboardInterrupt`` /
  ``CancelledError`` propagate out of ``run_stage`` (a cancelled turn is not a
  stage verdict),
* **the timeout taxonomy** — ``DeadlineExceeded`` / ``Stalled`` (the recurring
  intake hang) classify as ``timeout``, so every surface tells a wall-clock
  deadline apart from a provider outage,
* **the shared render** — ``render_sections`` carries the ``saddle [key]`` header,
  drops empty sections, and is never a header alone, and
* **the combined agent channel** — ``agent_context`` joins every stage's sections
  into one blob (the bubbles already went out per-stage).

``SADDLE_HOME`` is parked under tmp and the bubble-store singleton is swapped for
an :class:`InMemoryBubbleStore` so each assertion reads exactly what the stage
emitted, with no durable db or JSONL mirror touched.
"""
from __future__ import annotations

import asyncio

import pytest

from saddle import supervise
from saddle.bubble import InMemoryBubbleStore, set_bubble_store
from saddle.context import Context
from saddle.models import BUBBLE_ALERT, BUBBLE_NOTICE
from saddle.supervisor import (
    StageOutcome,
    StageResult,
    agent_context,
    classify_failure,
    render_sections,
    run_bounded,
    run_stage,
    system_message,
)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Park the saddle data dir under tmp so no durable db/mirror is touched."""
    monkeypatch.setenv("SADDLE_HOME", str(tmp_path))
    yield


@pytest.fixture
def store():
    """Swap the process bubble outbox for an in-memory one and hand it back so a
    test can read exactly what a stage emitted. The shared conftest resets the
    singleton around every test, so this never leaks past the test."""
    s = InMemoryBubbleStore()
    set_bubble_store(s)
    return s


def _ctx(tenant: str = "aeli", project: str = "rayxiv4") -> Context:
    return Context(tenant=tenant, project=project)


# --- silent success: ran, found nothing -> no bubble, still ok --------------

def test_stage_that_finds_nothing_stays_silent_but_ok(store):
    """The distinction the whole framework rests on: ``ok`` means RAN, not
    FOUND-SOMETHING. A stage returning None (or empty sections) emits no bubble
    yet is ``ok`` — it genuinely ran and nothing drifted."""
    res = run_stage(_ctx(), "intake", lambda: None)
    assert res.ok is True
    assert res.spoke is False
    assert res.bubble is None
    assert res.sections == []
    assert store.recent(_ctx()) == []


def test_outcome_with_only_blank_sections_is_silent(store):
    """Whitespace-only sections are not a finding — they're dropped, so a stage
    that 'spoke' only blanks is still a silent success (never a header-alone)."""
    res = run_stage(_ctx(), "code", lambda: StageOutcome(sections=["   ", ""]))
    assert res.ok is True
    assert res.spoke is False
    assert store.recent(_ctx()) == []


# --- a notice finding: ran, surfaced something at the routine level ----------

def test_notice_finding_emits_one_tagged_bubble(store):
    """A stage that ran and has something to say emits exactly one bubble, at the
    outcome's level, tagged with the stage so a client can filter by stage."""
    res = run_stage(
        _ctx(),
        "intent",
        lambda: StageOutcome(sections=["this prompt reopens a settled design"]),
        session="sess-1",
    )
    assert res.ok is True
    assert res.spoke is True
    assert res.level == BUBBLE_NOTICE
    bubbles = store.recent(_ctx())
    assert len(bubbles) == 1
    (b,) = bubbles
    assert b.stage == "intent"
    assert b.level == BUBBLE_NOTICE
    assert b.session == "sess-1"
    assert "this prompt reopens a settled design" in b.text
    assert f"saddle [{_ctx().key}]" in b.text  # carries the shared header


def test_outcome_meta_and_title_ride_on_the_bubble(store):
    """A richer client wants the structured payload — meta + title flow through
    run_stage onto the emitted bubble verbatim."""
    res = run_stage(
        _ctx(),
        "design",
        lambda: StageOutcome(
            sections=["band-aid: swallow-and-log instead of structural fix"],
            level=BUBBLE_ALERT,
            title="design review",
            meta={"findings": ["swallow"], "intake_id": "intk_1"},
        ),
    )
    (b,) = store.recent(_ctx())
    assert res.level == BUBBLE_ALERT
    assert b.title == "design review"
    assert b.meta == {"findings": ["swallow"], "intake_id": "intk_1"}


# --- an alert finding: a caught drift renders loudly ------------------------

def test_alert_level_finding_bubbles_at_alert(store):
    """A caught drift / band-aid sets ``level=alert``; the bubble and the result
    both carry alert, so the human sees it loudly, not buried in a notice."""
    res = run_stage(
        _ctx(),
        "design",
        lambda: StageOutcome(
            sections=["the diff added a try/except swallow"], level=BUBBLE_ALERT
        ),
    )
    assert res.ok is True  # it RAN — the finding is the point, not a failure
    assert res.level == BUBBLE_ALERT
    (b,) = store.recent(_ctx())
    assert b.level == BUBBLE_ALERT


# --- a failure: could-not-run -> classified, LOUD ALERT ---------------------

def test_raising_stage_is_caught_classified_and_alerted(store):
    """The anti-fail-open: a stage that RAISES never silently no-ops. It is
    caught, classified, and bubbled as an ALERT naming what saddle did NOT verify
    — and the result is ``ok=False`` with the classified category."""
    def boom() -> StageOutcome:
        raise supervise.DeadlineExceeded("decompose hung")

    res = run_stage(
        _ctx(), "intake", boom, what="the prompt decomposition"
    )
    assert res.ok is False
    assert res.spoke is True
    assert res.level == BUBBLE_ALERT
    assert res.failure == "timeout"
    assert res.remedy  # a one-line hint is always present
    (b,) = store.recent(_ctx())
    assert b.level == BUBBLE_ALERT
    assert b.stage == "intake"
    assert "could not run" in b.text
    assert "did NOT verify the prompt decomposition" in b.text
    assert "timeout" in b.text
    assert b.meta.get("failure") == "timeout"
    assert b.meta.get("what") == "the prompt decomposition"


def test_failure_subject_defaults_to_stage_name(store):
    """Without an explicit ``what``, the failure message still names the stage so
    the gap is never anonymous."""
    def boom() -> StageOutcome:
        raise ValueError("provider returned garbage")

    res = run_stage(_ctx(), "lesson", boom)
    assert res.ok is False
    assert res.failure == "other"
    (b,) = store.recent(_ctx())
    assert "did NOT verify stage lesson" in b.text


def test_control_flow_exceptions_propagate(store):
    """A cancelled / interrupted turn is NOT a stage finding — BaseException
    control-flow propagates out of run_stage instead of being bubbled."""
    def cancel() -> StageOutcome:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_stage(_ctx(), "intake", cancel)
    assert store.recent(_ctx()) == []  # nothing was swallowed into a bubble


def test_cancelled_error_propagates(store):
    """asyncio.CancelledError is BaseException-derived control flow — it must
    propagate, not classify as a timeout finding."""
    def cancel() -> StageOutcome:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_stage(_ctx(), "code", cancel)
    assert store.recent(_ctx()) == []


# --- the failure taxonomy: the recurring intake hang is a timeout -----------

@pytest.mark.parametrize(
    "exc",
    [
        supervise.DeadlineExceeded("wall-clock"),
        supervise.Stalled("no heartbeat"),
        TimeoutError("plain"),
        asyncio.TimeoutError(),
    ],
)
def test_liveness_errors_classify_as_timeout(exc):
    """Both supervise liveness errors subclass TimeoutError and so must classify
    as ``timeout`` — the recurring intake hang, told apart from a provider
    outage by every surface that reports it."""
    category, remedy = classify_failure(exc)
    assert category == "timeout"
    assert remedy


def test_non_timeout_classifies_as_other():
    """A plain value error is not a timeout — it stays uncategorized rather than
    being mislabeled, so the taxonomy means something."""
    category, _ = classify_failure(ValueError("bad"))
    assert category == "other"


def test_json_decode_error_classifies_as_parse_error():
    """A staged call_json reply that wasn't valid JSON raises json.JSONDecodeError;
    a stage that hits it must classify as ``parse_error`` (the JSON-contract gap),
    not the opaque ``other`` — so the surfaced ALERT carries the right remedy. It is
    matched by TYPE (JSONDecodeError subclasses ValueError), so a PLAIN ValueError
    is still ``other`` (above) and the taxonomy stays meaningful."""
    import json

    category, remedy = classify_failure(
        json.JSONDecodeError("Expecting ',' delimiter", "{bad json}", 5)
    )
    assert category == "parse_error"
    assert remedy


# --- shared presentation: the one render every channel uses -----------------

def test_render_sections_carries_header_and_drops_blanks():
    """The shared block-render: a ``saddle [key]`` header over non-blank
    sections, blanks dropped, so a render is never a header alone."""
    out = render_sections(_ctx(), ["first", "  ", "", "second"])
    assert out.startswith(f"━━ saddle [{_ctx().key}] ━━")
    assert "first" in out and "second" in out
    # the two blanks contributed nothing
    assert out.count("\n\n") == 1


def test_render_sections_all_blank_is_empty_string():
    """All-blank in means empty out — a hook can then skip the stdout emit
    entirely rather than print a lone header."""
    assert render_sections(_ctx(), ["", "   ", "\n"]) == ""


# --- the combined agent channel: one blob over every stage ------------------

def test_agent_context_joins_every_stage(store):
    """The agent-context channel is COMBINED: it joins all stages' sections into
    one additionalContext blob (the bubbles already went per-stage)."""
    r1 = run_stage(_ctx(), "intake", lambda: StageOutcome(sections=["A ran"]))
    r2 = run_stage(_ctx(), "intent", lambda: StageOutcome(sections=["B drift"]))
    blob = agent_context(_ctx(), [r1, r2])
    assert "A ran" in blob
    assert "B drift" in blob
    assert blob.startswith(f"━━ saddle [{_ctx().key}] ━━")


def test_agent_context_empty_when_all_silent(store):
    """When no stage had anything to say, the combined blob is empty so a hook
    can skip the stdout emit."""
    r1 = run_stage(_ctx(), "intake", lambda: None)
    r2 = run_stage(_ctx(), "code", lambda: StageOutcome(sections=["  "]))
    assert agent_context(_ctx(), [r1, r2]) == ""


# --- the user-screen channel: only what SPOKE, capped -----------------------

def test_system_message_only_includes_stages_that_spoke(store):
    """The on-screen herald is NARROWER than the agent channel: a silent-success
    stage contributes nothing; only a stage that emitted a bubble (found drift or
    failed) reaches the human's screen — so a clean run never adds noise."""
    silent = run_stage(_ctx(), "intake", lambda: None)
    spoke = run_stage(
        _ctx(), "design",
        lambda: StageOutcome(sections=["band-aid: swallowed the error"], level=BUBBLE_ALERT),
    )
    sm = system_message(_ctx(), [silent, spoke])
    assert "band-aid: swallowed the error" in sm
    assert sm.startswith(f"━━ saddle [{_ctx().key}] ━━")


def test_system_message_empty_when_every_stage_silent(store):
    """A turn where nothing drifted returns ``""`` so the caller skips the
    systemMessage field entirely — saddle stays quiet on screen, never a header
    alone announcing it had nothing to say."""
    r1 = run_stage(_ctx(), "intake", lambda: None)
    r2 = run_stage(_ctx(), "code", lambda: StageOutcome(sections=["   "]))
    assert system_message(_ctx(), [r1, r2]) == ""


def test_system_message_includes_a_could_not_run_failure(store):
    """A stage that COULD-NOT-RUN spoke (its own ALERT) — the human must SEE that
    saddle didn't verify something this turn, not just the agent. The classified
    failure text reaches the screen."""
    def boom() -> StageOutcome:
        raise supervise.DeadlineExceeded("decompose hung")

    failed = run_stage(_ctx(), "intake", boom, what="the prompt decomposition")
    sm = system_message(_ctx(), [failed])
    assert "could not run" in sm
    assert "did NOT verify the prompt decomposition" in sm


def test_system_message_caps_a_flooding_section(store):
    """A long design alert can't flood the on-screen surface: the herald is capped
    and points to the outbox for the full text (which the durable bubble kept)."""
    huge = run_stage(
        _ctx(), "design",
        lambda: StageOutcome(sections=["X" * 5000], level=BUBBLE_ALERT),
    )
    sm = system_message(_ctx(), [huge], max_chars=200)
    assert len(sm) <= 200 + len("\n… (full detail in saddle outbox)")
    assert sm.endswith("full detail in saddle outbox)")


# --- the async -> sync deadline bridge --------------------------------------

def test_run_bounded_drives_a_coro_to_completion():
    """The sync stage protocol drives an async step via run_bounded; a step that
    finishes under the deadline returns its value."""
    async def work() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_bounded(work(), seconds=5, what="a quick step") == 7


def test_run_bounded_deadline_raises_classifiable_timeout():
    """A step that exceeds the deadline raises a DeadlineExceeded that
    classify_failure turns into ``timeout`` — so a wedged decompose becomes a
    loud ALERT, not an unbounded hang."""
    async def hang() -> int:
        await asyncio.sleep(5)
        return 1

    with pytest.raises(supervise.DeadlineExceeded) as ei:
        run_bounded(hang(), seconds=0.05, what="a wedged step")
    assert classify_failure(ei.value)[0] == "timeout"


# --- the result contract ----------------------------------------------------

def test_stage_result_spoke_reflects_bubble_presence():
    """``spoke`` is exactly 'did this stage emit a bubble' — true for a finding
    or a failure, false for a silent clean run."""
    assert StageResult(stage="x").spoke is False
    from saddle.models import BubbleEvent

    spoke = StageResult(stage="x", bubble=BubbleEvent(text="hi"))
    assert spoke.spoke is True
