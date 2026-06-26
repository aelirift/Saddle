"""Conversational-intent tracker — parsing, the deterministic "pick a)" catch,
and the headline (tenant, project) isolation guarantee through real SQLite.

The failure these tests pin down: the user picks ``a``, the agent's context later
compacts away what ``a`` meant, and an action that is "some other a)" must be
caught. saddle's ledger outlives the context window, so the catch is a plain
equality check — and a pick in one project must NEVER resolve a fork in another.
"""
from __future__ import annotations

from saddle.action_store import InMemoryActionStore, SqliteActionStore
from saddle.context import Context
from saddle.dialog import (
    DeterministicMatcher,
    IntentTracker,
    parse_declared_choice,
    parse_declared_label,
    parse_fork,
    resolve_selection,
)
from saddle.dialog_store import InMemoryForkStore, SqliteForkStore
from saddle.models import (
    ACT_CONTESTED,
    ACT_DELETE,
    ACT_RECORDED,
    BIND_AMBIGUOUS,
    BIND_LABEL,
    BIND_POSITION,
    BIND_RECOMMENDED,
    DRIFT_ALIGNED,
    DRIFT_DRIFT,
    DRIFT_UNKNOWN,
    FORK_RESOLVED,
    Action,
    Fork,
    ForkOption,
)

_AGENT_LETTER = """\
Here's the decision on caching. Which do you want?
a) In-memory LRU — fast, lost on restart
b) Redis — survives restarts, adds a dependency
c) SQLite page cache — middle ground (recommended)
"""

_AGENT_DIGIT = """\
Pick a transport:
1. WebSocket
2. SSE
3. long-poll
"""

_AGENT_BOLD = """\
Two ways to do it:
- **Option A**: synchronous hook
- **Option B**: async queue
"""

_AGENT_EG = """\
We could cache things. e.g. an LRU in front of the store.
Two options:
a) yes, add caching
b) no, keep it simple
"""


# === parsing the agent's offered fork ====================================

def test_parse_letter_fork_with_prompt_and_recommendation():
    fork = parse_fork(_AGENT_LETTER)
    assert fork is not None
    assert fork.labels() == ["a", "b", "c"]
    assert "caching" in fork.prompt.lower()
    rec = fork.recommended_option()
    assert rec is not None and rec.label == "c"


def test_parse_digit_fork():
    fork = parse_fork(_AGENT_DIGIT)
    assert fork is not None
    assert fork.labels() == ["1", "2", "3"]


def test_parse_bold_option_fork():
    fork = parse_fork(_AGENT_BOLD)
    assert fork is not None
    assert fork.labels() == ["a", "b"]


def test_eg_and_ie_are_not_mistaken_for_options():
    # "e.g." has no space after its inner dot -> not an option line.
    fork = parse_fork(_AGENT_EG)
    assert fork is not None
    assert fork.labels() == ["a", "b"]  # no stray "e"


def test_prose_without_options_is_not_a_fork():
    assert parse_fork("Let me just refactor the resolver and run the tests.") is None
    assert parse_fork("") is None


# === resolving the user's pick (deterministic) ===========================

def _fork() -> Fork:
    f = parse_fork(_AGENT_LETTER)
    assert f is not None
    f.id = "frk_test"
    return f


def test_resolve_solo_label():
    b = resolve_selection(_fork(), "a")
    assert b is not None and b.resolved and b.label == "a" and b.method == BIND_LABEL


def test_resolve_verb_label():
    for text, lab in [("go with b", "b"), ("option c", "c"), ("let's do a", "a")]:
        b = resolve_selection(_fork(), text)
        assert b is not None and b.resolved and b.label == lab, text


def test_resolve_position():
    assert resolve_selection(_fork(), "the first one").label == "a"
    assert resolve_selection(_fork(), "second").label == "b"
    last = resolve_selection(_fork(), "last")
    assert last.label == "c" and last.method == BIND_POSITION


def test_resolve_recommended_phrase_binds_recommended_option():
    b = resolve_selection(_fork(), "your call")
    assert b is not None and b.resolved
    assert b.label == "c" and b.method == BIND_RECOMMENDED


def test_recommended_phrase_without_recommendation_is_ambiguous():
    plain = parse_fork(_AGENT_BOLD)  # no recommended option
    plain.id = "frk_plain"
    b = resolve_selection(plain, "your call")
    assert b is not None and not b.resolved and b.method == BIND_AMBIGUOUS


def test_label_not_offered_is_ambiguous_not_a_guess():
    b = resolve_selection(_fork(), "z")
    assert b is not None and not b.resolved
    assert b.method == BIND_AMBIGUOUS and b.label == ""


def test_non_pick_reply_returns_none():
    assert resolve_selection(_fork(), "do a refactor of the resolver") is None
    assert resolve_selection(_fork(), "why did you choose that earlier?") is None


# === an agent action declaring which option it acts on ===================

def test_parse_declared_label_explicit_forms():
    assert parse_declared_label("I'm going with option a now") == "a"
    assert parse_declared_label("approach b looks cleanest") == "b"
    assert parse_declared_label("chose c for the cache") == "c"


def test_parse_declared_label_ignores_prose_a():
    # "doing a refactor" must NOT yield a phantom label that fakes a drift.
    assert parse_declared_label("Let me start doing a refactor of the module") == ""
    assert parse_declared_label("added a helper") == ""


# === the tracker round-trip: offer -> pick -> verdict ====================

def test_tracker_catches_pick_a_then_action_b_as_drift():
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())

    fork = tracker.observe_agent_message(ctx, _AGENT_LETTER, session="s1")
    assert fork is not None and fork.id

    binding = tracker.observe_user_message(ctx, "a", session="s1")
    assert binding is not None and binding.resolved and binding.label == "a"

    # the fork is now resolved, the binding active
    assert tracker.active_binding(ctx, session="s1").label == "a"

    # the canonical catch — action declares "b", user bound "a"
    drift = tracker.check_action(ctx, "b", session="s1")
    assert drift.is_drift and drift.status == DRIFT_DRIFT
    assert drift.bound_label == "a" and drift.action_label == "b"

    # acting on the bound option is aligned
    assert tracker.check_action(ctx, "a", session="s1").status == DRIFT_ALIGNED
    # an action that declares no option can't be judged deterministically
    assert tracker.check_action(ctx, "", session="s1").status == DRIFT_UNKNOWN


def test_ambiguous_reply_does_not_become_an_active_binding():
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    tracker.observe_agent_message(ctx, _AGENT_LETTER)

    amb = tracker.observe_user_message(ctx, "z")  # not an offered label
    assert amb is not None and not amb.resolved
    assert tracker.active_binding(ctx) is None
    assert tracker.check_action(ctx, "a").status == DRIFT_UNKNOWN

    # a real pick afterward still resolves the still-open fork
    good = tracker.observe_user_message(ctx, "a")
    assert good is not None and good.resolved and good.label == "a"


def test_no_open_fork_means_a_bare_pick_binds_nothing():
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    # user says "a" with no fork ever offered -> nothing to bind
    assert tracker.observe_user_message(ctx, "a") is None
    assert tracker.check_action(ctx, "a").status == DRIFT_UNKNOWN


# === the headline: multi-project / multi-tenant isolation (real SQLite) ==

def test_pick_in_project_y_never_resolves_a_fork_offered_in_project_x(tmp_path):
    store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=store)
    ctx_x = Context(tenant="acme", project="projx")
    ctx_y = Context(tenant="acme", project="projy")

    fork_x = tracker.observe_agent_message(ctx_x, _AGENT_LETTER, session="s1")
    assert fork_x is not None

    # the SAME pick text, in a DIFFERENT project, binds nothing
    assert tracker.observe_user_message(ctx_y, "a", session="s1") is None
    assert tracker.check_action(ctx_y, "a", session="s1").status == DRIFT_UNKNOWN

    # project x still has its open fork and resolves correctly
    bx = tracker.observe_user_message(ctx_x, "a", session="s1")
    assert bx is not None and bx.label == "a"
    assert tracker.check_action(ctx_x, "b", session="s1").is_drift
    # and project y, which never had a binding, is still UNKNOWN (no false drift)
    assert tracker.check_action(ctx_y, "b", session="s1").status == DRIFT_UNKNOWN
    store.close()


def test_same_project_name_different_tenant_is_isolated(tmp_path):
    store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=store)
    acme = Context(tenant="acme", project="projx")
    globex = Context(tenant="globex", project="projx")  # same project name

    tracker.observe_agent_message(acme, _AGENT_LETTER)
    # globex shares the project NAME but not the tenant -> no open fork for it
    assert tracker.observe_user_message(globex, "a") is None
    assert tracker.check_action(globex, "a").status == DRIFT_UNKNOWN
    store.close()


def test_session_scopes_picks_within_a_project(tmp_path):
    store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=store)
    ctx = Context(tenant="acme", project="game")

    tracker.observe_agent_message(ctx, _AGENT_LETTER, session="s1")
    # a pick from a different session does not bind the s1 fork
    assert tracker.observe_user_message(ctx, "a", session="s2") is None
    # a project-wide pick (no session filter) does bind it
    b = tracker.observe_user_message(ctx, "a", session="")
    assert b is not None and b.label == "a"
    store.close()


def test_sqlite_store_scopes_get_and_open_forks(tmp_path):
    store = SqliteForkStore(tmp_path / "saddle.db")
    ctx_x = Context(tenant="acme", project="projx")
    ctx_y = Context(tenant="acme", project="projy")
    fork = store.add_fork(ctx_x, Fork(options=[ForkOption("a"), ForkOption("b")]))

    assert store.get_fork(ctx_x, fork.id) is not None
    assert store.get_fork(ctx_y, fork.id) is None          # cross-project read blocked
    assert [f.id for f in store.open_forks(ctx_x)] == [fork.id]
    assert store.open_forks(ctx_y) == []
    store.close()


def test_fork_resolution_flips_status_and_closes_it(tmp_path):
    store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=store)
    ctx = Context(tenant="acme", project="game")

    fork = tracker.observe_agent_message(ctx, _AGENT_LETTER)
    tracker.observe_user_message(ctx, "b")
    reloaded = store.get_fork(ctx, fork.id)
    assert reloaded.status == FORK_RESOLVED
    assert store.open_forks(ctx) == []   # resolved forks drop out of "open"
    store.close()


def test_deterministic_matcher_is_the_default():
    tracker = IntentTracker(store=InMemoryForkStore())
    assert isinstance(tracker._matcher, DeterministicMatcher)


def test_action_label_outside_bound_fork_is_unknown_not_drift():
    """A bare label from a DIFFERENT decision space must not FAKE a drift. The user
    bound '1' of a 1/2/3 fork; an action that declares 'a' (an option of NO tracked
    fork) cannot contradict the commitment, so it's a quiet UNKNOWN — not a fake
    drift, and not noisy must-confirm either. A different option OF THE SAME fork
    stays a real, deterministic, announced drift (that's what never goes silent)."""
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=InMemoryForkStore())
    tracker.observe_agent_message(ctx, _AGENT_DIGIT)        # offers 1 / 2 / 3
    b = tracker.observe_user_message(ctx, "1")
    assert b is not None and b.resolved and b.label == "1"
    # 'a' is an option of no tracked fork -> nothing to compare -> quiet UNKNOWN
    v = tracker.check_action(ctx, "a")
    assert v.status == DRIFT_UNKNOWN and not v.is_drift
    assert not v.surface and not v.announce  # unrelated decision space, not a drift
    # but a different option OF THE SAME fork is a real drift, and announced
    d = tracker.check_action(ctx, "2")
    assert d.is_drift and d.announce


# === fork-choice IDENTITY: the (fork, option) unit, not a bare letter ====
#
# The 22-hour failure: the user picked "a" of fork p1.f6; the agent did "a" of a
# DIFFERENT fork (p2.f8). A label-only check blesses that ("a" == "a"); a
# fork-choice check catches it ("p1.f6.a" != "p2.f8.a"). saddle numbers every fork
# (node_id "p<prompt>.f<seq>") and every pick (choice_id "p<prompt>.f<seq>.<label>")
# so the comparison is exact and survives the agent's context compaction.

def _bound_letter_fork(store=None):
    """A tracker with an a/b/c fork offered in exchange 1 and "a" bound -> the
    live commitment is choice_id "p1.f1.a". Returns (tracker, ctx, fork, binding)."""
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(store=store or InMemoryForkStore())
    tracker.observe_user_message(ctx, "let's decide caching")   # opens exchange 1
    fork = tracker.observe_agent_message(ctx, _AGENT_LETTER)    # node p1.f1
    binding = tracker.observe_user_message(ctx, "a")            # binds p1.f1.a
    return tracker, ctx, fork, binding


def test_fork_gets_node_id_and_binding_gets_choice_id():
    tracker, ctx, fork, binding = _bound_letter_fork()
    assert fork.node_id == "p1.f1"
    assert fork.choice_id("a") == "p1.f1.a"
    assert binding.resolved and binding.choice_id == "p1.f1.a"
    assert tracker.active_binding(ctx).choice_id == "p1.f1.a"


def test_cited_choice_aligns_with_committed_fork_choice():
    tracker, ctx, fork, _ = _bound_letter_fork()
    v = tracker.check_action(ctx, choice_id="p1.f1.a")
    assert v.status == DRIFT_ALIGNED and not v.is_drift
    assert v.action_choice == "p1.f1.a" and v.bound_choice == "p1.f1.a"
    assert not v.announce          # an aligned action is quiet


def test_cited_choice_same_fork_other_option_is_drift():
    tracker, ctx, fork, _ = _bound_letter_fork()
    v = tracker.check_action(ctx, choice_id="p1.f1.b")
    assert v.status == DRIFT_DRIFT and v.is_drift and v.announce
    assert v.bound_choice == "p1.f1.a" and v.action_choice == "p1.f1.b"


def test_cited_choice_wrong_fork_same_letter_is_drift_THE_22H_FAILURE():
    """THE canonical failure. The user committed to "a" of fork p1.f1; a SECOND real
    fork is also on the table; the agent cites the "a" of THAT other fork. Same
    letter, wrong fork -> DRIFT, surfaced. A label-only check (the bug) called this
    aligned for 22 hours."""
    tracker, ctx, fork, _ = _bound_letter_fork()
    other = tracker.observe_agent_message(ctx, _AGENT_LETTER)   # a real 2nd fork
    assert other.node_id != fork.node_id
    v = tracker.check_action(ctx, choice_id=other.choice_id("a"))
    assert v.status == DRIFT_DRIFT and v.is_drift and v.announce
    assert v.bound_choice == "p1.f1.a" and v.action_choice == other.choice_id("a")


def test_dotted_number_that_names_no_fork_is_not_a_citation():
    """The false-positive guard a real transcript exposed — both nets exercised.
    An untagged dotted number ("127.0.0", a localhost IP; "125.203.41", a line
    range) can't even parse as a fork-choice locator (no p/f tags), so it's
    rejected at the FORMAT level. A well-formed but unminted locator ("p9.f9.a")
    parses, yet names no fork saddle assigned, so the node-existence net rejects
    it. Neither is a citation; neither may fake a drift."""
    tracker, ctx, fork, _ = _bound_letter_fork()
    for stray in ("127.0.0", "125.203.41", "p9.f9.a"):
        v = tracker.check_action(ctx, choice_id=stray)
        assert v.status == DRIFT_UNKNOWN and not v.is_drift and not v.announce, stray


def test_same_choice_id_string_means_different_things_per_project(tmp_path):
    """Numbering is (tenant, project)-unique: both projects can mint node "p1.f1",
    yet a cited "p1.f1.a" is judged against THAT project's own commitment."""
    store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=store)
    cx = Context(tenant="acme", project="projx")
    cy = Context(tenant="acme", project="projy")

    tracker.observe_user_message(cx, "decide")
    fx = tracker.observe_agent_message(cx, _AGENT_LETTER)
    tracker.observe_user_message(cx, "a")                     # projx bound p1.f1.a
    tracker.observe_user_message(cy, "decide")
    fy = tracker.observe_agent_message(cy, _AGENT_LETTER)
    tracker.observe_user_message(cy, "b")                     # projy bound p1.f1.b

    assert fx.node_id == fy.node_id == "p1.f1"                  # same string, two projects
    assert tracker.check_action(cx, choice_id="p1.f1.a").status == DRIFT_ALIGNED
    # the SAME cited id is a drift in projy, whose commitment is p1.f1.b
    vy = tracker.check_action(cy, choice_id="p1.f1.a")
    assert vy.is_drift and vy.bound_choice == "p1.f1.b" and vy.announce
    store.close()


# === the never-silent guarantee (the "harmless fix isn't harmless" guard) =

def test_only_aligned_and_nothing_to_compare_are_quiet():
    """surface/announce is False for exactly two cases: an aligned action, and
    "nothing to compare" (no commitment, or the action declares no option).
    Everything else that touches the commitment is surfaced."""
    tracker, ctx, fork, _ = _bound_letter_fork()
    assert not tracker.check_action(ctx, choice_id="p1.f1.a").announce   # aligned
    assert not tracker.check_action(ctx, "").announce                  # no declared option

    fresh = Context(tenant="acme", project="empty")
    t2 = IntentTracker(store=InMemoryForkStore())
    assert not t2.check_action(fresh, "a").announce                    # no commitment


def test_every_contradiction_and_must_confirm_is_surfaced():
    """No real drift is downgraded to a silent UNKNOWN. This is the guard against
    a 'harmless' UNKNOWN hiding a drift the user needs announced."""
    # same-fork contradictions, with only the committed fork in play
    t1, c1, f1, _ = _bound_letter_fork()                  # committed p1.f1.a
    assert t1.check_action(c1, choice_id="p1.f1.b").announce    # cited same-fork drift
    assert t1.check_action(c1, "b").announce                  # bare same-fork drift

    # a real DIFFERENT fork's option, cited explicitly -> drift, surfaced; and a
    # bare label now ambiguous across the two forks -> surfaced must-confirm
    t2, c2, f2, _ = _bound_letter_fork()                  # committed p1.f1.a
    other = t2.observe_agent_message(c2, _AGENT_LETTER)       # a real 2nd fork
    assert t2.check_action(c2, choice_id=other.choice_id("a")).announce
    amb = t2.check_action(c2, "a")
    assert amb.status == DRIFT_UNKNOWN and not amb.is_drift and amb.announce


# === parsing a declared fork-choice id from an agent action ==============

def test_parse_declared_choice_extracts_qualified_id():
    assert parse_declared_choice("I'm acting on p1.f6.a now") == "p1.f6.a"
    assert parse_declared_choice("going with p2.f10.b for the cache") == "p2.f10.b"
    assert parse_declared_choice("doing p3.f4.2, the third option") == "p3.f4.2"


def test_parse_declared_choice_ignores_bare_label_and_prose():
    assert parse_declared_choice("option a looks cleanest") == ""
    assert parse_declared_choice("let me refactor the resolver") == ""
    assert parse_declared_choice("") == ""
    # a bare verb-label still parses as a label, but NOT as a qualified choice
    assert parse_declared_label("going with option a") == "a"
    assert parse_declared_choice("going with option a") == ""


# === action provenance: number what the agent DID, dispute the reason ====

def test_record_and_find_action_provenance():
    """The real example: the agent deleted f_detail_map(); months later
    "where's the map feature?" must answer with the action id, session, and the
    recorded reason."""
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(
        store=InMemoryForkStore(), action_store=InMemoryActionStore()
    )
    tracker.observe_user_message(ctx, "clean up dead code")     # exchange 1
    act = tracker.record_action(
        ctx,
        Action(
            summary="deleted map feature code", kind=ACT_DELETE, file="a.json",
            line_start=5, line_end=250, symbol="f_detail_map",
            reason="not wired properly, no caller for this function",
        ),
        session="sX",
    )
    assert act.aid == "p1.act1" and act.status == ACT_RECORDED
    assert act.pn == 1 and act.session == "sX"

    hits = tracker.find_actions(ctx, query="map")
    assert [h.aid for h in hits] == ["p1.act1"]
    h = hits[0]
    assert h.symbol == "f_detail_map" and h.session == "sX"
    assert "not wired" in h.reason and h.line_start == 5 and h.line_end == 250
    # findable by symbol too
    assert tracker.find_actions(ctx, symbol="f_detail_map")[0].aid == "p1.act1"


def test_dispute_action_marks_contested_with_counter_reason():
    """The user disputes the REASON, not the fact: "it wasn't hooked up, so the
    fix was to hook it up, not delete it." The action is CONTESTED, never erased."""
    ctx = Context(tenant="acme", project="game")
    tracker = IntentTracker(
        store=InMemoryForkStore(), action_store=InMemoryActionStore()
    )
    act = tracker.record_action(
        ctx, Action(summary="removed map", symbol="f_detail_map", kind=ACT_DELETE)
    )
    assert tracker.dispute_action(
        ctx, act.aid, "it wasn't hooked up properly — hook it up, don't delete it"
    )
    reloaded = tracker.find_actions(ctx, symbol="f_detail_map")[0]
    assert reloaded.status == ACT_CONTESTED
    assert "hook it up" in reloaded.dispute
    # disputing a non-existent action reports failure, doesn't raise
    assert tracker.dispute_action(ctx, "p9.act999", "no such action") is False


def test_action_numbering_and_provenance_are_project_isolated(tmp_path):
    """Each (tenant, project) runs its own act-series, and one project's actions
    never surface in another's lookups — through real SQLite."""
    store = SqliteActionStore(tmp_path / "saddle.db")
    fork_store = SqliteForkStore(tmp_path / "saddle.db")
    tracker = IntentTracker(store=fork_store, action_store=store)
    cx = Context(tenant="acme", project="projx")
    cy = Context(tenant="acme", project="projy")

    ax = tracker.record_action(cx, Action(summary="x work", symbol="fx", kind=ACT_DELETE))
    ay = tracker.record_action(cy, Action(summary="y work", symbol="fy", kind=ACT_DELETE))
    # neither project opened an exchange first, so both number from p0; both
    # start their act-series at p0.act1 independently
    assert ax.aid == "p0.act1" and ay.aid == "p0.act1"
    # and neither sees the other's action
    assert [a.symbol for a in tracker.find_actions(cx)] == ["fx"]
    assert [a.symbol for a in tracker.find_actions(cy)] == ["fy"]
    assert tracker.find_actions(cx, symbol="fy") == []
    store.close()
    fork_store.close()
