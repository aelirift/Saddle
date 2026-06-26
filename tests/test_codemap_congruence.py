"""Congruence: the WHOLE server->client mirror bug class, derived per-function
across files instead of named mutator-by-mutator.

This is the gap AuthoritySpec left open. Authority checks a HAND-LISTED set of
mutators in isolation ("does this named function call a guard in its own body?").
Congruence DERIVES the mutator set from the code and correlates each with its call
sites: a mirror service is any module defining ``mirror_apply`` (the client-side
snapshot applier); the fields ``mirror_apply`` writes ARE the replicated state; any
public function that writes one of them is a mutator. A replicated-state mutator is
safe on a real client ONLY when it is BOTH authority-gated AND routed (a server
``@rpc`` receiver carries a remote client's intent to the authority). The TWO ways
that safety breaks are the two halves of the bug class, and BOTH are gaps:

  * shape #1 -- UNGATED + a raw client/HUD caller: the HUD mutates the LOCAL mirror,
    the next server snapshot reverts it (the change flickers then vanishes).
  * shape #2 -- GATED but UNROUTED + a raw client/HUD caller: on a remote client the
    guard bails, the click silently no-ops, and the server never learns the intent.

That is the exact shape RayXI shipped five times (mount_collection, stat_allocation,
auction_house, profession_skill_curve, weekly_vault). AuthoritySpec couldn't see it
-- it never correlated a mutator with WHO calls it, and it had no notion of a route.
This axis finds both halves mechanically, with no hand-authored mutator list:
``mirror_apply`` + ``guard`` are project ENGINE tokens, and the mutator set, its
callers, and the ``@rpc`` routes are read from the AST.
"""
from __future__ import annotations

from saddle.codemap import gdref, pyref
from saddle.codemap.checks import check_congruence
from saddle.codemap.impact import impact_congruence
from saddle.codemap.specs import CongruenceSpec

# A real-shaped mirror service (Godot idiom). apply_replication_snapshot lands the
# server snapshot on the client, so the field it writes (_state) IS the replicated
# state. Three mutators of _state model the three outcomes:
#   * allocate_point -- UNGATED                    -> shape #1 gap
#   * reset_points   -- gated, but NO server route -> shape #2 gap
#   * spend_point    -- gated AND routed (the @rpc receiver below calls it) -> CLEAN
# get_points only reads it; register_stat writes a DIFFERENT field (a build-time
# catalog), so neither is a mirror mutator at all.
GD_SERVICE = '''
func apply_replication_snapshot(snap):
    _state = snap["state"]
    stat_points_changed.emit()

func _is_server() -> bool:
    return multiplayer.is_server()

func allocate_point(stat):
    _state[stat] += 1

func reset_points():
    if not _is_server():
        return
    _state = {}

func spend_point(stat):
    if not _is_server():
        return
    _state[stat] -= 1

func get_points(stat):
    return _state.get(stat, 0)

func register_stat(stat_name):
    stat_catalog[stat_name] = true
'''

# The HUD calls ALL THREE mutators with the identical `_runtime.X()` shape from its
# button handlers. It `extends CanvasLayer`, a Godot UI base, so the module is a
# CLIENT module (a real HUD) -- the signal that tells a genuine UI caller of a server
# mutator apart from a server-side service-to-service call. The spend handler ALSO
# routes through the intent router first (the proven fix's local-fallback pattern) --
# but the detector keys on the @rpc RECEIVER in the router, not the sender here, so
# this direct fallback call is what puts spend_point on a client's invocation path.
GD_HUD = '''
extends CanvasLayer

func _on_allocate_pressed(stat):
    _runtime.allocate_point(stat)

func _on_reset_pressed():
    _runtime.reset_points()

func _on_spend_pressed(stat):
    if IntentRouter.send_spend(stat):
        return
    _runtime.spend_point(stat)
'''

# The intent router: a client-side `send_spend` fires an @rpc to the server, whose
# `_recv_spend` receiver runs on the authority and calls the mutator. THIS receiver
# is the server route that makes spend_point safe -- gated AND reachable from a real
# server path. reset_points has no such receiver, which is exactly why it is a gap.
GD_ROUTER = '''
func send_spend(stat):
    if not _is_client_with_peer():
        return false
    _recv_spend.rpc(stat)
    return true

@rpc("any_peer", "call_remote", "reliable")
func _recv_spend(stat):
    _runtime.spend_point(stat)
'''

SPEC = CongruenceSpec(name="stat_allocation",
                      mirror_apply="apply_replication_snapshot", guard="_is_server")


def _gd_mods():
    return gdref.parse_modules([("stat_runtime.gd", GD_SERVICE),
                                ("stat_hud.gd", GD_HUD),
                                ("intent_router.gd", GD_ROUTER)])


def test_gd_derives_mutator_set_and_both_gap_shapes():
    mods = _gd_mods()
    imp = impact_congruence(mods, SPEC)
    # the mirror service is recognised by its apply fn
    assert {r.location for r in imp.services} == {"stat_runtime.gd:2"}
    # the mutator set is DERIVED: only the three functions that write the replicated
    # field _state -- the getter and the catalog-registrar are correctly excluded
    assert {m.name for m in imp.mutators} == {"allocate_point", "reset_points", "spend_point"}
    # TWO gaps: the ungated one (shape #1) AND the gated-but-unrouted one (shape #2).
    # spend_point is gated AND routed, so it is clean despite the identical HUD call.
    assert {m.name for m in imp.gap_mutators} == {"allocate_point", "reset_points"}
    kinds = {m.name: m.gap_kind for m in imp.gap_mutators}
    assert kinds == {"allocate_point": "ungated", "reset_points": "unrouted"}
    # the gate is exactly the projection of the impact set
    assert check_congruence(mods, SPEC) == imp.gaps()


def test_gd_ungated_gap_names_its_hud_caller():
    mods = _gd_mods()
    imp = impact_congruence(mods, SPEC)
    alloc = next(m for m in imp.mutators if m.name == "allocate_point")
    assert not alloc.guarded
    assert alloc.is_gap and alloc.gap_kind == "ungated"
    f = next(g for g in imp.gaps() if g.detail["func"] == "allocate_point")
    assert f.node_kind == "congruence"
    assert f.check == "client_server_congruence"
    assert f.severity == "error"
    assert f.detail["gap_kind"] == "ungated"
    # the finding names the HUD call site (the correlation AuthoritySpec can't make)
    assert "stat_hud.gd:5" in f.detail["external_calls"]
    assert "reverts" in f.message  # shape #1 explanation: the snapshot reverts it


def test_gd_gated_but_unrouted_is_a_gap():
    # reset_points is called by the HUD with the SAME shape as spend_point and it
    # calls _is_server() first -- but NO @rpc receiver routes a remote client's intent
    # to it, so on a real client the guard bails and the click is silently lost. The
    # refined contract turns on gating AND routing, not gating alone.
    mods = _gd_mods()
    imp = impact_congruence(mods, SPEC)
    reset = next(m for m in imp.mutators if m.name == "reset_points")
    assert reset.guarded            # it IS authority-gated ...
    assert not reset.rpc_routed     # ... but no server route reaches it ...
    assert reset.raw_client_callers  # ... and a raw HUD call invokes it
    assert reset.is_gap and reset.gap_kind == "unrouted"
    f = next(g for g in imp.gaps() if g.detail["func"] == "reset_points")
    assert f.detail["gap_kind"] == "unrouted"
    assert f.detail["routed"] is False
    assert "stat_hud.gd:8" in f.detail["external_calls"]
    assert "silently lost" in f.message  # shape #2 explanation


def test_gd_gated_and_routed_mutator_is_clean():
    # spend_point is the safe shape: authority-gated AND reached by an @rpc receiver
    # (_recv_spend in the router). The HUD's direct fallback call still puts it on a
    # client path, but gated+routed clears BOTH halves of the bug class.
    mods = _gd_mods()
    imp = impact_congruence(mods, SPEC)
    spend = next(m for m in imp.mutators if m.name == "spend_point")
    assert spend.guarded
    assert spend.rpc_routed            # the @rpc receiver is the server route
    # the @rpc receiver's call is told apart from the genuine HUD call ...
    assert {r.func for r in spend.raw_client_callers} == {"_on_spend_pressed"}
    assert "_recv_spend" not in {r.func for r in spend.raw_client_callers}
    assert not spend.is_gap            # gated AND routed -> not a divergence


def test_gd_multiline_dynamic_call_route_is_detected():
    # RayXI's real router routes through the DYNAMIC dispatch form, and the `.call(`
    # routinely opens on one line with the "mutator" string on the NEXT
    # (`str(svc.call(\n    "push_state", ...))`). A line-scoped caller scan can't see
    # the split call, so a genuinely-routed mutator reads as UNROUTED -- the exact
    # false gap auction_house.list_item produced on real code. The whole-source
    # dynamic scan crosses the newline, so the gated+routed mutator is correctly clean.
    svc = '''
func apply_replication_snapshot(snap):
    _state = snap["state"]

func _is_server() -> bool:
    return multiplayer.is_server()

func push_state(v):
    if not _is_server():
        return
    _state["v"] = v
'''
    hud = '''
extends CanvasLayer

func _on_push(v):
    _svc.push_state(v)
'''
    router = '''
@rpc("any_peer", "call_remote", "reliable")
func _recv_push(v):
    var ok: bool = bool(_svc.call(
        "push_state", v,
    ))
'''
    mods = gdref.parse_modules([("svc.gd", svc), ("svc_hud.gd", hud),
                                ("router.gd", router)])
    imp = impact_congruence(mods, SPEC)
    push = next(m for m in imp.mutators if m.name == "push_state")
    assert push.guarded
    assert push.rpc_routed              # the multi-line .call("push_state") IS the route
    assert {r.func for r in push.raw_client_callers} == {"_on_push"}
    assert not push.is_gap             # gated AND routed across the line break -> clean
    assert check_congruence(mods, SPEC) == []


def test_gd_mutator_writes_replicated_field_through_private_helper():
    # The mutator lands its replicated-state write ONLY through a PRIVATE helper it
    # calls -- chat_backend.send_message never touches _history directly; it delegates
    # to _record_history. A direct-writes-only scan reads post_message as a non-writer
    # and misses the gap entirely (a gated, client-called, UNROUTED mutator silently
    # absent from the report). Following the private-helper edge restores it, while a
    # public function that flows only into a private READER stays correctly unflagged.
    svc = '''
func apply_replication_snapshot(snap):
    _history = snap["history"]

func _is_server() -> bool:
    return multiplayer.is_server()

func post_message(body):
    if not _is_server():
        return
    _record(body)

func _record(body):
    _history[body] = true

func peek(key):
    return _read(key)

func _read(key):
    return _history.get(key, false)
'''
    hud = '''
extends CanvasLayer

func _on_send(body):
    _svc.post_message(body)
'''
    spec = CongruenceSpec(name="chat", mirror_apply="apply_replication_snapshot",
                          guard="_is_server")
    mods = gdref.parse_modules([("chat.gd", svc), ("chat_hud.gd", hud)])
    imp = impact_congruence(mods, spec)
    # post_message is a DERIVED mutator even though its only _history write is in _record
    post = next(m for m in imp.mutators if m.name == "post_message")
    assert {w.name for w in post.write_refs} == {"_history"}
    # peek flows ONLY into a private READER helper -> writes nothing -> not a mutator.
    # The edge-follow adds writes, never over-promotes a pure read path.
    assert "peek" not in {m.name for m in imp.mutators}
    # gated but unrouted + a raw HUD caller -> the shape-2 gap, now visible
    assert post.guarded and not post.rpc_routed
    assert {r.func for r in post.raw_client_callers} == {"_on_send"}
    assert post.is_gap and post.gap_kind == "unrouted"
    assert {f.detail["func"] for f in check_congruence(mods, spec)} == {"post_message"}


def test_gd_server_only_caller_is_not_a_gap():
    # An ungated mutator whose ONLY external caller is itself server-authoritative
    # (@rpc("authority")) is a server-internal call, not a client divergence. The
    # gap requires a RAW client caller, so this stays silent.
    svc = '''
func apply_replication_snapshot(snap):
    _state = snap

func push_state(v):
    _state = v
'''
    server_caller = '''
@rpc("authority")
func broadcast(v):
    _svc.push_state(v)
'''
    mods = gdref.parse_modules([("svc.gd", svc), ("sync.gd", server_caller)])
    imp = impact_congruence(mods, SPEC)
    push = next(m for m in imp.mutators if m.name == "push_state")
    assert not push.guarded             # it really is ungated ...
    assert push.external_calls          # ... and it really is called externally ...
    assert push.raw_client_callers == []  # ... but the only caller is server-domain
    assert not push.is_gap
    assert check_congruence(mods, SPEC) == []


def test_gd_harness_caller_is_not_a_gap():
    # A build-oracle probe / unit test drives an UNGATED mutator directly to exercise
    # it -- it runs with authority and is NOT the client UI. Matched by filename
    # convention (a `*_probe.gd` / `test_*` / `tests/` path), such a caller never
    # makes a mutator "client-reachable", so it is not a false-positive gap.
    svc = '''
func apply_replication_snapshot(snap):
    _state = snap

func bump(v):
    _state = v
'''
    probe = '''
func _run():
    _svc.bump(5)
'''
    mods = gdref.parse_modules([("svc.gd", svc), ("build_oracle_probe.gd", probe)])
    imp = impact_congruence(mods, SPEC)
    bump = next(m for m in imp.mutators if m.name == "bump")
    assert bump.external_calls == []    # the probe caller is filtered out entirely
    assert not bump.is_gap
    assert check_congruence(mods, SPEC) == []


def test_no_mirror_service_yields_no_findings():
    # If no module defines mirror_apply, there is no replication mirror to reason
    # about -- the axis stays silent rather than inventing a phantom service. This is
    # the safe failure mode that lets an auto-derived spec fire harmlessly on a
    # project that has no such service.
    plain = 'func set_x(v):\n    _state = v\n'
    mods = gdref.parse_modules([("plain.gd", plain), ("hud.gd", GD_HUD)])
    imp = impact_congruence(mods, SPEC)
    assert imp.services == []
    assert imp.mutators == []
    assert check_congruence(mods, SPEC) == []


def test_exempt_functions_are_not_flagged():
    # `exempt` is the escape hatch for functions a mirror service legitimately leaves
    # ungated/unrouted. Naming both gap mutators exempt suppresses their (otherwise
    # real) gaps without touching spend_point, which stays clean on its own merits.
    mods = _gd_mods()
    spec = CongruenceSpec(name="stat_allocation",
                          mirror_apply="apply_replication_snapshot",
                          guard="_is_server",
                          exempt=("allocate_point", "reset_points"))
    imp = impact_congruence(mods, spec)
    assert {m.name for m in imp.mutators} == {"spend_point"}
    assert check_congruence(mods, spec) == []


# --- caller attribution precision: same-named methods across services ----------

def test_gd_func_arity_counts_required_and_defaulted():
    # func_arity is the primitive the caller-scan uses to tell same-named methods
    # apart. A parameter carrying `=` (a default, incl. the `:=` form) is optional, so
    # it widens the max but not the minimum; a missing function yields None (the caller
    # then makes NO arity judgement -- an unknown never deletes a real caller).
    mod = gdref.parse_modules([("svc.gd", '''
func invite(guild_id, target, by):
    pass

func set_flag(name, value := true):
    pass

func nullary():
    pass
''')])[0]
    assert gdref.func_arity(mod, "invite") == (3, 3)
    assert gdref.func_arity(mod, "set_flag") == (1, 2)   # `value` is defaulted
    assert gdref.func_arity(mod, "nullary") == (0, 0)
    assert gdref.func_arity(mod, "does_not_exist") is None


def test_gd_same_name_mutators_disambiguated_by_arity():
    # Two mirror services both define `invite`, but with DIFFERENT arity: the party
    # service takes 2 parameters, the guild service 3. A HUD calls invite with THREE
    # arguments, which can only be the guild service's. Name-equality alone (the old
    # behaviour) charged the call to BOTH and invented a phantom gap on the party
    # service -- the exact group_framework.invite false positive the rayxi sweep hit.
    # Arity attributes the call to the guild service only.
    party = '''
func apply_replication_snapshot(snap):
    _members = snap["members"]

func _is_server() -> bool:
    return multiplayer.is_server()

func invite(target, inviter):
    if not _is_server():
        return
    _members[target] = inviter
'''
    guild = '''
func apply_replication_snapshot(snap):
    _roster = snap["roster"]

func _is_server() -> bool:
    return multiplayer.is_server()

func invite(guild_id, target, by):
    if not _is_server():
        return
    _roster[target] = guild_id
'''
    hud = '''
extends CanvasLayer

func _on_invite():
    _guild.invite("g1", "bob", "me")
'''
    spec = CongruenceSpec(name="social", mirror_apply="apply_replication_snapshot",
                          guard="_is_server")
    mods = gdref.parse_modules([("party.gd", party), ("guild.gd", guild),
                                ("social_hud.gd", hud)])
    imp = impact_congruence(mods, spec)
    guild_inv = next(m for m in imp.mutators
                     if m.module == "guild.gd" and m.name == "invite")
    party_inv = next(m for m in imp.mutators
                     if m.module == "party.gd" and m.name == "invite")
    # the 3-arg HUD call fits the guild signature (3 params) -> a gap there ...
    assert guild_inv.is_gap
    assert {r.func for r in guild_inv.raw_client_callers} == {"_on_invite"}
    # ... and CANNOT fit the party signature (2 params) -> not attributed, no phantom gap
    assert party_inv.raw_client_callers == []
    assert not party_inv.is_gap
    assert {f.location.split(":")[0] for f in check_congruence(mods, spec)} == {"guild.gd"}


def test_gd_same_arity_mutators_disambiguated_by_receiver_service():
    # The harder collision: two mirror services both define `kick` with the SAME 3-param
    # signature, so arity cannot separate them (the chat_backend.kick vs guild_system.kick
    # case). Each self-registers under its own service name; the HUD binds its receiver
    # from _service("guild") and calls receiver.kick(...). The receiver's resolved service
    # name is the only signal -- the call belongs to the guild service, not the chat one.
    guild = '''
func _enter_tree():
    get_node("/root/ServiceRegistry").call("register", "guild", self)

func apply_replication_snapshot(snap):
    _roster = snap["roster"]

func _is_server() -> bool:
    return multiplayer.is_server()

func kick(guild_id, target, by):
    if not _is_server():
        return
    _roster[target] = false
'''
    chat = '''
func _enter_tree():
    get_node("/root/ServiceRegistry").call("register", "chat", self)

func apply_replication_snapshot(snap):
    _channels = snap["channels"]

func _is_server() -> bool:
    return multiplayer.is_server()

func kick(channel_id, target, by):
    if not _is_server():
        return
    _channels[target] = false
'''
    hud = '''
extends CanvasLayer

func _on_kick(target):
    var svc: Node = _service("guild")
    svc.kick("g1", target, "me")
'''
    spec = CongruenceSpec(name="social", mirror_apply="apply_replication_snapshot",
                          guard="_is_server")
    mods = gdref.parse_modules([("guild.gd", guild), ("chat.gd", chat),
                                ("social_hud.gd", hud)])
    # the module declares its own locator identity, read off its register() call
    assert gdref.registered_service_names(mods[0]) == {"guild"}
    assert gdref.registered_service_names(mods[1]) == {"chat"}
    imp = impact_congruence(mods, spec)
    guild_kick = next(m for m in imp.mutators
                      if m.module == "guild.gd" and m.name == "kick")
    chat_kick = next(m for m in imp.mutators
                     if m.module == "chat.gd" and m.name == "kick")
    # the receiver was resolved from _service("guild"), so the call is the guild's ...
    assert guild_kick.is_gap
    assert {r.func for r in guild_kick.raw_client_callers} == {"_on_kick"}
    # ... and NOT the chat service's, despite the identical name and arity
    assert chat_kick.raw_client_callers == []
    assert not chat_kick.is_gap
    assert {f.location.split(":")[0] for f in check_congruence(mods, spec)} == {"guild.gd"}


def test_gd_gated_mutator_owns_replicated_write_through_public_delegate():
    # kick lands NO replicated write of its own -- it delegates to the PUBLIC `leave`,
    # which writes the replicated _roster. Because kick is authority-GATED (its guard is
    # the author's declaration "invoking me mutates authoritative state"), the write it
    # reaches through the public delegate is its responsibility, so the gated-but-
    # unrouted HUD-called kick is correctly a gap (the guild_system.kick -> leave shape).
    # `leave` is itself a mutator but has no HUD caller, so following the public delegate
    # did not double-count the divergence.
    svc = '''
func apply_replication_snapshot(snap):
    _roster = snap["roster"]

func _is_server() -> bool:
    return multiplayer.is_server()

func kick(guild_id, target, by):
    if not _is_server():
        return
    return leave(guild_id, target)

func leave(guild_id, target):
    if not _is_server():
        return
    _roster[target] = false
'''
    hud = '''
extends CanvasLayer

func _on_kick(target):
    _guild.kick("g1", target, "me")
'''
    spec = CongruenceSpec(name="guild", mirror_apply="apply_replication_snapshot",
                          guard="_is_server")
    mods = gdref.parse_modules([("guild.gd", svc), ("guild_hud.gd", hud)])
    imp = impact_congruence(mods, spec)
    kick = next(m for m in imp.mutators if m.name == "kick")
    # kick reaches the replicated _roster write only THROUGH the public `leave`
    assert {w.name for w in kick.write_refs} == {"_roster"}
    assert kick.guarded and not kick.rpc_routed
    assert {r.func for r in kick.raw_client_callers} == {"_on_kick"}
    assert kick.is_gap and kick.gap_kind == "unrouted"
    # `leave` IS a mutator (gated, writes _roster) but has no raw HUD caller -> not a gap
    leave = next(m for m in imp.mutators if m.name == "leave")
    assert leave.raw_client_callers == []
    assert not leave.is_gap
    assert {f.detail["func"] for f in check_congruence(mods, spec)} == {"kick"}


def test_gd_container_mutation_method_counts_as_a_replicated_write():
    # decline_request mutates the replicated _requests ONLY through `_requests.erase(rid)`
    # -- a Dictionary mutated IN PLACE, never reassigned. A write scan that saw only `=`
    # assignments read it as a pure reader and missed its unrouted client gap (the exact
    # friends_list.decline_request shape). Container-mutation methods (.erase / .append /
    # .clear / .pop_* / ...) count as a write of their base member, so the gated-but-
    # unrouted HUD-called decline_request is correctly a gap. The pure getter beside it
    # (a `.get` READ) is NOT a mutation, so following the rule does not over-flag readers.
    svc = '''
func apply_replication_snapshot(snap):
    _requests = {}
    for rid in snap["pending"]:
        _requests[rid] = snap["pending"][rid]

func _is_server() -> bool:
    return multiplayer.is_server()

func decline_request(rid):
    if not _is_server():
        return false
    _requests.erase(rid)
    return true

func pending_for(account):
    return _requests.get(account, [])
'''
    hud = '''
extends CanvasLayer

func _on_decline(rid):
    _social.decline_request(rid)
'''
    spec = CongruenceSpec(name="friends", mirror_apply="apply_replication_snapshot",
                          guard="_is_server")
    mods = gdref.parse_modules([("friends.gd", svc), ("friends_hud.gd", hud)])
    imp = impact_congruence(mods, spec)
    decline = next(m for m in imp.mutators if m.name == "decline_request")
    # the in-place `.erase` IS the replicated write, so decline_request is a mutator ...
    assert {w.name for w in decline.write_refs} == {"_requests"}
    # ... while the `.get` reader next to it is correctly NOT a mutator
    assert "pending_for" not in {m.name for m in imp.mutators}
    assert decline.guarded and not decline.rpc_routed
    assert {r.func for r in decline.raw_client_callers} == {"_on_decline"}
    assert decline.is_gap and decline.gap_kind == "unrouted"
    assert {f.detail["func"] for f in check_congruence(mods, spec)} == {"decline_request"}


# --- cross-language parity: the same derivation on the Python adapter ----------

PY_SERVICE = '''
_state = {}

def apply_replication_snapshot(snap):
    global _state
    _state = snap["state"]

def is_server():
    return True

def allocate_point(stat):
    _state[stat] += 1

def reset_points():
    if not is_server():
        return
    global _state
    _state = {}

def spend_point(stat):
    if not is_server():
        return
    _state[stat] -= 1
'''

PY_HUD = '''
# saddle-domain: client
def _on_allocate(stat):
    allocate_point(stat)

def _on_reset():
    reset_points()

def _on_spend(stat):
    spend_point(stat)
'''

# Python has no built-in @rpc; the parity fixture models the same shape with a
# decorator NAMED rpc, which the adapter's rpc_receivers recognises identically.
PY_ROUTER = '''
@rpc("any_peer")
def _recv_spend(stat):
    spend_point(stat)
'''


def test_py_parity_both_gap_shapes_and_clean_route():
    mods = pyref.parse_modules([("runtime.py", PY_SERVICE), ("hud.py", PY_HUD),
                                ("router.py", PY_ROUTER)])
    spec = CongruenceSpec(name="stat", mirror_apply="apply_replication_snapshot",
                          guard="is_server")
    imp = impact_congruence(mods, spec)
    assert {m.name for m in imp.mutators} == {"allocate_point", "reset_points", "spend_point"}
    # ungated -> shape #1, gated-but-unrouted -> shape #2, gated+routed -> clean
    assert {m.name for m in imp.gap_mutators} == {"allocate_point", "reset_points"}
    assert {m.name: m.gap_kind for m in imp.gap_mutators} == {
        "allocate_point": "ungated", "reset_points": "unrouted"}
    spend = next(m for m in imp.mutators if m.name == "spend_point")
    assert spend.rpc_routed and not spend.is_gap
    assert {f.detail["func"] for f in check_congruence(mods, spec)} == {
        "allocate_point", "reset_points"}


# Parity for the in-place container-mutation rule: a gated mutator whose only
# replicated write is `self._pending.pop(rid)` must read as a mutator on the Python
# adapter too (ast.Call to a known mutating method on a member receiver), not just on
# GDScript -- the AST `_mut_call_base` and the regex `_GD_METHOD_MUT` are kept in lockstep.
PY_MUT_SERVICE = '''
class Social:
    def __init__(self):
        self._pending = {}

    def apply_replication_snapshot(self, snap):
        self._pending = dict(snap["pending"])

    def is_server(self):
        return True

    def decline_request(self, rid):
        if not self.is_server():
            return False
        self._pending.pop(rid, None)
        return True

    def pending_for(self, account):
        return self._pending.get(account, [])
'''

PY_MUT_HUD = '''
# saddle-domain: client
def _on_decline(svc, rid):
    svc.decline_request(rid)
'''


def test_py_parity_container_mutation_method_counts_as_a_replicated_write():
    mods = pyref.parse_modules([("social.py", PY_MUT_SERVICE), ("hud.py", PY_MUT_HUD)])
    spec = CongruenceSpec(name="social", mirror_apply="apply_replication_snapshot",
                          guard="is_server")
    imp = impact_congruence(mods, spec)
    decline = next(m for m in imp.mutators if m.name == "decline_request")
    assert {w.name for w in decline.write_refs} == {"_pending"}
    assert "pending_for" not in {m.name for m in imp.mutators}
    assert decline.guarded and not decline.rpc_routed
    assert decline.is_gap and decline.gap_kind == "unrouted"
    assert {f.detail["func"] for f in check_congruence(mods, spec)} == {"decline_request"}
