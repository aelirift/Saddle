"""#11 — GDScript per-function domain from multiplayer-authority signals.

Godot doesn't split server/client by file path; it splits by multiplayer
authority. So the boundary check (a server-authoritative value must be packed
into the replication snapshot AND read by client code, or it's invisible on
screen) could never fire on GDScript while every .gd ref was "shared". This
derives the domain per function — `@rpc` annotation, `is_server()` guard, or a
replicated-state applier name — so boundary_mirror works on real Godot code.
"""
from __future__ import annotations

from saddle.codemap import gdref
from saddle.codemap.checks import check_boundary
from saddle.codemap.impact import impact_boundary
from saddle.codemap.specs import BoundarySpec

SPEC = BoundarySpec(name="health", key="health", replication_func="replicate")

# A complete loop: an authority-guarded tick WRITES health (server); replicate()
# packs it; an apply_*-named handler READS it onto the HUD (client).
GD_COMPLETE = '''
func _physics_process(delta):
    if not multiplayer.is_server():
        return
    self.health -= 1

func replicate():
    return {"health": self.health}

func apply_snapshot(snap):
    _hud.set_bar(snap["health"])
'''

# Same server write, but nothing packs it and no client reads it: two gaps.
GD_RAW = '''
func _physics_process(delta):
    if not multiplayer.is_server():
        return
    self.health -= 1
'''

# RPC-annotation path: @rpc authority = server sender, @rpc any_peer = client recv.
GD_RPC = '''
@rpc("authority")
func push_health():
    self.health -= 1

func replicate():
    return {"health": self.health}

@rpc("any_peer", "call_local")
func recv_health(v):
    _hud.set_bar(v if false else self.health)
'''


def test_authority_guard_makes_boundary_complete():
    mods = gdref.parse_modules([("unit.gd", GD_COMPLETE)])
    imp = impact_boundary(mods, SPEC)
    # the guarded tick is recognised as a server write
    assert {w.domain for w in imp.server_writes} == {"server"}
    assert {w.func for w in imp.server_writes} == {"_physics_process"}
    # replicate() packs it; apply_snapshot (client) reads it
    assert imp.packed
    assert {r.func for r in imp.client_reads} == {"apply_snapshot"}
    assert check_boundary(mods, SPEC) == []  # complete -> no gaps


def test_raw_server_write_flags_both_boundary_gaps():
    mods = gdref.parse_modules([("unit.gd", GD_RAW)])
    imp = impact_boundary(mods, SPEC)
    assert {w.domain for w in imp.server_writes} == {"server"}
    findings = check_boundary(mods, SPEC)
    assert findings == imp.gaps()
    # not packed + no client read = two boundary_mirror findings
    assert len(findings) == 2
    assert all(f.check == "boundary_mirror" for f in findings)


def test_rpc_annotations_set_domain():
    mods = gdref.parse_modules([("unit.gd", GD_RPC)])
    imp = impact_boundary(mods, SPEC)
    # @rpc("authority") is a server write; @rpc("any_peer") recv is a client read
    assert "push_health" in {w.func for w in imp.server_writes}
    assert {w.domain for w in imp.server_writes} == {"server"}
    assert "recv_health" in {r.func for r in imp.client_reads}
    assert imp.packed
    assert check_boundary(mods, SPEC) == []


def test_neutral_function_stays_shared():
    # No authority signal, no client-name => shared => not a server write, so the
    # boundary check stays silent (nothing authoritative to mirror).
    mods = gdref.parse_modules([("unit.gd", "func tick():\n    self.health -= 1\n")])
    imp = impact_boundary(mods, SPEC)
    assert imp.server_writes == []
    assert check_boundary(mods, SPEC) == []
