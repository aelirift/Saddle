"""impact.py + checks.py — the impact set IS the gate.

The load-bearing invariant of this whole layer: the gate is a thin PROJECTION of
the impact set, so a map you read for "what else must change" and a gate that
blocks the commit can never drift apart (RayXI's fatal split — a dataflow map
nobody imported beside gates that only checked declarations). Every test asserts
``check_X(mods, spec) == impact_X(mods, spec).gaps()`` alongside the partition.
"""
from __future__ import annotations

from saddle.codemap import (
    BoundarySpec,
    IdentitySpec,
    ValueSpec,
    check_boundary,
    check_identity,
    check_value,
    impact_boundary,
    impact_identity,
    impact_value,
    pyref,
)

# build_def CONSTRUCTS the base (producer), resolve_cd RESOLVES it (accessor),
# cast resolves in scope (covered), sweep + show_tooltip read raw (uncovered).
VALUE_SRC = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    cd = resolve_cd(inst)          # resolver in scope -> raw read below is COVERED
    if inst["cooldown_s"] > 0:
        return cd

def sweep(inst):
    return inst["cooldown_s"] - 1  # UNCOVERED: modifier never reaches the sweep

def show_tooltip(inst):
    return inst.get("cooldown_s")  # UNCOVERED: skill description shows raw base
'''

IDENTITY_SRC = '''
STATUS = {"burn", "slow", "stun"}

def apply(inst, kind):
    if kind == "burn":     # member - fine
        return 1
    if kind == "freeze":   # DRIFT - not in canonical, an if-== form RayXI never saw
        return 2
'''

DUP_SRC = '''
STATUS = {"burn", "slow"}
STATUS = {"burn", "slow", "stun"}
'''

BOUND_SERVER = '''
# saddle-domain: server
def tick(self):
    self.health = self.health - 1   # authoritative write

def replicate(self):
    return {"health": self.health}  # packed into the snapshot
'''

BOUND_CLIENT = '''
# saddle-domain: client
def render(snap):
    return draw(snap["health"])     # lands on screen
'''

BOUND_SERVER_RAW = '''
# saddle-domain: server
def tick(self):
    self.health = self.health - 1   # written, but never packed and never mirrored
'''


def test_value_impact_partitions_and_gate_projects():
    mods = pyref.parse_modules([("abil.py", VALUE_SRC)])
    spec = ValueSpec(name="cooldown", field="cooldown_s",
                     accessor="resolve_cd", producers=("build_def",))
    imp = impact_value(mods, spec)
    assert {r.func for r in imp.uncovered_reads} == {"sweep", "show_tooltip"}
    assert {r.func for r in imp.covered_reads} == {"cast"}
    assert {r.func for r in imp.producer_reads} == {"build_def", "resolve_cd"}
    assert check_value(mods, spec) == imp.gaps()              # gate == projection
    gaps = check_value(mods, spec)
    assert {f.detail["func"] for f in gaps} == {"sweep", "show_tooltip"}
    assert all(f.check == "value_propagation" for f in gaps)


def test_identity_impact_flags_only_drift():
    mods = pyref.parse_modules([("st.py", IDENTITY_SRC)])
    spec = IdentitySpec(name="status_type", canonical={"burn", "slow", "stun"},
                        source_symbol="STATUS", carriers={"kind"})
    imp = impact_identity(mods, spec)
    assert {lit for lit, _ in imp.member_refs} == {"burn"}    # only USED members
    assert {lit for lit, _ in imp.drift_refs} == {"freeze"}
    assert check_identity(mods, spec) == imp.gaps()           # gate == projection
    gaps = check_identity(mods, spec)
    assert len(gaps) == 1 and gaps[0].detail["literal"] == "freeze"


def test_identity_impact_flags_duplicate_canonical_set():
    mods = pyref.parse_modules([("dup.py", DUP_SRC)])
    spec = IdentitySpec(name="status_type", canonical={"burn", "slow", "stun"},
                        source_symbol="STATUS", carriers={"kind"})
    gaps = check_identity(mods, spec)
    assert len(gaps) == 1
    assert "declared in 2 places" in gaps[0].message


def test_boundary_impact_complete_has_no_gaps():
    mods = (pyref.parse_modules([("server/state.py", BOUND_SERVER)])
            + pyref.parse_modules([("client/hud.py", BOUND_CLIENT)]))
    spec = BoundarySpec(name="health", key="health", replication_func="replicate")
    imp = impact_boundary(mods, spec)
    assert imp.packed is True
    assert {r.domain for r in imp.client_reads} == {"client"}
    assert check_boundary(mods, spec) == imp.gaps() == []     # gate == projection


def test_boundary_impact_unmirrored_flags_both_legs():
    mods = pyref.parse_modules([("server/state.py", BOUND_SERVER_RAW)])
    spec = BoundarySpec(name="health", key="health", replication_func="replicate")
    imp = impact_boundary(mods, spec)
    assert check_boundary(mods, spec) == imp.gaps()           # gate == projection
    msgs = " ".join(f.message for f in imp.gaps())
    assert "not read into 'replicate'" in msgs                # not packed into snapshot
    assert "no client-domain code reads it" in msgs           # never lands on screen
    assert all(f.check == "boundary_mirror" for f in imp.gaps())
