"""Precision: false positives the naive function-local resolve heuristic raised.

#10a — one-hop interprocedural argument propagation. The function-local rule
("a read is covered iff a resolver is called in the SAME function") misses the
common hand-off shape: one function resolves the effective value and passes it to
a helper that applies it. The helper reads the base field too, but it is NOT a
modifier that fails to propagate — it received the resolved value through its
parameter. Each adapter (Python AST, GDScript regex) must recognise the hand-off
and stop flagging the helper, while a genuinely raw read (no resolution, no
resolved argument) still trips the gate.
"""
from __future__ import annotations

from saddle.codemap import gdref, pyref
from saddle.codemap.checks import check_value
from saddle.codemap.impact import impact_value
from saddle.codemap.specs import ValueSpec

SPEC = ValueSpec(name="cooldown", field="cooldown_s",
                 accessor="resolve_cd", producers=("build_def",))

# build_def = producer (builds base, exempt); resolve_cd = resolver (its own base
# read IS the resolution, exempt). cast resolves then hands `cd` to
# _commit_activation; cast_inline hands an inline resolve_cd(...) call to _apply —
# both helpers read the base raw but are covered one hop out. sweep reads it raw
# with no resolution and no resolved argument: the only true gap.
PY_SRC = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

def cast(inst):
    cd = resolve_cd(inst)
    _commit_activation(inst, cd)

def _commit_activation(inst, cd):
    schedule(inst["cooldown_s"])

def cast_inline(inst):
    _apply(inst, resolve_cd(inst))

def _apply(inst, cd):
    note(inst["cooldown_s"])

def sweep(inst):
    if inst["cooldown_s"] <= 0:
        expire(inst)
'''

GD_SRC = '''
func build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

func resolve_cd(inst):
    return inst["cooldown_s"] * 0.9

func cast(inst):
    var cd = resolve_cd(inst)
    _commit_activation(inst, cd)

func _commit_activation(inst, cd):
    schedule(inst["cooldown_s"])

func cast_inline(inst):
    _apply(inst, resolve_cd(inst))

func _apply(inst, cd):
    note(inst["cooldown_s"])

func sweep(inst):
    if inst["cooldown_s"] <= 0:
        expire(inst)
'''


def test_py_arg_propagation_kills_handoff_fp():
    mods = pyref.parse_modules([("abil.py", PY_SRC)])
    imp = impact_value(mods, SPEC)
    # the helpers that received the resolved value are now covered, not flagged
    assert imp.arg_covered_funcs == {"_commit_activation", "_apply"}
    assert {"_commit_activation", "_apply"} <= {r.func for r in imp.covered_reads}
    # only the genuinely raw read remains uncovered
    assert {r.func for r in imp.uncovered_reads} == {"sweep"}
    # the gate is still exactly the projection of the impact set
    assert check_value(mods, SPEC) == imp.gaps()
    assert {f.detail["func"] for f in check_value(mods, SPEC)} == {"sweep"}


def test_gd_arg_propagation_kills_handoff_fp():
    mods = gdref.parse_modules([("abil.gd", GD_SRC)])
    imp = impact_value(mods, SPEC)
    assert imp.arg_covered_funcs == {"_commit_activation", "_apply"}
    assert {"_commit_activation", "_apply"} <= {r.func for r in imp.covered_reads}
    assert {r.func for r in imp.uncovered_reads} == {"sweep"}
    assert check_value(mods, SPEC) == imp.gaps()
    assert {f.detail["func"] for f in check_value(mods, SPEC)} == {"sweep"}


def test_no_resolvers_is_noop():
    # An empty resolver set can't cover anything: the guard returns nothing and
    # no read is silently marked covered.
    assert pyref.resolved_arg_callees(pyref.parse_modules([("a.py", PY_SRC)])[0], set()) == set()
    assert gdref.resolved_arg_callees(gdref.parse_modules([("a.gd", GD_SRC)])[0], set()) == set()


# --- #10b qualified-carrier identity discrimination ----------------------
from saddle.codemap.checks import check_identity  # noqa: E402
from saddle.codemap.impact import impact_identity  # noqa: E402
from saddle.codemap.specs import IdentitySpec  # noqa: E402

# `kind` is overloaded: it carries the status namespace on `effect`, and an
# UNRELATED event namespace on `event`. A bare carrier flags "click" as drift; a
# qualified carrier `effect.kind` ignores the event namespace but still catches a
# real typo ("freeze") on the right object.
IDENT_PY = '''
STATUS = {"burn", "slow", "stun"}

def apply_status(effect):
    if effect.kind == "burn":
        ignite()
    elif effect.kind == "freeze":
        chill()

def route(event):
    if event.kind == "click":
        handle_click()
'''

IDENT_GD = '''
const STATUS = ["burn", "slow", "stun"]

func apply_status(effect):
    if effect.kind == "burn":
        ignite()
    elif effect.kind == "freeze":
        chill()

func route(event):
    if event.kind == "click":
        handle_click()
'''


def _qualified(carrier):
    return IdentitySpec(name="status_kind", canonical={"burn", "slow", "stun"},
                        source_symbol="STATUS", carriers={carrier})


def test_py_qualified_carrier_ignores_other_namespace():
    mods = pyref.parse_modules([("a.py", IDENT_PY)])
    # bare `kind` is the false positive: the unrelated "click" reads as drift
    bare = impact_identity(mods, _qualified("kind"))
    assert "click" in {lit for lit, _ in bare.drift_refs}
    # qualified `effect.kind` ignores event.kind, keeps the real "freeze" typo
    imp = impact_identity(mods, _qualified("effect.kind"))
    assert {lit for lit, _ in imp.drift_refs} == {"freeze"}
    assert {lit for lit, _ in imp.member_refs} == {"burn"}
    assert check_identity(mods, _qualified("effect.kind")) == imp.gaps()


def test_gd_qualified_carrier_ignores_other_namespace():
    mods = gdref.parse_modules([("a.gd", IDENT_GD)])
    bare = impact_identity(mods, _qualified("kind"))
    assert "click" in {lit for lit, _ in bare.drift_refs}
    imp = impact_identity(mods, _qualified("effect.kind"))
    assert {lit for lit, _ in imp.drift_refs} == {"freeze"}
    assert {lit for lit, _ in imp.member_refs} == {"burn"}
    assert check_identity(mods, _qualified("effect.kind")) == imp.gaps()
