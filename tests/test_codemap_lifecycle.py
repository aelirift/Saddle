"""Lifecycle: the dead-knob axis — a DECLARED symbol read by nobody.

This is the completeness gap ORTHOGONAL to value propagation. Propagation asks
whether a read sees the modifier; lifecycle asks whether the declaration has any
read at ALL. RayXI shipped a cluster of these: the status-effect
``max_stacks_per_kind`` / ``bleed_ignores_armor`` / cc-flag ``@export``s and the
loadout ``encoding`` export — every one declared (a designer could set it in the
inspector) and consumed by zero lines of code, so changing it did nothing.

The MiniMax audit surfaced them; this axis lets saddle find them mechanically.
Each adapter (Python AST, GDScript regex) must (a) see the declaration, (b)
over-count reads in the conservative direction so a live knob is never falsely
called dead, and (c) still flag a knob with genuinely zero reads. And the gate is
exactly the projection of the impact set: ``check_lifecycle == impact.gaps()``.
"""
from __future__ import annotations

from saddle.codemap import gdref, pyref
from saddle.codemap.checks import check_lifecycle
from saddle.codemap.impact import impact_lifecycle
from saddle.codemap.specs import LifecycleSpec

# MAX_STACKS is declared AND read (alive). BLEED_IGNORES_ARMOR and ENCODING are
# declared module constants that nothing reads — the dead knobs.
PY_SRC = '''
MAX_STACKS = 5
BLEED_IGNORES_ARMOR = True
ENCODING = "utf-8"

def apply_stack(n):
    return min(n, MAX_STACKS)
'''

# RayXI's real shape: @export knobs + a const + a signal. max_stacks_per_kind and
# stack_added are read/emitted (alive); bleed_ignores_armor and ENCODING are dead.
GD_SRC = '''
@export var max_stacks_per_kind: int = 3
@export var bleed_ignores_armor: bool = false
const ENCODING := "v2"
signal stack_added

func apply_stack(kind):
    if _count(kind) < max_stacks_per_kind:
        stack_added.emit()
'''


def _spec(symbol):
    return LifecycleSpec(name="knob", symbol=symbol)


def test_py_dead_export_is_flagged_and_live_one_is_not():
    mods = pyref.parse_modules([("status.py", PY_SRC)])
    # the live knob: declared and read -> no gap, and the read is seen
    alive = impact_lifecycle(mods, _spec("MAX_STACKS"))
    assert alive.decls and alive.uses
    assert alive.gaps() == []
    # the dead knobs: declared, read by nobody -> exactly one liveness finding each
    for dead in ("BLEED_IGNORES_ARMOR", "ENCODING"):
        imp = impact_lifecycle(mods, _spec(dead))
        assert imp.decls and not imp.uses
        gaps = imp.gaps()
        assert len(gaps) == 1
        assert gaps[0].node_kind == "lifecycle"
        assert gaps[0].detail["symbol"] == dead
        # the gate is exactly the projection of the impact set
        assert check_lifecycle(mods, _spec(dead)) == gaps


def test_gd_dead_export_is_flagged_and_live_ones_are_not():
    mods = gdref.parse_modules([("status.gd", GD_SRC)])
    # @export var read in a body, and a signal that is emitted -> both alive
    for live in ("max_stacks_per_kind", "stack_added"):
        imp = impact_lifecycle(mods, _spec(live))
        assert imp.decls and imp.uses, live
        assert imp.gaps() == []
    # the dead @export and the dead const -> flagged
    for dead in ("bleed_ignores_armor", "ENCODING"):
        imp = impact_lifecycle(mods, _spec(dead))
        assert imp.decls and not imp.uses, dead
        assert {f.detail["symbol"] for f in imp.gaps()} == {dead}
        assert check_lifecycle(mods, _spec(dead)) == imp.gaps()


def test_undeclared_symbol_is_silent_not_dead():
    # A spec naming a symbol that is declared NOWHERE has no declaration to be
    # dead — that's a spec/naming error, not a liveness gap, so the axis stays
    # quiet rather than crying about every typo in a manifest.
    py = pyref.parse_modules([("a.py", PY_SRC)])
    gd = gdref.parse_modules([("a.gd", GD_SRC)])
    for mods in (py, gd):
        imp = impact_lifecycle(mods, _spec("does_not_exist"))
        assert imp.decls == []
        assert imp.gaps() == []


def test_liveness_is_project_wide_declared_here_read_there():
    # Declared in one module, read in ANOTHER — alive. Liveness is a whole-project
    # question (a knob read by any module is live), so the impact unions across
    # modules before deciding, exactly like every other axis.
    decl = "SHARED_CAP = 9\n"
    use = "from x import SHARED_CAP\n\ndef f():\n    return SHARED_CAP\n"
    mods = pyref.parse_modules([("decl.py", decl), ("use.py", use)])
    imp = impact_lifecycle(mods, _spec("SHARED_CAP"))
    assert imp.decls and imp.uses
    assert imp.gaps() == []
