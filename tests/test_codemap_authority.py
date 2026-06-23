"""Authority: the write-side trust boundary — a mutator of server state that lets
a client call it ungated.

BoundarySpec covers the READ side (a server value must be mirrored to the client
to appear). This is the mirror on the WRITE side: a function that mutates
authoritative state must call an authority guard (`is_server` /
`is_multiplayer_authority` / a project `_is_server`) before it writes, or a client
can invoke it directly and desync / cheat. RayXI's talent/loadout mutators shipped
ungated — the MiniMax audit flagged it; this axis finds it mechanically.

A mutator with a guard call in its body is covered; a defined mutator with none is
the gap; a mutator the code never defines is named by nobody and stays silent. The
gate is exactly the projection of the impact set: ``check_authority == gaps()``.
"""
from __future__ import annotations

from saddle.codemap import gdref, pyref
from saddle.codemap.checks import check_authority
from saddle.codemap.impact import impact_authority
from saddle.codemap.specs import AuthoritySpec

# set_loadout guards with is_server; delete_loadout writes ungated -> the gap.
PY_SRC = '''
def is_server():
    return True

def set_loadout(player, data):
    if not is_server():
        return
    player["loadout"] = data

def delete_loadout(player):
    player["loadout"] = None
'''

# RayXI's real idiom: a project `_is_server()` helper gates the mutator.
GD_SRC = '''
func _is_server() -> bool:
    return multiplayer.is_server()

func set_loadout(player, data):
    if not _is_server():
        return
    player.loadout = data

func delete_loadout(player):
    player.loadout = null
'''


def test_py_ungated_mutator_is_flagged_guarded_one_is_not():
    mods = pyref.parse_modules([("loadout.py", PY_SRC)])
    spec = AuthoritySpec(name="loadout", guard="is_server",
                         mutators=("set_loadout", "delete_loadout"))
    imp = impact_authority(mods, spec)
    # both mutators are seen; only the guarded one is covered
    assert {r.name for r in imp.mutator_defs} == {"set_loadout", "delete_loadout"}
    assert imp.guarded_funcs == {"set_loadout"}
    gaps = imp.gaps()
    assert {f.detail["func"] for f in gaps} == {"delete_loadout"}
    assert gaps[0].node_kind == "authority"
    # the gate is exactly the projection of the impact set
    assert check_authority(mods, spec) == gaps


def test_gd_ungated_mutator_is_flagged_project_guard_idiom():
    mods = gdref.parse_modules([("loadout.gd", GD_SRC)])
    spec = AuthoritySpec(name="loadout", guard="_is_server",
                         mutators=("set_loadout", "delete_loadout"))
    imp = impact_authority(mods, spec)
    assert imp.guarded_funcs == {"set_loadout"}
    assert {f.detail["func"] for f in imp.gaps()} == {"delete_loadout"}
    assert check_authority(mods, spec) == imp.gaps()


def test_tuple_guard_either_accepted_form_covers():
    # A mutator guarded by ANY of the accepted guard functions is covered — real
    # codebases have more than one authority helper.
    src = '''
def is_server(): return True
def is_authority(): return True

def a(x):
    if not is_server(): return
    x["v"] = 1

def b(x):
    if not is_authority(): return
    x["v"] = 2

def c(x):
    x["v"] = 3
'''
    mods = pyref.parse_modules([("a.py", src)])
    spec = AuthoritySpec(name="state", guard=("is_server", "is_authority"),
                         mutators=("a", "b", "c"))
    imp = impact_authority(mods, spec)
    assert imp.guarded_funcs == {"a", "b"}
    assert {f.detail["func"] for f in imp.gaps()} == {"c"}


def test_undefined_mutator_is_silent_not_unguarded():
    # A mutator the code never defines is named by nobody — a spec/naming error,
    # not an authority gap. The axis stays quiet rather than flagging a phantom.
    py = pyref.parse_modules([("loadout.py", PY_SRC)])
    gd = gdref.parse_modules([("loadout.gd", GD_SRC)])
    for mods in (py, gd):
        spec = AuthoritySpec(name="x", guard="is_server",
                             mutators=("does_not_exist",))
        imp = impact_authority(mods, spec)
        assert imp.mutator_defs == []
        assert imp.gaps() == []


def test_all_mutators_guarded_is_clean():
    src = '''
def is_server(): return True

def set_a(x):
    if not is_server(): return
    x["a"] = 1

def set_b(x):
    if not is_server(): return
    x["b"] = 2
'''
    mods = pyref.parse_modules([("a.py", src)])
    spec = AuthoritySpec(name="state", guard="is_server",
                         mutators=("set_a", "set_b"))
    assert impact_authority(mods, spec).gaps() == []
    assert check_authority(mods, spec) == []
