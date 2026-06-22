"""refs / pyref / gdref — the symbol-inventory MENU and project walk.

These lock the grounding menu the surface stage feeds the LLM (so it names the
field the code really uses) and the deterministic, vendored-dir-excluding walk
that turns a project root into Modules. The menu is per-occurrence ranked so the
load-bearing symbols float to the top; the walk must never wander into .venv /
node_modules yet must keep a project that itself lives under a build/ dir.
"""
from __future__ import annotations

from pathlib import Path

from saddle.codemap import gdref, pyref, refs

# A Python ability module: cooldown_s is read 3 ways (two .get, one subscript);
# build_def constructs the base, resolve_cd resolves it, show_tooltip reads raw.
PY_SRC = '''
STATUS = {"burn", "slow", "stun"}


def build_def(d):
    return d["cooldown_s"]


def resolve_cd(d):
    return d.get("cooldown_s") * 0.9


def show_tooltip(d):
    return d.get("cooldown_s")
'''

# The GDScript twin: cooldown_s read twice (one .get, one subscript). The comment
# mentions ".mana" and "resolve_cd()" — both MUST be stripped before scanning, or
# prose leaks into the menu. resolve_cd is defined but never called.
GD_SRC = '''
const STATUS = ["burn", "slow", "stun"]

func resolve_cd(d):
    return d.get("cooldown_s") * 0.9   # cd = base.mana via resolve_cd()

func show_tooltip(d):
    var c = d["cooldown_s"]
    return c
'''


def test_pyref_symbols_separates_buckets():
    mod = pyref.parse_modules([("abil.py", PY_SRC)])[0]
    syms = pyref.symbols(mod)
    assert syms["fields"]["cooldown_s"] == 3        # 2x .get + 1x subscript
    assert "get" not in syms["fields"]              # a method call, not a field
    assert syms["calls"]["get"] == 2
    assert set(syms["funcs"]) == {"build_def", "resolve_cd", "show_tooltip"}
    assert set(syms["collections"]["STATUS"]) == {"burn", "slow", "stun"}


def test_gdref_symbols_strips_comments_and_defs():
    mod = gdref.parse_modules([("abil.gd", GD_SRC)])[0]
    syms = gdref.symbols(mod)
    assert syms["fields"]["cooldown_s"] == 2        # 1x .get + 1x subscript
    assert "mana" not in syms["fields"]             # comment ".mana" was stripped
    assert "resolve_cd" in syms["funcs"]
    assert "resolve_cd" not in syms["calls"]        # def head + comment call excluded
    assert set(syms["collections"]["STATUS"]) == {"burn", "slow", "stun"}


def test_refs_symbols_merge_and_determinism():
    mods = (pyref.parse_modules([("abil.py", PY_SRC)])
            + gdref.parse_modules([("abil.gd", GD_SRC)]))
    syms = refs.symbols(mods)
    assert syms.fields["cooldown_s"] == 5           # 3 python + 2 gdscript
    assert syms.collections["STATUS"] == ["burn", "slow", "stun"]  # union, sorted
    assert refs.symbols(mods).top() == refs.symbols(mods).top()    # deterministic
    assert next(iter(syms.top()["fields"])) == "cooldown_s"        # most load-bearing first


def test_refs_symbols_top_caps_and_ranks():
    s = refs.Symbols(
        fields={f"f{i}": i for i in range(50)},     # f49 has the highest count
        funcs={}, calls={},
        collections={f"c{i}": [] for i in range(30)},
    )
    top = s.top(fields=5, collections=10)
    assert list(top["fields"]) == ["f49", "f48", "f47", "f46", "f45"]  # ranked + capped
    assert len(top["collections"]) == 10            # capped


def test_project_files_excludes_vendored_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("y = 2\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("z = 3\n")
    found = refs.project_files(tmp_path)
    assert {Path(p).name for p in found} == {"a.py"}


def test_project_files_keeps_excluded_name_in_root_ancestry(tmp_path):
    root = tmp_path / "build" / "proj"              # 'build' is excluded — but in ANCESTRY
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "m.py").write_text("x = 1\n")
    found = refs.project_files(root)
    assert [Path(p).name for p in found] == ["m.py"]
