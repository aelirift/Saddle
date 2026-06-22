"""#9 — completeness over substrates other than code dataflow.

#9a reference_presence: a feature defined in code must also be REGISTERED in the
non-code substrates that declare it (config / docs / schema). #9b
persistence_symmetry: a value that must survive a session has to be referenced in
BOTH the save and the load function. Both keep the project invariant — the spec
only names the thing; the gap is derived from real files / real AST — and both
project cleanly: check_* IS impact_*.gaps().
"""
from __future__ import annotations

import json

from saddle.codemap import gdref, pyref
from saddle.codemap.checks import check_persistence, check_reference
from saddle.codemap.manifest import SurfaceManifest
from saddle.codemap.specs import PersistenceSpec, ReferenceSpec
from saddle.codemap.substrate import impact_persistence, impact_reference


# --- #9a reference_presence ----------------------------------------------
def test_reference_present_and_missing(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "config" / "abilities.json").write_text('{"firebolt": {"cd": 3}}')
    (tmp_path / "docs" / "spells.md").write_text("# Spells\n- frostbolt\n")  # no firebolt
    spec = ReferenceSpec(name="firebolt", key="firebolt",
                         substrates=("config/*.json", "docs/*.md"))
    imp = impact_reference(tmp_path, spec)
    assert {h.substrate: h.satisfied for h in imp.hits} == {
        "config/*.json": True, "docs/*.md": False}
    findings = check_reference(tmp_path, spec)
    assert findings == imp.gaps()
    assert len(findings) == 1
    assert findings[0].location == "docs/*.md"
    assert findings[0].check == "reference_presence"


def test_reference_whole_token_only(tmp_path):
    # `firebolt_v2` must NOT satisfy the token `firebolt`.
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "a.json").write_text('{"firebolt_v2": 1}')
    spec = ReferenceSpec(name="fb", key="firebolt", substrates=("config/*.json",))
    assert len(check_reference(tmp_path, spec)) == 1


def test_reference_absent_substrate(tmp_path):
    spec = ReferenceSpec(name="s", key="s", substrates=("schema/*.sql",))
    findings = check_reference(tmp_path, spec)
    assert len(findings) == 1
    assert "matched no files" in findings[0].message


def test_reference_skips_excluded_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.md").write_text("firebolt everywhere")
    spec = ReferenceSpec(name="fb", key="firebolt", substrates=("**/*.md",))
    # the only match is under an excluded dir -> treated as no files -> a gap
    assert len(check_reference(tmp_path, spec)) == 1


# --- #9b persistence_symmetry --------------------------------------------
PERSIST_PY = '''
def save(self):
    return {"gold": self.gold, "level": self.level}

def load(self, data):
    self.gold = data["gold"]
'''

PERSIST_GD = '''
func save(data):
    data["gold"] = gold
    data["level"] = level

func load(data):
    gold = data["gold"]
'''


def test_py_persistence_symmetry():
    mods = pyref.parse_modules([("p.py", PERSIST_PY)])
    gold = PersistenceSpec("gold", "gold", "save", "load")
    level = PersistenceSpec("level", "level", "save", "load")
    xp = PersistenceSpec("xp", "xp", "save", "load")
    assert check_persistence(mods, gold) == []          # both sides
    f = check_persistence(mods, level)                  # saved, never loaded
    assert f == impact_persistence(mods, level).gaps()
    assert len(f) == 1 and f[0].detail["side"] == "saved_not_loaded"
    g = check_persistence(mods, xp)                      # declared but absent
    assert len(g) == 1 and g[0].detail["side"] == "absent"


def test_py_persistence_loaded_not_saved():
    src = ('def save(self):\n    return {"gold": self.gold}\n\n'
           'def load(self, data):\n    self.gold = data["gold"]\n'
           '    self.mana = data["mana"]\n')
    mods = pyref.parse_modules([("p.py", src)])
    f = check_persistence(mods, PersistenceSpec("mana", "mana", "save", "load"))
    assert len(f) == 1 and f[0].detail["side"] == "loaded_not_saved"


def test_gd_persistence_symmetry():
    mods = gdref.parse_modules([("p.gd", PERSIST_GD)])
    assert check_persistence(mods, PersistenceSpec("gold", "gold", "save", "load")) == []
    f = check_persistence(mods, PersistenceSpec("level", "level", "save", "load"))
    assert len(f) == 1 and f[0].detail["side"] == "saved_not_loaded"


# --- manifest wiring ------------------------------------------------------
def test_manifest_roundtrip_includes_substrate_specs():
    m = SurfaceManifest(
        references=[ReferenceSpec("firebolt", "firebolt", ("config/*.json",))],
        persistence=[PersistenceSpec("gold", "gold", "save", "load")],
    )
    m2 = SurfaceManifest.from_dict(json.loads(json.dumps(m.to_dict())))
    assert not m2.is_empty()
    assert m2.references[0].substrates == ("config/*.json",)
    assert m2.persistence[0].save_func == "save"


def test_manifest_gate_references_need_root(tmp_path):
    (tmp_path / "abil.py").write_text(PERSIST_PY)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "spells.md").write_text("# Spells\n")  # firebolt missing
    mods = pyref.parse_paths([str(tmp_path / "abil.py")])
    m = SurfaceManifest(
        references=[ReferenceSpec("firebolt", "firebolt", ("docs/*.md",))],
        persistence=[PersistenceSpec("gold", "gold", "save", "load")],  # clean
    )
    # without a root, references can't be scanned and are skipped; persistence clean
    assert m.gate(mods) == []
    # with a root, the missing doc reference is gated
    gaps = m.gate(mods, root=tmp_path)
    assert [g.check for g in gaps] == ["reference_presence"]
