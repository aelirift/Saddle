"""No stale map — the gate always reads the code as it stands NOW.

The exact hazard the design layer must never have: step one edits the code, step
two validates against a map cached from BEFORE the edit and so blesses code it
never saw. (That is half of RayXI's bug — the dataflow map went stale because
nothing re-derived it at the gate.)

The codemap forecloses it structurally: it persists only the design's declared
INTENT (the SurfaceManifest), never a parsed AST map, and re-derives the map from
disk on every gate. These tests prove it operationally — one manifest, declared
once; the code and the doc substrate edited BETWEEN gate runs; the findings track
the current bytes every time.
"""
from __future__ import annotations

from saddle.codemap import ReferenceSpec, SurfaceManifest, ValueSpec, refs

_CLEAN = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    return resolve_cd(inst)

def sweep(inst):
    cd = resolve_cd(inst)        # routed through the resolver -> covered
    return cd - 1
'''

# Same file, but sweep now reads the base raw — a modifier never reaches it.
_DRIFTED = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    return resolve_cd(inst)

def sweep(inst):
    return inst["cooldown_s"] - 1   # DRIFT: raw base read, modifier misses
'''

_MANIFEST = SurfaceManifest(
    values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
)


def _gate(root):
    # Exactly the two-step the CLI / design hand-over performs: parse the tree as it
    # stands now, THEN run the persisted intent against that fresh parse.
    return _MANIFEST.gate(refs.parse_project(root), root=root)


def test_gate_tracks_code_edited_between_runs(tmp_path):
    abil = tmp_path / "abil.py"
    abil.write_text(_CLEAN)
    assert _gate(tmp_path) == []                       # clean code -> no gaps

    abil.write_text(_DRIFTED)                          # step one: edit the code
    gaps = _gate(tmp_path)                             # step two: re-derive + gate
    assert {f.detail["func"] for f in gaps} == {"sweep"}
    assert gaps[0].check == "value_propagation"

    abil.write_text(_CLEAN)                            # fix it back
    assert _gate(tmp_path) == []                       # gap gone — same manifest, new code


def test_reference_substrate_tracks_edited_docs(tmp_path):
    (tmp_path / "abil.py").write_text(_CLEAN)
    docs = tmp_path / "docs"
    docs.mkdir()
    skill = docs / "skill.md"
    skill.write_text("# Skill\n")                      # cooldown_s not registered yet
    m = SurfaceManifest(
        references=[ReferenceSpec("cooldown", "cooldown_s", ("docs/*.md",))]
    )
    g1 = m.gate(refs.parse_project(tmp_path), root=tmp_path)
    assert [f.check for f in g1] == ["reference_presence"]

    skill.write_text("# Skill\n\nBase cooldown_s is shown here.\n")  # now register it
    g2 = m.gate(refs.parse_project(tmp_path), root=tmp_path)
    assert g2 == []                                    # the current file is read, not a cache


def test_parse_project_never_memoizes(tmp_path):
    """The contract the freshness guarantee rests on: two parses of changed bytes
    yield different maps. A future cache would break the guarantee silently — this
    locks the re-derive-every-time behaviour."""
    abil = tmp_path / "abil.py"
    abil.write_text("def f(inst):\n    return inst['cooldown_s']\n")
    before = refs.field_reads(refs.parse_project(tmp_path)[0], "cooldown_s")
    abil.write_text("def f(inst):\n    return 0\n")    # remove the read
    after = refs.field_reads(refs.parse_project(tmp_path)[0], "cooldown_s")
    assert len(before) == 1 and len(after) == 0
