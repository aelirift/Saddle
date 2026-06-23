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
from saddle.codemap.checks import check_binding, check_persistence, check_reference
from saddle.codemap.manifest import SurfaceManifest
from saddle.codemap.specs import BindingSpec, PersistenceSpec, ReferenceSpec
from saddle.codemap.substrate import impact_binding
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


# --- #9c input_binding ----------------------------------------------------
def _ev_key(code: int) -> str:
    return (f'Object(InputEventKey,"resource_local_to_scene":false,"device":-1,'
            f'"pressed":false,"keycode":{code},"physical_keycode":0,"script":null)')


def _ev_pad(button: int) -> str:
    return (f'Object(InputEventJoypadButton,"resource_local_to_scene":false,'
            f'"device":-1,"button_index":{button},"pressure":0.0,"script":null)')


def _ev_mouse(button: int) -> str:
    # Faithful to Godot's serialization: the nested-paren Vector2(0, 0) fields land
    # BEFORE button_index. A naive `[^)]*` event regex stops at the inner ')' and
    # never sees button_index, so a mouse-only action parses as dead. This locks the
    # split-on-Object( parser that fixes that.
    return (f'Object(InputEventMouseButton,"resource_local_to_scene":false,'
            f'"device":-1,"position":Vector2(0, 0),"global_position":Vector2(0, 0),'
            f'"factor":1.0,"button_index":{button},"pressed":false,"script":null)')


def _keymap(actions: dict[str, list[str]]) -> str:
    """A minimal project.godot with just an [input] section. Each action maps to a
    list of pre-rendered event strings (empty list -> a dead action)."""
    lines = ["[application]", 'config/name="t"', "", "[input]", ""]
    for name, events in actions.items():
        lines.append(f'{name}={{"deadzone": 0.5, "events": [{", ".join(events)}]}}')
    return "\n".join(lines) + "\n"


# The RayXI bug, distilled: the number row '1' fires the first ability AND a menu
# panel AND a class-gated card. ability×card is context-exclusive (compatible);
# ability×panel is the gap.
_FAMILIES = {"ability": ("ability_",), "card": ("card_play_",),
             "panel": ("collection_panel", "world_map"),
             "move": ("forward", "jump"), "dismiss": ("cancel", "pause")}
_SPEC = BindingSpec(
    name="wow_keymap", keymap="project.godot", families=_FAMILIES,
    compatible=(("ability", "card"), ("dismiss", "dismiss")),
    programmatic=("debug_only",))


def test_binding_flags_number_row_panel_collision(tmp_path):
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49)],          # '1'
        "card_play_1": [_ev_key(49)],         # '1' — compatible with ability
        "collection_panel": [_ev_key(49)],    # '1' — the BUG: a panel on a cast key
        "forward": [_ev_key(87)],             # 'W' — clean
    }))
    findings = check_binding(tmp_path, _SPEC)
    assert findings == impact_binding(tmp_path, _SPEC).gaps()
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "input_binding" and f.node_kind == "binding"
    assert f.detail["trigger"] == "key:49"
    assert set(f.detail["actions"]) == {"ability_U", "card_play_1", "collection_panel"}
    # ability×card is tolerated; every flagged pair involves the panel
    pairs = [tuple(p) for p in f.detail["incompatible_pairs"]]
    assert pairs == [("ability_U", "collection_panel"), ("card_play_1", "collection_panel")]


def test_binding_same_family_overflow_collision(tmp_path):
    # Two DIFFERENT abilities on one key is a real gap — same-family co-binds are
    # not tolerated unless ("ability","ability") is declared compatible.
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_P": [_ev_key(55)],       # '7'
        "ability_DIGIT7": [_ev_key(55)],  # '7' — overflow ability re-uses the slot
    }))
    findings = check_binding(tmp_path, _SPEC)
    assert len(findings) == 1
    assert findings[0].detail["families"] == ["ability"]


def test_binding_ability_on_map_key(tmp_path):
    # The residual overflow case: ability_M and world_map both on 'M' (77).
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_M": [_ev_key(77)],
        "world_map": [_ev_key(77)],
    }))
    findings = check_binding(tmp_path, _SPEC)
    assert len(findings) == 1
    assert set(findings[0].detail["families"]) == {"ability", "panel"}


def test_binding_tolerates_compatible_and_dismiss_cluster(tmp_path):
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49)],
        "card_play_1": [_ev_key(49)],   # compatible pair -> no collision
        "cancel": [_ev_key(4194305)],   # Esc
        "pause": [_ev_key(4194305)],    # Esc — dismiss self-compatible
    }))
    assert check_binding(tmp_path, _SPEC) == []


def test_binding_dead_action_and_programmatic(tmp_path):
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49)],
        "settings": [],        # no trigger -> dead
        "debug_only": [],      # no trigger but declared programmatic -> exempt
    }))
    findings = check_binding(tmp_path, _SPEC)
    assert len(findings) == 1
    assert findings[0].detail == {"action": "settings", "reason": "no_trigger"}


def test_binding_gamepad_collision_detected(tmp_path):
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49), _ev_pad(2)],
        "collection_panel": [_ev_pad(2)],   # same pad button as an ability
    }))
    findings = check_binding(tmp_path, _SPEC)
    triggers = {f.detail["trigger"] for f in findings}
    assert "pad_btn:2" in triggers


def test_binding_mouse_button_parses_not_dead(tmp_path):
    # Regression: a mouse-only action carries the nested-paren Vector2(0, 0) fields
    # before button_index. It must yield a mouse trigger, never be miscounted dead.
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49)],
        "target_pick": [_ev_mouse(1)],        # mouse-only -> a real trigger, not dead
    }))
    imp = impact_binding(tmp_path, _SPEC)
    assert "target_pick" not in imp.dead_actions
    assert any(b.action == "target_pick" and b.trigger == "mouse:1"
               for b in imp.bindings)
    assert check_binding(tmp_path, _SPEC) == []


def test_binding_mouse_button_collision_detected(tmp_path):
    # And the trigger it produces still collides like any other: a panel sharing the
    # mouse button with an ability is the same gap class as a shared keycode.
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_mouse(1)],
        "collection_panel": [_ev_mouse(1)],
    }))
    findings = check_binding(tmp_path, _SPEC)
    assert len(findings) == 1
    assert findings[0].detail["trigger"] == "mouse:1"


def test_binding_missing_keymap_is_a_finding(tmp_path):
    findings = check_binding(tmp_path, _SPEC)  # no project.godot written
    assert len(findings) == 1
    assert "could not be read" in findings[0].message


def test_binding_manifest_gate_needs_root(tmp_path):
    (tmp_path / "project.godot").write_text(_keymap({
        "ability_U": [_ev_key(49)],
        "collection_panel": [_ev_key(49)],   # collision
    }))
    m = SurfaceManifest(bindings=[_SPEC])
    assert m.gate([]) == []                       # no root -> bindings skipped
    gaps = m.gate([], root=tmp_path)
    assert [g.check for g in gaps] == ["input_binding"]
