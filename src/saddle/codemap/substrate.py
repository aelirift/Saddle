"""Completeness over SUBSTRATES other than code dataflow.

impact.py answers "does this value/identity/boundary propagate through the
CODE?" Three completeness axes live off the AST and belong here:

  reference_presence    — a feature defined in code must also be REGISTERED in the
                          non-code substrates that declare it: a config file, the
                          docs, a DB schema. RayXI's talent cooldown lived in the
                          engine but never in the skill DESCRIPTION; that gap is
                          invisible to any AST check because the description isn't
                          code. A ReferenceSpec names the substrates (globs) the
                          key must appear in; a substrate with no match is a gap.

  persistence_symmetry  — a value that must survive a session must be referenced
                          in BOTH the save and the load function. Saved-but-not-
                          loaded resets every session; loaded-but-not-saved
                          restores garbage. This one IS code-derived (it reuses
                          field_reads/field_writes), but it's a save-file
                          completeness question, not a propagation one, so it sits
                          beside reference_presence rather than in impact.py.

  input_binding         — every physical input must do exactly ONE intended thing,
                          and every declared action must be reachable. The keymap
                          is a serialized engine resource (a Godot project.godot
                          ``[input]`` section), not code, so no AST check can see
                          when two binding mechanisms claim the same key — which is
                          exactly how RayXI shipped a build where the number row
                          opened store/social PANELS instead of casting abilities.
                          A BindingSpec names the keymap file and an intent-family
                          policy; a key firing two incompatible families (or an
                          action with no key at all) is the gap.

All three keep the project's discipline: the SPEC only names the thing; the gap is
derived from the actual files / actual AST, never from a second declaration that
can drift. And like the impact set, each ``gaps()`` IS the matching ``check_*``
in checks.py — one derivation, two readers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import refs
from .finding import Finding
from .pyref import Ref
from .refs import _EXCLUDE_DIRS
from .specs import BindingSpec, PersistenceSpec, ReferenceSpec


@dataclass
class SubstrateHit:
    """One declared substrate (a glob), the files it matched, and which of those
    actually carry the key."""
    substrate: str
    files: list[str] = field(default_factory=list)
    found_in: list[str] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return bool(self.found_in)


@dataclass
class ReferenceImpact:
    """Every substrate a reference key must live in, and where it was actually
    found — the complete 'where must this be registered' fan-out; the gaps are the
    substrates with no hit."""
    spec: ReferenceSpec
    hits: list[SubstrateHit] = field(default_factory=list)

    @property
    def missing(self) -> list[SubstrateHit]:
        return [h for h in self.hits if not h.satisfied]

    def gaps(self) -> list[Finding]:
        out: list[Finding] = []
        for h in self.missing:
            if not h.files:
                msg = (f"{self.spec.key!r} substrate {h.substrate!r} matched no "
                       f"files — the place this {self.spec.name} must be registered "
                       f"is absent")
            else:
                msg = (f"{self.spec.key!r} is not registered in any {h.substrate!r} "
                       f"file ({len(h.files)} scanned) — defined in code but missing "
                       f"from this substrate, so the feature is half-wired")
            out.append(Finding(
                check="reference_presence",
                severity="error",
                node_kind="reference",
                thing=self.spec.name,
                message=msg,
                location=h.substrate,
                detail={"substrate": h.substrate, "files": h.files},
            ))
        return out


@dataclass
class PersistenceImpact:
    """Where a persisted key is referenced on each side of the round-trip. A gap
    is an asymmetry (one side references it, the other doesn't) or total absence
    (the design declared it persists but neither side touches it)."""
    spec: PersistenceSpec
    save_refs: list[Ref] = field(default_factory=list)
    load_refs: list[Ref] = field(default_factory=list)

    def gaps(self) -> list[Finding]:
        s = self.spec
        saved, loaded = bool(self.save_refs), bool(self.load_refs)
        if saved and loaded:
            return []
        if saved and not loaded:
            return [Finding(
                check="persistence_symmetry", severity="error",
                node_kind="persistence", thing=s.name,
                message=(f"{s.key!r} is written in {s.save_func!r} but never read "
                         f"back in {s.load_func!r} — it silently resets every "
                         f"session"),
                location=self.save_refs[0].location,
                detail={"side": "saved_not_loaded", "load_func": s.load_func})]
        if loaded and not saved:
            return [Finding(
                check="persistence_symmetry", severity="error",
                node_kind="persistence", thing=s.name,
                message=(f"{s.key!r} is read in {s.load_func!r} but never written "
                         f"in {s.save_func!r} — it restores a default/garbage "
                         f"value"),
                location=self.load_refs[0].location,
                detail={"side": "loaded_not_saved", "save_func": s.save_func})]
        return [Finding(
            check="persistence_symmetry", severity="error",
            node_kind="persistence", thing=s.name,
            message=(f"{s.key!r} is declared to persist but neither {s.save_func!r} "
                     f"nor {s.load_func!r} references it — not wired at all"),
            location=f"persistence:{s.key}",
            detail={"side": "absent", "save_func": s.save_func,
                    "load_func": s.load_func})]


def _excluded(root: Path, p: Path) -> bool:
    try:
        rel = p.relative_to(root).parts
    except ValueError:
        return False
    return any(part in _EXCLUDE_DIRS for part in rel)


def impact_reference(root, spec: ReferenceSpec) -> ReferenceImpact:
    root = Path(root)
    token = re.compile(r"\b%s\b" % re.escape(spec.key))
    imp = ReferenceImpact(spec=spec)
    for sub in spec.substrates:
        files = sorted(
            str(p) for p in root.glob(sub)
            if p.is_file() and not _excluded(root, p)
        )
        found = [f for f in files if token.search(
            Path(f).read_text(encoding="utf-8", errors="replace"))]
        imp.hits.append(SubstrateHit(sub, files, found))
    return imp


def impact_persistence(mods: list, spec: PersistenceSpec) -> PersistenceImpact:
    imp = PersistenceImpact(spec=spec)
    for m in mods:
        touches = refs.field_reads(m, spec.key) + refs.field_writes(m, spec.key)
        for r in touches:
            if r.func == spec.save_func:
                imp.save_refs.append(r)
            elif r.func == spec.load_func:
                imp.load_refs.append(r)
    return imp


def format_reference_impact(imp: ReferenceImpact) -> str:
    s = imp.spec
    out = [f"REFERENCE {s.name!r}  (key {s.key!r})"]
    for h in imp.hits:
        mark = "OK" if h.satisfied else "MISSING"
        out.append(f"  [{mark}] {h.substrate}  ({len(h.found_in)}/{len(h.files)} file(s))")
        for f in h.found_in:
            out.append(f"      {f}")
    return "\n".join(out)


def format_persistence_impact(imp: PersistenceImpact) -> str:
    s = imp.spec
    return "\n".join([
        f"PERSISTENCE {s.name!r}  (key {s.key!r})",
        f"  saved in {s.save_func!r}: {len(imp.save_refs)} ref(s)",
        f"  loaded in {s.load_func!r}: {len(imp.load_refs)} ref(s)",
    ])


# === input_binding: the keymap substrate ====================================
# The third off-AST axis. The keymap is a serialized engine resource: parse it,
# fan every action out to the physical triggers that fire it, then project two gap
# kinds — a trigger firing incompatible intent families (a COLLISION, the number-
# row-opens-a-panel bug) and an action with no trigger at all (a DEAD binding).

# Godot keycodes are ASCII for the printable range, plus named keys in the
# 0x4000000 special block. This label table is the ENGINE's constants, not any
# game's vocabulary — it only makes a finding read "key '1'" instead of "key 49";
# the gap detection never depends on it.
_SPECIAL_KEYS = {
    4194305: "Escape", 4194306: "Tab", 4194308: "Enter", 4194309: "KpEnter",
    4194310: "Insert", 4194312: "Delete", 4194319: "Home", 4194321: "End",
    4194320: "PageUp", 4194322: "PageDown", 4194325: "Shift", 4194326: "Ctrl",
    4194327: "Meta", 4194328: "Alt", 16777234: "Left", 16777235: "Right",
    16777236: "Up", 16777237: "Down",
}


def _key_label(keycode: int) -> str:
    if keycode in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[keycode]
    if 32 <= keycode < 127:
        return chr(keycode)
    return f"#{keycode}"


def _trigger_label(trigger: str) -> str:
    kind, _, raw = trigger.partition(":")
    if kind == "key":
        return f"key '{_key_label(int(raw))}'"
    if kind == "mouse":
        return f"mouse button {raw}"
    if kind == "pad_btn":
        return f"gamepad button {raw}"
    if kind == "pad_axis":
        return f"gamepad axis {raw}"
    return trigger


@dataclass(frozen=True)
class TriggerBinding:
    """One physical input that fires one action — the atomic edge of the keymap
    fan-out. ``trigger`` is a stable id like ``key:49`` / ``mouse:1`` /
    ``pad_btn:3`` / ``pad_axis:5+`` so two actions collide iff their triggers are
    equal."""
    action: str
    family: str
    trigger: str
    location: str


@dataclass
class BindingImpact:
    """The complete 'what does each physical input do, and what can't be reached'
    fan-out for one keymap. Gaps are the trigger-level collisions and the
    unreachable (dead) actions — projected by :meth:`gaps`, which IS
    ``check_binding``."""
    spec: BindingSpec
    bindings: list[TriggerBinding] = field(default_factory=list)
    dead_actions: list[str] = field(default_factory=list)
    keymap_path: str = ""
    parse_error: str | None = None

    def by_trigger(self) -> dict[str, list[TriggerBinding]]:
        out: dict[str, list[TriggerBinding]] = {}
        for b in self.bindings:
            out.setdefault(b.trigger, []).append(b)
        return out

    def _collision_pairs(self, group: list[TriggerBinding]) -> list[tuple[str, str]]:
        """The (action, action) pairs on one trigger whose families aren't declared
        compatible. Distinct actions only — the same action listing a trigger twice
        is not a collision."""
        compat = self.spec.compatible_set
        actions = sorted({b.action: b for b in group}.values(), key=lambda b: b.action)
        bad: list[tuple[str, str]] = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                a, b = actions[i], actions[j]
                if frozenset((a.family, b.family)) in compat:
                    continue
                bad.append((a.action, b.action))
        return bad

    def gaps(self) -> list[Finding]:
        if self.parse_error is not None:
            return [Finding(
                check="input_binding", severity="error", node_kind="binding",
                thing=self.spec.name,
                message=(f"keymap {self.keymap_path or self.spec.keymap!r} could not "
                         f"be read: {self.parse_error} — the binding surface cannot "
                         f"be verified"),
                location=self.keymap_path or self.spec.keymap,
                detail={"parse_error": self.parse_error})]
        out: list[Finding] = []
        for trigger, group in sorted(self.by_trigger().items()):
            bad = self._collision_pairs(group)
            if not bad:
                continue
            fams = sorted({b.family for b in group})
            actions = sorted({b.action for b in group})
            out.append(Finding(
                check="input_binding", severity="error", node_kind="binding",
                thing=self.spec.name,
                message=(f"{_trigger_label(trigger)} fires {len(actions)} actions of "
                         f"incompatible intent ({', '.join(fams)}): "
                         f"{', '.join(actions)} — pressing it does the wrong thing / "
                         f"two things at once"),
                location=f"{self.keymap_path}:{trigger}",
                detail={"trigger": trigger, "actions": actions, "families": fams,
                        "incompatible_pairs": bad}))
        for action in sorted(self.dead_actions):
            out.append(Finding(
                check="input_binding", severity="error", node_kind="binding",
                thing=self.spec.name,
                message=(f"action {action!r} is declared with no trigger — it can "
                         f"never be invoked (a dead binding; add a key, or fire it "
                         f"in code and list it as programmatic)"),
                location=f"{self.keymap_path}:{action}",
                detail={"action": action, "reason": "no_trigger"}))
        return out


# Godot project.godot [input] entries look like:
#   action_name={"deadzone": 0.5, "events": [Object(InputEventKey,...,"keycode":77,...),
#                                             Object(InputEventJoypadButton,...,"button_index":3,...)]}
_SECTION_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
_ACTION_RE = re.compile(r"^(?P<action>[A-Za-z_][A-Za-z0-9_/]*)=\{(?P<body>.*)\}\s*$")
_KEYCODE_RE = re.compile(r'"keycode"\s*:\s*(-?\d+)')
_PHYS_KEYCODE_RE = re.compile(r'"physical_keycode"\s*:\s*(-?\d+)')
_BUTTON_RE = re.compile(r'"button_index"\s*:\s*(-?\d+)')
_AXIS_RE = re.compile(r'"axis"\s*:\s*(-?\d+)')
_AXIS_VALUE_RE = re.compile(r'"axis_value"\s*:\s*(-?[\d.]+)')


def _read_input_section(text: str) -> list[tuple[str, str]] | None:
    """The ``(action, body)`` pairs under ``[input]``. ``None`` when the file has
    no ``[input]`` section at all (nothing to gate)."""
    in_section = False
    seen = False
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            in_section = m.group("name").strip() == "input"
            seen = seen or in_section
            continue
        if not in_section:
            continue
        a = _ACTION_RE.match(line)
        if a:
            out.append((a.group("action"), a.group("body")))
    return out if seen else None


def _triggers_in_body(body: str) -> list[str]:
    """Every physical trigger id declared in one action's events body. Splits on
    ``Object(`` so each chunk holds exactly one event's fields — robust to nested
    parens like ``Vector2(0, 0)`` inside a mouse event (a naive ``[^)]*`` stops at
    that inner paren and loses the ``button_index`` that follows). An empty events
    list yields ``[]`` (a dead candidate)."""
    triggers: list[str] = []
    for chunk in body.split("Object(")[1:]:
        etype = chunk.split(",", 1)[0].strip()
        if etype == "InputEventKey":
            kc = _KEYCODE_RE.search(chunk)
            code = int(kc.group(1)) if kc else 0
            if code == 0:
                pk = _PHYS_KEYCODE_RE.search(chunk)
                code = int(pk.group(1)) if pk else 0
            if code:
                triggers.append(f"key:{code}")
        elif etype == "InputEventMouseButton":
            b = _BUTTON_RE.search(chunk)
            if b:
                triggers.append(f"mouse:{b.group(1)}")
        elif etype == "InputEventJoypadButton":
            b = _BUTTON_RE.search(chunk)
            if b:
                triggers.append(f"pad_btn:{b.group(1)}")
        elif etype == "InputEventJoypadMotion":
            ax = _AXIS_RE.search(chunk)
            if ax:
                val = _AXIS_VALUE_RE.search(chunk)
                sign = "-" if (val and float(val.group(1)) < 0) else "+"
                triggers.append(f"pad_axis:{ax.group(1)}{sign}")
    return triggers


def impact_binding(root, spec: BindingSpec) -> BindingImpact:
    """Parse ``root/spec.keymap`` and fan every action out to its physical
    triggers, classifying each by intent family. A missing or section-less file
    surfaces as a parse_error rather than raising — a half-wired map is a finding,
    not a crash."""
    path = Path(root) / spec.keymap
    imp = BindingImpact(spec=spec, keymap_path=str(path))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        imp.parse_error = type(e).__name__
        return imp
    entries = _read_input_section(text)
    if entries is None:
        imp.parse_error = "no [input] section"
        return imp
    programmatic = frozenset(spec.programmatic)
    for action, body in entries:
        triggers = _triggers_in_body(body)
        if not triggers:
            if action not in programmatic:
                imp.dead_actions.append(action)
            continue
        fam = spec.family_of(action)
        for t in dict.fromkeys(triggers):  # dedupe within one action
            imp.bindings.append(TriggerBinding(action, fam, t, f"{path}:{action}"))
    return imp


def format_binding_impact(imp: BindingImpact) -> str:
    s = imp.spec
    out = [f"BINDING {s.name!r}  (keymap {s.keymap!r})"]
    if imp.parse_error:
        out.append(f"  [UNREADABLE] {imp.parse_error}")
        return "\n".join(out)
    for trigger, group in sorted(imp.by_trigger().items()):
        actions = sorted({b.action for b in group})
        bad = imp._collision_pairs(group)
        if len(actions) > 1 or bad:
            out.append(f"  [{'COLLIDE' if bad else 'ok'}] "
                       f"{_trigger_label(trigger)} -> {', '.join(actions)}")
    for action in sorted(imp.dead_actions):
        out.append(f"  [DEAD] {action} (no trigger)")
    if len(out) == 1:
        out.append("  (no collisions, no dead actions)")
    return "\n".join(out)
