"""Completeness over SUBSTRATES other than code dataflow.

impact.py answers "does this value/identity/boundary propagate through the
CODE?" Two completeness axes live off the AST and belong here:

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

Both keep the project's discipline: the SPEC only names the thing; the gap is
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
from .specs import PersistenceSpec, ReferenceSpec


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
