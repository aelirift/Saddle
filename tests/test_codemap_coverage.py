"""The codemap's own anti-drift guard — the bug class saddle exists to prevent,
turned back on saddle itself.

RayXI shipped the talent bug because a map existed but was imported by nobody, and
because its gates checked a stale DECLARATION instead of the live code. The
codemap's defence against that is structural: every node KIND must be wired the
whole way through — a Spec to name it, an impact_* to fan it out, a check_* that
is exactly that impact's gaps(), a format_* to render it, an *Impact result type,
a public export so a gate can actually call it, and a SurfaceManifest field so a
design can hand it over. A kind wired through only SOME of those is exactly the
"half-wired map" that let the bug through.

These tests derive the kind set from the package's public surface (``__all__``)
and assert every seam agrees, so adding a sixth kind that forgets any one wiring
point fails the suite instead of silently shipping a map nobody runs. The one
place the canonical COUNT is asserted is here — change it on purpose, never by
drift.
"""
from __future__ import annotations

import json

import saddle.codemap as cm
from saddle.codemap import (
    AuthoritySpec,
    BindingSpec,
    BoundarySpec,
    IdentitySpec,
    LifecycleSpec,
    PersistenceSpec,
    ReferenceSpec,
    SurfaceManifest,
    ValueSpec,
)

EXPORTS = set(cm.__all__)


def _kinds() -> set[str]:
    """The authoritative kind set: one per exported ``check_*`` (the gate's view)."""
    return {n[len("check_"):] for n in EXPORTS if n.startswith("check_")}


def _singular(field: str) -> str:
    if field.endswith("ies"):
        return field[:-3] + "y"
    if field.endswith("s"):
        return field[:-1]
    return field


def test_kinds_are_the_expected_eight():
    # The ONE deliberate count. If a kind is added/removed this fails on purpose,
    # forcing every other seam below to be re-checked rather than drifting.
    assert _kinds() == {"value", "identity", "boundary", "reference",
                        "persistence", "lifecycle", "authority", "binding"}


def test_every_kind_is_exported_across_all_seams():
    """Each kind must expose all seven seams publicly — bound AND in ``__all__``,
    so a gate consumer can import it (RayXI's map-nobody-imported, made impossible)."""
    for k in _kinds():
        cap = k.capitalize()
        for name in (f"check_{k}", f"impact_{k}", f"format_{k}_impact",
                     f"{cap}Spec", f"{cap}Impact"):
            assert name in EXPORTS, f"{name} missing from codemap.__all__"
            assert hasattr(cm, name), f"{name} in __all__ but not bound in the package"
        assert callable(getattr(cm, f"check_{k}"))
        assert callable(getattr(cm, f"impact_{k}"))
        assert callable(getattr(cm, f"format_{k}_impact"))
        assert isinstance(getattr(cm, f"{cap}Spec"), type)
        assert isinstance(getattr(cm, f"{cap}Impact"), type)


def test_no_orphan_seams():
    """The reverse direction: every exported impact_/format_/Spec/Impact maps back
    to a known kind — no seam can exist for a kind the gate doesn't run."""
    kinds = _kinds()
    assert {n[len("impact_"):] for n in EXPORTS if n.startswith("impact_")} == kinds
    assert {n[len("format_"):-len("_impact")] for n in EXPORTS
            if n.startswith("format_") and n.endswith("_impact")} == kinds
    assert {n[:-len("Spec")].lower() for n in EXPORTS if n.endswith("Spec")} == kinds
    assert {n[:-len("Impact")].lower() for n in EXPORTS if n.endswith("Impact")} == kinds


def test_manifest_fields_match_the_kinds():
    """The design's hand-over surface must carry exactly the gate's kinds — a kind
    the gate runs but the manifest can't serialise could never be handed over."""
    fields = set(SurfaceManifest().to_dict())
    assert {_singular(f) for f in fields} == _kinds()


def test_manifest_round_trips_and_dispatches_one_spec_per_kind(tmp_path):
    """One spec of every kind survives the Design.meta JSON hop, and impacts()/gate()
    dispatch every kind without error — the manifest seam, end to end."""
    m = SurfaceManifest(
        values=[ValueSpec("v", "f", "r")],
        identities=[IdentitySpec("i", {"a"}, "S", {"k"})],
        boundaries=[BoundarySpec("b", "bk", "rep")],
        references=[ReferenceSpec("ref", "tok", ("docs/*.md",))],
        persistence=[PersistenceSpec("p", "pk", "save", "load")],
        lifecycle=[LifecycleSpec("l", "sym")],
        authority=[AuthoritySpec("a", "is_server", ("set_x",))],
        bindings=[BindingSpec("bind", "project.godot", {"ability": ("ability_",)},
                              compatible=(("ability", "card"),),
                              programmatic=("debug_toggle",))],
    )
    back = SurfaceManifest.from_dict(json.loads(json.dumps(m.to_dict())))
    d = back.to_dict()
    assert {k: len(v) for k, v in d.items()} == {
        "values": 1, "identities": 1, "boundaries": 1, "references": 1,
        "persistence": 1, "lifecycle": 1, "authority": 1, "bindings": 1,
    }
    # impacts() returns a fan-out bucket for every kind ...
    imps = back.impacts([], root=tmp_path)
    assert set(imps) == set(d)
    # ... and gate() dispatches every kind without raising (root present so the
    # file-substrate references actually run, not just the AST kinds).
    assert isinstance(back.gate([], root=tmp_path), list)
