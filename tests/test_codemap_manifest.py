"""SurfaceManifest — the Layer 2 <-> Layer 3 bridge.

The manifest is what a design HANDS OVER: every value/identity/boundary it
commits to touch. It must survive a real JSON hop (it persists in Design.meta as
plain JSON) and round-trip back to the typed specs the gate runs — nobody
re-types the touchpoints. And its gate/format must run the design's OWN specs
against real code, so declaration names the thing while code proves coverage.
"""
from __future__ import annotations

import json

from saddle.codemap import (
    BoundarySpec,
    IdentitySpec,
    SurfaceManifest,
    ValueSpec,
    pyref,
)

VALUE_SRC = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"]

def cast(inst):
    cd = resolve_cd(inst)
    return cd

def sweep(inst):
    return inst["cooldown_s"] - 1   # UNCOVERED

def show_tooltip(inst):
    return inst.get("cooldown_s")   # UNCOVERED
'''


def test_manifest_round_trip_preserves_specs():
    m = SurfaceManifest(
        values=[ValueSpec(name="cooldown", field="cooldown_s",
                          accessor=("resolve_cd", "apply_modifiers"),
                          producers=("build_def",))],
        identities=[IdentitySpec(name="status_type",
                                 canonical={"burn", "slow", "stun"},
                                 source_symbol="STATUS",
                                 carriers={"kind", "type"})],
        boundaries=[BoundarySpec(name="health", key="health",
                                 replication_func="replicate")],
    )
    d = json.loads(json.dumps(m.to_dict()))          # the real Design.meta hop
    back = SurfaceManifest.from_dict(d)
    v = back.values[0]
    assert v.resolvers == ("resolve_cd", "apply_modifiers")  # tuple accessor kept
    assert v.producers == ("build_def",)
    assert back.identities[0].canonical == {"burn", "slow", "stun"}  # set from sorted list
    assert back.boundaries[0].replication_func == "replicate"
    assert back.to_dict() == m.to_dict()             # idempotent


def test_manifest_scalar_accessor_round_trips():
    m = SurfaceManifest(values=[ValueSpec(name="cd", field="cooldown_s",
                                          accessor="resolve_cd")])
    back = SurfaceManifest.from_dict(m.to_dict())
    assert back.values[0].resolvers == ("resolve_cd",)   # scalar normalises through resolvers
    assert back.values[0].producers == ()


def test_manifest_from_dict_tolerates_none_and_empty():
    assert SurfaceManifest.from_dict(None).is_empty()
    assert SurfaceManifest.from_dict({}).is_empty()
    assert SurfaceManifest().format([]) == "(empty manifest)"


def test_manifest_gate_and_format_run_designs_own_specs():
    mods = pyref.parse_modules([("abil.py", VALUE_SRC)])
    m = SurfaceManifest(values=[ValueSpec(name="cooldown", field="cooldown_s",
                                          accessor="resolve_cd",
                                          producers=("build_def",))])
    findings = m.gate(mods)
    assert {f.detail["func"] for f in findings} == {"sweep", "show_tooltip"}
    text = m.format(mods)
    assert "UNCOVERED reads" in text
    assert "cooldown" in text
