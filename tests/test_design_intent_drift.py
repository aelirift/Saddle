"""intent_drift — a design's own surface, re-run against the code that now exists.

A design enumerates its completeness surface ONCE, at design time, and hands it
forward (the SurfaceManifest in ``Design.meta``). ``intent_drift`` runs those
exact specs against a fresh parse of the project, so when the implementation
later diverges from what the design committed to, the design's OWN gate catches
it — declared intent vs. live code, derived every time, never a re-typed gate or
a cached map.
"""
from __future__ import annotations

from saddle.codemap import SurfaceManifest, ValueSpec, refs
from saddle.design import intent_drift
from saddle.models import Design

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


def _design_with_surface() -> Design:
    m = SurfaceManifest(
        values=[ValueSpec("cooldown", "cooldown_s", "resolve_cd", ("build_def",))]
    )
    return Design(ask="route every cooldown read through the resolver",
                  meta={"surface": m.to_dict()})


def test_intent_drift_catches_code_diverging_after_design(tmp_path):
    design = _design_with_surface()
    abil = tmp_path / "abil.py"

    abil.write_text(_CLEAN)
    assert intent_drift(design, root=tmp_path) == []        # code matches the intent

    abil.write_text(_DRIFTED)                               # implementation drifts later
    gaps = intent_drift(design, root=tmp_path)              # the design's own gate catches it
    assert {f.detail["func"] for f in gaps} == {"sweep"}
    assert all(f.check == "value_propagation" for f in gaps)

    abil.write_text(_CLEAN)
    assert intent_drift(design, root=tmp_path) == []        # drift resolved


def test_intent_drift_accepts_preparsed_mods(tmp_path):
    design = _design_with_surface()
    (tmp_path / "abil.py").write_text(_DRIFTED)
    mods = refs.parse_project(tmp_path)
    gaps = intent_drift(design, mods=mods)                  # gate a tree parsed elsewhere
    assert {f.detail["func"] for f in gaps} == {"sweep"}


def test_intent_drift_no_surface_or_no_code_is_noop(monkeypatch):
    monkeypatch.delenv("SADDLE_CODE_ROOT", raising=False)
    assert intent_drift(Design(ask="x")) == []                       # no surface at all
    assert intent_drift(Design(ask="x", meta={"surface": {}})) == []  # empty surface
    assert intent_drift(_design_with_surface()) == []                # surface, but no code to compare
