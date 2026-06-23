"""design.py surface stage — the Layer 2 <-> Layer 3 seam, end to end.

This is the talent-cooldown bug, reproduced and CAUGHT. A project reads a base
number (cooldown_s) raw in several places; the design declares it now varies, the
surface stage grounds that declaration in the project's REAL symbols, derives the
existing fan-out, and forces the body prompt to address every uncovered site. The
manifest then persists on the Design so the gate can re-run the design's OWN specs
against the code after implementation and reproduce exactly the gaps.

No real LLM and no embedder: a fixture caller dispatches canned replies by label,
and a stub DKB returns no knowledge — the test is about the surface wiring, not
retrieval.
"""
from __future__ import annotations

import asyncio
import json

from saddle.codemap import SurfaceManifest, refs
from saddle.context import Context
from saddle.design import design_for
from saddle.models import DESIGN_FINAL

# A project that reads cooldown_s raw in sweep + show_tooltip (UNCOVERED), resolves
# it in cast (covered), builds the base in build_def (producer), via resolve_cd.
ABIL_PY = '''
def build_def(d):
    return {"cooldown_s": d["cooldown_s"]}

def resolve_cd(inst):
    return inst["cooldown_s"] - inst.get("cd_reduction", 0)

def cast(inst):
    cd = resolve_cd(inst)
    return cd if inst["cooldown_s"] > 0 else 0

def sweep(inst):
    return inst["cooldown_s"] - 1     # UNCOVERED: modifier never reaches the sweep

def show_tooltip(inst):
    return inst.get("cooldown_s")     # UNCOVERED: description shows the raw base
'''


class FixtureCaller:
    """Records every prompt by label and returns a canned reply per stage."""

    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    async def __call__(self, system: str, prompt: str, *, json_mode: bool = False,
                       label: str = "") -> str:
        self.prompts[label] = prompt
        if label == "design/diagnose":
            return json.dumps({
                "problem": "cooldown_s is read raw across the kit",
                "approach": "route every read through the resolver",
                "risks": ["the server/client split hides half the misses"],
            })
        if label == "design/surface":
            # Grounded in the menu the stage handed us (asserted below).
            return json.dumps({
                "values": [{"name": "cooldown", "field": "cooldown_s",
                            "accessor": ["resolve_cd"], "producers": ["build_def"]}],
                "identities": [], "boundaries": [],
            })
        if label in ("design/body", "design/revise"):
            return "## Design\nRoute cast, sweep, and the tooltip through resolve_cd."
        if label == "design/audit":
            return json.dumps({"ok": True, "issues": []})
        if label == "design/index":
            return json.dumps({"summary": "Resolver-routed cooldown",
                               "satisfies": [], "avoids": [], "heeds": []})
        return json.dumps({"entries": []})


class StubDKB:
    """No knowledge — isolates the surface wiring from retrieval."""

    def search_knowledge(self, ctx, query, *, k=8, kinds=None):
        return []


def test_surface_grounds_design_and_persists_runnable_manifest(tmp_path):
    (tmp_path / "abil.py").write_text(ABIL_PY)
    caller = FixtureCaller()
    design = asyncio.run(design_for(
        goal="let players spend a point to reduce a skill's cooldown",
        ctx=Context(tenant="acme", project="game"),
        caller=caller,
        dkb=StubDKB(),
        directives=[],
        persist=False,
        harvest=False,
        surface=True,
        code_root=str(tmp_path),
        max_audits=1,
    ))

    # 1. the surface stage was grounded in the project's REAL symbol menu —
    #    the actual field name and resolver were put in front of the LLM.
    surface_prompt = caller.prompts["design/surface"]
    assert "cooldown_s" in surface_prompt          # real field, from the menu
    assert "resolve_cd" in surface_prompt          # real resolver, from the menu

    # 2. the existing fan-out was injected into the body prompt, naming every
    #    uncovered site so the design cannot stop at the symptom.
    body_prompt = caller.prompts["design/body"]
    assert "COMPLETENESS SURFACE" in body_prompt
    assert "UNCOVERED reads" in body_prompt
    assert "sweep()" in body_prompt
    assert "show_tooltip()" in body_prompt

    # 3. the manifest persisted on the Design (plain JSON in meta), grounded.
    surface = design.meta["surface"]
    assert surface["values"][0]["field"] == "cooldown_s"
    assert design.status == DESIGN_FINAL           # audit converged clean

    # 4. the persisted manifest survives the storage JSON hop and, re-run against
    #    the real code, reproduces EXACTLY the two uncovered sites — the gate the
    #    design handed over, not a re-typed one.
    rehydrated = SurfaceManifest.from_dict(json.loads(json.dumps(surface)))
    mods = refs.parse_project(tmp_path)
    gaps = rehydrated.gate(mods)
    assert {f.detail["func"] for f in gaps} == {"sweep", "show_tooltip"}
    assert all(f.check == "value_propagation" for f in gaps)


def test_surface_off_still_designs_without_code(tmp_path):
    """surface=False (or no code root) must no-op cleanly — a brand-new project
    with no code still produces a design; the surface just rides empty."""
    caller = FixtureCaller()
    design = asyncio.run(design_for(
        goal="add a cooldown reduction talent",
        ctx=Context(tenant="acme", project="fresh"),
        caller=caller,
        dkb=StubDKB(),
        directives=[],
        persist=False,
        harvest=False,
        surface=False,
        max_audits=1,
    ))
    assert "design/surface" not in caller.prompts   # stage skipped entirely
    assert design.meta["surface"] == {
        "values": [], "identities": [], "boundaries": [],
        "references": [], "persistence": [], "lifecycle": [], "authority": [],
        "bindings": [],
    }
    assert "COMPLETENESS SURFACE" not in caller.prompts["design/body"]
