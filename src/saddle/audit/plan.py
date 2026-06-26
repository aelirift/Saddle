"""The audit PLAN — saddle's enumeration of everything that CAN be audited.

WHY THIS EXISTS
---------------
The mandate is not "audit some things" but "know everything that can be audited
and verify each one actually ran." That is a COVERAGE problem, and coverage is a
registry, not tribal knowledge: if the set of audit targets lives only in a
human's head (or a prompt), nobody can prove the audit was exhaustive.

An :class:`AuditPlan` is that registry, DERIVED from the target project's own
shape rather than hardcoded (so it stays game-agnostic — it works on any repo):

  - registry targets — each string-keyed JSON contract (``knowledge/**/*.json``,
    ``config/*.json``): does code implement every key it declares?
  - doc targets — each contract doc (``*.md``, ``docs/**/*.md``): does the code
    match what the doc claims is true?
  - package targets — each top-level code package: is it complete, or orphaned
    scaffolding nothing consumes?
  - concern targets — cross-cutting probes that span the whole tree (client/server
    congruence, user-facing UX/menus, dead declarations). These are generic
    mechanic-level concerns, never genre-specific.

Coverage is then trivially checkable: ``ran`` targets vs ``planned`` targets, with
the un-run ones named. Nothing is "done" until every planned target has a verdict.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

# Generic target kinds (the SOURCE that seeded the target) ----------------
REGISTRY = "registry"    # a string-keyed JSON contract
DOC = "doc"              # a markdown contract doc
PACKAGE = "package"      # a top-level code package
CONCERN = "concern"      # a cross-cutting probe spanning the whole tree
TARGET_KINDS: frozenset[str] = frozenset({REGISTRY, DOC, PACKAGE, CONCERN})

# Directories never worth enumerating (mirrors codemap.refs, plus data junk).
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", "dist", ".tox",
    "build", "target", ".cache", "coverage", "htmlcov",
})

# Auto-detected defaults — generic heuristics, no project-specific names baked in.
_DEFAULT_DOC_GLOBS = ["*.md", "docs/**/*.md", "doc/**/*.md"]
_DEFAULT_REGISTRY_GLOBS = [
    "knowledge/**/*.json", "config/**/*.json", "data/**/*.json",
    "schemas/**/*.json", "registry/**/*.json",
]
_DEFAULT_CODE_DIRS = ["src", "app", "apps", "lib", "pkg"]

# The generic cross-cutting concerns. Each is a mechanic-level question that holds
# for ANY genre — never "are the abilities right" but "does state the server owns
# reach the client", "can the player actually click what the screen renders".
_DEFAULT_CONCERNS: list[tuple[str, str]] = [
    (
        "client_server_congruence",
        "Find every feature where authoritative state and its mirror disagree, in "
        "EITHER direction. (a) Server writes the client never sees: a value the "
        "server mutates with no replication into the client snapshot, an @rpc the "
        "client calls that the server never defines, a HUD that reads a field the "
        "server never sends. (b) The opposite and just-as-broken shape — a "
        "client-side UI that WRITES authoritative state DIRECTLY: it calls a mutator "
        "on a service that DOES ship a client mirror, but the mutator is not "
        "gated to the server and there is no intent/request routed to the server, so "
        "the local change silently diverges and is reverted the instant the next "
        "server snapshot arrives. Name the feature, the write site, and the missing "
        "gate-or-mirror. A system that is server-only BY DESIGN (no client reader, "
        "deliberately excluded from the client scene — e.g. interest-management, "
        "anti-cheat, server AI) is NOT a gap; only flag a real divergence. Not just "
        "menus, EVERY feature.",
    ),
    (
        "menus_ux_reachable",
        "Find every user-facing panel/menu/HUD that renders but does not WORK: a "
        "panel drawn with a semi-transparent or missing background, an interactive "
        "panel whose buttons cannot receive the click (too-low CanvasLayer.layer, "
        "a higher layer eating the mouse), a declared input action bound to no key "
        "or a key that fires two conflicting actions, a screen hardcoded in pixels "
        "instead of anchored/percentage. Cite the scene/script site.",
    ),
    (
        "dead_declarations",
        "Find every DECLARED knob that nothing reads: an exported setting, a "
        "registry field, a config value, a signal, a constant that is defined and "
        "looks adjustable but is consumed by no code path — so changing it changes "
        "nothing. Cite the declaration and confirm the absence of a consumer.",
    ),
]


@dataclass(frozen=True)
class AuditTarget:
    """One unit of audit work — a thing that CAN be audited, with the grounding the
    probe needs to do it. ``id`` is stable across runs so coverage reconciles."""

    id: str
    kind: str                 # registry | doc | package | concern
    title: str
    question: str             # the specific audit question for this target
    paths: list[str] = field(default_factory=list)   # grounding files, relative to root
    source: str = ""          # the registry/doc/dir that seeded this target
    seeds: list[str] = field(default_factory=list)   # optional regex seeds to ground a
    #                          cross-cutting concern (e.g. the engine's replication /
    #                          authority tokens) — generic concerns ship empty; the
    #                          caller supplies project-specific seeds, so no engine
    #                          vocabulary is baked into saddle.

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "title": self.title,
            "question": self.question, "paths": list(self.paths),
            "source": self.source, "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AuditTarget":
        return cls(
            id=str(d.get("id", "")).strip(),
            kind=str(d.get("kind", "")).strip(),
            title=str(d.get("title", "")).strip(),
            question=str(d.get("question", "")).strip(),
            paths=[str(p) for p in d.get("paths", [])],
            source=str(d.get("source", "")).strip(),
            seeds=[str(s) for s in d.get("seeds", [])],
        )


@dataclass
class AuditPlan:
    """The full set of audit targets for one project root — the coverage registry.

    ``code_dirs`` records the SOURCE roots the audit reasons over (``src``/``apps``/
    …) so the driver parses the system under audit, NOT the build OUTPUT a pipeline
    scatters across the tree — see :func:`saddle.audit.ground.source_files`."""

    root: str
    targets: list[AuditTarget] = field(default_factory=list)
    code_dirs: list[str] = field(default_factory=list)
    # The project's string-keyed DATA files (knowledge/config/data/schemas/…). The
    # driver builds the data-field menu from these so doc/package probes can check a
    # doc-claimed contract field against the DATA it actually lives in, not only the
    # code symbol menu — the false-`absent` class for data-driven architectures.
    data_files: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[AuditTarget]:
        return [t for t in self.targets if t.kind == kind]

    def ids(self) -> set[str]:
        return {t.id for t in self.targets}

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "code_dirs": list(self.code_dirs),
            "data_files": list(self.data_files),
            "targets": [t.to_dict() for t in self.targets],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AuditPlan":
        return cls(
            root=str(d.get("root", "")),
            targets=[AuditTarget.from_dict(t) for t in d.get("targets", [])],
            code_dirs=[str(c) for c in d.get("code_dirs", [])],
            data_files=[str(p) for p in d.get("data_files", [])],
        )


def _rglob_globs(root: Path, globs: list[str]) -> list[str]:
    """Resolve a list of globs (relative to root) into a sorted, de-duped, skip-
    filtered list of relative file paths. Stable order ⇒ deterministic plan."""
    seen: set[str] = set()
    for g in globs:
        for p in root.glob(g):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            seen.add(str(rel))
    return sorted(seen)


def _is_string_keyed(path: Path) -> bool:
    """A registry worth a target is a JSON OBJECT (string keys) or a list of
    objects — a contract, not a bare scalar/array of primitives."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable / non-JSON ⇒ not a registry target
        return False
    if isinstance(data, dict):
        return bool(data)
    if isinstance(data, list):
        return any(isinstance(x, dict) for x in data)
    return False


_SRC_EXTS = (".py", ".gd", ".ts", ".tsx", ".go", ".js", ".rs", ".java")


def _source_dirs(base: Path) -> list[Path]:
    """Immediate child directories of ``base`` that (recursively) hold source."""
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name in _SKIP_DIRS:
            continue
        if any(f.suffix.lower() in _SRC_EXTS for f in child.rglob("*") if f.is_file()):
            out.append(child)
    return out


def _top_packages(root: Path, code_dirs: list[str]) -> list[tuple[str, str]]:
    """(package-id, relative-dir) for each real code package under the code dirs.

    Descends through a SINGLE project-namespace level (``src/`` that holds only
    ``rayxi/``) so the actual subsystems become targets, not the one namespace
    blob — but stops as soon as a level fans out into several sibling packages."""
    out: list[tuple[str, str]] = []
    for cd in code_dirs:
        base = root / cd
        if not base.is_dir():
            continue
        children = _source_dirs(base)
        depth = 0
        while len(children) == 1 and depth < 2:
            inner = _source_dirs(children[0])
            if not inner:
                break
            children = inner
            depth += 1
        for child in children:
            rel = str(child.relative_to(root))
            out.append((rel, rel))
    return out


def _collapse_groups(
    rels: list[str], *, threshold: int, sample: int, depth: int = 2,
) -> list[tuple[str, list[str], str]]:
    """Group files by their ancestor directory truncated to ``depth`` segments; a
    group holding MORE than ``threshold`` files collapses to ONE 'family' target
    (grounded on an evenly-spread ``sample`` of its files), while smaller groups
    stay one-target-per-file.

    Grouping by a SHALLOW ancestor (not the immediate parent) is what keeps a
    coverage plan meaningful on a repo that scatters hundreds of GENERATED
    instances across deep, timestamped sub-trees (rayxi's
    ``knowledge/level/reports/<game>/<run>/*.json``): the whole ``knowledge/level``
    tree is audited as ONE family, instead of ~200 near-identical probes drowning
    the real contract + package targets. Returns (group-id, paths, source-label)."""
    by_group: dict[str, list[str]] = {}
    for r in rels:
        parts = Path(r).parent.parts[:max(1, depth)]
        key = "/".join(parts) if parts else "."
        by_group.setdefault(key, []).append(r)
    out: list[tuple[str, list[str], str]] = []
    for key in sorted(by_group):
        members = sorted(by_group[key])
        if len(members) > threshold:
            step = max(1, len(members) // max(1, sample))
            picked = members[::step][:sample]
            fam = "./" if key == "." else f"{key}/"
            out.append((fam, picked, f"{fam} ({len(members)} files, {len(picked)} sampled)"))
        else:
            out.extend((m, [m], m) for m in members)
    return out


def build_plan(
    root: str | Path,
    *,
    registry_globs: list[str] | None = None,
    doc_globs: list[str] | None = None,
    code_dirs: list[str] | None = None,
    concerns: list[tuple[str, str]] | None = None,
    include_registries: bool = True,
    include_docs: bool = True,
    include_packages: bool = True,
    include_concerns: bool = True,
    collapse_threshold: int = 8,
    collapse_sample: int = 5,
) -> AuditPlan:
    """Enumerate every auditable target under ``root``, derived from its real shape.

    Globs/dirs default to generic heuristics (top-level + ``docs/`` markdown;
    JSON under common data dirs; packages under common source dirs) and are all
    overridable so the same builder serves any repo. A directory holding more than
    ``collapse_threshold`` registry/doc files is audited as ONE family target
    (grounded on a ``collapse_sample`` of its files) so generated-instance dirs do
    not drown the plan. The result is the COVERAGE registry: the complete set the
    driver must work through.
    """
    root = Path(root).expanduser().resolve()
    targets: list[AuditTarget] = []
    resolved_code_dirs = [
        d for d in (code_dirs or _DEFAULT_CODE_DIRS) if (root / d).is_dir()
    ]

    # The project's string-keyed data files — resolved ONCE here (not just for
    # registry targets) so the driver can build a data-field menu for doc/package
    # probes even when registry targets are excluded.
    data_file_rels = [
        rel for rel in _rglob_globs(root, registry_globs or _DEFAULT_REGISTRY_GLOBS)
        if _is_string_keyed(root / rel)
    ]

    if include_docs:
        doc_rels = _rglob_globs(root, doc_globs or _DEFAULT_DOC_GLOBS)
        for gid, paths, source in _collapse_groups(
            doc_rels, threshold=collapse_threshold, sample=collapse_sample
        ):
            targets.append(AuditTarget(
                id=f"doc:{gid}", kind=DOC, title=gid, source=source, paths=paths,
                question=(
                    "This doc (or family of docs) states how part of the system is "
                    "supposed to work. Audit the REAL code against it: every behavior, "
                    "contract, file, or invariant it claims exists — is it actually "
                    "implemented, wired, and consistent? Flag claims the code does not "
                    "honor and code that contradicts the doc."
                ),
            ))

    if include_registries:
        for gid, paths, source in _collapse_groups(
            data_file_rels, threshold=collapse_threshold, sample=collapse_sample
        ):
            targets.append(AuditTarget(
                id=f"registry:{gid}", kind=REGISTRY, title=gid, source=source, paths=paths,
                question=(
                    "This is a string-keyed registry/config (or family of them) that "
                    "declares a contract code must satisfy. Audit the dataflow: for "
                    "every key/id/field it declares, does code actually CONSUME it "
                    "(read it, branch on it, build behavior from it)? Flag declared "
                    "keys with no consumer (dead contract) and code that expects keys "
                    "the registry does not provide (drift)."
                ),
            ))

    if include_packages:
        for pkg_id, rel in _top_packages(root, resolved_code_dirs or _DEFAULT_CODE_DIRS):
            targets.append(AuditTarget(
                id=f"package:{pkg_id}", kind=PACKAGE, title=pkg_id, source=rel, paths=[rel],
                question=(
                    "Audit this code package for completeness and wiring: features "
                    "that are declared/stubbed but not implemented, public entry "
                    "points nothing calls (orphan/facade), TODO-shaped placeholders "
                    "presented as done, and logic that silently no-ops. Is every "
                    "promise this package makes actually kept?"
                ),
            ))

    if include_concerns:
        for cid, question in (concerns or _DEFAULT_CONCERNS):
            targets.append(AuditTarget(
                id=f"concern:{cid}", kind=CONCERN, title=cid, source="(cross-cutting)",
                question=question, paths=[],
            ))

    return AuditPlan(
        root=str(root), targets=targets,
        code_dirs=resolved_code_dirs or list(_DEFAULT_CODE_DIRS),
        data_files=data_file_rels,
    )
