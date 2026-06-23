"""Language dispatcher — the one seam the checks talk to.

Every Module/GDModule carries a ``lang`` tag; this routes the five reference
queries to the right adapter (Python AST vs GDScript regex) so checks.py is
written once and runs across languages. Adding a language = add an adapter with
this function set and a ``lang`` tag, then register it here. Nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import gdref, pyref

_ADAPTERS = {"python": pyref, "gdscript": gdref}

# Which extensions map to which adapter when parsing from disk.
_EXT_LANG = {".py": "python", ".gd": "gdscript"}

# Directories never worth parsing for a completeness map: VCS, caches, vendored
# deps, build output. Matched only on path segments BELOW the root (see
# project_files) so a project that itself lives under e.g. a `build/` dir — like
# RayXI's `src/rayxi/build/templates/` — is not wrongly skipped.
_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", "dist", ".tox",
})


def _adapter(mod):
    a = _ADAPTERS.get(getattr(mod, "lang", "python"))
    if a is None:
        raise ValueError(f"no codemap adapter for lang {getattr(mod, 'lang', None)!r}")
    return a


def field_reads(mod, field):
    return _adapter(mod).field_reads(mod, field)


def field_writes(mod, field):
    return _adapter(mod).field_writes(mod, field)


def identity_refs(mod, carriers):
    return _adapter(mod).identity_refs(mod, carriers)


def collection_decls(mod, name):
    return _adapter(mod).collection_decls(mod, name)


def calls_to(mod, callee):
    return _adapter(mod).calls_to(mod, callee)


def resolved_arg_callees(mod, resolvers):
    return _adapter(mod).resolved_arg_callees(mod, resolvers)


def name_decls(mod, name):
    return _adapter(mod).name_decls(mod, name)


def name_uses(mod, name):
    return _adapter(mod).name_uses(mod, name)


def function_defs(mod, name):
    return _adapter(mod).function_defs(mod, name)


def _ranked(counts: dict[str, int], n: int) -> dict[str, int]:
    """Most load-bearing first, ties broken by name so the menu is deterministic."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(items[:n])


@dataclass
class Symbols:
    """The project's real vocabulary, merged across every module — the grounding
    menu a manifest's specs must be drawn from. ``fields``/``funcs``/``calls`` map
    a symbol to its total occurrence count; ``collections`` maps a name to the
    union of string members declared under it. :meth:`top` returns a ranked,
    capped, JSON-ready slice small enough to drop into an LLM prompt without the
    long tail of one-off locals drowning the load-bearing symbols."""
    fields: dict[str, int] = field(default_factory=dict)
    funcs: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    collections: dict[str, list[str]] = field(default_factory=dict)

    def top(self, *, fields: int = 40, funcs: int = 40, calls: int = 40,
            collections: int = 20) -> dict:
        return {
            "fields": _ranked(self.fields, fields),
            "funcs": _ranked(self.funcs, funcs),
            "calls": _ranked(self.calls, calls),
            "collections": {k: self.collections[k] for k in sorted(self.collections)[:collections]},
        }


def symbols(mods) -> Symbols:
    """Aggregate per-module symbol inventories into one project-wide :class:`Symbols`.
    Counts sum across modules; a collection name seen in several modules merges to
    the union of its members (the duplicate-declaration *drift* signal stays the
    job of :func:`saddle.codemap.impact.impact_identity`, not the menu)."""
    agg = Symbols()
    for mod in mods:
        s = _adapter(mod).symbols(mod)
        for name, c in s["fields"].items():
            agg.fields[name] = agg.fields.get(name, 0) + c
        for name, c in s["funcs"].items():
            agg.funcs[name] = agg.funcs.get(name, 0) + c
        for name, c in s["calls"].items():
            agg.calls[name] = agg.calls.get(name, 0) + c
        for name, members in s["collections"].items():
            prev = agg.collections.get(name)
            agg.collections[name] = sorted(set(members) if prev is None else set(prev) | set(members))
    return agg


def parse_paths(paths):
    """Parse a mixed list of files, routing each by extension. Unknown
    extensions are skipped (a completeness map only reasons over code it can
    read; unreadable files are reported elsewhere, not silently mis-parsed)."""
    by_lang: dict[str, list[str]] = {}
    for p in paths:
        lang = _EXT_LANG.get(Path(p).suffix.lower())
        if lang:
            by_lang.setdefault(lang, []).append(str(p))
    mods = []
    for lang, group in by_lang.items():
        mods.extend(_ADAPTERS[lang].parse_paths(group))
    return mods


def project_files(root, *, exts: set[str] | None = None) -> list[str]:
    """Every parseable source file under ``root``, recursively, minus the
    excluded directories. ``exts`` defaults to all known languages. Returns a
    stable (sorted) list so a parse is deterministic."""
    root = Path(root)
    wanted = exts if exts is not None else set(_EXT_LANG)
    out: list[str] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in wanted or not p.is_file():
            continue
        # Exclude on segments BELOW root only — never on the root's own ancestry.
        rel_parts = p.relative_to(root).parts
        if any(part in _EXCLUDE_DIRS for part in rel_parts):
            continue
        out.append(str(p))
    return sorted(out)


def parse_project(root, *, exts: set[str] | None = None) -> list:
    """Parse a whole project tree into Modules — the one call Layer 2 makes to
    look at the target project's code (root from :func:`saddle.context.code_root`)."""
    return parse_paths(project_files(root, exts=exts))
