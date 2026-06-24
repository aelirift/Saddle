"""Grounding for the audit surface — assemble the REAL evidence a probe reasons
over, so a finding is anchored in the code/registry/doc, never hallucinated.

WHAT GROUNDS A PROBE
--------------------
  - the declared files (the doc text, the registry JSON, a sampled slice of a
    package's source) — the contract the probe audits against;
  - the project SYMBOL MENU (``codemap.refs`` — real field/func/call names ranked
    by use) so the probe writes findings in the code's actual vocabulary;
  - for a registry target, the STRING-KEYED DATAFLOW: every key the registry
    declares, mapped to the code sites that reference it — so "this key is
    consumed nowhere" (a dead contract) is a fact the probe is handed, not a guess;
  - for a cross-cutting concern, the SEED scan: the caller's project-specific
    patterns (e.g. the engine's replication / authority tokens) resolved to the
    code lines that match, so a congruence probe sees the actual write/read sites.

Everything is deterministic (no LLM) and size-capped so the bundle fits one
prompt. The probe (probe.py) turns this evidence into findings; this module never
judges — it only gathers.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from saddle.codemap import refs

# Size caps — generous enough to ground a real audit, bounded enough to fit a
# single prompt without crowding out the symbol menu.
_MAX_FILE_BYTES = 24_000        # per grounding file (a long doc/registry is truncated)
_MAX_TOTAL_FILE_BYTES = 90_000  # across all grounding files for one target
_MAX_KEYS = 80                  # registry keys whose dataflow we trace
_MAX_SITES_PER_KEY = 6          # code sites cited per key
_MAX_SEED_HITS = 40             # total code lines cited per concern seed
_PKG_SAMPLE_FILES = 14          # source files sampled to ground a package target

# The audit reasons over the SYSTEM UNDER AUDIT — the project's own source — never
# the build OUTPUT a pipeline emits. A generator like rayxi scatters tens of
# thousands of generated .gd/.py game instances across the tree (knowledge/level/…);
# parsing the whole repo to ground the audit is the slow, memory-heavy mistake that
# made a probe look hung. We parse/scan only the real code dirs.
_DEFAULT_CODE_DIRS: tuple[str, ...] = ("src", "app", "apps", "lib", "pkg")
# Extensions worth a raw-text scan (registry-key dataflow + concern seeds) — broader
# than what the AST parser reads, so a Go server / TS client congruence gap is seen.
_TEXT_SCAN_EXTS: set[str] = {
    ".py", ".gd", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cs",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".kt", ".swift", ".rb", ".php",
}


def source_files(root, code_dirs: list[str] | None = None, *, exts: set[str] | None = None) -> list[str]:
    """Parseable source under the project's real code dirs ONLY (``src``/``apps``/…),
    never the whole repo. Falls back to the whole root only for a flat project that
    has none of the conventional code dirs. ``exts`` defaults to the AST languages
    (py/gd); pass :data:`_TEXT_SCAN_EXTS` for a broader raw-text scan."""
    root = Path(root)
    dirs = code_dirs or list(_DEFAULT_CODE_DIRS)
    out: list[str] = []
    seen: set[str] = set()
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for f in refs.project_files(base, exts=exts):
            if f not in seen:
                seen.add(f)
                out.append(f)
    if not out and not any((root / d).is_dir() for d in dirs):
        return refs.project_files(root, exts=exts)  # flat repo: nothing else to scope to
    return sorted(out)


def parse_sources(root, code_dirs: list[str] | None = None) -> list:
    """Parse the project's source dirs into Modules — the scoped equivalent of
    :func:`refs.parse_project`, so the symbol menu reflects the pipeline's own
    vocabulary, not the generated games it outputs."""
    return refs.parse_paths(source_files(root, code_dirs))


@dataclass
class Grounding:
    """The evidence bundle a probe sees for one target. Pure data; ``format`` renders
    it to the prompt text."""

    target_id: str
    files: dict[str, str] = field(default_factory=dict)        # rel-path -> (capped) text
    symbol_menu: dict = field(default_factory=dict)            # codemap symbol menu
    key_dataflow: dict[str, list[str]] = field(default_factory=dict)  # registry key -> sites
    seed_hits: dict[str, list[str]] = field(default_factory=dict)     # seed -> "rel:line  <code>"
    notes: list[str] = field(default_factory=list)            # grounding caveats (truncation, no code)

    def is_empty(self) -> bool:
        return not (self.files or self.symbol_menu or self.key_dataflow or self.seed_hits)

    def format(self) -> str:
        out: list[str] = []
        for rel, text in self.files.items():
            out.append(f"=== FILE: {rel} ===\n{text}")
        if self.key_dataflow:
            lines = ["=== REGISTRY KEY DATAFLOW (declared key -> code sites that reference it) ==="]
            for key, sites in self.key_dataflow.items():
                where = ", ".join(sites) if sites else "NO REFERENCES FOUND IN CODE (dead contract?)"
                lines.append(f"  {key}: {where}")
            out.append("\n".join(lines))
        if self.seed_hits:
            lines = ["=== CONCERN SEED MATCHES (pattern -> code lines) ==="]
            for seed, hits in self.seed_hits.items():
                lines.append(f"  pattern /{seed}/:")
                if hits:
                    lines.extend(f"    {h}" for h in hits)
                else:
                    lines.append("    (no matches)")
            out.append("\n".join(lines))
        if self.symbol_menu:
            out.append("=== PROJECT SYMBOL MENU (real names, ranked by use — ground your finding sites here) ===\n"
                       + json.dumps(self.symbol_menu, indent=1))
        if self.notes:
            out.append("=== GROUNDING NOTES ===\n" + "\n".join(f"  - {n}" for n in self.notes))
        return "\n\n".join(out)


def _read_capped(path: Path, budget: int) -> tuple[str, bool]:
    """Read up to ``budget`` bytes of text; (text, truncated?)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — unreadable file is a grounding gap, not a crash
        return "", False
    if len(raw) <= budget:
        return raw, False
    return raw[:budget] + f"\n…[truncated {len(raw) - budget} bytes]", True


def _expand_paths(root: Path, rel_paths: list[str], *, sample: int) -> list[Path]:
    """Resolve declared paths (files or dirs) into concrete source files, sampling
    a dir down to its most-central files so a big package does not blow the budget."""
    files: list[Path] = []
    for rel in rel_paths:
        p = (root / rel)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            srcs = [f for f in refs.project_files(p)]  # parseable source only, sorted
            # Prefer package entry points + larger (more central) files.
            srcs.sort(key=lambda f: (Path(f).name not in ("__init__.py", "__init__.gd"),
                                     -Path(f).stat().st_size if Path(f).exists() else 0))
            files.extend(Path(s) for s in srcs[:sample])
    return files


def read_files(root: Path, rel_paths: list[str], *, sample: int = _PKG_SAMPLE_FILES) -> tuple[dict[str, str], list[str]]:
    """Read the declared grounding files (expanding dirs), capped per-file and in
    total. Returns (rel-path -> text, notes)."""
    out: dict[str, str] = {}
    notes: list[str] = []
    spent = 0
    for path in _expand_paths(root, rel_paths, sample=sample):
        if spent >= _MAX_TOTAL_FILE_BYTES:
            notes.append("grounding-file budget reached — some files omitted")
            break
        text, truncated = _read_capped(path, min(_MAX_FILE_BYTES, _MAX_TOTAL_FILE_BYTES - spent))
        if not text:
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        out[rel] = text
        spent += len(text)
        if truncated:
            notes.append(f"{rel} truncated to fit the prompt")
    return out, notes


def registry_keys(data) -> list[str]:
    """The string keys/ids a registry declares that code would reference: top-level
    object keys, and the id/name/key/type/slug values of nested objects. Deduped,
    capped, length-filtered (a 1-char or huge value is not a real key)."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(v) -> None:
        if isinstance(v, str):
            s = v.strip()
            if 2 <= len(s) <= 80 and s not in seen and not s.isspace():
                seen.add(s)
                keys.append(s)

    def _walk(node, depth: int) -> None:
        if len(keys) >= _MAX_KEYS or depth > 4:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                _add(k)
                if isinstance(v, (dict, list)):
                    _walk(v, depth + 1)
                elif k in ("id", "name", "key", "type", "slug", "kind"):
                    _add(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v, depth + 1)

    if isinstance(data, dict):
        _walk(data, 0)
    elif isinstance(data, list):
        _walk(data, 0)
    return keys[:_MAX_KEYS]


def _iter_code_text(code_files: list[str]) -> "list[tuple[str, str]]":
    """(path, text) for each code file, read once. Unreadable files skipped."""
    out: list[tuple[str, str]] = []
    for f in code_files:
        try:
            out.append((f, Path(f).read_text(encoding="utf-8", errors="replace")))
        except Exception:  # noqa: BLE001
            continue
    return out


def scan_key_sites(
    root: Path, keys: list[str], code_text: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Map each registry key to the code sites (``rel:line``) that reference it as a
    literal. A key with an empty list is a dead contract — declared, consumed by no
    code. Pre-filtered per file so only files actually containing a key are line-scanned."""
    hits: dict[str, list[str]] = {k: [] for k in keys}
    keyset = set(keys)
    for path, text in code_text:
        present = [k for k in keyset if k in text]
        if not present:
            continue
        try:
            rel = str(Path(path).relative_to(root))
        except ValueError:
            rel = path
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            for k in present:
                if len(hits[k]) >= _MAX_SITES_PER_KEY:
                    continue
                if k in line:
                    hits[k].append(f"{rel}:{i}")
    return hits


def scan_seeds(
    root: Path, seeds: list[str], code_text: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Map each concern seed (regex) to the code lines that match (``rel:line  <code>``),
    capped. The grounding for a cross-cutting probe — the actual write/read/guard sites."""
    out: dict[str, list[str]] = {}
    for seed in seeds:
        try:
            rx = re.compile(seed)
        except re.error:
            out[seed] = [f"(invalid regex: {seed})"]
            continue
        found: list[str] = []
        for path, text in code_text:
            if len(found) >= _MAX_SEED_HITS:
                break
            try:
                rel = str(Path(path).relative_to(root))
            except ValueError:
                rel = path
            for i, line in enumerate(text.splitlines(), 1):
                if len(found) >= _MAX_SEED_HITS:
                    break
                if rx.search(line):
                    found.append(f"{rel}:{i}  {line.strip()[:160]}")
        out[seed] = found
    return out


def ground_target(
    target,
    root: str | Path,
    *,
    mods: list | None = None,
    code_files: list[str] | None = None,
    code_text: list[tuple[str, str]] | None = None,
    code_dirs: list[str] | None = None,
) -> Grounding:
    """Assemble the full :class:`Grounding` for one :class:`~saddle.audit.plan.AuditTarget`.

    ``mods``/``code_files``/``code_text`` are parsed/read ONCE by the driver and
    passed in so a whole-plan run does not re-parse per target. Any missing piece
    is recomputed best-effort here (scoped to ``code_dirs`` — the project's real
    source, NOT the whole repo), so a single target can also be grounded alone
    without the whole-tree parse that made a probe look hung.
    """
    root = Path(root).expanduser().resolve()
    g = Grounding(target_id=target.id)

    files, notes = read_files(root, target.paths)
    g.files = files
    g.notes = notes

    if mods is None:
        try:
            mods = parse_sources(root, code_dirs)
        except Exception:  # noqa: BLE001 — code grounding is best-effort
            mods = []
    if mods:
        g.symbol_menu = refs.symbols(mods).top()
    else:
        g.notes.append("no parseable code found under root — symbol menu empty")

    # Registry dataflow + concern seeds both need the raw code text.
    needs_scan = target.kind == "registry" or bool(target.seeds)
    if needs_scan:
        if code_text is None:
            if code_files is None:
                code_files = source_files(root, code_dirs, exts=_TEXT_SCAN_EXTS)
            code_text = _iter_code_text(code_files)
        if target.kind == "registry" and target.paths:
            # A collapsed family carries several sample files — union their keys so
            # the dataflow is representative of the whole directory, not one file.
            keys: list[str] = []
            seen: set[str] = set()
            for rel in target.paths:
                if not rel.endswith(".json"):
                    continue
                try:
                    data = json.loads((root / rel).read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 — a bad registry file is itself a finding the probe can see
                    g.notes.append(f"could not parse registry {rel} for key dataflow")
                    continue
                for k in registry_keys(data):
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            if keys:
                g.key_dataflow = scan_key_sites(root, keys[:_MAX_KEYS], code_text)
        if target.seeds:
            g.seed_hits = scan_seeds(root, target.seeds, code_text)
    return g
