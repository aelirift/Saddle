"""Code-derived reference extraction (Python AST).

This is the borrow-and-upgrade of RayXI's value_impact_map.scan_consumers /
_symbol_resolve.collect_declared. RayXI scans GDScript with regex (can't tell a
read from a write, a comment from code, or scope boundaries). Here we use the
real `ast`, so every reference is precise: Load vs Store context, exact
enclosing function, exact node kind. A GDScript adapter (regex, borrowed from
RayXI) plugs in behind the same Ref/Module interface for the full build.

The map is ALWAYS derived from the parsed code — never from a declaration the
code can silently diverge from. That single property is what RayXI's declared
provides/reads_from and its write-only value_impact_map both lacked at the gate.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


def _domain_from(path: str, source: str) -> str:
    """Server / client / shared. An explicit `# saddle-domain: X` marker in the
    first few lines wins; otherwise the path decides. Models the trust/network
    boundary as a first-class axis (RayXI had no such axis, so server-only
    changes never surfaced as un-mirrored)."""
    for line in source.splitlines()[:5]:
        s = line.strip()
        if s.startswith("# saddle-domain:"):
            return s.split(":", 1)[1].strip()
    p = path.replace("\\", "/").lower()
    if "/server" in p or p.startswith("server/"):
        return "server"
    if "/client" in p or p.startswith("client/"):
        return "client"
    return "shared"


@dataclass
class Ref:
    path: str
    lineno: int
    kind: str          # attr_read | subscript_read | get_read | *_write | compare | assign | match_case | call
    name: str          # the field / identity-literal / function name
    func: str | None   # enclosing function name (None at module scope)
    snippet: str
    domain: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.lineno}"


@dataclass
class Module:
    path: str
    domain: str
    source: str
    tree: ast.Module
    lines: list[str]
    parents: dict[int, ast.AST]
    lang: str = "python"

    def line(self, n: int) -> str:
        return self.lines[n - 1].strip() if 0 < n <= len(self.lines) else ""

    def enclosing_func(self, node: ast.AST) -> str | None:
        cur = self.parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = self.parents.get(id(cur))
        return None


def parse_modules(sources: list[tuple[str, str]]) -> list[Module]:
    mods: list[Module] = []
    for path, src in sources:
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError:
            # A completeness map only reasons over code it can read; a file that
            # won't parse is skipped here (and surfaced as unreadable elsewhere),
            # never silently mis-parsed — and one bad file can't sink the project.
            continue
        parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        mods.append(Module(path, _domain_from(path, src), src, tree,
                           src.splitlines(), parents))
    return mods


def parse_paths(paths) -> list[Module]:
    out: list[tuple[str, str]] = []
    for p in paths:
        p = Path(p)
        out.append((str(p), p.read_text(encoding="utf-8", errors="replace")))
    return parse_modules(out)


def _str_const(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _mk(mod: Module, node: ast.AST, kind: str, name: str) -> Ref:
    ln = getattr(node, "lineno", 0)
    return Ref(mod.path, ln, kind, name, mod.enclosing_func(node), mod.line(ln), mod.domain)


def field_reads(mod: Module, field: str) -> list[Ref]:
    """Every Load-context read of `field`: `x.field`, `x['field']`, `x.get('field')`."""
    refs: list[Ref] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Attribute) and node.attr == field and isinstance(node.ctx, ast.Load):
            refs.append(_mk(mod, node, "attr_read", field))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _str_const(node.slice) == field:
                refs.append(_mk(mod, node, "subscript_read", field))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and _str_const(node.args[0]) == field:
                refs.append(_mk(mod, node, "get_read", field))
    return refs


def field_writes(mod: Module, field: str) -> list[Ref]:
    """Every Store-context write of `field`: `x.field = ...` / `x['field'] = ...`."""
    refs: list[Ref] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == field and isinstance(t.ctx, ast.Store):
                    refs.append(_mk(mod, t, "attr_write", field))
                elif isinstance(t, ast.Subscript) and isinstance(t.ctx, ast.Store):
                    if _str_const(t.slice) == field:
                        refs.append(_mk(mod, t, "subscript_write", field))
    return refs


def _carrier_matches(node, carrier: str) -> bool:
    """Does `node` hold this identity carrier?

    A BARE carrier (``"kind"``) matches the name anywhere it appears — a local
    ``kind``, any ``*.kind`` attribute, any ``*["kind"]`` key. A QUALIFIED carrier
    (``"effect.kind"``) matches ONLY ``effect.kind`` / ``effect["kind"]`` on an
    object named ``effect``. Qualification is how an overloaded attribute name
    (``kind``/``type`` reused across enum namespaces) is disambiguated so a
    literal from a *different* namespace isn't mis-flagged as drift."""
    obj, sep, attr = carrier.rpartition(".")
    if not sep:  # bare
        if isinstance(node, ast.Name):
            return node.id == carrier
        if isinstance(node, ast.Attribute):
            return node.attr == carrier
        if isinstance(node, ast.Subscript):
            return _str_const(node.slice) == carrier
        return False
    # qualified obj.attr — the object must be a plain Name == obj
    if isinstance(node, ast.Attribute) and node.attr == attr:
        return isinstance(node.value, ast.Name) and node.value.id == obj
    if isinstance(node, ast.Subscript) and _str_const(node.slice) == attr:
        return isinstance(node.value, ast.Name) and node.value.id == obj
    return False


def _is_carrier(node, carriers: set[str]) -> bool:
    return any(_carrier_matches(node, c) for c in carriers)


def identity_refs(mod: Module, carriers: set[str]) -> list[tuple[str, Ref]]:
    """String literals used as an identity: compared to / assigned to / matched
    against a carrier (a var/attr/key that holds this identity). Covers `==`,
    `!=`, `in (...)`, assignment, AND `match` — RayXI's owned_type_drift only
    covered `match` cases, so literal drift in an `if x == "freeze"` was invisible.
    """
    out: list[tuple[str, Ref]] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            if any(_is_carrier(o, carriers) for o in operands):
                for o in operands:
                    s = _str_const(o)
                    if s is not None:
                        out.append((s, _mk(mod, o, "compare", s)))
                    elif isinstance(o, (ast.Tuple, ast.List, ast.Set)):
                        for e in o.elts:
                            es = _str_const(e)
                            if es is not None:
                                out.append((es, _mk(mod, e, "compare", es)))
        elif isinstance(node, ast.Assign):
            s = _str_const(node.value)
            if s is not None and any(_is_carrier(t, carriers) for t in node.targets):
                out.append((s, _mk(mod, node.value, "assign", s)))
        elif isinstance(node, ast.Match):
            if _is_carrier(node.subject, carriers):
                for case in node.cases:
                    pats = [case.pattern]
                    if isinstance(case.pattern, ast.MatchOr):
                        pats = list(case.pattern.patterns)
                    for pat in pats:
                        if isinstance(pat, ast.MatchValue):
                            s = _str_const(pat.value)
                            if s is not None:
                                out.append((s, _mk(mod, pat.value, "match_case", s)))
    return out


def _string_collection(node) -> list[str] | None:
    container = node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set", "list", "tuple"):
        if not node.args:
            return []
        container = node.args[0]
    if isinstance(container, (ast.Set, ast.List, ast.Tuple)):
        vals = [_str_const(e) for e in container.elts]
        if vals and all(v is not None for v in vals):
            return [v for v in vals if v is not None]
    return None


def collection_decls(mod: Module, name: str) -> list[tuple[set[str], Ref]]:
    """Assignments `name = {...}/[...]/(...)/frozenset({...})` of string constants.
    More than one across the project == a duplicated canonical set (the drift
    enabler the user fought with 'lock all enums')."""
    out: list[tuple[set[str], Ref]] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    elts = _string_collection(node.value)
                    if elts is not None:
                        out.append((set(elts), _mk(mod, node, "assign", name)))
    return out


def calls_to(mod: Module, callee: str) -> list[Ref]:
    """Every call to a function named `callee` — `callee(...)` or `x.callee(...)`.
    Used to decide whether a consumer resolved the effective value *in scope*
    before reading a base field (RayXI's _function_has_resolve, in real AST)."""
    refs: list[Ref] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == callee:
                refs.append(_mk(mod, node, "call", callee))
            elif isinstance(f, ast.Attribute) and f.attr == callee:
                refs.append(_mk(mod, node, "call", callee))
    return refs


def _callee_name(func) -> str | None:
    """The simple name of a call's callee: `foo(...)`->'foo', `x.foo(...)`->'foo'."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_resolver_call(node, resolvers: set[str]) -> bool:
    return isinstance(node, ast.Call) and _callee_name(node.func) in resolvers


def resolved_arg_callees(mod: Module, resolvers: set[str]) -> set[str]:
    """Functions that RECEIVE an already-resolved value as a call argument.

    One-hop interprocedural coverage. If a caller computes the effective value
    (`cd = resolve_cd(inst)`, or inline `helper(inst, resolve_cd(inst))`) and
    hands it to `helper`, then `helper` reads the resolved value through its
    parameter — so `helper`'s own raw read of the base field is NOT a modifier
    that fails to propagate. Returns the set of callee names so impact_value can
    treat those functions' reads as covered.

    Deliberate, documented trade: a callee that receives the resolved value AND
    independently reads the base for a *different* purpose is also treated as
    covered. The hand-off pattern (resolve in caller, apply in helper) is common;
    the double-read is rare; killing the common false positive is worth missing
    the rare real one. Scoped one hop and per function body — module-scope
    hand-offs and chains deeper than one call are intentionally out of scope."""
    if not resolvers:
        return set()
    out: set[str] = set()
    for fn in ast.walk(mod.tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        resolved: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_resolver_call(node.value, resolvers):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        resolved.add(t.id)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node.func)
            if callee is None or callee in resolvers:
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            for arg in args:
                if _is_resolver_call(arg, resolvers) or (
                    isinstance(arg, ast.Name) and arg.id in resolved
                ):
                    out.add(callee)
                    break
    return out


def name_decls(mod: Module, name: str) -> list[Ref]:
    """Declaration sites of a script-scope symbol `name`: a module/class-level
    assignment (`NAME = ...`) or annotated assignment (`NAME: T = ...`). A
    declaration is at module or CLASS scope, never inside a function — a local of
    the same name is a different binding, not the design knob. (Python has no
    `@export`; a module/class constant is its analogue.)"""
    out: list[Ref] = []
    for node in ast.walk(mod.tree):
        target = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    target = t
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                target = node.target
        if target is not None and mod.enclosing_func(target) is None:
            out.append(_mk(mod, target, "declaration", name))
    return out


def name_uses(mod: Module, name: str) -> list[Ref]:
    """Every READ of `name`: a Load-context bare `ast.Name` (a module/class
    constant read by name) OR a Load-context `obj.NAME` attribute (a class field
    read through `self`/an instance). Deliberately permissive over the attribute
    base — the conservative bias for a liveness check is to over-count reads so a
    symbol is flagged DEAD only when truly nothing references it (a false 'dead' is
    worse than a missed one). A Store target is not a read, so a symbol with
    declarations but zero name_uses is dead."""
    out: list[Ref] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            out.append(_mk(mod, node, "use", name))
        elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
            out.append(_mk(mod, node, "use", name))
    return out


def function_defs(mod: Module, name: str) -> list[Ref]:
    """Definition site(s) of a function named `name` (`def name(...)`)."""
    out: list[Ref] = []
    for node in ast.walk(mod.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            out.append(_mk(mod, node, "func_def", name))
    return out


def symbols(mod: Module) -> dict:
    """Deterministic symbol inventory for ONE module — the raw vocabulary any
    spec must be grounded in, so the LLM names `cooldown_s` (what the code uses)
    not `cooldown` (what it imagines). Four buckets, each separated because each
    feeds a different spec slot:

      * fields      — attr/subscript/`.get()` accesses: candidate value `field`s
                      and identity `carriers`.
      * funcs       — `def` names: candidate `producers` / `replication_func`.
      * calls       — call sites: candidate `accessor`s (a resolver is CALLED, so
                      this is where a resolver surfaces, not in funcs alone).
      * collections — string-set assignments: candidate identity `canonical` sets
                      keyed by their `source_symbol`.

    Counts are per-occurrence so :func:`saddle.codemap.refs.symbols` can rank a
    symbol by how load-bearing it is across the whole project."""
    fields: dict[str, int] = {}
    funcs: dict[str, int] = {}
    calls: dict[str, int] = {}
    collections: dict[str, list[str]] = {}
    # An Attribute that is the callee of a Call is a METHOD, not a field read.
    call_attr_ids = {
        id(n.func) for n in ast.walk(mod.tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Attribute) and id(node) not in call_attr_ids:
            fields[node.attr] = fields.get(node.attr, 0) + 1
        elif isinstance(node, ast.Subscript):
            k = _str_const(node.slice)
            if k is not None:
                fields[k] = fields.get(k, 0) + 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = funcs.get(node.name, 0) + 1
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                calls[f.id] = calls.get(f.id, 0) + 1
            elif isinstance(f, ast.Attribute):
                calls[f.attr] = calls.get(f.attr, 0) + 1
                if f.attr == "get" and node.args:  # `.get("field")` names a field
                    k = _str_const(node.args[0])
                    if k is not None:
                        fields[k] = fields.get(k, 0) + 1
        if isinstance(node, ast.Assign):
            elts = _string_collection(node.value)
            if elts is not None:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        collections[t.id] = elts
    return {"fields": fields, "funcs": funcs, "calls": calls, "collections": collections}
