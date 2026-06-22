"""GDScript adapter — regex-derived references behind the same Ref interface.

GDScript has no Python `ast`, so this borrows RayXI's proven regex scanners
(value_impact_map.scan_consumers + _function_has_resolve, _symbol_resolve) and
produces the SAME Ref shape pyref does, so the checks run unchanged across
languages. Two upgrades over RayXI's originals:

  * read/write separation — RayXI lumped every occurrence together; here an
    accessor followed by an assignment operator is a WRITE, everything else a
    READ, so a modifier that WRITES the effective value isn't misread as a
    consumer of the base.
  * dynamic-call detection — RayXI's runtime leans on `obj.call("fn", ...)`;
    calls_to matches the direct, method, AND dynamic-string call forms, so the
    "is the resolver called in this consumer's scope" question is answered for
    real Godot code, not just static calls.

Regex is lower-fidelity than a real AST. It is deliberately conservative: it
errs toward calling a thing a READ (so coverage must be proven, never assumed) —
the same bias RayXI's scanner had, which is correct for a completeness gate.

Domain (server/client) is derived PER FUNCTION from authority signals, not file
path — Godot's split is by multiplayer authority, so a path heuristic would be
wrong more often than right. A function is `server` if it carries an `@rpc`
authority annotation or an `is_server()`/`is_multiplayer_authority()` guard;
`client` if it is an `@rpc("any_peer")` receiver or named like a replicated-state
applier (`_recv_*`, `apply_*`, `*_synced`); otherwise `shared`. An explicit
`# saddle-domain:` marker still overrides module-wide. This is what lets the
boundary check fire on GDScript (see GDModule.func_domain, item #11).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .pyref import Ref, _domain_from

_FUNC_RE = re.compile(r"^\s*(?:static\s+)?func\s+(\w+)")
# An accessor immediately followed by one of these is a WRITE, not a read.
_ASSIGN_AHEAD = re.compile(r"^\s*(?:=(?!=)|\+=|-=|\*=|/=|\|=|&=)")

# Per-function domain signals (item #11). Godot splits server/client by
# MULTIPLAYER AUTHORITY, not file path, so the boundary axis has to be read off
# the code: an RPC annotation, an authority guard, or a receive/apply naming
# convention. These are heuristics — deliberately coarse and function-scoped;
# block-level `if is_server(): ... else: ...` precision is out of scope (a single
# function gets a single domain). They only REFINE a module whose domain is the
# neutral "shared"; an explicit `# saddle-domain:` marker still wins module-wide.
_AUTHORITY_GUARD_RE = re.compile(
    r"\b(?:multiplayer\.is_server|is_multiplayer_authority)\s*\("
)
# Receivers/appliers of replicated state run on the client.
_CLIENT_FUNC_RE = re.compile(r"^_?recv_|^apply_|_synced$")

# GDScript keywords that read as `keyword(...)` but are control flow, not calls.
_GD_KEYWORDS = frozenset({
    "if", "elif", "else", "for", "while", "match", "return", "func", "var",
    "const", "and", "or", "not", "in", "is", "as", "await", "yield", "pass",
    "break", "continue", "when", "print", "assert", "preload", "range",
})


@dataclass
class GDModule:
    path: str
    domain: str
    source: str
    lines: list[str]
    lang: str = "gdscript"
    # def-line-index -> detected domain; ascending def indices for lookup.
    func_domains: dict[int, str] = field(default_factory=dict)
    _def_idxs: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Per-function detection only matters when the module domain is the
        # neutral "shared"; an explicit marker/path domain overrides it anyway, so
        # skip the body scans entirely in that case.
        if self.domain != "shared":
            return
        for i, ln in enumerate(self.lines):
            if _FUNC_RE.match(ln):
                self._def_idxs.append(i)
                self.func_domains[i] = self._compute_func_domain(i)

    def line(self, n: int) -> str:
        return self.lines[n - 1].strip() if 0 < n <= len(self.lines) else ""

    def enclosing_func(self, lineno: int) -> str | None:
        """Nearest preceding `func` (RayXI's heuristic, exact)."""
        for i in range(min(lineno, len(self.lines)) - 1, -1, -1):
            m = _FUNC_RE.match(self.lines[i])
            if m:
                return m.group(1)
        return None

    def func_domain(self, lineno: int) -> str:
        """The domain of the function enclosing `lineno`. An explicit module
        marker/path domain (anything but the neutral "shared") wins module-wide;
        otherwise the per-function authority signal decides; otherwise "shared"."""
        if self.domain != "shared":
            return self.domain
        best: int | None = None
        target = lineno - 1
        for idx in self._def_idxs:
            if idx <= target:
                best = idx
            else:
                break
        return self.func_domains.get(best, "shared") if best is not None else "shared"

    def _compute_func_domain(self, def_idx: int) -> str:
        m = _FUNC_RE.match(self.lines[def_idx])
        name = m.group(1) if m else ""
        # RPC annotations directly above the def — the explicit network contract.
        annot: list[str] = []
        j = def_idx - 1
        while j >= 0 and self.lines[j].strip().startswith("@"):
            annot.append(self.lines[j].strip())
            j -= 1
        ann = " ".join(annot)
        if "@rpc" in ann:
            return "client" if "any_peer" in ann else "server"
        # Authority guard anywhere in the body -> server-only execution.
        def_indent = len(self.lines[def_idx]) - len(self.lines[def_idx].lstrip())
        for k in range(def_idx + 1, len(self.lines)):
            ln = self.lines[k]
            if not ln.strip():
                continue
            if len(ln) - len(ln.lstrip()) <= def_indent:
                break
            if _AUTHORITY_GUARD_RE.search(ln):
                return "server"
        # Naming convention for replicated-state receivers/appliers.
        if _CLIENT_FUNC_RE.search(name):
            return "client"
        return "shared"


def parse_modules(sources: list[tuple[str, str]]) -> list[GDModule]:
    return [
        GDModule(path, _domain_from(path, src), src, src.splitlines())
        for path, src in sources
    ]


def parse_paths(paths) -> list[GDModule]:
    out: list[tuple[str, str]] = []
    for p in paths:
        p = Path(p)
        out.append((str(p), p.read_text(encoding="utf-8", errors="replace")))
    return parse_modules(out)


def _mk(mod: GDModule, lineno: int, kind: str, name: str) -> Ref:
    return Ref(mod.path, lineno, kind, name, mod.enclosing_func(lineno),
               mod.line(lineno), mod.func_domain(lineno))


def _code_lines(mod: GDModule):
    for idx, ln in enumerate(mod.lines):
        if ln.lstrip().startswith("#"):
            continue
        yield idx + 1, ln


def field_reads(mod: GDModule, field: str) -> list[Ref]:
    """`x.get("field"...)` (always read), `x["field"]` / `x.field` when NOT
    immediately assigned to."""
    f = re.escape(field)
    get_re = re.compile(r'\.get\(\s*["\']%s["\']' % f)
    sub_re = re.compile(r'\[\s*["\']%s["\']\s*\]' % f)
    attr_re = re.compile(r"\.%s\b" % f)
    refs: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        for _ in get_re.finditer(ln):
            refs.append(_mk(mod, lineno, "get_read", field))
        for m in sub_re.finditer(ln):
            if not _ASSIGN_AHEAD.match(ln[m.end():]):
                refs.append(_mk(mod, lineno, "subscript_read", field))
        for m in attr_re.finditer(ln):
            # don't double-count the `.get(` head as an attr read of `get`
            if not _ASSIGN_AHEAD.match(ln[m.end():]):
                refs.append(_mk(mod, lineno, "attr_read", field))
    return refs


def field_writes(mod: GDModule, field: str) -> list[Ref]:
    f = re.escape(field)
    sub_w = re.compile(r'\[\s*["\']%s["\']\s*\]\s*(?:=(?!=)|\+=|-=|\*=|/=)' % f)
    attr_w = re.compile(r"\.%s\b\s*(?:=(?!=)|\+=|-=|\*=|/=)" % f)
    refs: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        if sub_w.search(ln):
            refs.append(_mk(mod, lineno, "subscript_write", field))
        elif attr_w.search(ln):
            refs.append(_mk(mod, lineno, "attr_write", field))
    return refs


def calls_to(mod: GDModule, callee: str) -> list[Ref]:
    """`callee(...)`, `x.callee(...)`, and the dynamic `x.call("callee", ...)` /
    `x.call_deferred("callee", ...)` forms RayXI's runtime uses. The `func
    callee(...)` DEFINITION line is excluded, and `has_method("callee")` never
    matches (it isn't a `.call*(` form)."""
    c = re.escape(callee)
    direct = re.compile(r"(?<![\w.])%s\s*\(" % c)
    method = re.compile(r"\.%s\s*\(" % c)
    dynamic = re.compile(r'\.call\w*\(\s*["\']%s["\']' % c)
    def_re = re.compile(r"^\s*(?:static\s+)?func\s+%s\b" % c)
    refs: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        if def_re.match(ln):
            continue
        if direct.search(ln) or method.search(ln) or dynamic.search(ln):
            refs.append(_mk(mod, lineno, "call", callee))
    return refs


def _paren_span(text: str, open_idx: int) -> str | None:
    """The substring inside the parentheses that open at ``text[open_idx]`` ('('),
    nesting- and quote-aware. None if it doesn't balance on this line — a
    deliberately conservative bias (an unbalanced/multi-line call yields no
    coverage rather than a guess)."""
    if open_idx >= len(text) or text[open_idx] != "(":
        return None
    depth = 0
    quote: str | None = None
    esc = False
    start = open_idx + 1
    for i in range(open_idx, len(text)):
        ch = text[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


_GD_DIRECT_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")
_GD_METHOD_CALL = re.compile(r"\.([A-Za-z_]\w*)\s*\(")
_GD_IDENT = re.compile(r"[A-Za-z_]\w*")


def resolved_arg_callees(mod: GDModule, resolvers: set[str]) -> set[str]:
    """GDScript counterpart to :func:`saddle.codemap.pyref.resolved_arg_callees`
    — one-hop interprocedural coverage, regex-derived. Within each function body
    it tracks locals bound to a resolver result (`var cd = resolve_cd(inst)`),
    then for every call collects the callee whose argument list contains a
    resolver call OR one of those resolved locals. Same deliberate trade as the
    Python path: a callee that also reads the base for another purpose is treated
    as covered. Lower fidelity (line-scoped, balanced-paren args) by design."""
    if not resolvers:
        return set()
    res_alt = "|".join(re.escape(r) for r in resolvers)
    assign_re = re.compile(r"^\s*(?:var\s+)?(\w+)\s*=\s*(?:%s)\s*\(" % res_alt)
    res_call_re = re.compile(r"(?:%s)\s*\(" % res_alt)

    per_func: dict[str | None, list[str]] = {}
    for lineno, ln in _code_lines(mod):
        per_func.setdefault(mod.enclosing_func(lineno), []).append(
            _strip_inline_comment(ln)
        )

    out: set[str] = set()
    for codes in per_func.values():
        resolved: set[str] = set()
        for code in codes:
            m = assign_re.match(code)
            if m:
                resolved.add(m.group(1))
        for code in codes:
            for cm in list(_GD_DIRECT_CALL.finditer(code)) + list(
                _GD_METHOD_CALL.finditer(code)
            ):
                callee = cm.group(1)
                if callee in resolvers or callee in _GD_KEYWORDS:
                    continue
                args = _paren_span(code, cm.end() - 1)
                if args is None:
                    continue
                if res_call_re.search(args) or any(
                    tok in resolved for tok in _GD_IDENT.findall(args)
                ):
                    out.add(callee)
    return out


def _carrier_alt(carrier: str) -> str:
    """Regex alternation matching this carrier. A BARE carrier (``kind``) matches
    a bare name, any ``["kind"]`` key, or any ``.kind`` attribute. A QUALIFIED
    carrier (``effect.kind``) matches ONLY ``effect.kind`` / ``effect["kind"]`` —
    the regex mirror of :func:`saddle.codemap.pyref._carrier_matches`, so an
    overloaded attribute name reused across namespaces is disambiguated here too."""
    obj, sep, attr = carrier.rpartition(".")
    if not sep:
        ce = re.escape(carrier)
        return r'(?:\b%s\b|\["%s"\]|\.%s\b)' % (ce, ce, ce)
    oe, ae = re.escape(obj), re.escape(attr)
    return r'(?:\b%s\.%s\b|\b%s\["%s"\])' % (oe, ae, oe, ae)


def identity_refs(mod: GDModule, carriers: set[str]) -> list[tuple[str, Ref]]:
    """A string literal compared/assigned to a carrier, or a `match <carrier>:`
    case value. Covers `==`/`!=`, assignment, and match blocks — the spread of
    forms RayXI's match-only scan missed."""
    out: list[tuple[str, Ref]] = []
    cmp_res = []
    for c in carriers:
        alt = _carrier_alt(c)
        cmp_res.append(re.compile(alt + r'\s*(?:==|!=)\s*["\']([^"\']+)["\']'))
        cmp_res.append(re.compile(r'["\']([^"\']+)["\']\s*(?:==|!=)\s*' + alt))
        cmp_res.append(re.compile(alt + r'\s*=(?!=)\s*["\']([^"\']+)["\']'))
    case_re = re.compile(r'^\s*["\']([^"\']+)["\']\s*:')
    match_re = re.compile(r"^\s*match\s+(.+?):")
    in_match_for: str | None = None
    match_indent = -1
    for idx, ln in _code_lines(mod):
        line = mod.lines[idx - 1]
        for rx in cmp_res:
            for m in rx.finditer(line):
                out.append((m.group(1), _mk(mod, idx, "compare", m.group(1))))
        mm = match_re.match(line)
        if mm:
            subj = mm.group(1).strip()
            in_match_for = subj if any(
                re.fullmatch(_carrier_alt(c), subj) for c in carriers
            ) else None
            match_indent = len(line) - len(line.lstrip())
            continue
        if in_match_for is not None:
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= match_indent:
                in_match_for = None
            else:
                cm = case_re.match(line)
                if cm:
                    out.append((cm.group(1), _mk(mod, idx, "match_case", cm.group(1))))
    return out


def collection_decls(mod: GDModule, name: str) -> list[tuple[set[str], Ref]]:
    """`const/var NAME = [ "a", "b" ]` (or `{...}`/`(...)`) of string literals on
    one line. Multi-line collections are a later refinement."""
    ne = re.escape(name)
    decl = re.compile(
        r"^\s*(?:const\s+|var\s+)?%s\s*(?::\s*\w+)?\s*=\s*[\[{(](.*)[\]})]" % ne
    )
    lit = re.compile(r'["\']([^"\']+)["\']')
    out: list[tuple[set[str], Ref]] = []
    for lineno, ln in _code_lines(mod):
        m = decl.match(ln)
        if m:
            vals = lit.findall(m.group(1))
            if vals:
                out.append((set(vals), _mk(mod, lineno, "assign", name)))
    return out


# Symbol-inventory scanners — same buckets as pyref.symbols, regex-derived.
_SYM_GET = re.compile(r'\.get\(\s*["\']([^"\']+)["\']')
_SYM_SUB = re.compile(r'\[\s*["\']([^"\']+)["\']\s*\]')
_SYM_ATTR = re.compile(r"\.([A-Za-z_]\w*)\b(?!\s*\()")       # `.name` NOT a call
_SYM_DIRECT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")    # `name(`
_SYM_METHOD = re.compile(r"\.([A-Za-z_]\w*)\s*\(")            # `.name(`
_SYM_DYNAMIC = re.compile(r'\.call\w*\(\s*["\']([^"\']+)["\']')  # `.call("name"`
_SYM_COLL = re.compile(r"^\s*(?:const\s+|var\s+)?(\w+)\s*(?::\s*\w+)?\s*=\s*[\[{(](.*)[\]})]")
_SYM_LIT = re.compile(r'["\']([^"\']+)["\']')


def _strip_inline_comment(line: str) -> str:
    """Drop a trailing ``# comment`` — but never a ``#`` inside a string literal
    (e.g. a colour `"#ff0000"`). Keeps English prose in comments out of the menu."""
    quote: str | None = None
    esc = False
    for i, ch in enumerate(line):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def symbols(mod: GDModule) -> dict:
    """Deterministic symbol inventory for ONE GDScript module — same four buckets
    and contract as :func:`saddle.codemap.pyref.symbols`, regex-derived. Keyword
    `name(` forms (if/for/match/...) are dropped so the calls bucket holds real
    candidate accessors, not control flow; inline comments are stripped and a
    `func NAME(` definition is recorded as a def, never as a call to itself.
    Lower fidelity than the AST path by design — a menu, not a gate."""
    fields: dict[str, int] = {}
    funcs: dict[str, int] = {}
    calls: dict[str, int] = {}
    collections: dict[str, list[str]] = {}

    def _bump(d: dict, key: str) -> None:
        d[key] = d.get(key, 0) + 1

    for _lineno, ln in _code_lines(mod):
        code = _strip_inline_comment(ln)
        cm = _SYM_COLL.match(code)
        if cm:
            vals = _SYM_LIT.findall(cm.group(2))
            if vals:
                collections[cm.group(1)] = vals
        fm = _FUNC_RE.match(code)
        if fm:
            _bump(funcs, fm.group(1))
            code = code[fm.end():]  # the `func NAME` head is a def, not a call
        for m in _SYM_GET.finditer(code):
            _bump(fields, m.group(1))
        for m in _SYM_SUB.finditer(code):
            _bump(fields, m.group(1))
        for m in _SYM_ATTR.finditer(code):
            _bump(fields, m.group(1))
        for m in _SYM_DIRECT.finditer(code):
            if m.group(1) not in _GD_KEYWORDS:
                _bump(calls, m.group(1))
        for m in _SYM_METHOD.finditer(code):
            _bump(calls, m.group(1))
        for m in _SYM_DYNAMIC.finditer(code):  # the resolver behind a dynamic call
            _bump(calls, m.group(1))
    return {"fields": fields, "funcs": funcs, "calls": calls, "collections": collections}
