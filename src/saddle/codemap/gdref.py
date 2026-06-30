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
    # The engine's authority conventions (NOT game vocabulary): Godot's built-in
    # `multiplayer.is_server()` / `is_multiplayer_authority()`, AND a project's own
    # `_is_server()` / `is_server()` helper — the near-universal wrapper a Godot
    # multiplayer project writes around the built-in (RayXI's is exactly this). The
    # bare `is_server` alternation also covers the `multiplayer.is_server(` form (a
    # word boundary holds after the dot), so it is not listed separately.
    r"\b(?:is_multiplayer_authority|is_multiplayer_master|_?is_server)\s*\("
)
# Receivers/appliers of replicated state run on the client.
_CLIENT_FUNC_RE = re.compile(r"^_?recv_|^apply_|_synced$")

# A script that EXTENDS one of these Godot UI base classes is the client presentation
# layer — a HUD / panel / menu / dialog. These are ENGINE classes (the CanvasItem→
# Control UI tree plus CanvasLayer/Window/Popup roots), NOT game vocabulary, so this
# bakes in no genre. A UI module's code runs on the CLIENT, so its module domain is
# "client" (see parse_modules) — the signal that tells a real HUD caller of a server
# mutator apart from a server-side service-to-service call. Conservative by design: a
# UI root extending a class not listed reads as "shared" (a missed flag, never a false
# one) and is a one-line addition to fix.
_GD_UI_BASES = frozenset({
    "Control", "CanvasLayer", "Window", "Popup", "PopupMenu", "PopupPanel",
    "AcceptDialog", "ConfirmationDialog", "FileDialog",
    "Panel", "PanelContainer", "MarginContainer", "CenterContainer", "BoxContainer",
    "VBoxContainer", "HBoxContainer", "GridContainer", "ScrollContainer",
    "FlowContainer", "HFlowContainer", "VFlowContainer", "TabContainer",
    "AspectRatioContainer", "SplitContainer", "HSplitContainer", "VSplitContainer",
    "ColorRect", "TextureRect", "NinePatchRect", "ReferenceRect",
    "Button", "Label", "RichTextLabel", "Tree", "ItemList", "TextEdit", "LineEdit",
})
_GD_EXTENDS_RE = re.compile(r"^\s*extends\s+([A-Za-z_]\w*)")


def _ui_extends(source: str) -> bool:
    """True if the script's top-level ``extends`` names a Godot UI base class — i.e.
    this module is a client-side HUD/panel. ``extends`` is unique and sits near the
    top (after an optional ``@tool`` / ``class_name``); scan the head for it."""
    for ln in source.splitlines()[:25]:
        m = _GD_EXTENDS_RE.match(ln)
        if m:
            return m.group(1) in _GD_UI_BASES
    return False

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
    out: list[GDModule] = []
    for path, src in sources:
        domain = _domain_from(path, src)
        # A UI script (extends Control/CanvasLayer/…) runs on the client; promote the
        # neutral "shared" to "client" so its handlers read as client-domain. An
        # explicit path/marker domain still wins (never demote an authored "server").
        if domain == "shared" and _ui_extends(src):
            domain = "client"
        out.append(GDModule(path, domain, src, src.splitlines()))
    return out


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


def _string_spans(line: str) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` interiors of every CLOSED single-line string
    literal on ``line`` (the text between the quotes, delimiters excluded), quote-
    and escape-aware like :func:`_strip_inline_comment` and :func:`_paren_span`.

    A field-access pattern whose match STARTS inside one of these spans is text
    inside a string — a generated doc ``"description": "... primary_swing.damage ..."``
    or an attribute-path string ``"auto_target.range_m"`` — never a real read, so the
    caller rejects it. Only a string CLOSED on the same line yields a span: an
    UNTERMINATED quote contributes nothing, leaving its tail treated as code, so a
    real read after it is KEPT. That is the FP-safe bias — saddle never drops a
    finding on a tokenizer guess (a missed real read is the cardinal sin; an extra
    false positive is merely noise)."""
    spans: list[tuple[int, int]] = []
    quote: str | None = None
    esc = False
    start = 0
    for i, ch in enumerate(line):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                spans.append((start, i))
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            start = i + 1
    return spans


def _in_string(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def field_reads(mod: GDModule, field: str) -> list[Ref]:
    """`x.get("field"...)` (always read), `x["field"]` / `x.field` when NOT
    immediately assigned to.

    Inline comments are stripped and any match that starts INSIDE a string literal
    is rejected, so a field name merely MENTIONED in a comment or a generated doc /
    attribute-path string (``"... primary_swing.damage ..."``, ``"auto_target.range_m"``)
    is never miscounted as a real read. Those mentions are the regex adapter's
    false-positive class — pyref is immune because it walks the AST, where a string
    constant can't be confused with an attribute access — and unchecked they bury the
    genuine un-resolved base reads under noise (the meta-failure saddle exists to
    avoid). The legitimate forms all start in CODE (``.get(``, ``[``, a bare ``.``
    outside quotes), so this only ever removes false positives, never a real read."""
    f = re.escape(field)
    get_re = re.compile(r'\.get\(\s*["\']%s["\']' % f)
    sub_re = re.compile(r'\[\s*["\']%s["\']\s*\]' % f)
    attr_re = re.compile(r"\.%s\b" % f)
    refs: list[Ref] = []
    for lineno, raw in _code_lines(mod):
        ln = _strip_inline_comment(raw)
        spans = _string_spans(ln)
        for m in get_re.finditer(ln):
            if not _in_string(m.start(), spans):
                refs.append(_mk(mod, lineno, "get_read", field))
        for m in sub_re.finditer(ln):
            if _in_string(m.start(), spans):
                continue
            if not _ASSIGN_AHEAD.match(ln[m.end():]):
                refs.append(_mk(mod, lineno, "subscript_read", field))
        for m in attr_re.finditer(ln):
            if _in_string(m.start(), spans):
                continue
            # don't double-count the `.get(` head as an attr read of `get`
            if not _ASSIGN_AHEAD.match(ln[m.end():]):
                refs.append(_mk(mod, lineno, "attr_read", field))
    return refs


def field_writes(mod: GDModule, field: str) -> list[Ref]:
    """`x["field"] = ...` / `x.field = ...` (and the augmented forms). Inline
    comments are stripped and in-string matches rejected for the same reason as
    :func:`field_reads`: a ``"set foo.damage = 5"`` doc string is text, not a write.
    The FP-safe bias holds — a real write is in code, so it is never dropped."""
    f = re.escape(field)
    sub_w = re.compile(r'\[\s*["\']%s["\']\s*\]\s*(?:=(?!=)|\+=|-=|\*=|/=)' % f)
    attr_w = re.compile(r"\.%s\b\s*(?:=(?!=)|\+=|-=|\*=|/=)" % f)
    refs: list[Ref] = []
    for lineno, raw in _code_lines(mod):
        ln = _strip_inline_comment(raw)
        spans = _string_spans(ln)
        if any(not _in_string(m.start(), spans) for m in sub_w.finditer(ln)):
            refs.append(_mk(mod, lineno, "subscript_write", field))
        elif any(not _in_string(m.start(), spans) for m in attr_w.finditer(ln)):
            refs.append(_mk(mod, lineno, "attr_write", field))
    return refs


def calls_to(mod: GDModule, callee: str) -> list[Ref]:
    """`callee(...)`, `x.callee(...)`, and the dynamic `x.call("callee", ...)` /
    `x.call_deferred("callee", ...)` forms RayXI's runtime uses. The `func
    callee(...)` DEFINITION line is excluded, and `has_method("callee")` never
    matches (it isn't a `.call*(` form)."""
    c = re.escape(callee)
    direct = re.compile(r"(?<![\w.])%s\s*\(" % c)
    # A method call `recv.callee(` — but NOT when `recv` is a STRING LITERAL
    # (`"sep".join(...)`), which is a String/Array builtin, never a project function.
    # The negative lookbehind for a closing quote kills that whole class of generic-
    # method-name collision (e.g. a service's `join`/`collect`/`earn` mutator vs
    # `"\n".join(parts)`) that would otherwise attribute every string join to the
    # service. A service handle is always an identifier / node lookup, never a literal.
    method = re.compile("(?<![\"'])\\.%s\\s*\\(" % c)
    def_re = re.compile(r"^\s*(?:static\s+)?func\s+%s\b" % c)
    hit_lines: set[int] = set()
    # Direct / method forms keep the callee name and its `(` adjacent (GDScript never
    # parses `name\n(...)` as a call), so a line-scoped scan sees them whole.
    for lineno, ln in _code_lines(mod):
        if def_re.match(ln):
            continue
        if direct.search(ln) or method.search(ln):
            hit_lines.add(lineno)
    # The DYNAMIC form is different: `str(ah.call(\n    "list_item", ...))` puts the
    # `.call(` and the string that names the REAL callee on different lines, so a
    # line-scoped scan misses it — and a genuinely-routed mutator then reads as
    # unrouted (a false congruence gap, exactly what auction_house.list_item hit).
    # Scan it over the whole comment-stripped source instead: `\s*` already crosses
    # the newline, and the hit is attributed to the line the `.call(` opens on (the
    # call's own line, where its enclosing @rpc receiver is). Inline comments are
    # stripped so a `.call("x")` inside a comment never counts.
    dynamic = re.compile(r'\.call\w*\(\s*["\']%s["\']' % c)
    stripped = [_strip_inline_comment(ln) for ln in mod.lines]
    joined = "\n".join(stripped)
    for m in dynamic.finditer(joined):
        hit_lines.add(joined.count("\n", 0, m.start()) + 1)
    return [_mk(mod, n, "call", callee) for n in sorted(hit_lines)]


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
# Dynamic-dispatch methods: a resolved value sitting in their argument list is
# being dispatched THROUGH them (the real callee is the string they take), not
# handed to a consumer named after the method, so they are never the "covered"
# callee. (`def = obj.call("resolve_def", aid, def)` must not mark `call` covered.)
_GD_DISPATCH_METHODS = frozenset({"call", "call_deferred", "callv", "rpc", "rpc_id"})
# A `return <expr>` line and a simple `[var] name[: T] [:]= <rhs>` binding — the two
# statement shapes the return-polarity fixpoint reads (a function's own returns, and
# the locals it binds from project calls). `:?=` accepts GDScript's inferred-type
# `:=` alongside plain `=`; the `(?<!...)`-free form is safe because a return line
# never matches the assign RE and an augmented/compare line (`+=`, `==`) fails the
# bare `=` after the name (see _call_callee's callers, which reject non-call RHS).
_GD_RETURN_RE = re.compile(r"^\s*return\s+(.+?)\s*$")
_GD_ASSIGN_RE = re.compile(r"^\s*(?:var\s+)?(\w+)\s*(?::\s*[^=]+?)?\s*:?=\s*(.+?)\s*$")


def _call_callee(expr: str) -> str | None:
    """If ``expr`` is EXACTLY one direct call ``callee(...)`` spanning the WHOLE
    expression — not a method/attribute call (``obj.callee(...)``) and not part of a
    larger expression (``callee(...) + 1``, ``a if callee(...) else b``) — return
    ``callee``; else None. The GDScript mirror of pyref's bare-``ast.Name``-func
    guard: only an unambiguous top-level direct call carries a ``("call", fn)``
    return-polarity atom, so a method name colliding with an unrelated top-level
    wrapper cannot leak a polarity it does not have."""
    s = expr.strip()
    m = re.match(r"([A-Za-z_]\w*)\s*\(", s)
    if not m:
        return None
    callee = m.group(1)
    if callee in _GD_KEYWORDS:
        return None
    open_idx = m.end() - 1                       # index of the '(' (paren may follow ws)
    span = _paren_span(s, open_idx)
    if span is None:                             # unbalanced on this line -> no atom
        return None
    if s[open_idx + len(span) + 2:].strip():     # anything after the matching ')' -> compound
        return None
    return callee


def resolved_arg_callees(mod: GDModule, resolvers: set[str]) -> set[str]:
    """GDScript counterpart to :func:`saddle.codemap.pyref.resolved_arg_callees`
    — one-hop interprocedural coverage, regex-derived. Within each function body
    it tracks locals bound to a resolver result, then for every call collects the
    callee whose argument list contains a resolver call OR one of those resolved
    locals. Same deliberate trade as the Python path: a callee that also reads the
    base for another purpose is treated as covered. Lower fidelity (line-scoped,
    balanced-paren args) by design.

    Both call forms count as a resolver invocation: a direct/method call
    ``resolve_def(...)`` / ``obj.resolve_def(...)`` AND the DYNAMIC dispatch
    RayXI's runtime actually uses, ``obj.call("resolve_def", ...)`` /
    ``obj.call_deferred("resolve_def", ...)``. Recognising only the direct form
    (the original gap) read ``def = _talents.call("resolve_def", aid, def)`` as
    unresolved, so every helper handed that ``def`` was a false positive — the
    noise that buried the real HUD/tooltip gaps on RayXI."""
    if not resolvers:
        return set()
    res_alt = "|".join(re.escape(r) for r in resolvers)
    # A resolver invocation in either form, used to spot an inline resolve inside
    # a callee's argument list (`helper(x, resolve_def(x))` / the dynamic form).
    res_call_re = re.compile(
        r'(?:(?:%s)\s*\(|\.call\w*\(\s*["\'](?:%s)["\'])' % (res_alt, res_alt)
    )
    # A local bound to a resolver result, either call form. The optional `: Type`
    # annotation is tolerated (`var cd: float = resolve_cd(inst)`).
    assign_re = re.compile(
        r'^\s*(?:var\s+)?(\w+)\s*(?::\s*[^=]+?)?\s*=\s*'
        r'(?:(?:%s)\s*\(|[\w.]+\.call\w*\(\s*["\'](?:%s)["\'])' % (res_alt, res_alt)
    )

    # Group every code line under its enclosing func in ONE forward pass. The old
    # form called mod.enclosing_func(lineno) per line, and that scans BACKWARD to
    # line 0 each time — O(lines^2) per module, ~220s over a 237-file project. Only
    # `func` lines move the boundary and they all survive _code_lines (a func is
    # never a comment), so tracking the last-seen func forward is identical and O(n).
    per_func: dict[str | None, list[str]] = {}
    cur_fn: str | None = None
    for lineno, ln in _code_lines(mod):
        fm = _FUNC_RE.match(ln)
        if fm:
            cur_fn = fm.group(1)
        per_func.setdefault(cur_fn, []).append(_strip_inline_comment(ln))

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
                if (callee in resolvers or callee in _GD_KEYWORDS
                        or callee in _GD_DISPATCH_METHODS):
                    continue
                args = _paren_span(code, cm.end() - 1)
                if args is None:
                    continue
                if res_call_re.search(args) or any(
                    tok in resolved for tok in _GD_IDENT.findall(args)
                ):
                    out.add(callee)
    return out


def _ordered_param_names(mod: GDModule, func_name: str) -> list[str]:
    """Parameter names of ``func func_name`` IN ORDER (``_func_param_names`` returns
    a set; the pass-down fixpoint needs positions). Empty list for a zero-param or
    unparseable signature."""
    parts = _func_params(mod, func_name)
    if not parts:
        return []
    out: list[str] = []
    for part in parts:
        mm = re.match(r"\s*(\w+)", part)
        out.append(mm.group(1) if mm else "")
    return out


def _passdown_atom(arg: str, res_call_re, base_call_re):
    """Classify one call argument OR return expression for the pass-down fixpoint
    (see codemap/passdown.Atom): an inline resolver call -> ``"R"``, an inline
    base_source call -> ``"B"``, a bare identifier -> ``("n", name)``, a bare DIRECT
    call to a project function -> ``("call", fn)`` (carries fn's RETURN polarity, so a
    resolver-wrapper passes its resolved-ness through), anything else -> ``None``.
    Resolver/base checks win first, so ``apply_ability_modifiers(...)`` is ``"R"``,
    never ``("call", ...)``.

    What this CANNOT see is the documented fail-OPEN residual: a base value laundered
    through a cast/reassignment (``base = raw as Dictionary; return base``) reads as
    neither ``"B"`` nor a base-local, so a wrapper that returns it on one path and a
    resolved value on another is still blessed (existential-R, structural-B-veto).
    That is the return-axis twin of the laundered-base *param* residual — recorded as
    a known false-negative class, never papered over with a runtime assumption."""
    if res_call_re.search(arg):
        return "R"
    if base_call_re is not None and base_call_re.search(arg):
        return "B"
    s = arg.strip()
    if re.fullmatch(r"\w+", s):
        return ("n", s)
    callee = _call_callee(s)
    if callee is not None:
        return ("call", callee)
    return None


def passdown_facts(mod: GDModule, resolvers: set[str], base_sources=frozenset()):
    """GDScript extractor for the interprocedural pass-down fixpoint — the
    cross-module generalisation of :func:`resolved_arg_callees`. Returns a
    :class:`saddle.codemap.passdown.ModuleFacts`: this module's function parameter
    lists, its resolver-bound and base_source-bound locals per function, and every
    outgoing call site with its positional arguments pre-classified as resolver /
    base / bare-name / other. The fixpoint in ``passdown.py`` stitches these across
    modules; this stays purely per-module and regex-derived, same fidelity trade as
    the one-hop scan (line-scoped, balanced-paren args)."""
    from .passdown import ModuleFacts

    facts = ModuleFacts()
    if not resolvers:
        return facts
    res_alt = "|".join(re.escape(r) for r in resolvers)
    res_call_re = re.compile(
        r'(?:(?:%s)\s*\(|\.call\w*\(\s*["\'](?:%s)["\'])' % (res_alt, res_alt)
    )
    assign_re = re.compile(
        r'^\s*(?:var\s+)?(\w+)\s*(?::\s*[^=]+?)?\s*=\s*'
        r'(?:(?:%s)\s*\(|[\w.]+\.call\w*\(\s*["\'](?:%s)["\'])' % (res_alt, res_alt)
    )
    base_call_re = base_assign_re = None
    if base_sources:
        base_alt = "|".join(re.escape(b) for b in base_sources)
        base_call_re = re.compile(
            r'(?:(?:%s)\s*\(|\.call\w*\(\s*["\'](?:%s)["\'])' % (base_alt, base_alt)
        )
        base_assign_re = re.compile(
            r'^\s*(?:var\s+)?(\w+)\s*(?::\s*[^=]+?)?\s*=\s*'
            r'(?:(?:%s)\s*\(|[\w.]+\.call\w*\(\s*["\'](?:%s)["\'])' % (base_alt, base_alt)
        )

    # Ordered parameter names for every function this module defines.
    for name in symbols(mod)["funcs"].keys():
        facts.params[name] = _ordered_param_names(mod, name)

    # Group every code line under its enclosing func in ONE forward pass. The old
    # form called mod.enclosing_func(lineno) per line, and that scans BACKWARD to
    # line 0 each time — O(lines^2) per module, ~220s over a 237-file project. Only
    # `func` lines move the boundary and they all survive _code_lines (a func is
    # never a comment), so tracking the last-seen func forward is identical and O(n).
    per_func: dict[str | None, list[str]] = {}
    cur_fn: str | None = None
    for lineno, ln in _code_lines(mod):
        fm = _FUNC_RE.match(ln)
        if fm:
            cur_fn = fm.group(1)
        per_func.setdefault(cur_fn, []).append(_strip_inline_comment(ln))

    dyn_name_re = re.compile(r'\.call\w*\(\s*["\'](\w+)["\']')
    for fn, codes in per_func.items():
        resolved: set[str] = set()
        based: set[str] = set()
        called: dict[str, str] = {}   # local -> project callee it is bound from
        rets: list = []               # one classified atom per `return <expr>`
        for code in codes:
            am = assign_re.match(code)
            if am:
                resolved.add(am.group(1))
            bm = base_assign_re.match(code) if base_assign_re is not None else None
            if bm:
                based.add(bm.group(1))
            # call_locals: a local bound from a bare PROJECT call (not a resolver/base
            # binding, which the two branches above already claimed) carries that
            # callee's return polarity -> `var d = wrap(id)` resolves if wrap is a
            # resolver-wrapper. The fixpoint stitches d's polarity in passdown.py.
            if am is None and bm is None:
                gm = _GD_ASSIGN_RE.match(code)
                if gm:
                    callee = _call_callee(gm.group(2))
                    if (callee is not None and callee not in resolvers
                            and callee not in base_sources):
                        called[gm.group(1)] = callee
            # returns: classify this function's own return expressions so the fixpoint
            # can decide whether IT is a resolver-wrapper (R-returning, not B-returning).
            rm = _GD_RETURN_RE.match(code)
            if rm:
                rets.append(_passdown_atom(rm.group(1), res_call_re, base_call_re))
        if resolved:
            facts.resolved_locals[fn] = resolved
        if based:
            facts.base_locals[fn] = based
        if called:
            facts.call_locals[fn] = called
        if rets:
            facts.returns[fn] = rets
        for code in codes:
            if code.lstrip().startswith("func "):
                continue  # the signature line is a definition, not a call
            # direct + method calls: callee(args) / obj.callee(args)
            for cm in list(_GD_DIRECT_CALL.finditer(code)) + list(
                _GD_METHOD_CALL.finditer(code)
            ):
                callee = cm.group(1)
                if (callee in resolvers or callee in _GD_KEYWORDS
                        or callee in _GD_DISPATCH_METHODS):
                    continue
                span = _paren_span(code, cm.end() - 1)
                if span is None:
                    continue
                args = [a for a in _split_top_level(span)] if span.strip() else []
                atoms = [_passdown_atom(a, res_call_re, base_call_re) for a in args]
                facts.sites.append((fn, callee, atoms))
            # dynamic dispatch: obj.call("callee", args) — drop the method-name arg
            for dm in dyn_name_re.finditer(code):
                callee = dm.group(1)
                oi = code.find("(", dm.start())
                span = _paren_span(code, oi) if oi >= 0 else None
                if span is None:
                    continue
                parts = [a for a in _split_top_level(span)] if span.strip() else []
                args = parts[1:]
                atoms = [_passdown_atom(a, res_call_re, base_call_re) for a in args]
                facts.sites.append((fn, callee, atoms))
    return facts


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


def name_decls(mod: GDModule, name: str) -> list[Ref]:
    """Declaration sites of a script-scope symbol `name`: an `@export var NAME`,
    a script-scope (indent-0) `var NAME`, a `const NAME`, or a `signal NAME` —
    GDScript's design knobs (a value a designer or other code is meant to consume).
    Indent-0 only: a `var NAME` inside a function is a local, a different binding,
    not the knob (mirrors pyref's module/class-scope rule). Leading annotations
    (`@export`, `@export_range(...)`, `@onready`) are tolerated before the keyword."""
    ne = re.escape(name)
    decl_re = re.compile(
        r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:const|signal|var)\s+%s\b" % ne
    )
    out: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        if decl_re.match(_strip_inline_comment(ln)):
            out.append(_mk(mod, lineno, "declaration", name))
    return out


def name_uses(mod: GDModule, name: str) -> list[Ref]:
    r"""Every READ of `name` — any ``\bNAME\b`` token that is NOT the declaration's
    own LHS target. Deliberately permissive (matches bare `NAME`, `self.NAME`,
    `obj.NAME`, and a signal referenced by string in `emit_signal("NAME")`): the
    conservative bias for a liveness check is to over-count reads so a symbol is
    flagged DEAD only when truly nothing references it — a false 'dead' is worse
    than a missed one. The declaration's own target token is excluded so a knob
    isn't counted as its own reader (which would make the check a no-op)."""
    ne = re.escape(name)
    tok = re.compile(r"\b%s\b" % ne)
    decl_re = re.compile(
        r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:const|signal|var)\s+(%s)\b" % ne
    )
    decl_at: dict[int, int] = {}  # lineno -> start offset of the declared token
    for lineno, ln in _code_lines(mod):
        m = decl_re.match(_strip_inline_comment(ln))
        if m:
            decl_at[lineno] = m.start(1)
    out: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        code = _strip_inline_comment(ln)
        skip = decl_at.get(lineno)
        for m in tok.finditer(code):
            if skip is not None and m.start() == skip:
                continue
            out.append(_mk(mod, lineno, "use", name))
    return out


def function_defs(mod: GDModule, name: str) -> list[Ref]:
    """Definition site(s) of a function named `name` (`func name(...)`)."""
    out: list[Ref] = []
    for lineno, ln in _code_lines(mod):
        m = _FUNC_RE.match(ln)
        if m and m.group(1) == name:
            out.append(_mk(mod, lineno, "func_def", name))
    return out


def rpc_receivers(mod: GDModule) -> set[str]:
    """Function names carrying an ``@rpc`` annotation — the project's REMOTE entry
    points (Godot's ``@rpc(...)`` networking annotation). A replicated-state mutator
    CALLED from one of these has a real server route (the handler runs on the
    authority even though a remote peer triggers it); a mutator reached only from raw
    UI handlers has none. Engine vocabulary, not game vocabulary. The annotation sits
    in the contiguous ``@``-block directly above the ``func`` line (the same block
    :meth:`GDModule._compute_func_domain` reads for the RPC domain signal)."""
    out: set[str] = set()
    for i, ln in enumerate(mod.lines):
        m = _FUNC_RE.match(ln)
        if not m:
            continue
        j = i - 1
        while j >= 0 and mod.lines[j].strip().startswith("@"):
            if "@rpc" in mod.lines[j]:
                out.add(m.group(1))
                break
            j -= 1
    return out


# A write OPERATOR — assignment or any augmented form. `=(?!=)` excludes `==`.
_WRITE_OP = r"(?:=(?!=)|\+=|-=|\*=|/=|\|=|&=)"
# A subscript write `IDENT[...] =` / `obj.ATTR[...] =`; group is the BASE container
# being mutated (so `_state[key] = x` reports `_state`, the replicated dict, not the key).
_GD_SUB_W = re.compile(r"(?:\.(\w+)|\b(\w+))\s*\[[^\]]*\]\s*%s" % _WRITE_OP)
# A member write `obj.ATTR =`.
_GD_ATTR_W = re.compile(r"\.(\w+)\b\s*%s" % _WRITE_OP)
# A bare script-scope reassignment `NAME =` (a `var`/`const` decl is excluded by the
# caller). This is how GDScript writes a script-level var (`_state = {...}`), which a
# member-only scan would miss.
_GD_BARE_W = re.compile(r"^(\w+)\s*%s" % _WRITE_OP)
# Container-mutating METHOD calls — `_requests.erase(rid)`, `_log.append(x)`,
# `_slots.clear()`. GDScript mutates a Dictionary/Array IN PLACE through these
# built-ins, so a function whose ONLY state change is `_field.erase(k)` is still a
# MUTATOR even though it never assigns the field. Without this, a gated mutator that
# only erases/appends/clears replicated state (friends_list.decline_request →
# `_requests.erase`) reads as a pure reader and its unrouted client gap is missed.
# Longer names lead the alternation so `sort`/`append`/`pop` can't shadow
# `sort_custom`/`append_array`/`pop_*` (the trailing `(` already disambiguates; order
# keeps it obvious). The base must be a bare member identifier (optionally `self.`-
# qualified); a chained/subscripted receiver (`d.get(k).erase()`, `a[i].append()`) is
# deliberately NOT matched — its base is indeterminate, the conservative miss the
# adapter prefers — and the local-name exclusion in func_writes drops a mutation on a
# scratch local (the by-reference `friends_a.erase` case, whose real state write is the
# explicit `_state[account] = sa` reassignment alongside it).
_GD_MUT_METHODS = (
    "append_array", "push_back", "push_front", "pop_back", "pop_front", "pop_at",
    "remove_at", "sort_custom", "erase", "clear", "append", "insert", "merge",
    "assign", "fill", "resize", "sort", "reverse", "shuffle",
)
_GD_METHOD_MUT = re.compile(
    r"(?<![\w.\]])(?:self\.)?(\w+)\.(?:%s)\s*\(" % "|".join(_GD_MUT_METHODS)
)


def _func_body_lines(mod: GDModule, func_name: str):
    """Yield (lineno, raw-line) for every line in the BODY of each `func func_name`
    — indentation-delimited, the same body model :meth:`GDModule._compute_func_domain`
    uses. The body ends at the first non-blank line indented no deeper than the def."""
    fn_re = re.compile(r"^(\s*)(?:static\s+)?func\s+%s\b" % re.escape(func_name))
    i, n = 0, len(mod.lines)
    while i < n:
        m = fn_re.match(mod.lines[i])
        if not m:
            i += 1
            continue
        def_indent = len(m.group(1))
        j = i + 1
        while j < n:
            ln = mod.lines[j]
            if not ln.strip():
                j += 1
                continue
            if len(ln) - len(ln.lstrip()) <= def_indent:
                break
            yield j + 1, ln
            j += 1
        i = j


# A local binding declared inside a function body: `var NAME` / `const NAME`, or a
# `for NAME in` loop variable. Captured (not just skipped) so a later write to that
# name is known to be a SCRATCH-variable mutation, not a state write.
_GD_LOCAL_DECL = re.compile(r"^\s*(?:var|const)\s+(\w+)")
_GD_FOR_VAR = re.compile(r"^\s*for\s+(\w+)\s+in\b")


def _split_top_level(s: str) -> list[str]:
    """Split ``s`` on commas that sit at bracket/quote depth ZERO — so a default
    value or type argument that itself contains commas (``a, b = [1, 2], c := {}``)
    stays one part. The complement of a naive ``.split(",")``; needed wherever a
    comma count must mean argument/parameter count, not literal commas."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    esc = False
    for ch in s:
        if quote is not None:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _func_param_text(mod: GDModule, func_name: str) -> str | None:
    """The raw text BETWEEN the parens of ``func func_name(...)`` — a balanced,
    quote-aware scan so a multi-line ``func f(\\n  a: int,\\n  b := 0)`` and a string
    default containing a paren are both handled. ``None`` if the function is not
    defined in this module (the caller must then NOT make an arity judgement)."""
    fn_re = re.compile(r"^\s*(?:static\s+)?func\s+%s\s*\(" % re.escape(func_name))
    n = len(mod.lines)
    for i in range(n):
        if not fn_re.match(mod.lines[i]):
            continue
        buf: list[str] = []
        depth = 0
        quote: str | None = None
        esc = False
        done = False
        for j in range(i, n):
            for ch in mod.lines[j]:
                if quote is not None:
                    buf.append(ch)
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == quote:
                        quote = None
                    continue
                if ch in "\"'":
                    quote = ch
                    buf.append(ch)
                    continue
                if ch == "(":
                    depth += 1
                    if depth == 1:
                        continue
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        done = True
                        break
                if depth >= 1:
                    buf.append(ch)
            if done:
                break
        return "".join(buf)
    return None


def _func_params(mod: GDModule, func_name: str) -> list[str] | None:
    """The parameter list of ``func func_name`` as top-level parts (one per param),
    or ``None`` if the function is not defined here. Empty list = a zero-parameter
    function. Top-level splitting keeps a comma-bearing default in one part."""
    text = _func_param_text(mod, func_name)
    if text is None:
        return None
    return [p for p in _split_top_level(text) if p.strip()]


def _func_param_names(mod: GDModule, func_name: str) -> set[str]:
    """Parameter names of ``func func_name``. Each parameter is a LOCAL, so a write
    to it is not a state write."""
    parts = _func_params(mod, func_name)
    if not parts:
        return set()
    names: set[str] = set()
    for part in parts:
        mm = re.match(r"\s*(\w+)", part)
        if mm:
            names.add(mm.group(1))
    return names


def func_arity(mod: GDModule, func_name: str) -> tuple[int, int] | None:
    """``(min_required, max_total)`` parameter counts of ``func func_name``, or
    ``None`` if it is not defined in this module. A parameter carrying ``=`` (a
    default, including the ``:=`` inferred form) is optional, so ``min_required``
    counts only the parameters WITHOUT a default and ``max_total`` counts them all.
    GDScript user functions take no varargs, so ``max_total`` is exact.

    This is what lets the congruence caller-scan tell two same-named mutators on
    DIFFERENT services apart: ``guild_system.invite`` takes 3 parameters and
    ``group_framework.invite`` takes 2, so a 3-argument HUD call is attributable to
    the guild service only. Name-equality alone (the old behaviour) charged the call
    to both and invented a phantom gap on the party service."""
    parts = _func_params(mod, func_name)
    if parts is None:
        return None
    if not parts:
        return (0, 0)
    required = sum(1 for p in parts if "=" not in p)
    return (required, len(parts))


def call_arg_counts(mod: GDModule, callee: str) -> dict[int, int]:
    """Map each line that CALLS ``callee`` to the number of arguments passed at that
    site — the call-side complement of :func:`func_arity`, so the caller-scan can drop
    a name-collision attribution whose argument count cannot fit the definition.

    All three call forms :func:`calls_to` recognises are measured: the direct
    ``callee(a, b)`` and method ``x.callee(a, b)`` forms count their top-level
    arguments; the dynamic ``x.call("callee", a, b)`` form counts the arguments AFTER
    the method-name string (it is dispatched THROUGH ``call``, so the name itself is
    not one of ``callee``'s arguments). A call whose parentheses do not balance on its
    own line is OMITTED (unknown, never guessed) — the same conservative bias the rest
    of the adapter holds, so an undeterminable arity never deletes a real caller. When
    a line holds several calls to ``callee`` the largest count wins."""
    c = re.escape(callee)
    direct = re.compile(r"(?<![\w.])%s\s*\(" % c)
    method = re.compile("(?<![\"'])\\.%s\\s*\\(" % c)
    dynamic = re.compile(r'\.call\w*\(\s*["\']%s["\']' % c)
    def_re = re.compile(r"^\s*(?:static\s+)?func\s+%s\b" % c)
    out: dict[int, int] = {}
    for lineno, raw in _code_lines(mod):
        if def_re.match(raw):
            continue
        ln = _strip_inline_comment(raw)
        best: int | None = None
        # Dynamic dispatch first: the `(` belongs to `.call(`, and arg[0] is the
        # method-name string, so the real argument count is parts-minus-one.
        for dm in dynamic.finditer(ln):
            open_idx = ln.find("(", dm.start())
            if open_idx < 0:
                continue
            inside = _paren_span(ln, open_idx)
            if inside is None:
                continue
            n = max(0, len([p for p in _split_top_level(inside) if p.strip()]) - 1)
            best = n if best is None else max(best, n)
        # Direct / method forms: every top-level part is a real argument.
        for rx in (direct, method):
            for mm in rx.finditer(ln):
                open_idx = mm.end() - 1  # the regex ends ON the '('
                inside = _paren_span(ln, open_idx)
                if inside is None:
                    continue
                n = len([p for p in _split_top_level(inside) if p.strip()])
                best = n if best is None else max(best, n)
        if best is not None:
            out[lineno] = best
    return out


def _func_locals(mod: GDModule, func_name: str) -> set[str]:
    """The names LOCAL to ``func_name``: its parameters plus every ``var`` / ``const``
    / ``for`` binding in its body. A write whose base is one of these is a local-
    variable mutation, NOT replicated state — the difference between ``_slots[k] = v``
    (a member dict) and ``d[k] = null`` (a scratch dict built in a getter). Without
    this exclusion :func:`func_writes` misreads a pure getter that builds a local via a
    helper (``inventory_runtime.slots_of`` → ``_empty_slots``) as a state mutator, and
    the congruence axis raises a phantom gap on it."""
    locals_ = _func_param_names(mod, func_name)
    for _lineno, ln in _func_body_lines(mod, func_name):
        md = _GD_LOCAL_DECL.match(ln)
        if md:
            locals_.add(md.group(1))
        mf = _GD_FOR_VAR.match(ln)
        if mf:
            locals_.add(mf.group(1))
    return locals_


# A service-locator resolution: ``_service("name")`` / ``get_service("name")`` (the
# wrapper a HUD calls) OR the dynamic ``registry.call("get_service", "name")`` form. The
# captured group is the service NAME — what a receiver variable is bound to.
_GD_SERVICE_RESOLVE = re.compile(
    r'(?:_service|get_service|resolve_service)\s*\(\s*["\'](\w+)["\']'
    r'|\.call\w*\(\s*["\']get_service["\']\s*,\s*["\'](\w+)["\']'
)
# A self-registration ``register("name", self)`` (direct) or the dynamic
# ``registry.call("register", "name", self)`` — the module's OWN declared service name.
# ``self`` as the registered node is required so registering ANOTHER node never counts.
_GD_SELF_REGISTER = re.compile(
    r'(?:^|\.call\w*\(\s*["\']register["\']\s*,\s*|[^.\w]register\s*\(\s*)'
    r'["\'](\w+)["\']\s*,\s*self\b'
)
# `recv = ...` / `var recv: T = ...` — a binding whose left side is `recv`. The RHS is
# checked separately for a service-resolution call.
_GD_BIND = re.compile(r'^\s*(?:@onready\s+)?(?:var\s+)?(\w+)\s*(?::[^=]+?)?=\s*(.+)$')


def registered_service_names(mod: GDModule) -> set[str]:
    """The service name(s) this module registers ITSELF under — read off its own
    ``register("name", self)`` call (RayXI's ServiceRegistry convention: a service
    self-registers in ``_enter_tree``/``_ready`` under a string key). This is the
    module's authoritative identity for the locator, declared in code rather than
    guessed from the filename, so a same-named method on a DIFFERENT service can be
    told apart by which name a caller's receiver was resolved from."""
    out: set[str] = set()
    for _lineno, raw in _code_lines(mod):
        for m in _GD_SELF_REGISTER.finditer(_strip_inline_comment(raw)):
            out.add(m.group(1))
    return out


def _service_bindings(mod: GDModule) -> dict[str | None, dict[str, str]]:
    """Per-scope map of ``variable name -> service name`` for every local/member bound
    to a service-locator result (``var guild_svc := _service("guild_system")``). Keyed
    by enclosing function name (``None`` = module scope, e.g. an ``@onready`` member),
    so a receiver is resolved with function-local bindings first, module bindings as a
    fallback — the scope rule GDScript itself uses."""
    binds: dict[str | None, dict[str, str]] = {}
    for lineno, raw in _code_lines(mod):
        ln = _strip_inline_comment(raw)
        bm = _GD_BIND.match(ln)
        if not bm:
            continue
        rm = _GD_SERVICE_RESOLVE.search(bm.group(2))
        if not rm:
            continue
        svc = rm.group(1) or rm.group(2)
        binds.setdefault(mod.enclosing_func(lineno), {})[bm.group(1)] = svc
    return binds


def call_receiver_services(mod: GDModule, callee: str) -> dict[int, str]:
    """Map each line that calls ``recv.callee(...)`` to the service NAME ``recv`` was
    resolved from, when determinable — the receiver-side disambiguator the caller-scan
    uses when name AND arity collide (``chat_backend.kick`` and ``guild_system.kick``
    share both, but ``guild_panel_hud`` calls the one it bound from
    ``_service("guild_system")``). A receiver that is not a locator-bound variable
    (``self``, a raw node path, an unresolved member) yields no entry — unknown, never
    guessed, so a real caller is never dropped on a hunch. Both the method
    ``recv.callee(`` and dynamic ``recv.call("callee"`` forms are matched."""
    binds = _service_bindings(mod)
    module_binds = binds.get(None, {})
    c = re.escape(callee)
    method_recv = re.compile(r"(\w+)\.%s\s*\(" % c)
    dyn_recv = re.compile(r'(\w+)\.call\w*\(\s*["\']%s["\']' % c)
    out: dict[int, str] = {}
    for lineno, raw in _code_lines(mod):
        ln = _strip_inline_comment(raw)
        m = method_recv.search(ln) or dyn_recv.search(ln)
        if not m:
            continue
        recv = m.group(1)
        fn = mod.enclosing_func(lineno)
        svc = binds.get(fn, {}).get(recv) or module_binds.get(recv)
        if svc:
            out[lineno] = svc
    return out


def func_writes(mod: GDModule, func_name: str) -> list[Ref]:
    """Every STATE write inside the body of `func_name`, each Ref's ``name`` being
    the BASE identifier mutated (a member attr, or the container of a subscript, or
    a bare script-scope var). The per-function complement of :func:`field_writes`
    (which scopes to one field across the whole module): this scopes to one function
    across all fields, so a MUTATOR (writes state) is told apart from a pure getter
    without enumerating field names. A ``var``/``const`` line is a local binding, not
    a state write, and is skipped; a write whose base is a LOCAL of this function (a
    parameter or a ``var``/``const``/``for`` binding) is likewise a scratch-variable
    mutation, not state, and is excluded — only writes to names that outlive the call
    (members / script-scope vars) count, keeping the conservative read/write bias the
    rest of the adapter holds."""
    locals_ = _func_locals(mod, func_name)
    out: list[Ref] = []
    for lineno, ln in _func_body_lines(mod, func_name):
        if ln.lstrip().startswith("#"):
            continue
        code = _strip_inline_comment(ln).strip()
        if not code or code.startswith(("var ", "const ")):
            continue
        ms = _GD_SUB_W.match(code)
        if ms:
            base = ms.group(1) or ms.group(2)
            if base not in locals_:
                out.append(_mk(mod, lineno, "subscript_write", base))
            continue
        ma = _GD_ATTR_W.search(code)
        if ma:
            # `obj.field =` reports the ATTR (`field`), the member being mutated — not
            # the base object — so a local-name exclusion does not apply here (the
            # local-scratch case is the subscript/bare base above).
            out.append(_mk(mod, lineno, "attr_write", ma.group(1)))
            continue
        mb = _GD_BARE_W.match(code)
        if mb:
            if mb.group(1) not in locals_:
                out.append(_mk(mod, lineno, "bare_write", mb.group(1)))
            continue
        mm = _GD_METHOD_MUT.search(code)
        if mm and mm.group(1) not in locals_:
            out.append(_mk(mod, lineno, "method_mutate", mm.group(1)))
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
