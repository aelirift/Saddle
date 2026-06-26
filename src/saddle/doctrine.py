"""saddle doctrine — lessons compiled to a deterministic pre-action gate.

saddle exists because an LLM drifts, band-aids, and **does not learn**: a rule
you teach it in prose decays the moment it leaves the context window. saddle's
job is to be the part that DOES remember — deterministically.

Until now saddle remembered rules the same lossy way it distrusts. A standing
``directive`` was a free-text string pasted into the design prompt, and the
"was it honored?" check was *another LLM call* (design.py's audit stage). An
LLM grading an LLM drifts exactly where the first one did — which is how a
"never assume unused code is dead, classify it first" lesson failed live and
the agent proposed deleting load-bearing modules.

This module closes that gap. A lesson that is a *code-invariant* compiles to a
machine-checked rule that runs as a GATE *in front of* an action and BLOCKS it
— no LLM in the enforcement loop. saddle is the first passthrough: an action is
evaluated here before saddle ever prompts a provider, so a drifting model can't
route around the gate because the gate already ran. Two enforcement shapes (the
"hybrid"):

  * **data rules** — authorable as plain JSON in policy ``check_rules``:
      - ``require_evidence``: a verb on a target is forbidden unless named
        evidence is present (removing a symbol needs a *disposition*).
      - ``scope_fence``: a verb on a path outside the focus project is
        forbidden unless explicitly marked cross-project (the "don't wander
        off and 'fix' a sibling repo" guard).
  * **code predicates** — registered Python, for checks data can't express:
      - ``disposition_coherent``: a removal's disposition must NAME its
        replacement / wire-target / domain reason, not merely *exist*.

Judgment rules that can't be mechanized (taste: "architect not developer")
stay LLM-audited in design.py — this gate is only for invariants that can be
made red/green. :data:`SEED_RULES` is saddle's standing doctrine; :func:`evaluate`
is the gate; :func:`guard` is the passthrough entry point that converge calls
before it lets a coder mutation land (and that ``saddle guard`` exposes on the
CLI). The two seed rules are the exact lessons that drift kept defeating: stay
in the focus project, and never delete code without classifying it first.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:  # typing-only; avoids importing Context at module load
    from saddle.context import Context

_log = logging.getLogger("saddle.doctrine")

# Verb vocabulary is open at the edge (callers, shells, and LLMs all phrase the
# same act differently) and closed in the gate. Normalising here means a rule
# authored against "delete" also fires for "rm", "unlink", "purge", etc.
_VERB_SYNONYMS: dict[str, str] = {
    "rm": "delete", "del": "delete", "remove": "delete", "delete": "delete",
    "unlink": "delete", "drop": "delete", "purge": "delete",
    "edit": "edit", "modify": "edit", "change": "edit", "patch": "edit",
    "update": "edit",
    "write": "write", "overwrite": "write",
    "create": "create", "add": "create", "new": "create",
}

_CODE_SUFFIXES: tuple[str, ...] = (".py", ".pyi", ".gd")
# Extensions that name a *data/text* file, never a code symbol — so a bare
# "notes.txt" under --kind auto is judged a path, not code.
_DATA_SUFFIXES: tuple[str, ...] = (
    ".txt", ".md", ".rst", ".json", ".toml", ".yaml", ".yml", ".ini",
    ".cfg", ".lock", ".log", ".csv", ".tsv", ".xml", ".html", ".css",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip", ".tar",
    ".gz", ".db", ".sqlite", ".env",
)


def _norm_verb(verb: str) -> str:
    v = (verb or "").strip().lower()
    return _VERB_SYNONYMS.get(v, v)


def _truthy(v: Any) -> bool:
    """Evidence is submitted as strings from a CLI, so treat the usual
    string spellings of "no" as falsy rather than truthy-because-nonempty."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "0", "no", "none", "off")
    return bool(v)


def _looks_like_path(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    return (
        "/" in t or os.sep in t or t.startswith((".", "~"))
        or os.path.isabs(t) or t.endswith(_CODE_SUFFIXES)
    )


def _looks_like_code(t: str) -> bool:
    """A bare symbol/dotted-name (``foo.bar.baz``) reads as code; a slashed
    path or a phrase with spaces does not."""
    t = (t or "").strip()
    if not t:
        return False
    if t.endswith(_CODE_SUFFIXES):
        return True
    if "/" in t or os.sep in t:
        return False
    if t.lower().endswith(_DATA_SUFFIXES):
        return False
    return " " not in t


@dataclass(frozen=True)
class Action:
    """A single proposed mutation, submitted to the gate BEFORE it happens.

    ``target_kind`` is the caller's claim about what ``target`` is; "auto" lets
    the gate infer ``path`` / ``code`` facets from the string shape so a rule
    keyed on either still fires.
    """

    verb: str
    target: str
    target_kind: str = "auto"  # auto | path | code
    evidence: Mapping[str, Any] = field(default_factory=dict)
    project_root: str = ""  # explicit focus root; "" -> resolve via context

    @property
    def nverb(self) -> str:
        return _norm_verb(self.verb)

    def facets(self) -> frozenset[str]:
        """The set of target-kinds this action satisfies (``path``/``code``).

        A ``.py`` file is BOTH a path and code, so a code-scoped delete rule and
        a path-scoped scope-fence can both match the same target.
        """
        kind = self.target_kind
        out: set[str] = set()
        if kind == "path":
            out.add("path")
            if self.target.endswith(_CODE_SUFFIXES):
                out.add("code")
        elif kind == "code":
            out.add("code")
            if _looks_like_path(self.target):
                out.add("path")
        else:  # auto
            if _looks_like_path(self.target):
                out.add("path")
            if _looks_like_code(self.target):
                out.add("code")
        return frozenset(out)


@dataclass(frozen=True)
class CheckRule:
    """One machine-checked invariant. ``kind`` selects the evaluator in
    :func:`_apply`; the remaining fields parameterise that evaluator."""

    id: str
    kind: str  # require_evidence | scope_fence | predicate
    verbs: frozenset[str]
    target_kind: str = "any"  # any | path | code
    message: str = ""
    severity: str = "block"  # block | warn
    requires_evidence: tuple[str, ...] = ()
    override_evidence: str = ""
    predicate: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CheckRule":
        return cls(
            id=str(d["id"]),
            kind=str(d.get("kind", "require_evidence")),
            verbs=frozenset(_norm_verb(v) for v in d.get("verbs", [])),
            target_kind=str(d.get("target_kind", "any")),
            message=str(d.get("message", "")),
            severity=str(d.get("severity", "block")),
            requires_evidence=tuple(str(x) for x in d.get("requires_evidence", ())),
            override_evidence=str(d.get("override_evidence", "")),
            predicate=str(d.get("predicate", "")),
        )


@dataclass(frozen=True)
class Verdict:
    """The gate's answer. ``allowed`` is the bit a worker keys on; ``render``
    is the human/CLI line."""

    allowed: bool
    rule_id: str | None
    reason: str
    required_evidence: tuple[str, ...] = ()
    severity: str = "block"

    def render(self) -> str:
        if self.allowed and self.severity != "warn":
            return f"ALLOW: {self.reason}"
        tag = "WARN" if self.severity == "warn" else "BLOCK"
        head = f"{tag} [{self.rule_id}]: {self.reason}"
        if self.required_evidence:
            head += f"\n  needs evidence: {', '.join(self.required_evidence)}"
        return head


def _pred_disposition_coherent(action: Action, rule: CheckRule) -> tuple[bool, str]:
    """A removal's disposition must NAME its replacement, not merely exist.

    ``no-unwired-delete`` already forces *a* disposition to be present; this
    predicate enforces that the disposition is internally coherent — that the
    classification carries the companion fact that makes it falsifiable. This is
    exactly the lesson from the live drift: "unused" was asserted with no named
    replacement / wire-target / domain reason.
    """
    disp = str(action.evidence.get("disposition", "")).strip().lower()
    companion = {
        "scaffold": "wire_target",      # a stub to be wired -> name where
        "superseded": "replaced_by",    # replaced by newer code -> name it
        "domain_excluded": "reason",    # out of saddle's domain -> say why
    }
    if disp not in companion:
        return False, (
            f"disposition must be one of {sorted(companion)}; got {disp!r}"
        )
    need = companion[disp]
    if not _truthy(action.evidence.get(need)):
        return False, f"disposition '{disp}' must name its {need}"
    return True, ""


# Predicate registry — the "directive-as-code" half of the hybrid. Data rules
# can express "evidence X must be present"; a predicate expresses relationships
# between fields that JSON can't.
PREDICATES: dict[str, Callable[[Action, CheckRule], tuple[bool, str]]] = {
    "disposition_coherent": _pred_disposition_coherent,
}


# saddle's standing doctrine — the lessons this very conversation taught,
# compiled to gates so they survive the context window that taught them.
SEED_RULES: tuple[CheckRule, ...] = (
    CheckRule(
        id="stay-in-project-focus",
        kind="scope_fence",
        verbs=frozenset({"edit", "write", "create", "delete"}),
        target_kind="path",
        override_evidence="cross_project_task",
        message=(
            "this action targets a path OUTSIDE the focus project. Editing, "
            "creating, or deleting files that live outside the focus root shown "
            "below — in a sibling repo or any other project — is off-limits "
            "unless the task is explicitly cross-project (evidence "
            "cross_project_task=true)."
        ),
    ),
    CheckRule(
        id="no-unwired-delete",
        kind="require_evidence",
        verbs=frozenset({"delete"}),
        target_kind="code",
        requires_evidence=("disposition",),
        message=(
            "removing code requires a disposition. An unused or unreferenced "
            "symbol is NOT presumptively dead — classify it first: "
            "scaffold (+wire_target) | superseded (+replaced_by) | "
            "domain_excluded (+reason)."
        ),
    ),
    CheckRule(
        id="disposition-coherent",
        kind="predicate",
        verbs=frozenset({"delete"}),
        target_kind="code",
        predicate="disposition_coherent",
        message="the removal's disposition is incomplete.",
    ),
)


def _within(root: str, target: str) -> bool:
    """True if ``target`` resolves inside (or equals) ``root``. A relative
    target is taken relative to ``root`` — a worker naming a bare ``foo.py`` is
    talking about the focus project, not some arbitrary cwd."""
    try:
        r = Path(root).expanduser().resolve()
        t = Path(target).expanduser()
        if not t.is_absolute():
            t = r / t
        t = t.resolve()
        return t == r or r in t.parents
    except Exception:  # noqa: BLE001 — any resolution failure = "can't prove inside"
        return False


def _focus_root() -> str:
    """The focus project's root — the single boundary shared by the doctrine
    fence, Layer 3's code-map, and Layer 1 intake. Resolved via the focus
    authority (:func:`saddle.focus.focus_root`) so every layer agrees on what
    "the project" is."""
    from saddle.focus import focus_root
    return str(focus_root())


def _rule_triggers(rule: CheckRule, action: Action) -> bool:
    if action.nverb not in rule.verbs:
        return False
    if rule.target_kind == "any":
        return True
    return rule.target_kind in action.facets()


def _apply(
    rule: CheckRule,
    action: Action,
    predicates: Mapping[str, Callable[[Action, CheckRule], tuple[bool, str]]],
) -> Verdict | None:
    """Run one triggered rule. Returns a blocking/warning :class:`Verdict`, or
    ``None`` when the rule is satisfied (no objection)."""
    if rule.kind == "require_evidence":
        missing = tuple(
            k for k in rule.requires_evidence if not _truthy(action.evidence.get(k))
        )
        if missing:
            return Verdict(False, rule.id, rule.message, missing, rule.severity)
        return None

    if rule.kind == "scope_fence":
        if rule.override_evidence and _truthy(action.evidence.get(rule.override_evidence)):
            return None
        root = action.project_root or _focus_root()
        if not _within(root, action.target):
            needs = (rule.override_evidence,) if rule.override_evidence else ()
            return Verdict(
                False, rule.id,
                f"{rule.message}\n  target: {action.target}\n  focus:  {root}",
                needs, rule.severity,
            )
        return None

    if rule.kind == "predicate":
        fn = predicates.get(rule.predicate)
        if fn is None:
            _log.warning(
                "doctrine: rule %s names unknown predicate %r — skipping",
                rule.id, rule.predicate,
            )
            return None
        ok, reason = fn(action, rule)
        if not ok:
            return Verdict(False, rule.id, reason or rule.message, (), rule.severity)
        return None

    _log.warning("doctrine: rule %s has unknown kind %r — skipping", rule.id, rule.kind)
    return None


def evaluate(
    action: Action,
    rules: Sequence[CheckRule] | None = None,
    *,
    predicates: Mapping[str, Callable[[Action, CheckRule], tuple[bool, str]]] | None = None,
) -> Verdict:
    """Evaluate an action against the rule set. First BLOCK wins; warnings are
    collected and surfaced only if nothing blocks. An action no rule objects to
    is allowed."""
    rules = SEED_RULES if rules is None else rules
    predicates = PREDICATES if predicates is None else predicates
    warned: Verdict | None = None
    for rule in rules:
        if not _rule_triggers(rule, action):
            continue
        v = _apply(rule, action, predicates)
        if v is None or v.allowed:
            continue
        if v.severity == "block":
            return v
        warned = warned or v
    if warned is not None:
        return Verdict(
            True, warned.rule_id, f"allowed with warning: {warned.reason}",
            warned.required_evidence, "warn",
        )
    return Verdict(True, None, "no doctrine rule blocks this action")


def load_rules(ctx: "Context | None" = None) -> tuple[CheckRule, ...]:
    """SEED_RULES plus any tenant/project ``check_rules`` from policy. Policy
    rules are additive — a tenant can tighten doctrine, never loosen the seeds."""
    extra: list[CheckRule] = []
    try:
        from saddle.llm.policy import resolve_policy
        raw = resolve_policy(ctx).get("check_rules") or []
        for d in raw:
            try:
                extra.append(CheckRule.from_dict(d))
            except Exception as exc:  # noqa: BLE001 — one bad rule must not nuke doctrine
                _log.warning("doctrine: skipping malformed check_rule %r: %s", d, exc)
    except Exception:  # noqa: BLE001 — policy unavailable -> seeds still hold
        pass
    return SEED_RULES + tuple(extra)


def guard(action: Action, ctx: "Context | None" = None) -> Verdict:
    """The passthrough entry point: load context-aware rules and evaluate.

    converge calls this before it lets a coder mutation land, and ``saddle
    guard`` exposes it on the CLI — same gate, two front doors."""
    return evaluate(action, load_rules(ctx))


# --- tool-call adaptation: the passthrough's front edge ----------------------
# A coder (saddle's ChatSession) or any agent mutates the world through named
# tools (Edit / Write / Bash / ...). To gate those deterministically we
# translate a tool invocation into the doctrine Action(s) it represents, then
# run the same :func:`evaluate` the CLI uses. This is what makes doctrine a
# *passthrough* rather than advice: it sits in front of the tool, not inside the
# model — a PreToolUse hook or an SDK permission callback calls
# :func:`gate_tool_call` before the mutation is allowed to land.

_EDIT_TOOLS: frozenset[str] = frozenset({"Edit", "MultiEdit", "NotebookEdit", "Update"})
_WRITE_TOOLS: frozenset[str] = frozenset({"Write"})
_DELETE_CMDS: frozenset[str] = frozenset({"rm", "unlink", "shred"})
_SHELL_SEPS: frozenset[str] = frozenset({"&&", "||", ";", "|", "&"})


def _bash_actions(command: str, project_root: str) -> list[Action]:
    """Best-effort scan of a shell command for the obvious *mutating* operations
    the gate cares about — file deletes (``rm``/``unlink``/``shred``/``git rm``).

    The precise mutation channel is the Edit/Write tools, which map exactly; this
    closes the shell back-door so a delete can't dodge the gate merely by being a
    ``Bash`` call. A command we can't tokenise yields nothing — the gate does not
    assert on what it cannot read."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    segments: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok in _SHELL_SEPS:
            segments.append(cur)
            cur = []
        else:
            cur.append(tok)
    segments.append(cur)

    out: list[Action] = []
    for seg in segments:
        if not seg:
            continue
        head, rest = seg[0], seg[1:]
        if head == "git" and rest[:1] == ["rm"]:
            operands = [a for a in rest[1:] if not a.startswith("-")]
        elif head in _DELETE_CMDS:
            operands = [a for a in rest if not a.startswith("-")]
        else:
            continue
        out += [Action("delete", a, "path", project_root=project_root) for a in operands]
    return out


def actions_from_tool(
    tool_name: str, tool_input: Mapping[str, Any] | None, *, project_root: str = ""
) -> list[Action]:
    """Translate one tool invocation into the doctrine Action(s) it performs.

    Returns ``[]`` for tools that don't mutate code (Read / Grep / Glob / ...):
    the gate only speaks to mutations, so a read-only tool always passes."""
    name = (tool_name or "").strip()
    ti = tool_input or {}
    if name in _WRITE_TOOLS:
        fp = ti.get("file_path") or ti.get("path") or ""
        return [Action("write", str(fp), "path", project_root=project_root)] if fp else []
    if name in _EDIT_TOOLS:
        fp = ti.get("file_path") or ti.get("notebook_path") or ti.get("path") or ""
        return [Action("edit", str(fp), "path", project_root=project_root)] if fp else []
    if name == "Bash":
        return _bash_actions(str(ti.get("command") or ""), project_root)
    return []


def gate_tool_call(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    *,
    project_root: str = "",
    ctx: "Context | None" = None,
    rules: Sequence[CheckRule] | None = None,
) -> Verdict:
    """Evaluate a tool invocation against doctrine — the enforced passthrough's
    core. The first action that BLOCKS decides; otherwise the call is allowed
    (carrying a warning if any action warned). This is the single function a
    PreToolUse hook or an SDK ``can_use_tool`` callback calls."""
    actions = actions_from_tool(tool_name, tool_input, project_root=project_root)
    if not actions:
        return Verdict(True, None, "tool does not mutate code")
    rs = rules if rules is not None else load_rules(ctx)
    warned: Verdict | None = None
    for action in actions:
        v = evaluate(action, rs)
        if not v.allowed:  # evaluate never returns allowed=False except a block
            return v
        if v.severity == "warn":
            warned = warned or v
    return warned or Verdict(True, None, "no doctrine rule blocks this tool call")
