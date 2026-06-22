"""Domain models for saddle's intake (Layer 1) + design (Layer 2) layers.

An :class:`Intake` is one decomposition of one user prompt into a list of
:class:`Item` s. Each Item is one discrete ask, classified by ``kind`` and
restated clearly in ``ask``. The persistent "todo list" is not a separate
type — it's the subset of items whose kind is in :data:`TODO_KINDS` that
are still open (see ``store.todos``).

Layer 2 adds two more persisted types: :class:`Knowledge` — one entry in the
Design Knowledge Base (DKB) of best practices, anti-patterns, and hard-won
lessons — and :class:`Design`, the best-practice design Layer 2 produces for a
goal. Both are scope-laddered (global / tenant / project) like policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Item kinds: the taxonomy Layer 1 classifies every ask into ----------
QUESTION = "question"      # wants an answer
TASK = "task"              # an action to perform -> goes on the todo list
DIRECTIVE = "directive"    # a standing rule / preference / constraint
CONTEXT = "context"        # background, no action needed
DECISION = "decision"      # a fork left to the assistant to choose

ITEM_KINDS: frozenset[str] = frozenset(
    {QUESTION, TASK, DIRECTIVE, CONTEXT, DECISION}
)
# Kinds that land on the persistent, actionable todo list.
TODO_KINDS: frozenset[str] = frozenset({TASK, DIRECTIVE})

# --- Item statuses -------------------------------------------------------
OPEN = "open"              # not yet handled
ANSWERED = "answered"      # a question that's been answered
DONE = "done"             # a task that's been completed
NOTED = "noted"            # context/directive acknowledged, no further action
ITEM_STATUSES: frozenset[str] = frozenset({OPEN, ANSWERED, DONE, NOTED})


@dataclass
class Item:
    """One discrete ask pulled out of a prompt.

    ``kind``/``ask``/``source_text``/``detail`` are produced by Layer 1;
    ``id``/``intake_id``/``seq``/``ts`` are stamped by the store on save.
    """

    kind: str
    ask: str
    source_text: str = ""
    detail: str = ""
    status: str = OPEN
    id: str = ""
    intake_id: str = ""
    seq: int = 0
    ts: float = 0.0

    def is_todo(self) -> bool:
        return self.kind in TODO_KINDS


@dataclass
class Intake:
    """One decomposition of one prompt, with its items + coverage meta."""

    raw_prompt: str
    summary: str = ""
    items: list[Item] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    tenant: str = ""
    project: str = ""
    ts: float = 0.0


# === Layer 2 — Design Knowledge Base (DKB) + design artifacts ============
#
# Layer 2 turns Layer 1's items into a best-practice design. It reasons over
# accumulated knowledge — the DKB — and records each design it produces. Both
# are scoped (like policy) to global / tenant / project, so universal practice
# is shared while a project's hard-won lessons stay private.

# --- Knowledge kinds: the DKB taxonomy -----------------------------------
BEST_PRACTICE = "best_practice"   # a way of doing things that works — to follow
ANTI_PATTERN = "anti_pattern"     # a tempting bad practice — to avoid
LESSON = "lesson"                 # learnt from a real bug / issue / design gap
PRINCIPLE = "principle"           # a standing design value (e.g. "no band-aids")
KNOWLEDGE_KINDS: frozenset[str] = frozenset(
    {BEST_PRACTICE, ANTI_PATTERN, LESSON, PRINCIPLE}
)

# Provenance — where an entry came from. Powers the auto-harvest loop: the
# harness files AUDIT/RUNTIME entries itself as it catches flaws and hits bugs.
SEED = "seed"          # authored seed corpus (pre-researched, offline)
AUDIT = "audit"        # harvested from a Layer 2 audit catching a flaw
RUNTIME = "runtime"    # filed from a bug / issue observed using the app
MANUAL = "manual"      # entered by hand (e.g. `saddle lesson ...`)
KNOWLEDGE_SOURCES: frozenset[str] = frozenset({SEED, AUDIT, RUNTIME, MANUAL})

# Lifecycle — entries are retired, never hard-deleted, so the DKB only ever
# grows its record (deleting could erase a real lesson; see the cron no-delete
# rule). Retired entries drop out of retrieval but stay auditable.
ACTIVE = "active"
RETIRED = "retired"
KNOWLEDGE_STATUSES: frozenset[str] = frozenset({ACTIVE, RETIRED})


@dataclass
class Knowledge:
    """One entry in the Design Knowledge Base.

    ``kind``/``title``/``body``/``tags`` are the content; ``scope_tenant`` and
    ``scope_project`` place it on the visibility ladder — both empty = global
    (everyone), tenant set + project empty = tenant-wide, both set = that one
    project. ``id``/``ts`` are stamped by the store on save.
    """

    kind: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    scope_tenant: str = ""
    scope_project: str = ""
    source: str = SEED
    status: str = ACTIVE
    id: str = ""
    ts: float = 0.0

    @property
    def scope(self) -> str:
        """Readable scope label: ``global`` / ``<tenant>`` / ``<tenant>/<project>``."""
        if not self.scope_tenant:
            return "global"
        if not self.scope_project:
            return self.scope_tenant
        return f"{self.scope_tenant}/{self.scope_project}"


# --- Design artifacts: Layer 2's output ----------------------------------
DESIGN_DRAFT = "draft"        # generated, not yet audit-clean
DESIGN_FINAL = "final"        # passed the directive / anti-pattern audit
DESIGN_FLAGGED = "flagged"    # audit did not converge — left for review
DESIGN_STATUSES: frozenset[str] = frozenset(
    {DESIGN_DRAFT, DESIGN_FINAL, DESIGN_FLAGGED}
)


@dataclass
class Design:
    """One best-practice design produced by Layer 2 for one (folded) goal.

    ``problem`` and ``approach`` carry the higher-level thinking stage — the
    root cause behind the symptom and the structural direction chosen (with any
    reframe / alternatives considered) — so a design records *why* this shape,
    not just the shape. ``satisfies``/``avoids``/``heeds`` trace it back to the
    directives, anti-patterns, and lessons that shaped it. ``id``/``tenant``/
    ``project``/``ts`` are stamped by the store on save.
    """

    ask: str
    summary: str = ""
    problem: str = ""          # root cause vs. symptom (diagnose stage)
    approach: str = ""         # structural direction + any reframe / alternatives
    body: str = ""             # the design itself
    satisfies: list[str] = field(default_factory=list)  # directives / practices honored
    avoids: list[str] = field(default_factory=list)     # anti-patterns avoided
    heeds: list[str] = field(default_factory=list)      # lessons applied
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = DESIGN_DRAFT
    id: str = ""
    tenant: str = ""
    project: str = ""
    intake_id: str = ""
    ts: float = 0.0
