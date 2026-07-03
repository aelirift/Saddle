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

from saddle import ids

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
    # The project this ask ROUTES to (mediator design §4). Empty = the ambient
    # project the intake ran under; a slug names a KNOWN sibling project the
    # task would act on, so every downstream stage (drift, lessons, designs)
    # reads and writes THAT project's ledger — never a blended one.
    project: str = ""

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
# Two families share one store (one schema, one hybrid index, one scope ladder):
#
#   * DESIGN WISDOM — curated, hard-won, irreproducible. The reason saddle
#     designs on everything learned so far instead of re-deriving cold.
#   * MEMORY — plain facts saddle would otherwise re-derive every session by
#     re-reading files or re-searching the web. A FACT is a durable, curated
#     truth (saddle's own identity, a project's stable invariant); a REFERENCE
#     is a *cache* entry — a materialized file/web lookup that is reconstructible
#     and therefore bounded + evictable (see ``durable`` and the DKB's cleanup).
BEST_PRACTICE = "best_practice"   # a way of doing things that works — to follow
ANTI_PATTERN = "anti_pattern"     # a tempting bad practice — to avoid
LESSON = "lesson"                 # learnt from a real bug / issue / design gap
PRINCIPLE = "principle"           # a standing design value (e.g. "no band-aids")
FACT = "fact"                     # a durable truth (identity, a stable invariant)
REFERENCE = "reference"           # a cached file/web lookup — bounded + evictable
KNOWLEDGE_KINDS: frozenset[str] = frozenset(
    {BEST_PRACTICE, ANTI_PATTERN, LESSON, PRINCIPLE, FACT, REFERENCE}
)
# The cache tier: kinds whose entries are reconstructible from their source, so
# eviction may DELETE them (the no-delete rule protects irreproducible knowledge;
# a cache you can rebuild by re-reading its source is exempt and must be bounded).
# A REFERENCE is always non-``durable``; the DKB enforces that invariant on write.
CACHE_KINDS: frozenset[str] = frozenset({REFERENCE})

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

    The cache dimension (only meaningful for the MEMORY family; see
    :data:`CACHE_KINDS`):

    * ``durable`` — pinned. A durable entry is hard-won/curated and is NEVER
      auto-evicted (retire-not-delete still applies). A non-durable entry is a
      reconstructible cache row the DKB's cleanup may delete to stay bounded. A
      :data:`REFERENCE` is forced non-durable on write; everything else defaults
      durable.
    * ``hits`` / ``last_used`` — access tracking the cleanup uses to evict the
      COLDEST cache rows first (LRU with an LFU tie-break). Bumped on retrieval.
    * ``expires_at`` — optional TTL as an epoch second (0 = never). A cache fact
      derived from something that goes stale carries when it should be dropped;
      an expired entry is excluded from retrieval and purged by cleanup.
    * ``provenance`` — for a cache entry, where it came from so it can be
      re-derived / invalidated: e.g. ``{"source": "<path|url>", "fingerprint":
      "<hash>", "fetched_at": <ts>}``. Empty for curated knowledge.
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
    durable: bool = True
    hits: int = 0
    last_used: float = 0.0
    expires_at: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        """Readable scope label: ``global`` / ``<tenant>`` / ``<tenant>/<project>``."""
        if not self.scope_tenant:
            return "global"
        if not self.scope_project:
            return self.scope_tenant
        return f"{self.scope_tenant}/{self.scope_project}"

    @property
    def is_cache(self) -> bool:
        """True for a reconstructible cache row (non-durable) — the tier the
        DKB's cleanup is allowed to delete to stay bounded."""
        return not self.durable


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


# === Conversational intent: forks the agent OFFERED + the user's BINDINGS ===
#
# A distinct drift axis from Layer 2/3 (which compare CODE against a DESIGN).
# This tracks the live DIALOG: when the AGENT offers the user a set of labeled
# options (a Fork), and when the USER picks one ("pick a)"), the Binding is
# recorded in a durable, (tenant, project)-scoped ledger that OUTLIVES the
# agent's context window. An agent action that then contradicts the bound option
# is conversational DRIFT — the "I asked for a) and you did a different a) for
# 22 hours" failure. Because the ledger is saddle's, not the agent's, it survives
# however hard the agent thrashes or compacts: saddle still knows a) == that
# option of that fork.

# --- Fork lifecycle ------------------------------------------------------
FORK_OPEN = "open"              # offered, awaiting the user's pick
FORK_RESOLVED = "resolved"      # the user bound one of its options
FORK_SUPERSEDED = "superseded"  # explicitly retired before a pick (never automatic)
FORK_STATUSES: frozenset[str] = frozenset(
    {FORK_OPEN, FORK_RESOLVED, FORK_SUPERSEDED}
)

# --- How a user reply was bound to an option -----------------------------
BIND_LABEL = "label"            # explicit "a)" / "option b" / "go with c"
BIND_POSITION = "position"      # "the first one" / "second" / "last"
BIND_RECOMMENDED = "recommended"  # "go" / "your call" -> the fork's recommended option
BIND_SEMANTIC = "semantic"      # matched by meaning (the LLM brain) — a later layer
BIND_AMBIGUOUS = "ambiguous"    # looked like a pick but could not bind — clarify
BIND_METHODS: frozenset[str] = frozenset(
    {BIND_LABEL, BIND_POSITION, BIND_RECOMMENDED, BIND_SEMANTIC, BIND_AMBIGUOUS}
)

# --- Drift verdict on an agent action vs the active binding --------------
DRIFT_ALIGNED = "aligned"   # the action is consistent with the bound option
DRIFT_DRIFT = "drift"       # the action contradicts the bound option
DRIFT_UNKNOWN = "unknown"   # cannot tell deterministically — defer to the brain
DRIFT_STATUSES: frozenset[str] = frozenset(
    {DRIFT_ALIGNED, DRIFT_DRIFT, DRIFT_UNKNOWN}
)


@dataclass
class ForkOption:
    """One labeled choice within a :class:`Fork`. ``label`` is normalized
    (``a``, ``b``, ``1`` …); ``text`` is the option as the agent phrased it."""

    label: str
    text: str = ""
    recommended: bool = False


@dataclass
class Fork:
    """A decision point the AGENT offered the user — a set of labeled options.

    ``(tenant, project)`` scope it to ONE project so a pick in project X can
    never resolve a fork offered in project Y. ``session`` records which agent
    conversation produced it (a finer, optional filter — resolution is
    project-scoped by default, matching "my pick replies to your chats, nothing
    else"). ``id``/``ts`` are stamped by the store on save.
    """

    options: list[ForkOption] = field(default_factory=list)
    prompt: str = ""          # the question / framing the agent posed
    source_text: str = ""     # raw agent message the fork was extracted from
    status: str = FORK_OPEN
    id: str = ""
    tenant: str = ""
    project: str = ""
    area: str = ""        # optional descriptive sub-label (does NOT change isolation)
    session: str = ""
    ts: float = 0.0
    pn: int = 0           # the exchange (user prompt #) this fork answers
    seq: int = 0          # its sequence within that exchange -> node_id "p<pn>.f<seq>"

    def labels(self) -> list[str]:
        return [o.label for o in self.options]

    def option(self, label: str) -> "ForkOption | None":
        lab = (label or "").strip().lower()
        return next((o for o in self.options if o.label == lab), None)

    def recommended_option(self) -> "ForkOption | None":
        return next((o for o in self.options if o.recommended), None)

    @property
    def node_id(self) -> str:
        """Readable, (tenant, project)-unique fork LOCATOR ``"p<pn>.f<seq>"`` (e.g.
        ``p1.f6``). Letter-tagged so it can't be mistaken for an IP / line range /
        version; saddle assigns it even when the agent printed no id of its own."""
        return ids.fork_node(self.pn, self.seq)

    def choice_id(self, label: str) -> str:
        """Readable fork-CHOICE locator for one option ``"p<pn>.f<seq>.<label>"``
        (e.g. ``p1.f6.a``) — the unit a binding is on. A bare label like ``a`` is
        meaningless without this qualification; that is the whole point."""
        return ids.fork_choice(self.pn, self.seq, label)

    @property
    def scope_prefix(self) -> str:
        """Scope segment for this fork's qualified ids: ``tenant_project`` or
        ``tenant_project_area`` when an area is set."""
        return ids.scope_prefix(self.tenant, self.project, self.area)

    @property
    def qualified_node_id(self) -> str:
        """Globally-unique, self-describing fork id for persistence / provenance:
        ``<scope>_fork_p<pn>.f<seq>`` (e.g. ``rayxi_saddle_layer1_fork_p1.f6``)."""
        return ids.qualify(self.scope_prefix, ids.KIND_FORK, self.node_id)

    def qualified_choice_id(self, label: str) -> str:
        """Globally-unique, self-describing fork-choice id:
        ``<scope>_choice_p<pn>.f<seq>.<label>`` (e.g.
        ``rayxi_saddle_layer1_choice_p1.f6.a``). Empty when ``label`` is blank."""
        loc = self.choice_id(label)
        return ids.qualify(self.scope_prefix, ids.KIND_CHOICE, loc) if loc else ""


@dataclass
class Binding:
    """The user's selection of one option of one :class:`Fork`.

    A confident bind has ``resolved=True`` and a non-empty ``label``. A reply
    that looked like a pick but could not be bound (no matching open fork, label
    not offered, "go" with no recommendation) records ``resolved=False`` with
    ``method=ambiguous`` so the caller asks rather than guesses — the opposite of
    silently picking the wrong ``a)``. Scoped + stamped like every persisted row.
    """

    fork_id: str
    label: str = ""
    choice_id: str = ""      # "p<pn>.f<seq>.<label>" — the qualified bound unit
    user_text: str = ""
    method: str = BIND_LABEL
    confidence: float = 1.0
    resolved: bool = True
    reason: str = ""
    id: str = ""
    tenant: str = ""
    project: str = ""
    area: str = ""           # optional sub-label, mirrors the fork's (isolation unchanged)
    session: str = ""
    ts: float = 0.0


@dataclass
class DriftVerdict:
    """Whether an agent action is consistent with the active binding.

    The comparison is on QUALIFIED fork-choice ids, never bare labels:
    ``bound_choice`` is the commitment the user picked (e.g. ``p1.f6.a``);
    ``action_choice`` is what the agent's action cited (``p2.f8.a`` or a bare
    ``a``). Same letter on a different fork (``p2.f8.a`` vs the bound ``p1.f6.a``)
    is DRIFT, not a match — that wrong-fork case is the real "I did a different a)
    for 22 hours" failure. ``bound_label``/``action_label`` carry the bare
    letters for display.

    ``surface`` is the never-go-silent flag: True for every drift AND every
    must-confirm UNKNOWN (a bare label that cannot be pinned to the committed
    fork). It is False only for the genuinely quiet verdicts — an aligned action,
    or no commitment / no declared option to compare. A real contradiction is
    never downgraded to a silent UNKNOWN.
    """

    status: str = DRIFT_UNKNOWN
    fork_id: str = ""
    bound_label: str = ""
    action_label: str = ""
    bound_choice: str = ""       # qualified commitment, e.g. "p1.f6.a"
    action_choice: str = ""      # qualified/bare option the action cited
    reason: str = ""
    confidence: float = 0.0
    surface: bool = False        # caller MUST announce this verdict

    @property
    def is_drift(self) -> bool:
        return self.status == DRIFT_DRIFT

    @property
    def announce(self) -> bool:
        """Whether the caller must surface this to the agent/user. True for every
        drift and every must-confirm UNKNOWN; False only for the quiet verdicts
        (aligned, or nothing to compare). The guard against a "harmless" fix
        silently swallowing a real contradiction."""
        return self.surface


# === Action provenance: number + record every agent action point =========
#
# saddle numbers what the agent DOES, not just the forks it offers — so months
# later "where's the map feature?" resolves to a concrete, auditable record: it
# was removed in action ``p12.act5`` during session X, because <reason>. If the
# user disputes the REASON ("it wasn't wired, so the fix was to hook it up, not
# delete it"), the action is marked CONTESTED with that counter-reason and can be
# reversed. The id is saddle's, assigned even when the agent displayed none.

# --- Action kinds: what the action did to the target ---------------------
ACT_CREATE = "create"
ACT_EDIT = "edit"
ACT_DELETE = "delete"
ACT_MOVE = "move"
ACT_OTHER = "other"
ACTION_KINDS: frozenset[str] = frozenset(
    {ACT_CREATE, ACT_EDIT, ACT_DELETE, ACT_MOVE, ACT_OTHER}
)

# --- Action lifecycle ----------------------------------------------------
ACT_RECORDED = "recorded"     # logged as it happened
ACT_CONTESTED = "contested"   # the user disputed the action's reason
ACT_REVERSED = "reversed"     # undone after a dispute (kept for the audit trail)
ACTION_STATUSES: frozenset[str] = frozenset(
    {ACT_RECORDED, ACT_CONTESTED, ACT_REVERSED}
)


@dataclass
class Action:
    """One agent action point, numbered + recorded for provenance.

    ``aid`` is the readable, (tenant, project)-unique action LOCATOR
    (``p12.act5`` — action 5 of exchange 12, so the id itself says which prompt
    the agent was serving); ``summary``/``kind``/``file``/``line_start``/
    ``line_end``/``symbol`` say what changed and where; ``reason`` is *why* (the
    disputable field); ``choice_id`` links the fork-choice this action was
    executing (if any), so an action can be checked against the live commitment.
    ``dispute`` holds the user's counter-reason when ``status`` is ``contested``.
    Scoped + stamped on save.
    """

    summary: str = ""
    kind: str = ACT_OTHER
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    symbol: str = ""           # e.g. "f_detail_map()"
    reason: str = ""           # why it was done — the field a user can dispute
    choice_id: str = ""        # fork-choice it executed, e.g. "p1.f6.a"
    fork_id: str = ""          # internal link to that fork, if any
    status: str = ACT_RECORDED
    dispute: str = ""          # the user's counter-reason when contested
    aid: str = ""              # readable action locator "p<pn>.act<seq>"
    id: str = ""               # internal uuid
    tenant: str = ""
    project: str = ""
    area: str = ""             # optional sub-label (isolation stays (tenant, project))
    session: str = ""
    pn: int = 0                # the exchange the action belongs to
    ts: float = 0.0

    @property
    def scope_prefix(self) -> str:
        """Scope segment for this action's qualified id: ``tenant_project`` or
        ``tenant_project_area`` when an area is set."""
        return ids.scope_prefix(self.tenant, self.project, self.area)

    @property
    def qualified_id(self) -> str:
        """Globally-unique, self-describing action id for provenance:
        ``<scope>_action_p<pn>.act<seq>`` (e.g.
        ``rayxi_saddle_layer1_action_p12.act5``). Empty until ``aid`` is stamped."""
        return ids.qualify(self.scope_prefix, ids.KIND_ACTION, self.aid) if self.aid else ""


# === Bubble: saddle's client-agnostic outbound voice =====================
#
# saddle's supervisory voice (the itemized ask, a drift caught, a design-review
# verdict) must reach a HUMAN even when no human is at a terminal. A hook's
# stderr is shown only in an interactive TTY; under an SDK / service host
# (``CLAUDE_CODE_ENTRYPOINT=sdk-py``) it is swallowed, so an AFK user never sees
# what saddle bubbled. A :class:`BubbleEvent` is that voice made DURABLE and
# CLIENT-AGNOSTIC: every message saddle wants seen is persisted to a canonical
# table any client can query AND mirrored to a per-session JSONL any client can
# ``tail`` — independent of whichever agent or host happens to be in front of it.
# It is presentation/transport, not a verdict: the drift engines (Layer 2/3,
# dialog) still own *what* is wrong; a bubble only carries it to a surface.

# --- Bubble levels: how prominently a client should render the message ----
BUBBLE_INFO = "info"       # background FYI — no action expected
BUBBLE_NOTICE = "notice"   # saddle wants this read (a stage result, a commitment)
BUBBLE_ALERT = "alert"     # saddle needs a correction / decision — surface loudly
BUBBLE_LEVELS: frozenset[str] = frozenset({BUBBLE_INFO, BUBBLE_NOTICE, BUBBLE_ALERT})

# --- Supervisory stages: the five live drift-entry checkpoints + the two
#     non-stage channels a bubble can also carry. Drift enters a turn at five
#     distinct points, so supervision is STAGED across the whole turn (one check
#     per entry point) rather than concentrated in a single pre-code gate; these
#     are the labels :attr:`BubbleEvent.stage` carries, named once here so the
#     runner, the hooks, and any client filter on the SAME set instead of
#     re-spelling string literals (an identity that fans out — see the DKB).
STAGE_INTAKE = "intake"    # 1 — the prompt is broken down properly (fail LOUD)
STAGE_INTENT = "intent"    # 2 — this prompt vs project / design / history
STAGE_DESIGN = "design"    # 3 — the anti-band-aid pre-code design review
STAGE_CODE = "code"        # 4 — the code conforms to the agreed design
STAGE_LESSON = "lesson"    # 5 — the lesson from this turn is captured
STAGE_GUARD = "guard"      # the deterministic doctrine pre-action gate (Layer 0)
STAGE_DIALOG = "dialog"    # the back-and-forth correction channel the stages use
STAGE_VOICE = "voice"      # the plain-language check on saddle's OWN messages
STAGE_COMPLETION = "completion"  # the did-the-ACTUAL-goal-get-done gate
# The five live supervisory stages, in turn order — what the staged runner owns.
SUPERVISION_STAGES: tuple[str, ...] = (
    STAGE_INTAKE, STAGE_INTENT, STAGE_DESIGN, STAGE_CODE, STAGE_LESSON,
)
# Every label a bubble's ``stage`` may carry: the five stages + guard/dialog/voice.
BUBBLE_STAGES: frozenset[str] = frozenset(SUPERVISION_STAGES) | {
    STAGE_GUARD, STAGE_DIALOG, STAGE_VOICE, STAGE_COMPLETION,
}


@dataclass
class BubbleEvent:
    """One message saddle wants a human (or any client) to see.

    ``text`` is the rendered, human-readable body; ``level`` drives render
    prominence (:data:`BUBBLE_LEVELS`); ``stage`` is a descriptive label for the
    supervisory stage that emitted it (``intake`` / ``intent`` / ``design`` /
    ``code`` / ``lesson`` / ``guard`` / ``dialog`` — does NOT change isolation,
    like an :class:`Action`'s ``area``); ``title`` is an optional short headline
    for a compact render; ``meta`` carries any structured payload a richer client
    wants. ``session`` is a finer, optional filter WITHIN a project (which agent
    conversation) — never a substitute for the ``(tenant, project)`` fence.
    ``id``/``tenant``/``project``/``ts`` are stamped by the store on save.
    """

    text: str
    level: str = BUBBLE_NOTICE
    stage: str = ""
    title: str = ""
    session: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    tenant: str = ""
    project: str = ""
    ts: float = 0.0
