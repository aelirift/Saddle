"""saddle MCP server — saddle's surfaces exposed over the Model Context Protocol.

This is the channel saddle never had. saddle's whole reason to exist is to be the
part of the loop that does NOT drift: a third-party observer that holds the
high-level intent and design of a project independent of any coding agent's
context window. But a CLI no one remembers to run can't observe anything — which
is exactly how the "F1 = official MCP SDK, F2 = wire to the project" commitment got
locked and then silently dropped, with no live saddle around to catch it. This
server closes that gap: a running agent (Claude Code, or any MCP client) connects
here and saddle becomes a live participant — it can be ASKED, and it can talk back.

Every tool maps to a saddle surface the CLI already exposes — the same code paths,
no parallel implementation:

  intake        Layer 1   decompose a prompt into an itemized list (know / do)
  todos                   the running open backlog for this (tenant, project)
  kb_search/list  DKB     the Design Knowledge Base (lessons, principles)
  lesson          DKB     record a durable lesson/principle by hand
  recall          DKB     recall known facts/lessons before reading files or web
  remember        DKB     cache a fact/lookup so it isn't re-derived next session
  kb_gc           DKB     bound the memory: evict expired/over-cap cache entries
  codemap_symbols Layer 3 the grounding symbol menu a design must bind to
  guard           Layer 0 the deterministic pre-action doctrine gate
  commitment      dialog  the live fork-choice the user last locked
  observe_agent/user      feed the agent<->user loop to the drift tracker
  check_action    dialog  verdict an action against the commitment (never silent)
  bubble_recent   outbox  saddle's durable outbound voice (what it bubbled to a human)
  discuss                 saddle's own LLM voice — ask saddle to weigh in / audit
  whoami                  saddle introduces itself: who it speaks for, what it holds

Transport is stdio (FastMCP), so the server drops into any MCP client's table as a
``command`` + ``args`` entry. One server process speaks for exactly one
(tenant, project) — fixed at launch via ``SADDLE_TENANT`` / ``SADDLE_PROJECT`` /
``SADDLE_CODE_ROOT`` (the same resolution the CLI uses), and that pair IS saddle's
hard isolation boundary: a tool call can never reach another tenant's rows.

Run::

    saddle-mcp                    # console script
    python -m saddle.mcp_server   # module
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from saddle.context import Context, code_root, resolve

mcp = FastMCP("saddle")


# -- context: one server, one (tenant, project) ------------------------------

def _ctx() -> Context:
    """Resolve the (tenant, project) this server speaks for — from
    ``SADDLE_TENANT`` / ``SADDLE_PROJECT`` and the cwd, exactly as the CLI does.
    Resolved per call so a re-pointed env is honored without a restart."""
    return resolve(os.environ.get("SADDLE_TENANT"), os.environ.get("SADDLE_PROJECT"))


def _voice_model() -> tuple[str, str]:
    """saddle's own LLM voice model + effort (Opus 4.8 / xhigh by default)."""
    model = (os.environ.get("SADDLE_AGENT_MODEL") or "claude-opus-4-8").strip()
    effort = (os.environ.get("SADDLE_AGENT_EFFORT") or "xhigh").strip()
    return model, effort


# -- whoami: saddle introduces itself ----------------------------------------

@mcp.tool(name="whoami")
def whoami() -> str:
    """Who saddle is speaking for and what it currently holds.

    Returns the (tenant, project) isolation key, the code root saddle gates
    against, its database location, the model its own voice runs on, the open
    todo count, and the live commitment (if any). Call this first — it is how an
    agent learns which saddle it is talking to and whether there is recorded
    intent to honor before it starts work.
    """
    from saddle.dialog import get_tracker
    from saddle.store import default_db_path, get_store

    ctx = _ctx()
    model, effort = _voice_model()
    todos = get_store().todos(ctx)
    binding = get_tracker().active_binding(ctx)
    lines = [
        "saddle — third-party intent-drift observer",
        f"  speaking for : {ctx.key}   (tenant={ctx.tenant!r} project={ctx.project!r})",
        f"  code root    : {code_root()}",
        f"  database     : {default_db_path()}",
        f"  saddle voice : {model} (effort {effort})",
        f"  open todos   : {len(todos)}",
    ]
    if binding is not None:
        lines.append(
            f"  commitment   : {binding.choice_id or binding.label} "
            f"— {binding.user_text!r}"
        )
    else:
        lines.append("  commitment   : (none recorded yet)")
    return "\n".join(lines)


# -- Layer 1: the per-prompt itemization (the headline surface) --------------

@mcp.tool(name="intake")
async def intake(prompt: str, max_audits: int = 2) -> str:
    """Decompose a prompt into the itemized list of what the agent must KNOW and
    DO, then record it.

    This is saddle's Layer 1: a two-pass itemize-then-coverage-audit under the
    contract that **no ask may be dropped**. Each item is typed
    (question / task / directive / context / decision) so the agent can see, in
    one place, the full surface of a request before touching code — and so the
    same backlog accrues across prompts. The intake is persisted to the
    (tenant, project) store; pass an empty ``prompt`` to do nothing.
    """
    from saddle.intake import decompose, format_intake

    text = (prompt or "").strip()
    if not text:
        return "intake: empty prompt — nothing to itemize."
    intake_rec = await decompose(text, _ctx(), max_audits=max_audits)
    return format_intake(intake_rec)


# -- the running backlog -----------------------------------------------------

@mcp.tool(name="todos")
def todos() -> str:
    """The open actionable backlog (tasks + directives) for this
    (tenant, project) — the cross-intake running todo list."""
    from saddle.store import get_store

    ctx = _ctx()
    items = get_store().todos(ctx)
    if not items:
        return f"no open todos for {ctx.key}"
    out = [f"{len(items)} open todo(s) for {ctx.key}:"]
    out += [f"  [{it.kind:<9}] {it.ask}  ({it.id})" for it in items]
    return "\n".join(out)


# -- DKB: the Design Knowledge Base ------------------------------------------

@mcp.tool(name="kb_search")
def kb_search(query: str, k: int = 8) -> str:
    """Hybrid-search the Design Knowledge Base for lessons / principles /
    best-practices / anti-patterns visible to this (tenant, project)."""
    from saddle.dkb import get_dkb

    ctx = _ctx()
    q = (query or "").strip()
    if not q:
        return "kb_search: empty query"
    hits = get_dkb().search_knowledge(ctx, q, k=k)
    if not hits:
        return f"no matches for {q!r} in {ctx.key}"
    out = [f"{len(hits)} hit(s) for {q!r} in {ctx.key}:"]
    out += [f"  [{kn.kind:<13}] {score:.4f}  {kn.title}  <{kn.scope}>" for kn, score in hits]
    return "\n".join(out)


@mcp.tool(name="kb_list")
def kb_list(kind: str | None = None, limit: int = 50) -> str:
    """List the Design Knowledge Base entries visible to this (tenant, project),
    optionally filtered to a single ``kind``
    (best_practice / anti_pattern / lesson / principle)."""
    from saddle.dkb import get_dkb

    ctx = _ctx()
    kinds = [kind] if kind else None
    items = get_dkb().list_knowledge(ctx, kinds=kinds, limit=limit)
    if not items:
        return f"DKB empty for {ctx.key}"
    out = [f"{len(items)} DKB entr{'y' if len(items) == 1 else 'ies'} for {ctx.key}:"]
    out += [f"  [{kn.kind:<13}] {kn.title}  <{kn.scope}> ({kn.source})" for kn in items]
    return "\n".join(out)


@mcp.tool(name="lesson")
def lesson(
    text: str,
    kind: str = "lesson",
    scope: str = "project",
    title: str | None = None,
    tags: str = "",
) -> str:
    """Record a durable DKB entry by hand — a lesson / principle / best_practice /
    anti_pattern that should outlive the agent's context window.

    ``scope`` is one of global / tenant / project (default project). This is how a
    decision gets remembered the way saddle remembers: written down, addressable,
    immune to compaction.
    """
    from saddle.dkb import get_dkb
    from saddle.models import MANUAL, Knowledge

    ctx = _ctx()
    body = (text or "").strip()
    if not body:
        return "lesson: empty text"
    ttl = (title or body).strip()
    if len(ttl) > 80:
        ttl = ttl[:79].rstrip() + "…"
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if scope == "global":
        scope_tenant, scope_project = "", ""
    elif scope == "tenant":
        scope_tenant, scope_project = ctx.tenant, ""
    else:
        scope_tenant, scope_project = ctx.tenant, ctx.project
    kn = Knowledge(
        kind=kind, title=ttl, body=body, tags=tag_list,
        scope_tenant=scope_tenant, scope_project=scope_project, source=MANUAL,
    )
    get_dkb().add_knowledge(kn)
    return f"recorded {kn.kind} [{kn.scope}] {kn.id}: {ttl}"


@mcp.tool(name="recall")
def recall(query: str, k: int = 5, kinds: str = "") -> str:
    """Recall what saddle already KNOWS about this project relevant to ``query``,
    before you go read files or search the web.

    This is saddle's long-term memory: durable facts (identity, stable
    invariants), hard-won lessons, and cached file/web lookups, retrieved by
    hybrid search (semantic + keyword). Bounded to the top ``k`` matches so it
    never floods your context. ``kinds`` optionally restricts the search to a
    comma-separated set (e.g. ``fact,reference`` for plain knowledge, excluding
    design wisdom). Returns the entries' bodies — treat them as established, and
    only fall back to reading/searching when recall comes up empty.
    """
    from saddle.recall import format_recall
    from saddle.recall import recall as _recall

    ctx = _ctx()
    q = (query or "").strip()
    if not q:
        return "recall: empty query"
    kind_list = [k2.strip() for k2 in (kinds or "").split(",") if k2.strip()] or None
    entries = _recall(ctx, q, k=k, kinds=kind_list)
    if not entries:
        return f"recall: nothing known for {q!r} in {ctx.key} (read/search, then remember it)"
    return format_recall(entries)


@mcp.tool(name="remember")
def remember(
    text: str,
    kind: str = "fact",
    scope: str = "project",
    title: str | None = None,
    tags: str = "",
    source: str = "",
    ttl: float = 0.0,
) -> str:
    """Remember something so saddle (and you, next session) need not re-derive it.

    Use this to cache what you just learned the expensive way — a fact you
    confirmed by reading code, a lookup you did on the web — so future turns
    recall it instead of repeating the work. ``kind``: ``fact`` (durable, the
    default — identity, a stable project invariant) or ``reference`` (a
    reconstructible cache of a file/web lookup; the store keeps the cache tier
    bounded and evictable). ``source`` records where a cached value came from (a
    path/URL) so it can be re-derived; ``ttl`` (seconds) expires a stale cache
    entry. ``scope`` is global / tenant / project (default project).
    """
    import time

    from saddle.dkb import get_dkb
    from saddle.models import MANUAL, Knowledge

    ctx = _ctx()
    body = (text or "").strip()
    if not body:
        return "remember: empty text"
    ttl_title = (title or body).strip()
    if len(ttl_title) > 80:
        ttl_title = ttl_title[:79].rstrip() + "…"
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if scope == "global":
        scope_tenant, scope_project = "", ""
    elif scope == "tenant":
        scope_tenant, scope_project = ctx.tenant, ""
    else:
        scope_tenant, scope_project = ctx.tenant, ctx.project
    now = time.time()
    expires_at = now + ttl if ttl and ttl > 0 else 0.0
    provenance = {"source": source, "fetched_at": now} if source else {}
    kn = Knowledge(
        kind=kind, title=ttl_title, body=body, tags=tag_list,
        scope_tenant=scope_tenant, scope_project=scope_project, source=MANUAL,
        expires_at=expires_at, provenance=provenance,
    )
    get_dkb().add_knowledge(kn)
    tier = "cache" if kn.is_cache else "durable"
    return f"remembered {kn.kind} [{kn.scope}] ({tier}) {kn.id}: {ttl_title}"


@mcp.tool(name="kb_gc")
def kb_gc(max_cache: int = 0) -> str:
    """Run the memory cleanup: purge expired cache entries and evict the coldest
    of any scope over its cache cap, so saddle's store stays bounded. The durable
    tier (facts, lessons, principles) is never touched. ``max_cache`` overrides
    the per-scope cap for this run (0 = use the configured default)."""
    from saddle.dkb import get_dkb

    cap = max_cache if max_cache and max_cache > 0 else None
    r = get_dkb().cleanup(max_cache_per_scope=cap)
    return (
        f"kb gc: {r['expired']} expired + {r['evicted']} over-cap evicted "
        f"across {r['scopes']} scope(s)"
    )


# -- Layer 3: the grounding symbol menu --------------------------------------

@mcp.tool(name="codemap_symbols")
def codemap_symbols(root: str | None = None) -> str:
    """Dump the project's grounding symbol menu — the fields / funcs / calls /
    collections a design's completeness surface must bind to.

    This is the vocabulary Layer 3 gates against: a design that names a symbol
    outside this menu is hallucinating structure that does not exist.
    """
    from saddle.codemap import refs

    r = root or str(code_root())
    mods = refs.parse_project(r)
    menu = refs.symbols(mods).top()
    out = [f"symbol menu for {r} ({len(mods)} module(s)):"]
    for bucket in ("fields", "funcs", "calls"):
        out.append(f"\n{bucket}:")
        out += [f"  {cnt:>5}  {name}" for name, cnt in menu[bucket].items()]
    if menu.get("collections"):
        out.append("\ncollections:")
        out += [f"  {name} = {members}" for name, members in menu["collections"].items()]
    return "\n".join(out)


# -- Layer 0: the deterministic pre-action doctrine gate ---------------------

@mcp.tool(name="guard")
def guard(
    verb: str,
    target: str,
    kind: str = "auto",
    evidence: dict[str, str] | None = None,
) -> str:
    """Evaluate a proposed action against saddle's doctrine BEFORE it happens.

    Returns ALLOW or BLOCK with the rule and any required evidence — no LLM in the
    enforcement loop, so a drifting model can't reason around it. The standing
    rules: stay inside the focus project, and never delete code without first
    classifying it (scaffold +wire_target / superseded +replaced_by /
    domain_excluded +reason). ``verb`` is edit/write/create/delete/...; ``target``
    is the path or code symbol; ``evidence`` carries facts like
    ``{"disposition": "superseded", "replaced_by": "..."}``.
    """
    from saddle.doctrine import Action, guard as _guard

    action = Action(
        verb=verb, target=target, target_kind=kind,
        evidence=evidence or {}, project_root=str(code_root()),
    )
    return _guard(action, _ctx()).render()


# -- dialog axis: the live commitment + drift verdict ------------------------

@mcp.tool(name="commitment")
def commitment() -> str:
    """The live commitment — the fork-choice the user most recently, confidently
    locked for this (tenant, project).

    This is the anchor a drift check runs against: "I asked for a) and you did b)"
    becomes a single equality test against a ledger that outlives the context
    window. Returns the bound choice + the user's exact words, or a note that
    nothing is on record yet.
    """
    from saddle.dialog import get_tracker

    ctx = _ctx()
    b = get_tracker().active_binding(ctx)
    if b is None:
        return f"no active commitment for {ctx.key}"
    return (
        f"commitment for {ctx.key}:\n"
        f"  choice : {b.choice_id or b.label}\n"
        f"  method : {b.method} (confidence {b.confidence:.2f})\n"
        f"  said   : {b.user_text!r}"
    )


@mcp.tool(name="observe_agent")
def observe_agent(text: str) -> str:
    """Feed an AGENT message to the drift tracker so it records any fork the agent
    just offered (a set of >=2 labeled options).

    Returns the stored fork's node id + options, or a note that the message
    offered no clear choice. This is half of saddle watching the live loop — the
    other half is ``observe_user``.
    """
    from saddle.dialog import get_tracker

    fork = get_tracker().observe_agent_message(_ctx(), text or "")
    if fork is None:
        return "no fork detected (message offered no >=2-option labeled choice)"
    opts = "; ".join(f"{o.label}) {o.text}" for o in fork.options)
    return f"recorded fork {fork.node_id}: {opts}"


@mcp.tool(name="observe_user")
def observe_user(text: str) -> str:
    """Feed a USER message to the drift tracker so it binds the reply to the newest
    open fork — locking the commitment future actions are checked against.

    Returns the binding (resolved choice or an honest 'ambiguous, ask') or a note
    that the reply wasn't a pick. saddle never guesses a meaning-match here; an
    ambiguous reply is recorded unresolved so the agent asks rather than drifts.
    """
    from saddle.dialog import get_tracker

    b = get_tracker().observe_user_message(_ctx(), text or "")
    if b is None:
        return "no binding (no open fork, or the reply wasn't a pick)"
    if b.resolved and b.label:
        return f"bound {b.choice_id or b.label} via {b.method} (confidence {b.confidence:.2f})"
    return f"AMBIGUOUS — {b.reason} (recorded unresolved; ask the user)"


@mcp.tool(name="check_action")
def check_action(action_label: str = "", choice_id: str = "") -> str:
    """Verdict an action against the live commitment — saddle's never-silent drift
    check.

    Pass the qualified fork-choice the action acts on (``choice_id`` like
    ``p1.f6.a``, preferred) or a bare ``action_label``. Returns ALIGNED / DRIFT /
    UNKNOWN with the reason; any verdict that contradicts or can't be safely
    reconciled with the commitment is flagged ``MUST SURFACE`` — saddle's guarantee
    that a real drift is never downgraded to a quiet pass.
    """
    from saddle.dialog import get_tracker

    v = get_tracker().check_action(_ctx(), action_label, choice_id=choice_id)
    head = f"[{v.status.upper()}] {v.reason}"
    if getattr(v, "announce", False):
        head = f"⚠ MUST SURFACE\n{head}"
    return head


# -- the bubble outbox: saddle's outbound voice (read) -----------------------

@mcp.tool(name="bubble_recent")
def bubble_recent(
    session: str = "",
    level: str = "",
    since_seconds: float = 0.0,
    limit: int = 50,
) -> str:
    """Read saddle's recent outbound voice for this (tenant, project) — the bubbles
    the hooks (and the staged supervisory runner) emit so a human SEES what saddle
    said even when a hook's stderr is swallowed under an SDK / service host.

    This is the durable, client-agnostic outbox: the SAME events the launcher panel
    and the ``saddle bubble`` CLI render, through one shared render contract, so no
    client drifts in how saddle's voice is shown. Filter to one ``session``, one
    ``level`` (info / notice / alert — ``alert`` means saddle needs a correction or
    a decision), and/or the last ``since_seconds``. Returns the rendered block
    oldest-first (latest at the bottom, panel-style).
    """
    import time

    from saddle.bubble import recent_bubbles, render_bubbles

    ctx = _ctx()
    since_ts = (time.time() - since_seconds) if since_seconds and since_seconds > 0 else None
    events = recent_bubbles(
        ctx,
        session=session or None,
        since_ts=since_ts,
        level=level or None,
        limit=limit,
    )
    if not events:
        where = f" for session {session}" if session else ""
        return f"no bubbles{where} in {ctx.key}"
    return render_bubbles(events)


# -- saddle's own LLM voice --------------------------------------------------

@mcp.tool(name="discuss")
async def discuss(topic: str) -> str:
    """Ask saddle to weigh in, in its own LLM voice (Opus 4.8 lead, with the
    provider fallback chain).

    saddle answers as a third-party observer of the project's intent: it can audit
    a plan against recorded intent, pressure-test a design, or say plainly where it
    sees drift. This is the "saddle talks back" half of the loop — distinct from
    the deterministic tools above, which are saddle's red/green verdicts.
    """
    from saddle.llm.callers import build_callers
    from saddle.voice import VOICE_CONTRACT

    msg = (topic or "").strip()
    if not msg:
        return "discuss: empty topic"
    ctx = _ctx()
    caller = build_callers(ctx)["default"]
    system = (
        "You are saddle: a third-party intent-drift observer that sits OUTSIDE the "
        "coding agent. You hold the high-level intent and design of a project, "
        "independent of the agent's context window. Answer concisely and "
        "concretely. If you see drift from the stated or recorded intent, say so "
        "plainly. Do not pad, do not hedge, do not estimate effort. When the "
        "topic is engineer-to-engineer (the agent asking about design), match "
        "its technical depth; when your words will reach the project's human "
        "owner, follow the style contract below." + VOICE_CONTRACT
    )
    return await caller(system, msg, label="mcp/discuss")


def main() -> int:
    """Entry point: serve saddle over MCP on stdio."""
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
