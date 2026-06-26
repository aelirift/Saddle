"""saddle's id naming convention — the single source of truth for the grammar.

Every id saddle assigns is SELF-DESCRIBING and IP-PROOF, two properties learned
the hard way:

* **self-describing** — an id carries its TYPE (and, qualified, its scope) so a
  human reading ``rayxi_saddle_layer1_choice_p12.f1.a`` months later can decode it
  without asking: rayxi's saddle project, layer-1 area, a fork-CHOICE, prompt 12
  -> fork 1 -> option a. No bare ``12.1.5`` that means nothing on its own.
* **IP-proof** — every segment of a locator is LETTER-TAGGED (``p``, ``f``, ``q``,
  ``act``), so a locator can never be shaped like an IP address (``127.0.0.1``), a
  line range (``125.203``), or a version string (``1.6.2``). Those dotted numbers
  in ordinary agent prose no longer collide with the fork-choice grammar — the
  ambiguity that produced false drifts is removed at the FORMAT level, not merely
  caught after the fact by the node-existence check (which stays as a second net).

Two tiers:

* **LOCATOR** — the short, in-(tenant, project)-unique id the agent cites inline
  and saddle re-injects: ``p12`` (exchange), ``p12.f1`` (fork), ``p12.f1.a``
  (fork-choice), ``p12.q3`` (open question), ``p12.act5`` (action point).
* **QUALIFIED** — the globally-unique persistence / provenance form that prefixes
  the scope and type: ``<tenant>_<project>[_<area>]_<kind>_<locator>`` (e.g.
  ``rayxi_saddle_layer1_choice_p12.f1.a``).

This module is a leaf — stdlib only, nothing here imports saddle — so every layer
can build and parse ids through the one grammar without an import cycle.
"""

from __future__ import annotations

import re
import uuid

# --- id kinds: the self-describing TYPE word carried by every saddle id ---
# Full words, never abbreviations: an id prefix that needs decoding ("what is
# itm/dsg/frk?") defeats the whole point of a self-describing convention.
KIND_PROMPT = "prompt"        # one user exchange
KIND_ASK = "ask"              # an open question pulled from a prompt
KIND_FORK = "fork"            # a decision point the agent offered (a/b/c)
KIND_CHOICE = "choice"        # one option of a fork — the unit a binding is on
KIND_ACTION = "action"        # one agent action point
KIND_BINDING = "binding"      # the user's pick of one fork option
KIND_INTAKE = "intake"        # one decomposition of one user prompt
KIND_ITEM = "item"            # one discrete ask pulled out of a prompt
KIND_DESIGN = "design"        # one best-practice design (Layer 2)
KIND_KNOWLEDGE = "knowledge"  # one Design Knowledge Base entry
ID_KINDS = frozenset({
    KIND_PROMPT, KIND_ASK, KIND_FORK, KIND_CHOICE, KIND_ACTION,
    KIND_BINDING, KIND_INTAKE, KIND_ITEM, KIND_DESIGN, KIND_KNOWLEDGE,
})

# A fork-option label is a single letter or a 1-2 digit number (matching the
# labels the fork parser accepts), always lowercased in an id.
_LABEL = r"(?:[A-Za-z]|\d{1,2})"


# === locator builders — short, letter-tagged, IP-proof ===================

def exchange(pn: int) -> str:
    """Locator for one user exchange (prompt): ``p<pn>`` (e.g. ``p12``)."""
    return f"p{int(pn)}"


def fork_node(pn: int, seq: int) -> str:
    """Locator for a fork node: ``p<pn>.f<seq>`` (e.g. ``p12.f1``)."""
    return f"p{int(pn)}.f{int(seq)}"


def fork_choice(pn: int, seq: int, label: str) -> str:
    """Locator for one option of a fork: ``p<pn>.f<seq>.<label>`` (``p12.f1.a``).
    Empty when ``label`` is blank — a fork-choice id is meaningless without it,
    which is the whole point of qualifying a bare ``a``."""
    lab = (label or "").strip().lower()
    return f"{fork_node(pn, seq)}.{lab}" if lab else ""


def ask(pn: int, seq: int) -> str:
    """Locator for an open question: ``p<pn>.q<seq>`` (e.g. ``p12.q3``)."""
    return f"p{int(pn)}.q{int(seq)}"


def action(pn: int, seq: int) -> str:
    """Locator for an action point: ``p<pn>.act<seq>`` (e.g. ``p12.act5``) — the
    action's number within the exchange it was taken in, so the id itself says
    which prompt the agent was serving when it acted."""
    return f"p{int(pn)}.act{int(seq)}"


# === qualified ids — scope + type prefix, for persistence / provenance ====

def scope_prefix(tenant: str, project: str, area: str = "") -> str:
    """Underscore-joined scope: ``tenant_project`` or ``tenant_project_area``.
    Isolation stays on (tenant, project); ``area`` is a descriptive sub-label."""
    return "_".join(p for p in (tenant, project, area) if p)


def qualify(prefix: str, kind: str, locator: str) -> str:
    """Render a fully-qualified, self-describing id
    ``<scope-prefix>_<kind>_<locator>`` — e.g.
    ``rayxi_saddle_layer1_choice_p12.f1.a``."""
    return f"{prefix}_{kind}_{locator}"


# === internal record ids — opaque uuid primary keys, self-describing prefix =

def record_id(kind: str) -> str:
    """Mint a fresh internal record id ``<kind>_<uuid12>`` (e.g.
    ``item_a3f9c2b1e004``) — the DB primary key for a persisted row. The prefix
    is a full :data:`ID_KINDS` word so the key is legible on sight; the uuid keeps
    it globally unique without a counter round-trip. Isolation still rides on the
    row's ``(tenant, project)`` columns, never on this opaque key."""
    return f"{kind}_{uuid.uuid4().hex[:12]}"


# === parsing — a citation is only honored when it parses to this grammar ==
_FORK_NODE_RE = re.compile(r"^p(\d+)\.f(\d+)$")
_CHOICE_RE = re.compile(rf"^p(\d+)\.f(\d+)\.({_LABEL})$")
# Embedded-in-prose form, for spotting a citation inside an agent message. The
# word boundaries keep an IP / line range / version (no ``p``/``f`` tags) from
# ever matching, and tolerate trailing punctuation ("...go with p12.f1.a.").
DECLARE_CHOICE_RE = re.compile(rf"\b(p\d+\.f\d+\.{_LABEL})\b")


def parse_fork_node(s: str) -> tuple[int, int] | None:
    """``"p12.f1"`` -> ``(12, 1)``; anything else (a bare number, an IP, a line
    range) -> ``None``."""
    m = _FORK_NODE_RE.match((s or "").strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_choice(s: str) -> tuple[int, int, str] | None:
    """``"p12.f1.a"`` -> ``(12, 1, "a")`` with the label lowercased; else ``None``."""
    m = _CHOICE_RE.match((s or "").strip())
    return (int(m.group(1)), int(m.group(2)), m.group(3).lower()) if m else None


def normalize_choice(s: str) -> str:
    """Canonical fork-choice locator (lowercased label), or ``""`` if ``s`` is not
    one. Tolerant by design: a non-matching string yields ``""``, which resolves
    to no tracked fork and surfaces — it never fakes a drift."""
    parsed = parse_choice(s)
    return fork_choice(parsed[0], parsed[1], parsed[2]) if parsed else ""
