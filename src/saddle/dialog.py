"""Conversational-intent sensor — parse forks + picks, catch "pick a)" drift.

This is the live dialog axis of saddle, distinct from the code-vs-design drift
of Layer 2/3. It watches the running agent<->user loop for two events:

* the AGENT offering a set of labeled options (a :class:`~saddle.models.Fork`),
* the USER picking one (a :class:`~saddle.models.Binding`),

records both in the durable, (tenant, project)-scoped ledger
(:mod:`saddle.dialog_store`), and then — for any agent action that declares which
option it is acting on — returns a :class:`~saddle.models.DriftVerdict`. The
canonical catch is exact and deterministic: the user bound ``a``; an action that
declares ``b`` is DRIFT, no LLM in the loop. The hard "I asked for a) and you did
a different a) for 22 hours" failure becomes a single equality check against a
ledger that outlives the agent's context window.

Where meaning (not a label / position / "your call") is needed to bind a reply,
the deterministic matcher does NOT guess: it records an AMBIGUOUS, unresolved
binding so the caller asks, and leaves the real semantic judgment to the LLM
brain via the :class:`Matcher` seam (a later layer). Faking a verdict here would
be the band-aid; deferring honestly is the design.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from saddle import ids
from saddle.action_store import ActionStore, get_action_store
from saddle.context import Context
from saddle.dialog_store import ForkStore, get_fork_store
from saddle.models import (
    BIND_AMBIGUOUS,
    BIND_LABEL,
    BIND_POSITION,
    BIND_RECOMMENDED,
    DRIFT_ALIGNED,
    DRIFT_DRIFT,
    DRIFT_UNKNOWN,
    FORK_RESOLVED,
    Action,
    Binding,
    DriftVerdict,
    Fork,
    ForkOption,
)

# === Fork parsing: agent message -> labeled options ======================
#
# An option line is "a) ...", "a. ...", "a: ...", "(b) ...", "1) ...", or a
# markdown "- **Option C**: ...". The separator must be followed by whitespace —
# that single rule is what stops "e.g. foo" / "i.e. bar" / "3.14 pi" from being
# mistaken for options (no space after their inner dot).
_BOLD_RE = re.compile(
    r"^\s*[-*]?\s*\*\*(?:option\s+)?([A-Za-z]|\d{1,2})\*\*\s*[:.\)]?\s*(.+?)\s*$",
    re.I,
)
_BULLET_LABELED_RE = re.compile(
    r"^\s*[-*]\s+\(?([A-Za-z]|\d{1,2})[.\):]\s+(.+?)\s*$"
)
_LABELED_RE = re.compile(r"^\s*\(?([A-Za-z]|\d{1,2})[.\):]\s+(.+?)\s*$")
_OPT_PATTERNS = (_BOLD_RE, _BULLET_LABELED_RE, _LABELED_RE)

# Recommendation markers — on an option's own text, or a fork-level "I'd go with a".
_REC_OPT_RE = re.compile(r"\(recommended\)|\brecommended\b|\bbest choice\b", re.I)
_REC_FORK_RES = (
    re.compile(
        r"\b(?:i'?d\s+)?(?:recommend|suggest|go with|prefer|lean toward[s]?)\s+"
        r"(?:option\s+)?\(?([A-Za-z]|\d{1,2})\)?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:option\s+)?\(?([A-Za-z]|\d{1,2})\)?\s+is\s+(?:the\s+)?"
        r"(?:best|recommended|my recommendation)\b",
        re.I,
    ),
)


def _match_option_line(line: str) -> tuple[str, str] | None:
    for pat in _OPT_PATTERNS:
        m = pat.match(line)
        if m:
            lab = m.group(1).lower()
            txt = m.group(2).strip()
            if txt:
                return lab, txt
    return None


def parse_options(text: str) -> list[ForkOption]:
    """Pull the labeled options out of an agent message.

    Dedupes by label and keeps only one *consistent* group — all-letter or
    all-digit, whichever is larger — so a stray "1." in an a/b/c list can't
    masquerade as a fourth option. Returns ``[]`` unless >=2 options survive.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        hit = _match_option_line(line)
        if hit is None:
            continue
        lab, otext = hit
        if lab in seen:
            continue
        seen.add(lab)
        found.append((lab, otext))
    if len(found) < 2:
        return []
    alpha = [(l, t) for (l, t) in found if l.isalpha()]
    digit = [(l, t) for (l, t) in found if l.isdigit()]
    group = alpha if len(alpha) >= len(digit) else digit
    if len(group) < 2:
        return []
    return [ForkOption(label=l, text=t) for (l, t) in group]


def _mark_recommended(options: list[ForkOption], text: str) -> list[ForkOption]:
    labels = {o.label for o in options}
    for o in options:
        if _REC_OPT_RE.search(o.text):
            o.recommended = True
    if not any(o.recommended for o in options):
        for rx in _REC_FORK_RES:
            m = rx.search(text)
            if m and m.group(1).lower() in labels:
                lab = m.group(1).lower()
                for o in options:
                    if o.label == lab:
                        o.recommended = True
                break
    return options


def _extract_prompt(text: str) -> str:
    """The framing question: last non-empty line before the first option."""
    lines = (text or "").splitlines()
    first_opt = next(
        (i for i, ln in enumerate(lines) if _match_option_line(ln)), None
    )
    if first_opt is None:
        return ""
    for j in range(first_opt - 1, -1, -1):
        s = lines[j].strip()
        if s:
            return s
    return ""


def parse_fork(text: str) -> Fork | None:
    """Extract a :class:`Fork` from an agent message, or ``None`` if it offered
    no clear >=2-option choice. Unstamped — the store stamps id/ts/scope."""
    options = parse_options(text)
    if len(options) < 2:
        return None
    options = _mark_recommended(options, text or "")
    return Fork(
        options=options,
        prompt=_extract_prompt(text or ""),
        source_text=(text or "").strip(),
    )


# === Selection parsing: user message -> which option =====================
_SOLO_LABEL_RE = re.compile(r"^\(?([A-Za-z]|\d{1,2})\)?\s*[.!]?$")
_VERB_LABEL_RE = re.compile(
    r"\b(?:option|opt|choice|go with|go for|let'?s (?:do|go with|pick)|"
    r"i'?ll (?:take|go with|do|pick)|i(?:'d| would)? (?:like|prefer|pick|choose)|"
    r"going with|pick|choose|select)\s+\(?([A-Za-z]|\d{1,2})\)?(?=[\s.,;:!?\)]|$)",
    re.I,
)
_SOLO_POSITION_RE = re.compile(
    r"^(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|final)"
    r"(?:\s+(?:one|option|choice))?\s*[.!]?$",
    re.I,
)
_THE_X_POSITION_RE = re.compile(
    r"\bthe\s+(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|final)\s+"
    r"(?:one|option|choice)\b",
    re.I,
)
_RECOMMEND_RE = re.compile(
    r"^(?:go|go for it|go ahead|do it|your call|you (?:decide|choose|pick)|"
    r"up to you|whatever you (?:recommend|think|suggest|prefer)|i'?ll trust you|"
    r"trust your call|sounds good)\s*[.!]?$",
    re.I,
)
_POSITION_INDEX = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4,
}


def _bind(fork: Fork, label: str, user_text: str, method: str, conf: float) -> Binding:
    return Binding(
        fork_id=fork.id, label=label, user_text=user_text,
        method=method, confidence=conf, resolved=True,
    )


def _ambiguous(fork: Fork, user_text: str, reason: str) -> Binding:
    return Binding(
        fork_id=fork.id, label="", user_text=user_text,
        method=BIND_AMBIGUOUS, confidence=0.3, resolved=False, reason=reason,
    )


def _by_position(fork: Fork, word: str, user_text: str) -> Binding:
    if word in ("last", "final"):
        idx = len(fork.options) - 1
    else:
        idx = _POSITION_INDEX.get(word, -999)
    if 0 <= idx < len(fork.options):
        return _bind(fork, fork.options[idx].label, user_text, BIND_POSITION, 0.9)
    return _ambiguous(fork, user_text, f"position {word!r} is out of range for this fork")


def resolve_selection(fork: Fork, user_text: str) -> Binding | None:
    """Deterministically bind a user reply to one option of ``fork``.

    Returns a resolved :class:`Binding` on a confident label / position /
    "your call" match; an AMBIGUOUS unresolved binding when the reply clearly
    *looks* like a pick but can't be bound (label not offered, "go" with no
    recommendation) so the caller asks rather than guesses; ``None`` when the
    reply isn't a pick at all (leave the fork open).
    """
    t = (user_text or "").strip()
    if not t:
        return None
    labels = fork.labels()

    m = _SOLO_LABEL_RE.match(t)
    if m:
        lab = m.group(1).lower()
        if lab in labels:
            return _bind(fork, lab, t, BIND_LABEL, 1.0)
        return _ambiguous(fork, t, f"{lab!r} is not an offered option")

    m = _SOLO_POSITION_RE.match(t)
    if m:
        return _by_position(fork, m.group(1).lower(), t)

    if _RECOMMEND_RE.match(t):
        rec = fork.recommended_option()
        if rec is not None:
            return _bind(fork, rec.label, t, BIND_RECOMMENDED, 0.85)
        return _ambiguous(fork, t, "deferred to recommendation but no option was recommended")

    m = _VERB_LABEL_RE.search(t)
    if m:
        lab = m.group(1).lower()
        if lab in labels:
            return _bind(fork, lab, t, BIND_LABEL, 0.97)
        return _ambiguous(fork, t, f"{lab!r} is not an offered option")

    m = _THE_X_POSITION_RE.search(t)
    if m:
        return _by_position(fork, m.group(1).lower(), t)

    return None


# Agent action -> the option label it declares it is acting on. Kept to explicit
# forms ("option a", "approach b", "going with c", "chose a") so a sentence like
# "doing a refactor" never produces a phantom label that would fake a DRIFT.
_DECLARE_RE = re.compile(
    r"\b(?:option|approach|choice|alternative|going with|go with|chose|chosen|"
    r"selected|sticking with)\s+\(?([A-Za-z]|\d{1,2})\)?(?=[\s.,;:!?\)]|$)",
    re.I,
)


def parse_declared_label(text: str) -> str:
    """The bare option label an agent action declares (or ``""`` if none). A bare
    label is NOT a verdict on its own — it must be resolved against the committed
    fork (see :meth:`IntentTracker.check_action`), because ``a`` is meaningless
    without a fork."""
    if not text:
        return ""
    m = _DECLARE_RE.search(text)
    return m.group(1).lower() if m else ""


# A QUALIFIED fork-choice locator ("p1.f6.a") the agent cites — the enforced,
# letter-tagged form (see :mod:`saddle.ids`) that makes a drift check
# deterministic AND IP-proof: a dotted prose number like "127.0.0" has no p/f
# tags, so it can't even parse as a citation. saddle re-injects these locators so
# the agent cites the exact commitment rather than a fork-less label.


def parse_declared_choice(text: str) -> str:
    """The qualified fork-choice locator an agent action cites (e.g. ``p1.f6.a``),
    or ``""`` if it cites none. Preferred over :func:`parse_declared_label` — it
    names WHICH fork, so a same-letter / wrong-fork action is catchable."""
    if not text:
        return ""
    m = ids.DECLARE_CHOICE_RE.search(text)
    return ids.normalize_choice(m.group(1)) if m else ""


# === Matcher seam: deterministic now, semantic (LLM) later ===============
@runtime_checkable
class Matcher(Protocol):
    """Binds a user reply to one option of a fork. The deterministic matcher
    handles label / position / recommended; a future semantic matcher (the LLM
    brain) plugs in here for meaning-matched replies."""

    def match(self, fork: Fork, user_text: str) -> Binding | None: ...


class DeterministicMatcher:
    """Exact, no-LLM matcher — the canonical "pick a)" path."""

    def match(self, fork: Fork, user_text: str) -> Binding | None:
        return resolve_selection(fork, user_text)


# === The tracker: ties parser + matcher + durable ledger together ========
class IntentTracker:
    """Observes the live agent<->user dialog and verdicts agent actions.

    All three methods are (tenant, project)-scoped through the ledger, so a pick
    in one project can never resolve or drift-check a fork in another.
    """

    def __init__(
        self,
        store: ForkStore | None = None,
        matcher: Matcher | None = None,
        action_store: ActionStore | None = None,
    ) -> None:
        self._store = store or get_fork_store()
        self._matcher = matcher or DeterministicMatcher()
        # Lazy: only opened when the action API is actually used, so drift-only
        # callers (and tests) never touch the action store / real DB.
        self._action_store = action_store

    def _actions(self) -> ActionStore:
        if self._action_store is None:
            self._action_store = get_action_store()
        return self._action_store

    def observe_agent_message(
        self, ctx: Context, text: str, *, session: str = ""
    ) -> Fork | None:
        """Record any fork the agent just offered (numbered within the current
        exchange -> ``node_id`` ``"p<pn>.f<seq>"``). Returns the stored fork."""
        fork = parse_fork(text)
        if fork is None:
            return None
        fork.session = session
        return self._store.add_fork(ctx, fork)

    def observe_user_message(
        self, ctx: Context, text: str, *, session: str = ""
    ) -> Binding | None:
        """Open a new exchange for this user turn, then bind the reply to the
        newest open fork in this project.

        Every user turn advances the ``prompt`` counter (the exchange #), so
        saddle numbers the dialog whether or not the agent printed any id. Returns
        the stored :class:`Binding` (resolved or ambiguous), or ``None`` when there
        is no open fork or the reply isn't a pick. A resolved bind records the
        QUALIFIED ``choice_id`` (``"p<pn>.f<seq>.<label>"``) — the unit drift is
        checked against — and flips the fork to ``resolved``.
        """
        self._store.next_counter(ctx, "prompt")   # this turn is a new exchange
        forks = self._store.open_forks(ctx, session=session or None, limit=1)
        if not forks:
            return None
        fork = forks[0]
        binding = self._matcher.match(fork, text)
        if binding is None:
            return None
        binding.fork_id = fork.id
        binding.session = session
        if binding.resolved and binding.label:
            binding.choice_id = fork.choice_id(binding.label)
        self._store.add_binding(ctx, binding)
        if binding.resolved and binding.label:
            self._store.set_fork_status(ctx, fork.id, FORK_RESOLVED)
        return binding

    def active_binding(self, ctx: Context, *, session: str = "") -> Binding | None:
        """The latest confidently-resolved binding (the live commitment) for this
        project/session."""
        return self._store.latest_binding(
            ctx, session=session or None, resolved_only=True
        )

    # -- the drift verdict (fork-identity, never silent) ------------------

    def check_action(
        self,
        ctx: Context,
        action_label: str = "",
        *,
        choice_id: str = "",
        session: str = "",
    ) -> DriftVerdict:
        """Verdict an agent action against the live commitment, comparing
        QUALIFIED fork-choices — never bare labels.

        The unit is ``(fork, option)``: the user's commitment is e.g. ``p1.f6.a``.
        An action that cites a fork-choice id (``choice_id``) is judged exactly,
        but only when the id names a fork saddle actually minted: ``p1.f6.a``
        ALIGNED, ``p1.f6.b`` DRIFT (same fork, other option), and a real other
        fork's ``p2.f8.a`` DRIFT (the "same letter, wrong fork" failure a
        label-only check waved through). An untagged dotted number (``127.0.0``, a
        line range) can't parse as a locator, so it's prose, not a citation.

        An action that only states a bare ``action_label`` is resolved against the
        forks in scope. Because the committed fork is always a carrier candidate, a
        label IT offers can never be mistaken for "unrelated": same option ALIGNED,
        a different option of that fork DRIFT. A label that resolves to a single
        OTHER fork, or is ambiguous across several, is UNKNOWN **and surfaced**
        (``surface=True``) so saddle forces a fork-choice citation. A label that is
        an option of no tracked fork is a different decision space that cannot
        contradict the commitment — a quiet UNKNOWN.

        The never-silent guarantee: every action that contradicts or can't be
        safely reconciled with the commitment is surfaced. The only quiet verdicts
        are an aligned action and a genuine "nothing to compare" (no commitment, no
        declared option, or a label unrelated to every tracked fork). A real
        contradiction is never downgraded to a silent UNKNOWN.
        """
        b = self.active_binding(ctx, session=session)
        if b is None:
            return DriftVerdict(
                status=DRIFT_UNKNOWN,
                reason="no active commitment for this (tenant, project)",
            )
        bound_fork = self._store.get_fork(ctx, b.fork_id)
        bound_choice = b.choice_id or (
            bound_fork.choice_id(b.label) if bound_fork is not None else ""
        )
        bf_node = (
            bound_fork.node_id if bound_fork is not None
            else (bound_choice.rsplit(".", 1)[0] if bound_choice else "")
        )
        common = dict(fork_id=b.fork_id, bound_label=b.label, bound_choice=bound_choice)

        # --- Path 1: the action cites a qualified fork-choice locator --------
        # The locator grammar is letter-tagged (``p1.f6.a``), so an untagged prose
        # number like "127.0.0" (a localhost IP) or "125.203.41" (a line range)
        # can't even parse as a citation — the IP-collision class is gone at the
        # FORMAT level. A citation is then honored ONLY if it names a fork saddle
        # actually minted (saddle is the numbering authority); the node-existence
        # check below is the second net behind the grammar.
        cid = ids.normalize_choice(choice_id)
        if cid:
            c_node, c_label = cid.rsplit(".", 1)
            if c_node == bf_node:
                if c_label == b.label:
                    return DriftVerdict(
                        status=DRIFT_ALIGNED, action_label=c_label, action_choice=cid,
                        confidence=1.0, reason="action cites the committed fork-choice",
                        **common,
                    )
                return DriftVerdict(
                    status=DRIFT_DRIFT, action_label=c_label, action_choice=cid,
                    confidence=1.0, surface=True,
                    reason=(
                        f"action cites {cid} but the commitment is {bound_choice} "
                        "— same fork, different option"
                    ),
                    **common,
                )
            # A different node — a real drift only if it names a fork saddle minted.
            if self._store.get_fork_by_node(ctx, c_node) is not None:
                return DriftVerdict(
                    status=DRIFT_DRIFT, action_label=c_label, action_choice=cid,
                    confidence=1.0, surface=True,
                    reason=(
                        f"action cites {cid} but the commitment is {bound_choice} — a "
                        "DIFFERENT fork; record a new pick if you've switched"
                    ),
                    **common,
                )
            # The id names no tracked fork: it's prose, not a citation. Fall through
            # to the bare-label path rather than fake a drift from a stray number.
            cid = ""

        # --- Path 2: only a bare label --------------------------------------
        al = (action_label or "").strip().lower()
        if not al:
            return DriftVerdict(
                status=DRIFT_UNKNOWN, reason="action declares no option to compare",
                **common,
            )
        carriers = self._forks_with_label(ctx, al, session, bound_fork)
        if not carriers:
            # The committed fork is always among the carrier candidates, so a label
            # it offers can never land here. "No carriers" therefore means the
            # label is an option of NO tracked fork — a different decision space
            # entirely, which cannot contradict the commitment. Quiet, not a
            # swallowed drift: surfacing every unrelated label would be the noise.
            return DriftVerdict(
                status=DRIFT_UNKNOWN, action_label=al,
                reason=(
                    f"action declares {al!r}, an option of no tracked fork (the "
                    f"commitment {bound_choice} is unaffected); nothing to compare"
                ),
                **common,
            )
        if len(carriers) == 1:
            only = carriers[0]
            if only.id == b.fork_id:
                if al == b.label:
                    return DriftVerdict(
                        status=DRIFT_ALIGNED, action_label=al,
                        action_choice=only.choice_id(al), confidence=0.9,
                        reason="bare label resolves to the committed fork and matches",
                        **common,
                    )
                return DriftVerdict(
                    status=DRIFT_DRIFT, action_label=al,
                    action_choice=only.choice_id(al), confidence=1.0, surface=True,
                    reason=(
                        f"action does {only.choice_id(al)} but the commitment is "
                        f"{bound_choice} — same fork, different option"
                    ),
                    **common,
                )
            # a single, DIFFERENT fork carries this label: maybe legitimate new
            # work, maybe the wrong-fork drift — saddle can't tell deterministically,
            # so it surfaces (forces citation) rather than bless or fake it.
            return DriftVerdict(
                status=DRIFT_UNKNOWN, action_label=al,
                action_choice=only.choice_id(al), surface=True,
                reason=(
                    f"bare {al!r} resolves to fork {only.node_id} "
                    f"({only.choice_id(al)}), but the commitment is {bound_choice} "
                    "— confirm you've moved on, or cite the fork-choice id"
                ),
                **common,
            )
        nodes = [c.node_id for c in carriers]
        return DriftVerdict(
            status=DRIFT_UNKNOWN, action_label=al, surface=True,
            reason=(
                f"bare {al!r} is ambiguous across forks {nodes}; cite the "
                f"fork-choice id (commitment is {bound_choice})"
            ),
            **common,
        )

    def _forks_with_label(
        self, ctx: Context, label: str, session: str, bound_fork: Fork | None
    ) -> list[Fork]:
        """Forks in scope (still-open + the committed, now-resolved fork) that
        offer ``label`` — the candidates a bare label could refer to."""
        forks = list(self._store.open_forks(ctx, session=session or None, limit=50))
        if bound_fork is not None and all(f.id != bound_fork.id for f in forks):
            forks.append(bound_fork)
        return [f for f in forks if label in f.labels()]

    # -- action provenance: number + record what the agent DID ------------

    def record_action(
        self, ctx: Context, action: Action, *, session: str = ""
    ) -> Action:
        """Number + persist one agent action point (assigns ``p<pn>.act<seq>``),
        stamping the current exchange so it is traceable to the prompt it
        answered."""
        action.session = action.session or session
        if not action.pn:
            action.pn = self._store.current_counter(ctx, "prompt")
        return self._actions().add_action(ctx, action)

    def find_actions(
        self,
        ctx: Context,
        *,
        symbol: str | None = None,
        file: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Action]:
        """Provenance lookup — e.g. "where's the map feature?" by symbol/text."""
        return self._actions().find_actions(
            ctx, symbol=symbol, file=file, status=status, query=query, limit=limit
        )

    def dispute_action(self, ctx: Context, aid: str, reason: str) -> bool:
        """Mark an action CONTESTED with the user's counter-reason (the "that's the
        wrong reason — hook it up, don't delete it" case)."""
        return self._actions().dispute_action(ctx, aid, reason)


# --- process-wide singleton ---------------------------------------------
_tracker: IntentTracker | None = None


def get_tracker() -> IntentTracker:
    """Return the process-global tracker (lazily built over the default store)."""
    global _tracker
    if _tracker is None:
        _tracker = IntentTracker()
    return _tracker


def set_tracker(tracker: IntentTracker) -> None:
    """Inject a tracker (tests, alternate matchers/stores)."""
    global _tracker
    _tracker = tracker


def reset_tracker() -> None:
    """Drop the singleton — for tests."""
    global _tracker
    _tracker = None
