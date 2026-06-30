"""Claude Code transcript adapter — replay a live conversation into the tracker.

This is the PASSIVE sensor (the counterpart to the synchronous hook): saddle
reads the agent's own transcript JSONL — every user prompt, every assistant
message, and its thinking — and feeds the user<->agent loop into the
:class:`~saddle.dialog.IntentTracker`. Because saddle reads the log itself, a
thrashing or compacting agent cannot hide the dialog from it: the "you did a
different a) for 22 hours" moment is right there in the transcript, and replay
finds it as a :class:`~saddle.models.DriftVerdict`.

Parsed against the REAL on-disk shape (``~/.claude/projects/<slug>/<id>.jsonl``):

* one JSON object per line, top-level ``type`` in
  ``user|assistant|attachment|ai-title|last-prompt|queue-operation|system``;
* a genuine typed prompt is ``type=user`` whose ``message.content`` is a string
  (or a list of ``text`` blocks) — NOT one of the many ``tool_result`` blocks
  that Claude Code also injects under the user role;
* an assistant turn is ``type=assistant`` whose ``message.content`` is a list of
  ``thinking`` / ``text`` / ``tool_use`` blocks;
* ``sessionId`` is the conversation id (the tracker's ``session`` scope) and
  ``isSidechain`` marks sub-agent turns, which are NOT the user<->agent dialog
  and are skipped.

The deterministic drift check reads only the assistant's spoken ``text`` (where
commitments like "I'll go with option b" are stated cleanly); ``thinking`` is
carried on the event for the LLM brain but never deterministically parsed, so
messy reasoning ("b is wrong, do a") can't manufacture a false drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from saddle.context import Context
from saddle.dialog import (
    IntentTracker,
    get_tracker,
    parse_declared_choice,
    parse_declared_label,
)
from saddle.models import DriftVerdict

_DIALOG_TYPES = frozenset({"user", "assistant"})


@dataclass
class TranscriptEvent:
    """One dialog turn lifted out of the transcript. ``text`` is what was said
    (user prompt / assistant prose); ``thinking`` is the assistant's reasoning,
    carried for the brain but not parsed deterministically."""

    role: str            # "user" | "assistant"
    text: str = ""
    thinking: str = ""
    session: str = ""
    cwd: str = ""
    git_branch: str = ""
    timestamp: str = ""  # raw ISO8601 from the transcript (informational)
    uuid: str = ""


def _split_content(content: object) -> tuple[str, str, bool]:
    """Return ``(said_text, thinking_text, has_tool_result)`` for a message's
    ``content`` (a bare string, or a list of typed blocks)."""
    if isinstance(content, str):
        return content, "", False
    said: list[str] = []
    think: list[str] = []
    has_tool_result = False
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                said.append(str(b.get("text", "")))
            elif bt == "thinking":
                think.append(str(b.get("thinking", "")))
            elif bt == "tool_result":
                has_tool_result = True
    return "\n".join(said), "\n".join(think), has_tool_result


def parse_transcript_line(obj: object) -> TranscriptEvent | None:
    """Map one parsed JSONL object to a :class:`TranscriptEvent`, or ``None`` if
    it isn't a genuine user<->agent dialog turn (tool result, sidechain, system,
    title, attachment, empty)."""
    if not isinstance(obj, dict):
        return None
    t = obj.get("type")
    if t not in _DIALOG_TYPES:
        return None
    if obj.get("isSidechain"):
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    said, think, has_tool_result = _split_content(msg.get("content"))
    if t == "user":
        if has_tool_result:          # a tool result, not the user speaking
            return None
        if not said.strip():
            return None
    else:  # assistant
        if not said.strip() and not think.strip():
            return None
    return TranscriptEvent(
        role=t,
        text=said.strip(),
        thinking=think.strip(),
        session=str(obj.get("sessionId", "")),
        cwd=str(obj.get("cwd", "")),
        git_branch=str(obj.get("gitBranch", "")),
        timestamp=str(obj.get("timestamp", "")),
        uuid=str(obj.get("uuid", "")),
    )


def read_transcript(
    path: str | Path, *, after_uuid: str | None = None
) -> Iterator[TranscriptEvent]:
    """Yield dialog events from a transcript JSONL in order.

    Tolerant of a partially-written final line (a tail race): an unparseable
    line is skipped, never raised — this is log tailing, not enforcement.
    ``after_uuid`` is a resume cursor: skip everything up to and including that
    line's uuid, then yield only what follows (incremental tailing).
    """
    p = Path(path)
    if not p.exists():
        return
    skipping = after_uuid is not None
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if skipping:
                if isinstance(obj, dict) and obj.get("uuid") == after_uuid:
                    skipping = False
                continue
            ev = parse_transcript_line(obj)
            if ev is not None:
                yield ev


@dataclass
class TurnProposal:
    """The current turn's pre-edit state, lifted from the transcript for the
    pre-code design gate (Stage 3).

    ``goal`` is the user prompt that opened the latest turn; ``approach`` is the
    assistant prose spoken since (the design / plan the agent stated before the
    edit that triggered the read); ``anchor`` is the user prompt's uuid — the
    turn's identity, so the gate fires only on the FIRST code edit of a turn and
    stays silent on the rest. An empty ``anchor`` means no user turn was found
    (nothing to audit an approach against)."""

    goal: str = ""
    approach: str = ""
    anchor: str = ""


def latest_turn(path: str | Path) -> TurnProposal:
    """Lift the latest turn's ``(goal, approach, anchor)`` from a transcript.

    Walk the dialog in order; each user prompt opens a new turn (resetting the
    collected approach), so the assistant prose after the LAST user prompt is the
    approach the agent committed to before its edit. Only spoken ``text`` is read,
    never ``thinking`` — the *recorded design* is what the agent actually SAID it
    would do (the same discrimination the deterministic drift check makes), so
    private, messy reasoning ("b is wrong, do a") can't be mistaken for a stated
    approach. A transcript with no user turn yields an empty :class:`TurnProposal`.
    """
    goal = ""
    anchor = ""
    approach: list[str] = []
    for ev in read_transcript(path):
        if ev.role == "user":
            goal = ev.text
            anchor = ev.uuid
            approach = []  # a new turn starts — only THIS turn's prose is the approach
        elif ev.role == "assistant" and ev.text:
            approach.append(ev.text)
    return TurnProposal(goal=goal, approach="\n\n".join(approach).strip(), anchor=anchor)


@dataclass
class ReplayResult:
    """Outcome of feeding a transcript span into the tracker."""

    events: int = 0
    forks: int = 0          # forks the agent offered
    bindings: int = 0       # confidently-resolved user picks
    drifts: list[DriftVerdict] = field(default_factory=list)  # caught contradictions
    confirms: list[DriftVerdict] = field(
        default_factory=list
    )  # surfaced must-confirm UNKNOWNs (ambiguous / wrong-fork / uncited bare label)
    last_uuid: str = ""     # resume cursor for the next tail

    @property
    def surfaced(self) -> list[DriftVerdict]:
        """Everything saddle must speak up about — real drifts PLUS the
        must-confirm UNKNOWNs. The never-silent guarantee: an action that
        contradicts or can't be safely reconciled with the commitment is here,
        never swallowed."""
        return self.drifts + self.confirms


def replay(
    ctx: Context,
    path: str | Path,
    *,
    tracker: IntentTracker | None = None,
    after_uuid: str | None = None,
) -> ReplayResult:
    """Feed a transcript span into ``ctx``'s ledger and report what happened.

    For each assistant turn: record any fork offered, and — once a binding is
    active — check whether the turn declares an option that contradicts it. An
    action is read as a QUALIFIED fork-choice id first (``"p1.f6.a"`` — the form
    saddle re-injects), falling back to a bare label only when none is cited.
    A real drift lands in :attr:`ReplayResult.drifts`; a surfaced must-confirm
    UNKNOWN (ambiguous / wrong-fork / uncited bare label) lands in
    :attr:`ReplayResult.confirms` — both reachable via ``res.surfaced``, so a
    genuine contradiction is never silently dropped. For each user turn: open the
    exchange (numbering the dialog) and bind the pick. Returns counts plus a
    ``last_uuid`` cursor so a daemon can tail forward without reprocessing.
    """
    tracker = tracker or get_tracker()
    res = ReplayResult()
    for ev in read_transcript(path, after_uuid=after_uuid):
        res.events += 1
        if ev.uuid:
            res.last_uuid = ev.uuid
        if ev.role == "assistant":
            fork = tracker.observe_agent_message(ctx, ev.text, session=ev.session)
            if fork is not None:
                res.forks += 1
            choice = parse_declared_choice(ev.text)
            label = "" if choice else parse_declared_label(ev.text)
            if choice or label:
                verdict = tracker.check_action(
                    ctx, label, choice_id=choice, session=ev.session
                )
                if verdict.is_drift:
                    res.drifts.append(verdict)
                elif verdict.announce:
                    res.confirms.append(verdict)
        else:  # user
            binding = tracker.observe_user_message(ctx, ev.text, session=ev.session)
            if binding is not None and binding.resolved:
                res.bindings += 1
    return res
