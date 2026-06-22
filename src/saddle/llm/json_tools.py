"""Helpers for extracting JSON payloads from LLM responses."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from saddle.llm.protocol import LLMCaller


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks; keep everything else verbatim.

    Safe for free-form prose. Unlike :func:`strip_llm_wrappers` it does NOT pull
    the contents out of a ``` code fence, so a Markdown design body whose own
    text contains a fenced code block survives intact rather than being
    truncated to the first fenced span.
    """
    return _THINK_RE.sub("", text or "").strip()


def strip_llm_wrappers(text: str) -> str:
    """Think-strip, then unwrap a single enclosing code fence (a JSON helper)."""
    clean = strip_think(text)
    if not clean:
        return ""
    fence_match = _CODE_FENCE_RE.search(clean)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner:
            return inner
    return clean


def extract_json_text(text: str) -> str:
    clean = strip_llm_wrappers(text)
    if not clean:
        return ""
    try:
        json.loads(clean)
        return clean
    except Exception:
        pass

    # Scan every `{` / `[` position for a balanced span that parses.
    # When the SSE stream concatenates a draft and a final answer in one
    # buffer (Claude self-correcting mid-output: an incomplete first
    # emission followed by a complete second one), the first object is
    # malformed and the second is the model's committed final answer.
    # Walk all candidate spans and keep the LAST that parses cleanly —
    # that span is what the model committed to last.  Single-emission
    # responses still resolve on the very first balanced span, just via
    # a longer scan.
    candidates: list[str] = []
    n = len(clean)
    i = 0
    while i < n:
        ch_i = clean[i]
        if ch_i not in "{[":
            i += 1
            continue
        opener = ch_i
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(i, n):
            ch = clean[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end >= 0:
            candidate = clean[i : end + 1].strip()
            try:
                json.loads(candidate)
                candidates.append(candidate)
            except Exception:
                pass
            i = end + 1
        else:
            i += 1
    if candidates:
        return candidates[-1]
    return clean


def parse_json_response(text: str) -> Any:
    candidate = extract_json_text(text)
    if not candidate:
        raise json.JSONDecodeError("Empty JSON response", text or "", 0)
    return json.loads(candidate)


async def call_json(
    caller: "LLMCaller", system: str, prompt: str, *, label: str = ""
) -> dict:
    """Call an LLM in JSON mode and parse the reply to a dict.

    The shared primitive for every staged LLM step (intake, design): a
    non-dict or unparseable reply is surfaced loudly (the parse raises) rather
    than silently swallowed — a malformed structured reply is a contract bug to
    fix, not to paper over.
    """
    text = await caller(system, prompt, json_mode=True, label=label)
    payload = parse_json_response(text)
    return payload if isinstance(payload, dict) else {}
