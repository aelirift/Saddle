"""Categorize an LLM-call retry by root cause.

Per project memory `feedback_every_llm_retry_is_a_bug`: the only valid
retry is external rate-limiting (429). Every other category indicates
a CONTRACT GAP we own — input too large, output too large, schema
descriptor missing, field-name disagreement across stages, format
mismatch — and we want to surface that root cause loudly at observation
time, not bury it as an opaque exception string in `events.jsonl`.

Pre-fix the trace layer emitted generic `error="<str(exc)>"` strings.
Post-fix every retry carries a structured `retry_category` so the
creator panel + build-oracle + a future "no-non-rate-limit-retries"
contract test can see the categorized reason directly.

Categories (ordered by what's our responsibility to fix):

  external_rate_limit  → 429 / provider quota / explicit "rate" text
                         The only LEGIT retry reason — outside our
                         control, retry is correct.

  provider_outage      → 500/502/503/504 / "service unavailable" text
                         Outside our control but worth tracking so a
                         flaky provider gets noticed (move them down
                         the priority chain).

  timeout              → stream hung past the wall-clock cap.
                         OUR FAULT if recurring: likely input/output
                         size pushed past what the provider can stream
                         within the cap. Investigate call size.

  empty_response       → provider returned 0 bytes / whitespace only.
                         OUR FAULT (almost always): system prompt /
                         user prompt produced something the model
                         couldn't answer. Schema descriptor missing?

  parse_error          → response wasn't valid JSON.
                         OUR FAULT: output schema doesn't fit the
                         format the model defaults to. Tighten the
                         system prompt with an explicit "respond with
                         JSON, no prose" preamble + schema sample.

  validation_failed    → JSON parsed but schema validator rejected.
                         OUR FAULT: schema descriptor + system prompt
                         drift — model thinks the spec means X, we
                         enforce Y. Tighten the schema_for_prompt(...)
                         output OR loosen the validator.

  output_too_large     → response truncated mid-stream / hit max_tokens.
                         OUR FAULT: violates the "one artifact per LLM
                         call" contract (`feedback_llm_call_size_contract`).
                         Split the call.

  input_too_large      → provider rejected context-window size /
                         "context length exceeded" text.
                         OUR FAULT: violates same contract on the
                         input side. Trim brief / shed payload.

  provider_blocked     → "you've hit your limit" / account-level text.
                         Not a retry — surfaces here so the
                         categorization is exhaustive.

  other                → didn't match any of the above.
                         Surfaces as a known unknown — add a category
                         when this shows up repeatedly.

Generic across genres + providers. The categorizer reads the SHAPE of
the failure (status code, exception type, raw payload prefix), not
provider-specific error message strings — so Kimi, MiniMax, DeepSeek,
GLM all flow through the same categorizer.
"""

from __future__ import annotations

import json
import re
from typing import Any


# Category enum as a frozen string-set. New categories must be added
# here so the noqa-free narrow contract test ('all categories listed')
# stays green.
RETRY_CATEGORIES: frozenset[str] = frozenset({
    "external_rate_limit",
    "provider_outage",
    "timeout",
    "empty_response",
    "parse_error",
    "validation_failed",
    "output_too_large",
    "input_too_large",
    "provider_blocked",
    "other",
})

# ── Retry disposition: the ENFORCEMENT a category drives ──────────────
#
# The categorizer answers "what went wrong"; the disposition answers
# "what do we DO about it". This is the half of
# feedback_every_llm_retry_is_a_bug that was missing: pre-fix the
# category was computed and logged but never CONSUMED, so a
# validation_failed retry that eventually passed looked as clean as a
# first-try success. Three tiers:
#
#   legit_retry → outside our control AND re-issuing the same call can
#                 succeed. Retry silently; run health stays clean.
#                 external_rate_limit ONLY.
#
#   soft_retry  → re-issuing the SAME call may succeed (provider hiccup,
#                 or a model formatting / schema wobble the
#                 self-correction feedback loop can repair), but the
#                 retry is a smell worth surfacing: the run is marked
#                 DEGRADED and a categorized llm_retry trace event is
#                 emitted. Retry is still permitted up to max_retries.
#                 provider_outage lives here (was legit pre-2026-05-30):
#                 outside our control, but a flaky provider should show
#                 up as a degraded run, not a clean one.
#
#   hard_fail   → re-issuing the same call CANNOT fix it (input/output
#                 too large for the one-artifact contract; account
#                 block). Enforced at their detection points —
#                 load_schema() raises on a missing schema,
#                 ProviderBlockedError raises on account UI, the
#                 one-artifact size contract + oversize warning catch
#                 output bloat — so they never reach a silent retry.
#                 The map records the intent so any retry loop that
#                 ever sees one raises instead of looping.
DISPOSITION_LEGIT = "legit_retry"
DISPOSITION_SOFT = "soft_retry"
DISPOSITION_HARD = "hard_fail"

RETRY_DISPOSITIONS: dict[str, str] = {
    "external_rate_limit": DISPOSITION_LEGIT,
    "provider_outage": DISPOSITION_SOFT,
    "timeout": DISPOSITION_SOFT,
    "empty_response": DISPOSITION_SOFT,
    "parse_error": DISPOSITION_SOFT,
    "validation_failed": DISPOSITION_SOFT,
    "output_too_large": DISPOSITION_HARD,
    "input_too_large": DISPOSITION_HARD,
    "provider_blocked": DISPOSITION_HARD,
    "other": DISPOSITION_SOFT,
}

# Every category MUST have a disposition — keeps the tier map and the
# category enum from drifting apart. A new category with no disposition
# would silently default to soft and slip past the tiering.
assert set(RETRY_DISPOSITIONS) == set(RETRY_CATEGORIES), (
    "RETRY_DISPOSITIONS out of sync with RETRY_CATEGORIES: "
    f"{set(RETRY_CATEGORIES) ^ set(RETRY_DISPOSITIONS)}"
)

# Categories that retry SILENTLY with no run-health penalty. Derived
# from the disposition map so there's a single source of truth — every
# OTHER category either degrades the run (soft) or raises (hard).
LEGIT_RETRY_CATEGORIES: frozenset[str] = frozenset(
    cat for cat, disp in RETRY_DISPOSITIONS.items()
    if disp == DISPOSITION_LEGIT
)


# Provider-side text markers that imply each category. The categorizer
# checks the lowercased status string + raw response when the
# numeric status isn't enough (e.g. some providers wrap 200 around
# error JSON).
_RATE_LIMIT_TEXT_HINTS = (
    "rate limit",
    "too many requests",
    "rate-limited",
    "throttled",
    "ratelimit",
    "quota exceeded",
)

_OUTAGE_TEXT_HINTS = (
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "upstream connect error",
)

_INPUT_TOO_LARGE_HINTS = (
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "input is too long",
    "prompt is too long",
    "token limit",
)

_OUTPUT_TOO_LARGE_HINTS = (
    "max_tokens",
    "max tokens",
    "output truncated",
    "response truncated",
    "completion exceeded",
)

_BLOCKED_HINTS = (
    "you've hit your limit",
    "you have hit your limit",
    "hit your limit",
    "account is blocked",
    "subscription required",
)


def is_rate_limit_text(text: str) -> bool:
    """True when an error string looks like a provider rate-limit (429).

    Single source for the rate-limit vocabulary so the router's circuit
    breaker (``IntentRouter.record_failure``) and the structured-retry
    classifier (``categorize_retry``) agree on what 'throttled' means — a
    divergence would let the breaker miss a 429 the retry loop treats as the
    one legit retry, leaving the throttled provider in rotation to burn a
    pool slot per call.
    """
    if not text:
        return False
    t = text.lower()
    if "429" in t:
        return True
    return any(marker in t for marker in _RATE_LIMIT_TEXT_HINTS)


def categorize_retry(
    *,
    exception: BaseException | None = None,
    status_code: int | None = None,
    raw_response: str | None = None,
    parse_error: str | None = None,
    validation_errors: list[str] | None = None,
) -> str:
    """Classify one retry attempt's failure into a category.

    Callers pass whatever signals they have; the categorizer picks
    the MOST SPECIFIC category that matches. Ordering matters: a
    response that's both empty and 429 should categorize as
    external_rate_limit (rate-limit is the actionable signal).

    Returns one of `RETRY_CATEGORIES`. Never raises — unknown shapes
    fall through to "other" so the rest of the trace pipeline keeps
    flowing.
    """
    # 1. Provider-block — matches first because it's irrelevant to retry
    # but the categorizer should still return a known value so trace
    # events stay typed.
    response_text = (raw_response or "").lower()
    if response_text:
        for marker in _BLOCKED_HINTS:
            if marker in response_text:
                return "provider_blocked"

    # 2. Numeric status code. 429 is the canonical rate-limit signal;
    # 5xx is provider outage; 413/414 / context-window 400s are input
    # size; specific provider 4xx for output truncation is rare but
    # surfaces here when present.
    if status_code is not None:
        if status_code == 429:
            return "external_rate_limit"
        if status_code in (408,):  # request timeout
            return "timeout"
        if status_code in (413, 414):  # payload too large / URI too long
            return "input_too_large"
        if 500 <= status_code < 600:
            return "provider_outage"

    # 3. Exception type. Order matters here too — TimeoutError /
    # asyncio.TimeoutError beat generic Exception.
    if exception is not None:
        exc_name = type(exception).__name__
        exc_str = str(exception).lower()
        # isinstance, not an exact-name match: every TimeoutError SUBCLASS is the
        # wall-clock class and must classify as timeout. saddle's own typed
        # liveness errors (supervise.DeadlineExceeded / Stalled) and
        # asyncio.TimeoutError all subclass TimeoutError but carry their own
        # __name__, so an exact-name check let them fall through to "other" —
        # exactly the supervisory-stage deadline we need categorized. CancelledError
        # is BaseException (not a TimeoutError), so it stays an explicit name match.
        if isinstance(exception, TimeoutError) or exc_name == "CancelledError":
            return "timeout"
        if "timeout" in exc_name.lower() or "timeout" in exc_str:
            return "timeout"
        # A JSON parse failure that arrives as a RAISED exception (a staged
        # call_json reply that wasn't valid JSON — diagnose / surface / index /
        # harvest / intake still ride JSON) is the SAME contract gap as the
        # call_structured loop's parse_error param: classify it identically so the
        # surfacing layer gives the JSON-contract remedy, not an opaque "other".
        # json.JSONDecodeError subclasses ValueError, so match by TYPE, never by
        # message text. (The design-audit path no longer reaches here — it rides a
        # line contract whose own _AuditParseError surfaces loudly as a stage ALERT.)
        if isinstance(exception, json.JSONDecodeError):
            return "parse_error"
        # Match text hints before falling through.
        for marker in _RATE_LIMIT_TEXT_HINTS:
            if marker in exc_str:
                return "external_rate_limit"
        for marker in _OUTAGE_TEXT_HINTS:
            if marker in exc_str:
                return "provider_outage"
        for marker in _INPUT_TOO_LARGE_HINTS:
            if marker in exc_str:
                return "input_too_large"
        for marker in _OUTPUT_TOO_LARGE_HINTS:
            if marker in exc_str:
                return "output_too_large"

    # 4. Raw-response inspection (text markers in non-200 payloads).
    if response_text:
        for marker in _RATE_LIMIT_TEXT_HINTS:
            if marker in response_text:
                return "external_rate_limit"
        for marker in _OUTAGE_TEXT_HINTS:
            if marker in response_text:
                return "provider_outage"
        for marker in _INPUT_TOO_LARGE_HINTS:
            if marker in response_text:
                return "input_too_large"
        for marker in _OUTPUT_TOO_LARGE_HINTS:
            if marker in response_text:
                return "output_too_large"

    # 5. Structured retry reasons from the call_structured() retry loop.
    if validation_errors:
        return "validation_failed"
    if parse_error:
        # Empty / whitespace response is a different bug than a malformed
        # JSON — distinguish them so the surfacing layer can hint at the
        # right contract gap.
        if isinstance(raw_response, str) and not raw_response.strip():
            return "empty_response"
        return "parse_error"

    # 6. Empty response without a parse_error context — provider returned
    # nothing; still our fault, just categorized cleanly.
    if isinstance(raw_response, str) and not raw_response.strip():
        return "empty_response"

    return "other"


def retry_disposition(category: str) -> str:
    """Map a retry category to its enforcement disposition — one of
    DISPOSITION_LEGIT / DISPOSITION_SOFT / DISPOSITION_HARD. Unknown
    categories fall through to soft_retry: surface + retry, never crash
    on an unclassified shape. See RETRY_DISPOSITIONS for the tier
    rationale."""
    return RETRY_DISPOSITIONS.get(category, DISPOSITION_SOFT)


def is_legit_retry(category: str) -> bool:
    """Return True iff retrying this category is FREE — outside our
    control and re-issuing the same call can succeed (external rate
    limit only). False means the retry should mark the run degraded
    (soft) or should never have looped (hard). Binary shortcut over
    retry_disposition()."""
    return retry_disposition(category) == DISPOSITION_LEGIT


def describe_category(category: str) -> str:
    """One-sentence remediation hint for surfacing layers (creator
    panel, build oracle, trace.watch summary). Keep short — these
    appear inline in retry-event renders."""
    descriptions = {
        "external_rate_limit": (
            "provider rate-limited us (429); the retry is correct, "
            "but reorder providers or reduce concurrency if recurring."
        ),
        "provider_outage": (
            "provider returned 5xx / unavailable; outside our control "
            "but worth tracking for provider health."
        ),
        "timeout": (
            "stream hung past the wall-clock cap; investigate call "
            "size or provider routing — our fault if recurring."
        ),
        "empty_response": (
            "provider returned 0 bytes; system prompt / schema "
            "descriptor likely incomplete. CONTRACT GAP."
        ),
        "parse_error": (
            "response wasn't valid JSON; tighten the system prompt "
            "with explicit JSON-only preamble. CONTRACT GAP."
        ),
        "validation_failed": (
            "schema rejected the LLM output; schema_for_prompt + "
            "validator drift. CONTRACT GAP."
        ),
        "output_too_large": (
            "output truncated mid-stream; one-artifact-per-call "
            "contract violated. Split the call. CONTRACT GAP."
        ),
        "input_too_large": (
            "context-window exceeded on input; trim brief / shed "
            "payload. CONTRACT GAP."
        ),
        "provider_blocked": (
            "account-level block; not a retry, surfaces here only "
            "for categorization completeness."
        ),
        "other": (
            "uncategorized retry; add a category if this appears "
            "repeatedly."
        ),
    }
    return descriptions.get(category, descriptions["other"])
