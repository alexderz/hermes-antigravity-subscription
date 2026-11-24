"""Transport-aware context accounting for the ``agy`` CLI.

Two limits govern how much this provider can put in front of the model, and they
are not the model's advertised window.

1. **Single prompt frame.** ``agy`` caps one prompt frame. Measured against
   ``agy`` 1.5.15 on 2026-11-23 with two plain-text stdin prompts: 420,544 and
   250,378 bytes both kept exactly 191,583 bytes, and reported
   ``usage.input_tokens`` 47,345 and 47,353. The CLI now names the truncation in
   its reply ("228,961 bytes truncated") instead of dropping the tail silently,
   but the payload is still cut, so a prompt has to be sized to fit *before* it
   is sent.

2. **Live trajectory.** A multi-turn conversation persisted with
   ``agy --conversation`` is bounded by the compiled trajectory checkpoint
   (239,616 tokens), less the output reserve. That, not a per-family guess, is
   what Hermes should compact against, because continuation turns send only a
   delta and never approach the frame cap.

The declared window is the model's published context
(:func:`model_context_length`, 1,048,576 for the Gemini 4 family), not a
transport-round number: a live conversation grows through deltas and agy keeps
its own trajectory, auto-compacting above a threshold it allows to be raised to
the model window. What the transport does cap is a *single frame*, and
:func:`prompt_byte_budget` keeps every prompt inside the measured value. If a
reply still reports a truncation, :func:`note_truncation` learns the real cap
from it so the next prompt is sized against the observed value.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Declared windows are the models' own published context, taken from their
# OpenRouter cards on 2026-11-23 and, for the Antigravity-only ids, from the
# Gemini 4 family they route to (google/gemini-4-neon|carbon|argon|hydrogen are
# 1,048,576 with a 943,718 max output). agy keeps its own trajectory and
# auto-compacts at a threshold it allows to be set up to the model window
# (AGY_CLI_AUTO_COMPACT_OVERRIDE, per the 1.5.x changelog), so the model card is
# the honest number: Hermes compacts first, at its own threshold, and agy's
# internal compaction stays the backstop rather than the trigger.
GEMINI_4_CONTEXT = 1_048_576
CLAUDE_5_CONTEXT = 1_000_000
CLAUDE_LEGACY_CONTEXT = 200_000
GPT_OSS_CONTEXT = 131_072
DEFAULT_CONTEXT = 1_048_576

# Longest-prefix wins, so an effort suffix (``-high``, ``-xhigh``) needs no
# separate entry.
_CONTEXT_BY_PREFIX: tuple[tuple[str, int], ...] = (
    ("gemini-4-", GEMINI_4_CONTEXT),
    ("gemini-pro-orchestrator", GEMINI_4_CONTEXT),
    ("gemini-3.8-flash", GEMINI_4_CONTEXT),
    ("gemini-3.", GEMINI_4_CONTEXT),
    ("claude-opus-5-5", CLAUDE_5_CONTEXT),
    ("claude-sonnet-5-5", CLAUDE_5_CONTEXT),
    ("claude-opus-4-6", CLAUDE_LEGACY_CONTEXT),
    ("claude-sonnet-4-6", CLAUDE_LEGACY_CONTEXT),
    ("gpt-oss", GPT_OSS_CONTEXT),
)

#: Largest single prompt frame observed to survive agy 1.5.15 unchanged.
MEASURED_SINGLE_TURN_BYTES = 191_583

#: Fraction of the cap we are willing to fill. The measured value is exact for
#: the text that was probed; tokenizers and framing differ per payload, so keep a
#: margin rather than sending the cap itself.
BUDGET_FRACTION = 0.94

_BUDGET_ENV_VARS = ("HERMES_ANTIGRAVITY_PROMPT_BYTES", "ANTIGRAVITY_PROMPT_BYTES")

# agy names the cut in the reply. Two phrasings have been observed:
#   "Your message was truncated after `Record 00405: T` (`228,961 bytes` truncated)"
#   "Your message was truncated after `Record 00405: T` (`<truncated 58795 bytes>`)"
_TRUNCATION_PATTERNS = (
    re.compile(r"`(\d[\d,]*)\s*bytes`\s*truncated", re.IGNORECASE),
    re.compile(r"<truncated\s+(\d[\d,]*)\s*bytes>", re.IGNORECASE),
)
_TRUNCATION_HINT = re.compile(r"message was truncated", re.IGNORECASE)

_learned_cap: Optional[int] = None
_learned_lock = threading.Lock()


@dataclass(frozen=True)
class TruncationNotice:
    """What an agy reply says was cut from the prompt."""

    dropped_bytes: int
    observed_cap: Optional[int] = None


def _positive_int(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "").replace("_", "")
        if text.isdigit():
            value = int(text)
            return value if value > 0 else None
    return None


def _env_int(names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        value = _positive_int(os.environ.get(name))
        if value is not None:
            return value
    return None


def learned_single_turn_bytes() -> Optional[int]:
    """Cap observed from a real truncation notice, if one has been seen."""
    with _learned_lock:
        return _learned_cap


def single_turn_byte_cap() -> int:
    """Byte cap to size prompts against: the smallest of config, env and observation."""
    configured = _env_int(_BUDGET_ENV_VARS)
    if configured is not None:
        return configured
    observed = learned_single_turn_bytes()
    if observed is not None:
        return min(MEASURED_SINGLE_TURN_BYTES, observed)
    return MEASURED_SINGLE_TURN_BYTES


def prompt_byte_budget() -> int:
    """Budget for one assembled prompt: the cap less :data:`BUDGET_FRACTION` margin."""
    cap = single_turn_byte_cap()
    budget = int(cap * BUDGET_FRACTION)
    # Never let a pathological env value produce a zero-byte budget.
    return max(budget, 4_096)


def model_context_length(model: str) -> int:
    """Published context window for *model*, longest prefix first.

    Unknown ids fall back to :data:`DEFAULT_CONTEXT`; a model we have never seen
    is far more likely to be another 1M Gemini variant than a small one, and
    agy's own trajectory limit fails the turn rather than silently truncating if
    that guess is wrong for a run.
    """
    name = (model or "").strip().lower()
    best: tuple[int, int] | None = None
    for prefix, size in _CONTEXT_BY_PREFIX:
        if name.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), size)
    return best[1] if best else DEFAULT_CONTEXT


def prompt_bytes(text: str) -> int:
    """UTF-8 size of a prompt, which is what the transport counts."""
    return len(text.encode("utf-8", "replace"))


def parse_truncation_notice(text: str) -> Optional[TruncationNotice]:
    """Parse agy's truncation notice out of a reply, if it carries one.

    Returns ``None`` when the reply says nothing about truncation. A notice whose
    byte count cannot be parsed is still reported, with ``dropped_bytes = 0``, so
    callers can log it without inventing a number.
    """
    if not text or not isinstance(text, str):
        return None
    if not _TRUNCATION_HINT.search(text):
        return None
    for pattern in _TRUNCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            dropped = _positive_int(match.group(1))
            if dropped is not None:
                return TruncationNotice(dropped_bytes=dropped)
    return TruncationNotice(dropped_bytes=0)


def note_truncation(text: str, prompt_size: int) -> Optional[TruncationNotice]:
    """Record a truncation reported by agy, learning the real cap from it.

    ``prompt_size`` is the size of the prompt that was actually sent. The cap is
    ``prompt_size - dropped_bytes``: whatever survived is the transport limit.
    A learned cap only ever lowers the value we size against.
    """
    notice = parse_truncation_notice(text)
    if notice is None:
        return None
    observed: Optional[int] = None
    if notice.dropped_bytes and prompt_size > 0:
        surviving = prompt_size - notice.dropped_bytes
        if surviving > 0:
            observed = surviving
            global _learned_cap
            with _learned_lock:
                if _learned_cap is None or surviving < _learned_cap:
                    _learned_cap = surviving
    logger.warning(
        "agy truncated the prompt: %s bytes dropped from a %s-byte frame "
        "(observed cap %s, budget now %s bytes).",
        notice.dropped_bytes or "unknown",
        prompt_size or "unknown",
        observed if observed is not None else "unknown",
        prompt_byte_budget(),
    )
    return TruncationNotice(dropped_bytes=notice.dropped_bytes, observed_cap=observed)


def reset_learned_cap() -> None:
    """Forget a learned cap (tests, or after an agy upgrade)."""
    global _learned_cap
    with _learned_lock:
        _learned_cap = None


def omitted_notice(omitted_messages: int, budget: int) -> str:
    """Text that tells the model older turns were dropped to fit the frame."""
    plural = "s" if omitted_messages != 1 else ""
    return (
        "### EARLIER CONTEXT OMITTED\n"
        f"The transport accepts about {budget:,} bytes per prompt, so {omitted_messages} older "
        f"message{plural} were omitted from this transcript. Ask the user for specifics, or "
        "re-read a file, if the omitted turn matters."
    )


def hard_trim(text: str, budget: int) -> str:
    """Last-resort cut: keep the head and the tail, mark the hole.

    Only reached when even the harshest structured pass does not fit. Keeping
    both ends is deliberate: the head carries the system instructions and tool
    schema, the tail carries the live turn.
    """
    if prompt_bytes(text) <= budget:
        return text
    marker = f"\n\n[... {0} bytes of transcript omitted to fit the agy frame ...]\n\n"
    # Reserve room for the marker itself (~60 bytes) and split the remainder.
    room = max(budget - 200, 1_024)
    head_target = room // 2
    tail_target = room - head_target
    head = text[:head_target]
    tail = text[-tail_target:]
    marker = marker.format(max(prompt_bytes(text) - prompt_bytes(head) - prompt_bytes(tail), 0))
    trimmed = head + marker + tail
    # Byte-multibyte safety: cut until it fits.
    while prompt_bytes(trimmed) > budget and len(tail) > 0:
        overflow = prompt_bytes(trimmed) - budget
        tail = tail[overflow:]
        trimmed = head + marker + tail
    return trimmed


def describe_budget() -> dict[str, Any]:
    """Diagnostic snapshot for logs and tests."""
    return {
        "declared_context": DEFAULT_CONTEXT,
        "measured_cap": MEASURED_SINGLE_TURN_BYTES,
        "learned_cap": learned_single_turn_bytes(),
        "effective_cap": single_turn_byte_cap(),
        "budget": prompt_byte_budget(),
    }


def format_describe() -> str:
    return json.dumps(describe_budget(), sort_keys=True)
