"""Context accounting: the declared window and the per-frame byte budget.

agy 1.5.15 keeps only the first 191,583 bytes of a single prompt frame (measured
2026-11-23: two prompts of 420,544 and 250,378 bytes both kept 191,583, with
``usage.input_tokens`` 47,345 and 47,353) and now names the cut in its reply.
These tests pin the two halves of the fix: one declared window for every family
(the trajectory checkpoint), and a prompt builder that fits each frame instead of
letting the transport discard the tail.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import agy_context
from client import _format_delta_prompt, _format_messages_as_prompt
from agy_context import (
    MEASURED_SINGLE_TURN_BYTES,
    TRAJECTORY_CONTEXT,
    hard_trim,
    note_truncation,
    parse_truncation_notice,
    prompt_byte_budget,
    prompt_bytes,
    reset_learned_cap,
)


class DeclaredContextTest(unittest.TestCase):
    def test_every_family_declares_the_trajectory_checkpoint(self):
        import importlib

        entry = importlib.import_module("antigravity_plugin_entry")
        profile = entry.antigravity_profile
        for model in ("gemini-4-neon", "gemini-4-neon-high", "gemini-4-carbon-low", "claude-opus-5-5"):
            with self.subTest(model=model):
                self.assertEqual(profile.get_model_context_length(model), TRAJECTORY_CONTEXT)

    def test_env_override_still_wins(self):
        import importlib

        entry = importlib.import_module("antigravity_plugin_entry")
        profile = entry.antigravity_profile
        os.environ["HERMES_ANTIGRAVITY_CONTEXT_LENGTH"] = "123456"
        try:
            self.assertEqual(profile.get_model_context_length("gemini-4-neon"), 123456)
        finally:
            windows = ("HERMES_ANTIGRAVITY_CONTEXT_LENGTH",)
            os.environ["HERMES_ANTIGRAVITY_CONTEXT_LENGTH"] = ""
            for name in windows:
                os.environ.pop(name, None)


class BudgetTest(unittest.TestCase):
    def setUp(self):
        reset_learned_cap()
        for name in (
            "HERMES_ANTIGRAVITY_PROMPT_BYTES",
            "ANTIGRAVITY_PROMPT_BYTES",
            "HERMES_ANTIGRAVITY_CONTEXT_LENGTH",
            "ANTIGRAVITY_CONTEXT_LENGTH",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        reset_learned_cap()
        for name in ("HERMES_ANTIGRAVITY_PROMPT_BYTES", "ANTIGRAVITY_PROMPT_BYTES"):
            os.environ.pop(name, None)

    def test_budget_sits_under_the_measured_cap(self):
        self.assertLess(prompt_byte_budget(), MEASURED_SINGLE_TURN_BYTES)
        self.assertGreater(prompt_byte_budget(), MEASURED_SINGLE_TURN_BYTES * 0.9)

    def test_env_budget_override(self):
        os.environ["HERMES_ANTIGRAVITY_PROMPT_BYTES"] = "90000"
        self.assertEqual(agy_context.single_turn_byte_cap(), 90000)
        self.assertLessEqual(prompt_byte_budget(), 90000)

    @staticmethod
    def _big_history(tool_chars=120_000, tools=12, user_text="Fix the parser please."):
        messages = [{"role": "system", "content": "You are Hermes."}]
        for i in range(tools):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "big.py"}'},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": ("def f():\n    return 1\n" * (tool_chars // 22)),
                }
            )
        messages.append({"role": "user", "content": user_text})
        return messages

    def test_full_send_fits_the_budget_and_keeps_the_live_request(self):
        budget = prompt_byte_budget()
        text = _format_messages_as_prompt(self._big_history())
        self.assertLessEqual(prompt_bytes(text), budget)
        self.assertIn("Fix the parser please.", text)
        # The newest tool result survives intact-ish; the old ones are the ones cut.
        self.assertIn("Tool Result (call_11)", text)

    def test_tool_output_is_cut_before_turns_are_dropped(self):
        """The ladder prefers shrinking tool payloads over losing conversation."""
        text = _format_messages_as_prompt(self._big_history(tool_chars=300_000, tools=20))
        self.assertLessEqual(prompt_bytes(text), prompt_byte_budget())
        self.assertIn("tool output truncated", text)
        self.assertNotIn("EARLIER CONTEXT OMITTED", text)

    def test_dropped_history_is_announced_when_nothing_else_fits(self):
        """With no tool payload to shrink, whole turns go and the model is told."""
        os.environ["HERMES_ANTIGRAVITY_PROMPT_BYTES"] = "20000"
        messages = [{"role": "system", "content": "You are Hermes."}]
        for i in range(60):
            messages.append({"role": "user", "content": f"question {i} " + ("q" * 2_000)})
            messages.append({"role": "assistant", "content": f"answer {i} " + ("a" * 2_000)})
        messages.append({"role": "user", "content": "final question"})
        text = _format_messages_as_prompt(messages)
        self.assertLessEqual(prompt_bytes(text), prompt_byte_budget())
        self.assertIn("EARLIER CONTEXT OMITTED", text)
        self.assertIn("final question", text)

    def test_small_prompt_untouched(self):
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Say hi."},
        ]
        text = _format_messages_as_prompt(messages)
        self.assertNotIn("truncated to fit the prompt budget", text)
        self.assertNotIn("EARLIER CONTEXT OMITTED", text)
        self.assertIn("Say hi.", text)

    def test_delta_prompt_fits(self):
        huge = "x" * 400_000
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": huge},
            {"role": "user", "content": "keep going"},
        ]
        text = _format_delta_prompt(messages[1:], pending_user_text="keep going")
        self.assertLessEqual(prompt_bytes(text), prompt_byte_budget())

    def test_hard_trim_fits(self):
        blob = "z" * 500_000
        trimmed = hard_trim(blob, 100_000)
        self.assertLessEqual(prompt_bytes(trimmed), 100_000)
        self.assertIn("omitted to fit the agy frame", trimmed)


class TruncationNoticeTest(unittest.TestCase):
    def setUp(self):
        reset_learned_cap()

    def tearDown(self):
        reset_learned_cap()

    def test_parses_backticked_notice(self):
        reply = (
            "Your message was truncated after `Record 00405: T` (`228,961 bytes` truncated), "
            "so I didn't receive the rest of the records."
        )
        notice = parse_truncation_notice(reply)
        assert notice is not None
        self.assertEqual(notice.dropped_bytes, 228961)

    def test_parses_angle_bracket_notice(self):
        reply = "Your message was truncated after `Record 00405: T` (`<truncated 58795 bytes>`)."
        notice = parse_truncation_notice(reply)
        assert notice is not None
        self.assertEqual(notice.dropped_bytes, 58795)

    def test_ignores_ordinary_reply(self):
        self.assertIsNone(parse_truncation_notice("All twelve tests passed."))

    def test_matching_measurement_does_not_move_the_budget(self):
        """Re-observing the same 191,583-byte cap confirms it; nothing changes."""
        before = prompt_byte_budget()
        reply = "Your message was truncated after `x` (`<truncated 58795 bytes>`)."
        notice = note_truncation(reply, 250_378)
        self.assertIsNotNone(notice)
        assert notice is not None  # for type checkers
        self.assertEqual(notice.observed_cap, 250_378 - 58_795)
        self.assertEqual(prompt_byte_budget(), before)

    def test_a_smaller_observed_cap_lowers_the_budget(self):
        """A stricter limit than the measured one wins, so prompts shrink."""
        before = prompt_byte_budget()
        reply = "Your message was truncated after `x` (`<truncated 20000 bytes>`)."
        notice = note_truncation(reply, 100_000)
        assert notice is not None
        self.assertEqual(notice.observed_cap, 80_000)
        self.assertEqual(agy_context.single_turn_byte_cap(), 80_000)
        self.assertLess(prompt_byte_budget(), before)


if __name__ == "__main__":
    unittest.main()
