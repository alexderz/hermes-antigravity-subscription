"""Hermes-side hooks: setup status, picker rows, aux model, compact ordering.

These are the surfaces Hermes asks about outside inference, plus the two
behaviours that keep the harness from acting behind Hermes's back (its own
compaction ordering) and from aborting turns it is about to finish (transient
429s versus a real quota verdict).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import models
import process
import setup_hooks

AGY_MODELS = (
    "Fetching available models...\n"
    "gemini-4-neon-high\tGemini 4 Neon (High)\n"
    "gemini-4-neon-xhigh\tGemini 4 Neon (XHigh)\n"
    "gemini-4-neon-medium\tGemini 4 Neon (Medium)\n"
    "gemini-4-neon-low\tGemini 4 Neon (Low)\n"
    "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
    "claude-opus-5-5-high\tClaude Opus 5.5 (High)\n"
    "gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n"
)


class CatalogRowsTest(unittest.TestCase):
    def test_rows_group_efforts_and_keep_labels(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(
            by_id["gemini-4-neon"],
            {"id": "gemini-4-neon", "label": "Gemini 4 Neon", "note": "low, medium, high"},
        )
        self.assertEqual(by_id["claude-opus-5-5"]["label"], "Claude Opus 5.5")
        self.assertEqual(by_id["claude-opus-5-5"]["note"], "high")
        self.assertNotIn("gemini-4-neon-xhigh", by_id, "xhigh is a duplicate of high, not its own model")

    def test_empty_catalog(self):
        self.assertEqual(models.parse_catalog_rows(""), [])


class SetupStatusTest(unittest.TestCase):
    def test_missing_binary(self):
        with patch.object(setup_hooks, "_helpers", return_value=(lambda: {}, lambda: None, lambda: None)):
            status = setup_hooks.setup_status()
        self.assertFalse(status["available"])
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["detail"], setup_hooks.INSTALL_HINT)

    def test_signed_in_via_usage_read(self):
        usage = "Gemini Models\tFive Hour Limit Remaining\t62%\t2026-11-24T05:51:46Z\n"

        def fake_helpers():
            return (lambda: {}, lambda: "agy", lambda: None)

        with patch.object(setup_hooks, "_helpers", side_effect=fake_helpers), patch.object(
            setup_hooks, "_usage_probe", return_value=usage
        ):
            status = setup_hooks.setup_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["logged_in"])
        self.assertIn("62%", status["detail"])
        self.assertIsNone(status["login_command"], "agy has no non-interactive login verb")

    def test_signed_out(self):
        with patch.object(setup_hooks, "_helpers", return_value=(lambda: {}, lambda: "agy", lambda: None)), patch.object(
            setup_hooks, "_usage_probe", return_value=None
        ):
            status = setup_hooks.setup_status()
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["detail"], setup_hooks.LOGIN_HINT)


class DiscoveryHooksTest(unittest.TestCase):
    def test_discover_models_from_catalog(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(models, "load_catalog_rows", return_value=rows):
            discovered = setup_hooks.discover_models()
        assert discovered is not None
        self.assertEqual([row["id"] for row in discovered][:2], ["gemini-4-neon", "gemini-3.8-flash"])

    def test_discover_models_none_when_empty(self):
        with patch.object(models, "load_catalog_rows", return_value=[]):
            self.assertIsNone(setup_hooks.discover_models())

    def test_aux_model_prefers_flash_low(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.resolve_aux_model(), "gemini-3.8-flash-low")

    def test_vision_model(self):
        rows = models.parse_catalog_rows(AGY_MODELS)
        with patch.object(models, "load_catalog_rows", return_value=rows):
            self.assertEqual(setup_hooks.default_vision_model(), "gemini-4-neon")


class CompactOrderingTest(unittest.TestCase):
    def setUp(self):
        for name in ("AGY_CLI_AUTO_COMPACT_OVERRIDE", "ANTIGRAVITY_AUTO_COMPACT"):
            os.environ.pop(name, None)

    def tearDown(self):
        for name in ("AGY_CLI_AUTO_COMPACT_OVERRIDE", "ANTIGRAVITY_AUTO_COMPACT"):
            os.environ.pop(name, None)

    def test_default_keeps_agy_behind_hermes(self):
        self.assertEqual(setup_hooks.auto_compact_override(), "90%")

    def test_user_setting_wins(self):
        os.environ["AGY_CLI_AUTO_COMPACT_OVERRIDE"] = "70%"
        self.assertEqual(setup_hooks.auto_compact_override(), "70%")

    def test_provider_specific_knob(self):
        os.environ["ANTIGRAVITY_AUTO_COMPACT"] = "0.8M"
        self.assertEqual(setup_hooks.auto_compact_override(), "0.8M")

    def test_child_env_carries_the_override(self):
        env = process.build_child_env(Path(tempfile.mkdtemp(prefix="agy-env-")))
        self.assertEqual(env.get("AGY_CLI_AUTO_COMPACT_OVERRIDE"), "90%")


class QuotaWatchdogTest(unittest.TestCase):
    def _log(self, text: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix="agy-log-"))
        (root / "log").mkdir()
        (root / "log" / "cli-test.log").write_text(text, encoding="utf-8")
        return root

    def test_transient_429_does_not_abort(self):
        """agy retries these itself; the watchdog must not fail the turn."""
        root = self._log("RESOURCE_EXHAUSTED (code 429): rate limited, retrying in 3s\n")
        self.assertIsNone(process._check_early_quota_error(root))

    def test_real_quota_verdict_aborts(self):
        root = self._log(
            "RESOURCE_EXHAUSTED (code 429): Individual quota reached. Please upgrade your subscription. Resets in 2h55m37s.\n"
        )
        detail = process._check_early_quota_error(root)
        assert detail is not None
        self.assertIn("Individual quota reached", detail)

    def test_billing_cap_aborts(self):
        root = self._log("Your daily or billing cap has been reached.\n")
        self.assertIsNotNone(process._check_early_quota_error(root))


if __name__ == "__main__":
    unittest.main()
