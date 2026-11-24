"""Harness context discovery: parsing, cache, and lookup precedence."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import harness_context as hc


class EventParsingTest(unittest.TestCase):
    def test_effort_suffix_split(self):
        self.assertEqual(hc._split_effort("gemini-4-neon-high"), ("gemini-4-neon", "high"))
        self.assertEqual(hc._split_effort("gemini-4-hydrogen-xhigh"), ("gemini-4-hydrogen", "high"))
        self.assertEqual(hc._split_effort("gpt-oss-120b-medium"), ("gpt-oss-120b", "medium"))
        self.assertEqual(hc._split_effort("gemini-pro-orchestrator"), ("gemini-pro-orchestrator", ""))

    def test_last_usage_update_wins(self):
        events = [
            {"sessionUpdate": "usage_update", "sessionId": "s1", "size": 1_000_000},
            {"sessionUpdate": "available_commands_update", "sessionId": "s1"},
            {"sessionUpdate": "usage_update", "sessionId": "s2", "size": 131_072},
            {"sessionUpdate": "usage_update", "sessionId": "s1", "size": 1_048_576},
        ]
        self.assertEqual(hc._session_size(events, "s1"), 1_048_576)
        self.assertEqual(hc._session_size(events, "s2"), 131_072)
        self.assertIsNone(hc._session_size(events, "s3"))


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agy-windows-")
        self.path = Path(self.tmp) / "windows.json"
        self.env = patch.dict(os.environ, {"HERMES_ANTIGRAVITY_WINDOWS_CACHE": str(self.path)})
        self.env.start()
        hc.reset_memo()

    def tearDown(self):
        self.env.stop()
        hc.reset_memo()

    def _write(self, windows, *, fetched_at=None, agy="1.5.15"):
        self.path.write_text(
            json.dumps(
                {
                    "fetched_at": time.time() if fetched_at is None else fetched_at,
                    "agy_version": agy,
                    "windows": windows,
                }
            ),
            encoding="utf-8",
        )
        hc.reset_memo()

    def test_round_trip(self):
        with patch.object(hc, "agy_version", return_value="1.5.15"):
            hc.save_cached_windows({"gemini-4-neon": 1_048_576, "gpt-oss-120b": 131_072})
            windows = hc.load_cached_windows()
        self.assertEqual(windows["gemini-4-neon"], 1_048_576)
        self.assertEqual(windows["gpt-oss-120b"], 131_072)

    def test_lookup_prefers_longest_prefix(self):
        self._write({"gemini-4-neon": 1_048_576, "gemini-4-neon-high": 900_000})
        with patch.object(hc, "agy_version", return_value="1.5.15"):
            self.assertEqual(hc.lookup("gemini-4-neon-high"), 900_000)
            self.assertIsNone(hc.lookup("gemini-4-carbon-low"))

    def test_stale_cache_is_ignored(self):
        self._write({"gemini-4-neon": 1_048_576}, fetched_at=time.time() - hc.CACHE_TTL_SECONDS - 60)
        with patch.object(hc, "agy_version", return_value="1.5.15"):
            self.assertEqual(hc.load_cached_windows(), {})

    def test_new_agy_build_invalidates(self):
        self._write({"gemini-4-neon": 1_048_576}, agy="1.5.14")
        with patch.object(hc, "agy_version", return_value="1.5.15"):
            self.assertEqual(hc.load_cached_windows(), {})

    def test_lookup_missing_cache(self):
        self.assertIsNone(hc.lookup("gemini-4-neon"))


if __name__ == "__main__":
    unittest.main()
