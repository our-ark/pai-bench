from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FAKE_CLAUDE = ROOT / "tests" / "fixtures" / "fake-claude.py"

from identity_benchmark.claude_evaluator import (
    ClaudeEvaluator,
    ClaudeEvaluatorError,
    IMPLEMENTATION_ID,
)
from test_codex_evaluator import _request


class ClaudeEvaluatorTests(unittest.TestCase):
    def test_claude_evaluator_uses_isolated_noninteractive_flags(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "claude-log.json"
            with patch.dict(
                os.environ,
                {
                    "FAKE_CLAUDE_LOG": str(log),
                    "FAKE_CLAUDE_SCORE": "0.75",
                },
                clear=False,
            ):
                result = ClaudeEvaluator(
                    evaluator_id="claude-judge-v1",
                    model="sonnet",
                    reasoning_effort="high",
                    state_home=root / "state",
                    rubric_version="pai-model-judge-v2",
                    claude_bin=str(FAKE_CLAUDE),
                    max_budget_usd=0.25,
                    timeout_seconds=5,
                ).evaluate(_request())
            recorded = json.loads(log.read_text(encoding="utf-8"))

        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.metadata["implementation"], IMPLEMENTATION_ID)
        self.assertEqual(result.metadata["evaluator_id"], "claude-judge-v1")
        self.assertEqual(result.metadata["provider"], "anthropic")
        self.assertEqual(result.metadata["input_tokens"], 26)
        self.assertEqual(result.metadata["cached_input_tokens"], 3)
        self.assertEqual(result.metadata["total_cost_usd"], 0.0125)
        self.assertIn("--safe-mode", recorded["args"])
        self.assertIn("--restricted", recorded["args"])
        self.assertIn("--strict-mcp-config", recorded["args"])
        self.assertIn("--no-session-persistence", recorded["args"])
        self.assertEqual(
            recorded["args"][recorded["args"].index("--prompt-suggestions") + 1],
            "false",
        )
        self.assertEqual(
            recorded["args"][recorded["args"].index("--tools") + 1], ""
        )
        self.assertEqual(
            recorded["args"][recorded["args"].index("--model") + 1], "sonnet"
        )
        self.assertEqual(
            recorded["args"][recorded["args"].index("--effort") + 1], "high"
        )
        self.assertEqual(
            recorded["args"][recorded["args"].index("--max-budget-usd") + 1],
            "0.25",
        )
        self.assertIn("Target response", recorded["prompt"])
        self.assertEqual(
            recorded["schema"]["properties"]["score"]["enum"],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )

    def test_score_outside_frozen_scale_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {"FAKE_CLAUDE_SCORE": "0.3"},
                clear=False,
            ):
                with self.assertRaisesRegex(ClaudeEvaluatorError, "must be one of"):
                    ClaudeEvaluator(
                        evaluator_id="claude-judge-v1",
                        model="sonnet",
                        reasoning_effort="high",
                        state_home=root / "state",
                        claude_bin=str(FAKE_CLAUDE),
                        timeout_seconds=5,
                    ).evaluate(_request())

    @unittest.skipIf(sys.platform == "win32", "requires POSIX process groups")
    def test_timeout_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {"FAKE_CLAUDE_SLEEP": "30"},
                clear=False,
            ):
                started = time.monotonic()
                with self.assertRaisesRegex(
                    ClaudeEvaluatorError, "timed out after 0.05 seconds"
                ):
                    ClaudeEvaluator(
                        evaluator_id="claude-judge-v1",
                        model="sonnet",
                        reasoning_effort="high",
                        state_home=root / "state",
                        claude_bin=str(FAKE_CLAUDE),
                        timeout_seconds=0.05,
                    ).evaluate(_request())
                elapsed = time.monotonic() - started

        self.assertLess(elapsed, 3.0)

    def test_transient_exit_is_retried(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts.txt"
            with patch.dict(
                os.environ,
                {
                    "FAKE_CLAUDE_ATTEMPT_FILE": str(attempts),
                    "FAKE_CLAUDE_FAIL_ATTEMPTS": "2",
                },
                clear=False,
            ):
                result = ClaudeEvaluator(
                    evaluator_id="claude-judge-v1",
                    model="sonnet",
                    reasoning_effort="high",
                    state_home=root / "state",
                    claude_bin=str(FAKE_CLAUDE),
                    timeout_seconds=5,
                ).evaluate(_request())

        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metadata["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
