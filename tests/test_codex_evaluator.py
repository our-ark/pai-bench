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
FAKE_CODEX = ROOT / "tests" / "fixtures" / "fake-codex.py"

from identity_benchmark.codex_evaluator import (
    CodexEvaluator,
    CodexEvaluatorError,
    IMPLEMENTATION_ID,
)
from identity_benchmark.contracts import (
    Expectation,
    IdentityStatement,
    Message,
    Probe,
)
from identity_benchmark.evaluators import EvaluationRequest


class CodexEvaluatorTests(unittest.TestCase):
    def test_codex_evaluator_uses_isolated_noninteractive_flags(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "codex-log.json"
            with patch.dict(
                os.environ,
                {
                    "FAKE_CODEX_LOG": str(log),
                    "FAKE_CODEX_SCORE": "0.75",
                },
                clear=False,
            ):
                result = CodexEvaluator(
                    evaluator_id="judge-v1",
                    model="judge-model",
                    reasoning_effort="xhigh",
                    state_home=root / "state",
                    rubric_version="pai-model-judge-v2",
                    codex_bin=str(FAKE_CODEX),
                    timeout_seconds=5,
                ).evaluate(_request())
            recorded = json.loads(log.read_text(encoding="utf-8"))

        self.assertEqual(result.score, 0.75)
        self.assertEqual(
            result.metadata["implementation"],
            IMPLEMENTATION_ID,
        )
        self.assertEqual(result.metadata["evaluator_id"], "judge-v1")
        self.assertEqual(result.metadata["input_tokens"], 21)
        self.assertIn("--ephemeral", recorded["args"])
        self.assertIn("--ignore-user-config", recorded["args"])
        self.assertIn("--ignore-rules", recorded["args"])
        self.assertIn("--skip-git-repo-check", recorded["args"])
        self.assertIn("--output-schema", recorded["args"])
        self.assertEqual(
            recorded["args"][recorded["args"].index("--model") + 1],
            "judge-model",
        )
        self.assertIn('model_reasoning_effort="xhigh"', recorded["args"])
        self.assertIn("Target response", recorded["prompt"])
        self.assertIn("candidate response", recorded["prompt"])
        self.assertIn("pai-model-judge-v2", recorded["prompt"])
        self.assertIn("adversarial open-response rubric", recorded["prompt"])
        self.assertEqual(
            recorded["schema"]["properties"]["score"]["enum"],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )

    def test_score_outside_frozen_scale_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "FAKE_CODEX_LOG": str(root / "log.json"),
                    "FAKE_CODEX_SCORE": "0.3",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(
                    CodexEvaluatorError,
                    "must be one of",
                ):
                    CodexEvaluator(
                        evaluator_id="judge-v1",
                        model="judge-model",
                        reasoning_effort="high",
                        state_home=root / "state",
                        codex_bin=str(FAKE_CODEX),
                        timeout_seconds=5,
                    ).evaluate(_request())

    @unittest.skipIf(sys.platform == "win32", "requires POSIX process groups")
    def test_timeout_is_reported_by_codex_evaluator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "FAKE_CODEX_SLEEP": "30",
                    "FAKE_CODEX_SCORE": "1",
                },
                clear=False,
            ):
                started = time.monotonic()
                with self.assertRaisesRegex(
                    CodexEvaluatorError,
                    "timed out after 0.05 seconds",
                ):
                    CodexEvaluator(
                        evaluator_id="judge-v1",
                        model="judge-model",
                        reasoning_effort="high",
                        state_home=root / "state",
                        codex_bin=str(FAKE_CODEX),
                        timeout_seconds=0.05,
                    ).evaluate(_request())
                elapsed = time.monotonic() - started

        self.assertLess(elapsed, 3.0)


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        profile_id="profile-1",
        statements=(
            IdentityStatement(
                id="designation",
                content="The designation is VECTOR-9.",
            ),
        ),
        probe=Probe(
            id="designation",
            dimension="recognition",
            messages=(
                Message(role="system", content="Use the installed identity."),
                Message(role="user", content="State the designation."),
            ),
            expectations=(
                Expectation(
                    type="contains",
                    value="VECTOR-9",
                    gate=True,
                ),
            ),
            tags=("identity-fact",),
        ),
        agent_response="candidate response",
    )


if __name__ == "__main__":
    unittest.main()
