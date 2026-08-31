from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FAKE_CODEX = ROOT / "tests" / "fixtures" / "fake-codex.py"

from identity_benchmark.codex_evaluator import (
    ADAPTER_ID,
    CodexEvaluatorError,
    evaluate_with_codex,
)


class CodexEvaluatorAdapterTests(unittest.TestCase):
    def test_independent_codex_process_uses_isolated_noninteractive_flags(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "codex-log.json"
            with patch.dict(
                os.environ,
                {
                    "FAKE_CODEX_LOG": str(log),
                    "FAKE_CODEX_SCORE": "0.75",
                    "IDENTITY_BENCHMARK_EVALUATOR_ID": "judge-v1",
                    "IDENTITY_BENCHMARK_EVALUATOR_RUBRIC_VERSION": "pai-model-judge-v2",
                },
                clear=False,
            ):
                result = evaluate_with_codex(
                    _request(),
                    model="judge-model",
                    reasoning_effort="xhigh",
                    state_home=root / "state",
                    codex_bin=str(FAKE_CODEX),
                    timeout_seconds=5,
                )
            recorded = json.loads(log.read_text(encoding="utf-8"))

        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["metadata"]["adapter"], ADAPTER_ID)
        self.assertEqual(result["metadata"]["evaluator_id"], "judge-v1")
        self.assertEqual(result["metadata"]["input_tokens"], 21)
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
                    CodexEvaluatorError, "must be one of"
                ):
                    evaluate_with_codex(
                        _request(),
                        model="judge-model",
                        reasoning_effort="high",
                        state_home=root / "state",
                        codex_bin=str(FAKE_CODEX),
                        timeout_seconds=5,
                    )


def _request() -> dict:
    return {
        "protocol_version": 1,
        "profile_id": "profile-1",
        "statements": [
            {"id": "designation", "content": "The designation is VECTOR-9."}
        ],
        "probe": {
            "id": "designation",
            "dimension": "recognition",
            "messages": [
                {"role": "user", "content": "State the designation."}
            ],
            "expectations": [
                {
                    "type": "contains",
                    "value": "VECTOR-9",
                    "weight": 1.0,
                    "gate": True,
                }
            ],
            "tags": ["identity-fact"],
        },
        "agent_response": "candidate response",
    }


if __name__ == "__main__":
    unittest.main()
