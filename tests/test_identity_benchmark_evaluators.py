from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = ROOT / "tests" / "fixtures"

from identity_benchmark import (
    AgentAdapter,
    CodexEvaluator,
    EnochAdapter,
)
from identity_benchmark.contracts import BenchmarkRequest, InstanceResponse
from identity_benchmark.evaluators import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorError,
)
from identity_benchmark.experiments import load_experiment_spec, run_experiment
from identity_benchmark.runner import run_benchmark
from identity_benchmark.contracts import load_benchmark_profile
from evaluator_support import (
    ExpectationTestEvaluator,
    SyntheticAgent,
    expectation_evaluator_factory,
    synthetic_agent_factory,
)


PROFILE = FIXTURES / "synthetic-profile.json"
MODEL_EVALUATOR_MANIFEST = FIXTURES / "model-evaluator-smoke-experiment.json"


class IdentityBenchmarkEvaluatorTests(unittest.TestCase):
    def test_enoch_adapter_exposes_the_agent_contract(self) -> None:
        self.assertTrue(hasattr(EnochAdapter, "respond"))
        self.assertTrue(hasattr(AgentAdapter, "respond"))

    def test_evaluation_result_rejects_non_finite_or_boolean_scores(self) -> None:
        with self.assertRaises(EvaluatorError):
            EvaluationResult(score=float("nan"))
        with self.assertRaises(EvaluatorError):
            EvaluationResult(score=True)

    def test_codex_evaluator_exposes_the_evaluator_contract(self) -> None:
        with TemporaryDirectory() as directory:
            evaluator = CodexEvaluator(
                evaluator_id="codex-judge-v1",
                model="judge-model",
                reasoning_effort="xhigh",
                state_home=Path(directory),
            )

        self.assertEqual(evaluator.evaluator_id, "codex-judge-v1")
        self.assertTrue(callable(evaluator.evaluate))

    def test_evaluator_failure_is_separate_from_agent_response(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        agent = _StaticAgent()

        report = run_benchmark(
            profile,
            agent,
            evaluator=_FailingEvaluator(),
        )

        self.assertEqual(report.errors, len(profile.probes))
        self.assertTrue(
            all(result.response == "agent output" for result in report.results)
        )
        self.assertTrue(
            all("evaluation failed" in result.error for result in report.results)
        )

    def test_manifest_pins_reference_codex_evaluator(self) -> None:
        spec = load_experiment_spec(MODEL_EVALUATOR_MANIFEST)

        self.assertIsNotNone(spec.evaluator)
        assert spec.evaluator is not None
        self.assertEqual(spec.evaluator.evaluator_id, "codex-sol-xhigh-v1")
        self.assertEqual(spec.evaluator.model, "gpt-5.6-sol")
        self.assertEqual(spec.evaluator.reasoning_effort, "xhigh")
        self.assertEqual(spec.evaluator.rubric_version, "pai-model-judge-v1")
        self.assertFalse(hasattr(spec.evaluator, "command"))
        self.assertFalse(hasattr(spec.evaluator, "harness"))

    def test_matrix_constructs_codex_evaluator_and_records_it(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = temporary / "experiment.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "evaluator-fixture",
                        "profile": str(PROFILE),
                        "body_root": str(ROOT),
                        "models": ["agent-model"],
                        "reasoning_efforts": ["low"],
                        "identity_modes": ["full-context"],
                        "evaluator": {
                            "id": "codex-judge-v1",
                            "model": "judge-model",
                            "reasoning_effort": "xhigh",
                            "rubric_version": "pai-model-judge-v2",
                        },
                        "timeout_seconds": 10,
                    }
                ),
                encoding="utf-8",
            )
            evaluator = _MetadataEvaluator()
            with patch(
                "identity_benchmark.experiments.CodexEvaluator",
                return_value=evaluator,
            ) as constructor:
                report = run_experiment(
                    load_experiment_spec(manifest),
                    temporary / "reports",
                    agent_factory=synthetic_agent_factory,
                )
            saved = json.loads(
                (temporary / "reports" / "experiment-report.json").read_text(
                    encoding="utf-8"
                )
            )
            state_home = constructor.call_args.kwargs["state_home"]

        self.assertEqual(constructor.call_args.kwargs["model"], "judge-model")
        self.assertEqual(
            constructor.call_args.kwargs["reasoning_effort"],
            "xhigh",
        )
        self.assertEqual(report.runs[0].report.evaluator_id, "codex-judge-v1")
        self.assertEqual(report.aggregates[0]["evaluator_id"], "codex-judge-v1")
        self.assertEqual(saved["evaluator_ids"], ["codex-judge-v1"])
        self.assertFalse(state_home.exists())

    def test_matrix_constructs_enoch_adapter_directly(self) -> None:
        spec = load_experiment_spec(
            FIXTURES / "local-smoke-experiment.json"
        )
        with TemporaryDirectory() as directory, patch(
            "identity_benchmark.experiments.EnochAdapter",
            side_effect=lambda config: SyntheticAgent(config),
        ) as constructor:
            report = run_experiment(
                spec,
                Path(directory) / "reports",
                evaluator_factory=expectation_evaluator_factory,
            )
            config = constructor.call_args.args[0]

        self.assertEqual(len(report.runs), 2)
        self.assertEqual(config.agent_root, ROOT)
        self.assertEqual(config.model, spec.models[-1])
        self.assertEqual(config.reasoning_effort, spec.reasoning_efforts[-1])
        self.assertFalse(hasattr(spec, "instance_command"))


class _StaticAgent:
    instance_id = "static-agent"

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        del request
        return InstanceResponse(response="agent output")


class _FailingEvaluator:
    evaluator_id = "failing-evaluator"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        del request
        raise EvaluatorError("judge unavailable")


class _MetadataEvaluator(ExpectationTestEvaluator):
    evaluator_id = "codex-judge-v1"


if __name__ == "__main__":
    unittest.main()
