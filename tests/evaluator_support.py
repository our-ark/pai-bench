from __future__ import annotations

from pathlib import Path

from identity_benchmark.evaluators import EvaluationRequest, EvaluationResult
from identity_benchmark.experiments import (
    EvaluatorSpec,
    ExperimentSpec,
    run_experiment,
)
from identity_benchmark.runner import run_benchmark
from identity_benchmark.scoring import (
    DeterministicScorer,
    weighted_expectation_score,
)


class ExpectationTestEvaluator:
    """Test-only rule evaluator; production exposes only CodexEvaluator."""

    evaluator_id = "test-expectation-v1"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        scorer = DeterministicScorer()
        results = tuple(
            scorer.score(request.agent_response, expectation)
            for expectation in request.probe.expectations
        )
        return EvaluationResult(
            score=weighted_expectation_score(results),
            expectation_results=results,
            metadata={"evaluator_id": self.evaluator_id, "fixture": True},
        )


TEST_EVALUATOR = ExpectationTestEvaluator()


def run_test_benchmark(profile, instance):
    return run_benchmark(profile, instance, evaluator=TEST_EVALUATOR)


def expectation_evaluator_factory(
    spec: EvaluatorSpec | None,
    state_home: Path,
) -> ExpectationTestEvaluator:
    del spec, state_home
    return ExpectationTestEvaluator()


def run_test_experiment(
    spec: ExperimentSpec,
    output_dir: Path,
    **kwargs,
):
    return run_experiment(
        spec,
        output_dir,
        evaluator_factory=expectation_evaluator_factory,
        **kwargs,
    )
