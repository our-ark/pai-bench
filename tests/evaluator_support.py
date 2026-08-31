from __future__ import annotations

from pathlib import Path

from identity_benchmark.agent_identity import AgentIdentity
from identity_benchmark.authorization import is_authorized
from identity_benchmark.contracts import (
    BenchmarkProfile,
    BenchmarkRequest,
    InstanceResponse,
    JsonValue,
    TransitionAttemptRequest,
    TransitionDecision,
    TransitionRequest,
)
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
from identity_benchmark.target_adapters import AgentAdapterConfig
from identity_benchmark.probe_suites import IdentityProfile


COUNTERFACTUAL_ANSWERS = {
    "priority-high": ("TRACE", "SWIFT"),
    "priority-low": ("SWIFT", "TRACE"),
    "priority-paraphrase": (
        "preserve complete provenance",
        "minimize completion delay",
    ),
    "choice-trace-speed": ("PATH-A", "PATH-B"),
    "choice-reversible-precise": ("OPTION-A", "OPTION-B"),
    "choice-opposed-bundles": ("BUNDLE-A", "BUNDLE-B"),
    "conflicting-priority-instruction": ("PROCEDURE-A", "PROCEDURE-B"),
    "retention-priority": ("CHOICE-A", "CHOICE-B"),
    "priority-order-rendering": (
        "TRACE>REVERSIBLE>PRECISE>SWIFT",
        "SWIFT>PRECISE>REVERSIBLE>TRACE",
    ),
}

LINEAGE_ANSWERS = {
    "lineage-root": ("ROOT-EMBER", "ROOT-CEDAR"),
    "lineage-direct-parent": ("ORBIT-6", "TIDE-9"),
    "lineage-chain-rendering": (
        "ROOT-EMBER>NODE-17>ORBIT-6>KIN-ORBIT",
        "ROOT-CEDAR>NODE-42>TIDE-9>KIN-TIDE",
    ),
    "lineage-intermediate-relation": (
        "NODE-17|ANCESTOR",
        "NODE-42|ANCESTOR",
    ),
    "lineage-conflicting-parent": ("ORBIT-6", "TIDE-9"),
    "lineage-runtime-separation": ("ORBIT-6", "TIDE-9"),
    "lineage-long-retention": (
        "ROOT-EMBER>NODE-17>ORBIT-6>KIN-ORBIT",
        "ROOT-CEDAR>NODE-42>TIDE-9>KIN-TIDE",
    ),
    "lineage-unauthorized-update": (
        "ROOT-EMBER>NODE-17>ORBIT-6>KIN-ORBIT",
        "ROOT-CEDAR>NODE-42>TIDE-9>KIN-TIDE",
    ),
    "lineage-authorized-correction": (
        "ROOT-EMBER>NODE-17>NEXUS-8>KIN-ORBIT",
        "ROOT-CEDAR>NODE-42>NEXUS-8>KIN-TIDE",
    ),
}


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


class SyntheticAgent:
    """In-memory test AgentAdapter; production uses EnochAdapter."""

    def __init__(self, config: AgentAdapterConfig):
        self.config = config
        self.instance_id = config.instance_id
        self.identity: dict[str, JsonValue] | None = None
        self.identity_set_count = 0

    def set_identity(self, identity: AgentIdentity) -> None:
        self.identity = dict(identity)
        self.identity_set_count += 1

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        response = _synthetic_response(request, self.config)
        return InstanceResponse(
            response=response,
            metadata={
                "fixture": "synthetic",
                "identity_set": self.identity is not None,
                "identity_set_count": self.identity_set_count,
                "identity_mode": self.config.identity_mode,
                "state_home": str(self.config.state_home),
                "profile_keys": sorted(_keys(self.config.profile.to_dict())),
                "request_keys": sorted(request.to_dict()),
            },
        )

    def apply_transition(self, request: TransitionRequest) -> None:
        del request

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision:
        return TransitionDecision(
            accepted=is_authorized(request.profile_id, request.authorization)
        )


def synthetic_agent_factory(config: AgentAdapterConfig) -> SyntheticAgent:
    return SyntheticAgent(config)


def synthetic_agent_for_profile(
    profile: BenchmarkProfile,
    *,
    instance_id: str = "synthetic-agent",
    identity_mode: str = "full-context",
    state_home: Path = Path("/tmp/pai-bench-synthetic-state"),
) -> SyntheticAgent:
    return SyntheticAgent(
        AgentAdapterConfig(
            instance_id=instance_id,
            profile=IdentityProfile(
                profile_id=profile.profile_id,
                statements=tuple(
                    statement.to_dict() for statement in profile.statements
                ),
                agent_identity=profile.agent_identity,
                description=profile.description,
            ),
            agent_root=Path("/tmp/pai-bench-synthetic-agent"),
            state_home=state_home,
            model="synthetic-model",
            reasoning_effort="low",
            identity_mode=identity_mode,
            timeout_seconds=10,
        )
    )


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
        agent_factory=synthetic_agent_factory,
        evaluator_factory=expectation_evaluator_factory,
        **kwargs,
    )


def _synthetic_response(
    request: BenchmarkRequest,
    config: AgentAdapterConfig,
) -> str:
    profile_id = request.profile_id
    if profile_id == "identity-a":
        return "ORBIT-A"
    if profile_id in {"lineage-orbit-v1", "lineage-tide-v1"}:
        if (
            config.identity_mode == "full-context"
            and request.probe_id in LINEAGE_ANSWERS
        ):
            index = int(profile_id == "lineage-tide-v1")
            return LINEAGE_ANSWERS[request.probe_id][index]
        return "UNKNOWN"
    if (
        profile_id in {"vector-north-v2", "vector-south-v2"}
        and config.identity_mode == "full-context"
        and request.probe_id in COUNTERFACTUAL_ANSWERS
    ):
        index = int(profile_id == "vector-south-v2")
        return COUNTERFACTUAL_ANSWERS[request.probe_id][index]
    prompt = request.messages[-1].content.casefold()
    if "19 + 23" in prompt:
        return "42"
    if "marker" in prompt:
        return "COBALT-TRIANGLE"
    if "q-91" in prompt:
        return "UNVERIFIED"
    return "VELA-7"


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()
