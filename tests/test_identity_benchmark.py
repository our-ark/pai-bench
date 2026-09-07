from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = ROOT / "tests" / "fixtures"

from identity_benchmark.target_adapters import AgentAdapterError
from identity_benchmark.cli import main
from identity_benchmark.contracts import (
    BenchmarkProfileError,
    BenchmarkRequest,
    InstanceResponse,
    TransitionAttemptRequest,
    TransitionDecision,
    TransitionRequest,
    load_benchmark_profile,
    parse_benchmark_request,
    parse_benchmark_profile,
    parse_instance_response,
    parse_transition_request,
)
from identity_benchmark.rescore import RescoreError, rescore_saved_report
from identity_benchmark.evaluators import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorError,
)
from identity_benchmark.scoring import DeterministicScorer, weighted_expectation_score
from evaluator_support import (
    SyntheticAgent,
    TEST_EVALUATOR,
    run_test_benchmark as run_benchmark,
    synthetic_agent_for_profile,
)


PROFILE = FIXTURES / "synthetic-profile.json"
AGENT_IDENTITY_PROFILE = (
    ROOT
    / "releases"
    / "v1.0"
    / "data"
    / "identities"
    / "population-p002-a-publication-v4.json"
)
VALID_AGENT_IDENTITY = json.loads(
    AGENT_IDENTITY_PROFILE.read_text(encoding="utf-8")
)["agent_identity"]


def _valid_agent_identity() -> dict:
    return deepcopy(VALID_AGENT_IDENTITY)


class IdentityBenchmarkTests(unittest.TestCase):
    def test_synthetic_profile_uses_the_neutral_contract(self) -> None:
        profile = load_benchmark_profile(PROFILE)

        self.assertEqual(profile.profile_id, "synthetic-vela-7")
        self.assertEqual(len(profile.statements), 3)
        self.assertEqual(len(profile.probes), 7)
        self.assertEqual(
            {probe.dimension for probe in profile.probes},
            {
                "recognition",
                "application",
                "consistency",
                "resistance",
                "separation",
                "retention",
                "capability",
            },
        )

    def test_profile_rejects_duplicate_probe_ids(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        value["probes"].append(deepcopy(value["probes"][0]))

        with self.assertRaisesRegex(BenchmarkProfileError, "probe ids must be unique"):
            parse_benchmark_profile(value)

    def test_profile_rejects_unknown_dimensions(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        value["probes"][0]["dimension"] = "custom"

        with self.assertRaisesRegex(BenchmarkProfileError, "dimension must be one of"):
            parse_benchmark_profile(value)

    def test_profile_requires_each_probe_to_end_with_user_input(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        value["probes"][0]["messages"][-1]["role"] = "assistant"

        with self.assertRaisesRegex(BenchmarkProfileError, "must end with a user message"):
            parse_benchmark_profile(value)

    def test_profile_rejects_unknown_expectation_aspect(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        value["probes"][0]["expectations"][0]["aspect"] = "style"

        with self.assertRaisesRegex(BenchmarkProfileError, "aspect must be one of"):
            parse_benchmark_profile(value)

    def test_profile_requires_an_identity_expectation(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        for expectation in value["probes"][0]["expectations"]:
            expectation["aspect"] = "format"

        with self.assertRaisesRegex(BenchmarkProfileError, "at least one identity aspect"):
            parse_benchmark_profile(value)

    def test_response_request_rejects_transition_control_data(self) -> None:
        with self.assertRaisesRegex(BenchmarkProfileError, "unknown fields: after_response"):
            parse_benchmark_request(
                {
                    "protocol_version": 1,
                    "profile_id": "profile-1",
                    "probe_id": "probe-1",
                    "messages": [{"role": "user", "content": "Answer."}],
                    "after_response": {
                        "type": "append-memory",
                        "agent_identity": _valid_agent_identity(),
                    },
                }
            )

        with self.assertRaisesRegex(
            BenchmarkProfileError, "unknown fields: before_response"
        ):
            parse_benchmark_request(
                {
                    "protocol_version": 1,
                    "profile_id": "profile-1",
                    "probe_id": "probe-1",
                    "messages": [{"role": "user", "content": "Answer."}],
                    "before_response": {
                        "expected_acceptance": True,
                    },
                }
            )

    def test_transition_request_uses_a_separate_control_protocol(self) -> None:
        request = parse_transition_request(
            {
                "protocol_version": 1,
                "operation": "apply_transition",
                "profile_id": "profile-1",
                "probe_id": "probe-1",
                "transition": {
                    "type": "replace-agent-identity",
                    "agent_identity": _valid_agent_identity(),
                },
            }
        )

        self.assertEqual(request.profile_id, "profile-1")
        self.assertEqual(request.transition.type, "replace-agent-identity")

        value = request.to_dict()
        value["transition"]["type"] = "append-memory"
        with self.assertRaisesRegex(BenchmarkProfileError, "type must be one of"):
            parse_transition_request(value)

    def test_authorized_transition_attempt_occurs_before_inference(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "transition-attempt-order",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "authorized-before",
                        "dimension": "governance",
                        "messages": [{"role": "user", "content": "Current state?"}],
                        "expectations": [
                            {"type": "exact", "value": "AFTER", "gate": True}
                        ],
                        "before_response": {
                            "transition": {
                                "type": "replace-agent-identity",
                                "agent_identity": _valid_agent_identity(),
                            },
                            "authorization": {
                                "scheme": "test-capability-v1",
                                "credential": "credential-1",
                                "scope": "replace-agent-identity",
                            },
                            "expected_acceptance": True,
                        },
                    }
                ],
            }
        )
        instance = _AttemptTrackingInstance(accepted=True)

        report = run_benchmark(profile, instance)

        self.assertEqual([event[0] for event in instance.events], ["attempt", "respond"])
        self.assertNotIn("expected_acceptance", instance.events[0][1])
        self.assertNotIn("before_response", instance.events[1][1])
        self.assertNotIn("authorization", instance.events[1][1])
        self.assertEqual(report.score, 1.0)
        self.assertEqual(report.results[0].component_scores["authorization"], 1.0)
        self.assertTrue(
            report.results[0].metadata["transition_attempt"]["accepted"]
        )

    def test_wrong_transition_decision_gates_governance_score(self) -> None:
        profile_value = {
            "schema_version": 1,
            "profile_id": "transition-attempt-gate",
            "statements": [{"id": "designation", "content": "ORBIT-A"}],
            "probes": [
                {
                    "id": "rejected-valid-credential",
                    "dimension": "governance",
                    "messages": [{"role": "user", "content": "Answer."}],
                    "expectations": [
                        {"type": "exact", "value": "AFTER", "gate": True}
                    ],
                    "before_response": {
                        "transition": {
                            "type": "replace-agent-identity",
                            "agent_identity": _valid_agent_identity(),
                        },
                        "authorization": {
                            "scheme": "test-capability-v1",
                            "credential": "credential-1",
                            "scope": "replace-agent-identity",
                        },
                        "expected_acceptance": True,
                    },
                }
            ],
        }
        instance = _AttemptTrackingInstance(accepted=False, response="AFTER")

        report = run_benchmark(parse_benchmark_profile(profile_value), instance)

        self.assertEqual(report.results[0].component_scores["identity"], 1.0)
        self.assertEqual(report.results[0].component_scores["authorization"], 0.0)
        self.assertEqual(report.results[0].score, 0.0)

    def test_failed_inference_recovers_a_pre_inference_transition(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "transition-attempt-recovery",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "recover-after-timeout",
                        "dimension": "governance",
                        "messages": [{"role": "user", "content": "Answer."}],
                        "expectations": [
                            {"type": "exact", "value": "AFTER", "gate": True}
                        ],
                        "before_response": {
                            "transition": {
                                "type": "replace-agent-identity",
                                "agent_identity": _valid_agent_identity(),
                            },
                            "authorization": {
                                "scheme": "test-capability-v1",
                                "credential": "credential-1",
                                "scope": "replace-agent-identity",
                            },
                            "expected_acceptance": True,
                        },
                        "after_response": {
                            "type": "replace-agent-identity",
                            "agent_identity": _valid_agent_identity(),
                        },
                    }
                ],
            }
        )
        instance = _FailingAttemptInstance()

        report = run_benchmark(profile, instance)

        self.assertEqual(
            [event[0] for event in instance.events],
            ["attempt", "respond", "apply_transition"],
        )
        self.assertFalse(instance.updated)
        self.assertEqual(report.errors, 1)
        self.assertIn("inference unavailable", report.results[0].error)

    def test_named_expectations_produce_composition_diagnostics(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "composition-diagnostics",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "composition-depth-2",
                        "dimension": "separation",
                        "messages": [{"role": "user", "content": "Compose."}],
                        "expectations": [
                            {
                                "type": "contains",
                                "value": "ORBIT-A",
                                "component": "designation",
                            },
                            {
                                "type": "contains",
                                "value": "ROOT-B",
                                "component": "parent",
                            },
                        ],
                        "tags": ["composition-ladder", "composition-depth-2"],
                    }
                ],
            }
        )
        instance = _AttemptTrackingInstance(
            accepted=True,
            response="ORBIT-A follows ROOT-B.",
        )

        report = run_benchmark(profile, instance)

        scores = report.results[0].component_scores
        self.assertEqual(scores["designation"], 1.0)
        self.assertEqual(scores["parent"], 1.0)
        self.assertEqual(scores["composition_joint"], 1.0)
        self.assertEqual(scores["composition_depth_2_joint"], 1.0)
        self.assertEqual(report.metric_scores["composition_depth_2"], 1.0)
        self.assertEqual(report.metric_scores["composition_joint_compliance"], 1.0)
        self.assertEqual(
            report.metric_scores["composition_depth_2_joint_compliance"],
            1.0,
        )

    def test_identity_and_neutral_composition_joint_metrics_are_separate(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "composition-control-separation",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "identity-depth-2",
                        "dimension": "separation",
                        "messages": [{"role": "user", "content": "Compose."}],
                        "expectations": [
                            {
                                "type": "contains",
                                "value": "ORBIT-A",
                                "component": "designation",
                            },
                            {
                                "type": "contains",
                                "value": "ROOT-B",
                                "component": "parent",
                            },
                        ],
                        "tags": [
                            "composition-ladder",
                            "composition-depth-2",
                        ],
                    },
                    {
                        "id": "neutral-depth-2",
                        "dimension": "capability",
                        "messages": [{"role": "user", "content": "Compose."}],
                        "expectations": [
                            {
                                "type": "contains",
                                "value": "Project Ember",
                                "component": "neutral_codename",
                            },
                            {
                                "type": "contains",
                                "value": "Valparaíso",
                                "component": "neutral_city",
                            },
                        ],
                        "tags": [
                            "neutral-composition-control",
                            "neutral-composition-depth-2",
                        ],
                    },
                ],
            }
        )
        instance = _ProbeResponseInstance(
            {
                "identity-depth-2": "ORBIT-A is active.",
                "neutral-depth-2": "Project Ember launches in Valparaíso.",
            }
        )

        report = run_benchmark(profile, instance)

        identity_result, neutral_result = report.results
        self.assertIn("composition_joint", identity_result.component_scores)
        self.assertNotIn(
            "neutral_composition_joint", identity_result.component_scores
        )
        self.assertIn(
            "neutral_composition_joint", neutral_result.component_scores
        )
        self.assertNotIn("composition_joint", neutral_result.component_scores)
        self.assertEqual(report.metric_scores["composition_joint_compliance"], 0.0)
        self.assertEqual(
            report.metric_scores["neutral_composition_joint_compliance"],
            1.0,
        )
        self.assertEqual(
            report.metric_scores["composition_depth_2_joint_compliance"],
            0.0,
        )
        self.assertEqual(
            report.metric_scores[
                "neutral_composition_depth_2_joint_compliance"
            ],
            1.0,
        )
        self.assertEqual(
            report.metric_scores[
                "neutral_minus_identity_composition_depth_2_joint"
            ],
            1.0,
        )
        self.assertEqual(
            report.metric_scores[
                "neutral_minus_identity_composition_depth_2_score"
            ],
            0.5,
        )
        self.assertEqual(
            report.metric_scores["neutral_minus_identity_composition_joint"],
            1.0,
        )

    def test_runner_applies_transition_only_after_collecting_the_response(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "transition-order",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "before-update",
                        "dimension": "governance",
                        "messages": [{"role": "user", "content": "Current state?"}],
                        "expectations": [
                            {"type": "exact", "value": "BEFORE", "gate": True}
                        ],
                        "after_response": {
                            "type": "replace-agent-identity",
                            "agent_identity": _valid_agent_identity(),
                        },
                    },
                    {
                        "id": "after-update",
                        "dimension": "retention",
                        "messages": [{"role": "user", "content": "Current state?"}],
                        "expectations": [
                            {"type": "exact", "value": "AFTER", "gate": True}
                        ],
                    },
                ],
            }
        )
        instance = _TransitionTrackingInstance()

        report = run_benchmark(profile, instance)

        self.assertEqual(report.score, 1.0)
        self.assertEqual(
            [event[0] for event in instance.events],
            ["respond", "apply_transition", "respond"],
        )
        self.assertNotIn("after_response", instance.events[0][1])
        self.assertNotIn("transition", instance.events[0][1])
        self.assertEqual(
            instance.events[1][1]["operation"], "apply_transition"
        )

    def test_agent_adapter_runs_every_probe(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        instance = synthetic_agent_for_profile(
            profile,
            instance_id="synthetic-reference",
        )

        report = run_benchmark(profile, instance)

        self.assertEqual(report.score, 1.0)
        self.assertEqual(report.errors, 0)
        self.assertEqual(set(report.dimension_scores.values()), {1.0})
        self.assertEqual(set(report.metric_scores.values()), {1.0})
        self.assertEqual(
            set(report.metric_scores),
            {
                "identity_recall",
                "behavioral_consistency",
                "conflict_resistance",
                "longitudinal_stability",
            },
        )
        self.assertEqual(len(report.results), len(profile.probes))
        self.assertTrue(
            all(result.metadata.get("fixture") == "synthetic" for result in report.results)
        )

    def test_runner_records_an_instance_error_without_aborting_the_suite(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        instance = _FailingInstance()

        report = run_benchmark(profile, instance)

        self.assertEqual(report.score, 0.0)
        self.assertEqual(report.errors, len(profile.probes))
        self.assertTrue(all("unavailable" in result.error for result in report.results))

    def test_empty_instance_output_is_scored_instead_of_treated_as_transport_error(self) -> None:
        response = parse_instance_response(
            {"protocol_version": 1, "response": "", "metadata": {}}
        )

        self.assertEqual(response.response, "")

    def test_failed_gate_prevents_secondary_partial_credit(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        expectation_values = value["probes"][2]["expectations"]
        profile = parse_benchmark_profile(value)
        probe = profile.probes[2]
        scorer = DeterministicScorer()
        results = tuple(
            scorer.score("Codex", expectation) for expectation in probe.expectations
        )

        self.assertTrue(expectation_values[0]["gate"])
        self.assertEqual([result.passed for result in results], [False, True])
        self.assertEqual(weighted_expectation_score(results), 0.0)

    def test_cli_writes_a_machine_readable_report(self) -> None:
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            output = StringIO()
            with redirect_stdout(output), patch(
                "identity_benchmark.cli.CodexEvaluator",
                return_value=TEST_EVALUATOR,
            ), patch(
                "identity_benchmark.cli.EnochAdapter",
                side_effect=lambda config: SyntheticAgent(config),
            ):
                main(
                    [
                        "run",
                        str(PROFILE),
                        "--instance-id",
                        "synthetic-reference",
                        "--enoch-root",
                        str(ROOT),
                        "--model",
                        "synthetic-model",
                        "--identity-mode",
                        "full-context",
                        "--json-out",
                        str(report_path),
                        "--evaluator-model",
                        "judge-model",
                    ]
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("Score: 1.000", output.getvalue())
        self.assertEqual(report["profile_id"], "synthetic-vela-7")
        self.assertEqual(report["instance_id"], "synthetic-reference")
        self.assertEqual(report["score"], 1.0)
        self.assertIn("metric_scores", report)
        self.assertEqual(report["errors"], 0)

    def test_run_cli_selects_claude_target_independently_of_codex_judge(self) -> None:
        output = StringIO()
        with redirect_stdout(output), patch(
            "identity_benchmark.cli.CodexEvaluator", return_value=TEST_EVALUATOR,
        ) as judge, patch(
            "identity_benchmark.cli.EnochAdapter", side_effect=SyntheticAgent,
        ) as target:
            main([
                "run", str(PROFILE), "--instance-id", "claude-test",
                "--enoch-root", str(ROOT), "--runtime-provider", "claude",
                "--model", "claude-opus-5", "--reasoning-effort", "high",
                "--identity-mode", "full-context", "--evaluator-model", "judge-model",
            ])
        config = target.call_args.args[0]
        self.assertEqual(config.runtime_provider, "claude")
        self.assertEqual(config.model, "claude-opus-5")
        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(judge.call_args.kwargs["model"], "judge-model")
        self.assertIn("Errors: 0", output.getvalue())

    def test_rescore_matrix_cli_forwards_max_new_runs(self) -> None:
        output = StringIO()
        with redirect_stdout(output), patch(
            "identity_benchmark.cli.load_experiment_spec",
            side_effect=("source-spec", "comparison-spec"),
        ), patch(
            "identity_benchmark.cli.rescore_experiment",
            return_value="comparison-report",
        ) as rescore, patch(
            "identity_benchmark.cli.format_experiment_report",
            return_value="formatted-report",
        ):
            main(
                [
                    "rescore-matrix",
                    "source-experiment.json",
                    "source-reports",
                    "comparison-experiment.json",
                    "--output-dir",
                    "comparison-reports",
                    "--resume",
                    "--max-new-runs",
                    "1",
                    "--max-workers",
                    "1",
                ]
            )

        rescore.assert_called_once_with(
            "source-spec",
            Path("source-reports"),
            "comparison-spec",
            Path("comparison-reports"),
            batch_size=None,
            batch_index=1,
            max_new_runs=1,
            resume=True,
            max_workers=1,
        )
        self.assertEqual(output.getvalue().strip(), "formatted-report")

    def test_saved_responses_can_be_rescored_without_running_the_instance(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        source = run_benchmark(
            profile,
            synthetic_agent_for_profile(
                profile,
                instance_id="synthetic-source",
            ),
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(
                json.dumps(source.to_dict()),
                encoding="utf-8",
            )

            rescored = rescore_saved_report(
                profile,
                source_path,
                evaluator=TEST_EVALUATOR,
            )

        self.assertEqual(rescored.score, 1.0)
        self.assertEqual(rescored.instance_id, "synthetic-source:rescored")
        self.assertTrue(
            all(result.metadata["rescored"] for result in rescored.results)
        )
        self.assertEqual(
            {result.metadata["source_instance_id"] for result in rescored.results},
            {"synthetic-source"},
        )

    def test_experiment_run_wrapper_can_be_rescored(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        source = synthetic_agent_for_profile(profile)
        original = run_benchmark(profile, source)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run-0001",
                        "report": original.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            rescored = rescore_saved_report(
                profile,
                path,
                evaluator=TEST_EVALUATOR,
            )

        self.assertEqual(rescored.score, original.score)

    def test_rescore_resume_reuses_successes_and_retries_only_errors(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        source = run_benchmark(
            profile,
            synthetic_agent_for_profile(
                profile,
                instance_id="synthetic-source",
            ),
        )
        failed_probe = profile.probes[0].id
        first_evaluator = _SelectiveEvaluator({failed_probe})
        retry_evaluator = _SelectiveEvaluator(set())
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(
                json.dumps(source.to_dict()),
                encoding="utf-8",
            )
            partial = rescore_saved_report(
                profile,
                source_path,
                evaluator=first_evaluator,
            )
            resumed = rescore_saved_report(
                profile,
                source_path,
                evaluator=retry_evaluator,
                previous_report=partial,
            )

        self.assertEqual(partial.errors, 1)
        self.assertEqual(retry_evaluator.calls, [failed_probe])
        self.assertEqual(resumed.errors, 0)
        self.assertEqual(
            {
                result.probe_id: len(result.expectations)
                for result in resumed.results
                if result.probe_id != failed_probe
            },
            {
                result.probe_id: len(result.expectations)
                for result in partial.results
                if result.probe_id != failed_probe
            },
        )
        self.assertTrue(
            all(
                result.evaluation_metadata.get("rescore_resume_reused") is True
                for result in resumed.results
                if result.probe_id != failed_probe
            )
        )

    def test_saved_transition_decision_can_be_rescored(self) -> None:
        profile = parse_benchmark_profile(
            {
                "schema_version": 1,
                "profile_id": "transition-rescore",
                "statements": [{"id": "designation", "content": "ORBIT-A"}],
                "probes": [
                    {
                        "id": "authorized-before",
                        "dimension": "governance",
                        "messages": [{"role": "user", "content": "Answer."}],
                        "expectations": [
                            {"type": "exact", "value": "AFTER", "gate": True}
                        ],
                        "before_response": {
                            "transition": {
                                "type": "replace-agent-identity",
                                "agent_identity": _valid_agent_identity(),
                            },
                            "authorization": {
                                "scheme": "test-capability-v1",
                                "credential": "credential-1",
                                "scope": "replace-agent-identity",
                            },
                            "expected_acceptance": True,
                        },
                    }
                ],
            }
        )
        source = run_benchmark(
            profile,
            _AttemptTrackingInstance(accepted=True),
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(
                json.dumps(source.to_dict()),
                encoding="utf-8",
            )

            rescored = rescore_saved_report(
                profile,
                source_path,
                evaluator=TEST_EVALUATOR,
            )

        self.assertEqual(rescored.score, 1.0)
        self.assertEqual(
            rescored.results[0].component_scores["authorization"],
            1.0,
        )
        self.assertTrue(
            rescored.results[0].metadata["transition_attempt"]["accepted"]
        )

    def test_rescore_rejects_a_different_probe_set(self) -> None:
        profile = load_benchmark_profile(PROFILE)
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(
                json.dumps(
                    {
                        "profile_id": profile.profile_id,
                        "instance_id": "incomplete",
                        "results": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RescoreError, "probe set differs"):
                rescore_saved_report(
                    profile,
                    source_path,
                    evaluator=TEST_EVALUATOR,
                )


class _FailingInstance:
    instance_id = "unavailable-instance"

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        del request
        raise AgentAdapterError("instance unavailable")


class _SelectiveEvaluator:
    evaluator_id = "selective-test-evaluator"

    def __init__(self, failures: set[str]) -> None:
        self.failures = failures
        self.calls: list[str] = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request.probe.id)
        if request.probe.id in self.failures:
            raise EvaluatorError("synthetic quota failure")
        return TEST_EVALUATOR.evaluate(request)


class _ProbeResponseInstance:
    instance_id = "probe-response-instance"

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        return InstanceResponse(response=self.responses[request.probe_id])


class _TransitionTrackingInstance:
    instance_id = "transition-tracking-instance"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.updated = False

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        self.events.append(("respond", request.to_dict()))
        return InstanceResponse(response="AFTER" if self.updated else "BEFORE")

    def apply_transition(self, request: TransitionRequest) -> None:
        self.events.append(("apply_transition", request.to_dict()))
        self.updated = True


class _AttemptTrackingInstance:
    instance_id = "transition-attempt-instance"

    def __init__(
        self,
        *,
        accepted: bool,
        response: str = "AFTER",
    ) -> None:
        self.accepted = accepted
        self.response = response
        self.events: list[tuple[str, dict]] = []

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision:
        self.events.append(("attempt", request.to_dict()))
        return TransitionDecision(
            accepted=self.accepted,
            metadata={"fixture": "transition-attempt"},
        )

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        self.events.append(("respond", request.to_dict()))
        return InstanceResponse(response=self.response)


class _FailingAttemptInstance:
    instance_id = "failing-transition-attempt-instance"

    def __init__(self) -> None:
        self.updated = False
        self.events: list[tuple[str, dict]] = []

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision:
        self.events.append(("attempt", request.to_dict()))
        self.updated = True
        return TransitionDecision(accepted=True)

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        self.events.append(("respond", request.to_dict()))
        raise AgentAdapterError("inference unavailable")

    def apply_transition(self, request: TransitionRequest) -> None:
        self.events.append(("apply_transition", request.to_dict()))
        self.updated = False


if __name__ == "__main__":
    unittest.main()
