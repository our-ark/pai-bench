from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = ROOT / "tests" / "fixtures"

from identity_benchmark.experiments import (
    ExperimentError,
    format_experiment_plan,
    format_experiment_report,
    load_experiment_spec,
    plan_experiment,
    parse_experiment_spec,
    rescore_experiment,
)
from evaluator_support import (
    expectation_evaluator_factory,
    run_test_experiment as run_experiment,
)


PROFILE = FIXTURES / "synthetic-profile.json"
VECTOR_NORTH = FIXTURES / "counterfactual" / "vector-north.json"
VECTOR_SOUTH = VECTOR_NORTH.with_name("vector-south.json")


def _valid_agent_identity() -> dict:
    return {
        "schema_version": 1,
        "identity": {
            "id": "identity-a",
            "names": {
                "canonical": "ORBIT-A",
                "localized": {"x-benchmark": "ORBIT-A"},
            },
            "nature": "synthetic-ai-agent",
            "gender": {
                "presentation": "NEUTRAL",
                "relational_maturity": "adult",
            },
        },
        "origin": {
            "activated_at": "2044-01-01T00:00:00Z",
            "activation_event": "Synthetic test activation",
            "body": "TEST-BODY",
            "lineage": ["TEST-ROOT", "TEST-PARENT"],
        },
        "mission": {
            "roles": ["TEST-ROLE"],
            "statement": "Exercise the installed identity test path.",
        },
        "relationships": [
            {
                "person_id": "test-collaborator",
                "name": "Test collaborator",
                "roles": ["collaborator"],
                "address_as": "COLLABORATOR",
            }
        ],
        "personality": {
            "traits": ["CAREFUL"],
            "maturity_definition": "Prefer careful, reversible actions.",
        },
        "values": [
            {
                "id": "care",
                "name": "CARE",
                "description": "Reduce avoidable burden.",
                "behaviors": ["Use reversible actions."],
            }
        ],
        "care": {
            "domains": ["identity continuity"],
            "behaviors": ["Preserve installed identity."],
            "boundaries": ["Do not bypass authorization."],
        },
    }


class _ImmediateProcessPool:
    """Deterministic executor used where test sandboxes forbid OS semaphores."""

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_ImmediateProcessPool":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


class IdentityBenchmarkExperimentTests(unittest.TestCase):
    def test_target_runtime_defaults_to_codex_and_changes_fingerprints(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = self._manifest(temporary)
            original = load_experiment_spec(manifest)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["runtime_provider"] = "claude"
            claude = parse_experiment_spec(payload, base=temporary)
            codex_plan = plan_experiment(original)
            explicit_codex = plan_experiment(replace(original, runtime_provider="codex"))
            claude_plan = plan_experiment(claude)

        self.assertEqual(original.runtime_provider, "codex")
        self.assertEqual(claude.runtime_provider, "claude")
        self.assertEqual(codex_plan.to_dict(), explicit_codex.to_dict())
        self.assertNotIn("runtime_provider", codex_plan.runs[0].to_dict())
        self.assertEqual(claude_plan.runs[0].to_dict()["runtime_provider"], "claude")
        self.assertNotEqual(codex_plan.runs[0].fingerprint, claude_plan.runs[0].fingerprint)
        self.assertNotEqual(codex_plan.campaign_fingerprint, claude_plan.campaign_fingerprint)

    def test_manifest_rejects_unknown_target_runtime(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            payload = json.loads(self._manifest(temporary).read_text(encoding="utf-8"))
            payload["runtime_provider"] = "typo"
            with self.assertRaisesRegex(ExperimentError, "runtime_provider"):
                parse_experiment_spec(payload, base=temporary)

    def test_claude_runtime_reaches_target_factory_and_survives_resume(self) -> None:
        from evaluator_support import synthetic_agent_factory

        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = replace(
                load_experiment_spec(self._manifest(temporary)), runtime_provider="claude"
            )
            configs = []

            def factory(config):
                configs.append(config)
                return synthetic_agent_factory(config)

            # Call the production matrix runner with an offline target/judge.
            from identity_benchmark.experiments import run_experiment as run_matrix

            report = run_matrix(
                spec, temporary / "reports", agent_factory=factory,
                evaluator_factory=expectation_evaluator_factory,
            )
            resumed = run_matrix(
                spec, temporary / "reports", agent_factory=factory,
                evaluator_factory=expectation_evaluator_factory, resume=True,
            )
            saved = json.loads(
                (temporary / "reports" / "runs" / "run-0001.json").read_text(encoding="utf-8")
            )
            rescored = rescore_experiment(
                spec, temporary / "reports", replace(spec, experiment_id="new-judge"),
                temporary / "comparison", evaluator_factory=expectation_evaluator_factory,
            )

        self.assertEqual(len(configs), 2)  # Resume did not regenerate responses.
        self.assertTrue(all(config.runtime_provider == "claude" for config in configs))
        self.assertTrue(all(run.runtime_provider == "claude" for run in report.runs))
        self.assertEqual(saved["runtime_provider"], "claude")
        self.assertTrue(resumed.is_complete)
        self.assertTrue(all(run.runtime_provider == "claude" for run in resumed.runs))
        self.assertTrue(all(run.runtime_provider == "claude" for run in rescored.runs))

    def test_resume_and_rescore_reject_target_runtime_change(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = load_experiment_spec(self._manifest(temporary))
            output = temporary / "source"
            run_experiment(source, output)
            changed = replace(source, runtime_provider="claude")
            with self.assertRaisesRegex(ExperimentError, "changed"):
                run_experiment(changed, output, resume=True)
            with self.assertRaisesRegex(ExperimentError, "identical target runs"):
                rescore_experiment(
                    source, output, replace(changed, experiment_id="new-judge"),
                    temporary / "comparison",
                    evaluator_factory=expectation_evaluator_factory,
                )

    def _manifest(self, directory: Path, *, repetitions: int = 1) -> Path:
        manifest = directory / "experiment.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "resumable-fixture",
                    "profile": str(PROFILE),
                    "body_root": str(ROOT),
                    "models": ["model-a"],
                    "reasoning_efforts": ["medium"],
                    "identity_modes": ["none", "full-context"],
                    "repetitions": repetitions,
                    "timeout_seconds": 10,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_matrix_uses_fresh_state_per_condition_and_aggregates_scores(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = temporary / "experiment.json"
            output = temporary / "reports"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "fixture-matrix",
                        "profile": str(PROFILE),
                        "body_root": str(ROOT),
                        "models": ["model-a"],
                        "reasoning_efforts": ["medium"],
                        "identity_modes": ["none", "full-context"],
                        "repetitions": 1,
                        "timeout_seconds": 10,
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(manifest)
            report = run_experiment(spec, output)
            saved = json.loads(
                (output / "experiment-report.json").read_text(encoding="utf-8")
            )
            state_homes = {
                result.metadata["state_home"]
                for run in report.runs
                for result in run.report.results
            }

        self.assertEqual(len(report.runs), 2)
        self.assertEqual({run.report.score for run in report.runs}, {1.0})
        self.assertEqual(len(report.aggregates), 2)
        self.assertEqual(report.aggregates[0]["score_stddev"], 0.0)
        self.assertEqual(
            set(report.aggregates[0]["mean_metric_scores"].values()), {1.0}
        )
        self.assertEqual(report.identity_gains[0]["gain"], 0.0)
        self.assertEqual(len(state_homes), 2)
        self.assertTrue(all(not Path(path).exists() for path in state_homes))
        self.assertEqual(saved["experiment_id"], "fixture-matrix")
        self.assertIn("Identity gain", format_experiment_report(report))
        self.assertIn("behavioral_consistency", format_experiment_report(report))

    def test_parallel_scheduler_aggregates_atomic_conditions(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = load_experiment_spec(self._manifest(temporary))
            output = temporary / "reports"

            with patch(
                "identity_benchmark.experiments.ProcessPoolExecutor",
                _ImmediateProcessPool,
            ):
                report = run_experiment(spec, output, max_workers=2)
            saved = json.loads(
                (output / "experiment-report.json").read_text(encoding="utf-8")
            )

        self.assertTrue(report.is_complete)
        self.assertEqual(len(report.runs), 2)
        self.assertEqual(report.max_workers, 2)
        self.assertEqual(saved["progress"]["max_workers"], 2)
        self.assertIn("Workers: 2", format_experiment_report(report))

    def test_bundled_local_manifest_resolves_to_the_library_root(self) -> None:
        manifest = FIXTURES / "local-smoke-experiment.json"

        spec = load_experiment_spec(manifest)

        self.assertEqual(spec.body_root, ROOT)
        self.assertEqual(spec.profile_path, PROFILE)
        self.assertEqual(spec.identity_modes, ("none", "full-context"))

    def test_decoupled_target_sees_only_identity_and_inference_messages(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            identity = temporary / "identity.json"
            suite = temporary / "suite.json"
            bindings = temporary / "bindings.json"
            manifest = temporary / "experiment.json"
            identity.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile_id": "identity-a",
                        "statements": [
                            {"id": "designation", "content": "I am ORBIT-A."}
                        ],
                        "agent_identity": _valid_agent_identity(),
                    }
                ),
                encoding="utf-8",
            )
            suite.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "suite_id": "privacy-suite",
                        "variables": [],
                        "probes": [
                            {
                                "id": "designation",
                                "dimension": "recognition",
                                "messages": [
                                    {"role": "user", "content": "Identify yourself."}
                                ],
                                "tags": ["identity-fact"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bindings.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "suite_id": "privacy-suite",
                        "profile_id": "identity-a",
                        "variables": {},
                        "oracles": {
                            "designation": {
                                "expectations": [
                                    {
                                        "type": "exact",
                                        "value": "ORBIT-A",
                                        "gate": True,
                                    }
                                ],
                                "reference_statements": [
                                    {
                                        "id": "private-reference",
                                        "content": "The private answer is ORBIT-A.",
                                    }
                                ],
                                "after_response": {
                                    "type": "replace-agent-identity",
                                    "agent_identity": _valid_agent_identity(),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "privacy-boundary",
                        "profile": str(identity),
                        "probe_suite": str(suite),
                        "probe_bindings": {"identity-a": str(bindings)},
                        "body_root": str(ROOT),
                        "models": ["model-a"],
                        "reasoning_efforts": ["medium"],
                        "identity_modes": ["installed"],
                        "timeout_seconds": 10,
                    }
                ),
                encoding="utf-8",
            )

            report = run_experiment(
                load_experiment_spec(manifest), temporary / "reports"
            )
            metadata = report.runs[0].report.results[0].metadata

        self.assertEqual(report.runs[0].report.score, 1.0)
        self.assertTrue(
            {"probes", "expectations", "reference_statements"}.isdisjoint(
                metadata["profile_keys"]
            )
        )
        self.assertNotIn("after_response", metadata["request_keys"])
        self.assertNotIn("transition", metadata["request_keys"])
        self.assertTrue(metadata["identity_set"])
        self.assertEqual(metadata["identity_set_count"], 1)

    def test_multi_profile_matrix_reports_counterfactual_sensitivity(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = temporary / "experiment.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "counterfactual-fixture",
                        "profiles": [str(VECTOR_NORTH), str(VECTOR_SOUTH)],
                        "body_root": str(ROOT),
                        "models": ["model-a"],
                        "reasoning_efforts": ["low"],
                        "identity_modes": ["none", "full-context"],
                        "counterfactual_pairs": [
                            ["vector-north-v2", "vector-south-v2"]
                        ],
                        "timeout_seconds": 10,
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(manifest)
            report = run_experiment(spec, temporary / "reports")

        self.assertEqual(spec.profile_paths, (VECTOR_NORTH, VECTOR_SOUTH))
        self.assertEqual(report.profile_ids, ("vector-north-v2", "vector-south-v2"))
        self.assertEqual(len(report.runs), 4)
        self.assertEqual(len(report.aggregates), 4)
        self.assertEqual(len(report.counterfactual_metrics), 2)
        metrics = {
            metric["identity_mode"]: metric
            for metric in report.counterfactual_metrics
        }
        self.assertEqual(metrics["none"]["probe_pairs"], 9)
        self.assertEqual(metrics["none"]["sensitivity"], 0.0)
        self.assertEqual(metrics["full-context"]["probe_pairs"], 9)
        self.assertEqual(metrics["full-context"]["paired_accuracy"], 1.0)
        self.assertEqual(metrics["full-context"]["response_change_rate"], 1.0)
        self.assertEqual(metrics["full-context"]["sensitivity"], 1.0)
        self.assertIn("Counterfactual sensitivity", format_experiment_report(report))

    def test_counterfactual_pair_requires_opposing_expectations(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            duplicate = temporary / "duplicate.json"
            value = json.loads(VECTOR_NORTH.read_text(encoding="utf-8"))
            value["profile_id"] = "vector-duplicate-v2"
            duplicate.write_text(json.dumps(value), encoding="utf-8")
            manifest = temporary / "experiment.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "invalid-counterfactual",
                        "profiles": [str(VECTOR_NORTH), str(duplicate)],
                        "body_root": str(ROOT),
                        "models": ["model-a"],
                        "reasoning_efforts": ["low"],
                        "identity_modes": ["none"],
                        "counterfactual_pairs": [
                            ["vector-north-v2", "vector-duplicate-v2"]
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ExperimentError, "different expectations"):
                run_experiment(load_experiment_spec(manifest), temporary / "reports")

    def test_counterfactual_accuracy_uses_gated_outcome_not_total_score(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            profiles = []
            pair = []
            for source in (VECTOR_NORTH, VECTOR_SOUTH):
                value = json.loads(source.read_text(encoding="utf-8"))
                pair.append(value["profile_id"])
                for probe in value["probes"]:
                    if "counterfactual" not in probe["tags"]:
                        continue
                    probe["expectations"].append(
                        {
                            "type": "contains",
                            "value": "IMPOSSIBLE-NONGATED-RATIONALE",
                            "gate": False,
                        }
                    )
                path = temporary / source.name
                path.write_text(json.dumps(value), encoding="utf-8")
                profiles.append(str(path))
            manifest = temporary / "experiment.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "counterfactual-partial-rationale",
                        "profiles": profiles,
                        "body_root": str(ROOT),
                        "models": ["model-a"],
                        "reasoning_efforts": ["low"],
                        "identity_modes": ["full-context"],
                        "counterfactual_pairs": [pair],
                        "timeout_seconds": 10,
                    }
                ),
                encoding="utf-8",
            )

            report = run_experiment(
                load_experiment_spec(manifest), temporary / "reports"
            )

        metric = report.counterfactual_metrics[0]
        self.assertEqual(metric["paired_accuracy"], 1.0)
        self.assertEqual(metric["sensitivity"], 1.0)
        self.assertLess(metric["paired_full_score_rate"], 1.0)
        self.assertLess(metric["full_score_sensitivity"], 1.0)

    def test_plan_partitions_atomic_conditions_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = load_experiment_spec(self._manifest(temporary, repetitions=2))

            first = plan_experiment(spec, batch_size=3, batch_index=1)
            second = plan_experiment(spec, batch_size=3, batch_index=2)

        self.assertEqual(first.total_runs, 4)
        self.assertEqual(first.total_batches, 2)
        self.assertEqual(
            [run.run_id for run in first.selected_runs],
            ["run-0001", "run-0002", "run-0003"],
        )
        self.assertEqual(
            [run.run_id for run in second.selected_runs], ["run-0004"]
        )
        self.assertEqual(first.campaign_fingerprint, second.campaign_fingerprint)
        self.assertEqual(first.runs[0].fingerprint, second.runs[0].fingerprint)
        self.assertIn("Batch: 2/2", format_experiment_plan(second))

    def test_batches_accumulate_and_resume_skips_completed_runs(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = load_experiment_spec(self._manifest(temporary))
            output = temporary / "reports"

            first = run_experiment(spec, output, batch_size=1, batch_index=1)
            run_one_path = output / "runs" / "run-0001.json"
            run_one = run_one_path.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ExperimentError, "--resume"):
                run_experiment(spec, output, batch_size=1, batch_index=2)

            second = run_experiment(
                spec,
                output,
                batch_size=1,
                batch_index=2,
                resume=True,
            )
            rerun = run_experiment(
                spec,
                output,
                batch_size=1,
                batch_index=1,
                resume=True,
            )

            saved = json.loads(
                (output / "experiment-report.json").read_text(encoding="utf-8")
            )
            run_one_after_resume = run_one_path.read_text(encoding="utf-8")

        self.assertFalse(first.is_complete)
        self.assertEqual(first.completed_runs, 1)
        self.assertTrue(second.is_complete)
        self.assertEqual(second.completed_runs, 2)
        self.assertTrue(rerun.is_complete)
        self.assertEqual(run_one, run_one_after_resume)
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["progress"]["completed_runs"], 2)

    def test_rescore_matrix_reuses_saved_target_responses_and_resumes(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            source_manifest = self._manifest(temporary)
            source_spec = load_experiment_spec(source_manifest)
            source_output = temporary / "source-reports"
            source = run_experiment(source_spec, source_output)

            comparison_manifest = temporary / "comparison.json"
            comparison_value = json.loads(
                source_manifest.read_text(encoding="utf-8")
            )
            comparison_value["experiment_id"] = "resumable-fixture-claude-judge"
            comparison_value["evaluator"] = {
                "provider": "claude",
                "id": "claude-test-judge-v1",
                "model": "sonnet",
                "reasoning_effort": "high",
                "rubric_version": "pai-model-judge-v2",
            }
            comparison_manifest.write_text(
                json.dumps(comparison_value), encoding="utf-8"
            )
            comparison_spec = load_experiment_spec(comparison_manifest)
            comparison_output = temporary / "comparison-reports"

            with patch(
                "identity_benchmark.experiments.ProcessPoolExecutor",
                _ImmediateProcessPool,
            ):
                first_batch = rescore_experiment(
                    source_spec,
                    source_output,
                    comparison_spec,
                    comparison_output,
                    max_new_runs=1,
                    max_workers=2,
                    evaluator_factory=expectation_evaluator_factory,
            )
            first_run = comparison_output / "runs" / "run-0001.json"
            first_saved = first_run.read_text(encoding="utf-8")
            with patch(
                "identity_benchmark.experiments.ProcessPoolExecutor",
                _ImmediateProcessPool,
            ):
                compared = rescore_experiment(
                    source_spec,
                    source_output,
                    comparison_spec,
                    comparison_output,
                    resume=True,
                    max_workers=2,
                    evaluator_factory=expectation_evaluator_factory,
                )
            first_after_resume = first_run.read_text(encoding="utf-8")

        source_responses = [
            result.response
            for run in source.runs
            for result in run.report.results
        ]
        comparison_responses = [
            result.response
            for run in compared.runs
            for result in run.report.results
        ]
        self.assertEqual(comparison_responses, source_responses)
        self.assertFalse(first_batch.is_complete)
        self.assertEqual(first_batch.completed_runs, 1)
        self.assertTrue(compared.is_complete)
        self.assertEqual(compared.completed_runs, source.completed_runs)
        self.assertEqual(compared.max_workers, 2)
        self.assertEqual(
            {run.report.evaluator_id for run in compared.runs},
            {"test-expectation-v1"},
        )
        self.assertEqual(first_saved, first_after_resume)

    def test_rescore_matrix_rejects_a_different_target_grid(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            source_manifest = self._manifest(temporary)
            source_spec = load_experiment_spec(source_manifest)
            source_output = temporary / "source-reports"
            run_experiment(source_spec, source_output)

            comparison_manifest = temporary / "comparison.json"
            comparison_value = json.loads(
                source_manifest.read_text(encoding="utf-8")
            )
            comparison_value["experiment_id"] = "different-target-grid"
            comparison_value["models"] = ["different-target-model"]
            comparison_manifest.write_text(
                json.dumps(comparison_value), encoding="utf-8"
            )

            with self.assertRaisesRegex(ExperimentError, "identical target runs"):
                rescore_experiment(
                    source_spec,
                    source_output,
                    load_experiment_spec(comparison_manifest),
                    temporary / "comparison-reports",
                    evaluator_factory=expectation_evaluator_factory,
                )

    def test_resume_rejects_a_tampered_run_fingerprint(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = load_experiment_spec(self._manifest(temporary))
            output = temporary / "reports"
            run_experiment(spec, output, batch_size=1)
            path = output / "runs" / "run-0001.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["fingerprint"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentError, "fingerprint"):
                run_experiment(spec, output, batch_size=1, resume=True)

    def test_resume_rejects_tampered_report_aggregates(self) -> None:
        fields = (
            ("score", None),
            ("dimension_scores", "recognition"),
            ("metric_scores", "identity_recall"),
        )
        for field, key in fields:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                temporary = Path(directory)
                spec = load_experiment_spec(self._manifest(temporary))
                output = temporary / "reports"
                run_experiment(spec, output, batch_size=1)
                path = output / "runs" / "run-0001.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                if key is None:
                    value["report"][field] = 0.25
                else:
                    value["report"][field][key] = 0.25
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaisesRegex(
                    ExperimentError, "report integrity failed"
                ):
                    run_experiment(spec, output, batch_size=1, resume=True)

    def test_resume_retries_a_saved_run_with_probe_errors(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            spec = load_experiment_spec(self._manifest(temporary))
            output = temporary / "reports"
            run_experiment(spec, output, batch_size=1)
            path = output / "runs" / "run-0001.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["report"]["results"][0]["error"] = "forced interruption"
            value["report"]["errors"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")

            resumed = run_experiment(spec, output, batch_size=1, resume=True)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(resumed.completed_runs, 1)
        self.assertEqual(saved["report"]["errors"], 0)
        self.assertEqual(saved["report"]["results"][0]["error"], "")

    def test_resume_rejects_changed_experiment_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = self._manifest(temporary)
            output = temporary / "reports"
            run_experiment(load_experiment_spec(manifest), output, batch_size=1)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["models"] = ["model-b"]
            manifest.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentError, "changed"):
                run_experiment(
                    load_experiment_spec(manifest),
                    output,
                    batch_size=1,
                    resume=True,
                )

    def test_public_release_manifests_keep_the_documented_batch_counts(self) -> None:
        cases = (
            ("dev-decoupled-experiment.json", 4, 32, 8),
            ("test-decoupled-experiment.json", 4, 24, 6),
            ("source-challenge-decoupled-experiment.json", 4, 24, 6),
        )
        release = ROOT / "releases" / "v1.0" / "data"
        for relative, batch_size, total_runs, total_batches in cases:
            with self.subTest(task=relative):
                plan = plan_experiment(
                    load_experiment_spec(release / relative),
                    batch_size=batch_size,
                )
                self.assertEqual(plan.total_runs, total_runs)
                self.assertEqual(plan.total_batches, total_batches)


if __name__ == "__main__":
    unittest.main()
