from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from identity_benchmark.agent_identity import AgentIdentity
from identity_benchmark.contracts import (
    BenchmarkReport,
    BenchmarkProfile,
    BenchmarkRequest,
    InstanceResponse,
    JsonValue,
    StartupContext,
    TransitionAttemptRequest,
    TransitionDecision,
    TransitionRequest,
)
from identity_benchmark.evaluators import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
)
from identity_benchmark.runner import run_benchmark


class RescoreError(ValueError):
    """Raised when a saved report cannot be replayed against a profile."""


@dataclass(frozen=True)
class RecordedInstance:
    instance_id: str
    responses: dict[str, InstanceResponse]

    def set_identity(self, identity: AgentIdentity) -> None:
        # The saved responses already embody the original installed identity.
        del identity

    def set_startup_context(self, context: tuple[StartupContext, ...]) -> None:
        # The saved responses already embody the original startup context.
        del context

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        try:
            return self.responses[request.probe_id]
        except KeyError as error:
            raise RescoreError(
                f"saved report has no response for probe {request.probe_id!r}"
            ) from error

    def apply_transition(self, request: TransitionRequest) -> None:
        # Responses are already recorded after the original control-plane update.
        del request

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision:
        try:
            recorded = self.responses[request.probe_id].metadata[
                "transition_attempt"
            ]
        except KeyError as error:
            raise RescoreError(
                f"saved report lacks a transition decision for {request.probe_id!r}"
            ) from error
        if not isinstance(recorded, dict) or not isinstance(
            recorded.get("accepted"), bool
        ):
            raise RescoreError(
                f"saved transition decision for {request.probe_id!r} is invalid"
            )
        return TransitionDecision(
            accepted=recorded["accepted"],
            metadata={"rescored": True},
        )


def rescore_saved_report(
    profile: BenchmarkProfile,
    path: Path,
    *,
    evaluator: Evaluator,
    previous_report: BenchmarkReport | None = None,
):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RescoreError(f"could not load saved benchmark report: {error}") from error
    root = _mapping(value, "saved benchmark report")
    if "report" in root:
        root = _mapping(root["report"], "saved experiment run.report")
    if root.get("profile_id") != profile.profile_id:
        raise RescoreError(
            "saved report profile_id does not match the evaluation profile"
        )
    source_instance_id = root.get("instance_id")
    if not isinstance(source_instance_id, str) or not source_instance_id.strip():
        raise RescoreError("saved report instance_id must be non-empty text")
    raw_results = root.get("results")
    if not isinstance(raw_results, list):
        raise RescoreError("saved report results must be a list")

    responses: dict[str, InstanceResponse] = {}
    for index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, f"saved report results[{index}]")
        probe_id = result.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise RescoreError(f"saved report results[{index}].probe_id must be text")
        if probe_id in responses:
            raise RescoreError(f"saved report repeats probe {probe_id!r}")
        error = result.get("error", "")
        if error:
            raise RescoreError(
                f"saved report probe {probe_id!r} has no valid response: {error}"
            )
        response = result.get("response")
        if not isinstance(response, str):
            raise RescoreError(
                f"saved report probe {probe_id!r} response must be text"
            )
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RescoreError(
                f"saved report probe {probe_id!r} metadata must be an object"
            )
        replay_metadata = {
            str(key): _json_value(item, f"metadata.{key}")
            for key, item in metadata.items()
        }
        replay_metadata["rescored"] = True
        replay_metadata["source_instance_id"] = source_instance_id
        responses[probe_id] = InstanceResponse(
            response=response,
            metadata=replay_metadata,
        )

    expected_ids = {probe.id for probe in profile.probes}
    actual_ids = set(responses)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RescoreError(
            f"saved report probe set differs; missing={missing}, extra={extra}"
        )
    active_evaluator: Evaluator = evaluator
    if previous_report is not None:
        active_evaluator = _ResumeEvaluator(
            profile,
            responses,
            evaluator,
            previous_report,
        )
    return run_benchmark(
        profile,
        RecordedInstance(
            instance_id=f"{source_instance_id}:rescored",
            responses=responses,
        ),
        evaluator=active_evaluator,
    )


class _ResumeEvaluator:
    """Reuse successful stateless judgments and retry only failed probes."""

    def __init__(
        self,
        profile: BenchmarkProfile,
        responses: dict[str, InstanceResponse],
        evaluator: Evaluator,
        previous_report: BenchmarkReport,
    ) -> None:
        if previous_report.profile_id != profile.profile_id:
            raise RescoreError("previous report profile_id does not match the profile")
        if previous_report.evaluator_id != evaluator.evaluator_id:
            raise RescoreError("previous report evaluator does not match the evaluator")
        previous = {result.probe_id: result for result in previous_report.results}
        expected = {probe.id for probe in profile.probes}
        if set(previous) != expected:
            raise RescoreError("previous report probe set differs from the profile")
        reusable: dict[str, EvaluationResult] = {}
        for probe_id, result in previous.items():
            if result.response != responses[probe_id].response:
                raise RescoreError(
                    f"previous report response differs for probe {probe_id!r}"
                )
            if result.error:
                continue
            identity_score = result.component_scores.get("identity")
            if identity_score is None:
                raise RescoreError(
                    f"previous report lacks identity score for probe {probe_id!r}"
                )
            metadata = dict(result.evaluation_metadata)
            metadata["rescore_resume_reused"] = True
            reusable[probe_id] = EvaluationResult(
                score=identity_score,
                expectation_results=tuple(
                    item
                    for item in result.expectations
                    if item.expectation.aspect == "identity"
                ),
                metadata=metadata,
            )
        self._evaluator = evaluator
        self._reusable = reusable

    @property
    def evaluator_id(self) -> str:
        return self._evaluator.evaluator_id

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        previous = self._reusable.get(request.probe.id)
        if previous is not None:
            return previous
        return self._evaluator.evaluate(request)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RescoreError(f"{label} must be an object")
    return value


def _json_value(value: object, label: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
    raise RescoreError(f"{label} is not JSON-compatible")
