from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

from identity_benchmark.contracts import (
    ExpectationResult,
    IdentityStatement,
    JsonValue,
    Probe,
)


class EvaluatorError(RuntimeError):
    """Raised when an evaluator cannot return a valid probe judgment."""


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    expectation_results: tuple[ExpectationResult, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise EvaluatorError("evaluation score must be between 0 and 1")
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise EvaluatorError("evaluation metadata keys must be text")
            _json_value(value, f"metadata.{key}")


@dataclass(frozen=True)
class EvaluationRequest:
    """Blinded evaluator input; excludes profile description and private provenance."""

    profile_id: str
    statements: tuple[IdentityStatement, ...]
    probe: Probe
    agent_response: str


class Evaluator(Protocol):
    """Evaluate one complete probe response using any scoring harness."""

    @property
    def evaluator_id(self) -> str: ...

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


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
    raise EvaluatorError(f"{label} is not JSON-compatible")
