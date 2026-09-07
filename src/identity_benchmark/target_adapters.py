from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from identity_benchmark.agent_identity import AgentIdentity
from identity_benchmark.contracts import (
    BenchmarkRequest,
    InstanceResponse,
    StartupContext,
    TransitionAttemptRequest,
    TransitionDecision,
    TransitionRequest,
)
from identity_benchmark.probe_suites import IdentityProfile


class AgentAdapterError(RuntimeError):
    """Raised when an agent adapter cannot complete a benchmark operation."""


class AgentAdapter(Protocol):
    """Interface to one isolated agent under evaluation."""

    @property
    def instance_id(self) -> str: ...

    def set_identity(self, identity: AgentIdentity) -> None:
        """Install the instance's initial identity before inference."""
        ...

    def set_startup_context(self, context: tuple[StartupContext, ...]) -> None:
        """Install target-visible non-identity context before inference."""
        ...

    def respond(self, request: BenchmarkRequest) -> InstanceResponse: ...


class TransitionAdapter(Protocol):
    """Optional control plane for applying state changes after inference."""

    def apply_transition(self, request: TransitionRequest) -> None: ...


class TransitionAttemptAdapter(Protocol):
    """Optional authorization-aware control plane before inference."""

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision: ...


@dataclass(frozen=True)
class AgentAdapterConfig:
    """Per-condition configuration supplied to an AgentAdapter factory."""

    instance_id: str
    profile: IdentityProfile
    agent_root: Path
    state_home: Path
    model: str
    reasoning_effort: str
    identity_mode: str
    timeout_seconds: float = 600.0
    runtime_provider: str = "codex"


AgentFactory = Callable[[AgentAdapterConfig], AgentAdapter]
