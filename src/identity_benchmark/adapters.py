"""Compatibility exports for the v1 target-adapter API.

New code should import :mod:`identity_benchmark.target_adapters`. Evaluators
have a separate interface in :mod:`identity_benchmark.evaluators`.
"""

from identity_benchmark.target_adapters import (
    AgentAdapter,
    CommandAgentAdapter,
    CommandInstance,
    InstanceAdapter,
    InstanceError,
    TransitionAdapter,
    TransitionAttemptAdapter,
)

__all__ = [
    "AgentAdapter",
    "CommandAgentAdapter",
    "CommandInstance",
    "InstanceAdapter",
    "InstanceError",
    "TransitionAdapter",
    "TransitionAttemptAdapter",
]
