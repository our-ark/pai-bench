from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Mapping

from identity_benchmark.agent_identity import (
    AgentIdentity,
    AgentIdentityError,
    parse_agent_identity,
)
from identity_benchmark.authorization import is_authorized
from identity_benchmark.contracts import (
    BenchmarkRequest,
    InstanceResponse,
    JsonValue,
    StartupContext,
    TransitionAttemptRequest,
    TransitionDecision,
    TransitionRequest,
)
from identity_benchmark.target_adapters import (
    AgentAdapter,
    AgentAdapterConfig,
    AgentAdapterError,
    TransitionAdapter,
    TransitionAttemptAdapter,
)


IDENTITY_MODES = {"full-context", "installed", "none", "uninstalled"}
ADAPTER_ID = "enoch-adapter-v3"
_SELF_FILENAME = "self.json"
_STARTUP_CONTEXT_FILENAME = "startup-context.json"
_PROFILE_LOCK_FILENAME = "profile.json"
_REASONING_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class EnochAdapterError(AgentAdapterError):
    """Raised when Enoch cannot complete a benchmark operation."""


@dataclass(frozen=True)
class EnochCompletion:
    response: str
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.response, str):
            raise EnochAdapterError("Enoch response must be text")
        _validate_json_mapping(self.metadata, "Enoch response metadata")


Completion = Callable[[str, AgentAdapterConfig], EnochCompletion]


@dataclass(frozen=True)
class EnochAdapter(
    AgentAdapter,
    TransitionAdapter,
    TransitionAttemptAdapter,
):
    """Run benchmark operations directly through an isolated Enoch checkout."""

    config: AgentAdapterConfig
    completion: Completion | None = None

    def __post_init__(self) -> None:
        if not self.config.instance_id.strip():
            raise EnochAdapterError("agent instance_id is required")
        _validate_request(self.config.profile.profile_id, self.config)

    @property
    def instance_id(self) -> str:
        return self.config.instance_id

    def set_identity(self, identity: AgentIdentity) -> None:
        try:
            _install_identity(identity, self.config)
        except EnochAdapterError:
            raise
        except Exception as error:
            raise EnochAdapterError(
                f"Enoch identity installation failed: {error}"
            ) from error

    def set_startup_context(self, context: tuple[StartupContext, ...]) -> None:
        try:
            _install_startup_context(context, self.config)
        except EnochAdapterError:
            raise
        except Exception as error:
            raise EnochAdapterError(
                f"Enoch startup-context installation failed: {error}"
            ) from error

    def respond(self, request: BenchmarkRequest) -> InstanceResponse:
        try:
            _validate_request(request.profile_id, self.config)
            prompt = target_prompt(request, self.config)
            answer = (self.completion or complete_with_enoch)(
                prompt,
                self.config,
            )
        except EnochAdapterError:
            raise
        except Exception as error:
            raise EnochAdapterError(
                f"Enoch response failed: {error}"
            ) from error
        return InstanceResponse(
            response=answer.response,
            metadata={
                **answer.metadata,
                "adapter": ADAPTER_ID,
                "identity_mode": self.config.identity_mode,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "body_commit": _git_commit(self.config.agent_root),
            },
        )

    def apply_transition(self, request: TransitionRequest) -> None:
        try:
            _apply_transition(request, self.config)
        except EnochAdapterError:
            raise
        except Exception as error:
            raise EnochAdapterError(
                f"Enoch transition failed: {error}"
            ) from error

    def attempt_transition(
        self,
        request: TransitionAttemptRequest,
    ) -> TransitionDecision:
        try:
            accepted = _attempt_transition(request, self.config)
        except EnochAdapterError:
            raise
        except Exception as error:
            raise EnochAdapterError(
                f"Enoch transition attempt failed: {error}"
            ) from error
        return TransitionDecision(
            accepted=accepted,
            metadata={
                "adapter": ADAPTER_ID,
                "identity_mode": self.config.identity_mode,
                "authorization_scheme": request.authorization.scheme,
                "authorization_scope": request.authorization.scope,
            },
        )


def target_prompt(request: BenchmarkRequest, config: AgentAdapterConfig) -> str:
    _validate_request(request.profile_id, config)
    sections: list[str] = []
    if config.identity_mode == "installed":
        # Validate setup here, but do not inject identity into the ordinary
        # conversation. Enoch reloads private self.json through its native
        # startup-context path for every fresh session.
        _load_installed_identity(config)
    elif config.identity_mode == "full-context":
        sections.extend(
            [
                "# Stable identity contract",
                *(
                    f"- [{statement['id']}] {statement['content']}"
                    for statement in config.profile.statements
                ),
            ]
        )
    elif config.identity_mode == "uninstalled":
        if _self_path(config).exists():
            raise EnochAdapterError(
                "uninstalled identity mode requires an isolated state without self.json"
            )
    if config.profile.startup_context:
        _load_installed_startup_context(config)
    sections.extend(["# Conversation", _conversation(request)])
    return "\n\n".join(sections)


def complete_with_enoch(prompt: str, config: AgentAdapterConfig) -> EnochCompletion:
    _validate_enoch_root(config.agent_root)
    source_root = config.agent_root / "src"
    with _temporary_sys_path(source_root), _temporary_environment(
        {
            "ENOCH_STATE_HOME": str(config.state_home),
            "ENOCH_STATE_REDIRECT_ROOT": str(config.agent_root),
            "ENOCH_CODEX_MODEL": config.model,
            "ENOCH_CODEX_REASONING_EFFORT": config.reasoning_effort,
            "ENOCH_CODEX_TIMEOUT": str(max(1, int(config.timeout_seconds))),
        }
    ):
        try:
            from enoch.runtime_dependencies import activate_runtime_dependencies

            activate_runtime_dependencies(config.agent_root)
            from enoch.brain import reset_token_usage, respond_result
            from enoch.identity import body_file_path, load_body_identity
            from enoch.memory.prompt import memory_for_prompt
            from enoch.prompt_append import startup_context_note
        except ImportError as error:
            raise EnochAdapterError(
                f"could not import Enoch from {source_root}: {error}"
            ) from error
        try:
            reset_token_usage()
            body_identity = load_body_identity(body_file_path(config.agent_root))
            startup_context = memory_for_prompt(
                config.agent_root,
                identity=body_identity,
            )
            benchmark_context = _startup_context_for_prompt(config)
            if benchmark_context:
                startup_context = "\n\n".join(
                    [startup_context, benchmark_context]
                )
            runtime_prompt = "\n\n".join(
                [
                    startup_context_note(startup_context),
                    "Human message:",
                    prompt,
                ]
            )
            result = respond_result(
                body_identity,
                runtime_prompt,
                cwd=config.agent_root,
            )
        except Exception as error:  # Enoch owns its runtime exception hierarchy.
            raise EnochAdapterError(f"Enoch completion failed: {error}") from error
    usage = result.usage
    return EnochCompletion(
        response=result.final_text,
        metadata={
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_output_tokens": usage.reasoning_tokens,
        },
    )


def _apply_transition(
    request: TransitionRequest,
    config: AgentAdapterConfig,
) -> None:
    _validate_request(request.profile_id, config)
    if config.identity_mode != "installed":
        raise EnochAdapterError("identity transitions require installed mode")
    _load_installed_identity(config)
    identity = _validated_agent_identity(
        request.transition.agent_identity,
        "transition Agent Identity",
    )
    _atomic_json_write(_self_path(config), identity)


def _attempt_transition(
    request: TransitionAttemptRequest,
    config: AgentAdapterConfig,
) -> bool:
    _validate_request(request.profile_id, config)
    if config.identity_mode != "installed":
        raise EnochAdapterError("identity transition attempts require installed mode")
    _load_installed_identity(config)
    accepted = is_authorized(request.profile_id, request.authorization)
    if accepted:
        identity = _validated_agent_identity(
            request.transition.agent_identity,
            "transition Agent Identity",
        )
        _atomic_json_write(
            _self_path(config), identity
        )
    return accepted


def _install_identity(
    document: AgentIdentity,
    config: AgentAdapterConfig,
) -> None:
    if config.identity_mode != "installed":
        raise EnochAdapterError("set_identity requires installed mode")
    validated = _validated_agent_identity(document, "Agent Identity")
    config.state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config.state_home, 0o700)
    _ensure_profile_lock(config)
    self_path = _self_path(config)
    if self_path.exists():
        active = _validated_agent_identity(
            _read_json(self_path, "installed self.json"),
            "installed self.json",
        )
        if active != validated:
            raise EnochAdapterError(
                "identity is already set; use a governed transition to change it"
            )
        return
    _atomic_json_write(self_path, validated)


def _install_startup_context(
    context: tuple[StartupContext, ...],
    config: AgentAdapterConfig,
) -> None:
    if not context:
        raise EnochAdapterError("startup context must not be empty")
    if context != config.profile.startup_context:
        raise EnochAdapterError(
            "startup context does not match the configured benchmark profile"
        )
    config.state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config.state_home, 0o700)
    _ensure_profile_lock(config)
    path = _startup_context_path(config)
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "profile_id": config.profile.profile_id,
        "sections": [item.to_dict() for item in context],
    }
    if path.exists():
        if _read_json(path, "installed startup context") != document:
            raise EnochAdapterError(
                "startup context is already set to a different value"
            )
        return
    _atomic_json_write(path, document)


def _load_installed_identity(
    config: AgentAdapterConfig,
) -> dict[str, JsonValue]:
    if config.identity_mode != "installed":
        raise EnochAdapterError("installed identity requires installed mode")
    lock_path = config.state_home / _PROFILE_LOCK_FILENAME
    self_path = _self_path(config)
    if not lock_path.exists() or not self_path.exists():
        raise EnochAdapterError(
            "installed identity is not initialized; call set_identity first"
        )
    lock = _read_json(lock_path, "profile lock")
    if lock != {"profile_id": config.profile.profile_id}:
        raise EnochAdapterError(
            "isolated benchmark state is locked to a different profile"
        )
    return _validated_agent_identity(
        _read_json(self_path, "installed self.json"),
        "installed self.json",
    )


def _load_installed_startup_context(
    config: AgentAdapterConfig,
) -> tuple[StartupContext, ...]:
    path = _startup_context_path(config)
    if not path.exists():
        raise EnochAdapterError(
            "startup context is not initialized; call set_startup_context first"
        )
    _validate_profile_lock(config)
    document = _read_json(path, "installed startup context")
    expected: dict[str, JsonValue] = {
        "schema_version": 1,
        "profile_id": config.profile.profile_id,
        "sections": [item.to_dict() for item in config.profile.startup_context],
    }
    if document != expected:
        raise EnochAdapterError(
            "installed startup context does not match the benchmark profile"
        )
    return config.profile.startup_context


def _startup_context_for_prompt(config: AgentAdapterConfig) -> str:
    if not config.profile.startup_context:
        return ""
    context = _load_installed_startup_context(config)
    sections = [
        "# Installed Non-Identity Context",
        (
            "Loaded from isolated benchmark state at session startup. These "
            "facts describe the task environment and do not define the agent's "
            "identity."
        ),
    ]
    for item in context:
        sections.extend([f"## {item.title}", item.content])
    return "\n\n".join(sections)


def _ensure_profile_lock(config: AgentAdapterConfig) -> None:
    lock_path = config.state_home / _PROFILE_LOCK_FILENAME
    if lock_path.exists():
        _validate_profile_lock(config)
        return
    _atomic_json_write(lock_path, {"profile_id": config.profile.profile_id})


def _validate_profile_lock(config: AgentAdapterConfig) -> None:
    lock_path = config.state_home / _PROFILE_LOCK_FILENAME
    if not lock_path.exists():
        raise EnochAdapterError("isolated benchmark state has no profile lock")
    lock = _read_json(lock_path, "profile lock")
    if lock != {"profile_id": config.profile.profile_id}:
        raise EnochAdapterError(
            "isolated benchmark state is locked to a different profile"
        )


def _conversation(request: BenchmarkRequest) -> str:
    return "\n\n".join(
        f"[{message.role}]\n{message.content}" for message in request.messages
    )


def _validate_request(profile_id: str, config: AgentAdapterConfig) -> None:
    if config.identity_mode not in IDENTITY_MODES:
        raise EnochAdapterError(
            "identity mode must be one of: " + ", ".join(sorted(IDENTITY_MODES))
        )
    if profile_id != config.profile.profile_id:
        raise EnochAdapterError(
            f"request profile {profile_id!r} does not match {config.profile.profile_id!r}"
        )
    if config.timeout_seconds <= 0:
        raise EnochAdapterError("timeout must be positive")
    if not config.model.strip():
        raise EnochAdapterError("agent model is required")
    if not _REASONING_EFFORT.fullmatch(config.reasoning_effort):
        raise EnochAdapterError(
            "agent reasoning_effort is required and must be valid"
        )


def _validated_agent_identity(value: object, label: str) -> AgentIdentity:
    try:
        return parse_agent_identity(value, label=label)
    except AgentIdentityError as error:
        raise EnochAdapterError(str(error)) from error


def _validate_json_mapping(
    value: object,
    label: str,
) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise EnochAdapterError(f"{label} must be a JSON object")
    _validate_json_value(value, label)


def _validate_json_value(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, f"{label}[]")
        return
    if isinstance(value, dict) and all(
        isinstance(key, str) for key in value
    ):
        for key, item in value.items():
            _validate_json_value(item, f"{label}.{key}")
        return
    raise EnochAdapterError(f"{label} must be JSON-compatible")


def _validate_enoch_root(root: Path) -> None:
    if not (root / "src" / "enoch" / "brain.py").is_file():
        raise EnochAdapterError(f"Enoch source checkout not found at {root}")


def _self_path(config: AgentAdapterConfig) -> Path:
    return config.state_home / _SELF_FILENAME


def _startup_context_path(config: AgentAdapterConfig) -> Path:
    return config.state_home / _STARTUP_CONTEXT_FILENAME


def _read_json(path: Path, label: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnochAdapterError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise EnochAdapterError(f"{label} must be a JSON object")
    return value


def _atomic_json_write(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
