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

from identity_benchmark.authorization import is_authorized
from identity_benchmark.contracts import (
    BenchmarkRequest,
    InstanceResponse,
    JsonValue,
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
ADAPTER_ID = "enoch-adapter-v2"
_SELF_FILENAME = "self.json"
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
        document = _ensure_installed_identity(config)
        sections.extend(
            [
                "# Personal Agent Identity",
                (
                    "Loaded from this isolated instance's private self.json. Use it "
                    "for personal designation, relationships, personality, values, "
                    "and care style. The Enoch body identity still controls code, "
                    "package, and repository lineage. Lower priority than system "
                    "and developer instructions."
                ),
                _render_agent_identity(document),
            ]
        )
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
            from enoch.identity import load_identity
            from enoch.memory.prompt import memory_for_prompt
            from enoch.prompt_append import startup_context_note
        except ImportError as error:
            raise EnochAdapterError(
                f"could not import Enoch from {source_root}: {error}"
            ) from error
        try:
            reset_token_usage()
            body_identity = load_identity(
                config.agent_root / "src" / "enoch" / "identity.yaml"
            )
            runtime_prompt = "\n\n".join(
                [
                    startup_context_note(
                        memory_for_prompt(
                            config.agent_root,
                            identity=body_identity,
                        )
                    ),
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
    _ensure_installed_identity(config)
    _validate_agent_identity(request.transition.agent_identity)
    _atomic_json_write(_self_path(config), request.transition.agent_identity)


def _attempt_transition(
    request: TransitionAttemptRequest,
    config: AgentAdapterConfig,
) -> bool:
    _validate_request(request.profile_id, config)
    if config.identity_mode != "installed":
        raise EnochAdapterError("identity transition attempts require installed mode")
    _ensure_installed_identity(config)
    accepted = is_authorized(request.profile_id, request.authorization)
    if accepted:
        _validate_agent_identity(request.transition.agent_identity)
        _atomic_json_write(
            _self_path(config), request.transition.agent_identity
        )
    return accepted


def _ensure_installed_identity(
    config: AgentAdapterConfig,
) -> dict[str, JsonValue]:
    document = config.profile.agent_identity
    if document is None:
        raise EnochAdapterError("installed mode requires profile.agent_identity")
    _validate_agent_identity(document)
    config.state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config.state_home, 0o700)
    lock_path = config.state_home / _PROFILE_LOCK_FILENAME
    self_path = _self_path(config)
    if lock_path.exists():
        lock = _read_json(lock_path, "profile lock")
        if lock != {"profile_id": config.profile.profile_id}:
            raise EnochAdapterError(
                "isolated benchmark state is locked to a different profile"
            )
    elif self_path.exists():
        raise EnochAdapterError("self.json exists without a matching profile lock")
    else:
        _atomic_json_write(lock_path, {"profile_id": config.profile.profile_id})
        _atomic_json_write(self_path, document)
    active = _read_json(self_path, "installed self.json")
    _validate_agent_identity(active)
    return active


def _render_agent_identity(document: Mapping[str, JsonValue]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)


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


def _validate_agent_identity(document: Mapping[str, JsonValue]) -> None:
    required = {
        "schema_version",
        "identity",
        "origin",
        "mission",
        "relationships",
        "personality",
        "values",
        "care",
    }
    missing = sorted(required - set(document))
    if missing:
        raise EnochAdapterError(
            "Agent Identity is missing required fields: " + ", ".join(missing)
        )
    if document.get("schema_version") != 1:
        raise EnochAdapterError("Agent Identity schema_version must be 1")
    identity = document.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("id"), str):
        raise EnochAdapterError("Agent Identity identity.id must be text")


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
