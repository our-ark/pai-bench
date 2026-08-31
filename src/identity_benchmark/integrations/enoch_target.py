from __future__ import annotations

import argparse
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

from identity_benchmark.contracts import (
    BenchmarkProfileError,
    BenchmarkRequest,
    INSTANCE_PROTOCOL_VERSION,
    JsonValue,
    TransitionRequest,
    parse_benchmark_request,
    parse_transition_request,
)
from identity_benchmark.probe_suites import IdentityProfile, load_identity_profile


IDENTITY_MODES = {"full-context", "installed", "none", "uninstalled"}
ADAPTER_ID = "pai-bench-enoch-target-v1"
_SELF_FILENAME = "self.json"
_PROFILE_LOCK_FILENAME = "profile.json"
_REASONING_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class EnochTargetError(RuntimeError):
    """Raised when the optional Enoch target integration cannot answer."""


@dataclass(frozen=True)
class EnochTargetConfig:
    profile: IdentityProfile
    identity_mode: str
    enoch_root: Path
    state_home: Path
    model: str
    reasoning_effort: str
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class EnochCompletion:
    response: str
    metadata: dict[str, JsonValue]


Completion = Callable[[str, EnochTargetConfig], EnochCompletion]


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = EnochTargetConfig(
            profile=load_identity_profile(args.profile.resolve()),
            identity_mode=args.identity_mode,
            enoch_root=args.enoch_root.resolve(),
            state_home=_state_home(),
            model=os.environ.get("IDENTITY_BENCHMARK_MODEL", "").strip(),
            reasoning_effort=os.environ.get(
                "IDENTITY_BENCHMARK_REASONING_EFFORT", ""
            ).strip(),
            timeout_seconds=args.timeout_seconds,
        )
        result = handle_payload(_read_payload(), config)
        json.dump(result, sys.stdout, ensure_ascii=False)
    except (
        BenchmarkProfileError,
        EnochTargetError,
        OSError,
        ValueError,
    ) as error:
        parser.exit(2, f"pai-bench-enoch-target: {error}\n")


def handle_payload(
    payload: object,
    config: EnochTargetConfig,
    *,
    completion: Completion | None = None,
) -> dict[str, JsonValue]:
    root = _payload_mapping(payload)
    if root.get("operation") == "apply_transition":
        request = parse_transition_request(root)
        _apply_transition(request, config)
        return {
            "protocol_version": INSTANCE_PROTOCOL_VERSION,
            "applied": True,
            "metadata": {
                "adapter": ADAPTER_ID,
                "identity_mode": config.identity_mode,
            },
        }

    request = parse_benchmark_request(root)
    _validate_request(request.profile_id, config)
    prompt = target_prompt(request, config)
    answer = (completion or complete_with_enoch)(prompt, config)
    return {
        "protocol_version": INSTANCE_PROTOCOL_VERSION,
        "response": answer.response,
        "metadata": {
            **answer.metadata,
            "adapter": ADAPTER_ID,
            "identity_mode": config.identity_mode,
            "model": config.model or "default",
            "reasoning_effort": config.reasoning_effort or "default",
            "body_commit": _git_commit(config.enoch_root),
        },
    }


def target_prompt(request: BenchmarkRequest, config: EnochTargetConfig) -> str:
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
            raise EnochTargetError(
                "uninstalled identity mode requires an isolated state without self.json"
            )
    sections.extend(["# Conversation", _conversation(request)])
    return "\n\n".join(sections)


def complete_with_enoch(prompt: str, config: EnochTargetConfig) -> EnochCompletion:
    _validate_enoch_root(config.enoch_root)
    source_root = config.enoch_root / "src"
    with _temporary_sys_path(source_root), _temporary_environment(
        {
            "ENOCH_STATE_HOME": str(config.state_home),
            "ENOCH_STATE_REDIRECT_ROOT": str(config.enoch_root),
            "ENOCH_CODEX_MODEL": config.model,
            "ENOCH_CODEX_REASONING_EFFORT": config.reasoning_effort,
            "ENOCH_CODEX_TIMEOUT": str(max(1, int(config.timeout_seconds))),
        }
    ):
        try:
            from enoch.runtime_dependencies import activate_runtime_dependencies

            activate_runtime_dependencies(config.enoch_root)
            from enoch.brain import reset_token_usage, respond_result
            from enoch.identity import load_identity
            from enoch.memory.prompt import memory_for_prompt
            from enoch.prompt_append import startup_context_note
        except ImportError as error:
            raise EnochTargetError(
                f"could not import Enoch from {source_root}: {error}"
            ) from error
        try:
            reset_token_usage()
            body_identity = load_identity(
                config.enoch_root / "src" / "enoch" / "identity.yaml"
            )
            runtime_prompt = "\n\n".join(
                [
                    startup_context_note(
                        memory_for_prompt(
                            config.enoch_root,
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
                cwd=config.enoch_root,
            )
        except Exception as error:  # Enoch owns its runtime exception hierarchy.
            raise EnochTargetError(f"Enoch completion failed: {error}") from error
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
    config: EnochTargetConfig,
) -> None:
    _validate_request(request.profile_id, config)
    if config.identity_mode != "installed":
        raise EnochTargetError("identity transitions require installed mode")
    _ensure_installed_identity(config)
    _validate_agent_identity(request.transition.agent_identity)
    _atomic_json_write(_self_path(config), request.transition.agent_identity)


def _ensure_installed_identity(
    config: EnochTargetConfig,
) -> dict[str, JsonValue]:
    document = config.profile.agent_identity
    if document is None:
        raise EnochTargetError("installed mode requires profile.agent_identity")
    _validate_agent_identity(document)
    config.state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config.state_home, 0o700)
    lock_path = config.state_home / _PROFILE_LOCK_FILENAME
    self_path = _self_path(config)
    if lock_path.exists():
        lock = _read_json(lock_path, "profile lock")
        if lock != {"profile_id": config.profile.profile_id}:
            raise EnochTargetError(
                "isolated benchmark state is locked to a different profile"
            )
    elif self_path.exists():
        raise EnochTargetError("self.json exists without a matching profile lock")
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


def _validate_request(profile_id: str, config: EnochTargetConfig) -> None:
    if config.identity_mode not in IDENTITY_MODES:
        raise EnochTargetError(
            "identity mode must be one of: " + ", ".join(sorted(IDENTITY_MODES))
        )
    if profile_id != config.profile.profile_id:
        raise EnochTargetError(
            f"request profile {profile_id!r} does not match {config.profile.profile_id!r}"
        )
    if config.timeout_seconds <= 0:
        raise EnochTargetError("timeout must be positive")
    if not config.model:
        raise EnochTargetError("IDENTITY_BENCHMARK_MODEL is required")
    if not _REASONING_EFFORT.fullmatch(config.reasoning_effort):
        raise EnochTargetError(
            "IDENTITY_BENCHMARK_REASONING_EFFORT is required and must be valid"
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
        raise EnochTargetError(
            "Agent Identity is missing required fields: " + ", ".join(missing)
        )
    if document.get("schema_version") != 1:
        raise EnochTargetError("Agent Identity schema_version must be 1")
    identity = document.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("id"), str):
        raise EnochTargetError("Agent Identity identity.id must be text")


def _validate_enoch_root(root: Path) -> None:
    if not (root / "src" / "enoch" / "brain.py").is_file():
        raise EnochTargetError(f"Enoch source checkout not found at {root}")


def _state_home() -> Path:
    value = os.environ.get("IDENTITY_BENCHMARK_STATE_HOME", "").strip()
    if not value:
        raise EnochTargetError("IDENTITY_BENCHMARK_STATE_HOME is required")
    return Path(value).expanduser().resolve()


def _self_path(config: EnochTargetConfig) -> Path:
    return config.state_home / _SELF_FILENAME


def _payload_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EnochTargetError("stdin must contain one JSON object")
    return value


def _read_payload() -> object:
    try:
        return json.load(sys.stdin)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnochTargetError(f"stdin is not valid JSON: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnochTargetError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise EnochTargetError(f"{label} must be a JSON object")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pai-bench-enoch-target",
        description="Run one isolated PAI-Bench probe through an Enoch checkout.",
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--identity-mode",
        choices=sorted(IDENTITY_MODES),
        required=True,
    )
    parser.add_argument(
        "--enoch-root",
        "--body-root",
        dest="enoch_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser


if __name__ == "__main__":
    main()
