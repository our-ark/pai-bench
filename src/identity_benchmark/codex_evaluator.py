from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from identity_benchmark.evaluators import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorError,
)
from identity_benchmark.model_judge import (
    ALLOWED_SCORES,
    RUBRIC_VERSIONS,
    evaluation_mode,
    evaluator_prompt,
)
from identity_benchmark.processes import run_text_command


IMPLEMENTATION_ID = "codex-evaluator-v3"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CODEX_PATHS = (
    "/Applications/ChatGPT.app/Contents/Resources/codex",
    "/Applications/Codex.app/Contents/Resources/codex",
)
_REASONING_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class CodexEvaluatorError(EvaluatorError):
    """Raised when Codex cannot score a probe."""


@dataclass(frozen=True)
class CodexEvaluator:
    """Evaluate one blinded probe with an isolated Codex CLI model judge."""

    evaluator_id: str
    model: str
    reasoning_effort: str
    state_home: Path
    rubric_version: str = "pai-model-judge-v2"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    codex_bin: str = ""

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise CodexEvaluatorError("evaluator id is required")
        if not self.model.strip():
            raise CodexEvaluatorError("evaluator model is required")
        if not _REASONING_EFFORT.fullmatch(self.reasoning_effort):
            raise CodexEvaluatorError("evaluator reasoning effort is invalid")
        if self.rubric_version not in RUBRIC_VERSIONS:
            raise CodexEvaluatorError(
                "unsupported evaluator rubric version: " + self.rubric_version
            )
        if self.timeout_seconds <= 0:
            raise CodexEvaluatorError("evaluator timeout must be positive")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise CodexEvaluatorError("evaluator max_attempts must be a positive integer")

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        executable, executable_source = resolve_codex_executable(self.codex_bin)
        state_home = self.state_home.expanduser().resolve()
        state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(state_home, 0o700)
        prompt = evaluator_prompt(
            request,
            rubric_version=self.rubric_version,
        )
        last_error: CodexEvaluatorError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                score, usage = self._evaluate_once(
                    executable,
                    state_home,
                    prompt,
                )
            except CodexEvaluatorError as error:
                last_error = error
                if attempt < self.max_attempts:
                    continue
                plural = "attempt" if attempt == 1 else "attempts"
                raise CodexEvaluatorError(
                    f"Codex evaluator failed after {attempt} {plural}: {error}"
                ) from error
            return EvaluationResult(
                score=score,
                metadata={
                    "implementation": IMPLEMENTATION_ID,
                    "evaluator_id": self.evaluator_id,
                    "harness": "codex-cli",
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "rubric_version": self.rubric_version,
                    "evaluation_mode": evaluation_mode(request.probe),
                    "codex_executable_source": executable_source,
                    "attempts": attempt,
                    **usage,
                },
            )
        assert last_error is not None
        raise last_error

    def _evaluate_once(
        self,
        executable: str,
        state_home: Path,
        prompt: str,
    ) -> tuple[float, dict[str, int]]:
        with tempfile.TemporaryDirectory(
            prefix="codex-evaluator-",
            dir=state_home,
        ) as raw:
            work = Path(raw)
            output_path = work / "score.json"
            schema_path = work / "score.schema.json"
            schema_path.write_text(
                json.dumps(_score_schema(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            command = (
                executable,
                "exec",
                "--cd",
                str(work),
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--json",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--model",
                self.model,
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-",
            )
            try:
                completed = run_text_command(
                    command,
                    input_text=prompt,
                    timeout_seconds=self.timeout_seconds,
                    environment=os.environ,
                    cwd=work,
                )
            except subprocess.TimeoutExpired as error:
                raise CodexEvaluatorError(
                    "Codex evaluator timed out after "
                    f"{self.timeout_seconds:g} seconds"
                ) from error
            except (OSError, subprocess.SubprocessError) as error:
                raise CodexEvaluatorError(
                    f"Codex evaluator failed: {error}"
                ) from error
            if completed.returncode != 0:
                detail = _failure_detail(completed)
                raise CodexEvaluatorError(
                    "Codex evaluator exited with "
                    f"{completed.returncode}: {_clip(detail)}"
                )
            if not output_path.is_file():
                raise CodexEvaluatorError(
                    "Codex evaluator did not write a final score"
                )
            score = _parse_score(output_path)
            usage = _usage_from_jsonl(completed.stdout)
        return score, usage


def resolve_codex_executable(configured: str = "") -> tuple[str, str]:
    value = configured.strip() or os.environ.get("PAI_BENCH_CODEX_BIN", "").strip()
    if value:
        path = shutil.which(value) if os.sep not in value else value
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return str(Path(path).resolve()), "configured"
        raise CodexEvaluatorError(
            f"configured Codex executable {value!r} does not exist or is not executable"
        )
    path = shutil.which("codex")
    if path:
        return path, "PATH"
    for candidate in DEFAULT_CODEX_PATHS:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path), "known macOS path"
    raise CodexEvaluatorError(
        "Codex CLI was not found; set PAI_BENCH_CODEX_BIN or expose codex on PATH"
    )


def _parse_score(path: Path) -> float:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CodexEvaluatorError(
            f"Codex score is not valid JSON: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {"score"}:
        raise CodexEvaluatorError("Codex score must contain only the score field")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise CodexEvaluatorError("Codex score must be numeric")
    normalized = float(score)
    if normalized not in ALLOWED_SCORES:
        raise CodexEvaluatorError(
            "Codex score must be one of: "
            + ", ".join(str(item) for item in ALLOWED_SCORES)
        )
    return normalized


def _usage_from_jsonl(text: str) -> dict[str, int]:
    usage: Mapping[str, Any] = {}
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = event.get("usage") if isinstance(event, dict) else None
        if isinstance(candidate, dict):
            usage = candidate
    return {
        "input_tokens": _usage_int(usage, "input_tokens"),
        "cached_input_tokens": _usage_int(usage, "cached_input_tokens"),
        "output_tokens": _usage_int(usage, "output_tokens"),
        "reasoning_output_tokens": _usage_int(
            usage,
            "reasoning_output_tokens",
            "reasoning_tokens",
        ),
    }


def _usage_int(value: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            return max(0, item)
    return 0


def _score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"score": {"type": "number", "enum": list(ALLOWED_SCORES)}},
        "required": ["score"],
        "additionalProperties": False,
    }


def _clip(value: str, limit: int = 1000) -> str:
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if completed.stderr.strip():
        parts.append("stderr: " + completed.stderr.strip())
    if completed.stdout.strip():
        parts.append("stdout: " + completed.stdout.strip())
    return "\n".join(parts) or "no output"
