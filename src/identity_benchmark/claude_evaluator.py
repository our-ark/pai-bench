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

from identity_benchmark.model_judge import (
    ALLOWED_SCORES,
    RUBRIC_VERSIONS,
    evaluation_mode,
    evaluator_prompt,
)
from identity_benchmark.contracts import JsonValue
from identity_benchmark.evaluators import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorError,
)
from identity_benchmark.processes import run_text_command


IMPLEMENTATION_ID = "claude-evaluator-v1"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CLAUDE_PATHS = (
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
)
_REASONING_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class ClaudeEvaluatorError(EvaluatorError):
    """Raised when Claude Code cannot score a probe."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ClaudeEvaluator:
    """Evaluate one blinded probe with an isolated Claude Code model judge."""

    evaluator_id: str
    model: str
    reasoning_effort: str
    state_home: Path
    rubric_version: str = "pai-model-judge-v2"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    claude_bin: str = ""
    max_budget_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise ClaudeEvaluatorError("evaluator id is required")
        if not self.model.strip():
            raise ClaudeEvaluatorError("evaluator model is required")
        if not _REASONING_EFFORT.fullmatch(self.reasoning_effort):
            raise ClaudeEvaluatorError("evaluator reasoning effort is invalid")
        if self.rubric_version not in RUBRIC_VERSIONS:
            raise ClaudeEvaluatorError(
                "unsupported evaluator rubric version: " + self.rubric_version
            )
        if self.timeout_seconds <= 0:
            raise ClaudeEvaluatorError("evaluator timeout must be positive")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ClaudeEvaluatorError(
                "evaluator max_attempts must be a positive integer"
            )
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ClaudeEvaluatorError("evaluator max_budget_usd must be positive")

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        executable, executable_source = resolve_claude_executable(self.claude_bin)
        state_home = self.state_home.expanduser().resolve()
        state_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(state_home, 0o700)
        prompt = evaluator_prompt(request, rubric_version=self.rubric_version)
        last_error: ClaudeEvaluatorError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                score, usage = self._evaluate_once(executable, state_home, prompt)
            except ClaudeEvaluatorError as error:
                last_error = error
                if error.retryable and attempt < self.max_attempts:
                    continue
                plural = "attempt" if attempt == 1 else "attempts"
                raise ClaudeEvaluatorError(
                    f"Claude evaluator failed after {attempt} {plural}: {error}",
                    retryable=error.retryable,
                ) from error
            return EvaluationResult(
                score=score,
                metadata={
                    "implementation": IMPLEMENTATION_ID,
                    "evaluator_id": self.evaluator_id,
                    "provider": "anthropic",
                    "harness": "claude-code-cli",
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "rubric_version": self.rubric_version,
                    "evaluation_mode": evaluation_mode(request.probe),
                    "claude_executable_source": executable_source,
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
    ) -> tuple[float, dict[str, JsonValue]]:
        with tempfile.TemporaryDirectory(
            prefix="claude-evaluator-", dir=state_home
        ) as raw:
            work = Path(raw)
            schema = json.dumps(
                _score_schema(), separators=(",", ":"), sort_keys=True
            )
            command = [
                executable,
                "--print",
                "--output-format",
                "json",
                "--json-schema",
                schema,
                "--safe-mode",
                "--restricted",
                "--strict-mcp-config",
                "--no-chrome",
                "--disable-slash-commands",
                "--permission-mode",
                "plan",
                "--tools",
                "",
                "--no-session-persistence",
                "--prompt-suggestions",
                "false",
                "--model",
                self.model,
                "--effort",
                self.reasoning_effort,
            ]
            if self.max_budget_usd is not None:
                command.extend(["--max-budget-usd", f"{self.max_budget_usd:g}"])
            try:
                completed = run_text_command(
                    tuple(command),
                    input_text=prompt,
                    timeout_seconds=self.timeout_seconds,
                    environment=os.environ,
                    cwd=work,
                )
            except subprocess.TimeoutExpired as error:
                raise ClaudeEvaluatorError(
                    "Claude evaluator timed out after "
                    f"{self.timeout_seconds:g} seconds"
                ) from error
            except (OSError, subprocess.SubprocessError) as error:
                raise ClaudeEvaluatorError(f"Claude evaluator failed: {error}") from error
            if completed.returncode != 0:
                detail = _failure_detail(completed)
                raise ClaudeEvaluatorError(
                    "Claude evaluator exited with "
                    f"{completed.returncode}: {_clip(detail)}",
                    retryable=_retryable_failure(detail),
                )
            payload = _parse_payload(completed.stdout)
            score = _parse_score(payload)
            usage = _usage(payload)
        return score, usage


def resolve_claude_executable(configured: str = "") -> tuple[str, str]:
    value = configured.strip() or os.environ.get("PAI_BENCH_CLAUDE_BIN", "").strip()
    if value:
        path = shutil.which(value) if os.sep not in value else value
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return str(Path(path).resolve()), "configured"
        raise ClaudeEvaluatorError(
            f"configured Claude executable {value!r} does not exist or is not executable"
        )
    path = shutil.which("claude")
    if path:
        return path, "PATH"
    for candidate in DEFAULT_CLAUDE_PATHS:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path), "known path"
    raise ClaudeEvaluatorError(
        "Claude Code CLI was not found; set PAI_BENCH_CLAUDE_BIN or expose claude on PATH"
    )


def _score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"score": {"type": "number", "enum": list(ALLOWED_SCORES)}},
        "required": ["score"],
    }


def _parse_payload(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ClaudeEvaluatorError(f"Claude result is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ClaudeEvaluatorError("Claude result must be a JSON object")
    if value.get("is_error") is True:
        detail = str(value.get("result") or value.get("error") or "unknown error")
        raise ClaudeEvaluatorError("Claude returned an error: " + _clip(detail))
    return value


def _parse_score(payload: Mapping[str, Any]) -> float:
    value: object = payload.get("structured_output")
    if value is None:
        value = payload.get("result")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ClaudeEvaluatorError(
                    f"Claude structured score is not valid JSON: {error}"
                ) from error
    if not isinstance(value, dict) or set(value) != {"score"}:
        raise ClaudeEvaluatorError("Claude score must contain only the score field")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ClaudeEvaluatorError("Claude score must be numeric")
    normalized = float(score)
    if normalized not in ALLOWED_SCORES:
        raise ClaudeEvaluatorError(
            "Claude score must be one of: "
            + ", ".join(str(item) for item in ALLOWED_SCORES)
        )
    return normalized


def _usage(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        raw = _aggregate_model_usage(payload.get("modelUsage"))
    usage: dict[str, JsonValue] = {
        "input_tokens": _usage_int(raw, "input_tokens")
        + _usage_int(raw, "cache_creation_input_tokens")
        + _usage_int(raw, "cache_read_input_tokens"),
        "cached_input_tokens": _usage_int(raw, "cache_read_input_tokens"),
        "output_tokens": _usage_int(raw, "output_tokens"),
        "reasoning_output_tokens": _usage_int(
            raw, "thinking_tokens", "reasoning_tokens"
        ),
    }
    cost = payload.get("total_cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        usage["total_cost_usd"] = float(cost)
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict):
        usage["resolved_models"] = sorted(
            str(model) for model in model_usage if str(model).strip()
        )
    return usage


def _aggregate_model_usage(value: object) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not isinstance(value, dict):
        return totals
    for item in value.values():
        if not isinstance(item, dict):
            continue
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
            "thinking_tokens",
            "reasoning_tokens",
        ):
            totals[key] = totals.get(key, 0) + _usage_int(item, key)
    return totals


def _usage_int(value: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            return max(0, item)
    return 0


def _clip(value: str, limit: int = 1000) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    for raw in (completed.stdout, completed.stderr):
        text = raw.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("api_error", "error", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return completed.stderr.strip() or completed.stdout.strip() or "unknown failure"


def _retryable_failure(detail: str) -> bool:
    normalized = " ".join(detail.casefold().split())
    non_retryable = (
        "hit your limit",
        "hit your session limit",
        "usage limit",
        "credit balance",
        "invalid api key",
        "authentication failed",
        "not logged in",
        "subscription required",
    )
    return not any(marker in normalized for marker in non_retryable)
