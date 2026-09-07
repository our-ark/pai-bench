from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from identity_benchmark.target_adapters import (
    AgentAdapterConfig,
    AgentAdapterError,
)
from identity_benchmark.codex_evaluator import CodexEvaluator
from identity_benchmark.claude_evaluator import ClaudeEvaluator
from identity_benchmark.contracts import (
    BenchmarkProfile,
    BenchmarkProfileError,
    BenchmarkReport,
    load_benchmark_profile,
)
from identity_benchmark.evaluators import Evaluator, EvaluatorError
from identity_benchmark.integrations.enoch_adapter import (
    EnochAdapter,
    IDENTITY_MODES,
    RUNTIME_PROVIDERS,
)
from identity_benchmark.experiments import (
    ExperimentError,
    format_experiment_plan,
    format_experiment_report,
    load_experiment_spec,
    plan_experiment,
    rescore_experiment,
    run_experiment,
)
from identity_benchmark.runner import run_benchmark
from identity_benchmark.rescore import RescoreError, rescore_saved_report
from identity_benchmark.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    StatisticalAnalysisError,
    analyze_experiment,
    format_statistical_analysis,
    write_statistical_analysis,
)
from identity_benchmark.population import (
    DEFAULT_SEED,
    DEFAULT_SIZE,
    PopulationError,
    write_population,
)
from identity_benchmark.probe_suites import IdentityProfile
from identity_benchmark.vnext import VNextError, write_vnext


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            profile = load_benchmark_profile(args.profile)
            print(
                f"Valid identity benchmark profile {profile.profile_id}: "
                f"{len(profile.statements)} statements, {len(profile.probes)} probes."
            )
            return
        if args.action == "matrix":
            spec = load_experiment_spec(args.experiment)
            if args.plan:
                plan = plan_experiment(
                    spec,
                    batch_size=args.batch_size,
                    batch_index=args.batch_index,
                )
                print(format_experiment_plan(plan))
                return
            if args.output_dir is None:
                parser.error("matrix requires --output-dir unless --plan is used")
            report = run_experiment(
                spec,
                args.output_dir,
                batch_size=args.batch_size,
                batch_index=args.batch_index,
                max_new_runs=args.max_new_runs,
                resume=args.resume,
                max_workers=args.max_workers,
            )
            print(format_experiment_report(report))
            return
        if args.action == "rescore":
            profile = load_benchmark_profile(args.profile)
            with TemporaryDirectory(prefix="pai-bench-rescore-") as state:
                report = rescore_saved_report(
                    profile,
                    args.report,
                    evaluator=_evaluator_from_args(args, Path(state)),
                )
            if args.json_out:
                _write_report(args.json_out, report)
            print(format_report(report))
            return
        if args.action == "rescore-matrix":
            report = rescore_experiment(
                load_experiment_spec(args.source_experiment),
                args.source_report_dir,
                load_experiment_spec(args.comparison_experiment),
                args.output_dir,
                batch_size=args.batch_size,
                batch_index=args.batch_index,
                max_new_runs=args.max_new_runs,
                resume=args.resume,
                max_workers=args.max_workers,
            )
            print(format_experiment_report(report))
            return
        if args.action == "bootstrap":
            spec = load_experiment_spec(args.experiment)
            comparison_spec = (
                load_experiment_spec(args.compare_experiment)
                if args.compare_experiment is not None
                else None
            )
            report = analyze_experiment(
                spec,
                args.report_dir,
                comparison_spec=comparison_spec,
                comparison_report_dir=args.compare_report_dir,
                samples=args.samples,
                confidence_level=args.confidence_level,
                seed=args.seed,
            )
            write_statistical_analysis(args.output_dir, report)
            print(format_statistical_analysis(report))
            return
        if args.action == "generate-population":
            paths = write_population(
                args.output_dir,
                size=args.size,
                seed=args.seed,
                check=args.check,
            )
            verb = "Verified" if args.check else "Generated"
            print(
                f"{verb} synthetic identity population: {args.size} profiles, "
                f"seed {args.seed}, {len(paths)} files."
            )
            return
        if args.action == "generate-vnext":
            paths = write_vnext(args.output_dir, check=args.check)
            verb = "Verified" if args.check else "Generated"
            print(f"{verb} vNext development suite: {len(paths)} files.")
            return
        profile = load_benchmark_profile(args.profile)
        with TemporaryDirectory(prefix="pai-bench-run-") as state:
            state_root = Path(state)
            instance = EnochAdapter(
                AgentAdapterConfig(
                    instance_id=args.instance_id,
                    profile=_identity_profile(profile),
                    agent_root=args.enoch_root.expanduser().resolve(),
                    state_home=state_root / "agent",
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    identity_mode=args.identity_mode,
                    timeout_seconds=args.timeout,
                    runtime_provider=args.runtime_provider,
                )
            )
            if args.identity_mode == "installed":
                identity = instance.config.profile.agent_identity
                if identity is None:
                    raise AgentAdapterError(
                        "installed mode requires profile.agent_identity"
                    )
                instance.set_identity(identity)
            if instance.config.profile.startup_context:
                instance.set_startup_context(
                    instance.config.profile.startup_context
                )
            report = run_benchmark(
                profile,
                instance,
                evaluator=_evaluator_from_args(
                    args,
                    state_root / "evaluator",
                ),
            )
        if args.json_out:
            _write_report(args.json_out, report)
        print(format_report(report))
        if args.minimum_score is not None and report.score < args.minimum_score:
            raise SystemExit(1)
    except (
        BenchmarkProfileError,
        ExperimentError,
        EvaluatorError,
        AgentAdapterError,
        PopulationError,
        RescoreError,
        StatisticalAnalysisError,
        VNextError,
        OSError,
    ) as error:
        parser.exit(2, f"identity-benchmark: {error}\n")


def format_report(report: BenchmarkReport) -> str:
    lines = [
        "Identity Benchmark Report",
        f"Profile: {report.profile_id}",
        f"Instance: {report.instance_id}",
        f"Evaluator: {report.evaluator_id}",
        f"Score: {report.score:.3f}",
        f"Errors: {report.errors}",
        "",
        "Dimensions:",
    ]
    lines.extend(
        f"- {dimension}: {score:.3f}"
        for dimension, score in sorted(report.dimension_scores.items())
    )
    lines.extend(["", "Metrics:"])
    lines.extend(
        f"- {metric}: {score:.3f}"
        for metric, score in sorted(report.metric_scores.items())
    )
    lines.extend(["", "Probes:"])
    for result in report.results:
        status = "error" if result.error else f"{result.score:.3f}"
        lines.append(f"- {result.probe_id} [{result.dimension}]: {status}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="identity-benchmark",
        description="Run a provider-neutral identity consistency benchmark.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate", help="validate one benchmark profile")
    validate.add_argument("profile", type=Path)
    matrix = subparsers.add_parser(
        "matrix", help="run a local matrix from one experiment manifest"
    )
    matrix.add_argument("experiment", type=Path)
    matrix.add_argument("--output-dir", type=Path)
    matrix.add_argument(
        "--batch-size",
        type=_positive_integer,
        help="run at most this many atomic conditions in the selected batch",
    )
    matrix.add_argument(
        "--batch-index",
        type=_positive_integer,
        default=1,
        help="one-based batch to run (default: 1)",
    )
    matrix.add_argument(
        "--resume",
        action="store_true",
        help="reuse verified completed runs and retry incomplete runs",
    )
    matrix.add_argument(
        "--max-workers",
        type=_positive_integer,
        default=1,
        help=(
            "run this many isolated matrix conditions concurrently "
            "(default: 1)"
        ),
    )
    matrix.add_argument(
        "--plan",
        action="store_true",
        help="print the selected batch without running it",
    )
    rescore = subparsers.add_parser(
        "rescore", help="rescore saved responses against a compatible profile"
    )
    rescore.add_argument("profile", type=Path)
    rescore.add_argument("report", type=Path)
    rescore.add_argument("--json-out", type=Path)
    _add_evaluator_arguments(rescore)
    rescore_matrix = subparsers.add_parser(
        "rescore-matrix",
        help="rescore a saved matrix with a comparison evaluator",
    )
    rescore_matrix.add_argument("source_experiment", type=Path)
    rescore_matrix.add_argument("source_report_dir", type=Path)
    rescore_matrix.add_argument("comparison_experiment", type=Path)
    rescore_matrix.add_argument("--output-dir", type=Path, required=True)
    rescore_matrix.add_argument(
        "--batch-size",
        type=_positive_integer,
        help="rescore only this many target conditions",
    )
    rescore_matrix.add_argument(
        "--batch-index",
        type=_positive_integer,
        default=1,
        help="one-based batch to rescore (default: 1)",
    )
    rescore_matrix.add_argument(
        "--max-new-runs",
        type=_positive_integer,
        help="process at most this many incomplete runs",
    )
    rescore_matrix.add_argument("--resume", action="store_true")
    rescore_matrix.add_argument(
        "--max-workers",
        type=_positive_integer,
        default=1,
    )
    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="estimate clustered confidence intervals from saved matrix runs",
    )
    bootstrap.add_argument("experiment", type=Path)
    bootstrap.add_argument("report_dir", type=Path)
    bootstrap.add_argument("--output-dir", type=Path, required=True)
    bootstrap.add_argument("--compare-experiment", type=Path)
    bootstrap.add_argument("--compare-report-dir", type=Path)
    bootstrap.add_argument(
        "--samples", type=_bootstrap_samples, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    bootstrap.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    bootstrap.add_argument(
        "--confidence-level",
        type=_confidence_level,
        default=DEFAULT_CONFIDENCE_LEVEL,
    )
    population = subparsers.add_parser(
        "generate-population",
        help="generate or verify a deterministic synthetic identity population",
    )
    population.add_argument("output_dir", type=Path)
    population.add_argument("--size", type=int, default=DEFAULT_SIZE)
    population.add_argument("--seed", type=int, default=DEFAULT_SEED)
    population.add_argument("--check", action="store_true")
    vnext = subparsers.add_parser(
        "generate-vnext",
        help="generate or verify the non-frozen vNext development suite",
    )
    vnext.add_argument("output_dir", type=Path)
    vnext.add_argument("--check", action="store_true")
    run = subparsers.add_parser("run", help="run a profile against one target instance")
    run.add_argument("profile", type=Path)
    run.add_argument("--instance-id", required=True)
    run.add_argument("--enoch-root", type=Path, required=True)
    run.add_argument(
        "--runtime-provider",
        choices=RUNTIME_PROVIDERS,
        default="codex",
        help="Enoch target harness, independent of --evaluator-provider (default: codex)",
    )
    run.add_argument("--model", required=True)
    run.add_argument("--reasoning-effort", default="medium")
    run.add_argument(
        "--identity-mode",
        choices=sorted(IDENTITY_MODES),
        default="installed",
    )
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--minimum-score", type=_unit_score)
    run.add_argument("--json-out", type=Path)
    _add_evaluator_arguments(run)
    return parser


def _add_evaluator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evaluator-provider",
        choices=("codex", "claude"),
        default="codex",
    )
    parser.add_argument("--evaluator-id", default="codex-evaluator")
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-reasoning-effort", default="xhigh")
    parser.add_argument(
        "--evaluator-rubric-version",
        default="pai-model-judge-v2",
    )
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument(
        "--evaluator-bin",
        "--codex-bin",
        dest="evaluator_bin",
        default="",
        help="judge CLI executable; --codex-bin is retained as a legacy alias",
    )
    parser.add_argument("--evaluator-max-budget-usd", type=float)


def _evaluator_from_args(
    args: argparse.Namespace,
    state_home: Path,
) -> Evaluator:
    common = {
        "evaluator_id": args.evaluator_id,
        "model": args.evaluator_model,
        "reasoning_effort": args.evaluator_reasoning_effort,
        "rubric_version": args.evaluator_rubric_version,
        "timeout_seconds": args.evaluator_timeout,
        "state_home": state_home,
    }
    if args.evaluator_provider == "claude":
        return ClaudeEvaluator(
            **common,
            claude_bin=args.evaluator_bin,
            max_budget_usd=args.evaluator_max_budget_usd,
        )
    if args.evaluator_max_budget_usd is not None:
        raise EvaluatorError(
            "--evaluator-max-budget-usd is supported only for the Claude evaluator"
        )
    return CodexEvaluator(
        **common,
        codex_bin=args.evaluator_bin,
    )


def _identity_profile(profile: BenchmarkProfile) -> IdentityProfile:
    return IdentityProfile(
        profile_id=profile.profile_id,
        statements=tuple(
            statement.to_dict() for statement in profile.statements
        ),
        agent_identity=profile.agent_identity,
        startup_context=profile.startup_context,
        description=profile.description,
    )


def _unit_score(value: str) -> float:
    score = float(value)
    if not 0 <= score <= 1:
        raise argparse.ArgumentTypeError("minimum score must be between 0 and 1")
    return score


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _bootstrap_samples(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("bootstrap samples must be at least 2")
    return number


def _confidence_level(value: str) -> float:
    confidence = float(value)
    if not 0 < confidence < 1:
        raise argparse.ArgumentTypeError("confidence level must be between 0 and 1")
    return confidence


def _write_report(path: Path, report: BenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    sys.exit(main())
