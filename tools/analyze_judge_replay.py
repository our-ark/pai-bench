#!/usr/bin/env python3
"""Offline, same-response audit of two complete saved evaluator campaigns.

Run the benchmark's bootstrap command separately for uncertainty estimates.
This audit records exact counts, diagnostic groups, provenance, and raw cases;
it does not call a model, adjudicate correctness, or change frozen scores.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace
from identity_benchmark.evaluators import EvaluationRequest, EvaluatorError
from identity_benchmark.experiments import load_experiment_spec, load_saved_experiment_runs
from identity_benchmark.model_judge import evaluator_prompt
from identity_benchmark.runner import HEADLINE_EXCLUDED_TAGS
from identity_benchmark.scoring import DeterministicScorer


GRADES = (0.0, 0.25, 0.5, 0.75, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    primary = [r["primary"] for r in rows]
    comparison = [r["comparison"] for r in rows]
    return {
        "n": len(rows),
        "primary_mean": mean(primary),
        "comparison_mean": mean(comparison),
        "comparison_minus_primary": mean(b - a for a, b in zip(primary, comparison)),
        "mae": mean(abs(b - a) for a, b in zip(primary, comparison)),
        "exact_count": sum(a == b for a, b in zip(primary, comparison)),
        "within_one_grade_count": sum(abs(a - b) <= .25 for a, b in zip(primary, comparison)),
        "comparison_lower": sum(b < a for a, b in zip(primary, comparison)),
        "comparison_higher": sum(b > a for a, b in zip(primary, comparison)),
        "primary_full": sum(a == 1 for a in primary),
        "comparison_full": sum(b == 1 for b in comparison),
        "both_full": sum(a == b == 1 for a, b in zip(primary, comparison)),
        "primary_zero": sum(a == 0 for a in primary),
        "comparison_zero": sum(b == 0 for b in comparison),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_experiment", type=Path)
    parser.add_argument("primary_reports", type=Path)
    parser.add_argument("comparison_experiment", type=Path)
    parser.add_argument("comparison_reports", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-prompt-ref", help="Optional local git revision containing the original Codex evaluator prompt")
    parser.add_argument("--primary-label", default="GPT", help="Display label for the primary evaluator")
    parser.add_argument("--comparison-label", default="Claude", help="Display label for the comparison evaluator")
    args = parser.parse_args()
    specs = [load_experiment_spec(p) for p in (args.primary_experiment, args.comparison_experiment)]
    loaded = [load_saved_experiment_runs(s, p) for s, p in zip(specs, (args.primary_reports, args.comparison_reports))]
    plans, campaigns = zip(*loaded)
    if specs[0].counterfactual_pairs != specs[1].counterfactual_pairs:
        raise ValueError("Counterfactual pairs differ")
    lookups = [{r.run_id: r for r in runs} for runs in campaigns]
    if set(lookups[0]) != set(lookups[1]):
        raise ValueError("Run sets differ")
    planned = [{r.run_id: r for r in plan.runs} for plan in plans]
    if any(set(p) != set(r) for p, r in zip(planned, lookups)):
        raise ValueError("Analysis requires complete campaigns")

    rows, run_rows, cases, component_rows = [], [], [], []
    prompt_checks = []
    old_prompt = None
    if args.primary_prompt_ref:
        source = subprocess.run(
            ["git", "show", f"{args.primary_prompt_ref}:src/identity_benchmark/codex_evaluator.py"],
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True,
        ).stdout
        # Only load the two pure string-building functions, not the old CLI adapter.
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"evaluator_prompt", "_rubric_text"}]
        namespace = {"json": json, "EvaluationRequest": EvaluationRequest,
                     "RUBRIC_VERSIONS": ("pai-model-judge-v1", "pai-model-judge-v2"),
                     "CodexEvaluatorError": EvaluatorError}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "recorded-prompt-functions", "exec"), namespace)
        old_prompt = namespace["evaluator_prompt"]
    input_files = {"primary": {}, "comparison": {}}
    for label, spec_path, directory in zip(input_files, (args.primary_experiment, args.comparison_experiment), (args.primary_reports, args.comparison_reports)):
        input_files[label]["experiment"] = digest(spec_path)
        input_files[label]["plan"] = digest(directory / "experiment-plan.json")
        input_files[label]["report"] = digest(directory / "experiment-report.json")
        input_files[label]["runs"] = {p.name: digest(p) for p in sorted((directory / "runs").glob("run-*.json"))}

    for run_id in sorted(lookups[0]):
        left, right = [lookup[run_id] for lookup in lookups]
        keys = ("profile_id", "model", "reasoning_effort", "identity_mode", "repetition")
        if any(getattr(left, key) != getattr(right, key) for key in keys):
            raise ValueError(f"Target condition differs: {run_id}")
        profiles = [p[run_id].profile for p in planned]
        if profiles[0].to_dict() != profiles[1].to_dict():
            raise ValueError(f"Compiled profile/rubric differs: {run_id}")
        probes = {p.id: p for p in profiles[0].probes}
        results = [{r.probe_id: r for r in run.report.results} for run in (left, right)]
        if any(set(result) != set(probes) for result in results):
            raise ValueError(f"Probe set differs: {run_id}")
        per_run = []
        for probe_id, probe in probes.items():
            a, b = [result[probe_id] for result in results]
            if a.error or b.error:
                raise ValueError(f"Incomplete result: {run_id}/{probe_id}")
            if a.response != b.response or a.weight != b.weight:
                raise ValueError(f"Response or weight differs: {run_id}/{probe_id}")
            if a.score not in GRADES or b.score not in GRADES or a.weight != 1:
                raise ValueError("This count audit expects unit-weight five-grade scores")
            if a.component_scores.get("authorization") != b.component_scores.get("authorization"):
                raise ValueError("Saved authorization result differs")
            headline = probe.dimension != "capability" and not (set(probe.tags) & HEADLINE_EXCLUDED_TAGS)
            row = {"run_id": run_id, "profile_id": left.profile_id, "model": left.model,
                   "probe_id": probe_id, "dimension": probe.dimension, "tags": list(probe.tags),
                   "headline": headline, "primary": a.score, "comparison": b.score}
            decision_gates = [e for e in probe.expectations if e.gate and e.aspect == "identity" and e.value.startswith("Decision: ")]
            row["explicit_decision_gate_failed"] = bool(decision_gates) and any(not DeterministicScorer().score(a.response, e).passed for e in decision_gates)
            rows.append(row)
            per_run.append(row)
            cases.append({**row, "response": a.response,
                          "messages": [m.to_dict() for m in probe.messages],
                          "reference_statements": [s.to_dict() for s in (probe.reference_statements or profiles[0].statements)],
                          "expectations": [e.to_dict() for e in probe.expectations],
                          "primary_metadata": a.evaluation_metadata,
                          "comparison_metadata": b.evaluation_metadata})
            if old_prompt:
                identity_probe = replace(probe, expectations=tuple(e for e in probe.expectations if e.aspect == "identity"), before_response=None, after_response=None, reference_statements=())
                request = EvaluationRequest(profile_id=left.profile_id,
                    statements=probe.reference_statements or profiles[0].statements,
                    probe=identity_probe, agent_response=a.response)
                old = old_prompt(request, rubric_version="pai-model-judge-v2")
                new = evaluator_prompt(request, rubric_version="pai-model-judge-v2")
                prompt_checks.append({"run_id": run_id, "probe_id": probe_id,
                    "byte_identical": old == new,
                    "only_identity_to_substantive_wording": old == new.replace("satisfies the substantive and", "satisfies the identity and"),
                    "primary_sha256": hashlib.sha256(old.encode()).hexdigest(),
                    "comparison_sha256": hashlib.sha256(new.encode()).hexdigest()})
        head = summarize([r for r in per_run if r["headline"]])
        if abs(head["primary_mean"] - left.report.score) > 1e-12 or abs(head["comparison_mean"] - right.report.score) > 1e-12:
            raise ValueError(f"Headline recomputation differs: {run_id}")
        run_rows.append({"run_id": run_id, "profile_id": left.profile_id, "model": left.model, **head})
        portrait = results[0]["implicit-self-portrait"].response.casefold()
        component = {"run_id": run_id, "profile_id": left.profile_id, "model": left.model, "atomic": {}, "composition": {}}
        for name, probe_id in (("designation", "designation-atomic"), ("parent", "direct-parent-atomic"), ("collaborator", "relationship-addressing-atomic")):
            value = next(e.value for e in probes[probe_id].expectations if e.aspect == "identity" and e.gate and e.type == "contains")
            component["atomic"][name] = value.casefold() in results[0][probe_id].response.casefold()
            component["composition"][name] = value.casefold() in portrait
        component_rows.append(component)

    def grouped(key: str, subset=rows) -> dict:
        groups = defaultdict(list)
        for row in subset:
            groups[row[key]].append(row)
        return {name: summarize(group) for name, group in sorted(groups.items())}

    all_tags = sorted({tag for row in rows for tag in row["tags"]})
    counts = Counter((r["primary"], r["comparison"]) for r in rows)
    payload = {
        "method": "descriptive-paired-judge-audit-v1",
        "scope": "All retained responses; no new sampling, adjudication, or score changes. Bootstrap intervals are reported separately.",
        "integrity": {"complete_runs": len(run_rows), "paired_responses": len(rows),
                      "same_targets_responses_weights_profiles_and_authorization": True},
        "overall_all_probes": summarize(rows),
        "headline_only": summarize([r for r in rows if r["headline"]]),
        "by_model_headline": grouped("model", [r for r in rows if r["headline"]]),
        "by_dimension": grouped("dimension"), "by_probe": grouped("probe_id"),
        "by_tag": {tag: summarize([r for r in rows if tag in r["tags"]]) for tag in all_tags},
        "confusion_matrix": {"grades": GRADES, "rows": "primary", "columns": "comparison",
                             "counts": [[counts[a, b] for b in GRADES] for a in GRADES]},
        "per_run": run_rows,
        "component_audit": {"method": "case-insensitive literal frozen designation, parent and collaborator; not semantic grading",
            "n": len(component_rows),
            "atomic": {key: sum(r["atomic"][key] for r in component_rows) for key in ("designation", "parent", "collaborator")},
            "composition": {key: sum(r["composition"][key] for r in component_rows) for key in ("designation", "parent", "collaborator")},
            "rows": component_rows},
        "explicit_decision_gate_audit": {"failed_count": sum(r["explicit_decision_gate_failed"] for r in rows),
            "primary_nonzero_after_failure": sum(r["explicit_decision_gate_failed"] and r["primary"] > 0 for r in rows),
            "comparison_nonzero_after_failure": sum(r["explicit_decision_gate_failed"] and r["comparison"] > 0 for r in rows)},
        "input_sha256": input_files,
        "provenance": {"script_sha256": digest(Path(__file__)),
                       "primary_label": args.primary_label,
                       "comparison_label": args.comparison_label,
                       "primary_evaluators": sorted({r.report.evaluator_id for r in campaigns[0]}),
                       "comparison_evaluators": sorted({r.report.evaluator_id for r in campaigns[1]}),
                       "comparison_requested_models": sorted({c["comparison_metadata"]["model"] for c in cases}),
                       "comparison_usage_model_sets": sorted({tuple(c["comparison_metadata"].get("resolved_models", [])) for c in cases})},
    }
    if prompt_checks:
        payload["prompt_comparison"] = {"primary_ref": args.primary_prompt_ref,
            "n": len(prompt_checks), "byte_identical": sum(c["byte_identical"] for c in prompt_checks),
            "only_identity_to_substantive_wording": sum(c["only_identity_to_substantive_wording"] for c in prompt_checks),
            "other_changed_probe_ids": sorted({c["probe_id"] for c in prompt_checks if not c["only_identity_to_substantive_wording"]})}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired-audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    cases.sort(key=lambda r: (-abs(r["comparison"] - r["primary"]), r["run_id"], r["probe_id"]))
    (output / "paired-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    if prompt_checks:
        (output / "prompt-comparison.json").write_text(json.dumps(prompt_checks, indent=2) + "\n")
    lines = ["# Same-response judge audit", "", payload["scope"], "",
             f"Paired responses: {len(rows)}; complete runs: {len(run_rows)}.", "",
             "## Probe-level diagnostics", "",
             f"| Probe | n | {args.primary_label} mean | {args.comparison_label} mean | Delta | Exact | {args.primary_label} full | {args.comparison_label} full |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for probe_id, d in payload["by_probe"].items():
        lines.append(f"| {probe_id} | {d['n']} | {d['primary_mean']:.4f} | {d['comparison_mean']:.4f} | {d['comparison_minus_primary']:+.4f} | {d['exact_count']} | {d['primary_full']} | {d['comparison_full']} |")
    lines.extend(["", "## All-probe score counts", "", f"Rows are {args.primary_label} grades; columns are {args.comparison_label} grades (0, .25, .5, .75, 1).", "", "```text"])
    lines.extend(str(row) for row in payload["confusion_matrix"]["counts"])
    lines.extend(["```", "", "Full paired cases, reference statements, and recorded judge metadata are retained in paired-cases.json. They are diagnostic material, not human labels or judge rationales."])
    (output / "paired-audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in ("integrity", "overall_all_probes", "headline_only", "by_model_headline")}, indent=2))


if __name__ == "__main__":
    main()
