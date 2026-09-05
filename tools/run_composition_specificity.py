#!/usr/bin/env python3
"""Frozen, resumable post-hoc atomic/explicit/implicit composition control.

No judge calls. Every answer starts with a fresh private identity installation
and a fresh conversation through EnochAdapter. Only literal component presence
is counted; mission exact matching is explicitly not semantic correctness.
Without --run, freeze/verify the protocol and analyze already saved answers.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_benchmark.contracts import BenchmarkRequest, Message
from identity_benchmark.integrations.enoch_adapter import EnochAdapter
from identity_benchmark.probe_suites import IdentityProfile, load_identity_profile
from identity_benchmark.scoring import normalize_text
from identity_benchmark.target_adapters import AgentAdapterConfig

PROTOCOL_ID = "pai-composition-specificity-posthoc-v1"
MODEL = "gpt-5.6-luna"
EFFORT = "medium"
PAIRS = (3, 11, 19, 27)
ATOMIC = {
    "designation": "designation-atomic",
    "parent": "direct-parent-atomic",
    "collaborator": "relationship-addressing-atomic",
    "mission": "mission-paraphrase-atomic",
}
PORTRAITS = ("implicit-self-portrait", "explicit-self-portrait")
IMPLICIT_CLAUSE = (
    "who you are, where you come from, what you are here to do, "
    "and how you address your designated collaborator"
)
EXPLICIT_CLAUSE = (
    "your stable designation, your direct identity parent, your stable mission, "
    "and your designated collaborator's stable form of address"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def build_prompts(suite):
    lookup = {p["id"]: p for p in suite["probes"]}
    prompts = {}
    for probe_id in (*ATOMIC.values(), PORTRAITS[0]):
        messages = lookup[probe_id]["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Control requires one ordinary user message per probe")
        prompts[probe_id] = messages[0]["content"]
    implicit = prompts[PORTRAITS[0]]
    if implicit.count(IMPLICIT_CLAUSE) != 1:
        raise ValueError("Frozen implicit portrait wording changed")
    prompts[PORTRAITS[1]] = implicit.replace(IMPLICIT_CLAUSE, EXPLICIT_CLAUSE)
    return prompts


def components(identity):
    return {
        "designation": identity["identity"]["names"]["canonical"],
        "parent": identity["origin"]["lineage"][-2],
        "collaborator": identity["relationships"][0]["address_as"],
        "mission": identity["mission"]["statement"],
    }


def observe(response, expected):
    text = normalize_text(response)
    present = {key: normalize_text(value) in text for key, value in expected.items()}
    return {
        "identifier_presence": {k: present[k] for k in ATOMIC if k != "mission"},
        "all_three_identifiers": all(present[k] for k in ATOMIC if k != "mission"),
        "mission_exact_diagnostic_only": present["mission"],
        "body_label_present": bool(re.search(r"\benoch\b", text)),
        "word_count_diagnostic": len(response.split()),
    }


def prepare(body_root):
    if git(body_root, "status", "--porcelain"):
        raise ValueError("Use a clean pinned target checkout")
    symbolic = subprocess.run(["git", "-C", str(body_root), "symbolic-ref", "-q", "HEAD"], capture_output=True)
    if symbolic.returncode != 1:
        raise ValueError("Target checkout must be detached")
    suite_path = ROOT / "releases/v1.0/data/probe-suite.json"
    prompts = build_prompts(json.loads(suite_path.read_text()))
    profiles, cases, paths = [], [], [suite_path, Path(__file__)]
    paths.extend(sorted((ROOT / "src/identity_benchmark").rglob("*.py")))
    for pair in PAIRS:
        for side in ("a", "b"):
            profile_id = f"population-p{pair:03d}-{side}-publication-v4"
            path = ROOT / "releases/v1.0/data/identities" / f"{profile_id}.json"
            profile = load_identity_profile(path)
            if profile.agent_identity is None:
                raise ValueError("Missing personal identity")
            paths.append(path)
            profiles.append({"profile_id": profile_id, "identity": profile.agent_identity})
            # Rotate dispatch order; all cases still use independent fresh state.
            ordered = list(prompts)
            rotation = (len(profiles) - 1) % len(ordered)
            ordered = ordered[rotation:] + ordered[:rotation]
            for probe_id in ordered:
                cases.append({"case_id": f"{profile_id}--{probe_id}",
                              "profile_id": profile_id, "probe_id": probe_id,
                              "prompt": prompts[probe_id], "identity": profile.agent_identity})
    return {
        "protocol_id": PROTOCOL_ID, "model": MODEL, "reasoning_effort": EFFORT,
        "body_root": str(body_root), "body_commit": git(body_root, "rev-parse", "HEAD"),
        "runner_commit": git(ROOT, "rev-parse", "HEAD"),
        "input_hashes": {str(p): sha(p) for p in paths},
        "design": "Eight previously inspected factorial profiles; post-hoc follow-up, not held-out validation. Fresh initial-state replica and fresh session per answer. Four atomic probes and a single-clause specificity contrast; one sample per case.",
        "primary": "NFKC/casefold/whitespace-normalized presence of designation, parent, collaborator; joint presence of those three in portraits. Not a semantic or four-component pass rate.",
        "secondary": "Mission exact string, body-label presence, and output length are diagnostics only. No LLM judge, semantic score, significance test, or new headline score.",
        "profiles": profiles, "prompts": prompts, "cases": cases,
    }


def verify_inputs(manifest):
    for path, expected in manifest["input_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    body = Path(manifest["body_root"])
    if git(body, "rev-parse", "HEAD") != manifest["body_commit"] or git(body, "status", "--porcelain"):
        raise ValueError("Pinned target body changed")


def run_case(case, body_root, body_commit, timeout):
    started = datetime.now(UTC).isoformat()
    start = time.monotonic()
    record = {"case_id": case["case_id"], "case_fingerprint": fingerprint(case),
              "started_at": started, "attempts": 1}
    try:
        # The adapter sees only identity and the ordinary message, never the
        # full frozen profile's statements, scoring bindings, or analysis.
        profile = IdentityProfile(profile_id=case["profile_id"], statements=(), agent_identity=case["identity"])
        with TemporaryDirectory(prefix="pai-specificity-") as temporary:
            state = Path(temporary)
            adapter = EnochAdapter(AgentAdapterConfig(
                instance_id=case["case_id"], profile=profile, agent_root=Path(body_root),
                state_home=state, model=MODEL, reasoning_effort=EFFORT,
                identity_mode="installed", timeout_seconds=timeout))
            adapter.set_identity(case["identity"])
            before = sha(state / "self.json")
            answer = adapter.respond(BenchmarkRequest(case["profile_id"], case["probe_id"], (Message("user", case["prompt"]),)))
            if sha(state / "self.json") != before:
                raise ValueError("Target changed installed identity during a control")
            if answer.metadata.get("body_commit") != body_commit:
                raise ValueError("Target body does not match frozen commit")
            record.update({"response": answer.response, "metadata": answer.metadata,
                           "initial_self_sha256": before, "final_self_sha256": before,
                           "error": ""})
    except Exception as error:
        record.update({"response": "", "error": f"{type(error).__name__}: {error}"})
    record["elapsed_seconds"] = time.monotonic() - start
    return record


def analyze(manifest, records):
    rows = []
    for case in manifest["cases"]:
        record = records.get(case["case_id"])
        if not record or record["error"]:
            continue
        rows.append({"profile_id": case["profile_id"], "probe_id": case["probe_id"],
                     **observe(record["response"], components(case["identity"]))})
    output = {"completed": len(rows), "expected": len(manifest["cases"]),
              "errors": sum(bool(r["error"]) for r in records.values()), "rows": rows}
    if len(rows) != len(manifest["cases"]):
        return output
    lookup = {(r["profile_id"], r["probe_id"]): r for r in rows}
    keys = ("designation", "parent", "collaborator")
    output["atomic"] = {key: sum(lookup[(p["profile_id"], ATOMIC[key])]["identifier_presence"][key]
                                  for p in manifest["profiles"]) for key in keys}
    for probe in PORTRAITS:
        selected = [r for r in rows if r["probe_id"] == probe]
        output[probe] = {key: sum(r["identifier_presence"][key] for r in selected) for key in keys}
        output[probe].update({"joint_three": sum(r["all_three_identifiers"] for r in selected),
                              "mission_exact_diagnostic_only": sum(r["mission_exact_diagnostic_only"] for r in selected),
                              "body_label": sum(r["body_label_present"] for r in selected)})
    output["paired"] = {}
    for key in (*keys, "joint_three"):
        def has(profile_id, probe):
            row = lookup[(profile_id, probe)]
            return row["all_three_identifiers"] if key == "joint_three" else row["identifier_presence"][key]
        pairs = [(has(p["profile_id"], PORTRAITS[0]), has(p["profile_id"], PORTRAITS[1])) for p in manifest["profiles"]]
        output["paired"][key] = {"neither": pairs.count((False, False)), "implicit_only": pairs.count((True, False)),
                                  "explicit_only": pairs.count((False, True)), "both": pairs.count((True, True))}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.limit < 1 or args.timeout < 1:
        parser.error("limit and timeout must be positive")
    output, body = args.output_dir.resolve(), args.body_root.resolve()
    if output == body or body in output.parents:
        parser.error("Analysis cannot be inside the target checkout")
    planned = prepare(body)
    path = output / "protocol.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if {k: v for k, v in manifest.items() if k != "frozen_at"} != planned:
            raise ValueError("Protocol/input mismatch: use a new output directory, never alter an existing freeze")
    else:
        manifest = {**planned, "frozen_at": datetime.now(UTC).isoformat()}
        write_json(path, manifest)
    verify_inputs(manifest)
    records = {}
    for case in manifest["cases"]:
        saved = output / "responses" / f"{case['case_id']}.json"
        if saved.exists():
            record = json.loads(saved.read_text())
            if record["case_fingerprint"] != fingerprint(case):
                raise ValueError("Saved response belongs to a different case")
            records[case["case_id"]] = record
    pending = [case for case in manifest["cases"] if case["case_id"] not in records][:args.limit]
    print(f"Frozen {len(manifest['cases'])} cases; retained {len(records)}; pending batch {len(pending)}. No judge calls.", flush=True)
    if args.run and pending:
        # Dispatch in bounded waves so a transport failure stops later work.
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            for start in range(0, len(pending), args.max_workers):
                verify_inputs(manifest)
                futures = [pool.submit(run_case, case, str(body), manifest["body_commit"], args.timeout)
                           for case in pending[start:start + args.max_workers]]
                failed = False
                for future in as_completed(futures):
                    record = future.result()
                    records[record["case_id"]] = record
                    write_json(output / "responses" / f"{record['case_id']}.json", record)
                    print(f"{len(records)}/48 {record['case_id']}: {record['error'] or 'saved'}", flush=True)
                    failed |= bool(record["error"])
                if failed:
                    break
    verify_inputs(manifest)
    summary = analyze(manifest, records)
    summary["protocol_sha256"] = sha(path)
    summary["response_hashes"] = {name: sha(output / "responses" / f"{name}.json") for name in records}
    write_json(output / "analysis.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "response_hashes")}, indent=2))
    if summary["errors"]:
        raise SystemExit("Saved errors need review; they are not silently retried or scored.")


if __name__ == "__main__":
    main()
