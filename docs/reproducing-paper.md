# Reproducing the paper

PAI-Bench v1.0.0 is the first public software release. The paper's current
title is *Identity Is More Than Recall: A Benchmark for Persistent Identity
in Deployed AI Agents*. No arXiv identifier or conference acceptance is
asserted here. The manuscript is distributed separately.

## Three different version boundaries

| Material | Purpose | What is fixed |
| --- | --- | --- |
| `releases/v1.0/data/` | Frozen benchmark definition | 24 synthetic identities, 12 matched pairs, three splits, 32 shared probes |
| Git tag `v1.0.0` | Public implementation distribution | Installable package, target/evaluator adapters, tests, analysis tools |
| Release asset `pai-bench-v1.0.0-paper-evidence.zip` | Historical paper evidence | Sanitized records and source snapshots from the retained experiment export |

The `development/vnext/` suite is explicitly non-frozen and is **not** the
source of the paper's frozen headline results. Publication of old records
does not retroactively preregister them. Release time is not experiment time.

## Offline evidence inspection (no provider calls)

Download `pai-bench-v1.0.0-paper-evidence.zip` and `SHA256SUMS.txt` from the
[release](https://github.com/our-ark/pai-bench/releases/tag/v1.0.0).
Check the ZIP digest against the checksum file, extract it, and run:

```sh
cd pai-bench-v1.0.0-paper-evidence
python3 scripts/verify_artifact.py
python3 scripts/run_tests.py
```

The verifier checks per-file integrity and recorded numerical invariants:
1,536 locked-suite responses, 48 prompt-specificity control responses,
selected-primary scores, and gate/safety audit totals. Tests use local fakes.
Two historical regeneration tests are skipped because source-prototype notes
were intentionally omitted from the sanitized snapshot. No API key, network
connection, or model subscription is needed for these checks.

`data/reports/` contains responses and recorded judgments; `data/experiments/`
contains conditions; `analysis/` contains derived results and generating
scripts. The snapshot's README explains its source layout and redactions.

### Do not treat judge scores as ground truth

The selected Astra judge was chosen post hoc. Original Sol, current-rubric
Sol, and available Claude replays must remain distinguishable. The raw grades,
literal-gate sensitivity analysis, and Safety audit measure different things.
Checksums and judge agreement do not establish semantic accuracy. The
prompt-specificity follow-up is a small post-hoc study, not new held-out
validation. None of the published records supplies human calibration.

## Fresh benchmark runs

```sh
git clone --branch v1.0.0 https://github.com/our-ark/pai-bench.git
cd pai-bench
python3 -m pip install .
bin/release --check
python3 -m unittest discover -s tests
```

Copy a frozen decoupled experiment manifest to an untracked working
directory. Configure an Enoch checkout, target model, reasoning effort, and
independent evaluator using [the integration guide](integrations.md). Review
the plan and expected provider costs before launching a matrix. Current
provider endpoints and model aliases can change; identical output is not
guaranteed. Offline saved-response inspection is the stable reproduction
path for the paper's observations.

## What is intentionally not public in this release

- Real users' conversations, memory, and installed private identity files.
- Provider credentials, local account configuration, and runtime session state.
- Manuscript source/PDFs and the conference-specific submission package.

The evidence asset preserves neutral project labels from an already sanitized
export. It does not claim to contain untouched runtime logs, and no redacted
identifier has been reconstructed. `EXPORT.json` records every copied file's
hash and the packaging changes. Original frozen submission files are not
modified. The exporter can be inspected at `tools/build_paper_evidence.py`.
