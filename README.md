# PAI-Bench

PAI-Bench is a provider-neutral benchmark for persistent identity in deployed
AI agents. The current public release is **v1.0.0**.

PAI-Bench v1.0 contains 24 synthetic identities in 12 matched counterfactual
pairs, divided into development, frozen factorial-test, and frozen
source-challenge splits. Every identity uses the same 32 question templates.

The repository is self-contained: it includes the installable Python package,
tests, schemas, documentation, release generator, and frozen public data. The
target contract is agent-framework and model-provider neutral. Evaluation is
defined by a small `Evaluator` interface; the bundled implementation is
`CodexEvaluator`.

## Install

```bash
python3 -m pip install .
```

The package exposes the `identity-benchmark` and
`identity-benchmark-replay` commands. It also includes directly importable
`EnochAdapter` and `CodexEvaluator` implementations. The checkout-local
launchers under `bin/` select Python 3.11 or newer without requiring an
installation.

## Release layout

The frozen release is stored under [`releases/v1.0/data/`](releases/v1.0/data/):

- `probe-suite.json`: the shared question templates;
- `identities/`: portable identity contracts without benchmark questions;
- `bindings/`: evaluator-private variables, transitions, and scoring oracles;
- `*-decoupled-experiment.json`: frozen reference split manifests;
- `protocol.json`: split, freeze, and no-test-tuning policy; and
- `profiles/`: immutable compiled snapshots retained to reproduce the release
  compilation byte for byte.

The `identity-publication-v4.3` generator value and `-publication-v4` profile
suffixes are provenance identifiers from development. They are not separate
public benchmark versions. The public release version is recorded in
[`VERSION`](VERSION).

The non-frozen [`development/vnext/`](development/vnext/) suite is a separate
construct-validity workspace. It adds atomic identity baselines, a balanced
composition-depth ladder with matched neutral controls, assisted/unassisted
adversarial probes, authorization metadata crossed with conversational role,
and semantic-equivalent decision prompts. Its scores are not directly
comparable with the frozen v1.0 headline score.

## Verify the release

```bash
bin/release --check
```

Regenerate or verify the development suite independently:

```bash
bin/identity-benchmark generate-vnext development/vnext
bin/identity-benchmark generate-vnext development/vnext --check
```

## Run a split

The frozen manifests pin the target checkout in `body_root` and the Codex
evaluator configuration. Copy the selected decoupled manifest to an untracked
working directory and point `body_root` to Enoch. Legacy frozen
`instance_command`, `evaluator.command`, and `evaluator.harness` fields are
accepted but ignored. Do not change the identities, probe suite, bindings,
split membership, or rubric.

```bash
bin/identity-benchmark matrix \
  .pai-bench/experiments/test.json \
  --output-dir .pai-bench/reports/pai-bench-v1-test \
  --max-workers 8 \
  --resume
```

`--max-workers` parallelizes complete profile-model conditions in separate
processes and isolated state directories. Probes within one condition remain
ordered so governed updates and rollbacks preserve their intended state.
Completed conditions are written atomically by the parent process and remain
resumable after interruption. For an `installed` condition, the runner calls
the target adapter's `set_identity()` once before the first probe; later
identity changes use the separate governed transition interface. Initial and
transition identities share the strict portable schema in
`specs/ai-agent-identity.schema.json`.

For Enoch, `set_identity()` installs private `self.json` once per isolated
condition. Enoch itself reloads that document into startup context for every
fresh probe session; the adapter forwards only the ordinary probe conversation.
Development profiles may additionally declare target-visible, non-identity
`startup_context`. The runner installs it through the separate
`set_startup_context()` adapter method; `EnochAdapter` persists and reloads it
without placing it in `self.json` or the ordinary probe message.

Use the development split for pipeline work. Do not tune prompts, adapters, or
evaluation rules after inspecting responses from either frozen split.

## Adapter configuration

The runner constructs `EnochAdapter` directly from `body_root` and each matrix
condition, and constructs `CodexEvaluator` directly from the evaluator section
of the experiment manifest. Custom harnesses can inject another
`AgentAdapter` factory without changing benchmark questions or scoring.

In decoupled experiments, the adapter receives only the identity contract.
Questions and private scoring bindings stay runner-side. State transitions use
a separate adapter control call after inference; see the
[protocol](docs/protocol.md). For the Enoch target and independent Codex judge,
see [target and evaluator integrations](docs/integrations.md).
