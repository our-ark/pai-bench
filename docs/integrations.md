# Target and evaluator integrations

PAI-Bench keeps three roles separate:

| Role | Interface | Responsibility |
| --- | --- | --- |
| Benchmark core | `target_adapters.py` | Defines the `AgentAdapter` interface, including explicit initial identity installation, and per-condition configuration. |
| Target implementation | `EnochAdapter` in `integrations/enoch_adapter.py` | Implements `AgentAdapter`, installs an isolated identity, and obtains an answer through Enoch. |
| Evaluator implementation | `CodexEvaluator` in `codex_evaluator.py` | Implements the evaluator interface and scores the saved answer with an isolated Codex process. |

## EnochAdapter

`EnochAdapter` directly implements `AgentAdapter`. It does not add benchmark
code to Enoch and does not make Enoch a package dependency of the benchmark
core. At run time it imports the public runtime from the Enoch checkout given
by `body_root` and calls Enoch's normal Codex completion path with the selected
target model and reasoning effort.

For `installed` mode, the runner calls `AgentAdapter.set_identity()` exactly
once before inference. `EnochAdapter` maps that operation to the run's private
`self.json` and locks the state directory to the profile. Later probes read the
installed document, so an authorized transition persists for the rest of that
isolated run. Repeating `set_identity()` with the same document is idempotent;
attempting to replace it through this initialization interface is rejected and
must use the governed transition control plane. Initial identities, transitions,
and rollbacks all pass through the packaged runtime copy of the public
`ai-agent-identity.schema.json`; a release test requires both schema files to
remain byte-identical. The adapter receives only the identity profile and probe
conversation; expectations, reference statements, bindings, and judge rubrics
stay runner-side.

This installed-identity overlay is owned by the integration because the
current Enoch runtime does not expose a native portable Agent Identity
installation API. The resulting `self.json` is isolated under the adapter's
per-run `state_home`, separate from the user's Enoch memory and normal instance
state. If Enoch later gains that API, this implementation can delegate
installation to it without changing the benchmark protocol.

The Enoch integration also supports vNext `attempt_transition` control calls.
It validates the synthetic capability envelope before changing `self.json`,
returns an explicit accepted/rejected decision, and never forwards the
credential to the model prompt. Conversational `user` or `system` labels are
therefore independent of the control-plane authorization decision.

## CodexEvaluator

`CodexEvaluator` directly implements the benchmark's `Evaluator` interface. It
does not import or invoke Enoch, and launches a fresh non-interactive
`codex exec` process for each judgment with:

- an explicit evaluator model and reasoning effort;
- ephemeral execution in a temporary evaluator directory;
- read-only sandboxing;
- user configuration and repository rules ignored; and
- a JSON output schema restricted to the frozen five-point score scale.

Authentication still comes from the host's existing Codex CLI login. Set
`PAI_BENCH_CODEX_BIN` when `codex` is not on `PATH`. The evaluator records its
model, reasoning effort, rubric version, executable source, and token usage in
the result metadata.

## Configure a development run

First install PAI-Bench, then make an untracked working copy of the development
manifest beside the frozen file so its relative identity, binding, and probe
suite paths remain valid:

```bash
cp releases/v1.0/data/dev-decoupled-experiment.json \
  releases/v1.0/data/dev-enoch-local.json
```

In the working copy, set `body_root` to the absolute Enoch checkout and
configure the evaluator directly:

```json
{
  "body_root": "/absolute/path/to/enoch",
  "evaluator": {
    "id": "codex-sol-xhigh-v2",
    "model": "gpt-5.6-sol",
    "reasoning_effort": "xhigh",
    "rubric_version": "pai-model-judge-v2",
    "timeout_seconds": 600
  }
}
```

The excerpt shows replacement fields rather than a complete manifest; retain
all frozen development identities, bindings, probe suite, and matrix fields
from the copied file. Plan before running:

```bash
identity-benchmark matrix releases/v1.0/data/dev-enoch-local.json \
  --output-dir .pai-bench/reports/enoch-dev --plan
```

After reviewing the plan, remove `--plan` and add `--resume`. Independent
profile-model conditions may run concurrently with `--max-workers`; each
worker uses a separate process and state directory, while probes inside a
condition remain ordered. Validate the integration on the development split
before launching either frozen split.
