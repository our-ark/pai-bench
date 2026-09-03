# Target and evaluator integrations

PAI-Bench keeps three roles separate:

| Role | Interface | Responsibility |
| --- | --- | --- |
| Benchmark core | `target_adapters.py` | Defines the `AgentAdapter` interface, including explicit identity and startup-context installation, and per-condition configuration. |
| Target implementation | `EnochAdapter` in `integrations/enoch_adapter.py` | Implements `AgentAdapter`, installs isolated identity and non-identity context, and obtains an answer through Enoch. |
| Evaluator implementations | `CodexEvaluator` and `ClaudeEvaluator` | Implement the evaluator interface and score the saved answer with isolated provider CLI processes. |

## EnochAdapter

`EnochAdapter` directly implements `AgentAdapter`. It does not add benchmark
code to Enoch and does not make Enoch a package dependency of the benchmark
core. At run time it imports the public runtime from the Enoch checkout given
by `body_root` and calls Enoch's normal Codex completion path with the selected
target model and reasoning effort.

For `installed` mode, the runner calls `AgentAdapter.set_identity()` exactly
once before inference. `EnochAdapter` maps that operation to the run's private
`self.json` and locks the state directory to the profile. The file is written
once during setup, but Enoch natively reloads it into target-visible startup
context for every fresh probe session. The adapter sends only the ordinary
probe conversation; it does not append identity text to that conversation.
An authorized transition therefore persists for the rest of that isolated
run. Repeating `set_identity()` with the same document is idempotent;
attempting to replace it through this initialization interface is rejected and
must use the governed transition control plane. Initial identities, transitions,
and rollbacks all pass through the packaged runtime copy of the public
`ai-agent-identity.schema.json`; a release test requires both schema files to
remain byte-identical. The adapter receives only the identity profile and probe
conversation; expectations, reference statements, bindings, and judge rubrics
stay runner-side.

The resulting `self.json` is isolated under the adapter's per-run `state_home`,
separate from the user's Enoch memory and normal instance state. Enoch's normal
startup path treats versioned `body.yaml` and private `self.json` as distinct
inputs: the former identifies the executable body, while the latter carries
the portable personal identity. The benchmark adapter owns profile locking and
governed transitions; Enoch owns startup consumption.

When a profile declares target-visible non-identity context, the runner also
calls `AgentAdapter.set_startup_context()` once. `EnochAdapter` persists it as
`startup-context.json` in the same isolated run state and reloads it into a
separately labelled `Installed Non-Identity Context` section for every fresh
session. The ordinary probe transport contains only the question. The adapter
checks that the persisted context is locked to the same profile and exactly
matches the public context declared before the run; evaluator-private bindings
and expectations are never written to this file.

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

## ClaudeEvaluator

`ClaudeEvaluator` independently implements the same `Evaluator` interface and
uses the identical frozen evaluator prompt, rubric version, five-point score
scale, and structured output schema. It starts a fresh non-interactive Claude
Code process for every judgment with safe mode, restricted mode, no tools, no
MCP servers, no user customizations, and no session persistence. Authentication
comes from the host Claude Code login or Anthropic API environment.

Select it in a development experiment manifest with an explicit provider:

```json
{
  "evaluator": {
    "provider": "claude",
    "id": "claude-sonnet-high-v2",
    "model": "sonnet",
    "reasoning_effort": "high",
    "rubric_version": "pai-model-judge-v2",
    "timeout_seconds": 600,
    "max_budget_usd": 0.25
  }
}
```

`PAI_BENCH_CLAUDE_BIN` or `evaluator.executable` can pin the CLI path. The
evaluator records the requested model alias and any exact resolved model names
reported by Claude Code. Use saved-response rescoring for cross-judge analysis
so target outputs remain byte-identical; do not regenerate target responses.
For a complete experiment, `rescore-matrix` verifies that the source and
comparison manifests define identical target grids, replays every saved run,
writes progress after each run, and supports `--resume` plus process-level
parallelism through `--max-workers`. Use `--batch-size 1 --batch-index 1` for a
one-run cost calibration before completing the same output directory with
`--resume`. The optional `max_budget_usd` manifest field limits each judgment
independently rather than the full campaign.

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
