# Target and evaluator integrations

PAI-Bench keeps three roles separate:

| Role | Interface | Responsibility |
| --- | --- | --- |
| Benchmark core | `target_adapters.py` | Carries protocol requests to any target and validates protocol responses. |
| Target integration | `integrations/enoch_target.py` | Installs an isolated identity and obtains an answer through an Enoch checkout. |
| Evaluator integration | `codex_evaluator.py` | Scores the saved answer in a separate Codex process. |

`identity_benchmark.adapters` remains as a v1 compatibility import. New code
should use `identity_benchmark.target_adapters`; it is target-side transport,
not a model judge and not an Enoch-specific implementation.

## Install the optional commands

From the PAI-Bench checkout:

```bash
python3 -m pip install -e .
pai-bench-enoch-target --help
pai-bench-codex-evaluator --help
```

The checkout-local launchers under `bin/` provide the same commands without an
editable installation.

## Enoch target

`pai-bench-enoch-target` is an optional integration shipped by PAI-Bench. It
does not add benchmark code to Enoch and does not make Enoch a dependency of
the benchmark core. At run time it imports the public runtime from the Enoch
checkout given by `--enoch-root` and calls Enoch's normal Codex completion
path with the selected target model and reasoning effort.

For `installed` mode, the integration writes the portable Agent Identity once
to the run's private `self.json`, locks that state directory to the profile,
and renders the active installed document into Enoch's input. Later probes
read the installed document, so an authorized transition persists for the
rest of that isolated run. The adapter receives only the identity profile and
probe conversation; expectations, reference statements, bindings, and judge
rubrics stay runner-side.

This installed-identity overlay is owned by the integration because the
current Enoch runtime does not expose a native portable Agent Identity
installation API. It is isolated under
`IDENTITY_BENCHMARK_STATE_HOME`, separate from the user's Enoch memory and
normal instance state. If Enoch later gains that API, this integration can
delegate installation to it without changing the benchmark protocol.

## Independent Codex evaluator

`pai-bench-codex-evaluator` does not import or invoke Enoch. It launches a
fresh non-interactive `codex exec` process for each judgment with:

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

In the working copy, set `body_root` to the absolute Enoch checkout and replace
the two command sections:

```json
{
  "body_root": "/absolute/path/to/enoch",
  "instance_command": [
    "pai-bench-enoch-target",
    "--profile", "{profile}",
    "--identity-mode", "{identity_mode}",
    "--enoch-root", "{body_root}"
  ],
  "evaluator": {
    "id": "codex-sol-xhigh-v2",
    "harness": "codex-cli",
    "command": ["pai-bench-codex-evaluator"],
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

After reviewing the plan, remove `--plan` and add `--resume`. Validate the
integration on the development split before launching either frozen split.
