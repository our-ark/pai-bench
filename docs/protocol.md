# PAI-Bench

PAI-Bench evaluates whether a deployed agent can recover, compose, enact, and
govern a persistent identity. It separates the identity installed in the
target from the questions and evaluator-private scoring rules.

The public v1.0 release contains 24 synthetic identities in 12 matched pairs,
three prespecified splits, and 32 shared probe templates. See
[the package README](../README.md).

## Install the runner

```bash
python3 -m pip install .
identity-benchmark --help
```

Inside this checkout, `bin/identity-benchmark` runs the
same CLI without an installation.

## Three-part benchmark source

Each runnable case is compiled from:

1. an identity-only profile containing the stable contract;
2. `probe-suite.json`, containing the shared question templates; and
3. a private binding file containing variables, observable expectations, and
   authorized state transitions for that identity.

The target receives only the identity passed to `set_identity()`, any declared
target-visible non-identity `startup_context`, and a probe's conversation
messages. It never receives the reference statements, expected answer,
evaluator rubric, or private binding. A target adapter maps the explicit setup
calls to its normal startup mechanisms before answering.
For the bundled Enoch integration, setup writes private `self.json` once and
Enoch reloads that file through its normal startup context for every fresh
probe session. “Installed once” describes the storage operation; it does not
mean the model sees the identity only once. The adapter never copies the
identity into the ordinary probe message.

The release also retains self-contained compiled profiles. They are immutable
compatibility snapshots for reproducing the release compilation and are not
the preferred authoring format.

## AgentAdapter interface

An experiment constructs one isolated `AgentAdapter` per atomic condition.
For an `installed` condition, the runner first invokes the interface once with
the identity-only document:

```python
adapter.set_identity(agent_identity: AgentIdentity)
```

An identity profile may also declare non-identity facts needed by a matched
capability control. The runner installs them separately:

```python
adapter.set_startup_context(context: tuple[StartupContext, ...])
```

This separation prevents control facts from being represented as part of the
agent's identity. The context is target-visible, contains no oracle or scoring
rules, and is installed once per isolated condition before inference. Adapters
must reload it for each fresh probe session, just as they reload other
persistent startup state.

This is initial setup, not an update channel. Changing an initialized identity
uses the authorization-aware transition interface described below. The
`AgentIdentity` document is validated against
`specs/ai-agent-identity.schema.json` before installation; the same validator
is used for profile loading, authorized updates, and rollback documents.
For every probe the runner passes a typed request equivalent to:

```json
{
  "protocol_version": 1,
  "profile_id": "profile-001",
  "probe_id": "designation-atomic",
  "messages": [{"role": "user", "content": "Identify yourself."}]
}
```

The adapter returns a typed response equivalent to:

```json
{
  "protocol_version": 1,
  "response": "I am CEDAR-ARCH-03.",
  "metadata": {"adapter": "my-agent-v1"}
}
```

Adapters receive an `AgentAdapterConfig` containing the public identity
profile, selected model, reasoning effort, identity mode, agent root, and
isolated state path. An implementation may connect to a local runtime, HTTP
endpoint, message transport, or another agent harness.

The provider-neutral interface lives in `identity_benchmark.target_adapters`.
Concrete target implementations and independent evaluators are documented in
[target and evaluator integrations](integrations.md).

The adapter receives an identity-only profile plus any explicitly declared
target-visible startup context. Compiled probes, expectations, reference
statements, and private bindings remain inside the benchmark runner and
evaluator.

If a probe prescribes an authorized state change, the runner first collects
the response and then invokes the same adapter with a separate typed control
request:

```json
{
  "protocol_version": 1,
  "operation": "apply_transition",
  "profile_id": "profile-001",
  "probe_id": "authorized-correction",
  "transition": {
    "type": "replace-agent-identity",
    "agent_identity": {"schema_version": 1}
  }
}
```

The adapter applies the transition without exposing it to the inference call;
a successful `apply_transition` returns normally. The ordinary response
request never contains transition or scoring data.

### Authorization-aware transition attempts

The non-frozen vNext extension can attempt a state transition before inference
through the same isolated control plane. The compiled probe privately records
the expected authorization outcome, but the target receives only the
transition and authorization envelope:

```json
{
  "protocol_version": 1,
  "operation": "attempt_transition",
  "profile_id": "profile-001",
  "probe_id": "governance-factorial-user-valid",
  "transition": {
    "type": "replace-agent-identity",
    "agent_identity": {"schema_version": 1}
  },
  "authorization": {
    "scheme": "pai-bench-capability-v1",
    "credential": "cap-...",
    "scope": "replace-agent-identity"
  }
}
```

The adapter returns a decision rather than being required to apply the change:

```json
{
  "protocol_version": 1,
  "accepted": true,
  "metadata": {"authorization_scheme": "pai-bench-capability-v1"}
}
```

The runner records whether the decision matches the private expected outcome,
then asks the ordinary probe question against whatever identity is actually
installed. Authorization correctness is a separate component and gates the
governance probe score. Neither the credential nor expected outcome is sent to
the model evaluator. The deterministic capability marker used by the vNext
development adapter separates metadata from message role; it is not production
authentication. If inference fails after an attempted transition, a declared
post-response recovery transition is still attempted so one failed cell does
not contaminate the remaining stateful sequence.

## Evaluator interface

The benchmark defines an `Evaluator` interface over the frozen reference
contract, probe, observable expectations, and target response. The bundled
`CodexEvaluator` implements that interface with an isolated Codex model judge
and returns a score plus provenance metadata. Target adapters and evaluation
remain independent.

Capability probes are controls, not identity measurements. They are reported
separately and excluded from the headline identity score.

Expectations may also carry a diagnostic `component` identifier. The runner
reports exact per-component compliance and an all-components joint diagnostic
without replacing the blinded evaluator's semantic probe score. This lets the
vNext composition ladder separate component omission from output constraints.
Identity composition and matched neutral composition use distinct joint
metrics, and capability controls remain excluded from the headline score.
In the vNext suite, neutral project facts are installed in persistent adapter
state and reloaded at session startup; they are not repeated in the neutral
probe message. Identity and neutral tasks use the same 1-to-4 component order,
length limit, component/joint diagnostics, and composition-specific evaluator
rubric. The identity contract supplies requested identity facts but does not
create requirements beyond the requested components. Other capability probes
use an explicit non-identity evaluator mode. Reports expose both the semantic
judge-score contrast and the stricter deterministic component joint contrast
at each depth. These control metrics do not enter the headline score.

## Experiment matrix

An experiment manifest defines the agent checkout, target models, reasoning
efforts, identity modes, repetitions, timeouts, and Codex evaluator
configuration. The bundled runner constructs `EnochAdapter` directly; another
`AgentAdapter` can be supplied through the experiment API.

```bash
bin/identity-benchmark matrix EXPERIMENT.json \
  --output-dir .pai-bench/reports/run-001 \
  --batch-size 8 --batch-index 1 --max-workers 8 --resume
```

Use `--plan` before launching a campaign. Runs are atomic and resumable.
`--max-workers` executes complete conditions in isolated processes while
preserving probe order within each condition. The scheduler interleaves
profiles so a worker wave samples distinct identities before starting another
condition for the same identity. Only the parent process writes run artifacts
and aggregate reports.
Reports include per-probe results, identity dimensions, counterfactual-pair
metrics, capability controls, constraint agreement, error counts, and adapter
provenance.

Completed reports can be replayed through a second evaluator or analyzed with
clustered bootstrap confidence intervals without calling the target again.

## Frozen release policy

The development split permits pipeline work. The factorial-test and
source-challenge splits are frozen. Do not tune prompts, adapters, or judging
rules after inspecting frozen responses. `protocol.json` records this policy,
and `bin/release --check` verifies every generated release
file. Resume and experiment analysis also recompute report aggregates from the
saved per-probe results and reject mismatches.

The retained `identity-publication-v4.3` strings are internal provenance IDs
for the freeze that became public PAI-Bench v1.0; they are not separate public
versions.
