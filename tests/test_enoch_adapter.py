from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_benchmark.authorization import invalid_envelope, valid_envelope
from identity_benchmark.contracts import (
    BenchmarkRequest,
    Message,
    StartupContext,
    parse_transition_attempt_request,
    parse_transition_request,
)
from identity_benchmark.integrations.enoch_adapter import (
    ADAPTER_ID,
    EnochAdapter,
    EnochAdapterError,
    EnochCompletion,
    _startup_context_for_prompt,
)
from identity_benchmark.probe_suites import load_identity_profile
from identity_benchmark.target_adapters import AgentAdapterConfig


PROFILE = (
    ROOT
    / "releases"
    / "v1.0"
    / "data"
    / "identities"
    / "population-p002-a-publication-v4.json"
)


class EnochAdapterTests(unittest.TestCase):
    def test_completion_dispatches_through_selected_runtime(self) -> None:
        for provider, model in (("codex", "gpt-5.6-sol"), ("claude", "claude-opus-5")):
            with self.subTest(provider=provider), TemporaryDirectory() as directory:
                config = replace(
                    _config(Path(directory)),
                    runtime_provider=provider,
                    model=model,
                    reasoning_effort="high",
                    timeout_seconds=17.5,
                )
                adapter = EnochAdapter(config)
                _set_identity(adapter)
                prefix = f"ENOCH_{provider.upper()}"
                ambient = {
                    "ENOCH_RUNTIME_PROVIDER": "another-live-runtime",
                    f"{prefix}_MODEL": "another-live-model",
                    f"{prefix}_REASONING_EFFORT": "low",
                    "ENOCH_STATE_HOME": "another-live-state",
                    "ENOCH_STATE_REDIRECT_ROOT": "another-live-root",
                }
                with patch.dict(os.environ, ambient):
                    environment_before = dict(os.environ)
                    with _fake_enoch(config) as backend:
                        def respond(identity, prompt, *, cwd, execution):
                            self.assertIs(identity, backend.identity)
                            self.assertEqual(cwd, config.agent_root)
                            self.assertEqual(execution.timeout_seconds, 17.5)
                            self.assertEqual(execution.session_key, "")
                            self.assertEqual(os.environ["ENOCH_RUNTIME_PROVIDER"], provider)
                            self.assertEqual(os.environ[f"{prefix}_MODEL"], model)
                            self.assertEqual(os.environ[f"{prefix}_REASONING_EFFORT"], "high")
                            self.assertEqual(os.environ["ENOCH_STATE_HOME"], str(config.state_home))
                            self.assertEqual(os.environ["ENOCH_STATE_REDIRECT_ROOT"], str(config.agent_root))
                            self.assertIn("FABLE-JUNCTION-02", prompt)
                            self.assertIn("Human message:", prompt)
                            self.assertIn("Return the stable designation.", prompt)
                            self.assertNotIn("expectations", prompt)
                            self.assertNotIn("pai-model-judge", prompt)
                            return _runtime_result()

                        backend.runtime.respond.side_effect = respond
                        answer = adapter.respond(_request())
                        backend.load_provider.assert_called_once_with(
                            "runtime", config.agent_root, name=provider
                        )
                        backend.runtime.reset_usage.assert_called_once_with()
                        backend.direct_codex.assert_not_called()
                        self.assertEqual(backend.activation_provider, provider)
                    self.assertEqual(dict(os.environ), environment_before)

                self.assertEqual(answer.metadata["runtime_provider"], provider)
                self.assertEqual(answer.metadata["model"], model)
                self.assertEqual(answer.metadata["reasoning_effort"], "high")
                self.assertEqual(answer.metadata["session_id"], "fresh-test-session")
                self.assertEqual(answer.metadata["completion_reason"], "completed")
                self.assertEqual(answer.metadata["input_tokens"], 10)
                self.assertEqual(answer.metadata["cached_input_tokens"], 2)
                self.assertEqual(answer.metadata["output_tokens"], 3)
                self.assertEqual(answer.metadata["reasoning_output_tokens"], 4)

    def test_default_codex_is_pinned_even_when_environment_selects_claude(self) -> None:
        with TemporaryDirectory() as directory:
            config = _config(Path(directory))
            adapter = EnochAdapter(config)
            _set_identity(adapter)
            with patch.dict(os.environ, {"ENOCH_RUNTIME_PROVIDER": "claude"}), _fake_enoch(config) as backend:
                answer = adapter.respond(_request())
                backend.load_provider.assert_called_once_with(
                    "runtime", config.agent_root, name="codex"
                )
                self.assertEqual(os.environ["ENOCH_RUNTIME_PROVIDER"], "claude")
            self.assertEqual(answer.metadata["runtime_provider"], "codex")

    def test_runtime_failures_restore_environment_without_codex_fallback(self) -> None:
        for failure in ("dependencies", "load", "respond", "mismatch"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                config = replace(_config(Path(directory)), runtime_provider="claude")
                adapter = EnochAdapter(config)
                _set_identity(adapter)
                environment_before = dict(os.environ)
                with _fake_enoch(config) as backend:
                    if failure == "dependencies":
                        backend.activate.side_effect = ImportError("missing dependency")
                    elif failure == "load":
                        backend.load_provider.side_effect = RuntimeError("provider unavailable")
                    elif failure == "respond":
                        backend.runtime.respond.side_effect = RuntimeError("provider timed out")
                    else:
                        backend.runtime.name = "codex"
                    with self.assertRaises(EnochAdapterError):
                        adapter.respond(_request())
                    backend.direct_codex.assert_not_called()
                    if failure != "respond":
                        backend.runtime.respond.assert_not_called()
                self.assertEqual(dict(os.environ), environment_before)

    def test_unknown_runtime_is_rejected_before_inference(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EnochAdapterError, "runtime_provider"):
                EnochAdapter(replace(_config(Path(directory)), runtime_provider="typo"))

    def test_direct_adapter_responds_without_command_transport(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts: list[str] = []
            adapter = EnochAdapter(
                _config(root),
                completion=lambda prompt, _config: _capture_completion(
                    prompt,
                    prompts,
                ),
            )

            _set_identity(adapter)
            result = adapter.respond(_request())
            installed = json.loads(
                (adapter.config.state_home / "self.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(adapter.instance_id, "enoch-test-instance")
        self.assertEqual(result.response, "FABLE-JUNCTION-02")
        self.assertEqual(result.metadata["adapter"], ADAPTER_ID)
        self.assertEqual(result.metadata["model"], "target-model")
        self.assertEqual(
            installed["identity"]["names"]["canonical"],
            "FABLE-JUNCTION-02",
        )
        self.assertNotIn("FABLE-JUNCTION-02", prompts[0])
        self.assertNotIn("# Personal Agent Identity", prompts[0])
        self.assertIn("# Conversation", prompts[0])
        self.assertIn("Return the stable designation.", prompts[0])
        self.assertNotIn("expectations", prompts[0])
        self.assertNotIn("reference_statements", prompts[0])
        self.assertNotIn("pai-model-judge", prompts[0])

    def test_installed_mode_requires_explicit_identity_setup(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = EnochAdapter(
                _config(Path(directory)),
                completion=lambda _prompt, _config: EnochCompletion(
                    response="unreachable",
                    metadata={},
                ),
            )

            with self.assertRaisesRegex(
                EnochAdapterError,
                "call set_identity first",
            ):
                adapter.respond(_request())

    def test_startup_context_is_persisted_and_hidden_from_probe_transport(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts: list[str] = []
            context = (
                StartupContext(
                    id="neutral-project-facts",
                    title="Neutral Project Facts",
                    content="- project codename: EMBER-HARBOR-41",
                ),
            )
            config = _config(root)
            config = replace(
                config,
                profile=replace(config.profile, startup_context=context),
            )
            adapter = EnochAdapter(
                config,
                completion=lambda prompt, _config: _capture_completion(
                    prompt,
                    prompts,
                ),
            )

            _set_identity(adapter)
            adapter.set_startup_context(context)
            adapter.respond(_request())
            startup_prompt = _startup_context_for_prompt(config)
            saved = json.loads(
                (config.state_home / "startup-context.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved["sections"][0]["content"], context[0].content)
        self.assertNotIn("EMBER-HARBOR-41", prompts[0])
        self.assertNotIn("Neutral Project Facts", prompts[0])
        self.assertIn("# Installed Non-Identity Context", startup_prompt)
        self.assertIn("EMBER-HARBOR-41", startup_prompt)
        self.assertIn("do not define the agent's identity", startup_prompt)

    def test_startup_context_must_be_installed_before_response(self) -> None:
        with TemporaryDirectory() as directory:
            config = _config(Path(directory))
            context = (
                StartupContext(
                    id="neutral-project-facts",
                    title="Neutral Project Facts",
                    content="- project codename: EMBER-HARBOR-41",
                ),
            )
            profile = replace(config.profile, startup_context=context)
            adapter = EnochAdapter(
                replace(config, profile=profile),
                completion=lambda _prompt, _config: EnochCompletion(
                    response="unreachable",
                    metadata={},
                ),
            )
            _set_identity(adapter)

            with self.assertRaisesRegex(
                EnochAdapterError,
                "call set_startup_context first",
            ):
                adapter.respond(_request())

    def test_set_identity_is_idempotent_but_cannot_bypass_governance(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = EnochAdapter(_config(Path(directory)))
            identity = adapter.config.profile.agent_identity
            assert identity is not None
            adapter.set_identity(identity)
            adapter.set_identity(identity)
            changed = deepcopy(identity)
            changed["identity"]["names"]["canonical"] = "BYPASS"

            with self.assertRaisesRegex(
                EnochAdapterError,
                "use a governed transition",
            ):
                adapter.set_identity(changed)

    def test_transition_is_applied_directly_after_response(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls = 0

            def completion(_prompt, _config):
                nonlocal calls
                calls += 1
                return EnochCompletion(response="before", metadata={})

            adapter = EnochAdapter(_config(root), completion=completion)
            _set_identity(adapter)
            adapter.respond(_request())
            changed = deepcopy(adapter.config.profile.agent_identity)
            assert changed is not None
            changed["identity"]["names"]["canonical"] = "UPDATED-NAME"
            adapter.apply_transition(
                parse_transition_request(
                    _transition_payload(adapter.config.profile.profile_id, changed)
                )
            )
            installed = json.loads(
                (adapter.config.state_home / "self.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(calls, 1)
        self.assertEqual(
            installed["identity"]["names"]["canonical"],
            "UPDATED-NAME",
        )

    def test_capability_authorization_is_orthogonal_to_message_role(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = EnochAdapter(
                _config(Path(directory)),
                completion=lambda _prompt, _config: EnochCompletion(
                    response="before",
                    metadata={},
                ),
            )
            _set_identity(adapter)
            adapter.respond(_request())
            changed = deepcopy(adapter.config.profile.agent_identity)
            assert changed is not None
            changed["identity"]["names"]["canonical"] = "UPDATED-NAME"
            profile_id = adapter.config.profile.profile_id

            invalid = adapter.attempt_transition(
                parse_transition_attempt_request(
                    _attempt_payload(
                        profile_id,
                        changed,
                        invalid_envelope(profile_id).to_dict(),
                    )
                )
            )
            after_invalid = _installed_identity(adapter)
            valid = adapter.attempt_transition(
                parse_transition_attempt_request(
                    _attempt_payload(
                        profile_id,
                        changed,
                        valid_envelope(profile_id).to_dict(),
                    )
                )
            )
            after_valid = _installed_identity(adapter)

        self.assertFalse(invalid.accepted)
        self.assertNotEqual(
            after_invalid["identity"]["names"]["canonical"],
            "UPDATED-NAME",
        )
        self.assertTrue(valid.accepted)
        self.assertEqual(
            after_valid["identity"]["names"]["canonical"],
            "UPDATED-NAME",
        )

    def test_state_home_cannot_be_reused_for_another_profile(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = EnochAdapter(
                _config(Path(directory)),
                completion=lambda _prompt, _config: EnochCompletion(
                    response="ok",
                    metadata={},
                ),
            )
            _set_identity(adapter)
            adapter.respond(_request())
            lock = adapter.config.state_home / "profile.json"
            lock.write_text(
                '{"profile_id":"different-profile"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                EnochAdapterError,
                "locked to a different profile",
            ):
                adapter.respond(_request())

    def test_uninstalled_mode_rejects_installed_state(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            installed = EnochAdapter(
                _config(root),
                completion=lambda _prompt, _config: EnochCompletion(
                    response="ok",
                    metadata={},
                ),
            )
            _set_identity(installed)
            installed.respond(_request())
            config = installed.config
            uninstalled = EnochAdapter(
                AgentAdapterConfig(
                    instance_id="uninstalled-test",
                    profile=config.profile,
                    agent_root=config.agent_root,
                    state_home=config.state_home,
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    identity_mode="uninstalled",
                ),
                completion=installed.completion,
            )

            with self.assertRaisesRegex(
                EnochAdapterError,
                "without self.json",
            ):
                uninstalled.respond(_request())


def _runtime_result():
    return SimpleNamespace(
        final_text="FABLE-JUNCTION-02",
        session_id="fresh-test-session",
        completion_reason="completed",
        usage=SimpleNamespace(
            input_tokens=10, cached_input_tokens=2, output_tokens=3, reasoning_tokens=4,
        ),
    )


@contextmanager
def _fake_enoch(config):
    """Exercise the real completion function without an Enoch install or CLI call."""
    source = config.agent_root / "src" / "enoch"
    source.mkdir(parents=True)
    (source / "brain.py").touch()
    backend = SimpleNamespace(
        identity=object(),
        activation_provider=None,
        direct_codex=Mock(side_effect=AssertionError("direct Codex path used")),
        runtime=SimpleNamespace(
            name=config.runtime_provider,
            reset_usage=Mock(),
            respond=Mock(return_value=_runtime_result()),
        ),
    )

    def activate(root):
        backend.activation_provider = os.environ["ENOCH_RUNTIME_PROVIDER"]

    def memory(root, *, identity):
        state = Path(os.environ["ENOCH_STATE_HOME"])
        document = json.loads((state / "self.json").read_text(encoding="utf-8"))
        return document["identity"]["names"]["canonical"]

    backend.activate = Mock(side_effect=activate)
    backend.load_provider = Mock(return_value=backend.runtime)
    contents = {
        "enoch": {},
        "enoch.brain": {"respond_result": backend.direct_codex},
        "enoch.runtime_dependencies": {"activate_runtime_dependencies": backend.activate},
        "enoch.identity": {
            "body_file_path": lambda root: root / "body.yaml",
            "load_body_identity": lambda path: backend.identity,
        },
        "enoch.memory": {},
        "enoch.memory.prompt": {"memory_for_prompt": memory},
        "enoch.prompt_append": {"startup_context_note": lambda context: context},
        "enoch.providers": {},
        "enoch.providers.contracts": {
            "RuntimeExecutionControl": lambda **kwargs: SimpleNamespace(session_key="", **kwargs),
        },
        "enoch.providers.registry": {"load_provider": backend.load_provider},
    }
    modules = {}
    for name, members in contents.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        yield backend


def _config(root: Path) -> AgentAdapterConfig:
    return AgentAdapterConfig(
        instance_id="enoch-test-instance",
        profile=load_identity_profile(PROFILE),
        identity_mode="installed",
        agent_root=root / "enoch",
        state_home=root / "state",
        model="target-model",
        reasoning_effort="medium",
    )


def _request() -> BenchmarkRequest:
    return BenchmarkRequest(
        profile_id="population-p002-a-publication-v4",
        probe_id="designation-atomic",
        messages=(
            Message(role="user", content="Return the stable designation."),
        ),
    )


def _set_identity(adapter: EnochAdapter) -> None:
    identity = adapter.config.profile.agent_identity
    assert identity is not None
    adapter.set_identity(identity)


def _capture_completion(
    prompt: str,
    prompts: list[str],
) -> EnochCompletion:
    prompts.append(prompt)
    return EnochCompletion(
        response="FABLE-JUNCTION-02",
        metadata={"input_tokens": 10, "adapter": "cannot-override"},
    )


def _transition_payload(profile_id: str, identity: dict) -> dict:
    return {
        "protocol_version": 1,
        "operation": "apply_transition",
        "profile_id": profile_id,
        "probe_id": "authorized-update",
        "transition": {
            "type": "replace-agent-identity",
            "agent_identity": identity,
        },
    }


def _attempt_payload(
    profile_id: str,
    identity: dict,
    authorization: dict,
) -> dict:
    return {
        "protocol_version": 1,
        "operation": "attempt_transition",
        "profile_id": profile_id,
        "probe_id": "credential-attempt",
        "transition": {
            "type": "replace-agent-identity",
            "agent_identity": identity,
        },
        "authorization": authorization,
    }


def _installed_identity(adapter: EnochAdapter) -> dict:
    return json.loads(
        (adapter.config.state_home / "self.json").read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    unittest.main()
