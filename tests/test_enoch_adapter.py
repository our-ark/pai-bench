from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


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
