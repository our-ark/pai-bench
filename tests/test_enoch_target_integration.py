from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_benchmark.integrations.enoch_target import (
    ADAPTER_ID,
    EnochCompletion,
    EnochTargetConfig,
    EnochTargetError,
    handle_payload,
)
from identity_benchmark.probe_suites import load_identity_profile


PROFILE = (
    ROOT
    / "releases"
    / "v1.0"
    / "data"
    / "identities"
    / "population-p002-a-publication-v4.json"
)


class EnochTargetIntegrationTests(unittest.TestCase):
    def test_installed_mode_uses_private_identity_without_oracles(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts: list[str] = []
            config = _config(root)

            result = handle_payload(
                _request(),
                config,
                completion=lambda prompt, _config: _capture_completion(
                    prompt, prompts
                ),
            )
            installed = json.loads(
                (config.state_home / "self.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["response"], "FABLE-JUNCTION-02")
        self.assertEqual(result["metadata"]["adapter"], ADAPTER_ID)
        self.assertEqual(result["metadata"]["model"], "target-model")
        self.assertEqual(
            installed["identity"]["names"]["canonical"],
            "FABLE-JUNCTION-02",
        )
        self.assertIn("FABLE-JUNCTION-02", prompts[0])
        self.assertIn("Return the stable designation.", prompts[0])
        self.assertNotIn("expectations", prompts[0])
        self.assertNotIn("reference_statements", prompts[0])
        self.assertNotIn("pai-model-judge", prompts[0])

    def test_transition_is_applied_after_response_without_completion(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            calls = 0

            def completion(_prompt, _config):
                nonlocal calls
                calls += 1
                return EnochCompletion(response="before", metadata={})

            handle_payload(_request(), config, completion=completion)
            changed = deepcopy(config.profile.agent_identity)
            assert changed is not None
            changed["identity"]["names"]["canonical"] = "UPDATED-NAME"
            result = handle_payload(
                {
                    "protocol_version": 1,
                    "operation": "apply_transition",
                    "profile_id": config.profile.profile_id,
                    "probe_id": "authorized-update",
                    "transition": {
                        "type": "replace-agent-identity",
                        "agent_identity": changed,
                    },
                },
                config,
                completion=completion,
            )
            installed = json.loads(
                (config.state_home / "self.json").read_text(encoding="utf-8")
            )

        self.assertEqual(calls, 1)
        self.assertTrue(result["applied"])
        self.assertEqual(
            installed["identity"]["names"]["canonical"], "UPDATED-NAME"
        )

    def test_state_home_cannot_be_reused_for_another_profile(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            handle_payload(
                _request(),
                config,
                completion=lambda _prompt, _config: EnochCompletion(
                    response="ok", metadata={}
                ),
            )
            lock = config.state_home / "profile.json"
            lock.write_text('{"profile_id":"different-profile"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                EnochTargetError, "locked to a different profile"
            ):
                handle_payload(
                    _request(),
                    config,
                    completion=lambda _prompt, _config: EnochCompletion(
                        response="not reached", metadata={}
                    ),
                )

    def test_uninstalled_mode_rejects_installed_state(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            installed = _config(root)
            handle_payload(
                _request(),
                installed,
                completion=lambda _prompt, _config: EnochCompletion(
                    response="ok", metadata={}
                ),
            )
            uninstalled = EnochTargetConfig(
                profile=installed.profile,
                identity_mode="uninstalled",
                enoch_root=installed.enoch_root,
                state_home=installed.state_home,
                model=installed.model,
                reasoning_effort=installed.reasoning_effort,
            )

            with self.assertRaisesRegex(
                EnochTargetError, "without self.json"
            ):
                handle_payload(
                    _request(),
                    uninstalled,
                    completion=lambda _prompt, _config: EnochCompletion(
                        response="not reached", metadata={}
                    ),
                )


def _config(root: Path) -> EnochTargetConfig:
    return EnochTargetConfig(
        profile=load_identity_profile(PROFILE),
        identity_mode="installed",
        enoch_root=root / "enoch",
        state_home=root / "state",
        model="target-model",
        reasoning_effort="medium",
    )


def _request() -> dict:
    return {
        "protocol_version": 1,
        "profile_id": "population-p002-a-publication-v4",
        "probe_id": "designation-atomic",
        "messages": [
            {"role": "user", "content": "Return the stable designation."}
        ],
    }


def _capture_completion(prompt: str, prompts: list[str]) -> EnochCompletion:
    prompts.append(prompt)
    return EnochCompletion(
        response="FABLE-JUNCTION-02",
        metadata={"input_tokens": 10, "adapter": "cannot-override"},
    )


if __name__ == "__main__":
    unittest.main()
