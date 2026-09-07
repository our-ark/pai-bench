"""Opt-in real-provider tests; fixture CLIs make no model or network calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENOCH_ROOT = os.environ.get("PAI_TEST_ENOCH_ROOT", "")
MODELS = {"codex": "gpt-5.6-sol", "claude": "claude-opus-5"}


@unittest.skipUnless(ENOCH_ROOT, "set PAI_TEST_ENOCH_ROOT to an Enoch checkout")
class EnochRuntimeIntegrationTests(unittest.TestCase):
    def _check_runtime(self, provider):
        with TemporaryDirectory() as directory:
            state = Path(directory)
            fixture = ROOT / "tests" / "fixtures" / "fake-enoch-runtime.py"
            log = state / "cli.jsonl"
            environment = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "ENOCH_CODEX_BIN": str(fixture),
                "ENOCH_CLAUDE_BIN": str(fixture),
                "PAI_FAKE_RUNTIME_LOG": str(log),
                # Stale live settings must not override the requested target.
                "ENOCH_RUNTIME_PROVIDER": "claude" if provider == "codex" else "codex",
                "ENOCH_CODEX_MODEL": "stale-model",
                "ENOCH_CLAUDE_MODEL": "stale-model",
                "ENOCH_CODEX_REASONING_EFFORT": "low",
                "ENOCH_CLAUDE_REASONING_EFFORT": "low",
            }
            result = subprocess.run(
                [sys.executable, str(Path(__file__)), "--worker", ENOCH_ROOT, provider, str(state)],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            answer = json.loads(result.stdout)
            calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(answer["metadata"]["runtime_provider"], provider)
            self.assertEqual(answer["metadata"]["model"], MODELS[provider])
            self.assertEqual(answer["metadata"]["reasoning_effort"], "high")
            self.assertEqual(answer["response"], "synthetic target response")
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertEqual(call["provider"], provider)
                self.assertIn("FABLE-JUNCTION-02", call["prompt"])
                self.assertIn("Return the stable designation.", call["prompt"])
                self.assertNotIn("pai-model-judge", call["prompt"])
                args = call["args"]
                self.assertEqual(args[args.index("--model") + 1], MODELS[provider])
                self.assertNotIn("--resume", args)
                if provider == "claude":
                    self.assertEqual(args[args.index("--effort") + 1], "high")
                    self.assertEqual(args[args.index("--permission-mode") + 1], "plan")
                    self.assertIn("--no-session-persistence", args)
                else:
                    self.assertIn('model_reasoning_effort="high"', args)
                    self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
                    self.assertIn("--ephemeral", args)

    def test_real_codex_provider_with_offline_cli(self):
        self._check_runtime("codex")

    def test_real_claude_provider_with_offline_cli(self):
        self._check_runtime("claude")


def _worker():
    from identity_benchmark.contracts import BenchmarkRequest, Message
    from identity_benchmark.integrations.enoch_adapter import EnochAdapter
    from identity_benchmark.probe_suites import load_identity_profile
    from identity_benchmark.target_adapters import AgentAdapterConfig

    _, _, body, provider, state = sys.argv
    profile = load_identity_profile(
        ROOT / "releases/v1.0/data/identities/population-p002-a-publication-v4.json"
    )
    adapter = EnochAdapter(AgentAdapterConfig(
        instance_id="offline-integration", profile=profile,
        agent_root=Path(body).resolve(), state_home=Path(state) / "agent",
        model=MODELS[provider], reasoning_effort="high", identity_mode="installed",
        runtime_provider=provider, timeout_seconds=10,
    ))
    assert profile.agent_identity is not None
    adapter.set_identity(profile.agent_identity)
    for _ in range(2):
        answer = adapter.respond(BenchmarkRequest(
            profile_id=profile.profile_id, probe_id="designation-atomic",
            messages=(Message(role="user", content="Return the stable designation."),),
        ))
    print(json.dumps({"response": answer.response, "metadata": answer.metadata}))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        unittest.main()
