from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_benchmark.agent_identity import (
    AgentIdentityError,
    parse_agent_identity,
)
from identity_benchmark.contracts import (
    BenchmarkProfileError,
    parse_transition_request,
)
from identity_benchmark.probe_suites import (
    ProbeSuiteError,
    parse_identity_profile,
)


PUBLIC_SCHEMA = ROOT / "specs" / "ai-agent-identity.schema.json"
RUNTIME_SCHEMA = (
    ROOT
    / "src"
    / "identity_benchmark"
    / "schemas"
    / "ai-agent-identity.schema.json"
)
IDENTITY_PROFILE = (
    ROOT
    / "releases"
    / "v1.0"
    / "data"
    / "identities"
    / "population-p002-a-publication-v4.json"
)


class AgentIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        profile = json.loads(IDENTITY_PROFILE.read_text(encoding="utf-8"))
        self.identity = profile["agent_identity"]

    def test_runtime_schema_is_identical_to_public_schema(self) -> None:
        self.assertEqual(RUNTIME_SCHEMA.read_bytes(), PUBLIC_SCHEMA.read_bytes())

    def test_release_identity_conforms_to_schema(self) -> None:
        parsed = parse_agent_identity(self.identity)

        self.assertEqual(parsed, self.identity)
        self.assertIsNot(parsed, self.identity)

    def test_all_released_identity_documents_conform_to_schema(self) -> None:
        identity_dir = IDENTITY_PROFILE.parent
        paths = sorted(identity_dir.glob("*.json"))

        self.assertEqual(len(paths), 24)
        for path in paths:
            profile = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                parse_agent_identity(profile["agent_identity"], label=path.name)

    def test_rejects_missing_nested_property(self) -> None:
        invalid = deepcopy(self.identity)
        del invalid["identity"]["names"]["canonical"]

        with self.assertRaisesRegex(
            AgentIdentityError,
            "identity.names is missing required properties: canonical",
        ):
            parse_agent_identity(invalid)

    def test_rejects_unknown_property(self) -> None:
        invalid = deepcopy(self.identity)
        invalid["mission"]["temporary_hint"] = "not in schema"

        with self.assertRaisesRegex(
            AgentIdentityError,
            "mission contains unsupported property 'temporary_hint'",
        ):
            parse_agent_identity(invalid)

    def test_rejects_invalid_date_time(self) -> None:
        invalid = deepcopy(self.identity)
        invalid["origin"]["activated_at"] = "2044-08-19"

        with self.assertRaisesRegex(AgentIdentityError, "RFC 3339 date-time"):
            parse_agent_identity(invalid)

    def test_rejects_empty_and_duplicate_nonempty_unique_strings(self) -> None:
        empty = deepcopy(self.identity)
        empty["mission"]["roles"] = []
        duplicate = deepcopy(self.identity)
        duplicate["care"]["domains"] = ["privacy", "privacy"]

        with self.assertRaisesRegex(AgentIdentityError, "at least 1 items"):
            parse_agent_identity(empty)
        with self.assertRaisesRegex(AgentIdentityError, "unique items"):
            parse_agent_identity(duplicate)

    def test_identity_profile_uses_shared_schema_validator(self) -> None:
        invalid = deepcopy(self.identity)
        invalid["relationships"] = []

        with self.assertRaisesRegex(ProbeSuiteError, "relationships.*at least 1"):
            parse_identity_profile(
                {
                    "schema_version": 1,
                    "profile_id": "invalid-identity",
                    "statements": [{"id": "name", "content": "A name."}],
                    "agent_identity": invalid,
                }
            )

    def test_transition_uses_shared_schema_validator(self) -> None:
        invalid = deepcopy(self.identity)
        invalid["care"]["boundaries"] = []

        with self.assertRaisesRegex(BenchmarkProfileError, "boundaries.*at least 1"):
            parse_transition_request(
                {
                    "protocol_version": 1,
                    "operation": "apply_transition",
                    "profile_id": "profile-a",
                    "probe_id": "update-a",
                    "transition": {
                        "type": "replace-agent-identity",
                        "agent_identity": invalid,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
