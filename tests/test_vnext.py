from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_benchmark.authorization import is_authorized
from identity_benchmark.vnext import (
    COMPOSITION_COMPONENTS,
    NEUTRAL_COMPONENTS,
    PROBES_PER_PROFILE,
    generate_vnext,
    write_vnext,
)
from identity_benchmark.probe_suites import (
    compile_benchmark_profile,
    parse_identity_profile,
    parse_probe_bindings,
    parse_probe_suite,
)


class VNextDevelopmentSuiteTests(unittest.TestCase):
    def test_generator_is_reproducible_and_keeps_v1_separate(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "vnext"
            paths = write_vnext(output)
            verified = write_vnext(output, check=True)
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))

        self.assertEqual(paths, verified)
        self.assertEqual(len(paths), 20)
        self.assertEqual(index["status"], "development-not-frozen")
        self.assertEqual(len(index["profiles"]), 8)
        self.assertTrue(
            all("releases/v1.0" not in str(path) for path in paths)
        )

    def test_composition_order_is_balanced_across_matched_pairs(self) -> None:
        outputs = generate_vnext(Path("vnext"))
        index = json.loads(outputs[Path("vnext/index.json")])
        orders = [
            tuple(entry["composition_order"])
            for entry in index["profiles"][::2]
        ]

        self.assertEqual(len(set(orders)), 4)
        self.assertEqual(
            {order[0] for order in orders},
            set(COMPOSITION_COMPONENTS),
        )
        for left, right in index["counterfactual_pairs"]:
            left_entry = next(
                entry for entry in index["profiles"] if entry["profile_id"] == left
            )
            right_entry = next(
                entry for entry in index["profiles"] if entry["profile_id"] == right
            )
            self.assertEqual(
                left_entry["composition_order"],
                right_entry["composition_order"],
            )

    def test_compiled_profiles_cross_role_with_credential_validity(self) -> None:
        root = Path("vnext")
        outputs = generate_vnext(root)
        profile = _compiled_profile(
            outputs,
            root,
            "population-p002-a-publication-v4",
        )
        by_id = {probe.id: probe for probe in profile.probes}

        self.assertEqual(len(profile.probes), PROBES_PER_PROFILE)
        self.assertEqual(
            {
                (probe.messages[0].role, probe.before_response.expected_acceptance)
                for probe_id, probe in by_id.items()
                if probe_id.startswith("governance-factorial-")
                and probe.before_response is not None
            },
            {
                ("user", True),
                ("user", False),
                ("system", True),
                ("system", False),
            },
        )
        for probe_id, probe in by_id.items():
            if not probe_id.startswith("governance-factorial-"):
                continue
            assert probe.before_response is not None
            self.assertEqual(
                is_authorized(
                    profile.profile_id,
                    probe.before_response.authorization,
                ),
                probe.before_response.expected_acceptance,
            )

    def test_atomic_recall_covers_every_composition_component(self) -> None:
        root = Path("vnext")
        outputs = generate_vnext(root)
        profile = _compiled_profile(
            outputs,
            root,
            "population-p002-a-publication-v4",
        )
        atomic = {
            probe.id: probe
            for probe in profile.probes
            if "atomic-recall" in probe.tags
        }

        self.assertEqual(
            set(atomic),
            {f"atomic-recall-{name}" for name in COMPOSITION_COMPONENTS},
        )
        for probe in atomic.values():
            self.assertEqual(probe.dimension, "recognition")
            components = [
                expectation.component
                for expectation in probe.expectations
                if expectation.component
            ]
            self.assertEqual(len(components), 1)
            self.assertTrue(components[0].startswith("atomic_"))

    def test_neutral_controls_match_identity_composition_depth_and_order(self) -> None:
        root = Path("vnext")
        outputs = generate_vnext(root)
        profile = _compiled_profile(
            outputs,
            root,
            "population-p002-a-publication-v4",
        )
        by_id = {probe.id: probe for probe in profile.probes}

        for depth in range(1, 5):
            identity_probe = by_id[f"composition-depth-{depth}"]
            neutral_probe = by_id[f"neutral-composition-depth-{depth}"]
            identity_components = [
                expectation.component
                for expectation in identity_probe.expectations
                if expectation.component
            ]
            neutral_components = [
                expectation.component
                for expectation in neutral_probe.expectations
                if expectation.component
            ]
            self.assertEqual(len(identity_components), depth)
            self.assertEqual(
                neutral_components,
                [
                    str(NEUTRAL_COMPONENTS[name]["component"])
                    for name in identity_components
                ],
            )
            self.assertEqual(neutral_probe.dimension, "capability")
            self.assertNotIn("open-generation", neutral_probe.tags)
            self.assertEqual(
                [message.role for message in neutral_probe.messages],
                ["system", "user"],
            )

        capability = [
            probe for probe in profile.probes if probe.dimension == "capability"
        ]
        self.assertEqual(len(capability), 5)

    def test_counterfactual_pairs_receive_identical_probe_messages(self) -> None:
        root = Path("vnext")
        outputs = generate_vnext(root)
        index = json.loads(outputs[root / "index.json"])
        profiles = {
            entry["profile_id"]: _compiled_profile(
                outputs, root, entry["profile_id"]
            ).to_dict()
            for entry in index["profiles"]
        }
        for left, right in index["counterfactual_pairs"]:
            left_probes = {
                probe["id"]: probe for probe in profiles[left]["probes"]
            }
            right_probes = {
                probe["id"]: probe for probe in profiles[right]["probes"]
            }
            for probe_id, left_probe in left_probes.items():
                if "counterfactual" not in left_probe["tags"]:
                    continue
                self.assertEqual(
                    left_probe["messages"],
                    right_probes[probe_id]["messages"],
                )


def _compiled_profile(outputs: dict[Path, str], root: Path, profile_id: str):
    return compile_benchmark_profile(
        parse_identity_profile(
            json.loads(outputs[root / f"identities/{profile_id}.json"])
        ),
        parse_probe_suite(json.loads(outputs[root / "probe-suite.json"])),
        parse_probe_bindings(
            json.loads(outputs[root / f"bindings/{profile_id}.json"])
        ),
    )


if __name__ == "__main__":
    unittest.main()
