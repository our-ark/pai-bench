from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("specificity", ROOT / "tools/run_composition_specificity.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CompositionSpecificityTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads((ROOT / "releases/v1.0/data/probe-suite.json").read_text())

    def test_only_the_requested_element_clause_changes(self):
        prompts = module.build_prompts(self.suite)
        self.assertEqual(len(prompts), 6)
        self.assertEqual(prompts[module.PORTRAITS[0]].replace(module.IMPLICIT_CLAUSE, module.EXPLICIT_CLAUSE), prompts[module.PORTRAITS[1]])
        for probe in module.PORTRAITS:
            self.assertIn("in four sentences", prompts[probe])
            self.assertNotIn("80 words", prompts[probe])
            self.assertIn("Do not discuss identity layers", prompts[probe])

    def test_missing_frozen_clause_fails_closed(self):
        suite = deepcopy(self.suite)
        next(p for p in suite["probes"] if p["id"] == module.PORTRAITS[0])["messages"][0]["content"] = "Changed"
        with self.assertRaises(ValueError):
            module.build_prompts(suite)

    def test_joint_identifiers_do_not_require_exact_mission(self):
        expected = {"designation": "SELF-A", "parent": "PARENT-B", "collaborator": "ALLY-C", "mission": "Canonical mission"}
        observed = module.observe("self-a, PARENT-B, ally-c. I help people cooperate.", expected)
        self.assertTrue(observed["all_three_identifiers"])
        self.assertFalse(observed["mission_exact_diagnostic_only"])
        self.assertFalse(module.observe("SELF-A ALLY-C", expected)["all_three_identifiers"])

    def test_normalization_and_body_word_boundary(self):
        expected = dict.fromkeys(module.ATOMIC, "Ａ  B")
        observed = module.observe("a\n b Enochian", expected)
        self.assertTrue(observed["all_three_identifiers"])
        self.assertFalse(observed["body_label_present"])

    def test_incomplete_analysis_never_publishes_full_aggregate(self):
        report = module.analyze({"cases": [{"case_id": "missing"}]}, {})
        self.assertEqual(report["completed"], 0)
        self.assertNotIn("atomic", report)

    def test_analysis_is_paired_and_has_no_semantic_grade(self):
        identity = {"identity": {"names": {"canonical": "SELF-A"}}, "origin": {"lineage": ["PARENT-B", "SELF-A"]}, "relationships": [{"address_as": "ALLY-C"}], "mission": {"statement": "Canonical mission"}}
        cases = [{"case_id": p, "profile_id": "one", "probe_id": p, "identity": identity} for p in (*module.ATOMIC.values(), *module.PORTRAITS)]
        manifest = {"profiles": [{"profile_id": "one"}], "cases": cases}
        records = {c["case_id"]: {"response": "SELF-A PARENT-B ALLY-C", "error": ""} for c in cases}
        records[module.PORTRAITS[0]]["response"] = "SELF-A ALLY-C"
        report = module.analyze(manifest, records)
        self.assertEqual(report["atomic"]["parent"], 1)
        self.assertEqual(report["paired"]["parent"]["explicit_only"], 1)
        self.assertEqual(report[module.PORTRAITS[1]]["joint_three"], 1)
        self.assertNotIn("score", report)


if __name__ == "__main__":
    unittest.main()
