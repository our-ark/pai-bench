from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "paper_evidence_export", Path(__file__).resolve().parents[1] / "tools/build_paper_evidence.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class PaperEvidenceExportTests(unittest.TestCase):
    def make_snapshot(self, root: Path, extra: dict[str, bytes] | None = None) -> Path:
        path = root / "input.zip"
        verifier = '''def main():
    paper = (ROOT / "manuscript/paper.tex").read_text()
    assert "Anonymous" in paper
    numbers = load("numbers.json")
    print("manifest, anonymous manuscript markers,")
'''
        files = {
            "scripts/verify_artifact.py": verifier.encode(),
            "data/reports/case.json": b'{"response":"synthetic answer","score":0.5}\n',
            "manuscript/paper.tex": b"private manuscript",
            "SUPPLEMENT.md": b"private supplement",
            ".git/config": b"private git metadata",
            "README.md": b"old README",
            "LICENSE": b"Apache-2.0",
        }
        files.update(extra or {})
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in files.items():
                archive.writestr("snapshot/" + name, value)
        return path

    def test_whitelist_and_byte_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_snapshot(Path(temp))
            files = EXPORT.collect(source)
            self.assertNotIn("SUPPLEMENT.md", files)
            self.assertFalse(any(p.startswith(("manuscript/", ".git/")) for p in files))
            expected = b'{"response":"synthetic answer","score":0.5}\n'
            self.assertEqual(files["data/reports/case.json"], expected)
            self.assertNotIn(b"manuscript/paper.tex", files["scripts/verify_artifact.py"])
            report = json.loads(files["EXPORT.json"])
            self.assertEqual(report["target_or_judge_calls"], 0)
            self.assertEqual(report["preserved_sha256"]["data/reports/case.json"],
                             hashlib.sha256(expected).hexdigest())
            for entry in files["MANIFEST.sha256"].decode().splitlines():
                digest, name = entry.split("  ", 1)
                self.assertEqual(hashlib.sha256(files[name]).hexdigest(), digest)

    def test_reproducible_archive_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_snapshot(root)
            a, b = root / "a.zip", root / "b.zip"
            first, second = EXPORT.build(source, a), EXPORT.build(source, b)
            self.assertEqual(first, second)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with self.assertRaises(FileExistsError):
                EXPORT.build(source, a)

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_snapshot(Path(temp), {"../data/escape": b"no"})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                EXPORT.collect(source)

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_snapshot(Path(temp))
            with zipfile.ZipFile(source, "a") as archive:
                entry = zipfile.ZipInfo("snapshot/data/link")
                entry.external_attr = 0o120777 << 16
                archive.writestr(entry, "/private/state")
            with self.assertRaisesRegex(ValueError, "Symlinks"):
                EXPORT.collect(source)


if __name__ == "__main__":
    unittest.main()
