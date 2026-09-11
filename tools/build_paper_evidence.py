#!/usr/bin/env python3
"""Export a manuscript-free evidence asset from an already sanitized snapshot.

This is packaging, not anonymization. Audit the input before publishing the
output. Original response/judgment bytes are retained exactly. No model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ALLOWED = {
    "analysis", "data", "src", "releases", "tests", "specs", "bin",
    "tools", "scripts", "reference-body", "LICENSE", "VERSION", "pyproject.toml",
}
ROOT_NAME = "pai-bench-v1.0.0-paper-evidence"
README = """# PAI-Bench v1.0.0 paper evidence

This asset contains retained, previously sanitized experimental evidence:
synthetic profiles, 1,536 locked-suite target responses, recorded model-judge
outputs, follow-up controls, derived analyses, and the historical benchmark
and reference-body source snapshots used by the review export.

It deliberately excludes the manuscript and methodological supplement. The
paper is distributed separately. No private installed-agent state, Git history,
provider credentials, or raw source-prototype notes are included.

## Offline verification

With Python 3.11 or newer, from this extracted directory:

```sh
python3 scripts/verify_artifact.py
python3 scripts/run_tests.py
```

Verification checks every file checksum, recorded campaign sizes, selected
headline values, component counts, and gate/safety audit summaries. It checks
the retained evidence; it does not establish judge correctness or re-run a
model. Tests use fakes. Two regeneration tests are skipped because raw source
prototype notes were intentionally omitted from the historical export.

## Evidence map

- `data/reports/`: retained target responses and recorded judge outputs.
- `data/experiments/`: experiment specifications.
- `analysis/`: offline audits, follow-up controls, and their scripts.
- `releases/v1.0/data/`: the historical sanitized profiles/bindings/probe suite.
- `src/identity_benchmark/`: historical sanitized benchmark snapshot.
- `src/reference_agent/`: historical sanitized target-body snapshot.
- `EXPORT.json`: source-export checksum, exclusions, and preservation counts.
- `MANIFEST.sha256`: checksums of this asset (excluding the manifest itself).

## Provenance and limitations

The upstream projects are https://github.com/our-ark/pai-bench and
https://github.com/our-ark/enoch. The code is Apache-2.0; see LICENSE and NOTICE.
The release tag identifies this distribution, not a claim that today's source
commit generated historical results. Use this snapshot for historical audits
and the tagged repository for fresh runs and current integrations.

Neutral project/revision labels and redactions already present in the review
export are intentionally retained. These records are not untouched runtime
logs. No response or judgment is regenerated or edited by this exporter;
checksums here identify the published, sanitized bytes. Historical internal
hashes document provenance but are not checksums of those redacted files.
No original identifier has been guessed or silently reconstructed.

Primary Astra selection was post hoc. Original and replayed judge results,
literal-gate sensitivity, and the small prompt-specificity follow-up remain
separate analyses. The asset is not new replication or human validation.
It must not be used to infer that the current vNext suite produced the frozen
paper results. Re-running proprietary targets/judges requires provider access
and need not reproduce identical outputs from evolving services.
"""
NOTICE = """PAI-Bench and sanitized reference-body snapshot
Copyright 2026 Zhenyu Zhao and Roy Zhao / respective upstream contributors.

Upstream benchmark: https://github.com/our-ark/pai-bench
Upstream reference body: https://github.com/our-ark/enoch
Apache License 2.0; original source notices are retained where present.

The reference-body snapshot retains neutral labels from an earlier sanitized
export. This export removes manuscript files, replaces the package README,
adds provenance, and removes manuscript-only checks from the verifier.
Other included file bytes, including target responses and judgments, are
unchanged relative to the supplied sanitized export.
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(source: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    excluded = []
    original = {}
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) < 2:
                raise ValueError("Unsafe input archive path")
            relative = PurePosixPath(*name.parts[1:])
            if relative.parts[0] not in ALLOWED:
                excluded.append(str(relative))
                continue
            if str(relative) in files:
                raise ValueError("Duplicate input path")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not accepted")
            data = archive.read(entry)
            files[str(relative)] = data
            original[str(relative)] = sha(data)

    key = "scripts/verify_artifact.py"
    verifier = files[key].decode("utf-8")
    start = verifier.index('    paper = (ROOT / "manuscript/paper.tex")')
    end = verifier.index('    numbers = load(', start)
    verifier = verifier[:start] + verifier[end:]
    verifier = verifier.replace("manifest, anonymous manuscript markers,", "manifest, retained evidence,")
    files[key] = verifier.encode("utf-8")
    files["README.md"] = README.encode()
    files["NOTICE"] = NOTICE.encode()
    preserved = {name: digest for name, digest in original.items() if name != key}
    assert all(sha(files[name]) == digest for name, digest in preserved.items())
    files["EXPORT.json"] = (json.dumps({
        "format_version": 1, "release": "v1.0.0", "prepared": "2026-09-11",
        "input_sanitized_export_sha256": sha(source.read_bytes()),
        "excluded_paths": sorted(excluded),
        "modified_existing_files": [key],
        "added_files": ["README.md", "NOTICE", "EXPORT.json", "MANIFEST.sha256"],
        "preserved_file_count": len(preserved),
        "preserved_sha256": preserved,
        "target_or_judge_calls": 0,
        "privacy_scan_required": True,
    }, indent=2, sort_keys=True) + "\n").encode()
    files["MANIFEST.sha256"] = "".join(
        f"{sha(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    return files


def build(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("Refusing to replace an existing evidence asset")
    files = collect(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{ROOT_NAME}/{name}", date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return {"files": len(files), "sha256": sha(output.read_bytes()), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanitized_snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.sanitized_snapshot, args.output), indent=2))
