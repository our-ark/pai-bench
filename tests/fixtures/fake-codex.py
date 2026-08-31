#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


args = sys.argv[1:]
prompt = sys.stdin.read()
sleep = os.environ.get("FAKE_CODEX_SLEEP", "").strip()
if sleep:
    time.sleep(float(sleep))
output = Path(args[args.index("--output-last-message") + 1])
schema_index = args.index("--output-schema") + 1 if "--output-schema" in args else None
record = {"args": args, "prompt": prompt}
if schema_index is not None:
    schema = Path(args[schema_index])
    record["schema"] = json.loads(schema.read_text(encoding="utf-8"))
log = os.environ.get("FAKE_CODEX_LOG", "").strip()
if log:
    Path(log).write_text(json.dumps(record), encoding="utf-8")
attempt_file = os.environ.get("FAKE_CODEX_ATTEMPT_FILE", "").strip()
attempt = 1
if attempt_file:
    path = Path(attempt_file)
    attempt = int(path.read_text(encoding="utf-8")) + 1 if path.exists() else 1
    path.write_text(str(attempt), encoding="utf-8")
fail_attempts = int(os.environ.get("FAKE_CODEX_FAIL_ATTEMPTS", "0"))
if attempt <= fail_attempts:
    print(json.dumps({"type": "error", "message": "synthetic transient failure"}))
    print("synthetic warning", file=sys.stderr)
    raise SystemExit(1)
score = os.environ.get("FAKE_CODEX_SCORE", "").strip()
output.write_text(
    json.dumps({"score": float(score)}) if score else "synthetic target response",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 21,
                "cached_input_tokens": 5,
                "output_tokens": 4,
                "reasoning_output_tokens": 3,
            },
        }
    )
)
