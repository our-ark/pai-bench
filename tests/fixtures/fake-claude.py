#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


args = sys.argv[1:]
prompt = sys.stdin.read()
sleep = os.environ.get("FAKE_CLAUDE_SLEEP", "").strip()
if sleep:
    time.sleep(float(sleep))
schema = json.loads(args[args.index("--json-schema") + 1])
record = {"args": args, "prompt": prompt, "schema": schema}
log = os.environ.get("FAKE_CLAUDE_LOG", "").strip()
if log:
    Path(log).write_text(json.dumps(record), encoding="utf-8")
attempt_file = os.environ.get("FAKE_CLAUDE_ATTEMPT_FILE", "").strip()
attempt = 1
if attempt_file:
    path = Path(attempt_file)
    attempt = int(path.read_text(encoding="utf-8")) + 1 if path.exists() else 1
    path.write_text(str(attempt), encoding="utf-8")
fail_attempts = int(os.environ.get("FAKE_CLAUDE_FAIL_ATTEMPTS", "0"))
api_error = os.environ.get("FAKE_CLAUDE_API_ERROR", "").strip()
if api_error:
    print(
        json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "api_error": api_error,
            }
        )
    )
    raise SystemExit(1)
if attempt <= fail_attempts:
    print(json.dumps({"type": "result", "is_error": True, "result": "failure"}))
    print("synthetic warning", file=sys.stderr)
    raise SystemExit(1)
score = float(os.environ.get("FAKE_CLAUDE_SCORE", "1"))
print(
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps({"score": score}),
            "structured_output": {"score": score},
            "usage": {
                "input_tokens": 19,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 3,
                "output_tokens": 5,
            },
            "total_cost_usd": 0.0125,
        }
    )
)
