#!/usr/bin/env python3
"""Offline target CLI fixture for the real Enoch runtime-provider contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


args = sys.argv[1:]
prompt = sys.stdin.read()
log = Path(os.environ["PAI_FAKE_RUNTIME_LOG"])
provider = "claude" if "--print" in args else "codex"
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"provider": provider, "args": args, "prompt": prompt}) + "\n")
if provider == "claude":
    print(json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "synthetic target response", "session_id": "offline-claude-session",
        "usage": {"input_tokens": 21, "output_tokens": 4},
    }))
else:
    output = Path(args[args.index("--output-last-message") + 1])
    output.write_text("synthetic target response", encoding="utf-8")
    print(json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 21, "output_tokens": 4},
    }))
