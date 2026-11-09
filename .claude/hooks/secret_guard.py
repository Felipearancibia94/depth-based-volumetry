"""PreToolUse hook: block secrets and non-loopback endpoints in files Claude writes.

Reads the hook JSON from stdin and inspects the content of an Edit or Write call.
- Secret patterns are blocked in every file except Markdown (docs may show examples).
- URLs whose host is not loopback are blocked in ``src/``, ``examples/`` and ``*.toml``
  (project rule: no non-loopback endpoints in the repository). Tests are not checked because
  they exercise the refusal of remote hosts on purpose.
Exit code 2 blocks the call and shows stderr to Claude; anything else lets it through.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "private key": re.compile("-----BEGIN [A-Z ]*PRIV" + "ATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "API key (sk-)": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
URL = re.compile(r"https?://([^/\s\"'<>{}:@]+)(?::\d+)?", re.IGNORECASE)
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
ENDPOINT_DIRS = ("src", "examples")


def content_of(tool_input: dict) -> str:
    return tool_input.get("content") or tool_input.get("new_string") or ""


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool_input = data["tool_input"]
        file_path = tool_input["file_path"]
    except (ValueError, KeyError, TypeError):
        return 0

    text = content_of(tool_input)
    if not text:
        return 0

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".").resolve()
    path = Path(file_path)
    if not path.is_absolute():
        path = root / path
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return 0
    if rel.parts[:1] == (".claude",):
        return 0

    problems: list[str] = []
    if path.suffix.lower() != ".md":
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"looks like a {name}")

    if rel.parts[:1] and (rel.parts[0] in ENDPOINT_DIRS or path.suffix.lower() == ".toml"):
        hosts = {m.group(1).lower() for m in URL.finditer(text)}
        remote = sorted(h for h in hosts if h not in LOOPBACK)
        if remote:
            problems.append(f"non-loopback endpoint(s): {', '.join(remote)}")

    if problems:
        sys.stderr.write(
            f"Blocked write to {rel.as_posix()}: " + "; ".join(problems) + ".\n"
            "Project rule: no credentials and no non-loopback endpoints in the repository. "
            "Use a loopback address or a placeholder read from the environment or user config.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
