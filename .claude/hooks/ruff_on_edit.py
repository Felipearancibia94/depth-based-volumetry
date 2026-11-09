"""PostToolUse hook: format and lint a Python file right after Claude edits it.

Reads the hook JSON from stdin. Runs ``ruff format`` then ``ruff check`` on the edited
``.py`` file using the project's own ruff (``.venv`` first, then ``PATH``). Remaining lint
errors are written to stderr with exit code 2 so Claude sees and fixes them. Anything that
prevents the check (not a Python file, no ruff, outside the project) exits 0 silently.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_ruff(root: Path) -> str | None:
    for candidate in (root / ".venv" / "Scripts" / "ruff.exe", root / ".venv" / "bin" / "ruff"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ruff")


def main() -> int:
    try:
        data = json.load(sys.stdin)
        file_path = data["tool_input"]["file_path"]
    except (ValueError, KeyError, TypeError):
        return 0

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".").resolve()
    path = Path(file_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.suffix != ".py" or not path.is_file():
        return 0
    try:
        path.relative_to(root)
    except ValueError:
        return 0

    ruff = find_ruff(root)
    if ruff is None:
        return 0

    try:
        subprocess.run([ruff, "format", str(path)], cwd=root, capture_output=True, timeout=60)
        check = subprocess.run(
            [ruff, "check", str(path)], cwd=root, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0

    if check.returncode != 0:
        sys.stderr.write(
            f"ruff check reports problems in {path.name}:\n{check.stdout}{check.stderr}"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
