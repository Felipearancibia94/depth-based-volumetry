"""Keep the ```mermaid blocks in Markdown identical to their .mmd sources.

The source of a diagram is a `.mmd` file in docs/diagrams/mermaid/. A Markdown file embeds it
between two markers; the path is relative to that Markdown file. The block between the markers
is generated: edit the `.mmd`, then run this script.

    <!-- mermaid: diagrams/mermaid/flow.mmd -->
    ```mermaid
    ...copied from the .mmd...
    ```
    <!-- /mermaid -->

    S=.claude/skills/mermaid/scripts
    uv run --no-project python $S/sync.py --all            # rewrite stale blocks
    uv run --no-project python $S/sync.py --check --all    # exit 1 if any block is stale
    uv run --no-project python $S/sync.py docs/some.md     # only these files

`--all` scans README.md, ARCHITECTURE.md, ROADMAP.md and docs/. Run from the repository root.
Standard library only; line endings of each Markdown file are preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKED = re.compile(
    r"(?P<open><!-- mermaid: (?P<src>[^>]+?) -->\n)(?P<body>.*?)(?P<close><!-- /mermaid -->)",
    re.DOTALL,
)
ROOT_DOCS = ("README.md", "ARCHITECTURE.md", "ROADMAP.md")


def render_block(code: str) -> str:
    return "```mermaid\n" + code.strip("\n") + "\n```\n"


def sync_file(path: Path, write: bool) -> tuple[int, int]:
    """Return (blocks, stale). Raises SystemExit on a missing source or a dangling marker."""
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    blocks = stale = 0

    def replace(match: re.Match) -> str:
        nonlocal blocks, stale
        src = (path.parent / match.group("src").strip()).resolve()
        if not src.is_file():
            raise SystemExit(f"{path}: source not found: {match.group('src').strip()}")
        blocks += 1
        body = render_block(src.read_text(encoding="utf-8"))
        if body != match.group("body"):
            stale += 1
        return match.group("open") + body + match.group("close")

    new = MARKED.sub(replace, text)
    opens, closes = text.count("<!-- mermaid:"), text.count("<!-- /mermaid -->")
    if opens != closes or opens != blocks:
        raise SystemExit(f"{path}: unbalanced mermaid markers ({opens} open, {closes} close)")
    if write and new != text:
        path.write_bytes(new.replace("\n", eol).encode("utf-8"))
    return blocks, stale


def discover(repo: Path) -> list[Path]:
    files = [repo / name for name in ROOT_DOCS if (repo / name).is_file()]
    files += sorted((repo / "docs").rglob("*.md"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="scan the project documentation")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if stale")
    args = parser.parse_args()
    repo = Path.cwd()
    if not (repo / "pyproject.toml").exists():
        raise SystemExit("run from the repository root")
    files = discover(repo) if args.all else args.files
    if not files:
        raise SystemExit("give Markdown files or --all")

    total = stale_total = 0
    for path in files:
        blocks, stale = sync_file(path, write=not args.check)
        if blocks:
            state = "stale" if stale and args.check else ("updated" if stale else "in sync")
            print(f"{path}: {blocks} block(s), {state}")
        total += blocks
        stale_total += stale
    label = "stale" if args.check else "updated"
    print(f"{total} embedded diagram(s), {stale_total} {label}")
    if args.check and stale_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
