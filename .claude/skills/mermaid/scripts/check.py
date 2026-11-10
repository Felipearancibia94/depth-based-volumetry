"""Check Mermaid sources by parsing and rendering them in a browser.

Arguments are `.mmd` files (one diagram each) and/or Markdown files (every ```mermaid block).

    S=.claude/skills/mermaid/scripts
    uv run --no-project python $S/check.py docs/diagrams/mermaid/*.mmd

Serves a small page on 127.0.0.1 (free port, printed as `open: <url>`). Open that URL in a
browser (Claude: the built-in browser pane); the page loads Mermaid, runs `mermaid.parse` and
`mermaid.render` on every block and posts the results back. The script prints one line per
block (`file:line OK` or `file:line ERROR message`), then exits 0 if all blocks pass and 1
otherwise. The page fetches Mermaid from cdn.jsdelivr.net, so it needs network access.
A pass means the pinned Mermaid version accepts the block; GitHub may run another version.
"""

from __future__ import annotations

import http.server
import json
import re
import sys
import textwrap
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*mermaid\b")


def extract(path: Path) -> list[dict]:
    """Mermaid blocks of one file: file, 1-based line of the first code line, code.

    A `.mmd` / `.mermaid` file is a single block; anything else is searched for fences."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".mmd", ".mermaid"):
        return [{"file": path.as_posix(), "line": 1, "code": text}]
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        match = FENCE.match(lines[i])
        if match:
            fence = match.group(1)
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if len(s) >= len(fence) and set(s) == {fence[0]}:
                    break
                j += 1
            code = textwrap.dedent("\n".join(lines[i + 1 : j]))
            blocks.append({"file": path.as_posix(), "line": i + 2, "code": code})
            i = j
        i += 1
    return blocks


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: check.py <file.md> [more.md ...]")
    blocks: list[dict] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.is_file():
            raise SystemExit(f"not a file: {arg}")
        blocks.extend(extract(path))
    if not blocks:
        print("no mermaid blocks found")
        return

    page = (HERE / "check.html").read_bytes()
    payload = json.dumps(blocks).encode()
    results: list[list[dict]] = []
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/blocks"):
                self._send(payload, "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            results.append(json.loads(body))
            self._send(b"ok", "text/plain")
            done.set()

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"open: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    ok = done.wait(TIMEOUT_S)
    server.shutdown()
    if not ok:
        raise SystemExit(f"no result received within {TIMEOUT_S} s; nothing checked")

    failed = 0
    for r in results[0]:
        if r["ok"]:
            print(f"{r['file']}:{r['line']} OK ({r['type']})")
        else:
            failed += 1
            print(f"{r['file']}:{r['line']} ERROR {r['message']}")
    print(f"{len(results[0])} block(s), {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
