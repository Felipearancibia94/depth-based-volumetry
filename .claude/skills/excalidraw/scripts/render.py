"""Render an .excalidraw file to a PNG next to it, using a browser.

    S=.claude/skills/excalidraw/scripts
    uv run --no-project python $S/render.py docs/diagrams/roadmap.excalidraw

Serves a small page on 127.0.0.1 (free port, printed as `open: <url>`). Open that URL in a
browser (Claude: the built-in browser pane); the page draws the diagram with the Excalidraw
hand-drawn font and posts the PNG back. The server writes `<name>.png` and exits. Text is baked
into the PNG because GitHub does not render fonts embedded in Excalidraw SVG exports.
The page fetches the Virgil font from cdn.jsdelivr.net; without network it falls back to a
sans-serif font.
"""

from __future__ import annotations

import base64
import http.server
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: render.py <file.excalidraw>")
    src = Path(sys.argv[1]).resolve()
    out = src.with_suffix(".png")
    page = (HERE / "render.html").read_bytes()
    doc = src.read_bytes()
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/doc"):
                self._send(doc, "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            data = self.rfile.read(int(self.headers["Content-Length"])).decode()
            out.write_bytes(base64.b64decode(data.split(",", 1)[1]))
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
        raise SystemExit(f"no PNG received within {TIMEOUT_S} s; nothing written")
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
