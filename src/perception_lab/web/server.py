"""Local dashboard: status, live view, event timeline with evidence, search and review.

Standard library only. It binds to loopback by default and has no authentication, so
binding elsewhere requires ``web.allow_remote``. Requests must carry a loopback ``Host``
header (protection against DNS rebinding) and writes must be JSON (a cross-site form
cannot send that without a CORS preflight, which this server never approves).
"""

from __future__ import annotations

import ipaddress
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from perception_lab.core.config import UNKNOWN_PET
from perception_lab.spatial.scene import SceneCalibration
from perception_lab.store.db import EventStore
from perception_lab.store.media import resolve_media_path
from perception_lab.web.page import INDEX_HTML


@dataclass
class DashboardState:
    store: EventStore
    media_dir: str
    scene: SceneCalibration | None = None
    pipeline: Any = None  # PetPipeline when running live
    pet_names: tuple[str, ...] = ()  # configured household pets, for the dashboard filter
    allowed_hosts: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "lab-dashboard"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            pass  # keep the console for notifications

        # -- helpers ------------------------------------------------------------------

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            name = urlparse(f"//{host}").hostname or ""
            return name in state.allowed_hosts or _is_loopback(name)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        # -- routes -------------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = url.path
            if path == "/":
                return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            if path == "/api/status":
                pipeline = state.pipeline
                return self._json(
                    {
                        "events_total": state.store.count(),
                        "pipeline": pipeline.status.to_dict() if pipeline is not None else None,
                        "live": pipeline is not None,
                    }
                )
            if path == "/api/events":
                limit = _int(query, "limit", 50, 1, 500)
                offset = _int(query, "offset", 0, 0, 10**9)
                rule = query.get("rule", [None])[0]
                pet = query.get("pet", [None])[0]
                events = state.store.list(limit=limit, offset=offset, rule=rule, pet=pet)
                return self._json([e.to_dict() for e in events])
            if path == "/api/pets":
                return self._json({"pets": list(state.pet_names)})
            if path == "/api/search":
                q = query.get("q", [""])[0]
                return self._json([e.to_dict() for e in state.store.search(q, limit=100)])
            if path.startswith("/api/events/"):
                event = state.store.get(path.removeprefix("/api/events/"))
                if event is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such event")
                return self._json(event.to_dict())
            if path == "/api/zones":
                zones = state.scene.zones if state.scene is not None else ()
                return self._json(
                    [
                        {"name": z.name, "tags": list(z.tags), "polygon_xy_px": z.polygon_xy_px}
                        for z in zones
                    ]
                )
            if path == "/api/live.jpg":
                jpeg = getattr(state.pipeline, "live_jpeg", None)
                if not jpeg:
                    return self._error(HTTPStatus.NOT_FOUND, "no live view")
                return self._send(200, jpeg, "image/jpeg")
            if path.startswith("/media/"):
                target = resolve_media_path(state.media_dir, path.removeprefix("/media/"))
                if target is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such file")
                return self._send(200, Path(target).read_bytes(), "image/jpeg")
            return self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
            path = urlparse(self.path).path
            if not (path.startswith("/api/events/") and path.endswith("/review")):
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            event_id = path.removeprefix("/api/events/").removesuffix("/review")
            length = min(int(self.headers.get("Content-Length", 0) or 0), 10_000)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                label = body.get("label")
                if label is not None and (not isinstance(label, str) or len(label) > 100):
                    raise ValueError("label must be a short string")
                pet = body.get("pet")
                allowed = (*state.pet_names, UNKNOWN_PET)
                if pet is not None and pet not in allowed:
                    raise ValueError(f"pet must be one of {', '.join(allowed)}")
                event = state.store.set_review(event_id, body.get("status", ""), label, pet)
            except KeyError:
                return self._error(HTTPStatus.NOT_FOUND, "no such event")
            except (ValueError, AttributeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return self._json(event.to_dict())

    return Handler


def _int(query: dict[str, list[str]], name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(query.get(name, [default])[0])
    except ValueError:
        return default
    return max(low, min(high, value))


def start_server(
    state: DashboardState, host: str = "127.0.0.1", port: int = 8000, *, allow_remote: bool = False
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    if not _is_loopback(host) and not allow_remote:
        raise ValueError(
            f"refusing to serve the dashboard on {host!r}: it has no authentication; "
            "set web.allow_remote = true only on a trusted network"
        )
    if allow_remote:
        state.allowed_hosts = state.allowed_hosts | {host}
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="dashboard", daemon=True)
    thread.start()
    return server, thread
