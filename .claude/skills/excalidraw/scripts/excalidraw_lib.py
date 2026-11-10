"""Small standard-library helper to write Excalidraw files from Python specs.

A spec is a Python file under docs/diagrams/src/ that defines:

    OUTPUT = "docs/diagrams/<name>"        # without extension
    def build(d: Diagram, repo: Path) -> None: ...

`build.py` runs it and writes `<OUTPUT>.excalidraw`; `render.py` turns that into a PNG.
Output is deterministic (fixed seeds) so diffs stay readable.
"""

from __future__ import annotations

import base64
import json
import struct
import textwrap
from pathlib import Path

CHAR_W = 0.58  # approximate Virgil glyph width / font size
LINE_H = 1.25

# (background, stroke) pairs. Use the meaning in the legend of each diagram.
GREEN = ("#b2f2bb", "#2f9e44")  # done
BLUE = ("#a5d8ff", "#1971c2")  # measured on real hardware / infrastructure
ORANGE = ("#ffd8a8", "#e8590c")  # open or needs validation
GREY = ("#e9ecef", "#868e96")  # later / not tested
PURPLE = ("#d0bfff", "#6741d9")  # decision / VLM
RED = ("#ffc9c9", "#e03131")  # known limit
MUTED = "#555555"


def text_size(text: str, font_size: float) -> tuple[float, float]:
    lines = text.split("\n")
    return max(len(line) for line in lines) * font_size * CHAR_W, len(lines) * font_size * LINE_H


def wrap(text: str, width_px: float, font_size: float) -> str:
    """Wrap each paragraph so it fits inside a box `width_px` wide."""
    chars = max(8, int((width_px - 24) / (font_size * CHAR_W)))
    return "\n".join(textwrap.fill(p, chars) if p else "" for p in text.split("\n"))


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) of a PNG or JPEG, without third-party libraries."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data):
            marker, length = data[i + 1], struct.unpack(">H", data[i + 2 : i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return w, h
            i += 2 + length
    raise ValueError("only PNG and JPEG images are supported")


class Diagram:
    def __init__(self, seed: int = 1000) -> None:
        self.elements: list[dict] = []
        self.files: dict[str, dict] = {}
        self._seed = seed

    # -- primitives -----------------------------------------------------------------------
    def _next(self) -> int:
        self._seed += 1
        return self._seed

    def _base(self, eid: str, etype: str, x, y, w, h, **kw) -> dict:
        e = {
            "id": eid, "type": etype, "x": x, "y": y, "width": w, "height": h, "angle": 0,
            "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
            "groupIds": [], "frameId": None, "roundness": None, "seed": self._next(),
            "version": 1, "versionNonce": self._next(), "isDeleted": False,
            "boundElements": None, "updated": 1, "link": None, "locked": False,
        }  # fmt: skip
        e.update(kw)
        return e

    def find(self, eid: str) -> dict:
        return next(e for e in self.elements if e["id"] == eid)

    def label(self, eid, x, y, text, fs=18, color="#1e1e1e") -> None:
        w, h = text_size(text, fs)
        self.elements.append(
            self._base(eid, "text", x, y, w, h, strokeColor=color, text=text, originalText=text,
                       fontSize=fs, fontFamily=1, textAlign="left", verticalAlign="top",
                       containerId=None, autoResize=True, lineHeight=LINE_H)
        )  # fmt: skip

    def box(self, eid, x, y, w, h, text, colors, fs=16, dashed=False, stroke_width=2) -> None:
        bg, stroke = colors
        rect = self._base(
            eid, "rectangle", x, y, w, h, strokeColor=stroke, backgroundColor=bg,
            roundness={"type": 3}, strokeStyle="dashed" if dashed else "solid",
            boundElements=[{"id": eid + "_t", "type": "text"}], strokeWidth=stroke_width,
        )  # fmt: skip
        tw, th = text_size(text, fs)
        txt = self._base(
            eid + "_t", "text", x + (w - tw) / 2, y + (h - th) / 2, tw, th, text=text,
            originalText=text, fontSize=fs, fontFamily=1, textAlign="center",
            verticalAlign="middle", containerId=eid, autoResize=True, lineHeight=LINE_H,
        )  # fmt: skip
        self.elements.extend([rect, txt])

    def zone(self, eid, x, y, w, h, colors, opacity=25) -> None:
        bg, stroke = colors
        self.elements.append(
            self._base(eid, "rectangle", x, y, w, h, strokeColor=stroke, backgroundColor=bg,
                       opacity=opacity, roundness={"type": 3}, strokeWidth=1)
        )  # fmt: skip

    def ellipse(self, eid, x, y, w, h, bg, stroke="#1e1e1e", stroke_width=2) -> None:
        self.elements.append(
            self._base(eid, "ellipse", x, y, w, h, strokeColor=stroke, backgroundColor=bg,
                       strokeWidth=stroke_width, roundness={"type": 2})
        )  # fmt: skip

    def poly(self, eid, points, bg="transparent", stroke="#1e1e1e", closed=True, stroke_width=2):
        x0, y0 = points[0]
        pts = [[px - x0, py - y0] for px, py in points]
        if closed:
            pts.append(pts[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.elements.append(
            self._base(eid, "line", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
                       strokeColor=stroke, backgroundColor=bg if closed else "transparent",
                       strokeWidth=stroke_width, points=pts, lastCommittedPoint=None,
                       startBinding=None, endBinding=None, startArrowhead=None,
                       endArrowhead=None, polygon=closed)
        )  # fmt: skip

    def image(self, eid, path: Path, x, y, scale=1.0, frame=True) -> tuple[float, float]:
        """Embed a PNG/JPEG. Only use images that may be published (see CLAUDE.md)."""
        data = path.read_bytes()
        w, h = image_size(data)
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        fid = eid + "_file"
        self.files[fid] = {
            "mimeType": mime, "id": fid, "created": 1, "lastRetrieved": 1,
            "dataURL": f"data:{mime};base64," + base64.b64encode(data).decode(),
        }  # fmt: skip
        self.elements.append(
            self._base(eid, "image", x, y, w * scale, h * scale, fileId=fid, status="saved",
                       scale=[1, 1], crop=None, strokeColor="transparent")
        )  # fmt: skip
        if frame:
            self.elements.append(
                self._base(eid + "_frame", "rectangle", x, y, w * scale, h * scale,
                           strokeColor="#868e96", strokeWidth=1, roundness={"type": 3})
            )  # fmt: skip
        return w * scale, h * scale

    def arrow(self, aid, src, dst, dashed=False, text=None, elbow_y=None, stroke_width=2):
        """Arrow from `src` to `dst`: right edge to left edge when `dst` is to the right,
        else bottom to top. With `elbow_y`: down, across at that y, then down."""
        a, b = self.find(src), self.find(dst)
        if elbow_y is not None:
            sx, sy = a["x"] + a["width"] / 2, a["y"] + a["height"]
            ex, ey = b["x"] + b["width"] / 2, b["y"]
            pts = [[0, 0], [0, elbow_y - sy], [ex - sx, elbow_y - sy], [ex - sx, ey - sy]]
        else:
            sx, sy = a["x"] + a["width"], a["y"] + a["height"] / 2
            ex, ey = b["x"], b["y"] + b["height"] / 2
            if ex <= sx:
                sx, sy = a["x"] + a["width"] / 2, a["y"] + a["height"]
                ex, ey = b["x"] + b["width"] / 2, b["y"]
            pts = [[0, 0], [ex - sx, ey - sy]]
        self.elements.append(
            self._base(aid, "arrow", sx, sy, abs(ex - sx), ey - sy, roundness={"type": 2},
                       strokeStyle="dashed" if dashed else "solid", points=pts,
                       lastCommittedPoint=None,
                       startBinding={"elementId": src, "focus": 0, "gap": 1},
                       endBinding={"elementId": dst, "focus": 0, "gap": 1},
                       startArrowhead=None, endArrowhead="arrow", elbowed=False,
                       strokeWidth=stroke_width)
        )  # fmt: skip
        for e in (a, b):
            e["boundElements"] = (e["boundElements"] or []) + [{"id": aid, "type": "arrow"}]
        if text:
            self.label(aid + "_l", (sx + ex) / 2 - 40, (sy + ey) / 2 - 24, text, 14, MUTED)

    # -- layouts --------------------------------------------------------------------------
    def column(self, prefix, x, y, w, rows, fs=15, gap=12) -> float:
        """Stack auto-sized boxes of wrapped text; rows are (text, colors). Returns next y.
        Boxes coloured GREY are dashed."""
        for i, (text, colors) in enumerate(rows):
            t = wrap(text, w, fs)
            h = (t.count("\n") + 1) * fs * LINE_H + 26
            self.box(f"{prefix}{i}", x, y, w, h, t, colors, fs, dashed=colors == GREY)
            y += h + gap
        return y

    def row(self, prefix, x, y, w, h, step, rows, fs=17, arrows=True) -> list[str]:
        """Boxes left to right, `step` apart, optionally joined by arrows. Returns ids."""
        ids = []
        for i, (text, colors) in enumerate(rows):
            eid = f"{prefix}{i}"
            self.box(eid, x + i * step, y, w, h, text, colors, fs)
            ids.append(eid)
        if arrows:
            for a, b in zip(ids, ids[1:], strict=False):
                self.arrow(f"a_{a}_{b}", a, b)
        return ids

    def header(self, title, legend=None) -> None:
        self.label("title", 0, 0, title, 30)
        if legend:
            self.label("legend", 0, 42, legend, 16, MUTED)

    # -- output ---------------------------------------------------------------------------
    def check(self) -> None:
        ids = [e["id"] for e in self.elements]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate element ids: {sorted(dupes)}")
        known = set(ids)
        for e in self.elements:
            for ref in e.get("boundElements") or []:
                if ref["id"] not in known:
                    raise ValueError(f"{e['id']} is bound to missing {ref['id']}")
            for key in ("startBinding", "endBinding"):
                if e.get(key) and e[key]["elementId"] not in known:
                    raise ValueError(f"{e['id']} {key} points to a missing element")

    def save(self, path: Path) -> None:
        self.check()
        doc = {
            "type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
            "files": self.files,
        }  # fmt: skip
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# -- doodles ------------------------------------------------------------------------------
def cat(d: Diagram, prefix, cx, cy, s, fur, eyes, patch=None) -> None:
    """A cat face `s` wide centred on (cx, cy)."""
    for k, sign in enumerate((-1, 1)):
        d.poly(f"{prefix}_ear{k}", [(cx + sign * s * 0.46, cy - s * 0.52),
                                     (cx + sign * s * 0.12, cy - s * 0.30),
                                     (cx + sign * s * 0.5, cy - s * 0.12)], fur)  # fmt: skip
    d.ellipse(prefix + "_head", cx - s / 2, cy - s * 0.36, s, s * 0.8, fur)
    if patch:
        d.ellipse(prefix + "_patch", cx + s * 0.05, cy - s * 0.30, s * 0.38, s * 0.3, patch,
                  stroke=patch, stroke_width=1)  # fmt: skip
    for k, dx in enumerate((-0.2, 0.2)):
        d.ellipse(f"{prefix}_eye{k}", cx + s * dx - s * 0.07, cy - s * 0.1, s * 0.14, s * 0.18,
                  eyes, stroke_width=1)  # fmt: skip
    d.poly(prefix + "_nose", [(cx - s * 0.05, cy + s * 0.1), (cx + s * 0.05, cy + s * 0.1),
                              (cx, cy + s * 0.17)], "#ffa8a8", stroke_width=1)  # fmt: skip
    for k, sy in enumerate((0.14, 0.22)):
        d.poly(f"{prefix}_wl{k}", [(cx - s * 0.5, cy + s * sy), (cx - s * 0.2, cy + s * 0.16)],
               closed=False, stroke_width=1)  # fmt: skip
        d.poly(f"{prefix}_wr{k}", [(cx + s * 0.2, cy + s * 0.16), (cx + s * 0.5, cy + s * sy)],
               closed=False, stroke_width=1)  # fmt: skip


def dog(d: Diagram, prefix, cx, cy, s, fur="#e6c9a8", ears="#a47551") -> None:
    """A dog face `s` wide centred on (cx, cy)."""
    d.ellipse(prefix + "_earL", cx - s * 0.62, cy - s * 0.3, s * 0.34, s * 0.7, ears)
    d.ellipse(prefix + "_earR", cx + s * 0.28, cy - s * 0.3, s * 0.34, s * 0.7, ears)
    d.ellipse(prefix + "_head", cx - s / 2, cy - s * 0.36, s, s * 0.8, fur)
    d.ellipse(prefix + "_snout", cx - s * 0.22, cy, s * 0.44, s * 0.3, "#fff4e6", stroke_width=1)
    d.ellipse(prefix + "_nose", cx - s * 0.09, cy + s * 0.02, s * 0.18, s * 0.12, "#1e1e1e",
              stroke_width=1)  # fmt: skip
    for k, dx in enumerate((-0.2, 0.2)):
        d.ellipse(f"{prefix}_eye{k}", cx + s * dx - s * 0.05, cy - s * 0.12, s * 0.1, s * 0.12,
                  "#1e1e1e", stroke_width=1)  # fmt: skip
