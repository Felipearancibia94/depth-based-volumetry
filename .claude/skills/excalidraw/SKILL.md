---
name: excalidraw
description: Create or update an Excalidraw diagram in docs/diagrams (editable .excalidraw plus a PNG that renders on GitHub with the hand-drawn font). Use when asked for a diagram, an "excalidraw", a drawing of the roadmap, architecture, flow or pets, or when a change makes an existing generated diagram stale.
---

# Excalidraw diagrams

Diagrams are code: a small Python spec in `docs/diagrams/src/<name>.py` produces
`docs/diagrams/<name>.excalidraw` (editable at excalidraw.com) and `<name>.png` (what README
and docs embed; GitHub draws Excalidraw SVG text in a fallback font, so we ship PNG).
Specs use only the standard library through `scripts/excalidraw_lib.py`.

Existing specs: `roadmap.py`, `home_perception_lab_overview.py`, `pet_identification.py`.
`pet-monitoring-architecture.excalidraw` is hand-made and has no spec; leave it alone unless
asked.

## 1. Content first

- Take every status and number from its source of truth: `ROADMAP.md` for status,
  `ARCHITECTURE.md` for structure, `docs/pet-monitoring.md` for the lab. Do not invent figures.
- Mark measured values as measured (date, machine); never present a single run as accuracy.
- English text. No model-vendor hints for the detector (say "multi-class YOLO-layout ONNX").
- Photos only from `docs/gallery/` (owner-picked, no people). Never embed captures from
  `data/`.

## 2. Write or edit the spec

```python
"""One-line purpose. Where the content comes from."""

from excalidraw_lib import BLUE, GREEN, GREY, MUTED, ORANGE, PURPLE, RED, cat, dog

OUTPUT = "docs/diagrams/<name>"  # no extension
SEED = 9000  # any constant, keeps output deterministic


def build(d, repo):
    d.header("Title", "Legend: what each colour means in this diagram")
    d.row("s", 20, 100, 260, 90, 300, [("Step 1\ndetail", GREEN), ("Step 2", ORANGE)])
    y = d.column("c", 0, 240, 340, [("Long text is wrapped to the width", BLUE)])
    d.zone("z", 0, y + 20, 700, 200, ("#e5dbff", "#9775fa"))
    d.arrow("a1", "s0", "c0", dashed=True, text="label")  # ids from row/column: prefix + index
    d.image("ph", repo / "docs/gallery/negrita.jpg", 0, 500, scale=1.2)
    cat(d, "cat1", 800, 40, 60, "#343a40", "#ffd43b")  # doodles: cat(), dog()
```

Helpers: `label`, `box`, `zone`, `ellipse`, `poly`, `image`, `arrow` (straight, or `elbow_y=`
for down-across-down), `column` (auto-sized stacked boxes, returns next y), `row` (boxes joined
by arrows), `header`. Colours are `(background, stroke)` pairs; GREY boxes in a column are
dashed. Always state the colour meaning in the legend. Element ids must be unique (checked
on save).

## 3. Build

```bash
uv run --no-project python .claude/skills/excalidraw/scripts/build.py docs/diagrams/src/<name>.py
```

`--all` rebuilds every spec. Run from the repository root.

## 4. Render the PNG

Start the renderer in the background (it waits for the browser, writes the PNG, then exits):

```bash
uv run --no-project python .claude/skills/excalidraw/scripts/render.py docs/diagrams/<name>.excalidraw
```

Read the `open: http://127.0.0.1:<port>/` line from its output, open that URL in the built-in
browser, wait about 3 s and read `document.title`: `saved 200 WxH` means the PNG was written;
`ERR ...` means a render bug (fix `render.html`, restart the renderer). The page loads the
Virgil font from cdn.jsdelivr.net; without network the PNG falls back to a sans-serif font.
A user without Claude can open the same URL in any browser.

## 5. Check and wire up

- Look at the PNG (Read it): no overlapping text, no box text spilling out, arrows land on
  boxes, nothing far outside the layout (a stray arrow makes the image huge).
- Link the PNG and the `.excalidraw` from the doc that owns the topic (README, ROADMAP.md,
  `docs/pet-monitoring.md`); do not copy the diagram's content into prose elsewhere.
- When a status in `ROADMAP.md` changes, rebuild `roadmap.py` (and the overview if it shows
  that item) in the same change.
- Run `/verify` (ruff covers the specs and scripts). Commit spec, `.excalidraw` and `.png`
  together.
