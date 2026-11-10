---
paths:
  - "**/*.md"
---

# Documentation rules

- English only. Relative links must point to existing files.
- Do not duplicate: architecture lives in `ARCHITECTURE.md` (entry point, system view, package
  tables) and `docs/architecture/` (levels 2-4 with diagrams), status in `ROADMAP.md`, geometry
  in `docs/method.md`, lab design in `docs/pet-monitoring.md`. Link instead of copying.
- Architecture documents describe the code as implemented. Mark planned work as planned and
  never draw it; when a diagram and the code disagree, fix the diagram or record the gap in
  `docs/architecture/lab.md`.
- `docs/plans/` and `docs/reports/` are dated history. Do not update them to reflect the
  current state; record current state in `ROADMAP.md` and decisions in `docs/adr/`.
- Do not claim real-model inference, RealSense capture or real-world accuracy unless it was
  run on that hardware or model; mark such figures as measured, with the date and machine.
- README CLI tables must match the argparse definitions in `cli.py` for both packages.
- New architectural decisions get an ADR (`docs/adr/`), listed in its README.
- Diagrams in `docs/diagrams/` with a spec in `docs/diagrams/src/` are generated: edit the
  spec and rebuild with `/excalidraw`, never the `.excalidraw` or `.png` by hand. When a
  `ROADMAP.md` status changes, rebuild `roadmap.py` in the same change.
- Formal flows, call sequences, state machines and schemas are Mermaid sources in
  `docs/diagrams/mermaid/*.mmd` (`/mermaid`), embedded in the doc that explains them between
  `<!-- mermaid: path -->` markers; overviews, roadmaps and visual layouts are Excalidraw
  (`/excalidraw`). Do not draw the same content in both. Never edit a block between markers:
  change the `.mmd`, then run the skill's `sync.py` and `check.py`.
