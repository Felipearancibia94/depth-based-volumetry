# 0001. `perception_lab` as a sibling package

Status: Accepted (2026-11-01)

## Context

The plan in `docs/plans/2026-11-01-pet-monitoring-plan.md` proposed renaming `depth_volumetry`
to `perception_lab`. By implementation time `main` already held working, tested volumetry
code under `depth_volumetry`.

## Decision

Add `perception_lab` next to `depth_volumetry` in the same distribution and CLI entry points
(`depth-volumetry`, `lab`). The lab imports from `depth_volumetry`; the reverse is not allowed.

## Consequences

- Volumetry code and its tests were not touched.
- The lab reuses segmentation decoding, the lazy RealSense loader and image I/O instead of
  duplicating them (see `ARCHITECTURE.md`).
- Changing a reused function in `depth_volumetry` can break the lab; run the full test suite.
