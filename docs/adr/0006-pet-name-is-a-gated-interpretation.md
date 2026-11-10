# 0006. The pet name is a gated interpretation

Status: Accepted (2026-11-09)

## Context

The owner has three animals and wants events attributed to them. The lab has no recognition
model: track ids are not identities. A local VLM can read coat colour and species from the
evidence crops, but on saved crops it also named a pet (certainty high) in a room with no animal,
and its certainty did not warn about it.

## Decision

- The VLM may answer a short `pet` field, restricted by schema to the names in `[[vlm.pets]]`
  or `unknown`. It lives in `interpretation`, never in `observed_facts`, and no alert or rule
  depends on it (ADR 0003).
- The name is kept only if the event's mean detector score reaches `vlm.pet_min_detector_score`
  (0.4 by default) and the VLM's certainty is not `low`. Otherwise it is stored as `unknown`,
  the original answer is kept as `pet_vlm` and the reason is added to `uncertainty`.
- The owner can correct the name from the dashboard. The correction is stored apart as
  `review_pet`, wins in `PetEvent.pet` and is meant to become the label for evaluating the naming.
- The `pet` column in SQLite mirrors `PetEvent.pet` for filtering, search and the daily summary.

## Consequences

- A wrong name needs both a detection and a confident VLM, and can be fixed without losing what
  the model said.
- The threshold is a guess to be tuned on labelled events (corrections give those labels).
- Telling similar animals apart depends on the descriptions in the config; a trained 3-class
  detector remains the alternative if this proves too weak.
