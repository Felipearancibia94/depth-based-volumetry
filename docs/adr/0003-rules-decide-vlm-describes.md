# 0003. Rules decide, the VLM describes

Status: Accepted (2026-11-01)

## Context

A small local vision-language model (VLM) is slow, can fail to follow a schema and states
poorly calibrated confidence. Alerts must still be timely and explainable.

## Decision

- Events and notifications come from deterministic rules over measured facts.
- The VLM runs only after an event starts, receives a few evidence crops plus the measured
  facts, and its answer is merged into the stored event afterwards.
- An event keeps `observed_facts`, `hypothesis` (the rule's prior), `interpretation` (the VLM)
  and `uncertainty` as separate fields. `unknown` is a valid VLM answer. `confidence` stays
  empty until it can be calibrated against reviewed events.
- Missing depth gives `None` plus a quality flag, never a guessed value.

## Consequences

- Alerts never wait for the VLM; with the VLM off the app still works.
- Accuracy of the VLM is measured separately, with `unknown` counted apart.
- Hard activities (playing, grooming) may mostly return `unknown`.
