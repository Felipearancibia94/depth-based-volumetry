---
paths:
  - "src/perception_lab/**"
  - "examples/lab.example.toml"
  - "examples/labels.example.json"
---

# perception_lab rules

- Alerts and notifications come from deterministic rules. The VLM only describes and never
  gates an alert (ADR 0003).
- Keep observed facts, rule hypothesis, VLM interpretation and uncertainty in separate fields.
  `unknown` is a valid VLM answer. Do not fill `confidence` with model-stated numbers.
- Missing or invalid depth gives `None` plus a quality flag. Say whether speeds and distances
  are in metres or body lengths.
- Local only (ADR 0004): VLM, webhook and dashboard endpoints stay on loopback unless the
  config explicitly allows otherwise. Do not add outbound network calls, cloud services,
  credentials or non-loopback hosts.
- Standard library first (ADR 0002): TOML, SQLite, `urllib`, `http.server`. A new runtime
  dependency needs a stated reason and `uv add`.
- Import from `depth_volumetry` only through the reuse points listed in `ARCHITECTURE.md`; do
  not copy its code.
- Config changes go through `core/config.py` and `examples/lab.example.toml` together; a
  changed `lab` flag needs the README table and `tests/lab/test_cli.py` (`/cli-change`).
- Never describe a result as real-world accuracy. Synthetic runs only check the plumbing.
