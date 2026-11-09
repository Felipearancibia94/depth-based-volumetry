# 0004. Local only, loopback endpoints

Status: Accepted (2026-11-01)

## Context

The lab captures images of a household, including people. Nothing should leave the machine
by default.

## Decision

- No cloud services. The VLM endpoint, the webhook and the dashboard must be on loopback
  (`127.0.0.1`, `localhost`, `::1`) unless the configuration explicitly allows otherwise
  (for example `web.allow_remote = true`).
- No continuous recording: frames live in a short in-memory ring buffer; only event evidence
  is written, with retention and `lab purge`.
- Loopback defaults (such as the local Ollama address) may appear in `examples/`. Credentials,
  company or client names and non-loopback endpoints must not be added to the repository.
- Recordings, databases, media, models and calibration files are git-ignored.

## Consequences

- Safe defaults, at the cost of a manual step to expose the dashboard to other machines.
- The dashboard has no authentication, so remote access is the user's responsibility.
