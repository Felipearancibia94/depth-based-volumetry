---
name: commit
description: Create a Conventional Commit from the current changes. Only runs when the user invokes /commit.
disable-model-invocation: true
allowed-tools: Bash(git status*), Bash(git diff*), Bash(git log*), Bash(git branch --show-current), Bash(git add *), Bash(git commit *)
---

# Commit

Invoking this skill is the user's authorization for one commit. It does not authorize a push,
a branch change or a remote change.

1. `git branch --show-current`, `git status --short`, `git diff` (staged and unstaged). If on
   `main`, stop and ask which branch to use.
2. Confirm nothing sensitive or ignored is included: no `*.onnx`, `*.bag`, `*.db`, `.env*`,
   `lab.toml`, `scene.toml`, `calibration.toml`, credentials or non-loopback endpoints.
3. Run `/verify`. Do not commit with failing checks unless the user says so; mention it in the
   message if they do.
4. Stage only the files that belong to one logical change.
5. Message: Conventional Commits (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, with an
   optional scope such as `events`, `spatial`, `cli`), imperative, under about 72 characters,
   a body that explains why when it is not obvious. Match the style of `git log`.
6. Commit, show `git log -1 --stat`, and stop. Never push.
