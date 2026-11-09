---
name: cli-change
description: Add, rename or remove a command-line flag or subcommand of depth-volumetry or lab and keep parser, README, docs and tests in sync. Use whenever a cli.py file is edited.
---

# CLI change

1. Identify the package: `src/depth_volumetry/cli.py` (`measure`, `capture`) or
   `src/perception_lab/cli.py` (`lab ...`).
2. Edit the argparse definition. Units in names (`--seconds`, `--min-dwell-s`); an invalid
   value must raise an error, not be clamped.
3. Update the matching README table (`## Usage` for `depth-volumetry`, `## Home perception lab`
   for `lab`) and any command quoted in `docs/pet-monitoring.md` or `docs/method.md`.
4. Update the test: `tests/test_cli.py` or `tests/lab/test_cli.py`. Cover the new flag and an
   invalid value.
5. If a config key changed, update `core/config.py` and `examples/lab.example.toml` together.
6. Check that nothing else in the docs mentions the old flag:
   `grep -rn -- "--old-flag" README.md docs examples`.
7. Run `/verify` and report the results.
