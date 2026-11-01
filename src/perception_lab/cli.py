"""``lab`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from perception_lab import __version__
from perception_lab.core.config import ConfigError, LabConfig, dump_config

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_OR_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab", description="Home perception lab: local-first RGB-D pet monitoring."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-config", help="write the default configuration as TOML")
    p.add_argument("--out", type=Path, default=Path("lab.toml"))
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=_run_init_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_OR_INPUT


def _run_init_config(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.force:
        raise ValueError(f"{args.out} exists; use --force to overwrite it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_config(LabConfig()))
    print(f"wrote {args.out}")
    return EXIT_OK
