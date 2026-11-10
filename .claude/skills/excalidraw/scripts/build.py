"""Build .excalidraw files from diagram specs.

    S=.claude/skills/excalidraw/scripts
    uv run --no-project python $S/build.py docs/diagrams/src/roadmap.py
    uv run --no-project python $S/build.py --all

Run from the repository root. Each spec defines OUTPUT and build(d, repo); see excalidraw_lib.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from excalidraw_lib import Diagram  # noqa: E402

SPEC_DIR = Path("docs/diagrams/src")


def build(spec_path: Path, repo: Path) -> Path:
    spec = importlib.util.spec_from_file_location(spec_path.stem, spec_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {spec_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagram = Diagram(seed=getattr(module, "SEED", 1000))
    module.build(diagram, repo)
    out = repo / (module.OUTPUT + ".excalidraw")
    diagram.save(out)
    n_files = len(diagram.files)
    print(f"{spec_path} -> {out.relative_to(repo)} ({len(diagram.elements)} elements, "
          f"{n_files} images)")  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help=f"build every spec in {SPEC_DIR}")
    args = parser.parse_args()
    repo = Path.cwd()
    if not (repo / "pyproject.toml").exists():
        raise SystemExit("run from the repository root")
    specs = sorted((repo / SPEC_DIR).glob("*.py")) if args.all else args.specs
    if not specs:
        raise SystemExit("give a spec path or --all")
    for spec_path in specs:
        build(spec_path, repo)


if __name__ == "__main__":
    main()
