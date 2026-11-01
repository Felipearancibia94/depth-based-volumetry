"""Evidence images on disk: ``<media_dir>/<YYYY-MM-DD>/<event_id>/<name>.jpg``."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from numpy.typing import NDArray

from depth_volumetry.io import write_image


def event_dir(media_dir: str | Path, event_id: str, created_unix: float | None = None) -> Path:
    day = time.strftime("%Y-%m-%d", time.localtime(created_unix or time.time()))
    return Path(media_dir) / day / event_id


def save_images(
    media_dir: str | Path,
    event_id: str,
    images: Sequence[tuple[str, NDArray]],
    created_unix: float | None = None,
) -> list[str]:
    """Write JPEGs and return their paths relative to ``media_dir``."""
    folder = event_dir(media_dir, event_id, created_unix)
    paths = []
    for name, image in images:
        target = folder / f"{name}.jpg"
        write_image(target, image)
        paths.append(target.relative_to(media_dir).as_posix())
    return paths


def resolve_media_path(media_dir: str | Path, relative: str) -> Path | None:
    """Resolve a relative evidence path, refusing anything outside ``media_dir``."""
    root = Path(media_dir).resolve()
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():
        return None
    return target
