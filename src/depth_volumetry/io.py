"""Reading and writing images, depth maps and masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def read_color(path: str | Path) -> NDArray[np.uint8]:
    """Read a colour image as a BGR ``H x W x 3`` array."""
    _require_file(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode colour image: {path}")
    return image


def read_depth(path: str | Path) -> NDArray:
    """Read a depth map from a ``.npy`` file or a single-channel 16-bit ``.png``."""
    _require_file(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        depth = np.load(path, allow_pickle=False)
    elif suffix == ".png":
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"could not decode depth image: {path}")
    else:
        raise ValueError(f"unsupported depth format {suffix!r}: use .npy or .png")
    if depth.ndim != 2:
        raise ValueError(f"depth map must be 2-D (single channel), got shape {depth.shape}")
    return depth


def read_mask(path: str | Path) -> NDArray[np.bool_]:
    """Read a binary mask image; any non-zero pixel counts as part of the object."""
    _require_file(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode mask image: {path}")
    return image > 0


def write_image(path: str | Path, image: NDArray) -> None:
    """Write an image, creating parent directories and failing loudly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"could not write image: {target}")


def _require_file(path: str | Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"file not found: {path}")
