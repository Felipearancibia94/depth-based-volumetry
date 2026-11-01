"""Run records: every experiment writes down what produced its numbers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

from perception_lab.core.config import LabConfig, dump_config

_PACKAGES = (
    "depth-based-volumetry",
    "numpy",
    "opencv-python-headless",
    "onnxruntime",
    "onnxruntime-gpu",
    "pyrealsense2",
)


def git_revision() -> str | None:
    """Current commit, suffixed with ``-dirty`` when the working tree has changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def build_run_record(config: LabConfig, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config_text = dump_config(config)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "config_toml": config_text,
        "detector_model_sha256": file_sha256(config.detector.model_path),
        **(extra or {}),
    }


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, default=str))
    return target
