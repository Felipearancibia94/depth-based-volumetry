"""Lab configuration: one TOML file mapped onto frozen dataclasses.

Every field has a safe local-first default, so ``LabConfig()`` runs the synthetic
pipeline without any file, camera, GPU or model. Unknown keys and wrongly typed values
raise :class:`ConfigError`; nothing is silently substituted.
"""

from __future__ import annotations

import json
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    """Raised when a configuration file or mapping is invalid."""


@dataclass(frozen=True)
class SourceConfig:
    kind: Literal["synthetic", "video", "bag", "realsense"] = "synthetic"
    path: str | None = None  # video file or .bag file
    camera_id: str = "cam0"
    serial: str | None = None  # RealSense serial; None = first device found
    width_px: int = 640
    height_px: int = 480
    fps: int = 15
    depth: bool = True
    real_time: bool = False  # .bag replay: False = deterministic, every frame
    max_frames: int | None = None
    process_fps: float | None = None  # drop frames above this rate; None = keep all
    synthetic_seconds: float = 120.0
    seed: int = 0


@dataclass(frozen=True)
class DetectorConfig:
    kind: Literal["yolov8_onnx", "ground_truth"] = "ground_truth"
    model_path: str | None = None
    score_threshold: float = 0.35
    iou_threshold: float = 0.5
    class_ids: tuple[int, ...] = (15, 16)  # COCO ids of the classes to keep
    class_labels: tuple[str, ...] = ("cat", "dog")  # names for ``class_ids``, same order
    providers: tuple[str, ...] = ()  # ONNX Runtime providers; empty = all available
    every_n_frames: int = 1
    # ground_truth detector only (synthetic robustness tests): drop and jitter boxes
    gt_miss_rate: float = 0.0
    gt_jitter_px: float = 0.0


@dataclass(frozen=True)
class TrackerConfig:
    high_score: float = 0.5
    low_score: float = 0.15
    match_iou: float = 0.3
    low_match_iou: float = 0.5
    min_hits: int = 3
    max_lost_s: float = 2.0
    max_center_shift: float = 1.0  # last-resort match: centre shift / box diagonal, same label


@dataclass(frozen=True)
class SpatialConfig:
    scene_path: str | None = None  # scene TOML with floor plane and zones
    box_shrink: float = 0.2  # fraction trimmed from each side before sampling depth
    foot_band: float = 0.15  # lowest fraction of the box used for the foot depth
    foot_body_max_diff_m: float = 0.3  # larger gaps mean the band sees background
    min_valid_ratio: float = 0.2
    depth_min_m: float = 0.2
    depth_max_m: float = 8.0


@dataclass(frozen=True)
class RulesConfig:
    cooldown_s: float = 120.0
    release_s: float = 3.0  # a condition must be false this long before it can fire again
    flicker_tolerance_s: float = 1.0  # brief interruptions that do not break a condition
    history_s: float = 600.0
    zone_entry_tags: tuple[str, ...] = ("restricted",)
    zone_entry_min_s: float = 0.5
    dwell_tags: dict[str, float] = field(
        default_factory=lambda: {"food": 15.0, "water": 10.0, "bed": 60.0}
    )
    elevated_min_m: float = 0.5
    elevated_min_s: float = 2.0
    stationary_radius_m: float = 0.15
    stationary_radius_bl: float = 0.3  # in body lengths (box height), used without depth
    stationary_s: float = 300.0
    long_inactivity_s: float = 3600.0
    speed_window_s: float = 1.0
    speed_burst_mps: float = 1.2
    speed_burst_bl: float = 2.5  # body lengths per second, used without depth
    speed_burst_min_s: float = 1.0
    proximity_m: float = 0.5
    proximity_min_s: float = 5.0


@dataclass(frozen=True)
class VLMConfig:
    enabled: bool = False
    provider: Literal["ollama", "openai_compat"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:2b"
    max_images: int = 4
    image_max_side_px: int = 448
    montage: bool = False  # one combined image, for runtimes without multi-image input
    context_frame: bool = True  # add one uncropped frame
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_requests_per_minute: float = 4.0
    queue_size: int = 8
    keep_alive: str = "5m"
    # Ollama context window in tokens: prompt (about 1100 per image) plus the answer must fit.
    # Ollama's own default of 4096 cuts answers with three or more images. Not used by
    # ``openai_compat``: set the context when starting that server (``llama-server -c``).
    num_ctx: int = 8192
    allow_remote: bool = False  # refuse non-loopback endpoints unless explicitly allowed

    def __post_init__(self) -> None:
        if self.num_ctx < 1:
            raise ConfigError("num_ctx must be a positive number of tokens")


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "data/lab.db"
    media_dir: str = "data/media"
    ring_buffer_s: float = 20.0
    ring_buffer_fps: float = 4.0
    evidence_frames: int = 4
    evidence_max_side_px: int = 640
    media_retention_days: float = 14.0
    event_retention_days: float = 365.0


@dataclass(frozen=True)
class NotifyConfig:
    desktop: bool = True
    webhook_url: str | None = None
    allow_remote_webhook: bool = False
    min_interval_s: float = 60.0
    rules: tuple[str, ...] = ()  # empty = notify for every rule


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    allow_remote: bool = False  # the dashboard has no authentication


@dataclass(frozen=True)
class LabConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def __post_init__(self) -> None:
        if len(self.detector.class_ids) != len(self.detector.class_labels):
            raise ConfigError("detector.class_ids and detector.class_labels must have equal length")


def load_config(path: str | Path | None) -> LabConfig:
    """Read a TOML config; ``None`` returns the defaults."""
    if path is None:
        return LabConfig()
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return config_from_mapping(data)


def config_from_mapping(data: Mapping[str, Any]) -> LabConfig:
    return _build(LabConfig, data, "config")


def to_mapping(obj: Any) -> Any:
    """Dataclass -> plain dict (``None`` fields dropped, tuples as lists)."""
    if is_dataclass(obj):
        return {
            f.name: to_mapping(getattr(obj, f.name))
            for f in fields(obj)
            if getattr(obj, f.name) is not None
        }
    if isinstance(obj, tuple | list):
        return [to_mapping(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_mapping(v) for k, v in obj.items()}
    return obj


def dump_config(config: LabConfig) -> str:
    """Serialise a config as TOML that :func:`load_config` reads back unchanged."""
    lines: list[str] = []
    for section, values in to_mapping(config).items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def toml_value(value: Any) -> str:
    """TOML literal for a scalar, list or flat mapping (enough for lab files)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON basic strings are valid TOML basic strings
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        items = ", ".join(f"{json.dumps(str(k))} = {toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }" if items else "{}"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must be a table")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")
    values = {name: _coerce(hints[name], data[name], f"{where}.{name}") for name in data}
    try:
        return cls(**values)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _coerce(options[0], value, where)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where} must be one of {', '.join(map(repr, args))}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where} must be a list")
        return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} must be a table")
        return {str(k): _coerce(args[1], v, f"{where}.{k}") for k, v in value.items()}
    raise ConfigError(f"{where}: unsupported type {tp!r}")
