"""Per-camera scene calibration: floor plane and monitored zones, stored as TOML.

Example ``scene.toml``::

    camera_id = "living-room"

    [floor]                      # written by `lab calibrate`
    normal = [0.0, -0.82, -0.57]
    offset_m = 1.2

    [[zones]]
    name = "food_bowl"
    tags = ["food"]
    polygon_xy_px = [[100, 300], [200, 300], [200, 380], [100, 380]]

    [[zones]]
    name = "kitchen_counter"
    tags = ["restricted"]
    polygon_xy_px = [[400, 150], [600, 150], [600, 250], [400, 250]]
    min_elevation_m = 0.6        # optional: only counts when the animal is this high

A zone contains an animal when the bottom-centre of its box (its "foot" point) lies inside
the image polygon and, if an elevation range is given, the measured elevation is inside it.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perception_lab.core.config import ConfigError, toml_value
from perception_lab.spatial.geometry import Plane


@dataclass(frozen=True)
class Zone:
    name: str
    polygon_xy_px: tuple[tuple[float, float], ...]
    tags: tuple[str, ...] = ()
    min_elevation_m: float | None = None
    max_elevation_m: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("zone name must not be empty")
        if len(self.polygon_xy_px) < 3:
            raise ConfigError(f"zone {self.name!r} needs at least 3 polygon points")

    @property
    def needs_elevation(self) -> bool:
        return self.min_elevation_m is not None or self.max_elevation_m is not None

    def contains(self, point_xy: tuple[float, float], elevation_m: float | None = None) -> bool:
        if not point_in_polygon(point_xy, self.polygon_xy_px):
            return False
        if not self.needs_elevation:
            return True
        if elevation_m is None:
            return False
        if self.min_elevation_m is not None and elevation_m < self.min_elevation_m:
            return False
        return not (self.max_elevation_m is not None and elevation_m > self.max_elevation_m)


@dataclass(frozen=True)
class SceneCalibration:
    camera_id: str = "cam0"
    floor: Plane | None = None
    zones: tuple[Zone, ...] = ()

    def zones_with_tag(self, tag: str) -> tuple[Zone, ...]:
        return tuple(z for z in self.zones if tag in z.tags)

    def zone(self, name: str) -> Zone | None:
        return next((z for z in self.zones if z.name == name), None)


def point_in_polygon(point_xy: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd ray casting; points exactly on an edge may fall either way."""
    x, y = point_xy
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def scene_from_mapping(data: Mapping[str, Any]) -> SceneCalibration:
    unknown = sorted(set(data) - {"camera_id", "floor", "zones"})
    if unknown:
        raise ConfigError(f"unknown scene key(s): {', '.join(unknown)}")
    floor = None
    if "floor" in data:
        f = data["floor"]
        try:
            normal = tuple(float(v) for v in f["normal"])
            floor = Plane((normal[0], normal[1], normal[2]), float(f["offset_m"]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ConfigError("floor needs normal = [x, y, z] and offset_m") from exc
    zones = []
    for i, z in enumerate(data.get("zones", [])):
        allowed = {"name", "tags", "polygon_xy_px", "min_elevation_m", "max_elevation_m"}
        extra = sorted(set(z) - allowed)
        if extra:
            raise ConfigError(f"unknown key(s) in zones[{i}]: {', '.join(extra)}")
        try:
            polygon = tuple((float(p[0]), float(p[1])) for p in z["polygon_xy_px"])
            zones.append(
                Zone(
                    name=str(z["name"]),
                    polygon_xy_px=polygon,
                    tags=tuple(str(t) for t in z.get("tags", ())),
                    min_elevation_m=_opt_float(z.get("min_elevation_m")),
                    max_elevation_m=_opt_float(z.get("max_elevation_m")),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ConfigError(f"zones[{i}] needs name and polygon_xy_px = [[x, y], ...]") from exc
    names = [z.name for z in zones]
    if len(names) != len(set(names)):
        raise ConfigError("zone names must be unique")
    return SceneCalibration(str(data.get("camera_id", "cam0")), floor, tuple(zones))


def load_scene(path: str | Path | None) -> SceneCalibration:
    if path is None:
        return SceneCalibration()
    try:
        with open(path, "rb") as handle:
            return scene_from_mapping(tomllib.load(handle))
    except FileNotFoundError as exc:
        raise ConfigError(f"scene file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def dump_scene(scene: SceneCalibration) -> str:
    lines = [f"camera_id = {toml_value(scene.camera_id)}", ""]
    if scene.floor is not None:
        lines += [
            "[floor]",
            f"normal = {toml_value([round(v, 6) for v in scene.floor.normal])}",
            f"offset_m = {round(scene.floor.offset_m, 6)!r}",
            "",
        ]
    for zone in scene.zones:
        lines += [
            "[[zones]]",
            f"name = {toml_value(zone.name)}",
            f"tags = {toml_value(list(zone.tags))}",
            f"polygon_xy_px = {toml_value([list(p) for p in zone.polygon_xy_px])}",
        ]
        if zone.min_elevation_m is not None:
            lines.append(f"min_elevation_m = {zone.min_elevation_m!r}")
        if zone.max_elevation_m is not None:
            lines.append(f"max_elevation_m = {zone.max_elevation_m!r}")
        lines.append("")
    return "\n".join(lines)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number")
    return float(value)
