"""A scripted synthetic room with animals, rendered with exact colour and depth.

The scene is a flat floor and a back wall seen by a camera mounted ``camera_height_m``
above the floor and pitched down by ``pitch_deg``. Animals are upright rectangles that
move along keyframed paths and can rest on raised surfaces (``elevation_m``). Each frame
carries its ground-truth boxes in ``bundle.meta["ground_truth"]``; the scenario also
provides ground-truth activity segments for evaluating the whole pipeline without a
camera, a model or recordings.

This is a development and test fixture. It says nothing about real-world accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from perception_lab.core.types import BoxXYXY, FrameBundle, Intrinsics
from perception_lab.spatial.geometry import Plane
from perception_lab.spatial.scene import SceneCalibration, Zone


@dataclass(frozen=True)
class Keyframe:
    t_s: float
    x_m: float  # lateral, + right
    forward_m: float  # along the floor, away from the point below the camera
    elevation_m: float = 0.0


@dataclass(frozen=True)
class SyntheticAnimal:
    key: str
    label: str
    length_m: float
    height_m: float
    color_bgr: tuple[int, int, int]
    path: tuple[Keyframe, ...]

    def state(self, t_s: float) -> Keyframe:
        path = self.path
        if t_s <= path[0].t_s:
            return path[0]
        for a, b in zip(path, path[1:], strict=False):
            if t_s <= b.t_s:
                f = (t_s - a.t_s) / (b.t_s - a.t_s) if b.t_s > a.t_s else 1.0
                return Keyframe(
                    t_s,
                    a.x_m + f * (b.x_m - a.x_m),
                    a.forward_m + f * (b.forward_m - a.forward_m),
                    a.elevation_m + f * (b.elevation_m - a.elevation_m),
                )
        return path[-1]


@dataclass(frozen=True)
class FloorZone:
    """A zone defined on the floor (metres), projected to an image polygon."""

    name: str
    tags: tuple[str, ...]
    x_range_m: tuple[float, float]
    forward_range_m: tuple[float, float]
    elevation_m: float = 0.0
    min_elevation_m: float | None = None


@dataclass(frozen=True)
class ActivitySegment:
    """Ground truth: ``activity`` happens during ``[t_start_s, t_end_s]``.

    ``animal_class`` is ``None`` for pairwise activities involving several animals.
    """

    activity: str
    t_start_s: float
    t_end_s: float
    animal_class: str | None
    animal_key: str | None = None


@dataclass(frozen=True)
class Scenario:
    animals: tuple[SyntheticAnimal, ...]
    zones: tuple[FloorZone, ...]
    segments: tuple[ActivitySegment, ...]
    duration_s: float
    camera_height_m: float = 1.8
    pitch_deg: float = 25.0
    wall_forward_m: float = 5.0
    intrinsics: Intrinsics = field(
        default_factory=lambda: Intrinsics(640, 480, 525.0, 525.0, 320.0, 240.0)
    )


def default_scenario() -> Scenario:
    """Two minutes with a cat and a dog exercising every MVP rule once."""
    cat = SyntheticAnimal(
        "cat-1",
        "cat",
        0.45,
        0.25,
        (60, 140, 230),
        (
            Keyframe(0, -1.2, 2.2),
            Keyframe(5, -0.6, 2.4),  # reaches the bowl
            Keyframe(35, -0.6, 2.4),  # eats for 30 s
            Keyframe(40, 0.6, 2.6),
            Keyframe(41, 0.8, 2.6, 0.9),  # jumps onto the counter
            Keyframe(51, 0.8, 2.6, 0.9),
            Keyframe(52, 0.6, 2.6),
            Keyframe(57, 0.0, 3.2),  # lies down
            Keyframe(120, 0.0, 3.2),
        ),
    )
    dog = SyntheticAnimal(
        "dog-1",
        "dog",
        0.7,
        0.45,
        (40, 70, 120),
        (
            Keyframe(0, 1.4, 4.0),
            Keyframe(15, -0.4, 3.8),
            Keyframe(18, -1.2, 3.8),  # enters the restricted zone
            Keyframe(23, -1.2, 3.8),
            Keyframe(26, -0.2, 3.6),
            Keyframe(56, -1.4, 3.0),
            Keyframe(58, -1.4, 3.0),
            Keyframe(59.8, 1.4, 3.0),  # runs across the room (~1.6 m/s)
            Keyframe(70, 1.0, 3.3),
            Keyframe(80, 0.45, 3.2),  # stands next to the cat
            Keyframe(90, 0.45, 3.2),
            Keyframe(95, 1.2, 3.8),
            Keyframe(120, 1.2, 3.8),
        ),
    )
    zones = (
        FloorZone("food_bowl", ("food",), (-0.9, -0.3), (2.15, 2.65)),
        FloorZone("bedroom_door", ("restricted",), (-1.6, -0.8), (3.4, 4.2)),
        FloorZone("counter", ("elevated",), (0.4, 1.2), (2.3, 2.9), 0.9, 0.6),
    )
    segments = (
        ActivitySegment("eating", 5, 35, "cat", "cat-1"),
        ActivitySegment("entered_restricted_area", 18, 23, "dog", "dog-1"),
        ActivitySegment("on_elevated_surface", 41, 51, "cat", "cat-1"),
        ActivitySegment("resting_or_sleeping", 57, 120, "cat", "cat-1"),
        ActivitySegment("running", 58, 59.8, "dog", "dog-1"),
        ActivitySegment("interacting", 80, 90, None),
    )
    return Scenario((cat, dog), zones, segments, duration_s=120.0)


class SyntheticSource:
    """Renders a :class:`Scenario` at ``fps``; deterministic for a given ``seed``."""

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        fps: float = 15.0,
        duration_s: float | None = None,
        seed: int = 0,
        depth_noise: bool = True,
        camera_id: str = "synthetic",
    ) -> None:
        self.scenario = scenario or default_scenario()
        self.fps = fps
        self.duration_s = duration_s if duration_s is not None else self.scenario.duration_s
        self.seed = seed
        self.depth_noise = depth_noise
        self.camera_id = camera_id
        s = self.scenario
        t = math.radians(s.pitch_deg)
        self._down = np.array([0.0, math.cos(t), math.sin(t)])
        self._forward = np.array([0.0, -math.sin(t), math.cos(t)])
        self._lateral = np.array([1.0, 0.0, 0.0])
        self._background_depth, self._background_color = self._render_background()

    # -- geometry ---------------------------------------------------------------------

    def floor_plane(self) -> Plane:
        up = -self._down
        return Plane((float(up[0]), float(up[1]), float(up[2])), self.scenario.camera_height_m)

    def world_point(self, x_m: float, forward_m: float, elevation_m: float = 0.0) -> NDArray:
        h = self.scenario.camera_height_m
        return (h - elevation_m) * self._down + forward_m * self._forward + x_m * self._lateral

    def scene(self) -> SceneCalibration:
        """Exact calibration for this scenario: floor plane and projected zones."""
        k = self.scenario.intrinsics
        zones = []
        for z in self.scenario.zones:
            (x0, x1), (f0, f1) = z.x_range_m, z.forward_range_m
            corners = [(x0, f0), (x1, f0), (x1, f1), (x0, f1)]
            polygon = tuple(
                tuple(
                    round(v, 1) for v in k.project_px(tuple(self.world_point(x, f, z.elevation_m)))
                )
                for x, f in corners
            )
            zones.append(Zone(z.name, polygon, z.tags, z.min_elevation_m))
        return SceneCalibration(self.camera_id, self.floor_plane(), tuple(zones))

    # -- rendering --------------------------------------------------------------------

    def _render_background(self) -> tuple[NDArray, NDArray]:
        k = self.scenario.intrinsics
        ys, xs = np.mgrid[0 : k.height_px, 0 : k.width_px].astype(np.float64)
        rays = np.stack(
            [(xs - k.cx_px) / k.fx_px, (ys - k.cy_px) / k.fy_px, np.ones_like(xs)], axis=-1
        )
        along_down, along_fwd = rays @ self._down, rays @ self._forward
        with np.errstate(divide="ignore", invalid="ignore"):
            z_floor = np.where(
                along_down > 1e-6, self.scenario.camera_height_m / along_down, np.inf
            )
            z_wall = np.where(along_fwd > 1e-6, self.scenario.wall_forward_m / along_fwd, np.inf)
        depth = np.minimum(z_floor, z_wall)
        depth[~np.isfinite(depth)] = 0.0
        color = np.empty((k.height_px, k.width_px, 3), np.uint8)
        is_floor = z_floor <= z_wall
        color[is_floor] = (95, 110, 125)
        color[~is_floor] = (200, 205, 210)
        # Faint floor stripes give the image some texture.
        stripes = (np.floor(np.where(is_floor, z_floor, 0) * 2) % 2).astype(bool) & is_floor
        color[stripes] = (85, 100, 115)
        return depth.astype(np.float32), color

    def ground_truth_boxes(self, t_s: float) -> list[tuple[SyntheticAnimal, BoxXYXY, Keyframe]]:
        k = self.scenario.intrinsics
        out = []
        for animal in self.scenario.animals:
            state = animal.state(t_s)
            foot = self.world_point(state.x_m, state.forward_m, state.elevation_m)
            top = foot - animal.height_m * self._down
            half = animal.length_m / 2 * self._lateral
            pts = [
                k.project_px(tuple(p)) for p in (foot - half, foot + half, top - half, top + half)
            ]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = (
                max(0.0, min(xs)),
                max(0.0, min(ys)),
                min(k.width_px, max(xs)),
                min(k.height_px, max(ys)),
            )
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                out.append((animal, box, state))
        return out

    def render(self, index: int) -> FrameBundle:
        t_s = index / self.fps
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        depth = self._background_depth.copy()
        color = self._background_color.copy()
        objects = self.ground_truth_boxes(t_s)
        # Draw far animals first so near ones occlude them.
        objects.sort(key=lambda o: -self.world_point(o[2].x_m, o[2].forward_m, o[2].elevation_m)[2])
        truth = []
        for animal, box, state in objects:
            foot = self.world_point(state.x_m, state.forward_m, state.elevation_m)
            top = foot - animal.height_m * self._down
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            rows = np.arange(y1, y2)
            frac = (rows - box[1]) / max(1e-6, box[3] - box[1])
            z_rows = (top[2] + frac * (foot[2] - top[2])).astype(np.float32)
            depth[y1:y2, x1:x2] = z_rows[:, None]
            color[y1:y2, x1:x2] = animal.color_bgr
            truth.append({"key": animal.key, "label": animal.label, "box_xyxy": box})
        if self.depth_noise:
            noise = rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * 0.001 * depth**2
            depth = depth + noise
            depth[rng.random(depth.shape) < 0.02] = 0.0
        depth_raw = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        return FrameBundle(
            index=index,
            timestamp_s=t_s,
            color_bgr=color,
            depth=depth_raw,
            depth_unit_mm=1.0,
            intrinsics=self.scenario.intrinsics,
            camera_id=self.camera_id,
            meta={"ground_truth": truth},
        )

    def frames(self) -> Iterator[FrameBundle]:
        n = int(round(self.duration_s * self.fps))
        for index in range(n):
            yield self.render(index)

    def close(self) -> None:
        pass
