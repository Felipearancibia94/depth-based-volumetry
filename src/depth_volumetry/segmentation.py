"""Instance segmentation with an ONNX model that uses the YOLOv8-seg output layout.

The decoding functions are pure and operate on plain arrays, so they can be tested
without a model. :class:`SegmentationModel` is a thin wrapper around an ONNX Runtime
session.

Model contract
--------------
* Input: ``(1, 3, H, W)`` float32, RGB, scaled to ``[0, 1]``. The image is resized to
  ``(W, H)`` without letterboxing.
* ``output0``: ``(1, 4 + C + M, N)``. For each of the ``N`` candidates: box centre
  ``(cx, cy)`` and size ``(w, h)`` in input pixels, ``C`` class scores, ``M`` mask
  coefficients.
* ``output1``: ``(1, M, mh, mw)`` mask prototypes.

Conventions: points are ``(x, y)``; shapes are ``(height, width)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from depth_volumetry.measurement import MeasurementError

DEFAULT_INPUT_SIZE = (640, 640)


@dataclass(frozen=True)
class Detection:
    """One detected object, in original image coordinates."""

    box_xyxy: NDArray[np.float32]
    score: float
    class_id: int
    mask: NDArray[np.bool_]  # (H, W) of the original image

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in self.box_xyxy)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def preprocess(color_bgr: NDArray, input_size: tuple[int, int]) -> NDArray[np.float32]:
    """Convert a BGR image to the model input tensor ``(1, 3, H, W)``."""
    height, width = input_size
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError(f"expected a H x W x 3 colour image, got shape {color_bgr.shape}")
    resized = cv2.resize(color_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis])


def xywh_to_xyxy(boxes: NDArray) -> NDArray:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou(box: NDArray, boxes: NDArray) -> NDArray:
    """Intersection over union of one ``xyxy`` box against many."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - intersection
    iou = np.zeros_like(union, dtype=np.float32)
    np.divide(intersection, union, out=iou, where=union > 0)
    return iou


def nms(boxes_xyxy: NDArray, scores: NDArray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best score first."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes_xyxy[best], boxes_xyxy[rest]) <= iou_threshold]
    return keep


def decode_detections(
    predictions: NDArray,
    prototypes: NDArray,
    *,
    image_shape: tuple[int, int],
    input_size: tuple[int, int],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> list[Detection]:
    """Turn raw model outputs into detections with full-resolution boolean masks.

    Args:
        predictions: ``output0`` with shape ``(1, 4 + C + M, N)`` (or without the batch axis).
        prototypes: ``output1`` with shape ``(1, M, mh, mw)`` (or without the batch axis).
        image_shape: ``(height, width)`` of the original image.
        input_size: ``(height, width)`` of the model input.
        conf_threshold: Minimum best-class score.
        iou_threshold: NMS overlap threshold (class-agnostic).
        mask_threshold: Threshold applied to the mask probabilities.
    """
    if predictions.ndim == 3:
        predictions = predictions[0]
    if prototypes.ndim == 4:
        prototypes = prototypes[0]
    if predictions.ndim != 2 or prototypes.ndim != 3:
        raise ValueError("unexpected output shapes; expected (1, 4+C+M, N) and (1, M, mh, mw)")

    num_masks, mask_h, mask_w = prototypes.shape
    num_classes = predictions.shape[0] - 4 - num_masks
    if num_classes < 1:
        raise ValueError(
            f"output0 has {predictions.shape[0]} rows, too few for 4 box values, "
            f"{num_masks} mask coefficients and at least one class"
        )

    candidates = predictions.T  # (N, 4 + C + M)
    class_scores = candidates[:, 4 : 4 + num_classes]
    scores = class_scores.max(axis=1)
    keep = scores >= conf_threshold
    if not keep.any():
        return []
    candidates, scores = candidates[keep], scores[keep]
    class_ids = class_scores[keep].argmax(axis=1)
    boxes = xywh_to_xyxy(candidates[:, :4])
    coefficients = candidates[:, 4 + num_classes :]

    kept = nms(boxes, scores, iou_threshold)

    image_h, image_w = image_shape
    input_h, input_w = input_size
    scale = np.array([image_w / input_w, image_h / input_h] * 2, dtype=np.float32)
    flat_prototypes = prototypes.reshape(num_masks, -1)

    detections = []
    for index in kept:
        box = np.clip(boxes[index] * scale, 0, [image_w, image_h, image_w, image_h])
        probability = _sigmoid(coefficients[index] @ flat_prototypes).reshape(mask_h, mask_w)
        full = cv2.resize(probability, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        mask = full > mask_threshold
        x1, y1 = int(np.floor(box[0])), int(np.floor(box[1]))
        x2, y2 = int(np.ceil(box[2])), int(np.ceil(box[3]))
        cropped = np.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        detections.append(
            Detection(
                box_xyxy=box.astype(np.float32),
                score=float(scores[index]),
                class_id=int(class_ids[index]),
                mask=cropped,
            )
        )
    return detections


def select_target(
    detections: Sequence[Detection],
    image_shape: tuple[int, int],
    reference_point: tuple[float, float] | None = None,
) -> Detection:
    """Pick the detection whose box centre is nearest the reference point.

    The reference point defaults to the image centre.
    """
    if not detections:
        raise MeasurementError("no object: the model returned no detection above the threshold")
    if reference_point is None:
        reference_point = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    ref_x, ref_y = reference_point
    return min(
        detections,
        key=lambda d: (d.center_xy[0] - ref_x) ** 2 + (d.center_xy[1] - ref_y) ** 2,
    )


class SegmentationModel:
    """Run an ONNX segmentation model and decode its output."""

    def __init__(
        self,
        session: Any,
        *,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self.input_size = input_size or _input_size_from_session(session)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        input_size: tuple[int, int] | None = None,
    ) -> SegmentationModel:
        """Load a model file. ``providers`` defaults to everything ONNX Runtime offers."""
        import onnxruntime

        if not Path(path).is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        session = onnxruntime.InferenceSession(
            str(path), providers=list(providers or onnxruntime.get_available_providers())
        )
        return cls(
            session,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
        )

    def predict(self, color_bgr: NDArray) -> list[Detection]:
        """Detect objects in a BGR image."""
        tensor = preprocess(color_bgr, self.input_size)
        outputs = self._session.run(None, {self._input_name: tensor})
        predictions = next((o for o in outputs if np.ndim(o) == 3), None)
        prototypes = next((o for o in outputs if np.ndim(o) == 4), None)
        if predictions is None or prototypes is None:
            raise ValueError(
                "the model must return a 3-D detections tensor and a 4-D prototypes tensor"
            )
        return decode_detections(
            predictions,
            prototypes,
            image_shape=color_bgr.shape[:2],
            input_size=self.input_size,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )


def _input_size_from_session(session: Any) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return (height, width)
    return DEFAULT_INPUT_SIZE


def _sigmoid(values: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))
