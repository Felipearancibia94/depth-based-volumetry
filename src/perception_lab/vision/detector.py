"""Animal detectors.

:class:`YoloOnnxDetector` runs an ONNX model with the YOLOv8 output layout already used by
``depth_volumetry.segmentation`` (same input contract: RGB, ``[0, 1]``, resized without
letterboxing). Both layouts are accepted:

* detection: one output ``(1, 4 + C, N)``;
* instance segmentation: ``(1, 4 + C + M, N)`` plus prototypes ``(1, M, mh, mw)``; masks
  are then attached to detections and used for depth sampling.

Only the configured classes (COCO cat = 15, dog = 16 by default) are kept, before NMS, so
other classes cannot suppress an animal. No model is included; check the licence of the
weights you use (Ultralytics weights are AGPL-3.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from depth_volumetry.segmentation import decode_detections, nms, preprocess, xywh_to_xyxy
from perception_lab.core.config import DetectorConfig
from perception_lab.core.types import Detection, FrameBundle

DEFAULT_INPUT_SIZE = (640, 640)


class Detector(Protocol):
    def detect(self, bundle: FrameBundle) -> list[Detection]: ...


def decode_boxes(
    predictions: NDArray,
    *,
    image_shape: tuple[int, int],
    input_size: tuple[int, int],
    class_ids: Sequence[int],
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple[NDArray, float, int]]:
    """Decode a YOLOv8 detection output, keeping only ``class_ids``.

    Returns ``(box_xyxy, score, class_id)`` in original image pixels, best score first.
    """
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[0] < 5:
        raise ValueError(f"unexpected detection output shape {predictions.shape}")
    num_classes = predictions.shape[0] - 4
    ids = np.asarray(class_ids, dtype=int)
    if ids.size == 0 or ids.max() >= num_classes:
        raise ValueError(
            f"class ids {list(class_ids)} do not fit a model with {num_classes} classes"
        )
    candidates = predictions.T
    scores_all = candidates[:, 4:][:, ids]
    scores = scores_all.max(axis=1)
    keep = scores >= conf_threshold
    if not keep.any():
        return []
    boxes = xywh_to_xyxy(candidates[keep, :4])
    scores, labels = scores[keep], ids[scores_all[keep].argmax(axis=1)]
    image_h, image_w = image_shape
    input_h, input_w = input_size
    scale = np.array([image_w / input_w, image_h / input_h] * 2, dtype=np.float32)
    out = []
    for i in nms(boxes, scores, iou_threshold):
        box = np.clip(boxes[i] * scale, 0, [image_w, image_h, image_w, image_h])
        out.append((box, float(scores[i]), int(labels[i])))
    return out


def select_class_rows(predictions: NDArray, class_ids: Sequence[int], num_masks: int) -> NDArray:
    """Keep box rows, the rows of ``class_ids`` and the mask-coefficient rows."""
    if predictions.ndim == 3:
        predictions = predictions[0]
    num_classes = predictions.shape[0] - 4 - num_masks
    ids = np.asarray(class_ids, dtype=int)
    if num_classes < 1 or ids.size == 0 or ids.max() >= num_classes:
        raise ValueError(
            f"class ids {list(class_ids)} do not fit a model with {num_classes} classes"
        )
    rows = [0, 1, 2, 3, *(4 + ids).tolist(), *range(4 + num_classes, predictions.shape[0])]
    return predictions[rows]


class YoloOnnxDetector:
    def __init__(
        self,
        session: Any,
        *,
        class_ids: Sequence[int] = (15, 16),
        class_labels: Sequence[str] = ("cat", "dog"),
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        if len(class_ids) != len(class_labels):
            raise ValueError("class_ids and class_labels must have the same length")
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self.input_size = input_size or _input_size(session)
        self.class_ids = tuple(class_ids)
        self.labels = dict(zip(class_ids, class_labels, strict=True))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @classmethod
    def from_config(cls, config: DetectorConfig) -> YoloOnnxDetector:
        import onnxruntime

        if not config.model_path or not Path(config.model_path).is_file():
            raise FileNotFoundError(f"model file not found: {config.model_path}")
        providers = list(config.providers) or onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(config.model_path, providers=providers)
        return cls(
            session,
            class_ids=config.class_ids,
            class_labels=config.class_labels,
            conf_threshold=config.score_threshold,
            iou_threshold=config.iou_threshold,
        )

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        image = bundle.color_bgr
        outputs = self._session.run(None, {self._input_name: preprocess(image, self.input_size)})
        predictions = next((o for o in outputs if np.ndim(o) == 3), None)
        prototypes = next((o for o in outputs if np.ndim(o) == 4), None)
        if predictions is None:
            raise ValueError("the model must return a 3-D detections tensor")
        if prototypes is None:
            decoded = decode_boxes(
                predictions,
                image_shape=image.shape[:2],
                input_size=self.input_size,
                class_ids=self.class_ids,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
            return [
                Detection(tuple(float(v) for v in box), score, self.labels[cid], cid)
                for box, score, cid in decoded
            ]
        num_masks = np.shape(prototypes)[-3]
        selected = select_class_rows(predictions, self.class_ids, num_masks)
        found = decode_detections(
            selected,
            prototypes,
            image_shape=image.shape[:2],
            input_size=self.input_size,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        out = []
        for d in found:
            cid = self.class_ids[d.class_id]
            box = tuple(float(v) for v in d.box_xyxy)
            out.append(Detection(box, d.score, self.labels[cid], cid, d.mask))
        return out


class GroundTruthDetector:
    """Returns the boxes a :class:`SyntheticSource` stores in ``bundle.meta``.

    Optional degradations (missed detections, box jitter) make tracker and rule tests less
    idealised. Use it only with synthetic data.
    """

    def __init__(self, *, miss_rate: float = 0.0, jitter_px: float = 0.0, seed: int = 0) -> None:
        self.miss_rate = miss_rate
        self.jitter_px = jitter_px
        self._rng = np.random.default_rng(seed)

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        truth = bundle.meta.get("ground_truth")
        if truth is None:
            raise ValueError("GroundTruthDetector needs frames with meta['ground_truth']")
        out = []
        for obj in truth:
            if self.miss_rate and self._rng.random() < self.miss_rate:
                continue
            box = np.asarray(obj["box_xyxy"], dtype=np.float64)
            if self.jitter_px:
                box = box + self._rng.normal(0, self.jitter_px, 4)
            out.append(Detection(tuple(float(v) for v in box), 0.9, obj["label"]))
        return out


def create_detector(config: DetectorConfig) -> Detector:
    if config.kind == "ground_truth":
        return GroundTruthDetector(miss_rate=config.gt_miss_rate, jitter_px=config.gt_jitter_px)
    return YoloOnnxDetector.from_config(config)


def _input_size(session: Any) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return (height, width)
    return DEFAULT_INPUT_SIZE
