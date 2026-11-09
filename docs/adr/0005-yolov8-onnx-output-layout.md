# 0005. YOLOv8 ONNX output layout for the detector

Status: Accepted (2026-11-01)

## Context

The plan proposed YOLOX as the default detector. `depth_volumetry.segmentation` already
decodes the YOLOv8 output layout (detection and segmentation), with pure, tested functions.

## Decision

The lab detector (`vision/detector.py`) accepts any ONNX model with the YOLOv8 output layout
and reuses `decode_detections`, `nms`, `preprocess` and `xywh_to_xyxy` from `depth_volumetry`.
Segmentation models are preferred when available because masks improve depth sampling.

## Consequences

- No second decoder to maintain; one set of tests covers both uses.
- Ultralytics weights are AGPL-3.0: check the licence of whatever weights you use. No model
  binary is committed.
- Other detectors need an adapter or an export to the same layout.
