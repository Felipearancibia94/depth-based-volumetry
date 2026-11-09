---
paths:
  - "src/depth_volumetry/**"
---

# depth_volumetry rules

- `measurement.py` and the decode functions in `segmentation.py` are pure: no file, network or
  model access. Keep I/O in `io.py`, `cli.py` and the ONNX model wrapper.
- A failure to measure raises `MeasurementError`; invalid configuration raises
  `CalibrationError`. Never clamp, default or substitute a value to avoid an error.
- `pyrealsense2` is an optional extra and stays a lazy import (`realsense.load_pyrealsense2`).
  Do not import it at module top level.
- `perception_lab` imports from this package; this package must never import `perception_lab`.
  Changing `segmentation.decode_detections`, `nms`, `preprocess`, `xywh_to_xyxy`,
  `realsense.load_pyrealsense2` or `io.read_color` / `io.write_image` affects the lab: run the
  whole suite.
- Geometry and the model contract are defined in `docs/method.md`; update it with any change
  to the formulas or the expected model outputs.
- A changed CLI flag needs the README table and `tests/test_cli.py` updated (`/cli-change`).
