---
paths:
  - "tests/**"
---

# Test rules

- Synthetic arrays and tensors only. Tests need no network, camera, GPU or model file.
- One test module per source module; lab tests live in `tests/lab/`.
- Use `tests/lab/fake_rs.py` for the RealSense SDK and a fake HTTP server for the VLM
  providers. Do not call a real Ollama or open a real device.
- Never commit model binaries (`*.onnx`), recordings (`*.bag`), databases or real calibration
  files; build the data in the test or in a fixture.
- Assert behaviour and error types (`MeasurementError`, `CalibrationError`), not only that
  something ran. A fix for a bug gets a test that fails without it.
- Do not loosen a tolerance or delete a test to make a change pass; explain why instead.
