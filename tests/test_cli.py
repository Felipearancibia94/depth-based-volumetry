import json

import cv2
import numpy as np
import pytest

from depth_volumetry.cli import main
from depth_volumetry.io import write_image

SHAPE = (480, 640)
CALIBRATION_TEXT = "focal_length_px = 500\ncamera_height_cm = 200\n"


@pytest.fixture
def files(tmp_path):
    """A synthetic capture: 60 x 30 x 50 cm box (see test_measurement.py)."""
    corners = cv2.boxPoints(((320, 240), (200, 100), 30.0)).astype(np.int32)
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    depth = np.full(SHAPE, 2000, dtype=np.uint16)
    depth[mask > 0] = 1500
    color = np.full((*SHAPE, 3), 80, dtype=np.uint8)

    paths = {
        "mask": tmp_path / "mask.png",
        "depth": tmp_path / "depth.npy",
        "color": tmp_path / "color.png",
        "calibration": tmp_path / "calibration.toml",
        "overlay": tmp_path / "out" / "overlay.png",
    }
    write_image(paths["mask"], mask)
    np.save(paths["depth"], depth)
    write_image(paths["color"], color)
    paths["calibration"].write_text(CALIBRATION_TEXT)
    return paths


def measure_args(files, *extra):
    return [
        "measure",
        "--depth", str(files["depth"]),
        "--mask", str(files["mask"]),
        "--calibration", str(files["calibration"]),
        *extra,
    ]  # fmt: skip


def test_measure_with_mask_prints_json(files, capsys):
    assert main(measure_args(files, "--json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["length_cm"] == pytest.approx(60.0, rel=0.01)
    assert result["width_cm"] == pytest.approx(30.0, rel=0.02)
    assert result["height_cm"] == pytest.approx(50.0, rel=0.01)
    assert result["surface_distance_cm"] == pytest.approx(150.0)
    assert set(result) == {
        "length_cm",
        "width_cm",
        "height_cm",
        "surface_distance_cm",
        "volume_cm3",
    }


def test_measure_prints_plain_text_by_default(files, capsys):
    assert main(measure_args(files)) == 0
    out = capsys.readouterr().out
    assert "height_cm: 50.0" in out
    assert "volume_cm3:" in out


def test_measure_writes_overlay(files, capsys):
    args = measure_args(files, "--color", str(files["color"]), "--overlay", str(files["overlay"]))
    assert main(args) == 0
    overlay = cv2.imread(str(files["overlay"]))
    assert overlay.shape == (*SHAPE, 3)
    assert (overlay != 80).any()


def test_overlay_requires_color(files, capsys):
    assert main(measure_args(files, "--overlay", str(files["overlay"]))) == 2
    assert "--color is required" in capsys.readouterr().err


def test_mask_and_model_are_mutually_exclusive(files, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*measure_args(files), "--model", "m.onnx"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_a_source_is_required(files, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            ["measure", "--depth", str(files["depth"]), "--calibration", str(files["calibration"])]
        )
    assert exc.value.code == 2


def test_measurement_failure_exit_code(files, capsys):
    write_image(files["mask"], np.zeros(SHAPE, dtype=np.uint8))
    assert main(measure_args(files)) == 1
    assert "no object" in capsys.readouterr().err


def test_bad_calibration_exit_code(files, capsys):
    files["calibration"].write_text("focal_length_px = 500\n")
    assert main(measure_args(files)) == 2
    assert "camera_height_cm" in capsys.readouterr().err


def test_missing_input_file(files, capsys):
    files["depth"].unlink()
    assert main(measure_args(files)) == 2
    assert "file not found" in capsys.readouterr().err


def test_color_depth_shape_mismatch(files, capsys):
    write_image(files["color"], np.zeros((10, 10, 3), dtype=np.uint8))
    assert main(measure_args(files, "--color", str(files["color"]))) == 2
    assert "same" in capsys.readouterr().err


def test_capture_without_realsense_extra(tmp_path, capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    assert main(["capture", "--out", str(tmp_path)]) == 2
    assert "--extra realsense" in capsys.readouterr().err


@pytest.mark.parametrize("command", [[], ["measure"], ["capture"]])
def test_help_lists_flags(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*command, "--help"])
    assert exc.value.code == 0
    assert "usage: depth-volumetry" in capsys.readouterr().out


def test_measure_with_model_uses_nearest_detection(files, capsys, monkeypatch):
    from depth_volumetry import segmentation
    from depth_volumetry.segmentation import Detection

    mask = cv2.imread(str(files["mask"]), cv2.IMREAD_GRAYSCALE) > 0
    box = np.array([220, 190, 420, 290], dtype=np.float32)
    target = Detection(box_xyxy=box, score=0.9, class_id=0, mask=mask)
    distractor = Detection(
        box_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        score=0.95,
        class_id=0,
        mask=np.zeros(SHAPE, dtype=bool),
    )

    class StubModel:
        def predict(self, color):
            assert color.shape == (*SHAPE, 3)
            return [distractor, target]

    seen = {}

    def fake_from_file(path, **kwargs):
        seen.update(path=str(path), **kwargs)
        return StubModel()

    monkeypatch.setattr(segmentation.SegmentationModel, "from_file", staticmethod(fake_from_file))
    args = [
        "measure",
        "--depth", str(files["depth"]),
        "--color", str(files["color"]),
        "--model", "model.onnx",
        "--calibration", str(files["calibration"]),
        "--confidence", "0.7",
        "--json",
    ]  # fmt: skip
    assert main(args) == 0
    assert seen == {"path": "model.onnx", "conf_threshold": 0.7}
    assert json.loads(capsys.readouterr().out)["height_cm"] == pytest.approx(50.0, rel=0.01)


def test_model_requires_color(files, capsys):
    args = ["measure", "--depth", str(files["depth"]), "--model", "m.onnx",
            "--calibration", str(files["calibration"])]  # fmt: skip
    assert main(args) == 2
    assert "--color is required" in capsys.readouterr().err
