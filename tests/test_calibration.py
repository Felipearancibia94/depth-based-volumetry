from pathlib import Path

import pytest

from depth_volumetry.calibration import Calibration, CalibrationError, load_calibration

EXAMPLE = Path(__file__).parent.parent / "examples" / "calibration.example.toml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "calibration.toml"
    path.write_text(text)
    return path


def test_example_file_loads():
    calibration = load_calibration(EXAMPLE)
    assert calibration.focal_length_px == 600.0
    assert calibration.camera_height_cm == 200.0
    assert calibration.depth_percentiles == (20.0, 80.0)
    assert calibration.reference_point is None


def test_defaults_applied_for_optional_keys(tmp_path):
    calibration = load_calibration(
        write(tmp_path, "focal_length_px = 500\ncamera_height_cm = 150\n")
    )
    assert calibration == Calibration(focal_length_px=500.0, camera_height_cm=150.0)


def test_reference_point_is_read(tmp_path):
    calibration = load_calibration(
        write(
            tmp_path,
            "focal_length_px = 500\ncamera_height_cm = 150\nreference_point = [10, 20.5]\n",
        )
    )
    assert calibration.reference_point == (10.0, 20.5)


def test_missing_file(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path):
    with pytest.raises(CalibrationError, match="invalid TOML"):
        load_calibration(write(tmp_path, "focal_length_px = = 1"))


def test_missing_required_key(tmp_path):
    with pytest.raises(CalibrationError, match="camera_height_cm"):
        load_calibration(write(tmp_path, "focal_length_px = 500\n"))


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(CalibrationError, match="focal_length"):
        load_calibration(
            write(tmp_path, "focal_length = 500\nfocal_length_px = 500\ncamera_height_cm = 150\n")
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"focal_length_px": 0}, "focal_length_px"),
        ({"camera_height_cm": -1}, "camera_height_cm"),
        ({"base_height_cm": -1}, "base_height_cm"),
        ({"base_height_cm": 200}, "smaller than camera_height_cm"),
        ({"depth_unit_mm": 0}, "depth_unit_mm"),
        ({"distance_factor": float("nan")}, "distance_factor"),
        ({"depth_percentiles": (80, 20)}, "depth_percentiles"),
        ({"depth_percentiles": (-5, 50)}, "depth_percentiles"),
        ({"reference_point": (float("inf"), 0)}, "reference_point"),
    ],
)
def test_invalid_values(overrides, message):
    values = {"focal_length_px": 500.0, "camera_height_cm": 150.0, **overrides}
    with pytest.raises(CalibrationError, match=message):
        Calibration(**values)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"focal_length_px": "500", "camera_height_cm": 1}, "focal_length_px must be a number"),
        ({"focal_length_px": True, "camera_height_cm": 1}, "focal_length_px must be a number"),
        (
            {"focal_length_px": 5, "camera_height_cm": 9, "depth_percentiles": [1]},
            "list of two numbers",
        ),
    ],
)
def test_wrong_types(data, message):
    with pytest.raises(CalibrationError, match=message):
        Calibration.from_mapping(data)
