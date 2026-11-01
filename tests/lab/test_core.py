import numpy as np
import pytest

from perception_lab.core.config import (
    ConfigError,
    LabConfig,
    config_from_mapping,
    dump_config,
    load_config,
)
from perception_lab.core.runrecord import build_run_record, write_json
from perception_lab.core.types import FrameBundle, Intrinsics, box_iou, box_to_slices


def test_intrinsics_roundtrip():
    k = Intrinsics(640, 480, 600.0, 610.0, 320.0, 240.0)
    point = k.deproject_m(400.0, 100.0, 2.0)
    assert k.project_px(point) == pytest.approx((400.0, 100.0))
    assert point[2] == 2.0


def test_depth_m_uses_unit():
    depth = np.full((2, 2), 1500, np.uint16)
    bundle = FrameBundle(0, 0.0, np.zeros((2, 2, 3), np.uint8), depth, depth_unit_mm=1.0)
    assert bundle.depth_m()[0, 0] == pytest.approx(1.5)
    assert not bundle.has_depth  # no intrinsics


def test_box_helpers():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    rows, cols = box_to_slices((-5.0, 2.4, 700.0, 9.6), (480, 640))
    assert (rows.start, rows.stop, cols.start, cols.stop) == (2, 10, 0, 640)


def test_defaults_are_local_first():
    cfg = LabConfig()
    assert cfg.web.host == "127.0.0.1"
    assert cfg.vlm.enabled is False
    assert cfg.vlm.allow_remote is False
    assert cfg.source.kind == "synthetic"


def test_config_toml_roundtrip(tmp_path):
    path = tmp_path / "lab.toml"
    path.write_text(dump_config(LabConfig()))
    assert load_config(path) == LabConfig()


def test_partial_config_keeps_defaults():
    cfg = config_from_mapping({"vlm": {"enabled": True}, "rules": {"dwell_tags": {"food": 5}}})
    assert cfg.vlm.enabled and cfg.vlm.model == LabConfig().vlm.model
    assert cfg.rules.dwell_tags == {"food": 5.0}


@pytest.mark.parametrize(
    "data, message",
    [
        ({"source": {"bogus": 1}}, "unknown key"),
        ({"source": {"kind": "webcam"}}, "must be one of"),
        ({"source": {"fps": "15"}}, "integer"),
        ({"vlm": {"enabled": 1}}, "true or false"),
        ({"detector": {"class_ids": [15]}}, "equal length"),
        ({"rules": []}, "must be a table"),
    ],
)
def test_invalid_config_is_rejected(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(data)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_run_record(tmp_path):
    record = build_run_record(LabConfig(), {"suite": "unit"})
    assert record["suite"] == "unit"
    assert len(record["config_sha256"]) == 64
    assert write_json(tmp_path / "r" / "run.json", record).is_file()
