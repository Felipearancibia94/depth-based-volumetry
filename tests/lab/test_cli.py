import json

import pytest

from perception_lab.cli import main
from perception_lab.core.config import load_config
from perception_lab.spatial.scene import load_scene

from . import fake_rs


def _config_file(tmp_path, extra=""):
    path = tmp_path / "lab.toml"
    path.write_text(
        f"""
[source]
fps = 5
synthetic_seconds = 40.0

[storage]
db_path = "{(tmp_path / "lab.db").as_posix()}"
media_dir = "{(tmp_path / "media").as_posix()}"

[notify]
desktop = false
{extra}
"""
    )
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: lab" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    [
        "init-config",
        "run",
        "serve",
        "discover",
        "record",
        "replay",
        "calibrate",
        "eval",
        "summary",
        "purge",
    ],
)
def test_subcommand_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0


def test_init_config_writes_loadable_toml(tmp_path, capsys):
    out = tmp_path / "lab.toml"
    assert main(["init-config", "--out", str(out)]) == 0
    assert load_config(out).source.kind == "synthetic"
    assert main(["init-config", "--out", str(out)]) == 2
    assert "--force" in capsys.readouterr().err
    assert main(["init-config", "--out", str(out), "--force"]) == 0


def test_run_synthetic_prints_events_and_summary(tmp_path, capsys):
    assert main(["run", "--config", str(_config_file(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "possible eating" in out
    summary = json.loads(out[out.index("{") :])
    assert summary["frames"] == 200 and summary["events"] >= 2


def test_run_frames_override(tmp_path, capsys):
    assert main(["run", "--config", str(_config_file(tmp_path)), "--frames", "10", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 10


def test_bad_config_exit_code(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text("[source]\nkind = 'webcam'\n")
    assert main(["run", "--config", str(path)]) == 2
    assert "must be one of" in capsys.readouterr().err


def test_video_without_path_is_a_usage_error(capsys):
    assert main(["run", "--source", "video"]) == 2
    assert "source.path" in capsys.readouterr().err


def test_replay_synthetic_with_previews(tmp_path, capsys):
    cfg = _config_file(tmp_path)
    previews = tmp_path / "previews"
    args = [
        "replay",
        "--config",
        str(cfg),
        "--frames",
        "20",
        "--preview-dir",
        str(previews),
        "--preview-every",
        "10",
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 20 and report["mean_fps"] == pytest.approx(5.0)
    assert report["depth_valid_fraction_median"] > 0.9
    assert len(list(previews.glob("*.jpg"))) == 2


def test_calibrate_writes_floor_and_keeps_zones(tmp_path, capsys):
    cfg = _config_file(tmp_path)
    scene_path = tmp_path / "scene.toml"
    scene_path.write_text(
        '[[zones]]\nname = "bowl"\ntags = ["food"]\npolygon_xy_px = [[0, 0], [9, 0], [9, 9]]\n'
    )
    assert (
        main(["calibrate", "--config", str(cfg), "--out", str(scene_path), "--calib-frames", "3"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["camera_height_above_floor_m"] == pytest.approx(1.8, abs=0.05)
    assert report["tilt_from_vertical_deg"] == pytest.approx(25.0, abs=1.5)
    scene = load_scene(scene_path)
    assert scene.floor is not None and scene.zones[0].name == "bowl"


def test_purge_dry_run(tmp_path, capsys):
    assert main(["purge", "--config", str(_config_file(tmp_path)), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_summary_after_a_run(tmp_path, capsys):
    config = str(_config_file(tmp_path))
    assert main(["run", "--config", config, "--quiet"]) == 0
    capsys.readouterr()
    assert main(["summary", "--config", config]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["events"] > 0 and "unnamed" in report["pets"]
    assert main(["summary", "--config", config, "--date", "2001-01-01"]) == 0
    assert json.loads(capsys.readouterr().out)["events"] == 0
    assert main(["summary", "--config", config, "--date", "yesterday"]) == 2


def test_discover_without_sdk(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    assert main(["discover"]) == 2
    assert "--extra realsense" in capsys.readouterr().err


def test_discover_with_fake_sdk(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    assert main(["discover"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["serial_number"] == "000111"


def test_eval_synthetic_prints_report(tmp_path, capsys):
    args = ["eval", "--config", str(_config_file(tmp_path)), "--out-dir", str(tmp_path / "runs")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "## Rule events" in out and "results:" in out


def test_bench_detector_ground_truth(tmp_path, capsys):
    out = tmp_path / "bench.json"
    args = [
        "bench",
        "detector",
        "--config",
        str(_config_file(tmp_path)),
        "--frames",
        "5",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    assert json.loads(out.read_text())["latency"]["n"] == 5


def test_bench_vlm_unreachable_runtime_reports_errors(tmp_path, capsys):
    cfg = _config_file(tmp_path, '\n[vlm]\nbase_url = "http://127.0.0.1:9"\ntimeout_s = 2.0\n')
    assert main(["bench", "vlm", "--config", str(cfg), "--images", "1", "--repeats", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["runs"]["1"]["schema_valid"] == 0 and result["runs"]["1"]["errors"]
