import pytest

from perception_lab.cli import main
from perception_lab.core.config import load_config


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: lab" in capsys.readouterr().out


def test_init_config_writes_loadable_toml(tmp_path, capsys):
    out = tmp_path / "lab.toml"
    assert main(["init-config", "--out", str(out)]) == 0
    assert load_config(out).source.kind == "synthetic"
    assert main(["init-config", "--out", str(out)]) == 2
    assert "--force" in capsys.readouterr().err
    assert main(["init-config", "--out", str(out), "--force"]) == 0
