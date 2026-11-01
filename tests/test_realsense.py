import sys

import pytest

from depth_volumetry.realsense import RealSenseUnavailableError, capture_frames


def test_missing_dependency_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)  # makes `import pyrealsense2` fail
    with pytest.raises(RealSenseUnavailableError, match="--extra realsense"):
        capture_frames()
