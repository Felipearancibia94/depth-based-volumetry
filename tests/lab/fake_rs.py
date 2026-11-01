"""A minimal stand-in for ``pyrealsense2`` covering what the lab sources call.

It only checks that the code drives the API in a plausible way; it is not evidence that
the real SDK or a real camera behaves the same.
"""

from types import SimpleNamespace

import numpy as np


class _Enum:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    __str__ = __repr__


stream = SimpleNamespace(color=_Enum("stream.color"), depth=_Enum("stream.depth"))
format = SimpleNamespace(  # noqa: A001 - mirrors the SDK attribute name
    bgr8=_Enum("format.bgr8"), rgb8=_Enum("format.rgb8"), z16=_Enum("format.z16")
)
camera_info = SimpleNamespace(
    name="name",
    serial_number="serial_number",
    firmware_version="firmware_version",
    product_line="product_line",
    usb_type_descriptor="usb_type_descriptor",
)
INFO = {
    "name": "Fake RealSense D4xx",
    "serial_number": "000111",
    "firmware_version": "5.0.0",
    "product_line": "D400",
    "usb_type_descriptor": "3.2",
}
SHAPE = (48, 64)
calls = []


class _Intr:
    width, height, fx, fy, ppx, ppy = 64, 48, 50.0, 50.0, 32.0, 24.0


class _VideoProfile:
    intrinsics = _Intr()

    def __init__(self, fmt=None):
        self._fmt = fmt or format.bgr8

    def as_video_stream_profile(self):
        return self

    def format(self):
        return self._fmt

    def is_video_stream_profile(self):
        return True

    def stream_type(self):
        return stream.color

    def width(self):
        return 64

    def height(self):
        return 48

    def fps(self):
        return 30


class _Frame:
    def __init__(self, data, number, fmt=None):
        self._data, self._number = data, number
        self.profile = _VideoProfile(fmt)

    def __bool__(self):
        return True

    def get_data(self):
        return self._data

    def get_timestamp(self):
        return 1000.0 + self._number * 33.3

    def get_frame_number(self):
        return self._number

    def get_frame_timestamp_domain(self):
        return "global_time"


class _Frameset:
    def __init__(self, number, color_format=None):
        color = np.full((*SHAPE, 3), (10, 20, 30), np.uint8)
        self._color = _Frame(color, number, color_format)
        self._depth = _Frame(np.full(SHAPE, 2000, np.uint16), number)

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class _Sensor:
    def get_depth_scale(self):
        return 0.001

    def get_info(self, key):
        return "Stereo Module"

    def get_stream_profiles(self):
        return [_VideoProfile(), _VideoProfile()]


class _Playback:
    def set_real_time(self, value):
        calls.append(("set_real_time", value))


class _Device:
    def __init__(self, info=None):
        self._info = info or INFO

    def supports(self, key):
        return key in self._info

    def get_info(self, key):
        return self._info[key]

    def first_depth_sensor(self):
        return _Sensor()

    def query_sensors(self):
        return [_Sensor()]

    def as_playback(self):
        return _Playback()


class _Profile:
    def get_device(self):
        return _Device()


class config:
    def enable_stream(self, *args):
        calls.append(("enable_stream", args))

    def enable_device(self, serial):
        calls.append(("enable_device", serial))

    def enable_device_from_file(self, path, repeat_playback=True):
        calls.append(("enable_device_from_file", path, repeat_playback))

    def enable_record_to_file(self, path):
        calls.append(("enable_record_to_file", path))
        open(path, "wb").close()


class pipeline:
    frames_available = 3
    color_format = None

    def __init__(self):
        self._n = 0

    def start(self, cfg):
        calls.append(("start",))
        return _Profile()

    def stop(self):
        calls.append(("stop",))

    def wait_for_frames(self, timeout_ms=5000):
        self._n += 1
        return _Frameset(self._n, pipeline.color_format)

    def try_wait_for_frames(self, timeout_ms=5000):
        if self._n >= pipeline.frames_available:
            return False, None
        return True, self.wait_for_frames()


class align:
    def __init__(self, to):
        self.to = to

    def process(self, frames):
        calls.append(("align",))
        return frames


class context:
    devices = [_Device(), _Device({**INFO, "serial_number": "222", "usb_type_descriptor": "2.1"})]

    def query_devices(self):
        return context.devices
