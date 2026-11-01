import numpy as np

from depth_volumetry.measurement import BoxMeasurement, Footprint
from depth_volumetry.overlay import draw_measurement


def test_draw_measurement_returns_annotated_copy():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    corners = np.array([[40, 40], [120, 40], [120, 100], [40, 100]], dtype=np.float32)
    footprint = Footprint((80, 70), 80.0, 60.0, corners)
    measurement = BoxMeasurement(60.0, 30.0, 50.0, 150.0)

    result = draw_measurement(image, footprint, measurement)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.any()  # something was drawn
    assert not image.any()  # the input was not modified
    assert result[40, 80].tolist() != [0, 0, 0]  # a point on the top edge of the rectangle
