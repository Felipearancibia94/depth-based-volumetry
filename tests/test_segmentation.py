import numpy as np
import pytest

from depth_volumetry.measurement import MeasurementError
from depth_volumetry.segmentation import (
    Detection,
    SegmentationModel,
    decode_detections,
    nms,
    preprocess,
    select_target,
    xywh_to_xyxy,
)

INPUT_SIZE = (64, 64)  # (height, width)
IMAGE_SHAPE = (128, 128)  # (height, width): the image is twice the model input
MASK_SIZE = 16


def make_prototypes():
    """Two prototypes: the first is positive inside a rectangle, the second is all negative."""
    protos = np.full((1, 2, MASK_SIZE, MASK_SIZE), -10.0, dtype=np.float32)
    protos[0, 0, 6:10, 5:11] = 10.0  # rows 6-9, cols 5-10 -> image x 40..88, y 48..80
    return protos


def make_predictions(rows):
    """rows: list of (cx, cy, w, h, score) in model input pixels; one class, two coefficients."""
    n = len(rows)
    out = np.zeros((1, 4 + 1 + 2, n), dtype=np.float32)
    for i, (cx, cy, w, h, score) in enumerate(rows):
        out[0, :, i] = [cx, cy, w, h, score, 1.0, 0.0]
    return out


def decode(rows, **kwargs):
    return decode_detections(
        make_predictions(rows),
        make_prototypes(),
        image_shape=IMAGE_SHAPE,
        input_size=INPUT_SIZE,
        **kwargs,
    )


def test_xywh_to_xyxy():
    boxes = xywh_to_xyxy(np.array([[10.0, 20.0, 4.0, 6.0]]))
    assert boxes.tolist() == [[8.0, 17.0, 12.0, 23.0]]


def test_nms_suppresses_overlapping_lower_scores():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, 0.5) == [1, 2]


def test_nms_keeps_everything_when_threshold_is_high():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    assert sorted(nms(boxes, np.array([0.9, 0.8]), 0.95)) == [0, 1]


def test_preprocess_layout_and_range():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue channel in BGR
    tensor = preprocess(image, (8, 16))
    assert tensor.shape == (1, 3, 8, 16)
    assert tensor.dtype == np.float32
    assert tensor[0, 2].min() == pytest.approx(1.0)  # blue ends up last in RGB
    assert tensor[0, 0].max() == 0.0


def test_preprocess_rejects_grayscale():
    with pytest.raises(ValueError, match="H x W x 3"):
        preprocess(np.zeros((4, 4), dtype=np.uint8), (8, 8))


def test_decode_reproduces_known_rectangle():
    # Box centred at (32, 32) in model pixels, 24 x 16 -> image x 40..88, y 48..80.
    (detection,) = decode([(32, 32, 24, 16, 0.9)])
    assert detection.box_xyxy.tolist() == [40.0, 48.0, 88.0, 80.0]
    assert detection.score == pytest.approx(0.9)
    assert detection.mask.shape == IMAGE_SHAPE
    assert detection.mask.dtype == bool
    ys, xs = np.nonzero(detection.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == pytest.approx((40, 87, 48, 79), abs=5)
    assert detection.mask.sum() == pytest.approx(48 * 32, rel=0.1)


def test_decode_filters_low_confidence_and_overlaps():
    detections = decode(
        [
            (32, 32, 24, 16, 0.9),
            (33, 32, 24, 16, 0.8),  # near duplicate of the first
            (10, 10, 8, 8, 0.3),  # below the threshold
            (54, 54, 8, 8, 0.7),  # separate object
        ]
    )
    assert [round(d.score, 1) for d in detections] == [0.9, 0.7]


def test_decode_without_detections():
    assert decode([(32, 32, 24, 16, 0.2)]) == []


def test_decode_mask_is_cropped_to_box():
    # The prototype is positive in a region the second box does not cover.
    (detection,) = decode([(54, 54, 8, 8, 0.9)])
    assert not detection.mask.any()


def test_decode_accepts_arrays_without_batch_axis():
    detections = decode_detections(
        make_predictions([(32, 32, 24, 16, 0.9)])[0],
        make_prototypes()[0],
        image_shape=IMAGE_SHAPE,
        input_size=INPUT_SIZE,
    )
    assert len(detections) == 1


def test_decode_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="too few"):
        decode_detections(
            np.zeros((1, 5, 3), dtype=np.float32),
            make_prototypes(),
            image_shape=IMAGE_SHAPE,
            input_size=INPUT_SIZE,
        )


def make_detection(center, size=10.0):
    cx, cy = center
    box = np.array([cx - size, cy - size, cx + size, cy + size], dtype=np.float32)
    return Detection(box_xyxy=box, score=0.9, class_id=0, mask=np.zeros((100, 100), dtype=bool))


def test_select_target_defaults_to_image_centre():
    near, far = make_detection((48, 52)), make_detection((10, 10))
    assert select_target([far, near], (100, 100)) is near


def test_select_target_uses_reference_point():
    near, far = make_detection((48, 52)), make_detection((10, 10))
    assert select_target([far, near], (100, 100), reference_point=(5, 5)) is far


def test_select_target_without_detections():
    with pytest.raises(MeasurementError, match="no object"):
        select_target([], (100, 100))


class FakeIO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSession:
    """Stands in for an ONNX Runtime session; returns the synthetic outputs above."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.last_input = None

    def get_inputs(self):
        return [FakeIO("images", self.input_shape)]

    def run(self, _outputs, feeds):
        self.last_input = feeds["images"]
        return [make_prototypes(), make_predictions([(32, 32, 24, 16, 0.9)])]  # order swapped


def test_model_predict_with_fake_session():
    session = FakeSession([1, 3, 64, 64])
    model = SegmentationModel(session)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    (detection,) = model.predict(image)
    assert session.last_input.shape == (1, 3, 64, 64)
    assert detection.mask.shape == (128, 128)


def test_model_uses_default_size_for_dynamic_axes():
    model = SegmentationModel(FakeSession([1, 3, "height", "width"]))
    assert model.input_size == (640, 640)


def test_from_file_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="model file not found"):
        SegmentationModel.from_file(tmp_path / "missing.onnx")
