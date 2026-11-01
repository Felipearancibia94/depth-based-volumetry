import numpy as np
import pytest

from perception_lab.core.config import TrackerConfig
from perception_lab.core.types import Detection, FrameBundle, SpatialObs, TrackObservation
from perception_lab.sources.synthetic import SyntheticSource
from perception_lab.spatial.scene import Zone
from perception_lab.vision.detector import (
    GroundTruthDetector,
    YoloOnnxDetector,
    decode_boxes,
    select_class_rows,
)
from perception_lab.vision.draw import draw_observations
from perception_lab.vision.tracker import Tracker

NUM_CLASSES = 80


def _predictions(rows, num_masks=0, n=8):
    """(1, 4 + 80 + M, N) tensor in 640x640 input pixels; rows = [(cx, cy, w, h, cls, score)]."""
    pred = np.zeros((4 + NUM_CLASSES + num_masks, n), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        pred[:4, i] = (cx, cy, w, h)
        pred[4 + cls, i] = score
        if num_masks:
            pred[4 + NUM_CLASSES :, i] = 1.0
    return pred[np.newaxis]


def test_decode_boxes_keeps_only_animals_before_nms():
    rows = [
        (320, 320, 200, 200, 0, 0.95),  # person, overlapping the cat: must not suppress it
        (320, 320, 180, 180, 15, 0.80),  # cat
        (322, 322, 178, 178, 16, 0.40),  # dog duplicate of the same animal: suppressed
        (100, 100, 50, 50, 16, 0.70),  # dog
        (500, 500, 50, 50, 16, 0.10),  # below threshold
    ]
    out = decode_boxes(
        _predictions(rows),
        image_shape=(480, 640),
        input_size=(640, 640),
        class_ids=(15, 16),
        conf_threshold=0.3,
        iou_threshold=0.5,
    )
    assert [(cid, round(score, 2)) for _, score, cid in out] == [(15, 0.8), (16, 0.7)]
    box = out[0][0]
    assert box[1] == pytest.approx((320 - 90) * 480 / 640)  # rescaled to the image


def test_decode_boxes_rejects_bad_class_ids():
    with pytest.raises(ValueError, match="do not fit"):
        decode_boxes(
            _predictions([]),
            image_shape=(480, 640),
            input_size=(640, 640),
            class_ids=(99,),
            conf_threshold=0.3,
            iou_threshold=0.5,
        )


def test_select_class_rows_layout():
    pred = _predictions([(1, 2, 3, 4, 16, 0.9)], num_masks=2)
    selected = select_class_rows(pred, (15, 16), 2)
    assert selected.shape == (4 + 2 + 2, 8)
    assert selected[5, 0] == pytest.approx(0.9) and selected[4, 0] == 0.0


class _FakeSession:
    def __init__(self, outputs):
        self._outputs = outputs
        self.inputs = []

    def get_inputs(self):
        class _Input:
            name, shape = "images", [1, 3, 640, 640]

        return [_Input()]

    def run(self, _, feeds):
        self.inputs.append(feeds["images"].shape)
        return self._outputs


def _bundle():
    return FrameBundle(0, 0.0, np.zeros((480, 640, 3), np.uint8))


def test_yolo_detector_detection_layout():
    session = _FakeSession([_predictions([(320, 320, 100, 100, 15, 0.9)])])
    found = YoloOnnxDetector(session).detect(_bundle())
    assert session.inputs == [(1, 3, 640, 640)]
    assert [(d.label, d.class_id) for d in found] == [("cat", 15)]
    assert found[0].mask is None


def test_yolo_detector_segmentation_layout_attaches_masks():
    prototypes = np.full((1, 2, 160, 160), 5.0, np.float32)
    session = _FakeSession([_predictions([(320, 320, 100, 100, 16, 0.9)], num_masks=2), prototypes])
    (dog,) = YoloOnnxDetector(session).detect(_bundle())
    assert dog.label == "dog" and dog.mask.shape == (480, 640)
    assert dog.mask.any() and not dog.mask[0, 0]  # mask cropped to the box


def test_ground_truth_detector():
    bundle = SyntheticSource(fps=1, duration_s=1).render(0)
    found = GroundTruthDetector().detect(bundle)
    assert {d.label for d in found} == {"cat", "dog"}
    assert GroundTruthDetector(miss_rate=1.0).detect(bundle) == []
    with pytest.raises(ValueError):
        GroundTruthDetector().detect(_bundle())


def _det(x, y, label="cat", score=0.9, size=40):
    return Detection((x, y, x + size, y + size), score, label)


def test_tracker_confirms_after_min_hits_and_keeps_ids():
    tracker = Tracker(TrackerConfig(min_hits=3))
    outputs = [
        tracker.update([_det(10 + 5 * i, 10), _det(300 - 5 * i, 200, "dog")], i * 0.1)
        for i in range(6)
    ]
    assert outputs[0] == [] and outputs[1] == []
    ids = [{t.track_id: t.label for t in out} for out in outputs[2:]]
    assert all(i == ids[0] for i in ids)
    assert sorted(ids[0].values()) == ["cat", "dog"]


def test_tracker_survives_short_gap_and_uses_low_score_detections():
    tracker = Tracker(TrackerConfig(min_hits=2, max_lost_s=1.0))
    outputs = [tracker.update([_det(10 + 5 * i, 10)], i * 0.1) for i in range(3)]
    assert outputs[0] == []  # tentative until min_hits
    tid = outputs[-1][0].track_id
    assert tracker.update([], 0.3) == []  # missed frame: not reported, still alive
    (again,) = tracker.update([_det(30, 10, score=0.3)], 0.4)  # weak detection keeps it
    assert again.track_id == tid
    tracker.update([], 2.0)  # lost longer than max_lost_s
    assert tracker.tracks == []


def test_tracker_does_not_create_tracks_from_low_scores():
    tracker = Tracker(TrackerConfig(min_hits=1))
    assert tracker.update([_det(10, 10, score=0.2)], 0.0) == []
    assert tracker.tracks == []


def test_tracker_label_is_score_weighted_vote():
    tracker = Tracker(TrackerConfig(min_hits=1))
    tracker.update([_det(10, 10, "cat", 0.9)], 0.0)
    tracker.update([_det(10, 10, "dog", 0.55)], 0.1)
    (track,) = tracker.update([_det(10, 10, "cat", 0.9)], 0.2)
    assert track.label == "cat"


def test_draw_observations_marks_pixels():
    obs = TrackObservation(
        0.0, 3, "cat", (10, 20, 60, 70), 0.9, SpatialObs(2.0, 1.0, elevation_m=0.1)
    )
    zone = Zone("bowl", ((100, 100), (200, 100), (200, 200)), ("restricted",))
    canvas = draw_observations(np.zeros((240, 320, 3), np.uint8), [obs], [zone])
    assert canvas[20, 30].any() and canvas[100, 150].any()
