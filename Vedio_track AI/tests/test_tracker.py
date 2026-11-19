"""Tracker wrapper tests.

* The adapter / parsing / trail logic is tested with a stub backend.
* ``test_bytetrack_*`` / ``test_botsort_*`` exercise the REAL Ultralytics
  trackers and are skipped when ``ultralytics`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from backend.detector import Detections
from backend.tracker import ObjectTracker, TrackedObjects, TrailBuffer, TrackerError, build_backend, _DetectionBatch


def make_detections(boxes, conf=0.9, cls=0) -> Detections:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return Detections(boxes, np.full(len(boxes), conf, dtype=np.float32), np.full(len(boxes), cls, dtype=np.int64))


class StubBackend:
    """Records what it was called with and returns canned rows."""

    def __init__(self, output):
        self.output = output
        self.seen = []
        self.was_reset = False

    def update(self, batch, frame=None):
        self.seen.append(batch)
        return self.output

    def reset(self):
        self.was_reset = True


def test_batch_adapter_exposes_ultralytics_interface():
    det = make_detections([[10, 10, 30, 50], [100, 100, 140, 120]])
    batch = _DetectionBatch(det.xywh, det.conf, det.cls)
    assert len(batch) == 2
    assert batch.xywh[0].tolist() == [20.0, 30.0, 20.0, 40.0]
    sub = batch[np.array([False, True])]
    assert len(sub) == 1 and sub.xywh[0].tolist() == [120.0, 110.0, 40.0, 20.0]
    assert np.allclose(batch.xyxy, det.xyxy)


def test_update_parses_tracker_rows():
    rows = np.array([[10, 10, 50, 60, 7, 0.91, 0, 0], [80, 20, 120, 90, 9, 0.55, 2, 1]], dtype=np.float32)
    tracker = ObjectTracker(backend=StubBackend(rows))
    out = tracker.update(make_detections([[10, 10, 50, 60], [80, 20, 120, 90]]))
    assert out.track_ids.tolist() == [7, 9]
    assert out.cls.tolist() == [0, 2]
    assert out.conf.tolist() == pytest.approx([0.91, 0.55])
    assert out.centers.tolist() == [[30.0, 35.0], [100.0, 55.0]]


@pytest.mark.parametrize("empty_output", [np.empty((0,), dtype=np.float32), np.zeros((0, 8)), []])
def test_update_handles_no_tracks(empty_output):
    tracker = ObjectTracker(backend=StubBackend(empty_output))
    out = tracker.update(Detections.empty())
    assert len(out) == 0 and out.xyxy.shape == (0, 4)


def test_update_is_called_even_without_detections():
    backend = StubBackend(np.empty((0,), dtype=np.float32))
    ObjectTracker(backend=backend).update(Detections.empty())
    assert len(backend.seen) == 1  # lost tracks must still age


def test_reset_is_forwarded():
    backend = StubBackend([])
    ObjectTracker(backend=backend).reset()
    assert backend.was_reset


def test_unknown_tracker_type_is_rejected():
    with pytest.raises(TrackerError):
        build_backend("not-a-tracker")


def test_trail_buffer_limits_and_prunes():
    trails = TrailBuffer(max_len=5, max_stale_frames=10)
    for step in range(20):
        trails.update(np.array([1]), np.array([[step, step]], dtype=np.float32))
    assert len(trails.trail(1)) == 5 and trails.trail(1)[-1] == (19, 19)
    trails.set_max_len(3)
    assert len(trails.trail(1)) == 3
    for _ in range(60):  # id 1 disappears; id 2 keeps going
        trails.update(np.array([2]), np.array([[0, 0]], dtype=np.float32))
    assert trails.trail(1) == []
    trails.clear()
    assert trails.trail(2) == []


# --------------------------------------------------------------------------- #
# Real Ultralytics trackers
# --------------------------------------------------------------------------- #
def _moving_boxes(step: int):
    """Two well separated objects moving smoothly: ground truth identities are 'A' and 'B'."""
    a = [20 + 4 * step, 30, 80 + 4 * step, 130]
    b = [400 - 3 * step, 200, 470 - 3 * step, 330]
    return [a, b]


@pytest.mark.parametrize("tracker_type", ["bytetrack", "botsort"])
def test_real_tracker_keeps_ids_consistent(tracker_type):
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    tracker = ObjectTracker(tracker_type, frame_rate=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    history = []
    for step in range(25):
        det = make_detections(_moving_boxes(step), conf=0.9)
        history.append(tracker.update(det, frame))

    # ByteTrack/BoT-SORT confirm a new track on its 2nd frame, so skip the first frame.
    confirmed = history[2:]
    assert all(len(h) == 2 for h in confirmed)
    ids_a = {int(h.track_ids[np.argmin(h.centers[:, 0])]) for h in confirmed}
    ids_b = {int(h.track_ids[np.argmax(h.centers[:, 0])]) for h in confirmed}
    # The left-most object is A for the whole sequence, the other is B.
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_real_bytetrack_recovers_id_after_short_occlusion():
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    tracker = ObjectTracker("bytetrack", frame_rate=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    seen = set()
    for step in range(40):
        boxes = [] if 15 <= step < 18 else [[50 + 3 * step, 100, 130 + 3 * step, 220]]
        out = tracker.update(make_detections(boxes) if boxes else Detections.empty(), frame)
        seen.update(out.track_ids.tolist())
    assert len(seen) == 1  # same identity before and after the 3-frame gap
