"""Shared fixtures: a synthetic video plus deterministic stand-ins for YOLO and the tracker.

The stand-ins let the real pipeline (resize -> detect -> track -> record ->
annotate -> H.264 export -> session saving) run without PyTorch or Ultralytics.
Tests that need the real libraries use ``pytest.importorskip`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backend.detector import Detections  # noqa: E402
from backend.tracker import TrackedObjects  # noqa: E402

# Two coloured squares move across the frame; the "detector" finds them by colour.
RED = (0, 0, 255)  # BGR
BLUE = (255, 0, 0)
SQUARE = 40


def draw_scene(frame_idx: int, width: int = 320, height: int = 240, objects: bool = True) -> np.ndarray:
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    if objects:
        x_red = 10 + frame_idx * 4
        x_blue = width - SQUARE - 10 - frame_idx * 3
        cv2.rectangle(frame, (x_red, 40), (x_red + SQUARE, 40 + SQUARE), RED, -1)
        cv2.rectangle(frame, (x_blue, 150), (x_blue + SQUARE, 150 + SQUARE), BLUE, -1)
    return frame


class ColorBlobDetector:
    """Detector double with the same interface as ``YoloDetector``.

    Finds the red (class 0 'person') and blue (class 2 'car') squares by colour.
    """

    device = "cpu"
    names = {0: "person", 2: "car"}

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame, conf=0.35, iou=0.5, classes=None, imgsz=640) -> Detections:
        self.calls += 1
        boxes, confs, cls = [], [], []
        for class_id, (b, g, r) in ((0, RED), (2, BLUE)):
            if classes and class_id not in classes:
                continue
            mask = cv2.inRange(frame, (max(b - 25, 0), max(g - 25, 0), max(r - 25, 0)), (min(b + 25, 255), min(g + 25, 255), min(r + 25, 255)))
            ys, xs = np.nonzero(mask)
            if xs.size < 50:
                continue
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            confs.append(0.9)
            cls.append(class_id)
        if not boxes:
            return Detections.empty()
        return Detections(np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32), np.array(cls, dtype=np.int64))


class IoUTracker:
    """Minimal greedy IoU tracker used as a deterministic tracker double."""

    def __init__(self) -> None:
        self.tracks: dict[int, np.ndarray] = {}
        self.next_id = 1

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: Detections, frame=None) -> TrackedObjects:
        ids, used = [], set()
        for box in detections.xyxy:
            best_id, best = None, 0.2
            for tid, prev in self.tracks.items():
                if tid in used:
                    continue
                score = self._iou(box, prev)
                if score > best:
                    best_id, best = tid, score
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            self.tracks[best_id] = box
            ids.append(best_id)
        if len(detections) == 0:
            return TrackedObjects.empty()
        return TrackedObjects(detections.xyxy.copy(), np.array(ids, dtype=np.int64), detections.conf.copy(), detections.cls.copy())

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1


def iou_tracker_factory(settings, frame_rate):
    return IoUTracker()


@pytest.fixture
def blob_detector() -> ColorBlobDetector:
    return ColorBlobDetector()


@pytest.fixture
def settings() -> config.ProcessingSettings:
    return config.ProcessingSettings(max_width=0, conf=0.3)


def write_video(path: Path, frames: int = 30, fps: int = 15, objects: bool = True, size=(320, 240)) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG") if path.suffix == ".avi" else cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened(), "OpenCV could not create the synthetic test video"
    for i in range(frames):
        writer.write(draw_scene(i, size[0], size[1], objects))
    writer.release()
    return path


@pytest.fixture
def sample_video(tmp_path) -> Path:
    return write_video(tmp_path / "scene.mp4")


@pytest.fixture
def empty_video(tmp_path) -> Path:
    return write_video(tmp_path / "empty.mp4", objects=False)
