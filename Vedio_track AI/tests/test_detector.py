"""Detector wrapper tests.

Unit tests use a fake Ultralytics model that mimics ``model.predict`` output
(tensor-like objects with ``.cpu().numpy()``). Integration tests that need the
real library / weights are skipped automatically when those are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from backend.detector import Detections, YoloDetector


class FakeTensor:
    """Mimics a torch tensor: .cpu().numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = FakeTensor(xyxy), FakeTensor(conf), FakeTensor(cls)
        self._n = len(conf)

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYolo:
    names = {0: "person", 2: "car"}

    def __init__(self, results):
        self._results = results
        self.last_kwargs = {}

    def predict(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return self._results


FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def test_detections_empty_and_xywh():
    empty = Detections.empty()
    assert len(empty) == 0 and empty.xywh.shape == (0, 4)
    det = Detections(np.array([[10, 20, 50, 100]], dtype=np.float32), np.array([0.8], dtype=np.float32), np.array([0]))
    assert det.xywh.tolist() == [[30.0, 60.0, 40.0, 80.0]]


def test_detect_converts_tensor_output():
    model = FakeYolo([FakeResult(FakeBoxes([[1, 2, 30, 40], [50, 60, 90, 100]], [0.9, 0.5], [0, 2]))])
    detector = YoloDetector(model, "fake.pt", "cpu")
    result = detector.detect(FRAME, conf=0.4, iou=0.6, classes=[0, 2], imgsz=320)
    assert len(result) == 2
    assert result.cls.tolist() == [0, 2]
    assert result.xyxy.dtype == np.float32
    assert model.last_kwargs["conf"] == 0.4 and model.last_kwargs["classes"] == [0, 2]
    assert model.last_kwargs["imgsz"] == 320 and model.last_kwargs["verbose"] is False
    assert detector.class_name(2) == "car" and detector.class_name(99) == "99"


def test_detect_no_objects_does_not_crash():
    for results in ([], [FakeResult(None)], [FakeResult(FakeBoxes(np.zeros((0, 4)), [], []))]):
        detector = YoloDetector(FakeYolo(results), "fake.pt", "cpu")
        assert len(detector.detect(FRAME)) == 0


def test_detect_handles_empty_frame_and_empty_class_filter():
    model = FakeYolo([])
    detector = YoloDetector(model, "fake.pt", "cpu")
    assert len(detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))) == 0
    detector.detect(FRAME, classes=())
    assert model.last_kwargs["classes"] is None  # empty filter == all classes


def test_coco_class_list_is_complete():
    assert len(config.COCO_CLASSES) == 80
    assert config.COCO_CLASSES[0] == "person" and config.COCO_CLASSES[79] == "toothbrush"
    assert len(set(config.COCO_CLASSES)) == 80


def test_settings_roundtrip():
    s = config.ProcessingSettings(classes=(0, 2), conf=0.5)
    assert config.ProcessingSettings.from_dict(s.to_dict()) == s
    assert s.class_names == ["person", "bicycle"][:0] + ["person", "car"]
    assert config.ProcessingSettings.from_dict({"conf": 0.9, "unknown": 1}).conf == 0.9


def test_model_loader_reports_missing_ultralytics(monkeypatch):
    from backend import model_loader

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", None)  # makes `import ultralytics` fail
    model_loader.clear_cache()
    with pytest.raises(model_loader.ModelLoadError, match="ultralytics"):
        model_loader.load_detector("yolo11n.pt", "cpu")
    assert model_loader.last_error("yolo11n.pt")


def test_real_model_detects_something():
    """Integration: real YOLO weights on a blank frame must run and return a valid (possibly empty) result."""
    pytest.importorskip("ultralytics")
    from backend import model_loader

    try:
        detector = model_loader.load_detector(config.DEFAULT_MODEL, "cpu")
    except model_loader.ModelLoadError as exc:  # no internet / weights missing
        pytest.skip(f"model weights unavailable: {exc}")
    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), conf=0.25)
    assert isinstance(result, Detections)
    assert len(detector.names) >= 80
