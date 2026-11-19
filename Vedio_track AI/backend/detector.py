"""Object detection with a pretrained Ultralytics YOLO model."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class Detections:
    """Detector output for one frame (all arrays have length N)."""

    xyxy: np.ndarray  # (N, 4) float32, pixel coordinates (x1, y1, x2, y2)
    conf: np.ndarray  # (N,)  float32, 0..1
    cls: np.ndarray  # (N,)  int64 class ids

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @property
    def xywh(self) -> np.ndarray:
        """Boxes as (center_x, center_y, width, height) — the format trackers expect."""
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor (CPU or GPU) or array-like to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names or [])}


class YoloDetector:
    """Thin, thread-safe wrapper around an Ultralytics YOLO model.

    The wrapper only does detection. Tracking is handled separately by
    :class:`backend.tracker.ObjectTracker`, which keeps per-session state so a
    single (cached) model can safely be shared by several sessions.
    """

    def __init__(self, model: Any, model_name: str, device: str) -> None:
        self.model = model
        self.model_name = model_name
        self.device = device
        self.names: dict[int, str] = _names_to_dict(getattr(model, "names", {}))
        self._lock = threading.Lock()

    def class_name(self, class_id: int) -> str:
        return self.names.get(int(class_id), str(int(class_id)))

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Sequence[int] | None = None,
        imgsz: int = 640,
    ) -> Detections:
        """Run the model on one BGR frame and return its detections.

        Returns an empty :class:`Detections` (never raises) when the frame is
        empty or nothing is found.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return Detections.empty()

        class_ids = [int(c) for c in classes] if classes else None
        with self._lock:  # a shared model must not be used by two threads at once
            results = self.model.predict(
                frame,
                conf=float(conf),
                iou=float(iou),
                classes=class_ids,
                imgsz=int(imgsz),
                device=self.device,
                verbose=False,
            )
        return self._to_detections(results)

    @staticmethod
    def _to_detections(results: Any) -> Detections:
        if not results:
            return Detections.empty()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return Detections(
            xyxy=_to_numpy(boxes.xyxy).astype(np.float32).reshape(-1, 4),
            conf=_to_numpy(boxes.conf).astype(np.float32).reshape(-1),
            cls=_to_numpy(boxes.cls).astype(np.int64).reshape(-1),
        )
