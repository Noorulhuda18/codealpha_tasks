"""Multi-object tracking (ByteTrack / BoT-SORT) and movement trails.

Tracking IDs come from the real Ultralytics implementations of ByteTrack and
BoT-SORT. A fresh tracker instance is created per session, so tracker state is
never shared between users even though the YOLO model is.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config
from backend.detector import Detections


class TrackerError(RuntimeError):
    """Raised when the tracking backend cannot be created."""


@dataclass
class TrackedObjects:
    """Tracker output for one frame (all arrays have length N)."""

    xyxy: np.ndarray  # (N, 4) float32
    track_ids: np.ndarray  # (N,) int64 — produced by the tracking algorithm
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int64

    @classmethod
    def empty(cls) -> "TrackedObjects":
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            track_ids=np.zeros((0,), dtype=np.int64),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(self.track_ids.shape[0])

    @property
    def centers(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(
            [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0],
            axis=1,
        )


class _DetectionBatch:
    """Adapter exposing the attributes Ultralytics trackers read from a ``Boxes`` object.

    Ultralytics' ``BYTETracker.update`` / ``BOTSORT.update`` read ``conf``,
    ``cls`` and ``xywh`` (centre format) and, in some versions, slice the object
    with boolean masks. Using a small NumPy-only adapter means no torch tensors
    have to be built and works across Ultralytics releases.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, item: Any) -> "_DetectionBatch":
        return _DetectionBatch(self.xywh[item], self.conf[item], self.cls[item])

    @property
    def xyxy(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x, y, w, h = self.xywh.T
        return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _tracker_config(tracker_type: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge built-in defaults < Ultralytics' bundled YAML < explicit overrides."""
    cfg: dict[str, Any] = dict(config.TRACKER_DEFAULTS[tracker_type])
    try:
        import ultralytics
        import yaml

        yaml_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker_type}.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                cfg.update(yaml.safe_load(fh) or {})
    except Exception:  # keep built-in defaults if the YAML cannot be read
        pass
    cfg.update(overrides)
    cfg["tracker_type"] = tracker_type
    if tracker_type == "botsort":
        cfg["with_reid"] = False  # appearance model is not needed (and not shipped) for the demo
    cfg.setdefault("device", "cpu")
    return cfg


def build_backend(
    tracker_type: str,
    frame_rate: float = 30.0,
    track_buffer: int | None = None,
    match_thresh: float | None = None,
) -> Any:
    """Create an Ultralytics ByteTrack / BoT-SORT instance."""
    if tracker_type not in config.TRACKER_OPTIONS:
        raise TrackerError(f"Unknown tracker '{tracker_type}'. Choose one of {list(config.TRACKER_OPTIONS)}.")
    try:
        from ultralytics.trackers import BOTSORT, BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
    except ImportError as exc:
        raise TrackerError(
            "Tracking needs the 'ultralytics' package (and 'lap'). Run: pip install -r requirements.txt"
        ) from exc

    overrides: dict[str, Any] = {}
    if track_buffer is not None:
        overrides["track_buffer"] = int(track_buffer)
    if match_thresh is not None:
        overrides["match_thresh"] = float(match_thresh)

    args = IterableSimpleNamespace(**_tracker_config(tracker_type, overrides))
    args.frame_rate = max(1, int(round(frame_rate)))
    tracker_cls = BYTETracker if tracker_type == "bytetrack" else BOTSORT
    try:
        try:
            return tracker_cls(args=args)
        except TypeError as exc:
            if "frame_rate" not in str(exc):
                raise
            return tracker_cls(args=args, frame_rate=max(1, int(round(frame_rate))))
    except Exception as exc:
        raise TrackerError(f"Could not create the {config.TRACKER_OPTIONS[tracker_type]} tracker: {exc}") from exc


class ObjectTracker:
    """Per-session multi-object tracker.

    Parameters
    ----------
    tracker_type:
        ``"bytetrack"`` (default) or ``"botsort"``.
    frame_rate:
        Rate (updates per second) at which :meth:`update` will be called. It
        scales the "lost track" buffer so IDs survive the same *time* span
        regardless of the processing interval.
    backend:
        Optional pre-built tracker object exposing ``update(batch, frame)``
        (used by the tests to inject a deterministic tracker).
    """

    def __init__(
        self,
        tracker_type: str = config.DEFAULT_TRACKER,
        frame_rate: float = 30.0,
        track_buffer: int | None = None,
        match_thresh: float | None = None,
        backend: Any = None,
    ) -> None:
        self.tracker_type = tracker_type
        self._backend = backend or build_backend(tracker_type, frame_rate, track_buffer, match_thresh)

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> TrackedObjects:
        """Associate this frame's detections with existing tracks.

        Called on *every* processed frame — including frames without
        detections — so that lost tracks age and expire correctly.
        """
        batch = _DetectionBatch(detections.xywh, detections.conf, detections.cls)
        try:
            output = self._backend.update(batch, frame)
        except Exception:
            if len(detections) == 0:
                return TrackedObjects.empty()  # some backends reject empty input
            raise
        return self._parse(output)

    def reset(self) -> None:
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def _parse(output: Any) -> TrackedObjects:
        """Rows are ``[x1, y1, x2, y2, track_id, score, class, det_index]``."""
        arr = np.asarray(output, dtype=np.float32)
        if arr.size == 0:
            return TrackedObjects.empty()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return TrackedObjects(
            xyxy=arr[:, 0:4].astype(np.float32),
            track_ids=arr[:, 4].astype(np.int64),
            conf=arr[:, 5].astype(np.float32),
            cls=arr[:, 6].astype(np.int64),
        )


class TrailBuffer:
    """Keeps the recent centre points of every track for movement trails."""

    def __init__(self, max_len: int = 30, max_stale_frames: int = 90) -> None:
        self.max_len = max(2, int(max_len))
        self.max_stale_frames = max_stale_frames
        self._points: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._tick = 0

    def set_max_len(self, max_len: int) -> None:
        max_len = max(2, int(max_len))
        if max_len == self.max_len:
            return
        self.max_len = max_len
        for tid, pts in list(self._points.items()):
            self._points[tid] = deque(pts, maxlen=max_len)

    def update(self, track_ids: np.ndarray, centers: np.ndarray) -> None:
        self._tick += 1
        for tid, (cx, cy) in zip(track_ids.tolist(), centers.tolist()):
            if tid not in self._points:
                self._points[tid] = deque(maxlen=self.max_len)
            self._points[tid].append((int(cx), int(cy)))
            self._last_seen[tid] = self._tick
        if self._tick % 30 == 0:  # periodic cleanup of tracks that disappeared
            stale = [t for t, seen in self._last_seen.items() if self._tick - seen > self.max_stale_frames]
            for tid in stale:
                self._points.pop(tid, None)
                self._last_seen.pop(tid, None)

    def trail(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._points.get(int(track_id), ()))

    def clear(self) -> None:
        self._points.clear()
        self._last_seen.clear()
        self._tick = 0
