"""Frame pipeline, background video jobs and the WebRTC webcam processor.

``FramePipeline``   detect -> track -> record -> annotate for one frame.
``VideoJob``        processes an uploaded video frame by frame in a worker
                    thread (start / pause / resume / stop, progress, H.264 export,
                    automatic session saving). Frames are streamed, never held
                    in memory all at once.
``WebcamProcessor`` the same pipeline applied to live WebRTC frames.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import config
from backend import model_loader
from backend.analytics import DataView, DetectionLog, save_session
from backend.detector import YoloDetector
from backend.tracker import ObjectTracker, TrackedObjects, TrailBuffer
from backend.utils import (
    InvalidVideoError,
    VideoSink,
    annotate_frame,
    new_session_id,
    probe_video,
    resize_to_width,
)

TrackerFactory = Callable[[config.ProcessingSettings, float], Any]


# --------------------------------------------------------------------------- #
# Per-frame pipeline
# --------------------------------------------------------------------------- #
class FramePipeline:
    """Detect, track, record and annotate frames, one at a time.

    ``tracker_factory(settings, frame_rate)`` can be supplied to inject a
    different tracker (the tests use this); by default a real ByteTrack /
    BoT-SORT :class:`ObjectTracker` is created.
    """

    def __init__(
        self,
        detector: YoloDetector,
        settings: config.ProcessingSettings,
        source_fps: float = 30.0,
        log: DetectionLog | None = None,
        tracker_factory: TrackerFactory | None = None,
    ) -> None:
        self.detector = detector
        self.settings = settings
        self.source_fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        self.log = log if log is not None else DetectionLog()
        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self.tracker = self._tracker_factory(settings, self._tracker_rate(settings))
        self.trails = TrailBuffer(settings.trail_length)
        self._last_tracked = TrackedObjects.empty()

    # -- tracker management -------------------------------------------------- #
    @staticmethod
    def _default_tracker_factory(settings: config.ProcessingSettings, frame_rate: float) -> ObjectTracker:
        return ObjectTracker(
            settings.tracker,
            frame_rate=frame_rate,
            track_buffer=settings.track_buffer,
            match_thresh=settings.match_thresh,
        )

    def _tracker_rate(self, settings: config.ProcessingSettings) -> float:
        return self.source_fps / max(1, settings.frame_interval)

    def reset_tracking(self) -> None:
        self.tracker = self._tracker_factory(self.settings, self._tracker_rate(self.settings))
        self.trails.clear()
        self._last_tracked = TrackedObjects.empty()

    def set_detector(self, detector: YoloDetector) -> None:
        self.detector = detector

    def update_settings(self, new: config.ProcessingSettings) -> None:
        """Apply new settings between frames (tracker is rebuilt only if its parameters changed)."""
        old, self.settings = self.settings, new
        if (new.tracker, new.track_buffer, new.match_thresh) != (old.tracker, old.track_buffer, old.match_thresh):
            self.reset_tracking()
        self.trails.set_max_len(new.trail_length)

    # -- processing ---------------------------------------------------------- #
    def process(self, frame: np.ndarray, frame_idx: int, time_s: float, infer: bool | None = None) -> np.ndarray:
        """Process one BGR frame and return the annotated (and possibly resized) frame.

        When ``infer`` is false (frame skipping) the last known tracks are drawn
        on the new frame instead of running the detector again.
        """
        s = self.settings
        frame = resize_to_width(frame, s.max_width)
        if infer is None:
            infer = frame_idx % max(1, s.frame_interval) == 0

        if not infer:
            return annotate_frame(frame, self._last_tracked, self.detector.names, s, self.trails)

        started = time.perf_counter()
        detections = self.detector.detect(frame, s.conf, s.iou, s.classes, s.imgsz)
        tracked = self.tracker.update(detections, frame)
        self.trails.update(tracked.track_ids, tracked.centers)
        self._last_tracked = tracked
        annotated = annotate_frame(frame, tracked, self.detector.names, s, self.trails)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.log.add_frame(
            frame_idx=frame_idx,
            time_s=time_s,
            tracked=tracked,
            names=self.detector.names,
            latency_ms=latency_ms,
            raw_detections=len(detections),
            frame_size=(annotated.shape[1], annotated.shape[0]),
        )
        return annotated


# --------------------------------------------------------------------------- #
# Video jobs
# --------------------------------------------------------------------------- #
class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


ACTIVE_STATES = (JobState.RUNNING, JobState.PAUSED, JobState.STOPPING)


@dataclass
class JobSnapshot:
    state: JobState
    progress: float  # 0..1 (0 when the total number of frames is unknown)
    frames_read: int
    total_frames: int
    elapsed_s: float
    eta_s: float | None
    error: str | None
    latest_frame: np.ndarray | None
    output_path: Path | None
    playable: bool
    session_id: str


class VideoJob:
    """Process one video file in a background thread."""

    def __init__(
        self,
        source_path: str | Path,
        original_name: str,
        settings: config.ProcessingSettings,
        detector: YoloDetector,
        sessions_dir: Path | None = None,
        tracker_factory: TrackerFactory | None = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.original_name = original_name
        self.settings = settings
        self.detector = detector
        self.sessions_dir = Path(sessions_dir or config.SESSIONS_DIR)
        self.session_id = new_session_id()
        self.session_dir = self.sessions_dir / self.session_id
        self.output_path = self.session_dir / "processed.mp4"
        self.log = DetectionLog()
        self._tracker_factory = tracker_factory

        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()
        self._state = JobState.PENDING
        self._error: str | None = None
        self._frames_read = 0
        self._total_frames = 0
        self._latest_frame: np.ndarray | None = None
        self._playable = False
        self._active_since: float | None = None
        self._active_total = 0.0
        self.meta: dict[str, Any] = {}
        self.pipeline: FramePipeline | None = None

    # -- control ------------------------------------------------------------- #
    def start(self) -> None:
        """Start processing in a daemon thread and return immediately."""
        with self._lock:
            if self._state != JobState.PENDING:
                raise RuntimeError("This job has already been started.")
            self._state = JobState.RUNNING
        self._thread = threading.Thread(target=self.run, name=f"video-job-{self.session_id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self._state == JobState.RUNNING:
                self._resume.clear()
                self._state = JobState.PAUSED
                self._bank_active_time()

    def resume(self) -> None:
        with self._lock:
            if self._state == JobState.PAUSED:
                self._active_since = time.perf_counter()
                self._state = JobState.RUNNING
                self._resume.set()

    def stop(self) -> None:
        with self._lock:
            if self._state in (JobState.RUNNING, JobState.PAUSED):
                self._state = JobState.STOPPING
                self._stop.set()
                self._resume.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def update_settings(self, new: config.ProcessingSettings) -> None:
        """Apply live-safe options mid-run. Model, tracker, resolution and interval stay as started."""
        with self._lock:
            locked = self.settings
            merged = new.with_updates(
                model_name=locked.model_name, device=locked.device, tracker=locked.tracker,
                track_buffer=locked.track_buffer, match_thresh=locked.match_thresh,
                max_width=locked.max_width, frame_interval=locked.frame_interval, imgsz=locked.imgsz,
            )
            self.settings = merged
            if self.pipeline is not None:
                self.pipeline.update_settings(merged)

    @property
    def is_active(self) -> bool:
        return self._state in ACTIVE_STATES

    @property
    def state(self) -> JobState:
        return self._state

    # -- observation --------------------------------------------------------- #
    def _bank_active_time(self) -> None:
        if self._active_since is not None:
            self._active_total += time.perf_counter() - self._active_since
            self._active_since = None

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            elapsed = self._active_total + (time.perf_counter() - self._active_since if self._active_since else 0.0)
            total, done = self._total_frames, self._frames_read
            progress = min(done / total, 1.0) if total > 0 else 0.0
            if self._state == JobState.COMPLETED:
                progress = 1.0
            eta = (elapsed / done) * (total - done) if (total > 0 and done > 0 and self._state in ACTIVE_STATES) else None
            return JobSnapshot(
                state=self._state, progress=progress, frames_read=done, total_frames=total,
                elapsed_s=elapsed, eta_s=eta, error=self._error, latest_frame=self._latest_frame,
                output_path=self.output_path if self.output_path.exists() else None,
                playable=self._playable, session_id=self.session_id,
            )

    def data_view(self) -> DataView:
        snap_state = self._state
        finished = snap_state in (JobState.COMPLETED, JobState.STOPPED)
        return DataView(
            label=self.original_name,
            source="video",
            log=self.log,
            meta=self.meta,
            live=self.is_active,
            video_path=self.output_path if finished and self.output_path.exists() else None,
            session_id=self.session_id if finished else None,
            latest_frame_fn=lambda: self._latest_frame,
        )

    # -- worker -------------------------------------------------------------- #
    def run(self) -> None:
        """Process the whole video. Runs in the worker thread (or directly in tests)."""
        with self._lock:
            if self._state == JobState.PENDING:
                self._state = JobState.RUNNING
        cap: cv2.VideoCapture | None = None
        sink: VideoSink | None = None
        try:
            info = probe_video(self.source_path)
            self._total_frames = info.frame_count
            fps = info.fps
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.pipeline = FramePipeline(
                self.detector, self.settings, source_fps=fps, log=self.log, tracker_factory=self._tracker_factory
            )
            self.meta = {
                "session_id": self.session_id,
                "source": "video",
                "source_name": self.original_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model": self.settings.model_name,
                "device": self.detector.device,
                "tracker": self.settings.tracker,
                "settings": self.settings.to_dict(),
                "video": info.to_dict(),
            }
            cap = cv2.VideoCapture(str(self.source_path))
            if not cap.isOpened():
                raise InvalidVideoError(f"'{self.original_name}' could not be opened for processing.")

            with self._lock:
                self._active_since = time.perf_counter()
            index = 0
            while not self._stop.is_set():
                if not self._resume.is_set():  # paused
                    self._resume.wait(timeout=0.2)
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # end of video (or an undecodable tail)
                annotated = self.pipeline.process(frame, index, index / fps)
                if sink is None:
                    sink = VideoSink(self.output_path, fps, (annotated.shape[1], annotated.shape[0]))
                    self._playable = sink.playable
                sink.write(annotated)
                with self._lock:
                    self._latest_frame = annotated
                    self._frames_read = index + 1
                index += 1

            with self._lock:
                self._bank_active_time()
                self._state = JobState.STOPPED if self._stop.is_set() else JobState.COMPLETED
                # Frame counts reported by containers can be off; the run is complete now.
                self._total_frames = max(self._total_frames, self._frames_read) if self._state == JobState.COMPLETED else self._total_frames
        except InvalidVideoError as exc:
            self._fail(str(exc))
        except Exception as exc:  # keep the UI alive and report what went wrong
            self._fail(f"Processing failed: {type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if sink is not None:
                try:
                    sink.close()
                    self._playable = sink.playable
                except Exception as exc:
                    if self._state != JobState.FAILED:
                        self._fail(f"Could not finalise the output video: {exc}")
            self._finalise()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._bank_active_time()
            self._error = message
            self._state = JobState.FAILED

    def _finalise(self) -> None:
        """Persist the session (completed / stopped) or clean up (failed)."""
        try:
            if self._state in (JobState.COMPLETED, JobState.STOPPED) and self.log.frames_processed > 0:
                self.meta["status"] = self._state.value
                self.meta["frame_size"] = list(self.log.frame_size) if self.log.frame_size else None
                self.meta["processed_video"] = "processed.mp4" if self.output_path.exists() else None
                self.meta["video_playable_in_browser"] = self._playable
                save_session(self.session_id, self.meta, self.log.records_df(), self.log.frame_stats_df(), self.sessions_dir)
            elif self._state == JobState.FAILED:
                shutil.rmtree(self.session_dir, ignore_errors=True)
        except Exception as exc:  # saving must never crash the worker
            self._error = f"Results could not be saved: {exc}"


# --------------------------------------------------------------------------- #
# Webcam (streamlit-webrtc)
# --------------------------------------------------------------------------- #
try:  # streamlit-webrtc is optional at import time so the backend works without it
    from streamlit_webrtc import VideoProcessorBase as _WebRtcBase
except Exception:  # pragma: no cover - depends on the environment
    _WebRtcBase = object  # type: ignore[assignment,misc]

DetectorProvider = Callable[[config.ProcessingSettings], YoloDetector]


def _default_detector_provider(settings: config.ProcessingSettings) -> YoloDetector:
    return model_loader.load_detector(settings.model_name, settings.device)


class WebcamProcessor(_WebRtcBase):  # type: ignore[misc,valid-type]
    """Processes live WebRTC frames. One instance is created per webcam stream.

    ``recv`` is invoked by streamlit-webrtc on its own thread; settings are
    updated from the Streamlit script via :meth:`update_settings`.
    """

    def __init__(
        self,
        settings: config.ProcessingSettings,
        detector_provider: DetectorProvider | None = None,
        tracker_factory: TrackerFactory | None = None,
        assumed_fps: float = 30.0,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._provider = detector_provider or _default_detector_provider
        self._tracker_factory = tracker_factory
        self._assumed_fps = assumed_fps
        self._lock = threading.RLock()
        self._pipeline: FramePipeline | None = None
        self._pipeline_settings: config.ProcessingSettings | None = None
        self._reset_requested = False
        self._t0: float | None = None
        self._frame_idx = 0
        self.log = DetectionLog()
        self.paused = False
        self.latest_frame: np.ndarray | None = None
        self.frames_received = 0
        self.last_frame_time: float | None = None
        self.last_error: str | None = None
        self.session_id = new_session_id()
        self.created_at = datetime.now().isoformat(timespec="seconds")

    # -- called from the Streamlit script ------------------------------------ #
    def update_settings(self, settings: config.ProcessingSettings) -> None:
        self._settings = settings

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def reset(self) -> None:
        """Clear the log and tracker (applied on the next frame)."""
        self._reset_requested = True

    def seconds_since_last_frame(self) -> float | None:
        return None if self.last_frame_time is None else time.time() - self.last_frame_time

    def is_active(self, within_s: float = 3.0) -> bool:
        gap = self.seconds_since_last_frame()
        return gap is not None and gap < within_s

    @property
    def meta(self) -> dict[str, Any]:
        s = self._settings
        return {
            "session_id": self.session_id, "source": "webcam", "source_name": "Webcam stream",
            "created_at": self.created_at, "model": s.model_name, "tracker": s.tracker,
            "settings": s.to_dict(),
            "frame_size": list(self.log.frame_size) if self.log.frame_size else None,
        }

    def data_view(self) -> DataView:
        return DataView(
            label="Webcam stream", source="webcam", log=self.log, meta=self.meta,
            live=self.is_active(), latest_frame_fn=lambda: self.latest_frame,
        )

    # -- called from the WebRTC thread --------------------------------------- #
    def recv(self, frame: Any) -> Any:
        import av  # installed together with streamlit-webrtc

        image = frame.to_ndarray(format="bgr24")
        try:
            output = self.process_image(image)
        except Exception as exc:  # never kill the stream because of one bad frame
            self.last_error = f"{type(exc).__name__}: {exc}"
            output = image
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(output), format="bgr24")

    def process_image(self, image: np.ndarray, now: float | None = None) -> np.ndarray:
        """Process one BGR frame (separated from ``recv`` so it can be tested without WebRTC)."""
        now = time.time() if now is None else now
        settings = self._settings
        if self._t0 is None:
            self._t0 = now
        self.frames_received += 1
        self.last_frame_time = now

        if self._reset_requested:
            self._reset_requested = False
            self.log.reset()
            self._frame_idx = 0
            self._t0 = now
            if self._pipeline is not None:
                self._pipeline.reset_tracking()

        if self.paused:
            output = resize_to_width(image, settings.max_width)
        else:
            pipeline = self._ensure_pipeline(settings)
            output = pipeline.process(image, self._frame_idx, now - self._t0)
            self._frame_idx += 1
            self.last_error = None
        self.latest_frame = output
        return output

    def _ensure_pipeline(self, settings: config.ProcessingSettings) -> FramePipeline:
        with self._lock:
            if self._pipeline is None:
                detector = self._provider(settings)
                self._pipeline = FramePipeline(
                    detector, settings, source_fps=self._assumed_fps, log=self.log, tracker_factory=self._tracker_factory
                )
                self._pipeline_settings = settings
            elif settings is not self._pipeline_settings:
                old = self._pipeline_settings
                if old is not None and (settings.model_name, settings.device) != (old.model_name, old.device):
                    self._pipeline.set_detector(self._provider(settings))
                self._pipeline.update_settings(settings)
                self._pipeline_settings = settings
            return self._pipeline
