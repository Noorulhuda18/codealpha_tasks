"""Utility helpers: video probing, frame drawing, H.264 video writing, file handling."""

from __future__ import annotations

import colorsys
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import config

if TYPE_CHECKING:  # avoid a runtime import cycle
    from backend.tracker import TrackedObjects, TrailBuffer


class InvalidVideoError(ValueError):
    """The file is missing, has an unsupported format, or cannot be decoded."""


class VideoWriteError(RuntimeError):
    """The processed video could not be written."""


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def new_session_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "video"
    return stem[-80:]


# --------------------------------------------------------------------------- #
# Video probing / reading
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    path: str
    name: str
    width: int
    height: int
    fps: float
    frame_count: int  # 0 when the container does not report it
    duration_s: float
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_video(path: str | Path) -> VideoInfo:
    """Validate a video file and return its metadata.

    Raises :class:`InvalidVideoError` with a user-friendly message when the file
    is missing, has an unsupported extension, cannot be opened or has no frames.
    """
    p = Path(path)
    if not p.exists():
        raise InvalidVideoError(f"File not found: {p.name}")
    extension = p.suffix.lower().lstrip(".")
    if extension not in config.ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(e.upper() for e in config.ALLOWED_VIDEO_EXTENSIONS)
        raise InvalidVideoError(f"Unsupported format '.{extension}'. Please upload one of: {allowed}.")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise InvalidVideoError(
                f"'{p.name}' could not be opened. The file may be corrupt or use a codec OpenCV cannot decode."
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise InvalidVideoError(f"'{p.name}' does not contain any readable video frames.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 25.0
    frame_count = max(frame_count, 0)
    height, width = frame.shape[:2]
    return VideoInfo(
        path=str(p),
        name=p.name,
        width=int(width),
        height=int(height),
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_s=round(frame_count / fps, 2) if frame_count else 0.0,
        size_bytes=p.stat().st_size,
    )


def read_frame(path: str | Path, index: int = 0) -> np.ndarray | None:
    """Read a single frame (used for the preview thumbnail). Returns ``None`` on failure."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def save_uploaded_file(uploaded: Any, dest_dir: Path | None = None) -> Path:
    """Persist a Streamlit ``UploadedFile`` (or any object with ``name``/``getbuffer``) to disk."""
    dest_dir = Path(dest_dir or config.UPLOADS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{safe_filename(uploaded.name)}"
    with open(dest, "wb") as fh:
        fh.write(uploaded.getbuffer())
    return dest


def remove_file(path: str | Path | None) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Frame geometry
# --------------------------------------------------------------------------- #
def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Down-scale so the width is at most ``max_width`` (0 = keep size).

    Output dimensions are always even, as required by H.264 / yuv420p.
    """
    height, width = frame.shape[:2]
    if max_width and width > max_width:
        scale = max_width / float(width)
        new_w = max(2, int(round(width * scale)))
        new_h = max(2, int(round(height * scale)))
        new_w -= new_w % 2
        new_h -= new_h % 2
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    even_w, even_h = width - (width % 2), height - (height % 2)
    if (even_w, even_h) != (width, height):
        return np.ascontiguousarray(frame[:even_h, :even_w])
    return frame


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colour for a track ID (golden-ratio hue spacing)."""
    hue = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.98)
    return int(b * 255), int(g * 255), int(r * 255)


def annotate_frame(
    frame: np.ndarray,
    tracked: "TrackedObjects",
    names: dict[int, str],
    settings: config.ProcessingSettings,
    trails: "TrailBuffer | None" = None,
) -> np.ndarray:
    """Draw boxes, labels, IDs and trails on ``frame`` (modified in place and returned)."""
    height, width = frame.shape[:2]
    thickness = max(1, int(round(max(width, height) / 600)))
    font_scale = max(0.4, min(0.75, width / 1500))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if settings.show_trails and trails is not None:
        for tid in tracked.track_ids.tolist():
            points = trails.trail(tid)
            color = color_for_id(tid)
            for j in range(1, len(points)):
                fade = 0.35 + 0.65 * (j / len(points))
                faded = (int(color[0] * fade), int(color[1] * fade), int(color[2] * fade))
                cv2.line(frame, points[j - 1], points[j], faded, max(1, thickness), cv2.LINE_AA)

    for i in range(len(tracked)):
        x1, y1, x2, y2 = (int(v) for v in tracked.xyxy[i])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        tid = int(tracked.track_ids[i])
        color = color_for_id(tid)

        if settings.show_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            corner = max(6, int(min(x2 - x1, y2 - y1) * 0.22))
            for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thickness + 2, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thickness + 2, cv2.LINE_AA)

        parts: list[str] = []
        if settings.show_ids:
            parts.append(f"#{tid}")
        if settings.show_labels:
            parts.append(names.get(int(tracked.cls[i]), str(int(tracked.cls[i]))))
        if settings.show_conf:
            parts.append(f"{tracked.conf[i] * 100:.0f}%")
        if parts:
            text = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
            top = y1 - th - baseline - 6
            if top < 0:  # keep the label inside the frame
                top = y1 + 2
            cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + baseline + 6), color, -1)
            cv2.putText(frame, text, (x1 + 4, top + th + 2), font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Video writing (H.264 via FFmpeg when available, OpenCV mp4v fallback)
# --------------------------------------------------------------------------- #
def ffmpeg_executable() -> str | None:
    """Locate an FFmpeg binary: the one bundled with imageio-ffmpeg, else the system one."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


class VideoSink:
    """Writes frames incrementally to an MP4 file.

    Frames are piped straight into FFmpeg (libx264, yuv420p) so the result plays
    in every browser. If FFmpeg is unavailable, OpenCV's ``mp4v`` writer is used;
    that file is downloadable but may not preview in a browser (``playable`` is
    then ``False``).
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.size = (int(size[0]), int(size[1]))  # (width, height)
        self.frames_written = 0
        self.playable = False
        self._proc: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

        exe = ffmpeg_executable()
        if exe and self._start_ffmpeg(exe):
            self.playable = True
        else:
            self._start_opencv()

    # -- construction helpers ------------------------------------------------ #
    def _start_ffmpeg(self, exe: str) -> bool:
        width, height = self.size
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-framerate", f"{self.fps:.3f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags
            )
            return True
        except OSError:
            self._proc = None
            return False

    def _start_opencv(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
        if not self._writer.isOpened():
            self._writer = None
            raise VideoWriteError("No video encoder is available (FFmpeg and OpenCV mp4v both failed).")
        self.playable = False

    # -- public API ---------------------------------------------------------- #
    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise VideoWriteError("The video sink is already closed.")
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise VideoWriteError(f"Frame size {frame.shape[1]}x{frame.shape[0]} does not match {self.size}.")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._proc is not None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                if self.frames_written > 0:
                    raise VideoWriteError("FFmpeg stopped unexpectedly while writing the video.")
                self._abort_ffmpeg()  # encoder failed immediately -> fall back
                self._start_opencv()
                self._writer.write(frame)  # type: ignore[union-attr]
        elif self._writer is not None:
            self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Finalise the file. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if self._proc.returncode not in (0, None) and self.frames_written:
                self.playable = False
            self._proc = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _abort_ffmpeg(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
