"""Central configuration for VisionTrack AI.

Everything that is a constant, a default, or a path lives here so that the
backend (computer vision) and the components (Streamlit UI) share one source
of truth. This module must stay free of heavy imports (no torch / streamlit /
ultralytics) so that it can be imported anywhere, including from tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
APP_NAME = "VisionTrack AI"
TAGLINE = "See Everything. Track Anything."
VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
STYLE_PATH = ASSETS_DIR / "style.css"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
UPLOADS_DIR = OUTPUTS_DIR / "_uploads"
MODELS_DIR = ROOT_DIR / "models"

# --------------------------------------------------------------------------- #
# Theme (mirrors .streamlit/config.toml and assets/style.css)
# --------------------------------------------------------------------------- #
THEME = {
    "bg": "#212121",
    "bg_secondary": "#171717",
    "sidebar": "#171717",
    "card": "#262626",
    "border": "#3A3A3A",
    "text": "#F5F5F5",
    "text_muted": "#A1A1AA",
    "accent": "#19C37D",
    "accent_2": "#8B8BFF",
    "warning": "#F5A524",
    "danger": "#F0616D",
}

# --------------------------------------------------------------------------- #
# Navigation: (label, icon name)
# --------------------------------------------------------------------------- #
PAGES: list[tuple[str, str]] = [
    ("Overview", "layout"),
    ("Live Detection", "video"),
    ("Video Analysis", "film"),
    ("Object Tracking", "crosshair"),
    ("Detection History", "history"),
    ("Analytics", "chart"),
    ("Settings", "sliders"),
]

PAGE_DESCRIPTIONS: dict[str, str] = {
    "Overview": "Your computer vision workspace — detect, track and monitor objects.",
    "Live Detection": "Real-time detection and tracking from your webcam.",
    "Video Analysis": "Upload a video, process it on demand and export the results.",
    "Object Tracking": "Track identities, trajectories and lifetimes across frames.",
    "Detection History": "Browse, reload and export previous processing sessions.",
    "Analytics": "Statistics computed from the frames that were actually processed.",
    "Settings": "Model, tracker and performance configuration.",
}

# --------------------------------------------------------------------------- #
# Models and trackers
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "yolo11n.pt"
MODEL_OPTIONS: dict[str, str] = {
    "yolo11n.pt": "YOLO11 Nano — fastest, recommended for CPU laptops",
    "yolov8n.pt": "YOLOv8 Nano — fast and widely compatible",
    "yolo26n.pt": "YOLO26 Nano — newest nano model (needs a recent Ultralytics)",
    "yolo11s.pt": "YOLO11 Small — more accurate, slower on CPU",
    "yolov8s.pt": "YOLOv8 Small — more accurate, slower on CPU",
}
DEVICE_OPTIONS: dict[str, str] = {
    "auto": "Auto (GPU if available, otherwise CPU)",
    "cpu": "CPU",
    "cuda": "NVIDIA GPU (CUDA)",
}
TRACKER_OPTIONS: dict[str, str] = {
    "bytetrack": "ByteTrack",
    "botsort": "BoT-SORT",
}
DEFAULT_TRACKER = "bytetrack"
IMGSZ_OPTIONS: tuple[int, ...] = (320, 416, 512, 640, 800, 960)
MAX_WIDTH_OPTIONS: tuple[int, ...] = (0, 480, 640, 800, 960, 1280)  # 0 == keep original size

# Default tracker hyper-parameters. They are used when the YAML files that ship
# with Ultralytics cannot be located; otherwise the YAML values take priority
# and only the values the user changes in the UI are overridden.
TRACKER_DEFAULTS: dict[str, dict[str, Any]] = {
    "bytetrack": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
    },
    "botsort": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
    },
}

# The 80 COCO classes used by the pretrained YOLO models (index == class id).
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# --------------------------------------------------------------------------- #
# Video and limits
# --------------------------------------------------------------------------- #
ALLOWED_VIDEO_EXTENSIONS = ("mp4", "avi", "mov")
MAX_UPLOAD_MB = 1024  # keep in sync with server.maxUploadSize in config.toml
MAX_RECORDS = 150_000  # detection rows kept in memory per session (oldest dropped)
MAX_FRAME_STATS = 100_000  # per-frame statistics kept in memory per session
LIVE_REFRESH_SECONDS = 1.0  # refresh period of live panels
CHART_REFRESH_SECONDS = 3.0  # refresh period of live charts
WEBCAM_STALL_SECONDS = 6.0  # no frames for this long => show a warning

# WebRTC: a public STUN server helps when the browser is not on the same machine.
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# --------------------------------------------------------------------------- #
# Processing settings shared by the UI, the pipeline and the tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcessingSettings:
    """Immutable bundle of every user-tunable processing option.

    The dataclass is frozen so a new instance can be swapped into a running
    pipeline atomically from another thread (see ``dataclasses.replace``).
    """

    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    tracker: str = DEFAULT_TRACKER
    conf: float = 0.35
    iou: float = 0.5
    classes: tuple[int, ...] = ()  # COCO class ids; empty tuple == all classes
    imgsz: int = 640
    frame_interval: int = 1  # run the detector on every Nth frame
    max_width: int = 960  # frames wider than this are down-scaled (0 == keep)
    show_boxes: bool = True
    show_labels: bool = True
    show_ids: bool = True
    show_conf: bool = True
    show_trails: bool = False
    trail_length: int = 30
    track_buffer: int = 30
    match_thresh: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classes"] = list(self.classes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingSettings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if "classes" in clean:
            clean["classes"] = tuple(int(c) for c in clean["classes"])
        return cls(**clean)

    def with_updates(self, **changes: Any) -> "ProcessingSettings":
        return replace(self, **changes)

    @property
    def class_names(self) -> list[str]:
        return [COCO_CLASSES[i] for i in self.classes if 0 <= i < len(COCO_CLASSES)]
