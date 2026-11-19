"""Shared Streamlit helpers: icons, HTML/CSS injection, session state and settings.

Everything that touches ``st.session_state`` or global styling and is used by
more than one component lives here. Computer-vision logic stays in ``backend``.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import streamlit as st

import config
from backend import model_loader

# --------------------------------------------------------------------------- #
# Icons (Lucide-style, 24x24, 2px stroke)
# --------------------------------------------------------------------------- #
ICON_PATHS: dict[str, str] = {
    "layout": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "video": '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
    "film": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
    "crosshair": '<circle cx="12" cy="12" r="10"/><line x1="22" x2="18" y1="12" y2="12"/><line x1="6" x2="2" y1="12" y2="12"/><line x1="12" x2="12" y1="6" y2="2"/><line x1="12" x2="12" y1="22" y2="18"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "box": '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
    "square": '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
}


def icon_svg(name: str, size: int = 16, color: str = "currentColor", stroke: float = 2.0) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[name]}</svg>'
    )


def logo_svg(size: int = 34) -> str:
    """The VisionTrack AI mark: scan brackets around a tracked point."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 32 32">'
        f'<rect width="32" height="32" rx="8" fill="{config.THEME["accent"]}"/>'
        '<path d="M9 13v-2.5A1.5 1.5 0 0 1 10.5 9H13M19 9h2.5a1.5 1.5 0 0 1 1.5 1.5V13M23 19v2.5a1.5 1.5 0 0 1-1.5 1.5H19M13 23h-2.5A1.5 1.5 0 0 1 9 21.5V19" '
        'stroke="#171717" stroke-width="2" stroke-linecap="round" fill="none"/>'
        '<circle cx="16" cy="16" r="2.6" fill="#171717"/></svg>'
    )


def _icon_data_uri(name: str) -> str:
    svg = icon_svg(name, 24, color="black")
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="/:=,;'() ")


# Buttons that get an icon through CSS (``.st-key-<key> button::before``).
BUTTON_ICONS: dict[str, str] = {
    "hdr_settings": "sliders",
    "qa_webcam": "video",
    "qa_video": "film",
    "ctl_start": "play",
    "ctl_pause": "pause",
    "ctl_resume": "play",
    "ctl_stop": "square",
    "wc_reset": "refresh",
    "wc_save": "save",
    "set_load_model": "cpu",
    "hist_open": "chart",
    "hist_delete": "trash",
    "hist_refresh": "refresh",
    "ctl_remove": "trash",
    "set_clear_model": "refresh",
    "set_clear_uploads": "trash",
    "set_delete_all": "trash",
    "set_reset": "refresh",
}


@lru_cache(maxsize=1)
def _generated_css() -> str:
    """CSS rules that depend on Python data (icon masks)."""
    rules: list[str] = []
    for label, icon in config.PAGES:
        uri = _icon_data_uri(icon)
        for prefix in ("navon", "navoff"):
            rules.append(
                f'.st-key-{prefix}_{slugify(label)} button::before{{-webkit-mask-image:url("{uri}");mask-image:url("{uri}");}}'
            )
    for key, icon in BUTTON_ICONS.items():
        uri = _icon_data_uri(icon)
        rules.append(f'.st-key-{key} button::before{{-webkit-mask-image:url("{uri}");mask-image:url("{uri}");}}')
    download_uri = _icon_data_uri("download")
    rules.append(f'[class*="st-key-dl_"] button::before{{-webkit-mask-image:url("{download_uri}");mask-image:url("{download_uri}");}}')
    return "\n".join(rules)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def inject_css() -> None:
    try:
        base = Path(config.STYLE_PATH).read_text(encoding="utf-8")
    except OSError:
        base = ""
    st.markdown(f"<style>{base}\n{_generated_css()}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def html_block(markup: str) -> str:
    """Collapse indentation/newlines so Markdown never mistakes the HTML for a code block."""
    return "".join(line.strip() for line in markup.strip().splitlines())


def render_html(markup: str) -> None:
    st.markdown(html_block(markup), unsafe_allow_html=True)


def _streamlit_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(re.sub(r"\D.*", "", minor) or 0)
    except Exception:  # unknown version string -> assume a modern release
        return (99, 0)


def stretch(widget: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Version-proof 'fill the container width' keyword for charts, tables and images.

    Streamlit 1.50 replaced ``use_container_width=True`` with ``width="stretch"``
    (the old keyword was later removed), so the keyword is chosen by version.
    """
    if _streamlit_version() >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


def live_fragment(
    render: Callable[[], None],
    live: bool,
    every: float = 1.0,
    finished: Callable[[], bool] | None = None,
) -> None:
    """Draw ``render`` inside a fragment that refreshes itself every ``every`` seconds.

    ``live`` is evaluated by the caller at draw time. When ``finished`` is given and
    turns true inside a refresh (a background job ended), the whole page is re-run
    once so buttons, results and status chips switch to their idle layout.
    """

    @st.fragment(run_every=every if live else None)
    def _fragment() -> None:
        render()
        if live and finished is not None and finished():
            st.rerun()

    _fragment()


# --------------------------------------------------------------------------- #
# Session state and settings
# --------------------------------------------------------------------------- #
CFG_DEFAULTS: dict[str, Any] = {
    "cfg_model": config.DEFAULT_MODEL,
    "cfg_device": "auto",
    "cfg_tracker": config.DEFAULT_TRACKER,
    "cfg_conf": 0.35,
    "cfg_iou": 0.5,
    "cfg_classes": [],
    "cfg_imgsz": 640,
    "cfg_interval": 1,
    "cfg_max_width": 960,
    "cfg_show_boxes": True,
    "cfg_show_labels": True,
    "cfg_show_ids": True,
    "cfg_show_conf": True,
    "cfg_show_trails": False,
    "cfg_trail_len": 30,
    "cfg_track_buffer": 30,
    "cfg_match_thresh": 0.8,
}

STATE_DEFAULTS: dict[str, Any] = {
    "page": "Overview",
    "job": None,  # VideoJob
    "webcam_processor": None,  # WebcamProcessor
    "active_source": None,  # "video" | "webcam"
    "overview_source": "Video upload",
    "analytics_source": "live",  # "live" or a stored session id
    "webcam_paused": False,
    "upload_key": None,
    "upload_path": None,
    "upload_info": None,
    "upload_name": None,
    "upload_error": None,
    "upload_preview": None,
    "history_selected": None,
    "confirm_delete": None,
    "flash": None,
}


def init_state() -> None:
    for key, default in {**CFG_DEFAULTS, **STATE_DEFAULTS}.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default


def persist_widget_state() -> None:
    """Keep settings widgets' values alive while their page is not being rendered.

    Streamlit forgets the value of a keyed widget on any run in which it is not
    drawn; re-assigning the value at the top of every run prevents that, so
    settings survive page navigation.
    """
    for key in CFG_DEFAULTS:
        st.session_state[key] = st.session_state[key]


def set_page(name: str) -> None:
    st.session_state.page = name


def get_settings() -> config.ProcessingSettings:
    ss = st.session_state
    ids = tuple(sorted(config.COCO_CLASSES.index(n) for n in ss.cfg_classes if n in config.COCO_CLASSES))
    return config.ProcessingSettings(
        model_name=ss.cfg_model,
        device=ss.cfg_device,
        tracker=ss.cfg_tracker,
        conf=float(ss.cfg_conf),
        iou=float(ss.cfg_iou),
        classes=ids,
        imgsz=int(ss.cfg_imgsz),
        frame_interval=int(ss.cfg_interval),
        max_width=int(ss.cfg_max_width),
        show_boxes=bool(ss.cfg_show_boxes),
        show_labels=bool(ss.cfg_show_labels),
        show_ids=bool(ss.cfg_show_ids),
        show_conf=bool(ss.cfg_show_conf),
        show_trails=bool(ss.cfg_show_trails),
        trail_length=int(ss.cfg_trail_len),
        track_buffer=int(ss.cfg_track_buffer),
        match_thresh=float(ss.cfg_match_thresh),
    )


def model_status(settings: config.ProcessingSettings | None = None) -> tuple[str, str]:
    """(label, kind) describing the selected model for the header and sidebar."""
    settings = settings or get_settings()
    short = settings.model_name.replace(".pt", "")
    if model_loader.is_loaded(settings.model_name, settings.device):
        return f"{short} · Ready", "ok"
    if model_loader.last_error(settings.model_name):
        return f"{short} · Error", "error"
    return f"{short} · Not loaded", "idle"


def system_status() -> tuple[str, str]:
    """(label, kind) for the overall system: Idle / Processing / Live / Paused."""
    job = st.session_state.get("job")
    processor = st.session_state.get("webcam_processor")
    if job is not None and job.is_active:
        return ("Paused" if job.state.value == "paused" else "Processing"), ("warn" if job.state.value == "paused" else "busy")
    if processor is not None and processor.is_active():
        return "Live", "live"
    return "Idle", "ok"
