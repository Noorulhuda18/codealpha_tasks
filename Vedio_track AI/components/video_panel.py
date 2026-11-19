"""Main workspace panels: model cache, active data view, video upload/processing and webcam.

Heavy lifting (detection, tracking, recording, export) lives in ``backend``; this
module only wires Streamlit widgets to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import config
from backend import model_loader
from backend.analytics import DataView, save_session, summary_to_markdown
from backend.detector import YoloDetector
from backend.model_loader import ModelLoadError
from backend.tracker import TrackerError, build_backend
from backend.utils import (
    InvalidVideoError,
    format_seconds,
    human_size,
    probe_video,
    read_frame,
    remove_file,
    save_uploaded_file,
)
from backend.video_processor import JobSnapshot, JobState, VideoJob, WebcamProcessor
from components import get_settings, live_fragment
from components.metrics import card_title, chip, empty_state, info_grid, status_bar

try:  # the rest of the app keeps working (with a clear message) when this is missing
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - any import problem should be reported, not crash the app
    webrtc_streamer = None  # type: ignore[assignment]
    WebRtcMode = None  # type: ignore[assignment]
    WEBRTC_ERROR = f"{type(_exc).__name__}: {_exc}"


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _cached_detector(model_name: str, device: str) -> YoloDetector:
    """One detector per (model, device) for the whole server process — never per rerun."""
    return model_loader.load_detector(model_name, device)


def get_detector(settings: config.ProcessingSettings) -> YoloDetector:
    """Load (and on first use download) the selected model. Raises ``ModelLoadError``."""
    if model_loader.is_loaded(settings.model_name, settings.device):
        return _cached_detector(settings.model_name, settings.device)
    with st.spinner(f"Loading {settings.model_name} — the first run downloads the weights…"):
        return _cached_detector(settings.model_name, settings.device)


def clear_model_cache() -> None:
    _cached_detector.clear()
    model_loader.clear_cache()


def ensure_tracker(settings: config.ProcessingSettings) -> None:
    """Build a throw-away tracker so a missing dependency shows as a readable error."""
    ss = st.session_state
    if ss.get("tracker_checked") != settings.tracker:
        build_backend(settings.tracker)
        ss["tracker_checked"] = settings.tracker


# --------------------------------------------------------------------------- #
# Which session is shown
# --------------------------------------------------------------------------- #
def view_for(page: str) -> DataView:
    """The :class:`DataView` a page should display."""
    ss = st.session_state
    job, cam = ss.get("job"), ss.get("webcam_processor")
    if page == "Live Detection":
        return cam.data_view() if cam is not None else DataView.empty()
    if page == "Video Analysis":
        return job.data_view() if job is not None else DataView.empty()
    if ss.get("active_source") == "webcam" and cam is not None:
        return cam.data_view()
    if job is not None:
        return job.data_view()
    return cam.data_view() if cam is not None else DataView.empty()


def sync_live_settings() -> None:
    """Push live-adjustable options (confidence, classes, overlays …) into running sessions."""
    ss = st.session_state
    settings = get_settings()
    job, cam = ss.get("job"), ss.get("webcam_processor")
    if job is not None and job.is_active:
        job.update_settings(settings)
    if cam is not None:
        cam.update_settings(settings)


def job_is_active() -> bool:
    job = st.session_state.get("job")
    return bool(job is not None and job.is_active)


_STATE_CHIPS: dict[JobState, tuple[str, str]] = {
    JobState.PENDING: ("Ready", "idle"),
    JobState.RUNNING: ("Processing", "busy"),
    JobState.PAUSED: ("Paused", "warn"),
    JobState.STOPPING: ("Stopping…", "warn"),
    JobState.COMPLETED: ("Completed", "ok"),
    JobState.STOPPED: ("Stopped", "idle"),
    JobState.FAILED: ("Failed", "error"),
}


@st.cache_data(max_entries=2, show_spinner=False)
def _file_bytes(path: str, mtime_ns: int) -> bytes:
    """Read a result file once; ``mtime_ns`` is part of the cache key."""
    return Path(path).read_bytes()


def file_bytes(path: Path) -> bytes:
    return _file_bytes(str(path), path.stat().st_mtime_ns)


# --------------------------------------------------------------------------- #
# Generic preview (Overview and Object Tracking pages)
# --------------------------------------------------------------------------- #
def render_preview(page: str, show_links: bool = True) -> None:
    """Shows the processed frame of whichever session is active, with its status.

    ``show_links`` adds shortcuts to the two input pages (the Overview already has its own).
    """
    with st.container(key=f"card_preview_{page.lower().replace(' ', '_')}"):
        card_title("Live preview", "image", "Bounding boxes, labels and tracking IDs on the latest processed frame")

        def _body() -> None:
            view = view_for(page)
            frame = view.latest_frame()
            if frame is None:
                empty_state(
                    "image", "No active session",
                    "Start the webcam or analyze a video to see detections and tracking IDs here.", min_height=300,
                )
            else:
                st.image(frame, channels="BGR", output_format="JPEG")
            totals = view.totals()
            if view.source == "none":
                status_bar(chip("Idle", "idle"), "Nothing is being processed")
            else:
                status_bar(
                    chip("Live" if view.live else "Finished", "live" if view.live else "ok"),
                    view.label, f'{totals["frames_processed"]:,} frames',
                )

        live = view_for(page).live
        live_fragment(_body, live, every=0.7, finished=lambda: not view_for(page).live)
        if show_links:
            c1, c2, _ = st.columns([1, 1, 2])
            c1.button("Open Live Detection", key="qa_webcam", on_click=_goto, args=("Live Detection",))
            c2.button("Open Video Analysis", key="qa_video", on_click=_goto, args=("Video Analysis",))


def _goto(page: str) -> None:
    st.session_state["page"] = page


# --------------------------------------------------------------------------- #
# Video upload and processing
# --------------------------------------------------------------------------- #
def _handle_upload(uploaded: Any) -> None:
    """Persist a newly chosen file, validate it and remember its metadata."""
    ss = st.session_state
    if uploaded is None:
        if ss.get("upload_error"):  # the user removed an invalid file: allow choosing it again
            ss["upload_error"], ss["upload_key"] = None, None
        return
    key = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    if key == ss.get("upload_key"):
        return
    path: Path | None = None
    try:
        path = save_uploaded_file(uploaded)
        info = probe_video(path)
    except (InvalidVideoError, OSError) as exc:
        remove_file(path)
        ss.update(upload_key=key, upload_error=str(exc))
        return
    if not job_is_active():
        remove_file(ss.get("upload_path"))
    ss.update(
        upload_key=key, upload_path=str(path), upload_info=info, upload_name=uploaded.name,
        upload_preview=read_frame(path), upload_error=None, start_error=None,
    )


def _remove_upload() -> None:
    ss = st.session_state
    remove_file(ss.get("upload_path"))
    for key in ("upload_path", "upload_info", "upload_name", "upload_preview", "upload_error"):
        ss[key] = None


def _start_video_job() -> None:
    ss = st.session_state
    info = ss.get("upload_info")
    if info is None:
        ss["start_error"] = "Choose a video first."
        return
    settings = get_settings()
    try:
        detector = get_detector(settings)
        ensure_tracker(settings)
    except (ModelLoadError, TrackerError) as exc:
        ss["start_error"] = str(exc)
        return
    job = VideoJob(info.path, ss.get("upload_name") or info.name, settings, detector)
    job.start()
    ss.update(job=job, active_source="video", start_error=None, result_cache=None)


def _video_body() -> None:
    ss = st.session_state
    job: VideoJob | None = ss.get("job")
    snap: JobSnapshot | None = job.snapshot() if job is not None else None
    frame = snap.latest_frame if snap is not None and snap.latest_frame is not None else ss.get("upload_preview")
    if frame is None:
        empty_state("film", "No video selected", "Upload an MP4, AVI or MOV file above, then press Start.", min_height=300)
    else:
        st.image(frame, channels="BGR", output_format="JPEG")

    if snap is None:
        if ss.get("upload_preview") is not None:
            status_bar(chip("Ready", "idle"), "First frame shown — press Start to analyze the video")
        return
    label, kind = _STATE_CHIPS[snap.state]
    details = [f"frame {snap.frames_read:,}" + (f" / {snap.total_frames:,}" if snap.total_frames else "")]
    if snap.eta_s is not None:
        details.append(f"ETA {format_seconds(snap.eta_s)}")
    status_bar(chip(label, kind), *details)
    if snap.state in (JobState.RUNNING, JobState.PAUSED, JobState.STOPPING, JobState.COMPLETED):
        text = f"{snap.progress * 100:.0f}% complete" if snap.total_frames else f"{snap.frames_read:,} frames processed"
        st.progress(min(max(snap.progress, 0.0), 1.0), text=text)
    if snap.error:
        st.error(snap.error)


def _video_controls() -> None:
    ss = st.session_state
    job: VideoJob | None = ss.get("job")
    active = job_is_active()
    paused = bool(job is not None and job.state == JobState.PAUSED)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start", key="ctl_start", type="primary", disabled=ss.get("upload_info") is None or active):
            _start_video_job()
            st.rerun()
    with c2:
        can_toggle = bool(job is not None and job.state in (JobState.RUNNING, JobState.PAUSED))
        if paused:
            if st.button("Resume", key="ctl_resume", disabled=not can_toggle):
                job.resume()
                st.rerun()
        elif st.button("Pause", key="ctl_pause", disabled=not can_toggle):
            job.pause()
            st.rerun()
    with c3:
        if st.button("Stop", key="ctl_stop", disabled=not active):
            job.stop()
            st.rerun()
    if ss.get("start_error"):
        st.error(ss["start_error"])


def _result_downloads(job: VideoJob) -> None:
    """Processed video, CSV and summary of a finished job."""
    ss = st.session_state
    snap = job.snapshot()
    view = job.data_view()
    if not view.has_data:
        return
    cache = ss.get("result_cache")
    if not cache or cache.get("id") != job.session_id:
        cache = {
            "id": job.session_id,
            "csv": view.records().to_csv(index=False).encode("utf-8"),
            "summary": summary_to_markdown(view.summary()).encode("utf-8"),
        }
        ss["result_cache"] = cache
    stem = Path(job.original_name).stem or "video"

    card_title("Results", "download", "Processed video, detection records and analysis summary")
    if snap.output_path is not None:
        if snap.playable:
            st.video(str(snap.output_path))
        else:
            st.info(
                "The processed video was saved, but this computer has no H.264 encoder so the browser "
                "cannot preview it. Download it to play it locally (install `imageio-ffmpeg` for previews)."
            )
    d1, d2, d3 = st.columns(3)
    with d1:
        if snap.output_path is not None:
            st.download_button(
                "Processed video", file_bytes(snap.output_path), f"{stem}_tracked.mp4", "video/mp4", key="dl_video",
            )
    with d2:
        st.download_button("Detections (CSV)", cache["csv"], f"{stem}_detections.csv", "text/csv", key="dl_csv")
    with d3:
        st.download_button("Analysis summary", cache["summary"], f"{stem}_summary.md", "text/markdown", key="dl_summary")
    st.caption(f"Saved as session {job.session_id} — available later under Detection History.")


def render_video_panel() -> None:
    ss = st.session_state
    job: VideoJob | None = ss.get("job")
    active = job_is_active()

    with st.container(key="card_video_input"):
        card_title("Video input", "film", "Upload an MP4, AVI or MOV file. Frames are streamed one at a time.")
        uploaded = st.file_uploader(
            "Upload a video", type=list(config.ALLOWED_VIDEO_EXTENSIONS), disabled=active,
            label_visibility="collapsed", key="uploader",
        )
        _handle_upload(uploaded)
        if ss.get("upload_error"):
            st.error(ss["upload_error"])
        info = ss.get("upload_info")
        if info is not None:
            info_grid(
                [
                    ("File", ss.get("upload_name") or info.name),
                    ("Resolution", f"{info.width} × {info.height}"),
                    ("Frame rate", f"{info.fps:.1f} FPS"),
                    ("Frames", f"{info.frame_count:,}" if info.frame_count else "unknown"),
                    ("Duration", format_seconds(info.duration_s) if info.duration_s else "unknown"),
                    ("Size", human_size(info.size_bytes)),
                ]
            )
            if uploaded is None and not active:
                st.button("Remove video", key="ctl_remove", on_click=_remove_upload)

    with st.container(key="card_video_preview"):
        card_title("Main video preview", "image", "Processed frames with boxes, labels and tracking IDs")
        live_fragment(_video_body, active, every=0.6, finished=lambda: not job_is_active())
        _video_controls()
        if job is not None and not active and job.state in (JobState.COMPLETED, JobState.STOPPED):
            _result_downloads(job)


# --------------------------------------------------------------------------- #
# Webcam (streamlit-webrtc)
# --------------------------------------------------------------------------- #
WEBRTC_TRANSLATIONS = {
    "start": "Start camera",
    "stop": "Stop camera",
    "select_device": "Select camera",
    "media_api_not_available": "Camera access needs a secure page: open http://localhost:8501 in Chrome or Edge.",
    "device_ask_permission": "Allow camera access in your browser, then wait a moment…",
    "device_not_available": "No camera was found. Connect a webcam and reload the page.",
    "device_access_denied": "Camera access was denied. Click the padlock in the address bar and allow the camera.",
}

CAMERA_HELP = (
    "- Press **Start camera** and click **Allow** when the browser asks for camera access.\n"
    "- Camera blocked? Click the padlock in the address bar → *Camera* → *Allow*, then reload.\n"
    "- Close other apps using the camera (Zoom, Teams, Windows Camera, another browser tab).\n"
    "- On Windows also check *Settings → Privacy & security → Camera* and allow desktop apps.\n"
    "- Use Chrome or Edge and open the app at `http://localhost:8501` (remote hosts need HTTPS).\n"
    "- Stuck on *connecting*? A VPN or firewall may block WebRTC — turn it off or try another network.\n"
    "- Too slow? Pick **YOLO11 Nano**, a smaller image size, or raise *Run detection every Nth frame*."
)


def _save_webcam_session(cam: WebcamProcessor) -> str | None:
    """Store a webcam session under outputs/sessions (once per session)."""
    ss = st.session_state
    saved: set[str] = ss.setdefault("saved_webcam_sessions", set())
    view = cam.data_view()
    if cam.session_id in saved or not view.has_data:
        return None
    save_session(cam.session_id, cam.meta, view.records(), view.frame_stats())
    saved.add(cam.session_id)
    return cam.session_id


def _webcam_status() -> None:
    ss = st.session_state
    cam: WebcamProcessor | None = ss.get("webcam_processor")
    playing = bool(ss.get("webcam_playing"))
    if not playing or cam is None:
        status_bar(chip("Camera off", "idle"), "Press “Start camera” in the video box above")
        return
    gap = cam.seconds_since_last_frame()
    if cam.paused:
        status_bar(chip("Detection paused", "warn"), "Showing the raw camera image")
    elif gap is None:
        status_bar(chip("Starting…", "busy"), "Waiting for the first frame (the model may still be warming up)")
    elif gap > config.WEBCAM_STALL_SECONDS:
        status_bar(chip("No frames", "error"), f"nothing received for {gap:.0f} s")
    else:
        view = cam.data_view()
        status_bar(chip("Live", "live"), f'{view.totals()["frames_processed"]:,} frames processed', f"{cam.log.recent_fps():.1f} FPS")
    if cam.last_error:
        st.warning(f"A frame could not be processed: {cam.last_error}")


def render_webcam_panel() -> None:
    ss = st.session_state
    with st.container(key="card_webcam"):
        card_title("Live camera", "video", "Real-time detection and tracking from your webcam")
        if webrtc_streamer is None:
            st.error(
                "Real-time webcam streaming needs the `streamlit-webrtc` package "
                f"({WEBRTC_ERROR}). Run `pip install -r requirements.txt` and restart the app."
            )
            return
        settings = get_settings()
        try:
            get_detector(settings)
            ensure_tracker(settings)
        except (ModelLoadError, TrackerError) as exc:
            st.error(str(exc))
            st.caption("Fix the problem above (or choose another model under Settings), then reload this page.")
            with st.expander("Troubleshooting"):
                st.markdown(CAMERA_HELP)
            return

        ctx = webrtc_streamer(
            key="visiontrack-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=config.RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"width": {"ideal": 960}, "height": {"ideal": 540}, "frameRate": {"ideal": 30}},
                "audio": False,
            },
            video_processor_factory=lambda: WebcamProcessor(settings),
            async_processing=True,
            translations=WEBRTC_TRANSLATIONS,
        )
        ss["webcam_playing"] = bool(ctx.state.playing)
        processor = ctx.video_processor
        previous: WebcamProcessor | None = ss.get("webcam_processor")
        if processor is not None:
            if previous is not None and previous is not processor:
                _save_webcam_session(previous)  # a new stream started: keep the old session
            processor.update_settings(settings)
            ss["webcam_processor"] = processor
            ss["active_source"] = "webcam"
        cam: WebcamProcessor | None = ss.get("webcam_processor")

        playing = bool(ss["webcam_playing"])
        live_fragment(_webcam_status, True, every=1.0)

        has_data = bool(cam is not None and cam.data_view().has_data)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if cam is not None and cam.paused:
                if st.button("Resume", key="ctl_resume", disabled=not playing):
                    cam.set_paused(False)
                    st.rerun()
            elif st.button("Pause", key="ctl_pause", disabled=not (playing and cam is not None)):
                cam.set_paused(True)
                st.rerun()
        with c2:
            if st.button("Reset", key="wc_reset", disabled=cam is None, help="Clear the log and tracker; IDs restart from 1"):
                cam.reset()
                st.rerun()
        with c3:
            if st.button("Save session", key="wc_save", disabled=not has_data):
                saved = _save_webcam_session(cam)
                st.toast(f"Saved session {saved}" if saved else "This session is already saved.")
        with c4:
            csv = cam.data_view().records().to_csv(index=False).encode("utf-8") if has_data else b""
            st.download_button("Export CSV", csv, "webcam_detections.csv", "text/csv", key="dl_webcam", disabled=not has_data)
        with st.expander("Camera not working?"):
            st.markdown(CAMERA_HELP)
