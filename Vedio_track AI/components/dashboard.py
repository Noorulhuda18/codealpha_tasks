"""Page layouts: Overview, Live Detection, Video Analysis, Object Tracking,
Detection History, Analytics and Settings.

All numbers shown here come from a :class:`backend.analytics.DataView`, i.e. from
frames that were actually processed — nothing on these pages is hard-coded.
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd
import streamlit as st

import config
from backend import analytics, model_loader
from backend.analytics import DataView
from backend.model_loader import ModelLoadError
from backend.utils import human_size, remove_file
from components import (
    CFG_DEFAULTS,
    charts,
    live_fragment,
    model_status,
    render_html,
    set_page,
    slugify,
    system_status,
)
from components.charts import show_chart, show_table
from components.metrics import (
    activity_feed,
    card_title,
    category_chips,
    chip,
    empty_state,
    info_grid,
    metric_card,
    muted,
    render_metric_grid,
    render_stats_row,
    section_heading,
    tracked_list,
)
from components.video_panel import (
    clear_model_cache,
    get_detector,
    render_preview,
    render_video_panel,
    render_webcam_panel,
    sync_live_settings,
    view_for,
)

LIVE_PAGE = "Live Detection"

RECORD_LABELS = {
    "frame": "Frame", "video_time_s": "Time (s)", "detected_at": "Timestamp", "track_id": "Track ID",
    "class_name": "Class", "confidence_pct": "Confidence %", "x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2",
}
TRACK_LABELS = {
    "track_id": "Track ID", "class_name": "Class", "first_frame": "First frame", "last_frame": "Last frame",
    "first_time_s": "First seen (s)", "last_time_s": "Last seen (s)", "duration_s": "Duration (s)",
    "frames_seen": "Frames seen", "avg_confidence": "Avg confidence %",
}
CLASS_LABELS = {
    "class_name": "Class", "detections": "Detections", "unique_tracks": "Unique objects",
    "avg_confidence": "Avg confidence %", "share_pct": "Share %",
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _live(page: str) -> bool:
    """Should live panels on ``page`` refresh themselves?"""
    return page == LIVE_PAGE or view_for(page).live


def _finished(page: str) -> Callable[[], bool]:
    if page == LIVE_PAGE:  # the webcam page keeps polling; the stream is started/stopped by the browser
        return lambda: False
    return lambda: not view_for(page).live


def dynamic(page: str, render: Callable[[], None], every: float = 1.0) -> None:
    live_fragment(render, _live(page), every=every, finished=_finished(page))


def _display(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    cols = [c for c in labels if c in df.columns]
    return df[cols].rename(columns=labels)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header(page: str) -> None:
    model_label, model_kind = model_status()
    system_label, system_kind = system_status()
    left, middle, right = st.columns([5, 4, 1.5], vertical_alignment="center")
    with left:
        render_html(
            f'<div class="vt-header-title">{page}</div>'
            f'<div class="vt-header-desc">{config.PAGE_DESCRIPTIONS.get(page, "")}</div>'
        )
    with middle:
        render_html(f'<div class="vt-header-status">{chip(system_label, system_kind)}{chip(model_label, model_kind)}</div>')
    with right:
        st.button("Settings", key="hdr_settings", on_click=set_page, args=("Settings",), disabled=page == "Settings")


# --------------------------------------------------------------------------- #
# Detection configuration widgets (all bound to session state, see components.get_settings)
# --------------------------------------------------------------------------- #
def render_config_controls(full: bool = False) -> None:
    st.selectbox("Model", list(config.MODEL_OPTIONS), format_func=config.MODEL_OPTIONS.get, key="cfg_model")
    if full:
        st.selectbox("Compute device", list(config.DEVICE_OPTIONS), format_func=config.DEVICE_OPTIONS.get, key="cfg_device")
    st.selectbox("Tracker", list(config.TRACKER_OPTIONS), format_func=config.TRACKER_OPTIONS.get, key="cfg_tracker")
    st.slider("Confidence threshold", 0.05, 0.95, step=0.05, key="cfg_conf")
    if full:
        st.slider("IoU threshold (overlap suppression)", 0.1, 0.9, step=0.05, key="cfg_iou")
        st.selectbox("Inference image size (px)", list(config.IMGSZ_OPTIONS), key="cfg_imgsz")
        st.slider("Run detection every Nth frame", 1, 10, key="cfg_interval",
                  help="Skipped frames reuse the last tracks. Higher values are faster on slow CPUs.")
        st.selectbox("Maximum frame width", list(config.MAX_WIDTH_OPTIONS), key="cfg_max_width",
                     format_func=lambda w: "Original size" if w == 0 else f"{w} px")
        st.slider("Tracker memory (frames a lost track is kept)", 10, 120, step=5, key="cfg_track_buffer")
        st.slider("Tracker matching threshold", 0.5, 0.95, step=0.05, key="cfg_match_thresh")
    c1, c2 = st.columns(2)
    with c1:
        st.toggle("Bounding boxes", key="cfg_show_boxes")
        st.toggle("Tracking IDs", key="cfg_show_ids")
        if full:
            st.toggle("Confidence scores", key="cfg_show_conf")
    with c2:
        st.toggle("Object labels", key="cfg_show_labels")
        st.toggle("Movement trails", key="cfg_show_trails")
    if full:
        st.slider("Trail length (positions)", 5, 120, step=5, key="cfg_trail_len")
        st.multiselect("Detect only these classes (empty = all 80)", config.COCO_CLASSES, key="cfg_classes",
                       placeholder="All COCO classes")
    else:
        with st.expander("Object classes"):
            st.multiselect("Detect only these classes (empty = all 80)", config.COCO_CLASSES, key="cfg_classes",
                           placeholder="All COCO classes", label_visibility="collapsed")


def _reset_settings() -> None:
    for key, default in CFG_DEFAULTS.items():
        st.session_state[key] = list(default) if isinstance(default, list) else default


# --------------------------------------------------------------------------- #
# Right-hand panel and lower dashboard (shared by the workspace pages)
# --------------------------------------------------------------------------- #
def _right_panel(page: str) -> None:
    slug = slugify(page)

    def _body() -> None:
        view = view_for(page)
        with st.container(key=f"card_activity_{slug}"):
            card_title("Live detection activity", "zap")
            activity_feed(view.recent(8))
        with st.container(key=f"card_categories_{slug}"):
            card_title("Detected object categories", "scan")
            category_chips(view.class_summary())
        with st.container(key=f"card_tracks_{slug}"):
            card_title("Tracking IDs", "target", "Objects in the latest frame")
            tracked_list(view.current_objects())

    dynamic(page, _body, every=1.0)
    with st.container(key=f"card_config_{slug}"):
        card_title("Detection configuration", "cpu")
        render_config_controls(full=False)
        st.caption("Confidence, classes and overlays apply immediately. Model, tracker and frame settings "
                   "apply to the next run (see Settings for all options).")


def _recent_table(view: DataView) -> None:
    recent = view.recent(15)
    if recent.empty:
        muted("Detection records will appear here as frames are processed.")
    else:
        show_table(_display(recent, RECORD_LABELS), height=330)


def _lower_dashboard(page: str) -> None:
    section_heading("Analytics", "Charts update from the frames processed so far.")
    slug = slugify(page)

    def _body() -> None:
        view = view_for(page)
        stats = view.frame_stats()
        c1, c2, c3 = st.columns(3)
        with c1, st.container(key=f"card_dist_{slug}"):
            show_chart(charts.object_distribution(view.class_summary()), f"{slug}_dist")
        with c2, st.container(key=f"card_act_{slug}"):
            show_chart(charts.activity_over_time(analytics.activity_over_time(stats)), f"{slug}_act")
        with c3, st.container(key=f"card_perf_{slug}"):
            show_chart(charts.fps_chart(stats), f"{slug}_perf")
        with st.container(key=f"card_recent_{slug}"):
            card_title("Recent detection history", "history", "Newest detections first — see Detection History for all records")
            _recent_table(view)

    dynamic(page, _body, every=2.0)


def _workspace(page: str, main_panel: Callable[[], None], lower: Callable[[str], None] = _lower_dashboard) -> None:
    dynamic(page, lambda: render_stats_row(view_for(page).totals()), every=1.0)
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        main_panel()
    with right:
        _right_panel(page)
    lower(page)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    with st.container(key="card_welcome"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            render_html(
                '<div class="vt-kicker">Workspace</div>'
                '<div class="vt-hero-title">Computer Vision Workspace</div>'
                '<div class="vt-hero-text">Detect objects with a pretrained YOLO model and follow every one of them '
                "across frames with ByteTrack or BoT-SORT. Start with your webcam or analyze a video file — everything "
                "runs locally on this computer.</div>"
            )
        with right:
            c1, c2 = st.columns(2)
            c1.button("Start webcam", key="qa_webcam", type="primary", on_click=set_page, args=(LIVE_PAGE,))
            c2.button("Analyze video", key="qa_video", on_click=set_page, args=("Video Analysis",))
    _workspace("Overview", lambda: render_preview("Overview", show_links=False))


def page_live() -> None:
    _workspace(LIVE_PAGE, render_webcam_panel)


def page_video() -> None:
    _workspace("Video Analysis", render_video_panel)


def _lower_tracking(page: str) -> None:
    section_heading("Tracking analytics", "Identity lifetimes and activity, from the tracker's real output.")

    def _body() -> None:
        view = view_for(page)
        tracks = analytics.track_table(view.records())
        c1, c2 = st.columns(2)
        with c1, st.container(key="card_track_life"):
            show_chart(charts.track_timeline(tracks, height=340), "tracking_life")
        with c2, st.container(key="card_track_active"):
            show_chart(charts.active_tracks_chart(view.frame_stats(), height=340), "tracking_active")
        with st.container(key="card_track_history"):
            card_title("Tracking activity history", "history", "One row per tracking ID")
            if tracks.empty:
                muted("Tracks appear here once objects have been followed across frames.")
            else:
                show_table(_display(tracks, TRACK_LABELS), height=340)

    dynamic(page, _body, every=2.0)


def page_tracking() -> None:
    _workspace("Object Tracking", lambda: render_preview("Object Tracking"), lower=_lower_tracking)


# -- Detection History ----------------------------------------------------- #
def _history_current() -> None:
    view = view_for("Detection History")
    if not view.has_data or view.records().empty:
        empty_state(
            "history", "No detections recorded yet",
            "Run the webcam or process a video. Every detection is stored with its class, tracking ID, "
            "confidence, bounding box, frame number and timestamp.",
            min_height=260,
        )
        return
    records = view.records()
    ss = st.session_state
    classes = sorted(records["class_name"].unique())
    ids = sorted(int(i) for i in records["track_id"].unique())
    ss["flt_classes"] = [c for c in ss.get("flt_classes", []) if c in classes]
    ss["flt_ids"] = [i for i in ss.get("flt_ids", []) if i in ids]
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.multiselect("Class", classes, key="flt_classes", placeholder="All classes")
    c2.multiselect("Tracking ID", ids, key="flt_ids", placeholder="All IDs")
    min_conf = c3.slider("Minimum confidence %", 0, 100, 0, key="flt_min_conf")
    out = records
    if ss["flt_classes"]:
        out = out[out["class_name"].isin(ss["flt_classes"])]
    if ss["flt_ids"]:
        out = out[out["track_id"].isin(ss["flt_ids"])]
    out = out[out["confidence_pct"] >= min_conf]
    shown = out.tail(3000).iloc[::-1]
    st.caption(f"{len(out):,} of {len(records):,} records match · showing the newest {len(shown):,} · {view.label}")
    show_table(_display(shown, RECORD_LABELS), height=430)
    if view.live:
        st.info("This session is still running; the export contains the records up to now.")
    st.download_button("Export filtered records (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "detections.csv", "text/csv", key="dl_history_csv")


def _delete_session_flow(session_id: str) -> None:
    ss = st.session_state
    if ss.get("confirm_delete") == session_id:
        st.warning("Delete this session and its processed video permanently?")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Yes, delete", key="confirm_yes"):
            analytics.delete_session(session_id)
            ss["confirm_delete"] = None
            st.toast("Session deleted.")
            st.rerun()
        if c2.button("Cancel", key="confirm_no"):
            ss["confirm_delete"] = None
            st.rerun()
    elif st.button("Delete", key="hist_delete"):
        ss["confirm_delete"] = session_id
        st.rerun()


def _open_in_analytics(session_id: str) -> None:
    st.session_state["analytics_source"] = session_id
    st.session_state["page"] = "Analytics"


def _history_saved() -> None:
    sessions = analytics.list_sessions()
    top_left, top_right = st.columns([5, 1])
    top_right.button("Refresh", key="hist_refresh")
    if not sessions:
        with top_left:
            empty_state(
                "history", "No saved sessions yet",
                "Finished video analyses are saved automatically. Webcam sessions are saved with “Save session” "
                "(or automatically when a new stream starts).",
                min_height=240,
            )
        return
    labels = {
        s["session_id"]: f'{s["created_at"].replace("T", " ")[:16]} · {s["source"] or "?"} · '
                         f'{s["source_name"] or "n/a"} · {s["unique_tracks"]} objects'
        for s in sessions
    }
    chosen = top_left.selectbox("Saved session", list(labels), format_func=labels.get, key="saved_session")
    meta = next(s for s in sessions if s["session_id"] == chosen)
    try:
        view = analytics.load_session(chosen)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        st.error(f"This session could not be loaded: {exc}")
        return
    render_metric_grid([
        metric_card("scan", "Unique objects", f'{meta["unique_tracks"]:,}', f'{meta["detections"]:,} detections'),
        metric_card("layers", "Frames processed", f'{meta["frames"]:,}', f'{meta["source"]} · {meta["model"]}'),
        metric_card("zap", "Average FPS", f'{meta["avg_fps"]:.1f}', f'tracker: {meta["tracker"]}'),
        metric_card("target", "Most frequent",
                    str(view.class_summary().iloc[0]["class_name"]) if not view.class_summary().empty else "—",
                    "object class"),
    ])
    records = view.records()
    c1, c2 = st.columns([1, 2])
    with c1, st.container(key="card_hist_dist"):
        show_chart(charts.object_distribution(view.class_summary(), height=330), "hist_saved_dist")
    with c2, st.container(key="card_hist_table"):
        card_title("Detection records", "history", "Newest 3,000 rows — download the CSV for everything")
        if records.empty:
            muted("This session has no detection records.")
        else:
            show_table(_display(records.tail(3000).iloc[::-1], RECORD_LABELS), height=330)

    folder = config.SESSIONS_DIR / chosen
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if (folder / "detections.csv").exists():
            st.download_button("Detections (CSV)", (folder / "detections.csv").read_bytes(),
                               f"{chosen}_detections.csv", "text/csv", key="dl_saved_csv")
    with d2:
        st.download_button("Analysis summary", analytics.summary_to_markdown(view.summary()).encode("utf-8"),
                           f"{chosen}_summary.md", "text/markdown", key="dl_saved_summary")
    with d3:
        st.button("Open in Analytics", key="hist_open", on_click=_open_in_analytics, args=(chosen,))
    with d4:
        _delete_session_flow(chosen)
    if view.video_path is not None:
        if st.toggle(f"Show processed video ({human_size(view.video_path.stat().st_size)})", key="show_saved_video"):
            if (view.meta or {}).get("video_playable_in_browser", True):
                st.video(str(view.video_path))
            else:
                st.info("This video cannot be previewed in the browser; download it instead.")
            st.download_button("Processed video", view.video_path.read_bytes(), f"{chosen}_tracked.mp4",
                               "video/mp4", key="dl_saved_video")


def page_history() -> None:
    current, saved = st.tabs(["Current session", "Saved sessions"])
    with current:
        _history_current()
    with saved:
        _history_saved()


# -- Analytics ------------------------------------------------------------- #
def _analytics_view() -> DataView | None:
    ss = st.session_state
    sessions = [s["session_id"] for s in analytics.list_sessions()]
    options = ["live"] + sessions
    if ss.get("analytics_source") not in options:
        ss["analytics_source"] = "live"
    names = {"live": "Current session"}
    names.update({sid: f"Saved · {sid}" for sid in sessions})
    st.selectbox("Data source", options, format_func=names.get, key="analytics_source")
    if ss["analytics_source"] == "live":
        return view_for("Analytics")
    try:
        return analytics.load_session(ss["analytics_source"])
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load the session: {exc}")
        return None


def _analytics_body(get_view: Callable[[], DataView | None]) -> None:
    view = get_view()
    if view is None or not view.has_data:
        return
    records, stats, classes = view.records(), view.frame_stats(), view.class_summary()
    summary = view.summary()
    totals, perf, tracking = summary["totals"], summary["performance"], summary["tracking"]
    top = summary["most_frequent_object"]
    longest = tracking.get("longest_track")
    render_metric_grid([
        metric_card("target", "Unique tracked objects", f'{totals["unique_tracks"]:,}', "distinct tracking IDs"),
        metric_card("scan", "Total detections", f'{totals["total_detections"]:,}', "object-frames over the session"),
        metric_card("image", "Most frequent object", top["class_name"] if top else "—",
                    f'{top["detections"]:,} detections · {top["share_pct"]}%' if top else ""),
        metric_card("zap", "Average FPS", f'{perf["avg_fps"]:.1f}', f'median {perf["median_fps"]:.1f} · min {perf["min_fps"]:.1f}'),
    ])
    avg_conf = float(records["confidence_pct"].mean()) if not records.empty else 0.0
    render_metric_grid([
        metric_card("layers", "Frames processed", f'{totals["frames_processed"]:,}', f'{perf["processing_time_s"]:.1f} s of compute'),
        metric_card("cpu", "Average latency", f'{perf["avg_latency_ms"]:.0f} ms', f'p95 {perf["p95_latency_ms"]:.0f} ms'),
        metric_card("crosshair", "Average track duration", f'{tracking["avg_track_duration_s"]:.1f} s',
                    f'longest: #{longest["track_id"]} {longest["class_name"]} ({longest["duration_s"]:.1f} s)' if longest else ""),
        metric_card("scan", "Average confidence", f"{avg_conf:.0f}%", "across all detections"),
    ])
    activity = analytics.activity_over_time(stats)
    row1 = st.columns(3)
    row2 = st.columns(3)
    figures = [
        (row1[0], "an_dist", charts.object_distribution(classes)),
        (row1[1], "an_unique", charts.unique_by_class(classes)),
        (row1[2], "an_act", charts.activity_over_time(activity)),
        (row2[0], "an_fps", charts.fps_chart(stats)),
        (row2[1], "an_lat", charts.latency_chart(stats)),
        (row2[2], "an_conf", charts.confidence_histogram(records)),
    ]
    for column, key, figure in figures:
        with column, st.container(key=f"card_{key}"):
            show_chart(figure, key)
    tracks = analytics.track_table(records)
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_an_classact"):
        show_chart(charts.class_activity(analytics.class_activity_over_time(records)), "an_classact")
    with c2, st.container(key="card_an_active"):
        show_chart(charts.active_tracks_chart(stats), "an_active")
    t1, t2 = st.columns(2)
    with t1, st.container(key="card_an_classtable"):
        card_title("Total detections by object class", "scan")
        show_table(_display(classes, CLASS_LABELS), height=320)
    with t2, st.container(key="card_an_tracktable"):
        card_title("Tracking activity history", "history")
        show_table(_display(tracks, TRACK_LABELS), height=320)


def page_analytics() -> None:
    view = _analytics_view()
    if view is None:
        return
    if not view.has_data:
        empty_state(
            "chart", "No analytics yet",
            "Analytics are calculated from real processed frames. Start the webcam or process a video first, "
            "or choose a saved session above.",
            min_height=260,
        )
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("Open Live Detection", key="qa_webcam", on_click=set_page, args=(LIVE_PAGE,))
        c2.button("Open Video Analysis", key="qa_video", on_click=set_page, args=("Video Analysis",))
        return

    saved = view.source == "saved"
    live_fragment(
        lambda: _analytics_body((lambda: view) if saved else (lambda: view_for("Analytics"))),
        live=view.live and not saved, every=2.0, finished=lambda: not view_for("Analytics").live,
    )

    section_heading("Export", "Download the numbers behind these charts.")
    summary = view.summary()
    stem = view.session_id or "session"
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    with d1:
        st.download_button("Analysis summary", analytics.summary_to_markdown(summary).encode("utf-8"),
                           f"{stem}_summary.md", "text/markdown", key="dl_an_md")
    with d2:
        st.download_button("Summary (JSON)", json.dumps(summary, indent=2).encode("utf-8"),
                           f"{stem}_summary.json", "application/json", key="dl_an_json")
    with d3:
        st.download_button("Detections (CSV)", view.records().to_csv(index=False).encode("utf-8"),
                           f"{stem}_detections.csv", "text/csv", key="dl_an_csv")


# -- Settings -------------------------------------------------------------- #
def _clear_uploads() -> None:
    ss = st.session_state
    keep = ss.get("upload_path") if ss.get("job") is not None and ss["job"].is_active else None
    removed = 0
    if config.UPLOADS_DIR.exists():
        for path in config.UPLOADS_DIR.iterdir():
            if path.is_file() and str(path) != keep:
                remove_file(path)
                removed += 1
    if not keep:
        for key in ("upload_path", "upload_info", "upload_name", "upload_preview", "upload_key"):
            ss[key] = None
    st.toast(f"Removed {removed} temporary upload(s).")


def _delete_all_sessions() -> None:
    removed = sum(analytics.delete_session(s["session_id"]) for s in analytics.list_sessions())
    st.session_state["confirm_delete_all"] = False
    st.toast(f"Deleted {removed} saved session(s).")


def page_settings() -> None:
    left, right = st.columns([1.4, 1], gap="medium")
    with left, st.container(key="card_settings_main"):
        card_title("Detection, tracking and performance", "cpu")
        render_config_controls(full=True)
        st.caption("Confidence, IoU, image size, classes and overlays apply live to a running session. "
                   "Model, device, tracker, tracker memory, frame interval and frame width apply to the next run.")
        st.button("Reset all settings to defaults", key="set_reset", on_click=_reset_settings)
    with right:
        with st.container(key="card_settings_model"):
            card_title("Model", "layers", "Pretrained Ultralytics YOLO · runs locally")
            label, kind = model_status()
            render_html(chip(label, kind))
            settings = config.ProcessingSettings(model_name=st.session_state.cfg_model, device=st.session_state.cfg_device)
            c1, c2 = st.columns(2)
            if c1.button("Load model now", key="set_load_model"):
                try:
                    detector = get_detector(settings)
                    st.success(f"{settings.model_name} is ready on {detector.device} · {len(detector.names)} classes.")
                except ModelLoadError as exc:
                    st.error(str(exc))
            c2.button("Clear model cache", key="set_clear_model", on_click=clear_model_cache)
            local = sorted(p.name for p in config.MODELS_DIR.glob("*.pt")) if config.MODELS_DIR.exists() else []
            st.caption("Weights in models/: " + (", ".join(local) if local else "none yet — downloaded on first use"))
            err = model_loader.last_error(settings.model_name)
            if err:
                st.error(err)
        with st.container(key="card_settings_storage"):
            card_title("Storage", "history", f"Results are stored in {config.OUTPUTS_DIR.name}/sessions")
            sessions = analytics.list_sessions()
            muted(f"{len(sessions)} saved session(s)")
            st.button("Clear temporary uploads", key="set_clear_uploads", on_click=_clear_uploads)
            st.checkbox("I want to delete all saved sessions", key="confirm_delete_all")
            st.button("Delete all saved sessions", key="set_delete_all", on_click=_delete_all_sessions,
                      disabled=not st.session_state.get("confirm_delete_all"))
        with st.container(key="card_settings_about"):
            card_title("About", "scan")
            info_grid([("Application", config.APP_NAME), ("Version", config.VERSION),
                       ("Tracker", config.TRACKER_OPTIONS[st.session_state.cfg_tracker])])
            muted(config.TAGLINE)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGE_RENDERERS: dict[str, Callable[[], None]] = {
    "Overview": page_overview,
    LIVE_PAGE: page_live,
    "Video Analysis": page_video,
    "Object Tracking": page_tracking,
    "Detection History": page_history,
    "Analytics": page_analytics,
    "Settings": page_settings,
}


def render_current_page() -> None:
    ss = st.session_state
    page = ss.get("page", "Overview")
    if page not in PAGE_RENDERERS:
        page = ss["page"] = "Overview"
    sync_live_settings()
    if page != LIVE_PAGE:
        ss["webcam_playing"] = False  # leaving the page stops the browser stream
    render_header(page)
    PAGE_RENDERERS[page]()
