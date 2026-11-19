"""End-to-end tests of the video pipeline, analytics, exports and resource cleanup.

A synthetic video with two moving coloured squares is processed with a
colour-based detector double and an IoU tracker double (see conftest.py), so the
*real* pipeline code runs: resizing, drawing, record keeping, H.264 export,
session persistence and thread control.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from backend import analytics
from backend.utils import InvalidVideoError, VideoSink, annotate_frame, probe_video, resize_to_width
from backend.video_processor import FramePipeline, JobState, VideoJob, WebcamProcessor
from backend.tracker import TrackedObjects, TrailBuffer
from conftest import ColorBlobDetector, draw_scene, iou_tracker_factory, write_video


def run_job(video, tmp_path, detector, settings, **kwargs) -> VideoJob:
    job = VideoJob(video, video.name, settings, detector, sessions_dir=tmp_path / "sessions", tracker_factory=iou_tracker_factory, **kwargs)
    job.run()  # synchronous: same code path as the worker thread
    return job


# --------------------------------------------------------------------------- #
# Probing / validation
# --------------------------------------------------------------------------- #
def test_probe_valid_video(sample_video):
    info = probe_video(sample_video)
    assert (info.width, info.height) == (320, 240)
    assert info.frame_count == 30 and info.fps == pytest.approx(15, abs=0.1)
    assert info.size_bytes > 0


def test_probe_rejects_bad_inputs(tmp_path):
    with pytest.raises(InvalidVideoError, match="not found"):
        probe_video(tmp_path / "missing.mp4")
    wrong = tmp_path / "notes.txt"
    wrong.write_text("hello")
    with pytest.raises(InvalidVideoError, match="Unsupported"):
        probe_video(wrong)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not a video" * 100)
    with pytest.raises(InvalidVideoError):
        probe_video(corrupt)


def test_resize_keeps_even_dimensions():
    frame = np.zeros((541, 961, 3), dtype=np.uint8)
    for max_width in (0, 500, 960, 2000):
        out = resize_to_width(frame, max_width)
        assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0
    assert resize_to_width(frame, 480).shape[1] <= 480


# --------------------------------------------------------------------------- #
# Full job
# --------------------------------------------------------------------------- #
def test_video_job_processes_exports_and_saves(sample_video, tmp_path, blob_detector, settings):
    job = run_job(sample_video, tmp_path, blob_detector, settings)
    snap = job.snapshot()
    assert snap.state == JobState.COMPLETED and snap.error is None
    assert snap.progress == 1.0 and snap.frames_read == 30

    # Processed video exists, is readable and has every frame.
    assert snap.output_path is not None
    cap = cv2.VideoCapture(str(snap.output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    ok, last = cap.read()
    cap.release()
    assert ok and last.shape[:2] == (240, 320)

    # Records: two objects per frame with stable IDs from the tracker.
    records = job.log.records_df()
    assert list(records.columns) == analytics.RECORD_COLUMNS
    assert len(records) == 60
    assert records["track_id"].nunique() == 2
    assert records.groupby("class_name")["track_id"].nunique().to_dict() == {"car": 1, "person": 1}
    assert records["confidence_pct"].between(0, 100).all()
    assert (records["x2"] > records["x1"]).all()

    # Session folder was written and can be reloaded.
    sessions = analytics.list_sessions(tmp_path / "sessions")
    assert len(sessions) == 1 and sessions[0]["detections"] == 60 and sessions[0]["has_video"]
    view = analytics.load_session(job.session_id, tmp_path / "sessions")
    assert len(view.records()) == 60 and view.video_path is not None
    assert view.totals()["unique_tracks"] == 2


def test_ids_remain_consistent_across_frames(sample_video, tmp_path, blob_detector, settings):
    job = run_job(sample_video, tmp_path, blob_detector, settings)
    records = job.log.records_df()
    for cname, group in records.groupby("class_name"):
        assert group["track_id"].nunique() == 1, f"{cname} switched IDs"
        assert group["frame"].tolist() == list(range(30))


def test_analytics_are_computed_from_records(sample_video, tmp_path, blob_detector, settings):
    job = run_job(sample_video, tmp_path, blob_detector, settings)
    records, stats = job.log.records_df(), job.log.frame_stats_df()

    summary = analytics.class_summary(records)
    assert summary.set_index("class_name")["detections"].to_dict() == {"person": 30, "car": 30}
    assert analytics.most_frequent_class(records)["detections"] == 30
    tracks = analytics.track_table(records)
    assert len(tracks) == 2 and (tracks["frames_seen"] == 30).all()
    assert analytics.performance_summary(stats)["frames"] == 30
    activity = analytics.activity_over_time(stats)
    assert activity["detections"].sum() == 60

    # Live log counters agree with the DataFrame-based numbers.
    totals = job.log.totals()
    assert totals["total_detections"] == 60 and totals["unique_tracks"] == 2 and totals["frames_processed"] == 30
    assert job.log.class_summary().set_index("class_name")["detections"].to_dict() == {"person": 30, "car": 30}

    exported = analytics.summary_to_markdown(job.data_view().summary())
    assert "person" in exported and "Unique tracked objects: 2" in exported


def test_csv_export_roundtrip(sample_video, tmp_path, blob_detector, settings):
    job = run_job(sample_video, tmp_path, blob_detector, settings)
    csv_path = tmp_path / "sessions" / job.session_id / "detections.csv"
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == analytics.RECORD_COLUMNS and len(loaded) == 60
    exported = job.log.records_df().to_csv(index=False)
    assert exported.splitlines()[0].startswith("frame,video_time_s,detected_at,track_id")


def test_class_filter_and_resize(sample_video, tmp_path, blob_detector):
    only_person = config.ProcessingSettings(max_width=160, conf=0.3, classes=(0,))
    job = run_job(sample_video, tmp_path, blob_detector, only_person)
    records = job.log.records_df()
    assert set(records["class_name"]) == {"person"}
    cap = cv2.VideoCapture(str(job.snapshot().output_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
    cap.release()


def test_frame_interval_skips_inference_but_keeps_video_length(sample_video, tmp_path, blob_detector):
    job = run_job(sample_video, tmp_path, blob_detector, config.ProcessingSettings(max_width=0, frame_interval=3))
    assert blob_detector.calls == 10  # every 3rd of 30 frames
    assert job.log.frames_processed == 10
    cap = cv2.VideoCapture(str(job.snapshot().output_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()


def test_video_without_objects_does_not_crash(empty_video, tmp_path, blob_detector, settings):
    job = run_job(empty_video, tmp_path, blob_detector, settings)
    assert job.state == JobState.COMPLETED
    assert job.log.totals()["total_detections"] == 0 and job.log.totals()["frames_processed"] == 30
    view = job.data_view()
    assert view.records().empty and view.class_summary().empty
    assert analytics.most_frequent_class(view.records()) is None
    assert view.summary()["totals"]["total_detections"] == 0


def test_invalid_file_fails_cleanly(tmp_path, blob_detector, settings):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage" * 200)
    job = run_job(bad, tmp_path, blob_detector, settings)
    snap = job.snapshot()
    assert snap.state == JobState.FAILED and snap.error
    assert not (tmp_path / "sessions" / job.session_id).exists()  # nothing half-written is kept


def test_avi_input_supported(tmp_path, blob_detector, settings):
    video = write_video(tmp_path / "scene.avi", frames=10)
    job = run_job(video, tmp_path, blob_detector, settings)
    assert job.state == JobState.COMPLETED and job.log.frames_processed == 10


# --------------------------------------------------------------------------- #
# Threading: start / pause / resume / stop and resource release
# --------------------------------------------------------------------------- #
class SlowDetector(ColorBlobDetector):
    def detect(self, *args, **kwargs):
        time.sleep(0.02)
        return super().detect(*args, **kwargs)


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_job_pause_resume_stop(tmp_path, settings):
    video = write_video(tmp_path / "long.mp4", frames=150)
    job = VideoJob(video, "long.mp4", settings, SlowDetector(), sessions_dir=tmp_path / "sessions", tracker_factory=iou_tracker_factory)
    job.start()
    assert wait_for(lambda: job.snapshot().frames_read >= 5)

    job.pause()
    assert job.state == JobState.PAUSED
    time.sleep(0.2)
    frozen = job.snapshot().frames_read
    time.sleep(0.3)
    assert job.snapshot().frames_read == frozen  # nothing is processed while paused

    job.resume()
    assert wait_for(lambda: job.snapshot().frames_read > frozen + 3)

    job.stop()
    job.join(10)
    snap = job.snapshot()
    assert snap.state == JobState.STOPPED and snap.frames_read < 150
    # A stopped job still yields a valid partial video + stored session.
    assert snap.output_path is not None and cv2.VideoCapture(str(snap.output_path)).isOpened()
    assert analytics.list_sessions(tmp_path / "sessions")[0]["frames"] == snap.frames_read


def test_resources_are_released(sample_video, tmp_path, blob_detector, settings, monkeypatch):
    opened, released = [], []
    real_capture = cv2.VideoCapture

    class TrackingCapture:
        def __init__(self, *args):
            self._cap = real_capture(*args)
            opened.append(self)

        def release(self):
            released.append(self)
            self._cap.release()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", TrackingCapture)
    job = run_job(sample_video, tmp_path, blob_detector, settings)
    assert job.state == JobState.COMPLETED
    assert opened and len(opened) == len(released)  # every capture opened was released

    # A failing detector must not leak the capture or the writer either.
    class Boom(ColorBlobDetector):
        def detect(self, *a, **k):
            raise RuntimeError("boom")

    opened.clear(); released.clear()
    job = run_job(sample_video, tmp_path, Boom(), settings)
    assert job.state == JobState.FAILED and "boom" in job.snapshot().error
    assert opened and len(opened) == len(released)


def test_live_settings_update_locks_unsafe_fields(sample_video, tmp_path, blob_detector, settings):
    job = VideoJob(sample_video, "scene.mp4", settings, blob_detector, sessions_dir=tmp_path / "s", tracker_factory=iou_tracker_factory)
    job.update_settings(settings.with_updates(conf=0.8, show_trails=True, model_name="other.pt", max_width=100))
    assert job.settings.conf == 0.8 and job.settings.show_trails
    assert job.settings.model_name == settings.model_name and job.settings.max_width == settings.max_width


# --------------------------------------------------------------------------- #
# Drawing, sink and webcam processor
# --------------------------------------------------------------------------- #
def test_annotation_draws_boxes_labels_and_trails(settings):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracked = TrackedObjects(
        np.array([[50, 60, 150, 200]], dtype=np.float32), np.array([5]), np.array([0.87], dtype=np.float32), np.array([0])
    )
    trails = TrailBuffer(10)
    for i in range(6):
        trails.update(np.array([5]), np.array([[60 + i * 5, 100]], dtype=np.float32))
    drawn = annotate_frame(frame.copy(), tracked, {0: "person"}, settings.with_updates(show_trails=True), trails)
    assert drawn.any() and (drawn != frame).any()
    hidden = annotate_frame(frame.copy(), tracked, {0: "person"}, settings.with_updates(show_boxes=False, show_labels=False, show_ids=False, show_conf=False))
    assert not hidden.any()  # every toggle off -> frame untouched


def test_video_sink_writes_playable_file(tmp_path):
    with VideoSink(tmp_path / "out.mp4", 10, (320, 240)) as sink:
        for i in range(12):
            sink.write(draw_scene(i))
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    cap.release()
    sink.close()  # closing twice is harmless
    from backend.utils import VideoWriteError
    with pytest.raises(VideoWriteError):
        sink.write(draw_scene(0))


def test_webcam_processor_pipeline(blob_detector, settings):
    processor = WebcamProcessor(settings, detector_provider=lambda s: blob_detector, tracker_factory=iou_tracker_factory)
    t0 = 1000.0
    for i in range(12):
        out = processor.process_image(draw_scene(i), now=t0 + i / 15)
    assert out.shape == (240, 320, 3)
    totals = processor.log.totals()
    assert totals["frames_processed"] == 12 and totals["unique_tracks"] == 2
    assert processor.data_view().source == "webcam"

    processor.set_paused(True)
    processor.process_image(draw_scene(13), now=t0 + 1)
    assert processor.log.totals()["frames_processed"] == 12  # paused -> no new detections

    processor.set_paused(False)
    processor.reset()
    processor.process_image(draw_scene(14), now=t0 + 2)
    assert processor.log.totals()["frames_processed"] == 1  # session was reset

    processor.update_settings(settings.with_updates(conf=0.9, show_boxes=False))
    processor.process_image(draw_scene(15), now=t0 + 3)  # settings applied without error
    assert processor.last_error is None


def test_webcam_recv_survives_errors(settings):
    av = pytest.importorskip("av")

    class Boom:
        names = {}
        device = "cpu"

        def detect(self, *a, **k):
            raise RuntimeError("inference failed")

    processor = WebcamProcessor(settings, detector_provider=lambda s: Boom(), tracker_factory=iou_tracker_factory)
    frame = av.VideoFrame.from_ndarray(draw_scene(0), format="bgr24")
    out = processor.recv(frame)
    assert out.width == 320 and "inference failed" in processor.last_error
