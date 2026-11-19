"""Detection records, statistics and session persistence.

Everything here is computed from frames that were actually processed:

* :class:`DetectionLog` collects one row per tracked object per processed frame
  (kept as a Pandas DataFrame on demand) plus one statistics row per frame.
* The ``*_summary`` / ``*_over_time`` functions turn those DataFrames into the
  numbers and tables shown on the dashboard and written to the exports.
* ``save_session`` / ``list_sessions`` / ``load_session`` persist finished
  sessions to ``outputs/sessions`` so they can be reviewed later.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from backend.tracker import TrackedObjects
from backend.utils import SESSION_ID_PATTERN

RECORD_COLUMNS = [
    "frame", "video_time_s", "detected_at", "track_id", "class_id", "class_name",
    "confidence_pct", "x1", "y1", "x2", "y2",
]
STAT_COLUMNS = ["frame", "time_s", "raw_detections", "tracked", "latency_ms", "fps"]
CLASS_SUMMARY_COLUMNS = ["class_name", "detections", "unique_tracks", "avg_confidence", "share_pct"]

_RECORD_DTYPES = {
    "frame": "int64", "video_time_s": "float64", "detected_at": "object", "track_id": "int64",
    "class_id": "int64", "class_name": "object", "confidence_pct": "float64",
    "x1": "int64", "y1": "int64", "x2": "int64", "y2": "int64",
}
_STAT_DTYPES = {
    "frame": "int64", "time_s": "float64", "raw_detections": "int64", "tracked": "int64",
    "latency_ms": "float64", "fps": "float64",
}


def empty_records() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_RECORD_DTYPES[c]) for c in RECORD_COLUMNS})


def empty_stats() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_STAT_DTYPES[c]) for c in STAT_COLUMNS})


# --------------------------------------------------------------------------- #
# Live log
# --------------------------------------------------------------------------- #
class DetectionLog:
    """Thread-safe accumulator for detection records and per-frame statistics.

    Written by the processing thread, read by the Streamlit script. Counters are
    maintained incrementally so the live dashboard never has to rebuild a large
    DataFrame just to show totals. Only the newest ``max_records`` rows are kept
    in memory (older rows are dropped, counters stay exact).
    """

    def __init__(self, max_records: int = config.MAX_RECORDS, max_stats: int = config.MAX_FRAME_STATS) -> None:
        self._lock = threading.RLock()
        self._rows: deque[tuple] = deque(maxlen=max_records)
        self._stats: deque[tuple] = deque(maxlen=max_stats)
        self._class_counts: Counter[str] = Counter()
        self._class_conf_sum: defaultdict[str, float] = defaultdict(float)
        self._class_ids: defaultdict[str, set[int]] = defaultdict(set)
        self._all_ids: set[int] = set()
        self._version = 0
        self._records_cache: tuple[int, pd.DataFrame] | None = None
        self._stats_cache: tuple[int, pd.DataFrame] | None = None
        self.total_detections = 0
        self.frames_processed = 0
        self.dropped_records = 0
        self.frame_size: tuple[int, int] | None = None
        self.last_objects: list[dict[str, Any]] = []
        self.created_at = time.time()

    # -- writing ------------------------------------------------------------- #
    def add_frame(
        self,
        frame_idx: int,
        time_s: float,
        tracked: TrackedObjects,
        names: dict[int, str],
        latency_ms: float,
        raw_detections: int | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        detected_at = datetime.now().isoformat(timespec="milliseconds")
        objects: list[dict[str, Any]] = []
        rows: list[tuple] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (int(round(v)) for v in tracked.xyxy[i])
            tid = int(tracked.track_ids[i])
            cid = int(tracked.cls[i])
            cname = names.get(cid, str(cid))
            conf_pct = round(float(tracked.conf[i]) * 100.0, 1)
            rows.append((int(frame_idx), round(float(time_s), 3), detected_at, tid, cid, cname, conf_pct, x1, y1, x2, y2))
            objects.append({"track_id": tid, "class_name": cname, "confidence_pct": conf_pct})

        latency_ms = max(float(latency_ms), 0.001)
        stat_row = (
            int(frame_idx), round(float(time_s), 3),
            int(raw_detections if raw_detections is not None else len(tracked)),
            len(tracked), round(latency_ms, 2), round(1000.0 / latency_ms, 2),
        )
        with self._lock:
            for row in rows:
                if len(self._rows) == self._rows.maxlen:
                    self.dropped_records += 1
                self._rows.append(row)
                _, _, _, tid, _, cname, conf_pct = row[:7]
                self._class_counts[cname] += 1
                self._class_conf_sum[cname] += conf_pct
                self._class_ids[cname].add(tid)
                self._all_ids.add(tid)
            self.total_detections += len(rows)
            self._stats.append(stat_row)
            self.frames_processed += 1
            self.last_objects = objects
            if frame_size:
                self.frame_size = frame_size
            self._version += 1

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
            self._stats.clear()
            self._class_counts.clear()
            self._class_conf_sum.clear()
            self._class_ids.clear()
            self._all_ids.clear()
            self.total_detections = self.frames_processed = self.dropped_records = 0
            self.last_objects = []
            self._records_cache = self._stats_cache = None
            self._version += 1

    # -- reading ------------------------------------------------------------- #
    @property
    def version(self) -> int:
        return self._version

    @property
    def unique_tracks(self) -> int:
        with self._lock:
            return len(self._all_ids)

    def recent_fps(self, window: int = 20) -> float:
        with self._lock:
            latest = list(islice(reversed(self._stats), window))
        if not latest:
            return 0.0
        mean_latency_ms = float(np.mean([row[4] for row in latest]))
        return round(1000.0 / mean_latency_ms, 1) if mean_latency_ms > 0 else 0.0

    def totals(self) -> dict[str, Any]:
        with self._lock:
            last_tracked = self._stats[-1][3] if self._stats else 0
            return {
                "total_detections": self.total_detections,
                "unique_tracks": len(self._all_ids),
                "current_tracked": int(last_tracked),
                "frames_processed": self.frames_processed,
                "fps": self.recent_fps(),
            }

    def class_summary(self) -> pd.DataFrame:
        with self._lock:
            if not self._class_counts:
                return pd.DataFrame(columns=CLASS_SUMMARY_COLUMNS)
            total = sum(self._class_counts.values())
            data = [
                {
                    "class_name": name,
                    "detections": count,
                    "unique_tracks": len(self._class_ids[name]),
                    "avg_confidence": round(self._class_conf_sum[name] / count, 1),
                    "share_pct": round(100.0 * count / total, 1),
                }
                for name, count in self._class_counts.most_common()
            ]
        return pd.DataFrame(data, columns=CLASS_SUMMARY_COLUMNS)

    def records_df(self) -> pd.DataFrame:
        with self._lock:
            if self._records_cache and self._records_cache[0] == self._version:
                return self._records_cache[1]
            rows = list(self._rows)
            version = self._version
        df = _frame_from_rows(rows, RECORD_COLUMNS, _RECORD_DTYPES, empty_records)
        with self._lock:
            self._records_cache = (version, df)
        return df

    def frame_stats_df(self) -> pd.DataFrame:
        with self._lock:
            if self._stats_cache and self._stats_cache[0] == self._version:
                return self._stats_cache[1]
            rows = list(self._stats)
            version = self._version
        df = _frame_from_rows(rows, STAT_COLUMNS, _STAT_DTYPES, empty_stats)
        with self._lock:
            self._stats_cache = (version, df)
        return df

    def recent_records(self, n: int = 50) -> pd.DataFrame:
        with self._lock:
            rows = list(islice(reversed(self._rows), n))
        return _frame_from_rows(rows, RECORD_COLUMNS, _RECORD_DTYPES, empty_records)


def _frame_from_rows(rows: list[tuple], columns: list[str], dtypes: dict[str, str], empty: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return empty()
    return pd.DataFrame.from_records(rows, columns=columns).astype(dtypes)


# --------------------------------------------------------------------------- #
# Statistics on DataFrames
# --------------------------------------------------------------------------- #
def class_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Detections, unique tracks, mean confidence and share per object class."""
    if records.empty:
        return pd.DataFrame(columns=CLASS_SUMMARY_COLUMNS)
    grouped = (
        records.groupby("class_name")
        .agg(detections=("track_id", "size"), unique_tracks=("track_id", "nunique"), avg_confidence=("confidence_pct", "mean"))
        .reset_index()
        .sort_values(["detections", "class_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["avg_confidence"] = grouped["avg_confidence"].round(1)
    grouped["share_pct"] = (100.0 * grouped["detections"] / grouped["detections"].sum()).round(1)
    return grouped[CLASS_SUMMARY_COLUMNS]


def most_frequent_class(records: pd.DataFrame) -> dict[str, Any] | None:
    summary = class_summary(records)
    if summary.empty:
        return None
    top = summary.iloc[0]
    return {"class_name": str(top["class_name"]), "detections": int(top["detections"]), "share_pct": float(top["share_pct"])}


_BIN_CHOICES = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]


def choose_bin_seconds(duration_s: float, max_bins: int = 60) -> float:
    target = duration_s / max_bins if duration_s > 0 else 1
    for choice in _BIN_CHOICES:
        if choice >= target:
            return float(choice)
    return float(_BIN_CHOICES[-1])


def activity_over_time(stats: pd.DataFrame, max_bins: int = 60) -> pd.DataFrame:
    """Tracked detections per time bin (columns: time_s, detections, avg_per_frame, frames)."""
    columns = ["time_s", "detections", "avg_per_frame", "frames"]
    if stats.empty:
        return pd.DataFrame(columns=columns)
    duration = float(stats["time_s"].max() - stats["time_s"].min())
    bin_s = choose_bin_seconds(duration, max_bins)
    binned = stats.assign(bin=(stats["time_s"] // bin_s) * bin_s)
    out = (
        binned.groupby("bin")
        .agg(detections=("tracked", "sum"), avg_per_frame=("tracked", "mean"), frames=("frame", "size"))
        .reset_index()
        .rename(columns={"bin": "time_s"})
    )
    out["avg_per_frame"] = out["avg_per_frame"].round(2)
    return out[columns]


def class_activity_over_time(records: pd.DataFrame, top_n: int = 6, max_bins: int = 60) -> pd.DataFrame:
    """Detections per time bin per class (long format: time_s, class_name, detections)."""
    columns = ["time_s", "class_name", "detections"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    duration = float(records["video_time_s"].max() - records["video_time_s"].min())
    bin_s = choose_bin_seconds(duration, max_bins)
    top = records["class_name"].value_counts().head(top_n).index
    subset = records[records["class_name"].isin(top)]
    binned = subset.assign(bin=(subset["video_time_s"] // bin_s) * bin_s)
    out = binned.groupby(["bin", "class_name"]).size().reset_index(name="detections").rename(columns={"bin": "time_s"})
    return out[columns]


def performance_summary(stats: pd.DataFrame) -> dict[str, float]:
    if stats.empty:
        return {"frames": 0, "avg_fps": 0.0, "median_fps": 0.0, "min_fps": 0.0, "max_fps": 0.0,
                "avg_latency_ms": 0.0, "p95_latency_ms": 0.0, "processing_time_s": 0.0}
    return {
        "frames": int(len(stats)),
        "avg_fps": round(float(stats["fps"].mean()), 2),
        "median_fps": round(float(stats["fps"].median()), 2),
        "min_fps": round(float(stats["fps"].min()), 2),
        "max_fps": round(float(stats["fps"].max()), 2),
        "avg_latency_ms": round(float(stats["latency_ms"].mean()), 2),
        "p95_latency_ms": round(float(stats["latency_ms"].quantile(0.95)), 2),
        "processing_time_s": round(float(stats["latency_ms"].sum() / 1000.0), 2),
    }


TRACK_TABLE_COLUMNS = [
    "track_id", "class_name", "first_frame", "last_frame", "first_time_s", "last_time_s",
    "duration_s", "frames_seen", "avg_confidence",
]


def track_table(records: pd.DataFrame) -> pd.DataFrame:
    """One row per tracking ID: lifetime, class and confidence (tracking activity history)."""
    if records.empty:
        return pd.DataFrame(columns=TRACK_TABLE_COLUMNS)
    grouped = records.groupby("track_id").agg(
        class_name=("class_name", lambda s: s.mode().iat[0]),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        first_time_s=("video_time_s", "min"),
        last_time_s=("video_time_s", "max"),
        frames_seen=("frame", "size"),
        avg_confidence=("confidence_pct", "mean"),
    ).reset_index()
    grouped["duration_s"] = (grouped["last_time_s"] - grouped["first_time_s"]).round(2)
    grouped["avg_confidence"] = grouped["avg_confidence"].round(1)
    return grouped[TRACK_TABLE_COLUMNS].sort_values("first_frame").reset_index(drop=True)


def _py(value: Any) -> Any:
    """Convert NumPy / Pandas scalars so the structure is JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


def build_summary(records: pd.DataFrame, stats: pd.DataFrame, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the analysis summary that is shown, exported and stored with a session."""
    tracks = track_table(records)
    classes = class_summary(records)
    top = most_frequent_class(records)
    tracking = {
        "unique_tracks": int(tracks.shape[0]),
        "avg_track_duration_s": round(float(tracks["duration_s"].mean()), 2) if not tracks.empty else 0.0,
        "longest_track": None,
    }
    if not tracks.empty:
        longest = tracks.sort_values("frames_seen", ascending=False).iloc[0]
        tracking["longest_track"] = {
            "track_id": int(longest["track_id"]),
            "class_name": str(longest["class_name"]),
            "frames_seen": int(longest["frames_seen"]),
            "duration_s": float(longest["duration_s"]),
        }
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session": meta or {},
        "totals": {
            "total_detections": int(len(records)),
            "unique_tracks": int(records["track_id"].nunique()) if not records.empty else 0,
            "frames_processed": int(len(stats)),
        },
        "most_frequent_object": top,
        "classes": classes.to_dict(orient="records"),
        "performance": performance_summary(stats),
        "tracking": tracking,
    }
    return _py(summary)


def summary_to_markdown(summary: dict[str, Any]) -> str:
    session = summary.get("session", {})
    totals = summary.get("totals", {})
    perf = summary.get("performance", {})
    tracking = summary.get("tracking", {})
    top = summary.get("most_frequent_object")
    if top:
        top_text = "{} ({} detections, {}%)".format(top["class_name"], top["detections"], top["share_pct"])
    else:
        top_text = "—"
    lines = [
        f"# {config.APP_NAME} — Analysis Summary",
        "",
        f"- Generated: {summary.get('generated_at', '')}",
        f"- Source: {session.get('source', '—')} ({session.get('source_name', '—')})",
        f"- Model: {session.get('model', '—')} · Tracker: {session.get('tracker', '—')}",
        "",
        "## Totals",
        f"- Total detections (tracked objects, all frames): {totals.get('total_detections', 0)}",
        f"- Unique tracked objects: {totals.get('unique_tracks', 0)}",
        f"- Frames processed: {totals.get('frames_processed', 0)}",
        f"- Most frequent object: {top_text}",
        "",
        "## Detections by class",
        "",
        "| Class | Detections | Unique tracks | Avg confidence | Share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("classes", []):
        lines.append(
            f"| {row['class_name']} | {row['detections']} | {row['unique_tracks']} | {row['avg_confidence']}% | {row['share_pct']}% |"
        )
    lines += [
        "",
        "## Performance",
        f"- Average FPS: {perf.get('avg_fps', 0)} (min {perf.get('min_fps', 0)}, max {perf.get('max_fps', 0)})",
        f"- Latency: avg {perf.get('avg_latency_ms', 0)} ms, p95 {perf.get('p95_latency_ms', 0)} ms",
        f"- Total inference + tracking time: {perf.get('processing_time_s', 0)} s",
        "",
        "## Tracking",
        f"- Unique tracks: {tracking.get('unique_tracks', 0)}",
        f"- Average track duration: {tracking.get('avg_track_duration_s', 0)} s",
    ]
    longest = tracking.get("longest_track")
    if longest:
        lines.append(
            f"- Longest track: #{longest['track_id']} ({longest['class_name']}), "
            f"{longest['frames_seen']} frames, {longest['duration_s']} s"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# DataView: one interface for live logs and stored sessions
# --------------------------------------------------------------------------- #
@dataclass
class DataView:
    """What the UI renders. Backed either by a live :class:`DetectionLog` or by DataFrames."""

    label: str
    source: str  # "video" | "webcam" | "saved" | "none"
    log: DetectionLog | None = None
    records_df: pd.DataFrame | None = None
    stats_df: pd.DataFrame | None = None
    meta: dict[str, Any] | None = None
    live: bool = False
    video_path: Path | None = None
    session_id: str | None = None
    latest_frame_fn: Callable[[], np.ndarray | None] | None = None

    @classmethod
    def empty(cls) -> "DataView":
        return cls(label="No data yet", source="none", records_df=empty_records(), stats_df=empty_stats(), meta={})

    @property
    def has_data(self) -> bool:
        return self.totals()["frames_processed"] > 0

    def records(self) -> pd.DataFrame:
        if self.log is not None:
            return self.log.records_df()
        return self.records_df if self.records_df is not None else empty_records()

    def frame_stats(self) -> pd.DataFrame:
        if self.log is not None:
            return self.log.frame_stats_df()
        return self.stats_df if self.stats_df is not None else empty_stats()

    def class_summary(self) -> pd.DataFrame:
        return self.log.class_summary() if self.log is not None else class_summary(self.records())

    def recent(self, n: int = 50) -> pd.DataFrame:
        if self.log is not None:
            return self.log.recent_records(n)
        return self.records().iloc[::-1].head(n)

    def current_objects(self) -> list[dict[str, Any]]:
        return list(self.log.last_objects) if self.log is not None else []

    def latest_frame(self) -> np.ndarray | None:
        return self.latest_frame_fn() if self.latest_frame_fn else None

    def totals(self) -> dict[str, Any]:
        if self.log is not None:
            return self.log.totals()
        records, stats = self.records(), self.frame_stats()
        perf = performance_summary(stats)
        return {
            "total_detections": int(len(records)),
            "unique_tracks": int(records["track_id"].nunique()) if not records.empty else 0,
            "current_tracked": int(stats["tracked"].iloc[-1]) if not stats.empty else 0,
            "frames_processed": int(len(stats)),
            "fps": perf["avg_fps"],
        }

    def summary(self) -> dict[str, Any]:
        return build_summary(self.records(), self.frame_stats(), self.meta or {})


# --------------------------------------------------------------------------- #
# Session persistence
# --------------------------------------------------------------------------- #
def save_session(
    session_id: str,
    meta: dict[str, Any],
    records: pd.DataFrame,
    stats: pd.DataFrame,
    sessions_dir: Path | None = None,
) -> Path:
    """Write ``detections.csv``, ``frame_stats.csv`` and ``summary.json`` for a session."""
    root = Path(sessions_dir or config.SESSIONS_DIR)
    folder = root / session_id
    folder.mkdir(parents=True, exist_ok=True)
    records.to_csv(folder / "detections.csv", index=False)
    stats.to_csv(folder / "frame_stats.csv", index=False)
    summary = build_summary(records, stats, meta)
    with open(folder / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return folder


def list_sessions(sessions_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return metadata of stored sessions (newest first)."""
    root = Path(sessions_dir or config.SESSIONS_DIR)
    if not root.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir(), reverse=True):
        summary_path = folder / "summary.json"
        if not folder.is_dir() or not SESSION_ID_PATTERN.match(folder.name) or not summary_path.exists():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        meta = summary.get("session", {})
        video = folder / "processed.mp4"
        sessions.append(
            {
                "session_id": folder.name,
                "created_at": meta.get("created_at", ""),
                "source": meta.get("source", ""),
                "source_name": meta.get("source_name", ""),
                "model": meta.get("model", ""),
                "tracker": meta.get("tracker", ""),
                "frames": summary.get("totals", {}).get("frames_processed", 0),
                "detections": summary.get("totals", {}).get("total_detections", 0),
                "unique_tracks": summary.get("totals", {}).get("unique_tracks", 0),
                "avg_fps": summary.get("performance", {}).get("avg_fps", 0),
                "has_video": video.exists(),
            }
        )
    return sessions


def load_session(session_id: str, sessions_dir: Path | None = None) -> DataView:
    """Load a stored session into a :class:`DataView` (raises ``FileNotFoundError``)."""
    if not SESSION_ID_PATTERN.match(session_id):
        raise FileNotFoundError(f"Invalid session id: {session_id}")
    folder = Path(sessions_dir or config.SESSIONS_DIR) / session_id
    if not (folder / "summary.json").exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with open(folder / "summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    records = pd.read_csv(folder / "detections.csv") if (folder / "detections.csv").exists() else empty_records()
    stats = pd.read_csv(folder / "frame_stats.csv") if (folder / "frame_stats.csv").exists() else empty_stats()
    if not records.empty:
        records = records.astype({k: v for k, v in _RECORD_DTYPES.items() if k in records.columns and v != "object"})
    if not stats.empty:
        stats = stats.astype({k: v for k, v in _STAT_DTYPES.items() if k in stats.columns})
    video = folder / "processed.mp4"
    meta = summary.get("session", {})
    return DataView(
        label=f"{meta.get('source_name', session_id)}",
        source="saved",
        records_df=records,
        stats_df=stats,
        meta=meta,
        live=False,
        video_path=video if video.exists() else None,
        session_id=session_id,
    )


def delete_session(session_id: str, sessions_dir: Path | None = None) -> bool:
    if not SESSION_ID_PATTERN.match(session_id):
        return False
    folder = Path(sessions_dir or config.SESSIONS_DIR) / session_id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
        return True
    return False
