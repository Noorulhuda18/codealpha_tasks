"""Plotly figures (dark theme) and the helpers that display charts and tables.

Every figure is built from DataFrames produced by ``backend.analytics`` — i.e.
from frames that were really processed. With no data a figure shows a quiet
"no data yet" message instead of failing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components import stretch

T = config.THEME
SERIES = ["#19C37D", "#8B8BFF", "#F5A524", "#4FB3E8", "#F0616D", "#C792EA", "#7DD3C0", "#E6C26E"]
MAX_POINTS = 1500
GRID = "#323232"


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **stretch())


def show_table(df: pd.DataFrame, height: int | None = None, **kwargs: Any) -> None:
    if height:
        kwargs["height"] = height
    st.dataframe(df, hide_index=True, **stretch(), **kwargs)


# --------------------------------------------------------------------------- #
# Figure scaffolding
# --------------------------------------------------------------------------- #
def _base(fig: go.Figure, title: str, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=14, color=T["text"])),
        height=height,
        margin=dict(l=6, r=10, t=44, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text_muted"], size=12),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1f1f1f", bordercolor=T["border"], font=dict(color=T["text"])),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    return fig


def _empty(title: str, height: int, message: str = "No data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=T["text_muted"], size=13), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, title, height)


def _thin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep charts responsive: at most ``MAX_POINTS`` evenly spaced rows."""
    if len(df) <= MAX_POINTS:
        return df
    step = -(-len(df) // MAX_POINTS)
    return df.iloc[::step]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def object_distribution(summary: pd.DataFrame, height: int = 300, top: int = 10) -> go.Figure:
    """Detections per object class (horizontal bars, largest on top)."""
    title = "Object distribution"
    if summary is None or summary.empty:
        return _empty(title, height)
    data = summary.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["detections"], y=data["class_name"], orientation="h",
            marker=dict(color=T["accent"]), customdata=data[["unique_tracks", "share_pct"]].to_numpy(),
            hovertemplate="<b>%{y}</b><br>%{x:,} detections · %{customdata[0]} unique objects · %{customdata[1]}%<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Detections")
    fig.update_yaxes(showgrid=False)
    return _base(fig, title, height)


def unique_by_class(summary: pd.DataFrame, height: int = 300, top: int = 10) -> go.Figure:
    """Distinct tracked objects per class."""
    title = "Unique tracked objects by class"
    if summary is None or summary.empty:
        return _empty(title, height)
    data = summary.sort_values("unique_tracks", ascending=False).head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["unique_tracks"], y=data["class_name"], orientation="h", marker=dict(color=T["accent_2"]),
            hovertemplate="<b>%{y}</b><br>%{x} unique objects<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Unique tracking IDs")
    fig.update_yaxes(showgrid=False)
    return _base(fig, title, height)


def activity_over_time(activity: pd.DataFrame, height: int = 300) -> go.Figure:
    """Tracked detections per time bin."""
    title = "Detection activity over time"
    if activity is None or activity.empty:
        return _empty(title, height)
    fig = go.Figure(
        go.Scatter(
            x=activity["time_s"], y=activity["detections"], mode="lines", line=dict(color=T["accent"], width=2, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor="rgba(25,195,125,0.12)",
            hovertemplate="t = %{x:.1f} s<br>%{y:,} detections<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Detections per interval", rangemode="tozero")
    return _base(fig, title, height)


def class_activity(class_activity_df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Detections over time, one line per class (long format input)."""
    title = "Activity by class"
    if class_activity_df is None or class_activity_df.empty:
        return _empty(title, height)
    fig = go.Figure()
    for i, (name, part) in enumerate(class_activity_df.groupby("class_name")):
        fig.add_trace(
            go.Scatter(
                x=part["time_s"], y=part["detections"], mode="lines", name=str(name),
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                hovertemplate=f"<b>{name}</b><br>t = %{{x:.1f}} s · %{{y:,}}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Detections per interval", rangemode="tozero")
    return _base(fig, title, height, legend=True)


def fps_chart(stats: pd.DataFrame, height: int = 300) -> go.Figure:
    """Processing throughput (frames per second the model + tracker can sustain)."""
    title = "Processing performance (FPS)"
    if stats is None or stats.empty:
        return _empty(title, height)
    data = _thin(stats)
    window = max(1, len(stats) // 100)
    smooth = data["fps"].rolling(window, min_periods=1).mean()
    fig = go.Figure(
        go.Scatter(
            x=data["time_s"], y=smooth, mode="lines", line=dict(color=T["accent_2"], width=2),
            hovertemplate="t = %{x:.1f} s<br>%{y:.1f} FPS<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frames per second", rangemode="tozero")
    return _base(fig, title, height)


def latency_chart(stats: pd.DataFrame, height: int = 300) -> go.Figure:
    """Per-frame detection + tracking latency."""
    title = "Latency per frame (ms)"
    if stats is None or stats.empty:
        return _empty(title, height)
    data = _thin(stats)
    fig = go.Figure(
        go.Scatter(
            x=data["time_s"], y=data["latency_ms"], mode="lines", line=dict(color=T["warning"], width=1.5),
            hovertemplate="t = %{x:.1f} s<br>%{y:.0f} ms<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Milliseconds", rangemode="tozero")
    return _base(fig, title, height)


def confidence_histogram(records: pd.DataFrame, height: int = 300) -> go.Figure:
    title = "Confidence distribution"
    if records is None or records.empty:
        return _empty(title, height)
    fig = go.Figure(
        go.Histogram(
            x=records["confidence_pct"], xbins=dict(start=0, end=100, size=5), marker=dict(color=T["accent_2"]),
            hovertemplate="%{x}% confidence<br>%{y:,} detections<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Confidence (%)", range=[0, 100])
    fig.update_yaxes(title_text="Detections")
    return _base(fig, title, height)


def active_tracks_chart(stats: pd.DataFrame, height: int = 300) -> go.Figure:
    """Number of objects tracked in each processed frame."""
    title = "Active tracks over time"
    if stats is None or stats.empty:
        return _empty(title, height)
    data = _thin(stats)
    fig = go.Figure(
        go.Scatter(
            x=data["time_s"], y=data["tracked"], mode="lines", line=dict(color=T["accent"], width=2, shape="hv"),
            fill="tozeroy", fillcolor="rgba(25,195,125,0.10)",
            hovertemplate="t = %{x:.1f} s<br>%{y} tracked<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Tracked objects", rangemode="tozero", dtick=1 if data["tracked"].max() < 8 else None)
    return _base(fig, title, height)


def track_timeline(tracks: pd.DataFrame, height: int = 300, max_tracks: int = 40) -> go.Figure:
    """Lifetime of each tracking ID (first to last time it was seen)."""
    title = "Track lifetimes"
    if tracks is None or tracks.empty:
        return _empty(title, height, "No tracks yet")
    data = tracks.sort_values("frames_seen", ascending=False).head(max_tracks).sort_values("first_time_s")
    fig = go.Figure()
    classes = list(dict.fromkeys(data["class_name"]))
    for i, name in enumerate(classes):
        part = data[data["class_name"] == name]
        xs: list[float | None] = []
        ys: list[str | None] = []
        for r in part.itertuples(index=False):
            xs += [r.first_time_s, max(r.last_time_s, r.first_time_s + 0.05), None]
            ys += [f"#{int(r.track_id)}"] * 2 + [None]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=str(name), line=dict(color=SERIES[i % len(SERIES)], width=6),
                hovertemplate=f"<b>{name}</b><br>%{{y}} · %{{x:.1f}} s<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False)
    return _base(fig, title, height, legend=True)
