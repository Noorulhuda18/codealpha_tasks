"""HTML building blocks: metric cards, status chips, cards, empty states and live lists.

Every function returns (or renders) markup that uses the classes defined in
``assets/style.css``. All dynamic text is HTML-escaped.
"""

from __future__ import annotations

from typing import Any, Sequence

from components import esc, icon_svg, render_html


# --------------------------------------------------------------------------- #
# Chips and headings
# --------------------------------------------------------------------------- #
def chip(label: str, kind: str = "idle") -> str:
    """Rounded status pill. ``kind``: ok | live | busy | warn | error | idle."""
    return f'<span class="vt-chip {esc(kind)}"><span class="dot"></span>{esc(label)}</span>'


def card_title(title: str, icon: str | None = None, subtitle: str = "") -> None:
    glyph = icon_svg(icon, 16) if icon else ""
    sub = f'<div class="vt-card-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(f'<div class="vt-card-title">{glyph}<span>{esc(title)}</span></div>{sub}')


def section_heading(title: str, subtitle: str = "") -> None:
    sub = f'<div class="vt-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(f'<div class="vt-section"><div class="vt-section-title">{esc(title)}</div>{sub}</div>')


def subtitle(text: str) -> None:
    render_html(f'<div class="vt-subtitle">{esc(text)}</div>')


def muted(text: str) -> None:
    render_html(f'<div class="vt-muted">{esc(text)}</div>')


# --------------------------------------------------------------------------- #
# Metric cards
# --------------------------------------------------------------------------- #
def metric_card(icon: str, label: str, value: str, sub: str = "") -> str:
    return (
        '<div class="vt-metric"><div class="vt-metric-top">'
        f'<span class="vt-metric-label">{esc(label)}</span>'
        f'<span class="vt-metric-icon">{icon_svg(icon, 16)}</span></div>'
        f'<div class="vt-metric-value">{esc(value)}</div>'
        f'<div class="vt-metric-sub">{esc(sub)}</div></div>'
    )


def render_metric_grid(cards: Sequence[str]) -> None:
    render_html(f'<div class="vt-metrics">{"".join(cards)}</div>')


def render_stats_row(totals: dict[str, Any]) -> None:
    """The four headline cards, computed from a ``DataView.totals()`` dictionary."""
    frames = int(totals.get("frames_processed", 0))
    fps = float(totals.get("fps", 0.0))
    latency = f"≈ {1000.0 / fps:.0f} ms per frame" if fps > 0 else "detection + tracking speed"
    render_metric_grid(
        [
            metric_card(
                "scan", "Total objects detected", f"{int(totals.get('unique_tracks', 0)):,}",
                f"unique tracked objects · {int(totals.get('total_detections', 0)):,} detections",
            ),
            metric_card(
                "target", "Currently tracked", f"{int(totals.get('current_tracked', 0)):,}",
                "objects in the latest processed frame",
            ),
            metric_card("zap", "Processing FPS", f"{fps:.1f}", latency),
            metric_card(
                "layers", "Total frames processed", f"{frames:,}",
                "from the active session" if frames else "no frames yet",
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Empty states, info tiles
# --------------------------------------------------------------------------- #
def empty_state(icon: str, title: str, text: str, min_height: int = 0) -> None:
    style = f' style="min-height:{int(min_height)}px"' if min_height else ""
    render_html(
        f'<div class="vt-empty"{style}><div class="ico">{icon_svg(icon, 22, stroke=1.6)}</div>'
        f'<div class="t">{esc(title)}</div><div class="s">{esc(text)}</div></div>'
    )


def info_grid(items: Sequence[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="vt-info"><div class="k">{esc(k)}</div><div class="v" title="{esc(v)}">{esc(v)}</div></div>'
        for k, v in items
    )
    render_html(f'<div class="vt-info-grid">{cells}</div>')


def status_bar(chip_html: str, *details: str) -> None:
    parts = "".join(f'<span class="vt-muted">{esc(d)}</span>' for d in details if d)
    render_html(f'<div class="vt-statusbar">{chip_html}{parts}</div>')


# --------------------------------------------------------------------------- #
# Live lists (right-hand panel)
# --------------------------------------------------------------------------- #
def _bar(percent: float) -> str:
    return f'<div class="vt-bar"><span style="width:{max(0.0, min(100.0, float(percent))):.0f}%"></span></div>'


def activity_feed(recent: Any) -> None:
    """Latest detections (newest first). ``recent`` is a records DataFrame."""
    if recent is None or len(recent) == 0:
        muted("Waiting for detections…")
        return
    rows = []
    for rec in recent.head(8).itertuples(index=False):
        rows.append(
            f'<div class="vt-obj"><span class="id">#{int(rec.track_id)}</span>'
            f'<div><div class="name">{esc(rec.class_name)} <span class="vt-muted">· frame {int(rec.frame):,}</span></div>'
            f'{_bar(rec.confidence_pct)}</div>'
            f'<span class="conf">{rec.confidence_pct:.0f}%</span></div>'
        )
    render_html("".join(rows))


def category_chips(class_summary: Any, limit: int = 12) -> None:
    if class_summary is None or len(class_summary) == 0:
        muted("No object categories yet.")
        return
    tags = "".join(
        f'<span class="vt-tag">{esc(r.class_name)}<b>{int(r.detections):,}</b></span>'
        for r in class_summary.head(limit).itertuples(index=False)
    )
    render_html(f'<div class="vt-chips">{tags}</div>')


def tracked_list(objects: Sequence[dict[str, Any]], limit: int = 10) -> None:
    """Objects tracked in the latest frame: tracking ID, class and confidence."""
    if not objects:
        muted("No objects are being tracked right now.")
        return
    rows = []
    for obj in sorted(objects, key=lambda o: o["track_id"])[:limit]:
        rows.append(
            f'<div class="vt-obj"><span class="id">ID {int(obj["track_id"])}</span>'
            f'<div><div class="name">{esc(obj["class_name"])}</div>{_bar(obj["confidence_pct"])}</div>'
            f'<span class="conf">{float(obj["confidence_pct"]):.0f}%</span></div>'
        )
    extra = len(objects) - limit
    if extra > 0:
        rows.append(f'<div class="vt-muted" style="padding-top:6px">+ {extra} more</div>')
    render_html("".join(rows))
