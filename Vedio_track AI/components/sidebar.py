"""Custom ChatGPT-style vertical navigation with a compact status footer."""

from __future__ import annotations

import streamlit as st

import config
from components import esc, logo_svg, model_status, render_html, set_page, slugify, system_status
from components.metrics import chip


def render_sidebar() -> None:
    current = st.session_state.get("page", "Overview")
    with st.sidebar:
        render_html(
            f'<div class="vt-brand">{logo_svg(36)}'
            f'<div><div class="vt-brand-name">{esc(config.APP_NAME)}</div>'
            f'<div class="vt-brand-tag">{esc(config.TAGLINE)}</div></div></div>'
        )
        render_html('<div class="vt-nav-label">Workspace</div>')
        for label, _icon in config.PAGES:
            state = "navon" if label == current else "navoff"
            st.button(
                label,
                key=f"{state}_{slugify(label)}",
                on_click=set_page,
                args=(label,),
                help=config.PAGE_DESCRIPTIONS.get(label),
            )

        model_label, model_kind = model_status()
        system_label, system_kind = system_status()
        render_html(
            '<div class="vt-sidebar-footer">'
            f'<div class="row"><span>AI model</span>{chip(model_label, model_kind)}</div>'
            f'<div class="row"><span>System</span>{chip(system_label, system_kind)}</div>'
            f'<div class="row"><span>Version</span><b>v{esc(config.VERSION)}</b></div>'
            "</div>"
        )
