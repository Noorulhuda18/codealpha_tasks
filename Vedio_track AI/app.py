"""VisionTrack AI — entry point.

Launch with::

    streamlit run app.py
"""

import streamlit as st

import config


def _page_icon():
    try:
        from PIL import Image

        return Image.open(config.LOGO_PATH)
    except Exception:  # noqa: BLE001 - a missing logo must never stop the app
        return "🎯"


# Must be the first Streamlit command.
st.set_page_config(
    page_title=f"{config.APP_NAME} — {config.TAGLINE}",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

from components import inject_css, init_state, persist_widget_state  # noqa: E402  (after set_page_config)
from components.dashboard import render_current_page  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402

init_state()
persist_widget_state()
inject_css()
render_current_page()  # first, so the sidebar footer below shows up-to-date model/system status
render_sidebar()
