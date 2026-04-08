"""
VioLane — Streamlit dashboard (primary UI).

Run: `streamlit run dashboard.py` → http://localhost:8501

Layout matches the former FastAPI/HTML monitor: header, control strip, video + HUD,
thumb strip, and right column (LIVE STATS · RULES · RECENT EVENTS).
Processing uses `utils/video_decode.py` (same as `web_app.py` if you still run it).
"""

from __future__ import annotations

import streamlit as st

import config
from utils import streamlit_helpers as sh
from utils.dashboard_theme import inject_theme
from utils.ui_common import model_options

st.set_page_config(
    page_title="VioLane",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚦",
)
inject_theme()

with st.sidebar:
    st.markdown("## VioLane")
    st.caption("Step 1: pick detectors · step 2: upload — only selected weights load.")
    st.caption("Catalog & model paths")
    st.caption(
        f"Plate checkpoint: `{config.PLATE_MODEL_PATH.split('/')[-1]}` · "
        f"scoped plates: **{config.TRUCK_SCOPED_PLATE_ONLY}**"
    )
    sh.render_sidebar_catalog()

labels, default_labels, label_to_id = model_options()
sh.render_video_tab(labels, default_labels, label_to_id)
