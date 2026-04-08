"""Viola Lane — Streamlit theme: glassmorphism, Inter, slate gradient."""

from __future__ import annotations

from typing import List

import html as _html_mod
import streamlit as st


def html_escape(s: str) -> str:
    return _html_mod.escape(str(s), quote=True)


VIOLA_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --v-bg-0: #0f172a;
  --v-bg-1: #1e1b4b;
  --v-card: rgba(30, 41, 59, 0.72);
  --v-card-border: rgba(148, 163, 184, 0.12);
  --v-text: #f1f5f9;
  --v-text-2: #94a3b8;
  --v-cyan: #06b6d4;
  --v-red: #ef4444;
  --v-green: #10b981;
  --v-orange: #f97316;
  --v-purple: #a855f7;
  --v-radius: 14px;
  --v-blur: 16px;
}

html, body, [class*="stApp"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif !important;
  color: var(--v-text) !important;
}

.stApp {
  background: linear-gradient(145deg, var(--v-bg-0) 0%, #0c1222 45%, var(--v-bg-1) 100%) !important;
  background-attachment: fixed !important;
}

[data-testid="stHeader"] {
  background: rgba(15, 23, 42, 0.85) !important;
  backdrop-filter: blur(var(--v-blur));
  border-bottom: 1px solid var(--v-card-border) !important;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1280px !important;
}

@media (max-width: 1024px) {
  div[data-testid="column"] {
    width: 100% !important;
    min-width: 100% !important;
  }
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
}

/* Glass panels */
.viola-glass {
  background: var(--v-card) !important;
  backdrop-filter: blur(var(--v-blur));
  -webkit-backdrop-filter: blur(var(--v-blur));
  border: 1px solid var(--v-card-border) !important;
  border-radius: var(--v-radius) !important;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}

/* Upload drop zone */
.viola-drop {
  border: 2px dashed rgba(6, 182, 212, 0.35);
  border-radius: var(--v-radius);
  padding: 1.25rem 1rem;
  text-align: center;
  background: rgba(15, 23, 42, 0.4);
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 0.75rem;
}
.viola-drop:hover {
  border-color: rgba(6, 182, 212, 0.65);
  background: rgba(6, 182, 212, 0.06);
}

/* Buttons */
.stButton > button {
  min-height: 44px !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0891b2 0%, #2563eb 55%, #06b6d4 100%) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 4px 20px rgba(6, 182, 212, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 6px 28px rgba(6, 182, 212, 0.45) !important;
}
.stButton > button[kind="primary"]:disabled {
  opacity: 0.45 !important;
  transform: none !important;
}

/* Inputs & selects */
.stTextInput input, .stNumberInput input, [data-baseweb="input"] input {
  background: rgba(15, 23, 42, 0.6) !important;
  border: 1px solid var(--v-card-border) !important;
  border-radius: 10px !important;
  color: var(--v-text) !important;
}
[data-baseweb="select"] > div {
  background: rgba(15, 23, 42, 0.6) !important;
  border-radius: 10px !important;
  border-color: var(--v-card-border) !important;
}

.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
  border-radius: 999px !important;
}

[data-testid="stExpander"] {
  background: rgba(30, 41, 59, 0.5) !important;
  border: 1px solid var(--v-card-border) !important;
  border-radius: var(--v-radius) !important;
}

/* Checkbox / toggle area */
.stCheckbox label, label[data-baseweb="checkbox"] {
  color: var(--v-text) !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
  padding: 0 !important;
}
[data-testid="stFileUploader"] section > button {
  border-radius: 10px !important;
  border: 1px solid var(--v-card-border) !important;
  background: rgba(6, 182, 212, 0.12) !important;
  color: var(--v-cyan) !important;
}

/* Animations */
@keyframes viola-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }
  50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}
@keyframes viola-shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-3px); }
  75% { transform: translateX(3px); }
}
.viola-stat-card.viol.pulse .viola-stat-val {
  animation: viola-pulse 1.2s ease infinite;
}
.viola-card-shake {
  animation: viola-shake 0.35s ease;
}

/* Injected HTML stats grid */
.viola-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  padding: 0.5rem 0 1.25rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}
.viola-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.viola-logo {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(135deg, #06b6d4, #3b82f6);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.35rem;
  box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
}
.viola-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(90deg, #f1f5f9, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.viola-sub {
  margin: 0.15rem 0 0;
  font-size: 0.8rem;
  color: var(--v-text-2);
}
.viola-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(30, 41, 59, 0.6);
  color: var(--v-text-2);
}
.viola-status-pill.ready { color: #10b981; border-color: rgba(16, 185, 129, 0.35); }
.viola-status-pill.run { color: #06b6d4; border-color: rgba(6, 182, 212, 0.45); animation: viola-pulse 2s ease infinite; }
.viola-status-pill.done { color: #a855f7; border-color: rgba(168, 85, 247, 0.35); }

.viola-stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
  margin-bottom: 0.75rem;
}
.viola-stat-card {
  border-radius: 12px;
  padding: 0.85rem 0.75rem;
  border: 1px solid var(--v-card-border);
  background: rgba(15, 23, 42, 0.45);
  text-align: left;
  transition: transform 0.2s ease;
}
.viola-stat-card:hover { transform: scale(1.02); }
.viola-stat-card .viola-stat-ico { font-size: 1.1rem; margin-bottom: 0.35rem; }
.viola-stat-card .viola-stat-lbl {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--v-text-2);
  margin-bottom: 0.2rem;
}
.viola-stat-card .viola-stat-val {
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.viola-stat-card.v1.shake {
  animation: viola-shake 0.45s ease;
}
.viola-stat-card.v1 .viola-stat-val { color: var(--v-red); }
.viola-stat-card.v2 .viola-stat-val { color: #38bdf8; }
.viola-stat-card.v3 .viola-stat-val { color: var(--v-orange); }
.viola-stat-card.v4 .viola-stat-val { color: var(--v-purple); }

.viola-zones {
  border-radius: var(--v-radius);
  padding: 1rem;
  border: 1px solid var(--v-card-border);
  background: rgba(15, 23, 42, 0.35);
  margin-bottom: 0.75rem;
}
.viola-zones h4 {
  margin: 0 0 0.75rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--v-text-2);
  display: flex;
  align-items: center;
  gap: 0.35rem;
}
.viola-zone-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.45rem 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.08);
  font-size: 0.85rem;
}
.viola-zone-row:last-child { border-bottom: none; }
.viola-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 0.5rem; }
.viola-dot.np { background: var(--v-red); }
.viola-dot.sl { background: #eab308; }
.viola-dot.sr { background: var(--v-green); }

.viola-events {
  border-radius: var(--v-radius);
  padding: 0.85rem;
  border: 1px solid var(--v-card-border);
  background: rgba(15, 23, 42, 0.35);
  max-height: 200px;
  overflow-y: auto;
  font-size: 0.72rem;
  color: var(--v-text-2);
}
.viola-events .ev { padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.viola-mini-preview {
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--v-card-border);
  aspect-ratio: 16/9;
  max-height: 120px;
  background: #020617;
  margin-bottom: 0.65rem;
}
.viola-mini-preview img { width: 100%; height: 100%; object-fit: cover; }

.viola-video-main {
  position: relative;
  border-radius: var(--v-radius);
  overflow: hidden;
  border: 1px solid var(--v-card-border);
  background: #020617;
  aspect-ratio: 16 / 9;
  max-height: min(70vh, 620px);
  margin-top: 0.5rem;
}
.viola-video-main img { width: 100%; height: 100%; object-fit: contain; display: block; }
.viola-hud {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 0.65rem;
  padding: 0.45rem 0.6rem;
  background: rgba(15, 23, 42, 0.88);
  backdrop-filter: blur(8px);
  border-radius: 8px;
  border: 1px solid var(--v-card-border);
  color: var(--v-text-2);
  line-height: 1.45;
}
.viola-hud strong { color: var(--v-cyan); }
.viola-ph {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--v-text-2);
  font-size: 0.88rem;
  text-align: center;
  padding: 1rem;
}

.viola-badge-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.5rem; }
.viola-badge {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  background: rgba(6, 182, 212, 0.12);
  color: var(--v-cyan);
  border: 1px solid rgba(6, 182, 212, 0.25);
}

.viola-hint {
  font-size: 0.75rem;
  color: var(--v-text-2);
  margin-top: 0.5rem;
  line-height: 1.45;
}

section[data-testid="stSidebar"] {
  background: rgba(15, 23, 42, 0.95) !important;
  backdrop-filter: blur(12px);
  border-right: 1px solid var(--v-card-border) !important;
}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{VIOLA_CSS}</style>", unsafe_allow_html=True)


def viola_header_html(*, status: str = "ready", show_live: bool = False) -> str:
    """status: ready | processing | complete"""
    cls = "ready"
    label = "Ready"
    if status == "processing":
        cls, label = "run", "Processing"
    elif status == "complete":
        cls, label = "done", "Complete"
    live = '<span style="margin-left:0.5rem;padding:0.2rem 0.5rem;border-radius:6px;background:rgba(239,68,68,0.2);color:#fca5a5;font-size:0.65rem;font-weight:700">LIVE</span>' if show_live else ""
    return f"""
<div class="viola-header">
  <div class="viola-brand">
    <div class="viola-logo" aria-hidden="true">🚦</div>
    <div>
      <h1 class="viola-title">Viola Lane</h1>
      <p class="viola-sub">Restricted hours · triple seat · helmet · plates</p>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
    <span class="viola-status-pill {cls}"><span style="opacity:0.85">●</span> {label}</span>
    {live}
  </div>
</div>
"""


def viola_upload_shell_html() -> str:
    return """
<div class="viola-drop">
  <div style="font-size:2rem;line-height:1;margin-bottom:0.35rem" aria-hidden="true">☁️</div>
  <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">Drop video, photo, or browse</div>
  <div style="font-size:0.78rem;color:#94a3b8">Video: MP4 · AVI · MOV · MKV · Photo: JPG · PNG · WebP · BMP · TIFF</div>
  <div class="viola-badge-row">
    <span class="viola-badge">MP4</span><span class="viola-badge">MOV</span>
    <span class="viola-badge">JPG</span><span class="viola-badge">PNG</span><span class="viola-badge">WebP</span>
  </div>
</div>
"""


def viola_stats_grid_html(
    viol: int,
    plates: int,
    frame_viol: str,
    fi: int,
    ft: int,
    *,
    pulse_viol: bool = False,
    shake_viol: bool = False,
    fps_line: str = "—",
) -> str:
    pv = " pulse" if pulse_viol else ""
    sv = " shake" if shake_viol else ""
    return f"""
<div class="viola-stats-grid">
  <div class="viola-stat-card v1 viol{pv}{sv}">
    <div class="viola-stat-ico">⚠️</div>
    <div class="viola-stat-lbl">Violations</div>
    <div class="viola-stat-val">{int(viol)}</div>
  </div>
  <div class="viola-stat-card v2">
    <div class="viola-stat-ico">🔤</div>
    <div class="viola-stat-lbl">Plate reads</div>
    <div class="viola-stat-val">{int(plates)}</div>
  </div>
  <div class="viola-stat-card v3">
    <div class="viola-stat-ico">◎</div>
    <div class="viola-stat-lbl">This frame</div>
    <div class="viola-stat-val">{html_escape(str(frame_viol))}</div>
  </div>
  <div class="viola-stat-card v4">
    <div class="viola-stat-ico">▣</div>
    <div class="viola-stat-lbl">Frame</div>
    <div class="viola-stat-val">{int(fi)}<span style="font-size:0.55em;opacity:0.7"> / {int(ft)}</span></div>
  </div>
</div>
<div style="font-size:0.72rem;color:#94a3b8;margin:0.25rem 0 0.5rem">Decode FPS · {html_escape(fps_line)}</div>
"""


def viola_zones_html() -> str:
    return """
<div class="viola-zones">
  <h4>ⓘ Active rules</h4>
  <div class="viola-zone-row"><span>Restricted hours (truck)</span><span style="opacity:0.5">●</span></div>
  <div class="viola-zone-row"><span>Triple seat</span><span style="opacity:0.5">●</span></div>
  <div class="viola-zone-row"><span>Helmet (no-helmet class)</span><span style="opacity:0.5">●</span></div>
  <div style="margin-top:0.5rem;font-size:0.68rem;color:#64748b">Detection-based only — no polygon zones</div>
</div>
"""


def viola_events_html(lines: List[str]) -> str:
    if not lines:
        inner = '<div class="ev" style="opacity:0.65">Events appear here during analysis.</div>'
    else:
        inner = "".join(f'<div class="ev">{html_escape(x)}</div>' for x in lines[:24])
    return f'<div class="viola-events">{inner}</div>'


# Back-compat
violane_header_html = viola_header_html
