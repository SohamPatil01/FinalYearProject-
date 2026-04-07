"""Shared Streamlit UI — Viola Lane glass layout (upload · models · live stats · zones)."""

from __future__ import annotations

import base64
import gc
import html
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

import config
from utils.dashboard_theme import (
    viola_events_html,
    viola_header_html,
    viola_stats_grid_html,
    viola_upload_shell_html,
    viola_zones_html,
)
from utils.pipeline import TrafficPipeline
from utils.ui_common import (
    model_options,
    paths_from_labels,
    resize_preview_rgb,
    write_upload_to_temp,
)
from utils.video_decode import iter_decode_image, iter_decode_video


def _is_upload_image(filename: str) -> bool:
    suf = Path(filename).suffix.lower()
    return suf in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    )


def _show_upload_idle_preview(mini_preview: Any, uploaded: Any) -> None:
    """Show the raw file in the preview column before/without a pipeline run."""
    if uploaded is None:
        return
    try:
        if _is_upload_image(uploaded.name):
            data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is not None and bgr.size > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mini_preview.image(
                    rgb,
                    caption="Uploaded · click **Run analysis**",
                    use_container_width=True,
                )
                return
        mini_preview.markdown(
            f"<p style='color:#94a3b8;margin:0'>Video <strong>{html.escape(uploaded.name)}</strong> — click "
            "<strong>Run analysis</strong></p>",
            unsafe_allow_html=True,
        )
    except Exception:
        mini_preview.caption(f"{html.escape(uploaded.name)} — preview failed; you can still run analysis.")


def _persist_viola_snapshot(
    *,
    main_uri: str,
    done_ev: Dict[str, Any],
    prog_text: str,
) -> None:
    caps = list(done_ev.get("captures") or [])
    unq = len({int(c["tid"]) for c in caps}) if caps else 0
    st.session_state["viola_snapshot"] = {
        "upload_uid": st.session_state.get("vl_upload_uid"),
        "main_uri": main_uri,
        "cum_viol": int(done_ev["cum_viol"]),
        "captures_n": len(caps),
        "frame_idx": int(done_ev["frame_idx"]),
        "est_decoded": int(done_ev["est_decoded"]),
        "fps": float(done_ev["fps"]),
        "unique_plates": unq,
        "prog_text": prog_text,
    }


def _restore_viola_snapshot(
    frame_slot: Any,
    mini_preview: Any,
    stats_panel: Any,
    prog_ph: Any,
) -> bool:
    """Redraw last annotated result after Streamlit rerun (Run button is only True for one run)."""
    snap = st.session_state.get("viola_snapshot")
    if not snap or not snap.get("main_uri"):
        return False
    if snap.get("upload_uid") != st.session_state.get("vl_upload_uid"):
        return False
    if str(st.session_state.get("viola_ui_status")) != "complete":
        return False
    u2 = str(snap["main_uri"])
    stats_panel.markdown(
        viola_stats_grid_html(
            int(snap["cum_viol"]),
            int(snap["captures_n"]),
            "0",
            int(snap["frame_idx"]),
            int(snap["est_decoded"]),
            fps_line=f"{float(snap['fps']):.1f} · done",
        ),
        unsafe_allow_html=True,
    )
    mini_preview.markdown(
        f'<div class="viola-mini-preview"><img src="{u2}" alt="annotated"/></div>',
        unsafe_allow_html=True,
    )
    frame_slot.markdown(
        _viola_frame_html(
            u2,
            int(snap["frame_idx"]),
            int(snap["est_decoded"]),
            float(snap["fps"]),
            3,
            int(snap["cum_viol"]),
            int(snap["unique_plates"]),
        ),
        unsafe_allow_html=True,
    )
    prog_ph.progress(1.0, text=str(snap.get("prog_text", "Complete")))
    return True


def _init_plate_capture_session_state() -> None:
    if "vl_plate_captures" not in st.session_state:
        st.session_state.vl_plate_captures = []
    if "vl_violation_captures" not in st.session_state:
        st.session_state.vl_violation_captures = []
    if "viola_ui_status" not in st.session_state:
        st.session_state.viola_ui_status = "ready"
    if "vl_prev_cum_viol" not in st.session_state:
        st.session_state.vl_prev_cum_viol = 0


def _bgr_jpeg_data_uri(bgr: np.ndarray, quality: int = 78) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _viola_frame_html(
    data_uri: str,
    fi: int,
    ft: int,
    fps: float,
    zones: int,
    viol: int,
    tracked: int,
) -> str:
    return f"""
<div class="viola-video-main">
  <img src="{data_uri}" alt="frame" />
  <div class="viola-hud">
    FRAME <strong>{int(fi)}</strong> / <span style="color:#38bdf8">{int(ft)}</span><br/>
    FPS <span style="color:#38bdf8">{fps:.1f}</span> · ZONES <strong>{int(zones)}</strong><br/>
    VIOLATIONS <strong style="color:#f87171">{int(viol)}</strong> · TRACKED <span style="color:#38bdf8">{int(tracked)}</span>
  </div>
</div>
"""


def _viola_placeholder_html() -> str:
    return """
<div class="viola-video-main">
  <div class="viola-ph">
    Output preview appears here when you run.<br/>
    <span style="font-size:0.88em;opacity:0.85">Annotated frames update <strong>live</strong> during processing.</span>
  </div>
</div>
"""


def _vl_model_key(mid: str) -> str:
    return f"vl_model_active_{mid}"


def _ensure_model_toggle_defaults(
    labels: List[str],
    label_to_id: Dict[str, str],
    default_labels: List[str],
) -> None:
    """Seed session_state once per model id so toggles have stable defaults."""
    for lab in labels:
        mid = label_to_id.get(lab)
        if not mid:
            continue
        k = _vl_model_key(mid)
        if k not in st.session_state:
            st.session_state[k] = lab in default_labels


def _selected_labels_from_toggles(labels: List[str], label_to_id: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for lab in labels:
        mid = label_to_id.get(lab)
        if mid and st.session_state.get(_vl_model_key(mid), False):
            out.append(lab)
    return out


def _refresh_evidence_strips(
    viol_holder: Any,
    plate_holder: Any,
    viol_items: List[Dict[str, Any]],
    plate_items: List[Dict[str, Any]],
    *,
    viol_thumb_max: int,
    plate_thumb_max: int,
    max_each: int = 9,
) -> None:
    viol_holder.empty()
    plate_holder.empty()
    with viol_holder.container():
        st.markdown(
            '<p style="font-size:0.85rem;font-weight:600;color:#f87171;margin:0.5rem 0 0.25rem">Violation evidence</p>',
            unsafe_allow_html=True,
        )
        st.caption("Auto snapshot when a rule fires — **once per incident** (padded crop of the subject).")
        if viol_items:
            show_v = list(reversed(viol_items))[:max_each]
            cols = st.columns(len(show_v))
            for i, it in enumerate(show_v):
                with cols[i]:
                    short = (it.get("message") or "?")[:20]
                    st.caption(f"Frame **{it.get('frame', '?')}** · {short}")
                    try:
                        st.image(it["thumb_rgb"], use_container_width=True)
                    except TypeError:
                        st.image(it["thumb_rgb"], width=viol_thumb_max)
        else:
            st.caption("_No violation crops yet._")
    with plate_holder.container():
        st.markdown(
            '<p style="font-size:0.85rem;font-weight:600;color:#38bdf8;margin:0.5rem 0 0.25rem">Number plates</p>',
            unsafe_allow_html=True,
        )
        st.caption("Finalized EasyOCR reads — newest first.")
        if plate_items:
            show_p = list(reversed(plate_items))[:max_each]
            cols2 = st.columns(len(show_p))
            for i, it in enumerate(show_p):
                with cols2[i]:
                    cap = (it.get("text") or "")[:14] or "—"
                    try:
                        st.image(it["thumb_rgb"], caption=cap, use_container_width=True)
                    except TypeError:
                        st.image(it["thumb_rgb"], caption=cap, width=plate_thumb_max)
        else:
            st.caption("_No plate crops yet._")


def render_sidebar_catalog() -> None:
    st.markdown("### Catalog")
    pe = max(1, int(getattr(config, "PLATE_YOLO_EVERY_N_FRAMES", 1)))
    bx = float(getattr(config, "PLATE_BBOX_EXPAND_FRAC", 0.0) or 0.0)
    inner = bool(getattr(config, "PLATE_DRAW_INNER_YOLO_BOX", True))
    st.caption(
        f"Plate weights: `{Path(config.PLATE_MODEL_PATH).name}` · "
        f"Truck-scoped plate YOLO: **{'on' if config.TRUCK_SCOPED_PLATE_ONLY else 'off'}** · "
        f"plate YOLO every **{pe}** frame(s) · bbox expand **{bx:.0%}** · inner YOLO box: **{'on' if inner else 'off'}** · "
        f"OCR upgrade: **{getattr(config, 'PLATE_OCR_ALLOW_UPGRADE', False)}**"
    )
    with st.expander("Model files", expanded=False):
        for entry in config.MODEL_CATALOG:
            p = config.catalog_model_paths()[entry["id"]]
            ok = "Ready" if config.is_model_file_usable(p) else "Missing"
            st.markdown(f"**{entry['title']}** — {ok}")
            st.caption(entry["summary"])


def render_video_tab(
    labels: List[str],
    default_labels: List[str],
    label_to_id: Dict[str, str],
) -> None:
    _init_plate_capture_session_state()
    preview_max = int(getattr(config, "DASHBOARD_PREVIEW_MAX_WIDTH", 1024))
    viol_thumb_max = int(getattr(config, "VIOLATION_SNAPSHOT_THUMB_MAX_WIDTH", 140))
    plate_strip_w = min(120, int(getattr(config, "DASHBOARD_SIDEBAR_PLATE_THUMB", 132)))
    ui_cfg = max(1, int(getattr(config, "DASHBOARD_UI_UPDATE_EVERY_N", 2)))

    header_slot = st.empty()

    col_left, col_right = st.columns([2, 3])

    with col_left:
        with st.container(border=True):
            st.markdown(viola_upload_shell_html(), unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Video or photo",
                type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                key="vid_up",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                uid = f"{uploaded.name}:{uploaded.size}"
                if st.session_state.get("vl_upload_uid") != uid:
                    st.session_state.vl_upload_uid = uid
                    st.session_state.viola_ui_status = "ready"
                    st.session_state.vl_toast_u = None
                    st.session_state.pop("viola_snapshot", None)
                    st.session_state.vl_run_events = []
                    st.session_state.vl_violation_captures = []
                    st.session_state.vl_plate_captures = []
                st.caption(f"**{html.escape(uploaded.name)}** · ready to analyze")
                if st.session_state.get("vl_toast_u") != uid:
                    st.session_state.vl_toast_u = uid
                    if hasattr(st, "toast"):
                        st.toast(f"Uploaded {uploaded.name}", icon="✅")

            _ensure_model_toggle_defaults(labels, label_to_id, default_labels)
            st.markdown("**Models**")
            st.caption("Click each toggle to activate or deactivate that detector for the next run.")
            for lab in labels:
                mid = label_to_id[lab]
                k = _vl_model_key(mid)
                if mid == "plate":
                    disp = f"{lab} · OCR"
                    h = "YOLO plate weights + EasyOCR on crops."
                elif mid == "triple":
                    disp = lab
                    h = "Triple riding — flags 3+ person boxes on one vehicle."
                else:
                    disp = lab
                    h = f"Use `{mid}` weights in the pipeline."
                st.toggle(disp, key=k, help=h)

            truck_s = config.TRUCK_VIOLATIONS_ACTIVE_START_HOUR
            truck_e = config.TRUCK_VIOLATIONS_ACTIVE_END_HOUR
            selected = _selected_labels_from_toggles(labels, label_to_id)
            if selected and any(label_to_id.get(x) == "truck" for x in selected):
                st.caption(
                    "**Truck rules clock** (half-open `[start, end)`) · same window for no-parking, signal line, and restricted-truck alerts."
                )
                tz_note = getattr(config, "TRUCK_RULES_TIMEZONE", None)
                if tz_note:
                    st.caption(f"Timezone: **{tz_note}** (`config.TRUCK_RULES_TIMEZONE`).")
                else:
                    st.caption(
                        "Using **system local** time. Set `TRUCK_RULES_TIMEZONE` in `config.py` if hours look wrong."
                    )
                st.caption("Truck window · start – end (exclusive)")
                tc1, tc2 = st.columns(2)
                with tc1:
                    truck_s = st.number_input("Start hour", 0, 23, config.TRUCK_VIOLATIONS_ACTIVE_START_HOUR, key="v_tr_s")
                with tc2:
                    truck_e = st.number_input("End hour (exclusive)", 1, 24, config.TRUCK_VIOLATIONS_ACTIVE_END_HOUR, key="v_tr_e")

            pacing = st.selectbox(
                "Playback pacing",
                options=["Real-time (match video clock)", "Fast (no sleep)"],
                index=0 if config.VIDEO_REALTIME_PACING_DEFAULT else 1,
                help="Real-time sleeps between frames so progress follows wall-clock video time. Fast runs as quickly as CPU allows.",
            )
            realtime = pacing.startswith("Real-time")

            ui_refresh = st.selectbox(
                "Side panel refresh",
                options=[
                    f"Every frame (heavier UI)",
                    f"Every {ui_cfg} frame(s) (recommended)",
                    "Every 3 frames",
                    "Every 5 frames",
                ],
                index=1,
                help="How often stats, plate list, event log, and evidence thumbnails refresh. The main video preview updates every frame.",
            )
            if "Every frame" in ui_refresh:
                ui_every_val = 1
            elif "Every 3" in ui_refresh:
                ui_every_val = 3
            elif "Every 5" in ui_refresh:
                ui_every_val = 5
            else:
                ui_every_val = ui_cfg

            export_mp4 = st.checkbox(
                "Export annotated MP4",
                value=False,
                help="Downloads an annotated clip after the run. For a single photo, this is a one-frame MP4.",
            )

            st.caption(
                "📥 **After a run**, use **Download** for the annotated clip (if enabled) or right-click thumbnails to save images."
            )

            run = st.button(
                "▶  Run analysis",
                type="primary",
                key="run_vid",
                use_container_width=True,
                disabled=(uploaded is None or not selected),
            )

        status_slot = st.empty()
        mode_slot = st.empty()
        viol_slot = st.empty()
        plate_slot = st.empty()
        viol_strip_holder = st.empty()
        plate_strip_holder = st.empty()
        dl_holder = st.empty()

    with col_right:
        mini_preview = st.empty()
        stats_panel = st.empty()
        stats_panel.markdown(
            viola_stats_grid_html(0, 0, "0", 0, 0, fps_line="—"),
            unsafe_allow_html=True,
        )
        prog_ph = st.empty()
        prog_ph.progress(0, text="—")
        st.markdown(viola_zones_html(), unsafe_allow_html=True)
        events_panel = st.empty()
        events_panel.markdown(viola_events_html([]), unsafe_allow_html=True)

    if run:
        hdr_status: str = "processing"
        hdr_live = True
    else:
        hdr_status = str(st.session_state.viola_ui_status)
        hdr_live = False
    header_slot.markdown(
        viola_header_html(status=hdr_status, show_live=hdr_live),
        unsafe_allow_html=True,
    )

    frame_slot = st.empty()

    if not labels:
        frame_slot.markdown(_viola_placeholder_html(), unsafe_allow_html=True)
        st.error("No usable `.pt` files found. Add weights under `models/`.")
        return
    if uploaded is None:
        frame_slot.markdown(_viola_placeholder_html(), unsafe_allow_html=True)
        st.info("Upload a **video** or **photo** (JPG, PNG, …), then click **Run analysis**.")
        return
    if not selected:
        _show_upload_idle_preview(mini_preview, uploaded)
        frame_slot.markdown(_viola_placeholder_html(), unsafe_allow_html=True)
        st.warning(
            "Turn on at least one **model** above (Truck, Triple, or Number plate). "
            "Otherwise **Run analysis** stays disabled."
        )
        return
    if not run:
        if not _restore_viola_snapshot(frame_slot, mini_preview, stats_panel, prog_ph):
            _show_upload_idle_preview(mini_preview, uploaded)
            frame_slot.markdown(_viola_placeholder_html(), unsafe_allow_html=True)
        events_panel.markdown(
            viola_events_html(list(st.session_state.get("vl_run_events", []))),
            unsafe_allow_html=True,
        )
        _refresh_evidence_strips(
            viol_strip_holder,
            plate_strip_holder,
            list(st.session_state.get("vl_violation_captures", [])),
            list(st.session_state.get("vl_plate_captures", [])),
            viol_thumb_max=viol_thumb_max,
            plate_thumb_max=plate_strip_w,
        )
        return

    paths = paths_from_labels(selected, label_to_id)
    media_path = write_upload_to_temp(uploaded)
    is_image = _is_upload_image(uploaded.name)
    out_mp4_path: Optional[Path] = None
    if export_mp4:
        fd, out_mp4_path_str = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        out_mp4_path = Path(out_mp4_path_str)

    try:
        with st.spinner("Loading YOLO…"):
            pipeline = TrafficPipeline(
                model_paths=paths,
                truck_violation_active_start_hour=int(truck_s),
                truck_violation_active_end_hour=int(truck_e),
            )
    except Exception as e:
        try:
            os.unlink(media_path)
        except OSError:
            pass
        if out_mp4_path and out_mp4_path.exists():
            try:
                os.unlink(out_mp4_path)
            except OSError:
                pass
        st.session_state.viola_ui_status = "ready"
        if hasattr(st, "toast"):
            st.toast("Pipeline failed to load", icon="❌")
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
        return

    status_slot.success(f"Active: **{', '.join(sorted(pipeline.active_models))}**")

    if pipeline.use_plate:
        with st.spinner("Loading EasyOCR (first time can take a while on CPU)…"):
            try:
                pipeline._get_ocr_reader()
            except Exception as e:
                st.warning(f"EasyOCR preload: {e}")

    st.session_state.vl_plate_captures = []
    st.session_state.vl_violation_captures = []
    st.session_state.vl_run_events = []
    st.session_state.vl_prev_cum_viol = 0
    ui_every = ui_every_val
    gc_every = int(getattr(config, "DASHBOARD_GC_EVERY_N_FRAMES", 0))

    done_ev: Optional[Dict[str, Any]] = None
    frame_idx = 0
    est_total = 1
    dec_skip = 1
    fps = 25.0

    decode_iter = (
        iter_decode_image(
            Path(media_path),
            out_mp4_path if export_mp4 else None,
            pipeline,
            write_annotated_mp4=bool(export_mp4),
        )
        if is_image
        else iter_decode_video(
            Path(media_path),
            out_mp4_path if export_mp4 else None,
            pipeline,
            write_annotated_mp4=bool(export_mp4),
        )
    )

    try:
        for ev in decode_iter:
            if ev["kind"] == "done":
                done_ev = ev
                break

            t0 = time.perf_counter()
            frame_idx = int(ev["frame_idx"])
            est_total = max(1, int(ev["frame_total_est"]))
            dec_skip = int(ev["dec_skip"])
            fps = float(ev["fps"])
            frame_interval = dec_skip / max(fps, 1.0)
            violations = ev["violations"]
            meta = ev["meta"]
            processed = ev["processed"]
            cum_viol = int(ev["cum_viol"])
            prev_v = int(st.session_state.vl_prev_cum_viol)
            pulse = cum_viol > prev_v
            st.session_state.vl_prev_cum_viol = cum_viol

            for c in ev["new_captures"]:
                st.session_state.vl_plate_captures.append(dict(c))
                t_sec = round((frame_idx * dec_skip) / max(fps, 1e-6), 1)
                line = (
                    f"{t_sec}s · V{int(c['tid'])} · plate {str(c.get('text') or '')} "
                    f"· OCR {float(c.get('ocr', 0)):.2f}"
                )
                st.session_state.vl_run_events.insert(0, line)
                while len(st.session_state.vl_run_events) > 50:
                    st.session_state.vl_run_events.pop()

            for snap in meta.get("violation_snapshots") or []:
                rgb_snap = snap.get("thumb_rgb")
                if rgb_snap is None:
                    continue
                arr = np.asarray(rgb_snap)
                if arr.size == 0:
                    continue
                small_v = resize_preview_rgb(arr, viol_thumb_max)
                st.session_state.vl_violation_captures.append(
                    {
                        "message": str(snap.get("message") or ""),
                        "thumb_rgb": small_v,
                        "frame": frame_idx,
                    }
                )
            while len(st.session_state.vl_violation_captures) > 48:
                st.session_state.vl_violation_captures.pop(0)

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            uv = list(dict.fromkeys(violations))
            if uv:
                ev_line = f"f{frame_idx} · " + " · ".join(v for v in uv[:8])
                st.session_state.vl_run_events.insert(0, ev_line)
                while len(st.session_state.vl_run_events) > 50:
                    st.session_state.vl_run_events.pop()

            # Always refresh the main preview + progress every processed frame. Previously the whole
            # block was gated by "Browser update frequency", so the video appeared to load only every N frames.
            small_bgr = cv2.cvtColor(resize_preview_rgb(rgb, preview_max), cv2.COLOR_RGB2BGR)
            uri = _bgr_jpeg_data_uri(small_bgr, 78)
            if uri:
                mini_preview.markdown(
                    f'<div class="viola-mini-preview"><img src="{uri}" alt="current frame"/></div>',
                    unsafe_allow_html=True,
                )
                frame_slot.markdown(
                    _viola_frame_html(
                        uri,
                        frame_idx,
                        est_total,
                        fps,
                        3,
                        cum_viol,
                        ev["unique_plate_tracks"],
                    ),
                    unsafe_allow_html=True,
                )
            prog_ph.progress(
                min(frame_idx / est_total, 1.0),
                text=(
                    "Photo · 1 frame"
                    if is_image
                    else f"Frame {frame_idx} / ~{est_total} (decode every {dec_skip})"
                ),
            )

            do_heavy_ui = ui_every <= 1 or (frame_idx - 1) % ui_every == 0
            if do_heavy_ui:
                fv = str(len(uv)) if uv else "0"
                shake = len(uv) > 0
                fps_line = f"{fps:.1f} · decode every {dec_skip} frame(s)"
                stats_panel.markdown(
                    viola_stats_grid_html(
                        cum_viol,
                        ev["plates_locked_count"],
                        fv,
                        frame_idx,
                        est_total,
                        pulse_viol=pulse,
                        shake_viol=shake,
                        fps_line=fps_line,
                    ),
                    unsafe_allow_html=True,
                )

                mim = meta.get("plate_infer_mode", "—")
                nbox = meta.get("plate_yolo_boxes", 0)
                rules_h = meta.get("truck_rules_clock_hour")
                tz = meta.get("truck_rules_tz")
                extra = ""
                if rules_h is not None and pipeline.use_truck:
                    tzs = f" ({tz})" if tz else ""
                    extra = f" · rule clock hour: **{rules_h}**{tzs}"
                mode_slot.caption(f"Plate infer: **{mim}** · boxes: **{nbox}**{extra}")

                if uv:
                    viol_slot.error("**Violations (this frame)**\n\n" + "\n".join(f"• {v}" for v in uv[:10]))
                else:
                    viol_slot.success("No active violations on this frame.")

                plates = meta.get("plates") or []
                if plates:
                    plines = []
                    for p in plates[:12]:
                        tid = p.get("track_id", "?")
                        if p.get("ocr_error") and not p.get("text"):
                            plines.append(f"- #{tid}: OCR failed")
                            continue
                        if p.get("pending"):
                            plines.append(
                                f"- #{tid}: stabilizing… (sharp {p.get('sharpness', 0):.0f}, "
                                f"stable {p.get('stable_frames', 0)})"
                            )
                            continue
                        t = html.escape(str(p.get("text") or "—"))
                        cf = float(p.get("confidence") or 0.0)
                        tags = []
                        if p.get("near_truck"):
                            tags.append("on/near truck")
                        suf = f" _{', '.join(tags)}_" if tags else ""
                        plines.append(f"- **#{tid}** `{t}` · OCR {cf:.2f}{suf}")
                    plate_slot.markdown("**Plate reads (EasyOCR on YOLO crops)**\n\n" + "\n".join(plines))
                elif pipeline.use_plate:
                    plate_slot.caption("No plate boxes this frame.")
                else:
                    plate_slot.caption("Enable **Number plate** toggle for OCR.")

                events_panel.markdown(
                    viola_events_html(list(st.session_state.vl_run_events)),
                    unsafe_allow_html=True,
                )
                _refresh_evidence_strips(
                    viol_strip_holder,
                    plate_strip_holder,
                    st.session_state.vl_violation_captures,
                    st.session_state.vl_plate_captures,
                    viol_thumb_max=viol_thumb_max,
                    plate_thumb_max=plate_strip_w,
                )

            if gc_every > 0 and frame_idx % gc_every == 0:
                gc.collect()

            if realtime and not is_image:
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, frame_interval - elapsed))
    except Exception as e:
        try:
            os.unlink(media_path)
        except OSError:
            pass
        if out_mp4_path and out_mp4_path.exists():
            try:
                os.unlink(out_mp4_path)
            except OSError:
                pass
        st.session_state.viola_ui_status = "ready"
        if hasattr(st, "toast"):
            st.toast("Processing stopped", icon="❌")
        st.error("Processing stopped:")
        st.exception(e)
        return

    try:
        os.unlink(media_path)
    except OSError:
        pass

    if not done_ev:
        st.error("Processing produced no frames.")
        st.session_state.viola_ui_status = "ready"
        if out_mp4_path and out_mp4_path.exists():
            try:
                os.unlink(out_mp4_path)
            except OSError:
                pass
        return

    st.session_state.viola_ui_status = "complete"
    st.session_state.vl_plate_captures = list(done_ev["captures"])
    stats_panel.markdown(
        viola_stats_grid_html(
            int(done_ev["cum_viol"]),
            len(done_ev["captures"]),
            "0",
            int(done_ev["frame_idx"]),
            int(done_ev["est_decoded"]),
            fps_line=f"{float(done_ev['fps']):.1f} · done",
        ),
        unsafe_allow_html=True,
    )
    if done_ev.get("last_bgr") is not None:
        last_rgb = cv2.cvtColor(done_ev["last_bgr"], cv2.COLOR_BGR2RGB)
        last_small = cv2.cvtColor(resize_preview_rgb(last_rgb, preview_max), cv2.COLOR_RGB2BGR)
        u2 = _bgr_jpeg_data_uri(last_small, 82)
        if u2:
            mini_preview.markdown(
                f'<div class="viola-mini-preview"><img src="{u2}" alt="last frame"/></div>',
                unsafe_allow_html=True,
            )
            frame_slot.markdown(
                _viola_frame_html(
                    u2,
                    int(done_ev["frame_idx"]),
                    int(done_ev["est_decoded"]),
                    float(done_ev["fps"]),
                    3,
                    int(done_ev["cum_viol"]),
                    len({int(c["tid"]) for c in done_ev["captures"]}),
                ),
                unsafe_allow_html=True,
            )
            _persist_viola_snapshot(
                main_uri=u2,
                done_ev=done_ev,
                prog_text="Photo · done" if is_image else "Complete",
            )

    prog_ph.progress(1.0, text="Photo · done" if is_image else "Complete")
    status_slot.caption(
        f"**Done** · frames **{done_ev['frame_idx']}** · violations **{done_ev['cum_viol']}** "
        f"· plate reads **{len(done_ev['captures'])}**"
    )
    events_panel.markdown(viola_events_html(list(st.session_state.vl_run_events)), unsafe_allow_html=True)
    _refresh_evidence_strips(
        viol_strip_holder,
        plate_strip_holder,
        st.session_state.vl_violation_captures,
        st.session_state.vl_plate_captures,
        viol_thumb_max=viol_thumb_max,
        plate_thumb_max=plate_strip_w,
    )

    if not is_image and frame_idx > 0 and frame_idx < max(5, int(est_total * 0.35)):
        st.warning(
            "Few frames decoded — re-encode to **H.264** in `.mp4` or lower decode stride in `config.py`."
        )

    if hasattr(st, "toast"):
        st.toast("Analysis complete", icon="✅")

    with dl_holder.container():
        if export_mp4 and out_mp4_path and out_mp4_path.exists():
            try:
                mp4_bytes = out_mp4_path.read_bytes()
                st.download_button(
                    label="⬇ Download annotated MP4",
                    data=mp4_bytes,
                    file_name="violane_annotated.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="dl_mp4_viola",
                )
            finally:
                try:
                    os.unlink(out_mp4_path)
                except OSError:
                    pass
        else:
            st.success(
                "**Thumbnails** — right-click to save. "
                "For **photos**, enable **Export annotated MP4** to download a one-frame clip, or save the preview above."
            )


__all__ = [
    "model_options",
    "render_sidebar_catalog",
    "render_video_tab",
]
