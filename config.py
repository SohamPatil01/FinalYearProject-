"""Central configuration for the traffic violation system."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Truck YOLO weights (`models/truck.pt`).
TRUCK_MODEL_PATH = str(MODELS_DIR / "truck.pt")
# Helmet / rider checkpoint. ``helmet_best.pt`` is a 2-class (With/Without Helmet) model
# trained on close-up rider crops; see HELMET_CROP_DETECTION below for how it is fed.
HELMET_MODEL_PATH = str(MODELS_DIR / "helmet_best.pt")

# Video source:
# - 0 for webcam
# - "path/to/video.mp4" for a video file
VIDEO_SOURCE = 0

# Model paths (default for main.py / full pipeline)
MODEL_PATHS = {
    "truck": TRUCK_MODEL_PATH,
    "triple": str(MODELS_DIR / "triple.pt"),
    "helmet": HELMET_MODEL_PATH,
}

# Number-plate detector — **only** this checkpoint is used to localize plates for OCR.
# Truck / triple / helmet models may expose plate-like class names; those boxes are dropped when plate is on.
PLATE_MODEL_PATH = str(MODELS_DIR / "plate_best.pt")
RED_LIGHT_MODEL_PATH = str(MODELS_DIR / "yolov10s.pt")
NO_PARKING_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
RED_LIGHT_CONFIG_DIR = BASE_DIR / "config" / "red_light"
ROI_SESSIONS_DIR = BASE_DIR / "config" / "roi" / "sessions"

# When both **truck** and **plate** are active:
# False (default): plate YOLO runs on full frame (recommended).
# True: plate YOLO can focus on truck ROIs when trucks are present.
TRUCK_SCOPED_PLATE_ONLY: bool = False
PLATE_DETECT_INSIDE_TRUCK_ROI: bool = True  # legacy; keep in sync with TRUCK_SCOPED_PLATE_ONLY
TRUCK_PLATE_ROI_PAD_FRAC: float = 0.18
TRUCK_ROI_PLATE_PAD_FRAC: float = 0.18  # legacy; keep in sync with TRUCK_PLATE_ROI_PAD_FRAC
# When TRUCK_SCOPED_PLATE_ONLY is True and trucks are present, still run full-frame plate YOLO and
# merge with ROI detections so plates outside truck ROIs are not missed.
PLATE_INCLUDE_FULL_FRAME_WITH_TRUCK_ROI: bool = True
# Ignore truck detections below this confidence when building ROIs for plate (0 = no filter).
TRUCK_PLATE_MIN_TRUCK_CONF: float = 0.0
# Scoped plate YOLO: search only the lower part of each expanded truck ROI (bumper / plate region).
# Matches the reference `cut_y = y1 + int(h * 0.5)` on the truck box. Set False to search the full ROI.
TRUCK_PLATE_ROI_BOTTOM_HALF_ONLY: bool = True
# Start of plate-search strip as a fraction from the top of the expanded truck ROI (0.5 = bottom half).
TRUCK_PLATE_ROI_VERTICAL_START_FRAC: float = 0.50
# Multi-class truck checkpoints often label cars / autos / bikes as separate classes. If empty,
# class IDs are inferred from `model.names` when TRUCK_AUTO_CLASS_FILTER is True.
TRUCK_CLASS_IDS: List[int] = []
# Infer truck vs non-truck classes from YOLO names (e.g. keep "truck", drop "car", "auto").
TRUCK_AUTO_CLASS_FILTER: bool = True
# Drop truck-head boxes below this confidence (0 = no filter). Raise to ~0.35–0.5 to cut soft false positives.
TRUCK_YOLO_MIN_CONF: float = 0.0
# Merge duplicate plate boxes from overlapping truck ROIs (IoU ≥ this → keep higher conf).
PLATE_ROI_MERGE_IOU: float = 0.45
PLATE_MODEL_KEY = "plate"

# --- Per-frame pipeline (same thread as the UI / video loop; nothing runs “OCR-only” in the background) ---
# 1) YOLO: truck / triple / helmet / … via ``MultiModelDetector.infer`` on the full frame (each enabled head).
# 2) YOLO: number-plate detector ``plate.pt`` via ``infer_plate`` (full frame or truck ROI per config).
# 3) Rules: violations from detections (triple, helmet, truck restricted-time); no polygon zones.
# 4) OCR: **only** on small BGR crops cut from the frame using each plate box (``_safe_crop`` → ``read_plate_from_crop``).
#    The full frame is never passed to OCR.

# Plate text recognition engine for **plate crops only**.
# - "easyocr": EasyOCR (default)
# - "paddle": PaddleOCR (optional)
PLATE_OCR_ENGINE: str = "easyocr"
# If PaddleOCR is unavailable / fails, optionally fall back to EasyOCR.
PLATE_OCR_FALLBACK_TO_EASYOCR: bool = True
# PaddleOCR options
PADDLEOCR_LANG: str = "en"
PADDLEOCR_USE_GPU: bool = False
PADDLEOCR_USE_ANGLE_CLS: bool = True

# EasyOCR languages (e.g. add "hi" for Devanagari if your plates need it)
EASYOCR_LANGS: List[str] = ["en"]
# Use GPU for EasyOCR when CUDA is available (much faster than CPU OCR)
EASYOCR_USE_GPU: bool = True

# YOLO speed (lower imgsz = faster; 416 is a good balance for real-time on GPU)
YOLO_IMGSZ: int = 416
# Smaller size for the plate model only (plates are small; 320 cuts cost vs full YOLO_IMGSZ)
YOLO_PLATE_IMGSZ: int = 320
# FP16 on CUDA only; auto-fallback on CPU in detectors.py
YOLO_HALF_PRECISION: bool = True
# Run plate **detector** YOLO every N processed frames (1 = every frame). Cached boxes between runs still get OCR on crops when gates pass.
PLATE_YOLO_EVERY_N_FRAMES: int = 2 or 3
# Drop dedicated plate-head boxes below this conf before tracking/OCR (reduces flicker and bad crops).
# Raised to reject the weak detector's false-positive boxes ("anything looks like a plate").
PLATE_YOLO_MIN_CONF: float = 0.5
# Expand each raw YOLO plate box by this fraction of its width/height before OCR (captures plate edges).
PLATE_BBOX_EXPAND_FRAC: float = 0.12
# When expansion > 0, draw the raw YOLO box (thin) inside the expanded OCR box (thick) on the video.
PLATE_DRAW_INNER_YOLO_BOX: bool = True

# --- Plate OCR gating (EasyOCR: **only** on plate-image crops; runs when crop quality + stability gates pass) ---
# If True (recommended): ``PlateOCRGate`` tracks plate boxes over time, waits for stable/sharp crops, then OCR.
# If False: no tracker — OCR runs on plate YOLO boxes directly (``ocr_plate_detections_one_shot``), throttled by
# ``PLATE_OCR_ATTEMPT_EVERY_N_FRAMES``. Simpler but heavier CPU, jittery track IDs, and no stability voting.
PLATE_USE_TRACK_OCR_GATE: bool = True
PLATE_TRACK_MAX_DISTANCE: int = 95
PLATE_TRACK_MAX_DISAPPEARED: int = 22
# EMA on plate box per track (0 = off). Smooths jitter between YOLO refreshes for steadier crops and reads.
PLATE_BBOX_SMOOTH_ALPHA: float = 0.42
PLATE_OCR_MIN_YOLO_CONF: float = 0.6
PLATE_OCR_MIN_AREA: int = 2000
# width/height; include portrait-ish / square plate boxes (truck fronts, angles).
PLATE_OCR_MIN_ASPECT: float = 0.35
PLATE_OCR_MAX_ASPECT: float = 8.0
# Min IoU between tracked box and current YOLO box to reuse that det's score (else use last_yolo_conf).
PLATE_OCR_DET_IOU_MIN: float = 0.08
PLATE_OCR_MIN_SHARPNESS: float = 35.0
# If aspect/area fails but Laplacian sharpness reaches this, still allow OCR (stable + YOLO ok).
PLATE_OCR_BYPASS_GEOM_SHARPNESS: float = 120.0
# Require this many stable frames before OCR — plate YOLO still runs every frame; EasyOCR waits until the box settles.
PLATE_OCR_STABLE_FRAMES: int = 3
PLATE_OCR_MAX_CENTROID_DRIFT: float = 20.0
PLATE_OCR_MIN_TEXT_LEN: int = 4
# Confirm a plate read by agreement across this many OCR reads (confidence-weighted vote)
# before locking it — corrects single-frame OCR mistakes. Set to 1 to lock on the first read.
PLATE_OCR_CONFIRM_READS: int = 2
# A single read at/above this OCR confidence locks immediately (skips the confirm vote),
# so clearly-readable plates still lock fast.
PLATE_OCR_SINGLE_LOCK_CONF: float = 0.80
# Ignore OCR reads below this confidence — stops junk crops from locking garbage text.
PLATE_OCR_MIN_ACCEPT_CONF: float = 0.35
# When False, EasyOCR runs at most once per track after gates pass (best for video throughput + stable read).
PLATE_OCR_ALLOW_UPGRADE: bool = False
PLATE_OCR_UPGRADE_QUALITY_RATIO: float = 1.25
# Minimum frames between OCR attempts while plate still has no valid read (higher = less CPU churn).
# Low enough to collect a couple of confirm-vote reads, high enough to avoid CPU stalls.
PLATE_OCR_RETRY_MIN_FRAMES: int = 10
# Stop retrying a no-text plate track after this many failed OCR attempts.
# (Successful OCR resets this naturally.)
PLATE_OCR_MAX_TRIES_PER_TRACK: int = 4
# If True, each track gets only one OCR attempt (no continuous background retries).
# False enables the multi-read confirm vote (see PLATE_OCR_CONFIRM_READS) for better accuracy.
PLATE_OCR_ONE_SHOT_PER_TRACK: bool = False
# Limit OCR calls per processed frame to avoid random multi-second stalls when multiple tracks gate-pass together.
# 1 = smoothest timeline. Increase to 2 only if you need faster lock-on for many simultaneous plates.
PLATE_OCR_MAX_TRIES_PER_FRAME: int = 1
# Optional throttle: only *attempt* OCR on every Nth pipeline frame (still **plate crops only**; plate YOLO runs every frame).
# Higher = less EasyOCR load and less UI lag. First stable read can bypass stride (see PLATE_OCR_SKIP_STRIDE_ON_STABLE_EDGE).
PLATE_OCR_ATTEMPT_EVERY_N_FRAMES: int = 6
# If True, the first OCR try for a track fires on the first frame where quality+stability gates pass, without waiting for the stride counter.
PLATE_OCR_SKIP_STRIDE_ON_STABLE_EDGE: bool = True
# Plate text stabilization window size (majority vote over last N successful OCR reads per track).
PLATE_TEXT_STABILIZE_WINDOW: int = 7
# Association thresholds for plate <-> vehicle matching.
PLATE_VEHICLE_ASSOC_IOU_MIN: float = 0.08
PLATE_VEHICLE_ASSOC_MAX_CENTROID_DIST: float = 140.0
# Association thresholds for helmet(no-helmet) <-> rider matching.
HELMET_RIDER_ASSOC_IOU_MIN: float = 0.08
HELMET_RIDER_ASSOC_MAX_CENTROID_DIST: float = 120.0
# Downscale plate crop so EasyOCR does not block for seconds on huge regions (inference unchanged).
PLATE_OCR_MAX_CROP_SIDE: int = 520
# Min shorter side after gray preprocess upscaling (helps small plates in truck ROI).
PLATE_OCR_PREPROCESS_MIN_SIDE: int = 140
# Extra border around YOLO crop before OCR (fraction of min side) — reduces clipped characters.
PLATE_OCR_INNER_PAD_FRAC: float = 0.10
# Multiple preprocess views per OCR attempt — accurate but slower; False cuts lag a lot.
# Off by default: the cross-frame confirm-vote already improves accuracy without the 3x CPU cost
# (this was the main cause of the mid-run video stalls).
PLATE_OCR_MULTI_VARIANT: bool = False
PLATE_OCR_MAX_VARIANTS: int = 2
# Recognition: beam search is slower but usually more accurate on plate strings than greedy.
PLATE_OCR_DECODER: str = "beamsearch"  # "greedy" | "beamsearch"
PLATE_OCR_BEAM_WIDTH: int = 8
# Magnify small plate crops for CRAFT + recognizer (helps tiny boxes).
PLATE_OCR_MAG_RATIO: float = 1.9
# When True, OCR uses EasyOCR **recognize** on the whole plate crop only (one line, no CRAFT text-detection pass
# inside the crop). When False, **readtext** runs detection + recognition on the crop (can pick extra regions).
PLATE_OCR_RECOGNIZER_ONLY: bool = True
# If recognizer-only yields no text, optionally fall back to readtext (slower; helps hard crops).
PLATE_OCR_RECOGNIZER_FALLBACK_READTEXT: bool = True
# Detection thresholds on the crop (used when PLATE_OCR_RECOGNIZER_ONLY is False or for fallback readtext).
PLATE_OCR_TEXT_THRESHOLD: float = 0.52
PLATE_OCR_LOW_TEXT: float = 0.32
# Try 90° rotations on the crop (slower; helps vertical / skewed plates).
PLATE_OCR_ROTATION_TRIALS: bool = False

# --- Indian / yellow-black plates (better contrast + spacing for e.g. CG04 JD 7398) ---
PLATE_OCR_INDIAN_STYLE: bool = True
# Extra padding when taking plate crops from the frame (reduces clipped first/last character).
PLATE_OCR_SAFE_CROP_PAD_FRAC: float = 0.14
# Minimum width (px) after preprocess upscaling — larger helps blurry phone crops.
PLATE_OCR_MIN_WIDTH_PX: int = 320
# Aggressive upscale before EasyOCR (similar to 3× in reference scripts).
PLATE_OCR_UPSCALE_MAX_FACTOR: float = 3.0

# Evidence crops when a violation fires (first frame per incident only; see TrafficPipeline).
VIOLATION_SNAPSHOT_PAD_FRAC: float = 0.12
VIOLATION_SNAPSHOT_THUMB_MAX_WIDTH: int = 140


class ModelCatalogEntry(TypedDict):
    id: str
    file: str
    title: str
    summary: str
    description: str


# UI (dashboard.py) / catalog: models you ship.
MODEL_CATALOG: List[ModelCatalogEntry] = [
    {
        "id": "truck",
        "file": "truck.pt",
        "title": "Truck",
        "summary": "Detects trucks / heavy vehicles; restricted-hours rule when the clock is in the active window.",
        "description": "When enabled: **Truck in restricted hours** uses the configured time window (and optional separate `TRUCK_RESTRICTED_*` hours if `TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW` is off). **Plates** use **Number plate** (`plate.pt`) + EasyOCR on crops, optionally ROI-prioritized around trucks.",
    },
    {
        "id": "triple",
        "file": "triple.pt",
        "title": "Triple seat",
        "summary": "Triple-seat violations: assigns **person** detections to **vehicles** (bike, rickshaw, …) and fires when count ≥ `TRIPLE_MIN_PERSONS_ON_MOTORCYCLE` (configurable).",
        "description": "Turn this on for **triple-seat** rules. The checkpoint is usually multi-class (vehicle + person; some also include a plate label). Plate **text** still comes only from **Number plate** (`plate.pt` + OCR) — any plate-like class here is ignored for OCR when Number plate is enabled, so counts and overlays stay correct.",
    },
    {
        "id": "helmet",
        "file": "helmet.pt",
        "title": "Helmet",
        "summary": "Detects riders / helmets; flags **no-helmet** classes (e.g. `no_helmet`).",
        "description": "Uses `models/helmet.pt`. Violations fire on detections whose class name matches **without helmet** (see `HELMET_VIOLATION_CLASS_IDS` to override). Number plates still use **Number plate** (`plate.pt`) + OCR.",
    },
    {
        "id": "plate",
        "file": "plate_best.pt",
        "title": "Number plate",
        "summary": "Universal plate detector + OCR on crops.",
        "description": "All plate localization for OCR uses `models/plate_best.pt` only. When off, no plate YOLO or OCR runs.",
    },
]


def catalog_model_paths() -> Dict[str, str]:
    """Map catalog id -> absolute path (plate / truck use resolved paths)."""
    out: Dict[str, str] = {}
    for entry in MODEL_CATALOG:
        mid = entry["id"]
        if mid == PLATE_MODEL_KEY:
            out[mid] = PLATE_MODEL_PATH
        elif mid == "truck":
            out[mid] = TRUCK_MODEL_PATH
        elif mid == "helmet":
            out[mid] = HELMET_MODEL_PATH
        else:
            out[mid] = str(MODELS_DIR / entry["file"])
    return out


def is_model_file_usable(path: str, min_bytes: int = 1024) -> bool:
    """True if a .pt file exists and looks non-empty (filters tiny placeholders)."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        return p.stat().st_size >= min_bytes
    except OSError:
        return False

# Basic visualization options
WINDOW_NAME = "Traffic Violation Detection"
FONT_SCALE = 0.6
THICKNESS = 2

# Triple-riding model. If your `.pt` uses classes like motorcycle + person + numberplate, leave this
# empty: the pipeline auto-detects and only flags when >= TRIPLE_MIN_PERSONS_ON_MOTORCYCLE persons
# align with one motorcycle. If non-empty, forces legacy single-head class filtering instead.
TRIPLE_VIOLATION_CLASS_IDS: List[int] = []
# When True (default), load triple YOLO `.names` and only count classes that look like "triple"
# (and exclude "double"-like names when the model has multiple classes).
TRIPLE_AUTO_CLASS_FILTER: bool = True
# Drop low-confidence triple detections (try 0.4–0.55 if you still get double FPs on a single-class model).
TRIPLE_MIN_CONFIDENCE: float = 0.0
# Overlapping triple boxes (same bike, duplicate heads) merge into one violation above this IoU.
TRIPLE_MERGE_IOU: float = 0.5
# Require this many consecutive frames with a detection in the same screen cell (reduces one-off FPs).
# Set to 1 to flag triple riding as soon as geometry passes (same frame).
TRIPLE_MIN_CONSECUTIVE_FRAMES: int = 1
# Cell size (px) for grouping the same bike across frames for the streak counter.
TRIPLE_STREAK_CELL_PX: int = 72

# --- Triple model with separate classes: vehicle + person (+ plate, etc.) ---
# Violation when at least this many **person** boxes are assigned to the **same** carrier vehicle
# (motorcycle, scooter, rickshaw, etc. — see `violations._carrier_vehicle_class_name`). Default **3** = triple seat.
TRIPLE_MIN_PERSONS_ON_MOTORCYCLE: int = 3
# Grow the motorcycle box by this fraction of its width/height when deciding if a person is on that bike.
TRIPLE_MC_EXPAND_FRAC: float = 0.2
# Also count the person if IoU with the (unexpanded) motorcycle box is at least this much.
TRIPLE_MC_PERSON_IOU_MIN: float = 0.04
# NMS among person boxes on the triple head before counting (reduces duplicate heads).
TRIPLE_PERSON_NMS_IOU: float = 0.42

# Helmet model: if non-empty, these class indices count as violations (overrides name-based auto-detect).
HELMET_VIOLATION_CLASS_IDS: List[int] = []
# Min YOLO confidence for a no-helmet box before it can trigger a violation.
HELMET_MIN_CONFIDENCE: float = 0.35
HELMET_MERGE_IOU: float = 0.45
# Same idea as triple streak: require this many consecutive frames in the same screen cell (1 = immediate).
HELMET_MIN_CONSECUTIVE_FRAMES: int = 1
HELMET_STREAK_CELL_PX: int = 64

# --- Absence-based helmet rule -------------------------------------------------
# Many helmet checkpoints (incl. the bundled yolov8n one) mislabel bare/capped heads
# as "helmet", so the `no_helmet` class rarely fires. When this is on, a rider/
# motorcycle that has NO sufficiently confident "helmet" box over its head region is
# flagged as a no-helmet violation — even if the model never emits a `no_helmet` box.
HELMET_ABSENCE_RULE: bool = True
# A "helmet" detection must be at least this confident to count as "this rider has a helmet".
# Keep relatively high so weak/false helmet boxes on bare heads do not suppress the violation.
HELMET_PRESENT_MIN_CONF: float = 0.55
# A rider/motorcycle box must be at least this confident to be considered a carrier needing a helmet.
HELMET_RIDER_MIN_CONF: float = 0.35
# Fraction of the rider box height (from the top) treated as the head region for helmet coverage.
HELMET_HEAD_REGION_FRAC: float = 0.45

# --- Crop-based helmet detection ----------------------------------------------
# Helmet checkpoints are trained on close-up riders, so running them on a full wide
# frame misses small/distant heads. When on, a carrier detector (COCO) locates
# motorcycles/bicycles, and the helmet model runs on each expanded rider crop (where
# the head is large), then results are mapped back to frame coordinates.
HELMET_CROP_DETECTION: bool = True
HELMET_CARRIER_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
HELMET_CARRIER_CLASS_IDS: List[int] = [1, 3]  # COCO: bicycle, motorcycle
HELMET_CARRIER_MIN_CONF: float = 0.30
HELMET_CARRIER_IMGSZ: int = 960
HELMET_CROP_PAD_FRAC: float = 0.25       # widen carrier box before cropping
HELMET_CROP_TOP_EXTEND_FRAC: float = 0.7  # extend upward to include the rider's head/torso
HELMET_CROP_IMGSZ: int = 640
HELMET_CROP_CONF: float = 0.20  # low: the per-crop run is only a signal source

# Per-rider decision (ensemble of crop variants).
# A rider counts as helmeted only if a "With Helmet" box reaches this confidence;
# otherwise (explicit "Without Helmet", OR no confident helmet at all) it's a violation.
HELMET_CROP_PRESENT_CONF: float = 0.45   # confidently helmeted -> no violation
HELMET_CROP_VIOL_CONF: float = 0.25      # explicit "Without Helmet" signal
HELMET_CROP_ABSENCE: bool = True         # flag riders with no confident helmet
HELMET_CROP_ABSENCE_CONF: float = 0.60   # confidence stamped on absence violations
HELMET_CROP_MAX_RIDERS: int = 10         # cap per frame (cost control)
# Per-rider tracking: each tracked rider is counted at most once.
HELMET_TRACK_MAX_DISAPPEARED: int = 30   # frames a rider can vanish before its id is dropped
HELMET_TRACK_MAX_DISTANCE: int = 120     # px centroid distance to keep the same rider id
HELMET_RIDER_MERGE_IOU: float = 0.4      # merge overlapping rider/bike boxes (one rider = one track)
# Temporal stabilisation (the helmet model is noisy frame-to-frame).
HELMET_STATUS_EMA_ALPHA: float = 0.45    # weight of the newest frame in the running evidence
HELMET_STATUS_FLIP_MARGIN: float = 0.18  # extra evidence needed to flip a settled decision
HELMET_CONFIRM_FRAMES: int = 3           # frames a rider must persist before its violation counts

# Separate “restricted hours” for the truck-only rule (used only if TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW is False).
TRUCK_RESTRICTED_START_HOUR = 7
TRUCK_RESTRICTED_END_HOUR = 10

# When True (default), “Truck in restricted hours” uses the same window as TRUCK_VIOLATIONS_ACTIVE_* / dashboard sliders.
# When False, that message uses TRUCK_RESTRICTED_* above; the active window only gates tracking/plate vs full rule evaluation.
TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW: bool = True

# Optional IANA timezone for rule hours, e.g. "Asia/Kolkata". None = system local time from datetime.now().
TRUCK_RULES_TIMEZONE: Optional[str] = None

# When truck model is on: only between these hours do we evaluate the truck **restricted-hours** violation.
# Outside this window we still detect trucks, track IDs, and read number plates — no truck-rule violations.
# Semantics: active if start <= hour < end (half-open). If start == end, treated as always active.
# Overnight: start > end wraps midnight (e.g. 22–6). End may be 24 for “through end of day”.
TRUCK_VIOLATIONS_ACTIVE_START_HOUR = 6
TRUCK_VIOLATIONS_ACTIVE_END_HOUR = 22

# Tracker config
TRACKER_MAX_DISTANCE = 60
TRACKER_MAX_DISAPPEARED = 20

# Streamlit: Real-time sleeps between frames to mimic video clock. Bogus low FPS in files causes long pauses.
# When DASHBOARD_ANNOTATED_REALTIME_SYNC is True, video runs at source FPS regardless of this default.
VIDEO_REALTIME_PACING_DEFAULT: bool = True

# Used only when VIDEO_TARGET_PROCESS_FPS is 0. Otherwise stride comes from target FPS below.
VIDEO_DECODE_EVERY_N_FRAME: int = 1
# Cap processed frame rate (~30 keeps UI smooth and cuts load vs decoding full 60 FPS sources).
# Computed as ceil(source_fps / VIDEO_TARGET_PROCESS_FPS). Set 0 to use VIDEO_DECODE_EVERY_N_FRAME only.
VIDEO_TARGET_PROCESS_FPS: float = 30.0

# Streamlit video: show **annotated** frames (violations/boxes on-image) paced to wall clock — not a separate
# `st.video` playing ahead of inference. Set False to restore native player + periodic overlay (can desync).
DASHBOARD_ANNOTATED_REALTIME_SYNC: bool = True

# Streamlit: downscale preview so each frame upload stays small (full frame still used for YOLO).
DASHBOARD_PREVIEW_MAX_WIDTH: int = 960
# When not using native `st.video`, JPEG HUD refresh every N frames. Progress bar still every frame.
DASHBOARD_PREVIEW_UPDATE_EVERY_N_FRAMES: int = 2
# With native video: show annotated overlay in the side column every N processed frames (lightweight).
DASHBOARD_ANNOTATED_PREVIEW_EVERY_N_FRAMES: int = 8
# Real-time pacing: if source FPS is below this, assume bad metadata and use DASHBOARD_PACING_ASSUMED_FPS.
DASHBOARD_PACING_MIN_SOURCE_FPS: float = 12.0
DASHBOARD_PACING_ASSUMED_FPS: float = 25.0
# Throttle heavy Streamlit updates (stats, plate list, events, evidence strips).
DASHBOARD_UI_UPDATE_EVERY_N: int = 2
# Periodic gc on long videos (0 = off — gc often causes visible hitches).
DASHBOARD_GC_EVERY_N_FRAMES: int = 0
# Sidebar plate crop thumbnails (max width px).
DASHBOARD_SIDEBAR_PLATE_THUMB: int = 132
# Max plate captures kept per run (oldest dropped).
DASHBOARD_PLATE_GALLERY_MAX_ITEMS: int = 80

# FastAPI live stream: max width (px) for JPEG frames over SSE; 0 = full resolution (heavy).
WEB_STREAM_PREVIEW_MAX_WIDTH: int = 960
# Emit one stream frame every N decoded frames over SSE (ignored when WEB_STREAM_REALTIME_PACE forces every frame).
WEB_STREAM_FRAME_EVERY_N: int = 2
# Sleep per decoded frame so preview tracks ~source FPS (per-frame budget, not cumulative — avoids multi-second catch-up stalls).
WEB_STREAM_REALTIME_PACE: bool = True
# Match Streamlit: cap SSE pacing so preview doesn’t demand >30 FPS wall time.
WEB_PACE_MAX_FPS: float = 30.0

# Min YOLO conf for one-shot plate OCR when `process_frame(..., force_immediate_plate_ocr=True)` is used (APIs/tools).
SAMPLE_OCR_MIN_YOLO_CONF: float = 0.35


class RuleCatalogEntry(TypedDict, total=False):
    id: str
    title: str
    summary: str
    requires_model: Optional[str]
    needs_roi: List[str]


RULE_CATALOG: List[RuleCatalogEntry] = [
    {
        "id": "helmet",
        "title": "Helmet violation",
        "summary": "Flags no-helmet detections from the helmet model.",
        "requires_model": "helmet",
    },
    {
        "id": "triple",
        "title": "Triple seat",
        "summary": "Triple-seat riding when triple model is enabled.",
        "requires_model": "triple",
    },
    {
        "id": "truck_restricted",
        "title": "Truck restricted hours",
        "summary": "Trucks during the configured time window.",
        "requires_model": "truck",
    },
    {
        "id": "plate_ocr",
        "title": "Plate OCR",
        "summary": "Read plate text from plate detector crops.",
        "requires_model": "plate",
    },
    {
        "id": "red_light",
        "title": "Red light violation",
        "summary": "Vehicles crossing stop line on red (yolov10s + ROIs).",
        "needs_roi": ["signal_roi", "violation_rois"],
    },
    {
        "id": "no_parking",
        "title": "No parking",
        "summary": "Vehicles inside a forbidden zone (yolov8n + zone ROI).",
        "needs_roi": ["no_parking_zone"],
    },
]

# Red-light engine (TrafficLight branch)
RED_LIGHT_TRACK_DIRECTION: str = "none"
RED_LIGHT_MIN_MOVE_PIXELS: int = 1
RED_LIGHT_VEHICLE_CONF: float = 0.22
RED_LIGHT_MIN_ROI_BOX_OVERLAP: float = 0.17
RED_LIGHT_GRACE_FRAMES: int = 6
RED_LIGHT_FRAME_SIZE: Tuple[int, int] = (1020, 600)

# No-parking COCO class ids: car, motorcycle, bus, truck
NO_PARKING_VEHICLE_CLASS_IDS: List[int] = [2, 3, 5, 7]
NO_PARKING_VIOLATION_LABEL: str = "Wrong parking violation"
# Violation only after the vehicle stays in-zone with little movement for this long.
NO_PARKING_MIN_STAND_SEC: float = 5.0
# Reset dwell timer if centroid moves more than this many pixels between frames.
NO_PARKING_MAX_MOVE_PX: int = 15
