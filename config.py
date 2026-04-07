"""Central configuration for the traffic violation system."""

from pathlib import Path
from typing import Dict, List, Optional, TypedDict

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Truck YOLO weights (`models/truck.pt`).
TRUCK_MODEL_PATH = str(MODELS_DIR / "truck.pt")

# Video source:
# - 0 for webcam
# - "path/to/video.mp4" for a video file
VIDEO_SOURCE = 0

# Model paths (default for main.py / full pipeline)
MODEL_PATHS = {
    "truck": TRUCK_MODEL_PATH,
    "triple": str(MODELS_DIR / "triple.pt"),
}

# Number-plate detector weights (`models/plate.pt`).
PLATE_MODEL_PATH = str(MODELS_DIR / "plate.pt")

# When both **truck** and **plate** are active: run your dedicated plate YOLO only inside
# expanded truck boxes (then EasyOCR on crops). If no qualifying truck boxes, falls back to full frame.
TRUCK_SCOPED_PLATE_ONLY: bool = True
PLATE_DETECT_INSIDE_TRUCK_ROI: bool = True  # legacy; keep in sync with TRUCK_SCOPED_PLATE_ONLY
TRUCK_PLATE_ROI_PAD_FRAC: float = 0.18
TRUCK_ROI_PLATE_PAD_FRAC: float = 0.18  # legacy; keep in sync with TRUCK_PLATE_ROI_PAD_FRAC
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

# Plate text recognition uses EasyOCR on YOLO plate crops (see PLATE_OCR_* below).

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
# Run plate YOLO every N processed frames (1 = every frame). 2–3 cuts GPU load a lot; gate reuses cached boxes between runs.
PLATE_YOLO_EVERY_N_FRAMES: int = 2
# Drop dedicated plate-head boxes below this conf before tracking/OCR (reduces flicker and bad crops).
PLATE_YOLO_MIN_CONF: float = 0.28
# Expand each raw YOLO plate box by this fraction of its width/height before OCR (captures plate edges).
PLATE_BBOX_EXPAND_FRAC: float = 0.12
# When expansion > 0, draw the raw YOLO box (thin) inside the expanded OCR box (thick) on the video.
PLATE_DRAW_INNER_YOLO_BOX: bool = True

# --- Plate OCR gating (EasyOCR runs only when plate looks good + stable, not every frame) ---
PLATE_TRACK_MAX_DISTANCE: int = 95
PLATE_TRACK_MAX_DISAPPEARED: int = 22
# EMA on plate box per track (0 = off). Smooths jitter between YOLO refreshes for steadier crops and reads.
PLATE_BBOX_SMOOTH_ALPHA: float = 0.42
PLATE_OCR_MIN_YOLO_CONF: float = 0.42
PLATE_OCR_MIN_AREA: int = 1400
# width/height; include portrait-ish / square plate boxes (truck fronts, angles).
PLATE_OCR_MIN_ASPECT: float = 0.35
PLATE_OCR_MAX_ASPECT: float = 8.0
# Min IoU between tracked box and current YOLO box to reuse that det's score (else use last_yolo_conf).
PLATE_OCR_DET_IOU_MIN: float = 0.08
PLATE_OCR_MIN_SHARPNESS: float = 35.0
# If aspect/area fails but Laplacian sharpness reaches this, still allow OCR (stable + YOLO ok).
PLATE_OCR_BYPASS_GEOM_SHARPNESS: float = 120.0
PLATE_OCR_STABLE_FRAMES: int = 3
PLATE_OCR_MAX_CENTROID_DRIFT: float = 20.0
PLATE_OCR_MIN_TEXT_LEN: int = 4
# When False, EasyOCR runs at most once per track after gates pass (best for video throughput + stable read).
PLATE_OCR_ALLOW_UPGRADE: bool = False
PLATE_OCR_UPGRADE_QUALITY_RATIO: float = 1.25
# Minimum frames between OCR attempts while plate still has no valid read (higher = less CPU churn).
PLATE_OCR_RETRY_MIN_FRAMES: int = 24
# Limit OCR calls per processed frame to avoid random multi-second stalls when multiple tracks gate-pass together.
# 1 = smoothest timeline. Increase to 2 only if you need faster lock-on for many simultaneous plates.
PLATE_OCR_MAX_TRIES_PER_FRAME: int = 1
# Downscale plate crop so EasyOCR does not block for seconds on huge regions (inference unchanged).
PLATE_OCR_MAX_CROP_SIDE: int = 520
# Min shorter side after gray preprocess upscaling (helps small plates in truck ROI).
PLATE_OCR_PREPROCESS_MIN_SIDE: int = 140
# Extra border around YOLO crop before OCR (fraction of min side) — reduces clipped characters.
PLATE_OCR_INNER_PAD_FRAC: float = 0.10
# Run EasyOCR on several preprocessed views (base / sharpened / inverted / adaptive) and keep best read.
PLATE_OCR_MULTI_VARIANT: bool = True
PLATE_OCR_MAX_VARIANTS: int = 4
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


# UI (dashboard.py) / catalog: models you ship. Add entries when you drop real `.pt` files into `models/`
# (e.g. helmet / dedicated signal / no-parking YOLO heads were omitted until checkpoints exist).
MODEL_CATALOG: List[ModelCatalogEntry] = [
    {
        "id": "truck",
        "file": "truck.pt",
        "title": "Truck",
        "summary": "Detects trucks / heavy vehicles and powers time- and zone-based rules.",
        "description": "When enabled: restricted-hour violations, no-parking dwell timer, and stop-line checks combined with simple red/green signal estimation from a crop of the frame.",
    },
    {
        "id": "triple",
        "file": "triple.pt",
        "title": "Triple seat",
        "summary": "Counts **person** boxes on each **vehicle** (bike, rickshaw, …); violation when count ≥ 3 (configurable).",
        "description": "Multi-class head (vehicle + person + optional plate): persons are assigned per vehicle; **triple seat** fires when `TRIPLE_MIN_PERSONS_ON_MOTORCYCLE` persons sit on the same carrier. **Numberplate** boxes are ignored when the **Number plate** model is also loaded.",
    },
    {
        "id": "plate",
        "file": "plate.pt",
        "title": "Number plate",
        "summary": "YOLO plate detector + EasyOCR on crops.",
        "description": "Uses `models/plate.pt` for detection; plate text via EasyOCR on crops. When off, no plate YOLO or OCR runs.",
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

# Violation rules
NO_PARKING_TIME_THRESHOLD_SEC = 10

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

# Separate “restricted hours” for the truck-only rule (used only if TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW is False).
TRUCK_RESTRICTED_START_HOUR = 7
TRUCK_RESTRICTED_END_HOUR = 10

# When True (default), “Truck in restricted hours” uses the same window as TRUCK_VIOLATIONS_ACTIVE_* / dashboard sliders.
# When False, that message uses TRUCK_RESTRICTED_* above while no-parking/signal still follow the active window only.
TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW: bool = True

# Optional IANA timezone for rule hours, e.g. "Asia/Kolkata". None = system local time from datetime.now().
TRUCK_RULES_TIMEZONE: Optional[str] = None

# When truck model is on: only between these hours do we evaluate truck *violations*
# (no-parking, signal line, restricted-hour rule). Outside this window we still detect trucks,
# track IDs, and read number plates — no truck-rule violations.
# Semantics: active if start <= hour < end (half-open). If start == end, treated as always active.
# Overnight: start > end wraps midnight (e.g. 22–6). End may be 24 for “through end of day”.
TRUCK_VIOLATIONS_ACTIVE_START_HOUR = 6
TRUCK_VIOLATIONS_ACTIVE_END_HOUR = 22

# Tracker config
TRACKER_MAX_DISTANCE = 60
TRACKER_MAX_DISAPPEARED = 20

# Streamlit dashboard: when True, only sleep the *remainder* of each frame interval so wall clock
# tracks video time when your PC is fast enough. When False, process as fast as possible (no pacing).
VIDEO_REALTIME_PACING_DEFAULT: bool = True

# Process every Nth decoded frame from file (2 ≈ 2× faster; use with real-time pacing to stretch sleep)
VIDEO_DECODE_EVERY_N_FRAME: int = 1

# Streamlit: downscale preview so each frame upload stays small (full frame still used for YOLO).
DASHBOARD_PREVIEW_MAX_WIDTH: int = 960
# Throttle heavy Streamlit updates (stats, plate list, events, evidence strips). Main video preview is every frame.
DASHBOARD_UI_UPDATE_EVERY_N: int = 2
# Periodic gc on long videos (0 = off — gc often causes visible hitches).
DASHBOARD_GC_EVERY_N_FRAMES: int = 0
# Sidebar plate crop thumbnails (max width px).
DASHBOARD_SIDEBAR_PLATE_THUMB: int = 132
# Max plate captures kept per run (oldest dropped).
DASHBOARD_PLATE_GALLERY_MAX_ITEMS: int = 80

# FastAPI live stream: max width (px) for JPEG frames over SSE; 0 = full resolution (heavy).
WEB_STREAM_PREVIEW_MAX_WIDTH: int = 960
# Emit one stream frame every N decoded frames (1 = every frame shown live).
WEB_STREAM_FRAME_EVERY_N: int = 1

# Min YOLO conf for one-shot plate OCR when `process_frame(..., force_immediate_plate_ocr=True)` is used (APIs/tools).
SAMPLE_OCR_MIN_YOLO_CONF: float = 0.35
