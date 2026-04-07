"""Number plate text extraction with EasyOCR (see ``config`` for preprocessing and Indian-style options)."""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

# EasyOCR → PyTorch DataLoader uses pin_memory=True; Apple MPS ignores it and warns (noise).
warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*not supported on MPS.*",
    category=UserWarning,
)

_reader = None

# Alphanumeric plates (India / generic Latin); spaces allowed when ``PLATE_OCR_INDIAN_STYLE``.
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _ocr_allowlist() -> str:
    if bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False)):
        return OCR_ALLOWLIST + " "
    return OCR_ALLOWLIST


def get_easyocr_reader(lang_list: Optional[List[str]] = None):
    """Lazy singleton EasyOCR reader (GPU when available and enabled in config)."""
    global _reader
    if _reader is None:
        import easyocr
        import torch

        langs = lang_list or getattr(config, "EASYOCR_LANGS", ["en"])
        use_gpu = bool(getattr(config, "EASYOCR_USE_GPU", True)) and torch.cuda.is_available()
        # PyTorch DataLoader warns on Apple MPS when EasyOCR uses pin_memory=True (harmless).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*pin_memory.*not supported on MPS.*",
                category=UserWarning,
            )
            _reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    return _reader


def reset_reader() -> None:
    """Clear cached EasyOCR reader."""
    global _reader
    _reader = None


def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_ratio: Optional[float] = None) -> np.ndarray:
    if pad_ratio is None:
        pad_ratio = float(getattr(config, "PLATE_OCR_SAFE_CROP_PAD_FRAC", 0.08) or 0.08)
    h, w = frame.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    xa = max(0, x1 - px)
    ya = max(0, y1 - py)
    xb = min(w, x2 + px)
    yb = min(h, y2 + py)
    return frame[ya:yb, xa:xb].copy()


def _inner_pad_crop_bgr(crop_bgr: np.ndarray, pad_frac: float) -> np.ndarray:
    """Replicate-pad the plate crop so characters are not flush against the crop edge."""
    if crop_bgr is None or crop_bgr.size == 0 or pad_frac <= 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    px = max(2, int(min(h, w) * pad_frac))
    return cv2.copyMakeBorder(crop_bgr, px, px, px, px, cv2.BORDER_REPLICATE)


def _upscale_bgr_min_width(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale small crops so characters have enough pixels (critical for phone / blurry video)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    min_w = int(getattr(config, "PLATE_OCR_MIN_WIDTH_PX", 0) or 0)
    max_f = float(getattr(config, "PLATE_OCR_UPSCALE_MAX_FACTOR", 3.0) or 1.0)
    f = 1.0
    if min_w > 0 and w < min_w:
        f = max(f, min_w / float(max(1, w)))
    f = min(f, max(1.0, max_f))
    if f > 1.02:
        return cv2.resize(crop_bgr, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_CUBIC)
    return crop_bgr


def _indian_plate_extra_grays(bgr: np.ndarray) -> List[np.ndarray]:
    """Yellow / reflective plates: extra contrast views for EasyOCR."""
    out: List[np.ndarray] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Dark text on bright yellow → after invert, Otsu often separates strokes
    inv = cv2.bitwise_not(gray)
    _, otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(otsu)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 30, 40], dtype=np.uint8), np.array([50, 255, 255], dtype=np.uint8))
    if int(mask.sum()) > 50:
        v = gray.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out.append(clahe.apply(v))
    return out


def preprocess_plate_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """Enhance contrast / sharpness before OCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gh, gw = gray.shape[:2]
    target_min = int(getattr(config, "PLATE_OCR_PREPROCESS_MIN_SIDE", 96) or 96)
    target_min = max(64, target_min)
    scale = max(1.0, target_min / min(gh, gw, 1))
    if scale > 1.01:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _gray_variants_for_ocr(base_gray: np.ndarray) -> List[np.ndarray]:
    """Multiple single-channel views of the same plate — OCR picks the best."""
    gray = np.asarray(base_gray)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    variants: List[np.ndarray] = [gray]

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    unsharp = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)
    variants.append(np.clip(unsharp, 0, 255).astype(np.uint8))

    variants.append(cv2.bitwise_not(gray))

    try:
        mind = int(min(gray.shape[0], gray.shape[1]))
        if mind >= 40:
            bs = max(11, (mind // 4) | 1)
            bs = min(bs, mind - 2) if mind > 15 else 11
            if bs >= 3 and bs % 2 == 1 and bs < mind:
                at = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    bs,
                    5,
                )
                variants.append(at)
    except Exception:
        pass

    max_v = max(1, int(getattr(config, "PLATE_OCR_MAX_VARIANTS", 4) or 4))
    return variants[:max_v]


def _append_rotation_trials(variants: List[np.ndarray]) -> List[np.ndarray]:
    if not getattr(config, "PLATE_OCR_ROTATION_TRIALS", False) or not variants:
        return variants
    g0 = variants[0]
    extra = [
        cv2.rotate(g0, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(g0, cv2.ROTATE_180),
        cv2.rotate(g0, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    return variants + extra


def _bbox_x_center(bbox: Any) -> float:
    arr = np.asarray(bbox, dtype=float)
    if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
        return float(np.mean(arr[:, 0]))
    if arr.size >= 2:
        return float(arr.flat[0])
    return 0.0


def _normalize_plate_text(text: str) -> str:
    """Keep A–Z / 0–9; optionally single spaces (Indian plates like ``CG04 JD 7398``)."""
    if bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False)):
        t = re.sub(r"[^A-Za-z0-9\s]", "", text)
        t = re.sub(r"\s+", " ", t).strip()
    else:
        t = re.sub(r"[^A-Za-z0-9]", "", text)
    return t.upper()


def _merge_readtext_results(results: Sequence[Tuple[Any, str, Any]]) -> Tuple[str, float]:
    """Sort detections left-to-right, then join tokens (natural plate reading order)."""
    if not results:
        return "", 0.0
    items = sorted(results, key=lambda it: _bbox_x_center(it[0]))
    parts: List[str] = []
    confs: List[float] = []
    for _bbox, text, conf in items:
        token = _normalize_plate_text(str(text))
        if token:
            parts.append(token)
            confs.append(float(conf))
    if not parts:
        return "", 0.0
    if bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False)):
        combined = " ".join(parts)
    else:
        combined = "".join(parts)
    mean_conf = sum(confs) / len(confs)
    return combined, mean_conf


def _ocr_score(text: str, mean_conf: float) -> float:
    """Prefer confident reads with plausible plate length."""
    min_len = int(getattr(config, "PLATE_OCR_MIN_TEXT_LEN", 4))
    if not text or len(text.replace(" ", "")) < min_len:
        return -1.0
    n = len(text.replace(" ", ""))
    peak = 10.0 if bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False)) else 8.0
    len_bonus = 1.0 - min(abs(n - peak) / 16.0, 0.45)
    return float(mean_conf) * (0.55 + 0.45 * len_bonus)


def _readtext_kwargs() -> Dict[str, Any]:
    dec = str(getattr(config, "PLATE_OCR_DECODER", "beamsearch") or "beamsearch").lower()
    if dec not in ("greedy", "beamsearch"):
        dec = "beamsearch"
    indian = bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False))
    mag = float(getattr(config, "PLATE_OCR_MAG_RATIO", 1.45))
    if indian:
        mag = max(mag, 1.85)
    tw = float(getattr(config, "PLATE_OCR_TEXT_THRESHOLD", 0.52))
    lw = float(getattr(config, "PLATE_OCR_LOW_TEXT", 0.32))
    if indian:
        tw = min(tw, 0.42)
        lw = min(lw, 0.28)
    return {
        "detail": 1,
        "paragraph": False,
        "allowlist": _ocr_allowlist(),
        "decoder": dec,
        "beamWidth": max(1, int(getattr(config, "PLATE_OCR_BEAM_WIDTH", 8))),
        "width_ths": 0.45,
        "height_ths": 0.45,
        "mag_ratio": mag,
        "text_threshold": tw,
        "low_text": lw,
        "min_size": 8,
    }


def _recognize_on_gray(reader: Any, gray: np.ndarray) -> Tuple[str, float]:
    """
    Recognizer-only: treat the full crop as a single text line (no CRAFT detection inside the image).
    """
    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.shape[2] == 3 else g[:, :, 0]
    dec = str(getattr(config, "PLATE_OCR_DECODER", "beamsearch") or "beamsearch").lower()
    if dec not in ("greedy", "beamsearch"):
        dec = "beamsearch"
    bw = max(1, int(getattr(config, "PLATE_OCR_BEAM_WIDTH", 8)))
    try:
        results = reader.recognize(
            g,
            horizontal_list=None,
            free_list=None,
            decoder=dec,
            beamWidth=bw,
            allowlist=_ocr_allowlist(),
            detail=1,
            paragraph=False,
        )
    except TypeError:
        try:
            results = reader.recognize(
                g,
                decoder="greedy",
                beamWidth=bw,
                allowlist=_ocr_allowlist(),
                detail=1,
                paragraph=False,
            )
        except Exception:
            return "", 0.0
    except Exception:
        return "", 0.0
    return _merge_readtext_results(results)


def _readtext_on_gray(reader: Any, gray: np.ndarray) -> Tuple[str, float]:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    try:
        results = reader.readtext(rgb, **_readtext_kwargs())
    except TypeError:
        try:
            results = reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                allowlist=_ocr_allowlist(),
                width_ths=0.45,
                height_ths=0.45,
            )
        except Exception:
            return "", 0.0
    except Exception:
        return "", 0.0
    return _merge_readtext_results(results)


def _readtext_on_rgb(reader: Any, rgb: np.ndarray) -> Tuple[str, float]:
    """Color image (H,W,3) RGB — helps blurry yellow plates."""
    try:
        results = reader.readtext(rgb, **_readtext_kwargs())
    except TypeError:
        try:
            results = reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                allowlist=_ocr_allowlist(),
                width_ths=0.45,
                height_ths=0.45,
            )
        except Exception:
            return "", 0.0
    except Exception:
        return "", 0.0
    return _merge_readtext_results(results)


def _ocr_gray_dispatch(reader: Any, gray: np.ndarray, *, recognizer_only: bool) -> Tuple[str, float]:
    return _recognize_on_gray(reader, gray) if recognizer_only else _readtext_on_gray(reader, gray)


def _ocr_color_dispatch(reader: Any, work_bgr: np.ndarray, *, recognizer_only: bool) -> Tuple[str, float]:
    """First pass on the plate crop: either grayscale recognizer-only or full readtext (detect+rec)."""
    if recognizer_only:
        g = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        return _recognize_on_gray(reader, g)
    rgb_full = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    return _readtext_on_rgb(reader, rgb_full)


def _read_plate_easyocr(reader: Any, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    EasyOCR on the plate crop only (input is already the YOLO box from the frame).

    With ``PLATE_OCR_RECOGNIZER_ONLY`` (default True), uses EasyOCR ``recognize`` on the whole crop as one
    line — no second-stage text detection inside the crop. Otherwise uses ``readtext`` (CRAFT + recognizer).
    """
    if reader is None:
        reader = get_easyocr_reader()
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    max_side = int(getattr(config, "PLATE_OCR_MAX_CROP_SIDE", 0) or 0)
    work = crop_bgr
    if max_side > 0:
        ch, cw = work.shape[:2]
        m = max(ch, cw)
        if m > max_side:
            scale = max_side / float(m)
            work = cv2.resize(
                work,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

    work = _upscale_bgr_min_width(work)

    pad_frac = float(getattr(config, "PLATE_OCR_INNER_PAD_FRAC", 0.0) or 0.0)
    if pad_frac > 0:
        work = _inner_pad_crop_bgr(work, pad_frac)

    recognizer_only = bool(getattr(config, "PLATE_OCR_RECOGNIZER_ONLY", True))
    fallback_readtext = bool(getattr(config, "PLATE_OCR_RECOGNIZER_FALLBACK_READTEXT", True))

    indian = bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False))
    best_text, best_conf = "", 0.0
    best_score = -1.0

    def _try(txt: str, ocf: float) -> None:
        nonlocal best_text, best_conf, best_score
        sc = _ocr_score(txt, ocf)
        if sc > best_score:
            best_score = sc
            best_text, best_conf = txt, ocf

    def _run_variants(use_ro: bool) -> None:
        if indian:
            t, c = _ocr_color_dispatch(reader, work, recognizer_only=use_ro)
            _try(t, c)
            for ig in _indian_plate_extra_grays(work):
                t2, c2 = _ocr_gray_dispatch(reader, ig, recognizer_only=use_ro)
                _try(t2, c2)

        base_gray = preprocess_plate_bgr(work)
        if base_gray is None or base_gray.size == 0:
            return
        if getattr(config, "PLATE_OCR_MULTI_VARIANT", True):
            grays = _gray_variants_for_ocr(base_gray)
        else:
            grays = [base_gray]
        grays = _append_rotation_trials(grays)
        for g in grays:
            t, c = _ocr_gray_dispatch(reader, g, recognizer_only=use_ro)
            _try(t, c)

    _run_variants(recognizer_only)
    if best_text:
        return best_text, best_conf
    if recognizer_only and fallback_readtext:
        _run_variants(False)
    if best_text:
        return best_text, best_conf

    base_gray = preprocess_plate_bgr(work)
    if base_gray is None or base_gray.size == 0:
        return "", 0.0
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    try:
        results = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            allowlist=_ocr_allowlist(),
            width_ths=0.5,
            height_ths=0.5,
        )
    except Exception:
        return "", 0.0
    return _merge_readtext_results(results)


def read_plate_from_crop(reader: Any, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Decode text from a **plate crop only** (BGR region from the YOLO box + optional padding).

    Never receives the full frame. See ``PLATE_OCR_RECOGNIZER_ONLY`` for recognizer vs detect+read.
    ``reader`` may be None to lazy-load EasyOCR.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    return _read_plate_easyocr(reader, crop_bgr)


def _point_in_bbox(px: float, py: float, bbox: Sequence[int]) -> bool:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return x1 <= px <= x2 and y1 <= py <= y2


def ocr_plate_detections_one_shot(
    frame: np.ndarray,
    plate_dets: List[dict],
    reader: Any,
    *,
    min_yolo_conf: float = 0.35,
    min_chars: Optional[int] = None,
    truck_boxes: Optional[List[Sequence[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run EasyOCR once per plate YOLO box (no temporal gating). For sampled stills / debugging.
    """
    mc = min_chars if min_chars is not None else int(getattr(config, "PLATE_OCR_MIN_TEXT_LEN", 4))
    out: List[Dict[str, Any]] = []
    for d in plate_dets:
        yc = float(d.get("confidence", 0.0))
        if yc < min_yolo_conf:
            continue
        x1, y1, x2, y2 = (int(d["bbox"][0]), int(d["bbox"][1]), int(d["bbox"][2]), int(d["bbox"][3]))
        crop = _safe_crop(frame, x1, y1, x2, y2)
        if crop.size == 0:
            continue
        try:
            txt, ocf = read_plate_from_crop(reader, crop)
        except Exception:
            txt, ocf = "", 0.0
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        near_truck = False
        if truck_boxes:
            for tb in truck_boxes:
                if _point_in_bbox(cx, cy, tb):
                    near_truck = True
                    break
        raw = d.get("bbox_raw")
        out.append(
            {
                "text": txt if txt and len(txt) >= mc else "",
                "confidence": float(ocf),
                "yolo_conf": yc,
                "bbox": [x1, y1, x2, y2],
                "bbox_raw": [int(x) for x in raw] if raw is not None else [x1, y1, x2, y2],
                "ocr_error": not bool(txt) or (len(txt) < mc if txt else True),
                "near_truck": near_truck,
            }
        )
    return out


def read_plates_for_detections(reader, frame: np.ndarray, plate_detections: list) -> List[dict]:
    """
    plate_detections: list of dicts with 'bbox' [x1,y1,x2,y2].
    Returns list of {text, confidence, bbox}.
    """
    out: List[dict] = []
    for det in plate_detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = _safe_crop(frame, x1, y1, x2, y2)
        text, conf = read_plate_from_crop(reader, crop)
        out.append(
            {
                "text": text,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return out
