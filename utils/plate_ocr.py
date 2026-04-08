"""Number plate text extraction on crops (EasyOCR by default; optional PaddleOCR)."""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

import config

# EasyOCR → PyTorch DataLoader uses pin_memory=True; Apple MPS ignores it and warns (noise).
warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*not supported on MPS.*",
    category=UserWarning,
)

_reader = None
_lprnet_reader = None
_paddle_reader = None

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
    """Clear cached OCR readers."""
    global _reader, _lprnet_reader, _paddle_reader
    _reader = None
    _lprnet_reader = None
    _paddle_reader = None


class _LPRSmallBasicBlock(torch.nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class _LPRNet(torch.nn.Module):
    def __init__(self, class_num: int, lpr_max_len: int = 8, dropout_rate: float = 0.5):
        super().__init__()
        self.class_num = int(class_num)
        self.lpr_max_len = int(lpr_max_len)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            _LPRSmallBasicBlock(ch_in=64, ch_out=128),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            _LPRSmallBasicBlock(ch_in=64, ch_out=256),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            _LPRSmallBasicBlock(ch_in=256, ch_out=256),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(in_channels=256, out_channels=self.class_num, kernel_size=(13, 1), stride=1),
            torch.nn.BatchNorm2d(num_features=self.class_num),
            torch.nn.ReLU(),
        )
        self.container = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1)),
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = []
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = torch.nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = torch.nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)
        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


def get_lprnet_reader():
    """Lazy singleton LPRNet reader loaded from ``config.LPRNET_MODEL_PATH``."""
    global _lprnet_reader
    if _lprnet_reader is not None:
        return _lprnet_reader
    model_path = str(getattr(config, "LPRNET_MODEL_PATH", "") or "")
    if not model_path:
        return None
    if not config.is_model_file_usable(model_path):
        return None
    model = None
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception:
        pass
    if model is None:
        try:
            loaded = torch.load(model_path, map_location="cpu")
            if hasattr(loaded, "eval"):
                model = loaded
            elif isinstance(loaded, dict):
                state_dict = None
                if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                    state_dict = loaded["state_dict"]
                elif "model" in loaded and isinstance(loaded["model"], dict):
                    state_dict = loaded["model"]
                elif all(isinstance(v, torch.Tensor) for v in loaded.values()):
                    state_dict = loaded
                if state_dict is not None:
                    out_w = state_dict.get("container.0.weight")
                    class_num = int(out_w.shape[0]) if isinstance(out_w, torch.Tensor) else (
                        len(str(getattr(config, "LPRNET_CHARSET", ""))) + 1
                    )
                    net = _LPRNet(class_num=class_num)
                    net.load_state_dict(state_dict, strict=False)
                    model = net
        except Exception:
            model = None
    if model is None or not hasattr(model, "eval"):
        return None
    model.eval()
    _lprnet_reader = model
    return _lprnet_reader


def get_paddle_reader():
    """Lazy singleton PaddleOCR reader."""
    global _paddle_reader
    if _paddle_reader is not None:
        return _paddle_reader
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None
    lang = str(getattr(config, "PADDLEOCR_LANG", "en") or "en")
    use_gpu = bool(getattr(config, "PADDLEOCR_USE_GPU", False))
    use_cls = bool(getattr(config, "PADDLEOCR_USE_ANGLE_CLS", True))
    try:
        _paddle_reader = PaddleOCR(use_textline_orientation=use_cls, lang=lang, use_gpu=use_gpu)
    except TypeError:
        try:
            _paddle_reader = PaddleOCR(use_angle_cls=use_cls, lang=lang, use_gpu=use_gpu)
        except TypeError:
            _paddle_reader = PaddleOCR(use_angle_cls=use_cls, lang=lang)
    except Exception:
        return None
    return _paddle_reader


def get_plate_reader(lang_list: Optional[List[str]] = None):
    """Return configured OCR reader object (EasyOCR by default; PaddleOCR when ``PLATE_OCR_ENGINE`` is paddle)."""
    engine = str(getattr(config, "PLATE_OCR_ENGINE", "easyocr") or "easyocr").lower()
    if engine == "paddle":
        rdr = get_paddle_reader()
        if rdr is not None:
            return rdr
        if bool(getattr(config, "PLATE_OCR_FALLBACK_TO_EASYOCR", True)):
            return get_easyocr_reader(lang_list)
        return None
    return get_easyocr_reader(lang_list)


def _ctc_decode_lprnet(logits: torch.Tensor) -> Tuple[str, float]:
    """
    Greedy CTC decode for logits shaped [B,C,T] or [B,T,C].
    Blank index is assumed as the last class.
    """
    if logits.ndim != 3:
        return "", 0.0
    t = logits.detach().float().cpu()
    if t.shape[1] > t.shape[2]:
        # [B, C, T] -> [B, T, C]
        t = t.permute(0, 2, 1)
    probs = torch.softmax(t, dim=-1)
    ids = torch.argmax(probs, dim=-1)[0].tolist()
    conf_seq = torch.max(probs, dim=-1).values[0].tolist()
    charset = str(getattr(config, "LPRNET_CHARSET", OCR_ALLOWLIST) or OCR_ALLOWLIST)
    blank = int(probs.shape[-1] - 1)
    out_chars: List[str] = []
    out_confs: List[float] = []
    prev = None
    for i, ch_id in enumerate(ids):
        if ch_id == blank:
            prev = ch_id
            continue
        if prev == ch_id:
            continue
        if 0 <= ch_id < len(charset):
            out_chars.append(charset[ch_id])
            out_confs.append(float(conf_seq[i]))
        prev = ch_id
    if not out_chars:
        return "", 0.0
    txt = _normalize_plate_text("".join(out_chars))
    return txt, float(sum(out_confs) / max(1, len(out_confs)))


def _read_plate_lprnet(model: Any, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """LPRNet inference on a plate crop (BGR) with CTC greedy decode."""
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    work = _upscale_bgr_min_width(crop_bgr)
    gray = preprocess_plate_bgr(work)
    if gray is None or gray.size == 0:
        return "", 0.0
    iw = int(getattr(config, "LPRNET_INPUT_WIDTH", 94) or 94)
    ih = int(getattr(config, "LPRNET_INPUT_HEIGHT", 24) or 24)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    ten = (ten - 0.5) / 0.5
    with torch.no_grad():
        out = model(ten)
    if isinstance(out, (tuple, list)) and out:
        out = out[0]
    if not isinstance(out, torch.Tensor):
        return "", 0.0
    return _ctc_decode_lprnet(out)


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


def _read_plate_paddle(reader: Any, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """PaddleOCR on a plate crop. Returns merged text and mean confidence."""
    if reader is None or crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    work = _upscale_bgr_min_width(crop_bgr)
    pad_frac = float(getattr(config, "PLATE_OCR_INNER_PAD_FRAC", 0.0) or 0.0)
    if pad_frac > 0:
        work = _inner_pad_crop_bgr(work, pad_frac)
    best_t, best_c, best_s = "", 0.0, -1.0
    variants: List[np.ndarray] = [work]
    if bool(getattr(config, "PLATE_OCR_MULTI_VARIANT", True)):
        pg = preprocess_plate_bgr(work)
        if pg is not None and pg.size > 0:
            variants.append(cv2.cvtColor(pg, cv2.COLOR_GRAY2BGR))
    for im in variants[:2]:
        try:
            raw = reader.ocr(im, cls=bool(getattr(config, "PADDLEOCR_USE_ANGLE_CLS", True)))
        except TypeError:
            raw = reader.ocr(im)
        except Exception:
            continue
        cand: List[Tuple[float, str, float]] = []
        rows = raw if isinstance(raw, list) else []
        if rows and isinstance(rows[0], list) and len(rows) == 1:
            rows = rows[0]
        for r in rows:
            if not isinstance(r, (list, tuple)) or len(r) < 2:
                continue
            box, txtc = r[0], r[1]
            if not isinstance(txtc, (list, tuple)) or len(txtc) < 2:
                continue
            txt = _normalize_plate_text(str(txtc[0]))
            if not txt:
                continue
            conf = float(txtc[1])
            try:
                xc = float(np.mean(np.asarray(box)[:, 0]))
            except Exception:
                xc = 0.0
            cand.append((xc, txt, conf))
        if not cand:
            continue
        cand.sort(key=lambda x: x[0])
        merged = " ".join(c[1] for c in cand) if bool(getattr(config, "PLATE_OCR_INDIAN_STYLE", False)) else "".join(
            c[1] for c in cand
        )
        ocf = float(sum(c[2] for c in cand) / len(cand))
        score = _ocr_score(merged, ocf)
        if score > best_s:
            best_t, best_c, best_s = merged, ocf, score
    return best_t, best_c


def read_plate_from_crop(reader: Any, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Decode text from a **number-plate image crop only** (BGR patch from plate YOLO + ``_safe_crop`` padding).

    Violation models and plate YOLO run on the full frame in the pipeline; this function is the **only**
    place OCR sees pixels, and it never receives the full frame.
    ``reader`` may be None to lazy-load the configured OCR engine.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    engine = str(getattr(config, "PLATE_OCR_ENGINE", "easyocr") or "easyocr").lower()
    if engine == "paddle":
        pr = reader if reader is not None else get_paddle_reader()
        txt, conf = _read_plate_paddle(pr, crop_bgr)
        if txt:
            return txt, conf
        if bool(getattr(config, "PLATE_OCR_FALLBACK_TO_EASYOCR", True)):
            eo_reader = get_easyocr_reader()
            return _read_plate_easyocr(eo_reader, crop_bgr)
        return "", 0.0
    return _read_plate_easyocr(reader if reader is not None else get_easyocr_reader(), crop_bgr)


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
