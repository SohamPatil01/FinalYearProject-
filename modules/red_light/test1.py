import cv2
import numpy as np
from collections import deque

# Raw per-frame guesses (trees / glare) are smoothed over this window
_LIGHT_HISTORY = deque(maxlen=11)
# Analyze only the center of the drawn signal box (edges often show trees/sky)
_SIGNAL_INNER_MARGIN_FRAC = 0.22


def _bgr_red_dominance(bgr, v_ch):
    """Lit pixels where red channel wins (common for red LEDs when HSV hue is unreliable)."""
    b, g, r = cv2.split(bgr)
    bright = (v_ch > 38).astype(np.uint8) * 255
    rd = (r > 75).astype(np.uint8) * 255
    rg = (r.astype(np.int16) > g.astype(np.int16) + 15).astype(np.uint8) * 255
    rb = (r.astype(np.int16) > b.astype(np.int16) + 15).astype(np.uint8) * 255
    return cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(rd, rg), rb), bright)


def _bgr_green_dominance(bgr, v_ch):
    """Lit green LED: green should beat red clearly (reduces false green on red lamps)."""
    b, g, r = cv2.split(bgr)
    bright = (v_ch > 48).astype(np.uint8) * 255
    gd = (g > 70).astype(np.uint8) * 255
    gr = (g.astype(np.int16) > r.astype(np.int16) + 22).astype(np.uint8) * 255
    gb = (g.astype(np.int16) > b.astype(np.int16) + 12).astype(np.uint8) * 255
    return cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(gd, gr), gb), bright)


def _build_light_masks(hsv, bgr):
    """
    Red: dual HSV wraps + BGR red dominance (handles bloom / clipping).
    Green: narrow saturated green hue; must be G-dominant in BGR; no pixel votes both.
    """
    _h, _s, v_ch = cv2.split(hsv)
    # Orange–red and full red wrap; looser S so dim reds still count
    red_a = cv2.inRange(hsv, (0, 38, 48), (20, 255, 255))
    red_b = cv2.inRange(hsv, (158, 38, 48), (179, 255, 255))
    red = cv2.bitwise_or(red_a, red_b)
    red = cv2.bitwise_or(red, _bgr_red_dominance(bgr, v_ch))

    # True green lamps only: tight hue, high saturation (excludes yellow / cyan / grey housing)
    green_h = cv2.inRange(hsv, (44, 88, 68), (92, 255, 255))
    green = cv2.bitwise_and(green_h, _bgr_green_dominance(bgr, v_ch))

    # Any pixel we call red must not count toward green (fixes red→green misreads)
    green = cv2.bitwise_and(green, cv2.bitwise_not(red))
    return red, green


def _refine_masks(red, green, k=3):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.dilate(red, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    green = cv2.dilate(green, kernel, iterations=1)
    return red, green


def _draw_dots_in_signal_box(frame, off_x, off_y, mask, radius=5, max_dots=4):
    """
    Blue dots at blob centroids inside the signal ROI (BGR blue = 255,0,0).
    Coordinates are in crop space; shifted by off_x, off_y onto the full frame.
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    min_area = max(4, int(0.0005 * mask.shape[0] * mask.shape[1]))
    drawn = 0
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"]) + off_x
        cy = int(M["m01"] / M["m00"]) + off_y
        cv2.circle(frame, (cx, cy), radius, (255, 0, 0), -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)
        drawn += 1
        if drawn >= max_dots:
            return
    if drawn == 0:
        M = cv2.moments(mask)
        if M["m00"] > 1e-6:
            cx = int(M["m10"] / M["m00"]) + off_x
            cy = int(M["m01"] / M["m00"]) + off_y
            cv2.circle(frame, (cx, cy), radius, (255, 0, 0), -1)
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)


def _signal_inner_box(x1, y1, x2, y2, margin_frac=_SIGNAL_INNER_MARGIN_FRAC):
    """Shrink to central region so background at ROI edges (trees, sky) is ignored."""
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20:
        return x1, y1, x2, y2
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    ix1, iy1 = x1 + mx, y1 + my
    ix2, iy2 = x2 - mx, y2 - my
    if ix2 <= ix1 + 4 or iy2 <= iy1 + 4:
        return x1, y1, x2, y2
    return ix1, iy1, ix2, iy2


def _lit_mask(v_ch, percentile=74, floor_v=58):
    """Keep only brighter pixels (the lamp), not dark foliage / housing."""
    if v_ch.size == 0:
        return None
    thr = max(floor_v, int(np.percentile(v_ch, percentile)))
    return ((v_ch >= thr).astype(np.uint8)) * 255


def _smooth_light_label(hist):
    """Majority over recent frames; ties → RED. Blue dots do not affect this — they only visualize."""
    non = [x for x in hist if x is not None]
    if not non:
        return None
    rc = non.count("RED")
    gc = non.count("GREEN")
    if rc > gc:
        return "RED"
    if gc > rc:
        return "GREEN"
    return "RED"


def _classify_from_counts(red_count, green_count, roi_pixels):
    """Pick RED / GREEN; green needs a clearer lead to avoid false GREEN on red lights."""
    min_px = max(8, int(0.003 * roi_pixels))
    margin_red = 1.12
    margin_green = 1.45

    if red_count < min_px and green_count < min_px:
        return None
    if red_count >= min_px and red_count > green_count * margin_red:
        return "RED"
    if green_count >= min_px and green_count > red_count * margin_green:
        return "GREEN"
    if red_count >= min_px or green_count >= min_px:
        # Ambiguous: prefer RED (safer for violation logic)
        if red_count >= green_count:
            return "RED"
        return "GREEN" if green_count > red_count * 1.08 else "RED"
    return None


def process_frame(frame, signal_roi=None, signal_label_on_frame=True):
    """
    Classify traffic light as RED / GREEN.
    With signal_roi: classifies using the *center* of the box + bright-pixel gate + temporal smoothing
    (reduces tree background and flicker). Full user box is still outlined for reference.
    If signal_label_on_frame is False, the RED/GREEN/? text on the signal box is omitted (e.g. HUD elsewhere).
    """
    h, w = frame.shape[:2]
    off_x, off_y = 0, 0
    work = frame
    legacy_x_cutoff = 915
    outer_signal = None  # (x1,y1,x2,y2) full drawn box for overlay

    if signal_roi is not None:
        x1, y1, x2, y2 = signal_roi
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        outer_signal = (x1, y1, x2, y2)
        ix1, iy1, ix2, iy2 = _signal_inner_box(x1, y1, x2, y2)
        work = frame[iy1:iy2, ix1:ix2]
        off_x, off_y = ix1, iy1
    else:
        xc = min(legacy_x_cutoff, w)
        if xc < 2:
            return frame, None
        work = frame[:, 0:xc]

    if work.size == 0:
        return frame, None

    bgr = cv2.GaussianBlur(work, (3, 3), 0)
    # Do not CLAHE before HSV — it skews hue and often turns red blobs into false greens
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red_m, green_m = _build_light_masks(hsv, bgr)
    red_m, green_m = _refine_masks(red_m, green_m, k=3)
    green_m = cv2.bitwise_and(green_m, cv2.bitwise_not(red_m))

    _h2, _s2, v_ch = cv2.split(hsv)
    lit = _lit_mask(v_ch)
    if lit is not None:
        red_m = cv2.bitwise_and(red_m, lit)
        green_m = cv2.bitwise_and(green_m, lit)

    roi_pixels = work.shape[0] * work.shape[1]
    red_count = int(cv2.countNonZero(red_m))
    green_count = int(cv2.countNonZero(green_m))
    raw_label = _classify_from_counts(red_count, green_count, roi_pixels)

    if outer_signal is not None:
        _LIGHT_HISTORY.append(raw_label)
        detected_label = _smooth_light_label(_LIGHT_HISTORY)
    else:
        detected_label = raw_label

    if signal_roi is not None and outer_signal is not None:
        sx1, sy1, sx2, sy2 = outer_signal
        if detected_label == "RED":
            border = (0, 0, 255)
            text_color = (0, 0, 255)
        elif detected_label == "GREEN":
            border = (0, 255, 0)
            text_color = (0, 255, 0)
        else:
            border = (128, 128, 128)
            text_color = (200, 200, 200)
        # Outline the full ROI you drew; classification uses the inner region only
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), border, 2)
        if signal_label_on_frame:
            txt = detected_label if detected_label else "?"
            cv2.putText(
                frame,
                txt,
                (sx1, max(18, sy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2,
            )
        # Dots only over the inner analysis crop (same pixels used for RED/GREEN counts)
        rw, rh = work.shape[1], work.shape[0]
        dot_r = max(3, min(rw, rh) // 14)
        if detected_label == "RED":
            _draw_dots_in_signal_box(frame, off_x, off_y, red_m, radius=dot_r)
        elif detected_label == "GREEN":
            _draw_dots_in_signal_box(frame, off_x, off_y, green_m, radius=dot_r)
        else:
            both = cv2.bitwise_or(red_m, green_m)
            _draw_dots_in_signal_box(frame, off_x, off_y, both, radius=max(2, dot_r - 1))
    else:
        if detected_label == "RED":
            color, text_color = (0, 0, 255), (0, 0, 255)
        elif detected_label == "GREEN":
            color, text_color = (0, 255, 0), (0, 255, 0)
        else:
            return frame, None
        ys, xs = np.where(cv2.bitwise_or(red_m, green_m) > 0)
        if len(xs) == 0:
            return frame, None
        gx1, gy1 = int(xs.min()) + off_x, int(ys.min()) + off_y
        gx2, gy2 = int(xs.max()) + off_x, int(ys.max()) + off_y
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 2)
        if signal_label_on_frame:
            cv2.putText(
                frame,
                detected_label,
                (gx1, gy1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )

    return frame, detected_label
