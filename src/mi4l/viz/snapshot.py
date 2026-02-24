from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C_SEG_A    = (30, 160, 30)        # Dark green  – segment A (trunk / reference side)
_C_SEG_B    = (60, 210, 60)        # Lime green  – segment B (limb being measured)
_C_REF      = (175, 175, 175)      # Light grey  – static reference lines (dashed)
_C_ARC_LINE = (255, 150, 0)        # Deep Cyan   – arc outline
_C_ARC_FILL = (255, 200, 100)      # Light Blue  – arc wedge fill (drawn at low alpha)
_C_PIVOT    = (40, 40, 220)        # Red         – pivot / vertex dot
_C_END      = (0, 200, 0)          # Green       – endpoint dots
_C_TXT_FG   = (255, 255, 255)      # White       – text face
_C_TXT_BG   = (30, 30, 30)         # Dark grey   – text badge background

# Drawing constants
_LINE_T      = 5                    # line thickness
_AA          = cv2.LINE_AA
_DOT_R_PVT   = 9                    # pivot dot radius
_DOT_R_END   = 7                    # endpoint dot radius
_ARC_R       = 60                   # arc / wedge radius (px)
_ARC_LINE_T  = 4                    # arc outline thickness
_WEDGE_ALPHA = 0.50                 # opacity of filled angle wedge


# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------

def _to_px(coord: tuple[float, float], w: int, h: int) -> tuple[int, int]:
    return int(round(coord[0] * w)), int(round(coord[1] * h))


def _seg(frame, p1, p2, color, thickness=_LINE_T, dash=False):
    """Solid or dashed anti-aliased line."""
    if p1 is None or p2 is None:
        return
    if not dash:
        cv2.line(frame, p1, p2, color, thickness, _AA)
    else:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        d = math.hypot(dx, dy)
        if d < 1:
            return
        n_dashes = max(1, int(d // 16))
        for i in range(n_dashes):
            t0 = i / n_dashes
            t1 = (i + 0.5) / n_dashes
            s = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
            e = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
            cv2.line(frame, s, e, color, thickness, _AA)


def _dot(frame, pt, color, radius, border=True):
    """Dot with a dark border for contrast."""
    if pt is None:
        return
    if border:
        cv2.circle(frame, pt, radius, (0, 0, 0), -1, _AA)
        cv2.circle(frame, pt, max(1, radius - 2), color, -1, _AA)
    else:
        cv2.circle(frame, pt, radius, color, -1, _AA)


def _angle_of(vec):
    """atan2 in degrees, normalised 0-360."""
    a = math.degrees(math.atan2(vec[1], vec[0]))
    return a % 360


def _short_arc_range(a1_deg, a2_deg):
    """
    Return (start, end) in cv2 ellipse convention such that we sweep
    the shorter arc between two directions.
    """
    a1 = a1_deg % 360
    a2 = a2_deg % 360
    diff = (a2 - a1) % 360
    if diff > 180:
        a1, a2 = a2, a1
        diff = 360 - diff
    return a1, a1 + diff


def _draw_wedge(frame, center, vec_a, vec_b, radius=_ARC_R,
                line_color=_C_ARC_LINE, fill_color=_C_ARC_FILL,
                alpha=_WEDGE_ALPHA, line_thick=_ARC_LINE_T):
    """
    Draw a filled + outlined pie-sector wedge between two direction vectors
    emanating from *center*. Uses polyFillPoly for a crisp filled sector.
    """
    a1 = _angle_of(vec_a)
    a2 = _angle_of(vec_b)
    start, end = _short_arc_range(a1, a2)

    # Build a filled polygon: centre + arc points
    n_pts = 64
    pts = [center]
    for i in range(n_pts + 1):
        ang = math.radians(start + (end - start) * i / n_pts)
        pts.append((int(center[0] + radius * math.cos(ang)),
                    int(center[1] + radius * math.sin(ang))))
    pts_arr = np.array(pts, dtype=np.int32)

    # Semi-transparent fill via overlay
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_arr], fill_color, _AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Crisp arc outline on the perimeter (only for the angle being measured)
    cv2.ellipse(frame, center, (radius, radius), 0.0,
                start, end, line_color, line_thick, _AA)


def _badge(frame, text: str, anchor: tuple[int, int]):
    """
    Crisp text badge with a rounded semi-transparent background.
    `anchor` = desired top-left of the text block.
    """
    font      = cv2.FONT_HERSHEY_DUPLEX
    scale     = 1.1
    thick     = 2
    pad_x, pad_y = 14, 10

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    fh, fw = frame.shape[:2]

    # Clamp so badge stays on-screen
    bx = max(pad_x, min(anchor[0], fw - tw - pad_x * 2))
    by = max(th + pad_y, min(anchor[1], fh - pad_y))

    x1 = bx - pad_x
    y1 = by - th - pad_y
    x2 = bx + tw + pad_x
    y2 = by + baseline + pad_y
    r  = 8  # corner radius

    # Semi-transparent rounded rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), _C_TXT_BG, -1, _AA)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x1 + r, y1 + r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x2 - r, y1 + r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x1 + r, y2 - r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x2 - r, y2 - r), r, _C_TXT_BG, -1, _AA)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Thin light border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1, _AA)

    # Text (white)
    cv2.putText(frame, text, (bx, by), font, scale, _C_TXT_FG, thick, _AA)


# ---------------------------------------------------------------------------
# Smart text placement (avoid person / vectors)
# ---------------------------------------------------------------------------

def _badge_anchor(A, B, C, mode, frame_shape, badge_w=200, badge_h=50):
    """
    Return (x, y) top-left anchor for the text badge.

    Strategy: sample 8 candidate positions at a fixed radius around the
    pivot B, pick the one furthest from all key points and farthest from
    the image edge — guarantees the badge stays clear of vectors and body.
    """
    fh, fw = frame_shape[:2]
    if B is None:
        return (20, 40)

    bx, by = B

    if mode == "bilateral":
        # Above pelvis centre
        tx = max(10, min(bx - badge_w // 2, fw - badge_w - 10))
        ty = max(10, by - 130)
        return (tx, ty)

    if mode == "distance":
        if A is not None:
            mx = (A[0] + bx) // 2
            my = min(A[1], by) - 70
        else:
            mx, my = bx, by - 70
        tx = max(10, min(mx - badge_w // 2, fw - badge_w - 10))
        ty = max(10, my)
        return (tx, ty)

    # Collect key landmark positions to avoid
    avoid_pts = [p for p in (A, B, C) if p is not None]

    # Sample 8 directions around B at two radii
    best_anchor = None
    best_score  = -1.0
    DIST = 150   # placement radius

    for angle_deg in range(0, 360, 45):
        ang_r = math.radians(angle_deg)
        cx = bx + DIST * math.cos(ang_r)
        cy = by + DIST * math.sin(ang_r)

        # Badge top-left so badge is centred at (cx, cy)
        tx = int(cx - badge_w / 2)
        ty = int(cy - badge_h / 2)

        # Keep on-screen
        tx = max(10, min(tx, fw - badge_w - 10))
        ty = max(10, min(ty, fh - badge_h - 10))

        # Score = min distance to any key point
        cx_clamped = tx + badge_w / 2
        cy_clamped = ty + badge_h / 2
        min_d = min(math.hypot(cx_clamped - px, cy_clamped - py)
                    for px, py in avoid_pts)

        # Bonus for being far from image edges
        edge_margin = min(tx, ty, fw - tx - badge_w, fh - ty - badge_h)
        score = min_d + 0.3 * edge_margin

        if score > best_score:
            best_score  = score
            best_anchor = (tx, ty)

    return best_anchor if best_anchor is not None else (max(10, bx + 70), max(40, by - 30))


# ---------------------------------------------------------------------------
# Mode inference
# ---------------------------------------------------------------------------

def _infer_mode(a_name, b_name, c_name) -> str:
    b = b_name or ""
    a = a_name or ""
    c = c_name or ""

    # Shoulder flexion: hip_midpoint → shoulder → wrist
    if "shoulder" in b and ("hip" in a or "midpoint" in a) and ("wrist" in c or "elbow" in c):
        return "internal"
    if "knee" in b:
        return "horizontal"
    if "hip" in b and ("shoulder" in a or "midpoint" in a):
        return "vertical"
    if "hip" in b:
        return "vertical"
    if "midpoint" in b and "shoulder" in b:
        return "horizontal"
    if "pelvis" in b and "ankle" in a and "ankle" in c:
        return "bilateral"
    if c is None or c == "":
        return "distance"
    return "internal"


# ---------------------------------------------------------------------------
# Render helpers (per mode)
# ---------------------------------------------------------------------------

def _render_internal(frame, A, B, C):
    """
    Two-segment angle: A→B (amber) + B→C (green) + shaded wedge at B.
    """
    if B is None:
        return
    if A is None or C is None:
        if C is not None:
            _seg(frame, B, C, _C_SEG_B)
        _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
        return

    ax, ay = A
    bx, by = B
    cx, cy = C

    ab_dx, ab_dy = bx - ax, by - ay
    ba_dx, ba_dy = ax - bx, ay - by
    bc_dx, bc_dy = cx - bx, cy - by
    ab_len = math.hypot(ab_dx, ab_dy)
    bc_len = math.hypot(bc_dx, bc_dy)

    # Extend A-B slightly beyond B to hint at direction
    if ab_len > 0:
        B_far = (int(bx + ab_dx * 0.25), int(by + ab_dy * 0.25))
    else:
        B_far = B

    # Extend B-C slightly beyond C
    if bc_len > 0:
        C_far = (int(cx + bc_dx * 0.15), int(cy + bc_dy * 0.15))
    else:
        C_far = C

    _seg(frame, A, B_far, _C_SEG_A, thickness=_LINE_T)
    _seg(frame, B, C_far, _C_SEG_B, thickness=_LINE_T)

    # Shaded wedge
    _draw_wedge(frame, B, (ba_dx, ba_dy), (bc_dx, bc_dy))

    _dot(frame, A, _C_END,   _DOT_R_END)
    _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
    _dot(frame, C, _C_END,   _DOT_R_END)


def _render_orientation(frame, A, B, C, mode, landmarks_row, w, h):
    """Vertical / horizontal reference with shaded wedge."""
    if B is None or C is None:
        return

    bcx = C[0] - B[0]
    bcy = C[1] - B[1]
    seg_len = math.hypot(bcx, bcy)

    # Extend the measurement segment both ways
    ext = max(220.0, seg_len * 2.2)
    if seg_len > 0:
        s = ext / seg_len
        C_ext  = (int(B[0] + bcx * s),    int(B[1] + bcy * s))
        C_back = (int(B[0] - bcx * 0.4),  int(B[1] - bcy * 0.4))
    else:
        C_ext, C_back = C, B

    _seg(frame, C_back, C_ext, _C_SEG_B)

    if mode == "vertical":
        ref_len = 220.0
        D = (B[0], int(B[1] + ref_len))
        _seg(frame, B, D, _C_REF, thickness=2, dash=True)
        _draw_wedge(frame, B, (0, 1), (bcx, bcy))

    elif mode == "horizontal":
        floor_y = _floor_y(landmarks_row, h)
        if abs(bcy) > 0.5:
            t = (floor_y - B[1]) / bcy
            fp_x = int(B[0] + t * bcx)
        else:
            fp_x = B[0]
        fp = (fp_x, floor_y)

        ref_len = 200
        horiz_end = (fp[0] - ref_len, fp[1])
        _seg(frame, fp, horiz_end, _C_REF, thickness=2, dash=True)

        if seg_len > 0:
            ux, uy = bcx / seg_len, bcy / seg_len
            vec_end = (int(fp[0] + ux * ref_len), int(fp[1] + uy * ref_len))
            _seg(frame, fp, vec_end, _C_SEG_B)

        _draw_wedge(frame, fp, (-1, 0), (bcx, bcy), radius=60)

    _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
    _dot(frame, C, _C_END,   _DOT_R_END)


def _render_bilateral(frame, A, B, C):
    """Bilateral straddle: pelvis → both ankles + shaded wedge."""
    if B is None or A is None or C is None:
        return
    _seg(frame, B, A, _C_SEG_B)
    _seg(frame, B, C, _C_SEG_A)

    va = (A[0] - B[0], A[1] - B[1])
    vc = (C[0] - B[0], C[1] - B[1])
    _draw_wedge(frame, B, va, vc, radius=50)

    _dot(frame, A, _C_END,   _DOT_R_END)
    _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
    _dot(frame, C, _C_END,   _DOT_R_END)


def _render_distance(frame, A, B, _pt):
    """Stick pass-through: wrist-to-wrist + shoulder reference."""
    if A is None or B is None:
        return
    _seg(frame, A, B, _C_SEG_B)
    _dot(frame, A, _C_END, _DOT_R_END)
    _dot(frame, B, _C_END, _DOT_R_END)

    l_sh = _pt("left_shoulder")
    r_sh = _pt("right_shoulder")
    if l_sh and r_sh:
        _seg(frame, l_sh, r_sh, _C_REF, thickness=2, dash=True)
        _dot(frame, l_sh, _C_REF, 5, border=False)
        _dot(frame, r_sh, _C_REF, 5, border=False)


def _floor_y(landmarks_row, h) -> int:
    for name in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
        yv = landmarks_row.get(f"{name}_y")
        if yv is not None and np.isfinite(float(yv)):
            return int(float(yv) * h)
    return h - 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_snapshot(
    video_path: str | Path,
    landmarks_row: dict,
    a_name: str | None,
    b_name: str,
    c_name: str | None,
    out_path: str | Path,
    angle_deg: float | None = None,
    angle_mode: str = "auto",
) -> None:
    """
    Render an annotated snapshot frame and save to *out_path*.

    Modes: auto | internal | horizontal | vertical | bilateral | distance
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = int(landmarks_row.get("frame_idx", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    fh, fw = frame.shape[:2]
    w = int(landmarks_row.get("image_w", fw))
    h = int(landmarks_row.get("image_h", fh))

    def _pt(name: str | None) -> tuple[int, int] | None:
        if name is None:
            return None
        xv = landmarks_row.get(f"{name}_x")
        yv = landmarks_row.get(f"{name}_y")
        if xv is None or yv is None:
            return None
        xv, yv = float(xv), float(yv)
        if not np.isfinite(xv) or not np.isfinite(yv):
            return None
        return _to_px((xv, yv), w, h)

    A = _pt(a_name)
    B = _pt(b_name)
    C = _pt(c_name)

    mode = angle_mode if angle_mode != "auto" else _infer_mode(a_name, b_name, c_name)

    # --- render geometry ---
    if mode == "internal":
        _render_internal(frame, A, B, C)
    elif mode in ("vertical", "horizontal"):
        _render_orientation(frame, A, B, C, mode, landmarks_row, w, h)
    elif mode == "bilateral":
        _render_bilateral(frame, A, B, C)
    elif mode == "distance":
        _render_distance(frame, A, B, _pt)

    # --- text badge ---
    if angle_deg is not None and B is not None:
        if mode == "distance":
            txt = f"{float(angle_deg):.2f}x shoulder"
        else:
            txt = f"{float(angle_deg):.1f} deg"
        anchor = _badge_anchor(A, B, C, mode, frame.shape)
        _badge(frame, txt, anchor)

    cv2.imwrite(str(p), frame)
