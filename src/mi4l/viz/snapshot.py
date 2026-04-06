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
_C_ARC_LINE = (255, 150, 0)        # Cyan        – arc outline
_C_ARC_FILL = (255, 200, 100)      # Light blue  – arc wedge fill
_C_PIVOT    = (40, 40, 220)        # Red         – pivot / vertex dot
_C_END      = (0, 200, 0)          # Green       – endpoint dots
_C_TXT_FG   = (255, 255, 255)      # White       – text
_C_TXT_BG   = (30, 30, 30)         # Dark grey   – badge background

# Drawing constants
_LINE_T      = 5
_AA          = cv2.LINE_AA
_DOT_R_PVT   = 9
_DOT_R_END   = 7
_ARC_R       = 60
_ARC_LINE_T  = 4
_WEDGE_ALPHA = 0.50


# ---------------------------------------------------------------------------
# Low-level drawing primitives  (shared, stable – do not change)
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
    """Return (start, end) for the shorter arc between two directions."""
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
    """Draw a filled + outlined pie-sector wedge between two direction vectors."""
    a1 = _angle_of(vec_a)
    a2 = _angle_of(vec_b)
    start, end = _short_arc_range(a1, a2)

    n_pts = 64
    pts = [center]
    for i in range(n_pts + 1):
        ang = math.radians(start + (end - start) * i / n_pts)
        pts.append((int(center[0] + radius * math.cos(ang)),
                    int(center[1] + radius * math.sin(ang))))
    pts_arr = np.array(pts, dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_arr], fill_color, _AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.ellipse(frame, center, (radius, radius), 0.0,
                start, end, line_color, line_thick, _AA)


def _badge(frame, text: str, anchor: tuple[int, int]):
    """Crisp text badge with a rounded semi-transparent background."""
    font      = cv2.FONT_HERSHEY_DUPLEX
    scale     = 1.1
    thick     = 2
    pad_x, pad_y = 14, 10

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    fh, fw = frame.shape[:2]

    bx = max(pad_x, min(anchor[0], fw - tw - pad_x * 2))
    by = max(th + pad_y, min(anchor[1], fh - pad_y))

    x1 = bx - pad_x
    y1 = by - th - pad_y
    x2 = bx + tw + pad_x
    y2 = by + baseline + pad_y
    r  = 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), _C_TXT_BG, -1, _AA)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x1 + r, y1 + r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x2 - r, y1 + r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x1 + r, y2 - r), r, _C_TXT_BG, -1, _AA)
    cv2.circle(overlay, (x2 - r, y2 - r), r, _C_TXT_BG, -1, _AA)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1, _AA)
    cv2.putText(frame, text, (bx, by), font, scale, _C_TXT_FG, thick, _AA)


def _badge_anchor(A, B, C, mode, frame_shape, badge_w=200, badge_h=50):
    """Return (x, y) top-left anchor for the text badge."""
    fh, fw = frame_shape[:2]
    if B is None:
        return (20, 40)

    bx, by = B

    if mode == "bilateral":
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

    avoid_pts = [p for p in (A, B, C) if p is not None]
    best_anchor = None
    best_score  = -1.0
    DIST = 150

    for angle_deg in range(0, 360, 45):
        ang_r = math.radians(angle_deg)
        cx = bx + DIST * math.cos(ang_r)
        cy = by + DIST * math.sin(ang_r)

        tx = int(cx - badge_w / 2)
        ty = int(cy - badge_h / 2)
        tx = max(10, min(tx, fw - badge_w - 10))
        ty = max(10, min(ty, fh - badge_h - 10))

        cx_clamped = tx + badge_w / 2
        cy_clamped = ty + badge_h / 2
        min_d = min(math.hypot(cx_clamped - px, cy_clamped - py)
                    for px, py in avoid_pts)
        edge_margin = min(tx, ty, fw - tx - badge_w, fh - ty - badge_h)
        score = min_d + 0.3 * edge_margin

        if score > best_score:
            best_score  = score
            best_anchor = (tx, ty)

    return best_anchor if best_anchor is not None else (max(10, bx + 70), max(40, by - 30))


# ---------------------------------------------------------------------------
# Generic render helpers  (used by poses that share a common structure)
# ---------------------------------------------------------------------------

def _render_internal(frame, A, B, C):
    """Two-segment angle: A→B + B→C + shaded wedge at B."""
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

    B_far = (int(bx + ab_dx * 0.25), int(by + ab_dy * 0.25)) if ab_len > 0 else B
    C_far = (int(cx + bc_dx * 0.15), int(cy + bc_dy * 0.15)) if bc_len > 0 else C

    _seg(frame, A, B_far, _C_SEG_A, thickness=_LINE_T)
    _seg(frame, B, C_far, _C_SEG_B, thickness=_LINE_T)
    _draw_wedge(frame, B, (ba_dx, ba_dy), (bc_dx, bc_dy))
    _dot(frame, A, _C_END,   _DOT_R_END)
    _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
    _dot(frame, C, _C_END,   _DOT_R_END)


def _render_vertical(frame, B, C):
    """Segment vector vs. downward vertical reference."""
    if B is None or C is None:
        return
    bcx = C[0] - B[0]
    bcy = C[1] - B[1]
    seg_len = math.hypot(bcx, bcy)

    ext = max(110.0, seg_len * 1.3)
    if seg_len > 0:
        s = ext / seg_len
        C_ext  = (int(B[0] + bcx * s),   int(B[1] + bcy * s))
        C_back = (int(B[0] - bcx * 0.2), int(B[1] - bcy * 0.2))
    else:
        C_ext, C_back = C, B

    _seg(frame, C_back, C_ext, _C_SEG_B)
    ref_len = 120.0
    D = (B[0], int(B[1] + ref_len))
    _seg(frame, B, D, _C_REF, thickness=2, dash=True)
    _draw_wedge(frame, B, (0, 1), (bcx, bcy))
    _dot(frame, B, _C_PIVOT, _DOT_R_PVT)
    _dot(frame, C, _C_END,   _DOT_R_END)


def _render_kneeling_knee(frame, B, C):
    """Kneeling knee flexion: segment vs. horizontal-right from knee pivot."""
    if B is None or C is None:
        return
    bcx = C[0] - B[0]
    bcy = C[1] - B[1]
    seg_len = math.hypot(bcx, bcy)

    ext = max(110.0, seg_len * 1.3)
    if seg_len > 0:
        s = ext / seg_len
        C_ext  = (int(B[0] + bcx * s),   int(B[1] + bcy * s))
        C_back = (int(B[0] - bcx * 0.2), int(B[1] - bcy * 0.2))
    else:
        C_ext, C_back = C, B

    _seg(frame, C_back, C_ext, _C_SEG_B)
    ref_len = 150.0
    horiz_end = (int(B[0] + ref_len), B[1])
    _seg(frame, B, horiz_end, _C_REF, thickness=2, dash=True)
    _draw_wedge(frame, B, (1, 0), (bcx, bcy), radius=60)
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


# ---------------------------------------------------------------------------
# Per-pose isolated snapshot render functions
# Each function receives (frame, lm_row, w, h, angle_deg).
# They ONLY draw geometry — badge is added after by save_snapshot.
# Returns the (A, B, C, mode) tuple used for badge anchor.
# ---------------------------------------------------------------------------

def _render_kneeling_knee_flexion(frame, lm_row, w, h, side, **_):
    """Knee flexion: Hip – Knee – Ankle, kneeling_knee mode (horiz ref from knee)."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    B = pt(f"{side}_knee")
    C = pt(f"{side}_ankle")
    _render_kneeling_knee(frame, B, C)
    return pt(f"{side}_hip"), B, C, "kneeling_knee"


def _render_unilateral_hip_extension(frame, lm_row, w, h, side=None, **_):
    """Hip extension: pelvis_center → both knees, bilateral wedge."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    # Compute pelvis_center if not already present
    if lm_row.get("pelvis_center_x") is None:
        lhx = lm_row.get("left_hip_x"); lhy = lm_row.get("left_hip_y")
        rhx = lm_row.get("right_hip_x"); rhy = lm_row.get("right_hip_y")
        if None not in (lhx, lhy, rhx, rhy):
            lm_row["pelvis_center_x"] = (float(lhx) + float(rhx)) / 2.0
            lm_row["pelvis_center_y"] = (float(lhy) + float(rhy)) / 2.0

    A = pt("left_knee")
    B = pt("pelvis_center")
    C = pt("right_knee")
    _render_bilateral(frame, A, B, C)
    return A, B, C, "bilateral"


def _render_standing_hip_abduction(frame, lm_row, w, h, side, **_):
    """Hip abduction: contralateral_hip – ipsilateral_hip – ipsilateral_ankle vs. vertical."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    other = "right" if side == "left" else "left"
    B = pt(f"{side}_hip")
    C = pt(f"{side}_ankle")
    _render_vertical(frame, B, C)
    return pt(f"{other}_hip"), B, C, "vertical"


def _render_bilateral_leg_straddle(frame, lm_row, w, h, **_):
    """Bilateral straddle: pelvis → both ankles."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    A = pt("left_ankle")
    B = pt("pelvis_center")
    C = pt("right_ankle")
    _render_bilateral(frame, A, B, C)
    return A, B, C, "bilateral"


def _render_prone_trunk_extension(frame, lm_row, w, h, **_):
    """
    Prone trunk extension – ISOLATED renderer.

    Anatomy:
      - Pelvis (hip midpoint) is the ORIGIN / pivot.
      - Trunk vector: hip_midpoint → shoulder_midpoint.
      - Reference: horizontal floor line extending to the RIGHT of the hip.
      - Inside angle is between trunk vector and the horizontal.
    """
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    hip   = pt("hip_midpoint")       # origin / pivot
    sho   = pt("shoulder_midpoint")  # trunk end

    if hip is None:
        return None, None, None, "trunk_extension"

    hx, hy = hip

    # Trunk vector: hip → shoulder
    if sho is not None:
        sx, sy = sho
        trunk_dx = sx - hx
        trunk_dy = sy - hy
        trunk_len = math.hypot(trunk_dx, trunk_dy)

        # Extend trunk vector beyond shoulder for clarity
        if trunk_len > 0:
            scale = max(trunk_len * 1.2, 120.0) / trunk_len
            sho_ext = (int(hx + trunk_dx * scale), int(hy + trunk_dy * scale))
        else:
            sho_ext = sho

        _seg(frame, hip, sho_ext, _C_SEG_B)
        _dot(frame, sho, _C_END, _DOT_R_END)
    else:
        trunk_dx, trunk_dy = 0, -1  # fallback upward

    # Horizontal reference: extend to the LEFT of the hip (inside angle side)
    ref_len = 160.0
    horiz_end = (int(hx - ref_len), hy)
    _seg(frame, hip, horiz_end, _C_REF, thickness=2, dash=True)

    # Wedge between trunk vector and leftward horizontal
    _draw_wedge(frame, hip, (-1, 0), (trunk_dx, trunk_dy), radius=70)

    _dot(frame, hip, _C_PIVOT, _DOT_R_PVT)

    return None, hip, sho, "trunk_extension"


def _render_shoulder_flexion(frame, lm_row, w, h, side, **_):
    """Shoulder flexion: hip_midpoint – shoulder – wrist, internal angle."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    A = pt("hip_midpoint")
    B = pt(f"{side}_shoulder")
    C = pt(f"{side}_wrist")
    _render_internal(frame, A, B, C)
    return A, B, C, "internal"


def _render_shoulder_stick_pass_through(frame, lm_row, w, h, _pt_fn, **_):
    """Stick pass-through: left_wrist – right_wrist distance."""
    def pt(name):
        xv = lm_row.get(f"{name}_x"); yv = lm_row.get(f"{name}_y")
        if xv is None or yv is None: return None
        if not (np.isfinite(float(xv)) and np.isfinite(float(yv))): return None
        return _to_px((float(xv), float(yv)), w, h)

    A = pt("left_wrist")
    B = pt("right_wrist")
    _render_distance(frame, A, B, _pt_fn)
    return A, B, None, "distance"


# ---------------------------------------------------------------------------
# Dispatch table — maps pose name → isolated renderer
# ---------------------------------------------------------------------------

SNAPSHOT_RENDERERS = {
    "kneeling_knee_flexion":      _render_kneeling_knee_flexion,
    "unilateral_hip_extension":   _render_unilateral_hip_extension,
    "standing_hip_abduction":     _render_standing_hip_abduction,
    "bilateral_leg_straddle":     _render_bilateral_leg_straddle,
    "prone_trunk_extension":      _render_prone_trunk_extension,
    "shoulder_flexion":           _render_shoulder_flexion,
    "shoulder_stick_pass_through": _render_shoulder_stick_pass_through,
}


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
    pose_name: str | None = None,
) -> None:
    """
    Render an annotated snapshot frame and save to *out_path*.

    If *pose_name* is provided and exists in SNAPSHOT_RENDERERS, the
    pose-specific renderer is used (fully isolated, no cross-pose side effects).
    Otherwise falls back to the legacy generic mode dispatcher.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = int(landmarks_row.get("frame_idx", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        # Requested frame is beyond the video length — seek to the last readable frame.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fallback_idx = max(0, min(frame_idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"  [WARN] snapshot: frame {frame_idx} out of range "
                  f"(total={total}), using frame {fallback_idx} instead.")
        else:
            cap.release()
            print(f"  [WARN] snapshot: could not read any frame from {video_path} — skipping snapshot.")
            return
    cap.release()

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

    # Determine which side is active from the b_name (pivot) / a_name hints
    side = "left"
    for hint in (b_name or "", a_name or ""):
        if hint.startswith("right_"):
            side = "right"
            break
        if hint.startswith("left_"):
            side = "left"
            break

    # --- Pose-specific dispatch ---
    if pose_name and pose_name in SNAPSHOT_RENDERERS:
        renderer = SNAPSHOT_RENDERERS[pose_name]
        A, B, C, mode = renderer(
            frame, landmarks_row, w, h,
            side=side, _pt_fn=_pt,
        )
    else:
        # Legacy fallback (for any future poses not yet in the table)
        A = _pt(a_name)
        B = _pt(b_name)
        C = _pt(c_name)
        mode = angle_mode if angle_mode != "auto" else _infer_mode_legacy(a_name, b_name, c_name)

        if mode == "internal":
            _render_internal(frame, A, B, C)
        elif mode == "vertical":
            _render_vertical(frame, B, C)
        elif mode == "kneeling_knee":
            _render_kneeling_knee(frame, B, C)
        elif mode == "bilateral":
            _render_bilateral(frame, A, B, C)
        elif mode == "distance":
            _render_distance(frame, A, B, _pt)

    # --- text badge ---
    if angle_deg is not None and B is not None:
        if mode == "distance":
            # angle_deg stores shoulder/wrist; invert to show grip as a multiple of shoulder width
            val = float(angle_deg)
            grip_multiple = (1.0 / val) if val > 0 else float("inf")
            txt = f"{grip_multiple:.2f}x shoulder"
        else:
            txt = f"{float(angle_deg):.1f} deg"
        anchor = _badge_anchor(A, B, C, mode, frame.shape)
        _badge(frame, txt, anchor)

    cv2.imwrite(str(p), frame)


def _infer_mode_legacy(a_name, b_name, c_name) -> str:
    """Legacy mode inference – only used when pose_name is not supplied."""
    b = b_name or ""
    a = a_name or ""
    c = c_name or ""
    if "shoulder" in b and ("hip" in a or "midpoint" in a) and ("wrist" in c or "elbow" in c):
        return "internal"
    if "knee" in b:
        return "kneeling_knee"
    if "hip" in b:
        return "vertical"
    if "pelvis" in b and "ankle" in a and "ankle" in c:
        return "bilateral"
    if c is None or c == "":
        return "distance"
    return "internal"
