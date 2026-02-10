from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import math
import numpy as np


def _to_px(coord: tuple[float, float], w: int, h: int) -> tuple[int, int]:
    x, y = coord
    return int(round(x * w)), int(round(y * h))


def save_snapshot(
    video_path: str | Path,
    landmarks_row: dict,
    a_name: str,
    b_name: str,
    c_name: str,
    out_path: str | Path,
    angle_deg: float | None = None,
) -> None:
    """
    Save a snapshot image for a single frame described by `landmarks_row`.

    `landmarks_row` should contain normalized coords like 'left_hip_x', 'left_hip_y'
    and `image_w`/`image_h` pixel dims.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # landmarks_row expected to have 'frame_idx' key with video frame number
    frame_idx = int(landmarks_row.get("frame_idx", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    h, w = frame.shape[:2]
    # Prefer explicit image_w/image_h if present (should match)
    if "image_w" in landmarks_row and "image_h" in landmarks_row:
        try:
            w = int(landmarks_row.get("image_w", w))
            h = int(landmarks_row.get("image_h", h))
        except Exception:
            pass

    def _pt(name: str) -> tuple[int, int] | None:
        xk = f"{name}_x"
        yk = f"{name}_y"
        if xk not in landmarks_row or yk not in landmarks_row:
            return None
        xv = landmarks_row.get(xk)
        yv = landmarks_row.get(yk)
        if xv is None or yv is None or not np.isfinite(xv) or not np.isfinite(yv):
            return None
        return _to_px((float(xv), float(yv)), w, h)

    A = _pt(a_name)
    B = _pt(b_name)
    C = _pt(c_name)

    # draw lines and points
    if A is not None and B is not None:
        cv2.line(frame, A, B, (0, 200, 0), thickness=4)
        cv2.circle(frame, A, radius=6, color=(0, 200, 0), thickness=-1)
    if B is not None and C is not None:
        cv2.line(frame, B, C, (0, 200, 0), thickness=4)
        cv2.circle(frame, C, radius=6, color=(0, 200, 0), thickness=-1)
    if B is not None:
        cv2.circle(frame, B, radius=7, color=(0, 0, 255), thickness=-1)

    # draw angle arc at B
    # To visualize the external angle (flexion from straight), we want the angle between:
    # Vector 1: Extension of thigh (A->B extended)
    # Vector 2: Lower leg (B->C)
    if A is not None and B is not None and C is not None:
        # Vector BA = A - B
        bax = A[0] - B[0]
        bay = A[1] - B[1]
        
        # Extend BA through B to find "straight" reference D
        # D = B - (A-B) = 2B - A
        # Or simpler vector direction from B is -BA
        dx = -bax
        dy = -bay
        
        # Normalize extension vector for drawing
        d_len = math.hypot(dx, dy)
        if d_len > 0:
            scale = 100.0 / d_len
            D = (int(B[0] + dx * scale), int(B[1] + dy * scale))
            # Draw gray extension line
            cv2.line(frame, B, D, (180, 180, 180), thickness=2, lineType=cv2.LINE_AA)

        # Vector BC = C - B
        bcx = C[0] - B[0]
        bcy = C[1] - B[1]
        
        # Angle of extension vector (reference 0)
        ang_ref = math.degrees(math.atan2(dy, dx))
        
        # Angle of lower leg
        ang_leg = math.degrees(math.atan2(bcy, bcx))
        
        # OpenCV ellipse draws clockwise? No, standard is counter-clockwise but y-axis is down.
        # Let's use simple start/end.
        start = float(ang_ref)
        end = float(ang_leg)
        
        # normalize to positive range
        while end < start:
             end += 360.0
        
        # Adjust direction if needed (users prefer acute/obtuse correctly)
        # If difference > 180, interpret as the other way
        if (end - start) > 180:
            start += 360
            start, end = end, start

        # choose radius
        r = max(40, int(min(math.hypot(bax, bay), math.hypot(bcx, bcy)) / 2.5))
        center = B
        axes = (r, r)
        
        # draw ellipse arc (cyan)
        # 0.0 is angle of main axes rotation
        cv2.ellipse(frame, center, axes, 0.0, start, end, (255, 200, 0), thickness=3)

    # write angle text (no degree symbol to avoid artifacts)
    if angle_deg is not None and B is not None:
        txt = f"{float(angle_deg):.1f}"
        pos = (B[0] + 10, max(20, B[1] - 10))
        cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(p), frame)
    cap.release()
