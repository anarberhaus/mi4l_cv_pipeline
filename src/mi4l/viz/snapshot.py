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
    a_name: str | None,
    b_name: str,
    c_name: str,
    out_path: str | Path,
    angle_deg: float | None = None,
    angle_mode: str = "auto",  # "auto", "internal", "vertical", "horizontal"
) -> None:
    """
    Save a snapshot image for a single frame described by `landmarks_row`.
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
        cap.release()
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    h, w = frame.shape[:2]
    if "image_w" in landmarks_row and "image_h" in landmarks_row:
        try:
            w = int(landmarks_row.get("image_w", w))
            h = int(landmarks_row.get("image_h", h))
        except Exception:
            pass

    def _pt(name: str | None) -> tuple[int, int] | None:
        if name is None:
            return None
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

    # Auto-infer mode if needed
    mode = angle_mode
    if mode == "auto":
        if "knee" in b_name:
            mode = "horizontal"
        elif "hip" in b_name or "shoulder" in b_name:
            if "midpoint" in b_name and "shoulder" in b_name:
                mode = "horizontal"
            elif "midpoint" in a_name and "shoulder" in a_name and "hip" in b_name:    
                mode = "vertical"
            elif "midpoint" in a_name and "hip" in a_name and "shoulder" in b_name:
                 mode = "vertical"
            elif "hip" in b_name:
                 mode = "vertical"
            elif "shoulder" in b_name:
                 mode = "vertical"
            else:
                 mode = "internal"
        elif "pelvis" in b_name:
            mode = "internal"
        else:
            mode = "internal"

    # Draw points based on mode
    # For orientation modes (vertical/horizontal), only draw B and C
    # For internal mode, draw A, B, C
    if mode in ["vertical", "horizontal"]:
        # Orientation-based: only show pivot (B) and endpoint (C)
        if B is not None:
            cv2.circle(frame, B, radius=7, color=(0, 0, 255), thickness=-1)
        if C is not None:
            cv2.circle(frame, C, radius=6, color=(0, 200, 0), thickness=-1)
    else:
        # Internal angle: show all three points
        if A is not None:
            cv2.circle(frame, A, radius=6, color=(0, 200, 0), thickness=-1)
        if B is not None:
            cv2.circle(frame, B, radius=7, color=(0, 0, 255), thickness=-1)
        if C is not None:
            cv2.circle(frame, C, radius=6, color=(0, 200, 0), thickness=-1)

    # Draw lines and arc based on mode
    if B is not None and C is not None:
        start_angle = 0.0
        end_angle = 0.0
        draw_arc = False
        
        # 1. Orientation Modes
        if mode in ["vertical", "horizontal"]:
            # Draw segment B->C
            cv2.line(frame, B, C, (0, 200, 0), thickness=4)
            
            # Vector B->C
            bcx = C[0] - B[0]
            bcy = C[1] - B[1]
            seg_len = math.hypot(bcx, bcy)
            ref_len = max(100.0, seg_len * 0.8)
            
            if mode == "vertical":
                # Reference is DOWN (0, 1) -> +y in image
                D = (B[0], int(B[1] + ref_len))
                cv2.line(frame, B, D, (180, 180, 180), thickness=2, lineType=cv2.LINE_AA)
                # CV2: 0°=Right, 90°=Down
                ref_angle_cv2 = 90.0
                
            elif mode == "horizontal":
                # Reference is RIGHT (1, 0)
                D = (int(B[0] + ref_len), B[1])
                cv2.line(frame, B, D, (180, 180, 180), thickness=2, lineType=cv2.LINE_AA)
                # CV2: 0°=Right
                ref_angle_cv2 = 0.0

            # Calculate B->C angle in CV2 coordinates
            vec_angle_cv2 = math.degrees(math.atan2(bcy, bcx))
            
            # Draw arc from reference to vector (interior angle)
            start_angle = ref_angle_cv2
            end_angle = vec_angle_cv2
            
            # Normalize angles to 0-360
            if start_angle < 0:
                start_angle += 360
            if end_angle < 0:
                end_angle += 360
            
            # Calculate the angular difference
            diff = (end_angle - start_angle) % 360
            
            # Choose the shorter arc direction
            if diff > 180:
                # Go the other way (swap and use 360 - diff)
                start_angle, end_angle = end_angle, start_angle
            
            draw_arc = True
            
        # 2. Internal Angle Mode
        elif mode == "internal":
            if A is not None:
                cv2.line(frame, A, B, (0, 200, 0), thickness=4)
                cv2.line(frame, B, C, (0, 200, 0), thickness=4)
                
                # Extension of A->B
                bax = A[0] - B[0]
                bay = A[1] - B[1]
                dx, dy = -bax, -bay
                d_len = math.hypot(dx, dy)
                if d_len > 0:
                    scale = 100.0 / d_len
                    D = (int(B[0] + dx * scale), int(B[1] + dy * scale))
                    cv2.line(frame, B, D, (180, 180, 180), thickness=2, lineType=cv2.LINE_AA)
                    
                    start_angle = math.degrees(math.atan2(dy, dx))
                    bcx = C[0] - B[0]
                    bcy = C[1] - B[1]
                    end_angle = math.degrees(math.atan2(bcy, bcx))
                    
                    # Normalize and ensure shorter arc
                    start_angle = start_angle % 360
                    end_angle = end_angle % 360
                    diff = (end_angle - start_angle) % 360
                    if diff > 180:
                        start_angle, end_angle = end_angle, start_angle
                    if end_angle < start_angle:
                        end_angle += 360
                        
                    draw_arc = True
            else:
                cv2.line(frame, B, C, (0, 200, 0), thickness=4)

        # Draw Arc
        if draw_arc:
            r = 40
            cv2.ellipse(frame, B, (r, r), 0.0, start_angle, end_angle, (255, 200, 0), thickness=3)

    # write angle text - position it next to the arc, not on top of points
    if angle_deg is not None and B is not None:
        txt = f"{float(angle_deg):.1f} deg"
        
        # Position text to the right and slightly below the arc
        # Calculate a position that's offset from B in a clear area
        if C is not None:
            # Position text along the bisector of the angle, outside the arc
            bcx = C[0] - B[0]
            bcy = C[1] - B[1]
            
            # Normalize and scale to create offset
            bc_len = math.hypot(bcx, bcy)
            if bc_len > 0:
                # Offset perpendicular to B->C, to the right side
                offset_dist = 60  # pixels from B
                # Use perpendicular direction (rotate 90°)
                perp_x = -bcy / bc_len
                perp_y = bcx / bc_len
                
                text_x = int(B[0] + perp_x * offset_dist)
                text_y = int(B[1] + perp_y * offset_dist)
            else:
                # Fallback: offset to the right
                text_x = B[0] + 50
                text_y = B[1]
        else:
            # Fallback: offset to the right and down
            text_x = B[0] + 50
            text_y = B[1] + 20
            
        pos = (text_x, text_y)
        cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(p), frame)
    cap.release()
