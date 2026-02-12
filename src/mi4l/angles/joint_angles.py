from __future__ import annotations

import numpy as np
import pandas as pd


def angle_orientation_deg(v_x: np.ndarray, v_y: np.ndarray, ref_axis: str = "vertical") -> np.ndarray:
    """
    Computes orientation of vector (v_x, v_y) relative to a reference axis.
    
    Returns angles where reference = 0° and angles increase as segment moves away.
    For horizontal reference, angles > 180° are remapped to keep starting position near 0°.
    """
    if ref_axis == "vertical":
        # 0 is down (0,1)
        theta = np.arctan2(v_x, v_y)
        angle_deg = np.degrees(theta)
    elif ref_axis == "horizontal":
        # 0 is flat (horizontal), 90 is up
        theta = np.arctan2(-v_y, v_x)
        angle_deg = np.degrees(theta)
    else:
        raise ValueError(f"Unknown reference axis: {ref_axis}")
        
    # Normalize to 0-360
    angle_deg = np.where(angle_deg < 0, angle_deg + 360.0, angle_deg)
    
    # For horizontal reference: remap angles > 180 to maintain "distance from reference"
    # This ensures 357° becomes 3° (small angle near reference)
    if ref_axis == "horizontal":
        angle_deg = np.where(angle_deg > 180, 360.0 - angle_deg, angle_deg)
        
    return angle_deg


def angle_abc_deg(a_xy: np.ndarray, b_xy: np.ndarray, c_xy: np.ndarray) -> np.ndarray:
    """
    Internal angle at point B for triangle A-B-C (in degrees).
    Returns 0-180.
    """
    ba = a_xy - b_xy
    bc = c_xy - b_xy

    dot = np.einsum("ij,ij->i", ba, bc)
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    denom = ba_norm * bc_norm

    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = dot / denom

    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))

    invalid = (denom <= 0) | np.isnan(denom) | np.isnan(a_xy).any(axis=1) | np.isnan(b_xy).any(axis=1) | np.isnan(c_xy).any(axis=1)
    ang[invalid] = np.nan
    return ang


def compute_knee_flexion_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 1 - Kneeling Knee Flexion
    Vector: Ankle - Knee (Lower Leg)
    Reference: Horizontal Floor (0=Flat, 90=Vertical Up)
    """
    def _xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            nan = np.full(len(landmarks_df), np.nan, dtype=np.float32)
            return nan, nan
        return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    lx_k, ly_k = _xy("left_knee")
    lx_a, ly_a = _xy("left_ankle")
    rx_k, ry_k = _xy("right_knee")
    rx_a, ry_a = _xy("right_ankle")

    # Vector = Ankle - Knee
    lv_x, lv_y = lx_a - lx_k, ly_a - ly_k
    rv_x, rv_y = rx_a - rx_k, ry_a - ry_k

    # Ref Horizontal
    left_deg = angle_orientation_deg(lv_x, lv_y, ref_axis="horizontal")
    right_deg = angle_orientation_deg(rv_x, rv_y, ref_axis="horizontal")

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_knee_flexion_deg": left_deg,
            "right_knee_flexion_deg": right_deg,
        }
    )
    return out


def compute_trunk_extension_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 2 - Prone Trunk Extension
    Vector: Shoulder_Mid - Hip_Mid
    Reference: Horizontal (0=Flat, Increasing=Extension Up)
    """
    def _xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            nan = np.full(len(landmarks_df), np.nan, dtype=np.float32)
            return nan, nan
        return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    lx_h, ly_h = _xy("left_hip")
    rx_h, ry_h = _xy("right_hip")
    hx_mid, hy_mid = (lx_h + rx_h) / 2.0, (ly_h + ry_h) / 2.0

    lx_s, ly_s = _xy("left_shoulder")
    rx_s, ry_s = _xy("right_shoulder")
    sx_mid, sy_mid = (lx_s + rx_s) / 2.0, (ly_s + ry_s) / 2.0

    # Vector = ShoulderMid - HipMid (Torso vector pointing to head)
    vx = sx_mid - hx_mid
    vy = sy_mid - hy_mid

    deg = angle_orientation_deg(vx, vy, ref_axis="horizontal")
    
    # Special handling for trunk extension:
    # When lying flat, torso points backward (180°), which should be ~0°
    # When extended, torso points up (90°)
    # So we want: 180° → 0°, 90° → 90°, 0° → 0°
    # Solution: remap angles > 90° as (180 - angle)
    deg = np.where(deg > 90, 180.0 - deg, deg)

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "trunk_extension_deg": deg,
        }
    )
    return out


def compute_hip_abduction_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 3 - Standing Hip Abduction
    Vector: Ankle - Hip (Leg)
    Reference: Vertical (0=Down)
    """
    def _xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            nan = np.full(len(landmarks_df), np.nan, dtype=np.float32)
            return nan, nan
        return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    lx_h, ly_h = _xy("left_hip")
    rx_h, ry_h = _xy("right_hip")
    lx_a, ly_a = _xy("left_ankle")
    rx_a, ry_a = _xy("right_ankle")

    # Vector = Ankle - Hip
    lv_x, lv_y = lx_a - lx_h, ly_a - ly_h
    rv_x, rv_y = rx_a - rx_h, ry_a - ry_h

    left_deg = angle_orientation_deg(lv_x, lv_y, ref_axis="vertical")
    right_deg = angle_orientation_deg(rv_x, rv_y, ref_axis="vertical")

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_hip_abduction_deg": left_deg,
            "right_hip_abduction_deg": right_deg,
        }
    )
    return out


def compute_bilateral_leg_straddle_angle(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 4 - Bilateral Leg Straddle
    Relative-Segment: Internal angle between left and right leg vectors.
    """
    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    l_hip = _xy("left_hip")
    r_hip = _xy("right_hip")
    l_ank = _xy("left_ankle")
    r_ank = _xy("right_ankle")

    # Legs vectors: Hip -> Ankle
    v_left = l_ank - l_hip
    v_right = r_ank - r_hip

    dot = np.einsum("ij,ij->i", v_left, v_right)
    nm_l = np.linalg.norm(v_left, axis=1)
    nm_r = np.linalg.norm(v_right, axis=1)
    denom = nm_l * nm_r

    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = dot / denom
    
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    
    invalid = (denom <= 0) | np.isnan(denom)
    ang[invalid] = np.nan

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "bilateral_leg_straddle_deg": ang,
        }
    )
    return out


def compute_hip_extension_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 5 - Unilateral Hip Extension
    Vector: Knee - Hip (Thigh)
    Reference: Vertical (0=Vertical)
    """
    def _xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            nan = np.full(len(landmarks_df), np.nan, dtype=np.float32)
            return nan, nan
        return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    lx_h, ly_h = _xy("left_hip")
    rx_h, ry_h = _xy("right_hip")
    lx_k, ly_k = _xy("left_knee")
    rx_k, ry_k = _xy("right_knee")

    # Vector = Knee - Hip
    lv_x, lv_y = lx_k - lx_h, ly_k - ly_h
    rv_x, rv_y = rx_k - rx_h, ry_k - ry_h

    left_deg = angle_orientation_deg(lv_x, lv_y, ref_axis="vertical")
    right_deg = angle_orientation_deg(rv_x, rv_y, ref_axis="vertical")

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_hip_extension_deg": left_deg,
            "right_hip_extension_deg": right_deg,
        }
    )
    return out


def compute_shoulder_flexion_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 6 - Shoulder Flexion
    Vector: Wrist - Shoulder (Arm)
    Reference: Vertical (0=Down)
    """
    def _xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            nan = np.full(len(landmarks_df), np.nan, dtype=np.float32)
            return nan, nan
        return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    lx_s, ly_s = _xy("left_shoulder")
    rx_s, ry_s = _xy("right_shoulder")
    lx_w, ly_w = _xy("left_wrist")
    rx_w, ry_w = _xy("right_wrist")

    # Vector = Wrist - Shoulder
    lv_x, lv_y = lx_w - lx_s, ly_w - ly_s
    rv_x, rv_y = rx_w - rx_s, ry_w - ry_s

    left_deg = angle_orientation_deg(lv_x, lv_y, ref_axis="vertical")
    right_deg = angle_orientation_deg(rv_x, rv_y, ref_axis="vertical")

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_shoulder_flexion_deg": left_deg,
            "right_shoulder_flexion_deg": right_deg,
        }
    )
    return out


def compute_stick_pass_through_metrics(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 7 - Stick Pass-Through
    Normalized distance.
    """
    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    l_w = _xy("left_wrist")
    r_w = _xy("right_wrist")
    l_s = _xy("left_shoulder")
    r_s = _xy("right_shoulder")

    wrist_dist = np.linalg.norm(l_w - r_w, axis=1)
    shoulder_dist = np.linalg.norm(l_s - r_s, axis=1)
    
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_dist = wrist_dist / shoulder_dist
        
    invalid = (shoulder_dist <= 0) | np.isnan(shoulder_dist)
    norm_dist[invalid] = np.nan

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "stick_pass_through_dist_norm": norm_dist,
        }
    )
    return out
