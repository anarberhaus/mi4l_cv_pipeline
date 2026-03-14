from __future__ import annotations

import numpy as np
import pandas as pd


def angle_orientation_deg(v_x: np.ndarray, v_y: np.ndarray, ref_axis: str = "vertical") -> np.ndarray:
    """
    Computes orientation of vector (v_x, v_y) relative to a reference axis.
    
    Returns angles where reference = 0° and angles increase as segment moves away.
    For both vertical and horizontal references, angles > 180° are remapped 
    to keep starting position near 0°.
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
    
    # Remap angles > 180 to maintain "distance from reference"
    # This ensures 357° becomes 3° (small angle near reference)
    # Applies to both vertical (hip/shoulder abduction) and horizontal (trunk extension)
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


def angle_with_reference_deg(v_x: np.ndarray, v_y: np.ndarray, ref_x: float, ref_y: float, aspect: np.ndarray | None = None) -> np.ndarray:
    """
    Computes the internal angle (0-180 degrees) between a segment vector (v_x, v_y) and a constant
    reference vector (ref_x, ref_y). Incorporates optional aspect ratio correction.
    """
    if aspect is not None:
        v_x = v_x * aspect
        
    dot = v_x * ref_x + v_y * ref_y
    v_norm = np.sqrt(v_x**2 + v_y**2)
    ref_norm = np.sqrt(ref_x**2 + ref_y**2)
    denom = v_norm * ref_norm
    
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = dot / denom
        
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    
    invalid = (denom <= 0) | np.isnan(denom) | np.isnan(v_x) | np.isnan(v_y)
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
    
    Computes the internal leg separation angle using angular positions in the
    image plane, corrected for video aspect ratio.
    
    MediaPipe landmarks are stored as normalized (0-1) coordinates.
    A 16:9 video has x range 0-1 mapping to 1920px and y range 0-1 mapping
    to 1080px. Without correction, horizontal spread is underweighted,
    making angles appear smaller than they are.
    
    Method:
        1. pelvis_center = midpoint(left_hip, right_hip)
        2. Scale x-coordinates by (image_w / image_h) to correct aspect ratio
        3. angle_to_left = atan2(dy_left, dx_left_scaled)
        4. angle_to_right = atan2(dy_right, dx_right_scaled)
        5. separation = abs(angle_right - angle_left)
        6. if separation > 180°: separation = 360° - separation
    """

    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32),
                                y.to_numpy(dtype=np.float32)])

    l_hip = _xy("left_hip")
    r_hip = _xy("right_hip")
    l_ank = _xy("left_ankle")
    r_ank = _xy("right_ankle")

    # Aspect ratio correction: scale x so that x and y represent equal physical distances
    # MediaPipe stores normalized coords (0-1), so x=0.1 and y=0.1 span different pixel counts
    image_w = landmarks_df.get("image_w")
    image_h = landmarks_df.get("image_h")
    if image_w is not None and image_h is not None:
        aspect = (image_w.to_numpy(dtype=np.float32) / image_h.to_numpy(dtype=np.float32))
    else:
        aspect = np.ones(len(landmarks_df), dtype=np.float32)

    # Pelvis center is the reference point
    pelvis_center = (l_hip + r_hip) / 2.0
    
    # Compute deltas from pelvis center to each ankle
    dx_left = (l_ank[:, 0] - pelvis_center[:, 0]) * aspect   # scale x by aspect ratio
    dy_left = l_ank[:, 1] - pelvis_center[:, 1]               # y stays as-is
    angle_to_left = np.arctan2(dy_left, dx_left)
    
    dx_right = (r_ank[:, 0] - pelvis_center[:, 0]) * aspect   # scale x by aspect ratio
    dy_right = r_ank[:, 1] - pelvis_center[:, 1]               # y stays as-is
    angle_to_right = np.arctan2(dy_right, dx_right)
    
    # Convert to degrees
    angle_to_left_deg = np.degrees(angle_to_left)
    angle_to_right_deg = np.degrees(angle_to_right)
    
    # Compute angular separation (always positive)
    separation = np.abs(angle_to_right_deg - angle_to_left_deg)
    
    # If separation > 180°, take the complementary angle
    separation = np.where(separation > 180, 360 - separation, separation)
    
    # Mark invalid where inputs are NaN
    invalid = np.isnan(dx_left) | np.isnan(dy_left) | np.isnan(dx_right) | np.isnan(dy_right)
    separation[invalid] = np.nan

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "bilateral_leg_straddle_deg": separation,
        }
    )

    return out


def compute_hip_extension_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 5 - Unilateral Hip Extension

    Computes the angular separation between the two thigh segments
    (hip→knee for each leg) using angular positions in the image plane,
    corrected for video aspect ratio.

    This is a vector-vector measurement (same approach as the bilateral leg
    straddle) — it captures the true range of motion between both legs
    regardless of trunk orientation.

    Method:
        1. pelvis_center = midpoint(left_hip, right_hip)
        2. Scale x-coordinates by (image_w / image_h) to correct aspect ratio
        3. angle_to_left  = atan2(dy_left_knee,  dx_left_knee_scaled)
        4. angle_to_right = atan2(dy_right_knee, dx_right_knee_scaled)
        5. separation = abs(angle_right - angle_left)
        6. if separation > 180°: separation = 360° - separation
    """

    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32),
                                y.to_numpy(dtype=np.float32)])

    l_hip = _xy("left_hip")
    r_hip = _xy("right_hip")
    l_knee = _xy("left_knee")
    r_knee = _xy("right_knee")

    # Aspect ratio correction
    image_w = landmarks_df.get("image_w")
    image_h = landmarks_df.get("image_h")
    if image_w is not None and image_h is not None:
        aspect = (image_w.to_numpy(dtype=np.float32) / image_h.to_numpy(dtype=np.float32))
    else:
        aspect = np.ones(len(landmarks_df), dtype=np.float32)

    # Pelvis center is the reference point
    pelvis_center = (l_hip + r_hip) / 2.0

    # Compute deltas from pelvis center to each knee
    dx_left = (l_knee[:, 0] - pelvis_center[:, 0]) * aspect
    dy_left = l_knee[:, 1] - pelvis_center[:, 1]
    angle_to_left = np.arctan2(dy_left, dx_left)

    dx_right = (r_knee[:, 0] - pelvis_center[:, 0]) * aspect
    dy_right = r_knee[:, 1] - pelvis_center[:, 1]
    angle_to_right = np.arctan2(dy_right, dx_right)

    # Convert to degrees
    angle_to_left_deg = np.degrees(angle_to_left)
    angle_to_right_deg = np.degrees(angle_to_right)

    # Compute angular separation (always positive)
    separation = np.abs(angle_to_right_deg - angle_to_left_deg)

    # If separation > 180°, take the complementary angle
    separation = np.where(separation > 180, 360 - separation, separation)

    # Mark invalid where inputs are NaN
    invalid = np.isnan(dx_left) | np.isnan(dy_left) | np.isnan(dx_right) | np.isnan(dy_right)
    separation[invalid] = np.nan

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "hip_extension_deg": separation,
        }
    )
    return out


def compute_shoulder_flexion_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    POSE 6 - Shoulder Flexion
    Trunk-relative angle at the shoulder: angle_abc(hip_mid, shoulder, wrist).

    Using hip_midpoint as the proximal trunk reference instead of a global
    vertical axis makes the measurement body-orientation-independent. This
    means PROM (e.g. kneeling/bent-over) and AROM (standing) are measured on
    the same anatomical scale: 0° = arm alongside trunk, 180° = fully raised.
    """
    def _col(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32),
                                y.to_numpy(dtype=np.float32)])

    l_hip = _col("left_hip")
    r_hip = _col("right_hip")
    l_sho = _col("left_shoulder")
    r_sho = _col("right_shoulder")
    l_wri = _col("left_wrist")
    r_wri = _col("right_wrist")

    # Trunk reference: midpoint of both hips
    hip_mid = (l_hip + r_hip) / 2.0

    # angle_abc_deg(A=hip_mid, B=shoulder, C=wrist)
    # → internal angle at B between trunk line and arm line
    left_deg  = angle_abc_deg(hip_mid, l_sho, l_wri)
    right_deg = angle_abc_deg(hip_mid, r_sho, r_wri)

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec":  landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_shoulder_flexion_deg":  left_deg,
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
        norm_dist = shoulder_dist / wrist_dist

    invalid = (wrist_dist <= 0) | np.isnan(wrist_dist) | (shoulder_dist <= 0) | np.isnan(shoulder_dist)
    norm_dist[invalid] = np.nan

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "stick_pass_through_dist_norm": norm_dist,
        }
    )
    return out
