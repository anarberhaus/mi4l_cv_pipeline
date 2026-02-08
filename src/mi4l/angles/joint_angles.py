from __future__ import annotations

import numpy as np
import pandas as pd


def angle_abc_deg(a_xy: np.ndarray, b_xy: np.ndarray, c_xy: np.ndarray) -> np.ndarray:
    """
    Angle at point B for triangle A-B-C (in degrees).
    a_xy, b_xy, c_xy: (N,2) arrays.
    Returns (N,) with NaN where undefined.
    """
    ba = a_xy - b_xy
    bc = c_xy - b_xy

    dot = np.einsum("ij,ij->i", ba, bc)
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    denom = ba_norm * bc_norm

    # Avoid divide by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = dot / denom

    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))

    # Mark invalid where denom==0 or any NaN inputs
    invalid = (denom <= 0) | np.isnan(denom) | np.isnan(a_xy).any(axis=1) | np.isnan(b_xy).any(axis=1) | np.isnan(c_xy).any(axis=1)
    ang[invalid] = np.nan
    return ang


def compute_knee_flexion_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Knee flexion is defined as: 180 - internal_knee_angle(hip-knee-ankle)
    Full extension ~ 0 deg; increasing flexion -> larger degrees.
    """
    required = ["frame_idx", "time_sec"]
    for c in required:
        if c not in landmarks_df.columns:
            raise ValueError(f"Missing required column in landmarks_df: {c}")

    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    left_hip = _xy("left_hip")
    left_knee = _xy("left_knee")
    left_ankle = _xy("left_ankle")

    right_hip = _xy("right_hip")
    right_knee = _xy("right_knee")
    right_ankle = _xy("right_ankle")

    left_internal = angle_abc_deg(left_hip, left_knee, left_ankle)
    right_internal = angle_abc_deg(right_hip, right_knee, right_ankle)

    left_flex = 180.0 - left_internal
    right_flex = 180.0 - right_internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_knee_flexion_deg": left_flex.astype(float),
            "right_knee_flexion_deg": right_flex.astype(float),
            "left_knee_internal_deg": left_internal.astype(float),
            "right_knee_internal_deg": right_internal.astype(float),
        }
    )
    return out


def compute_trunk_extension_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Angle defined as: angle(hip_midpoint - shoulder_midpoint - head_or_nose)
    Returns DataFrame with `frame_idx`, `time_sec`, `trunk_extension_deg`.
    """
    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    # hip midpoint
    left_hip = _xy("left_hip")
    right_hip = _xy("right_hip")
    hip_mid = (left_hip + right_hip) / 2.0

    # shoulder midpoint
    left_sh = _xy("left_shoulder")
    right_sh = _xy("right_shoulder")
    sh_mid = (left_sh + right_sh) / 2.0

    # head or nose
    head = _xy("nose")

    trunk_internal = angle_abc_deg(hip_mid, sh_mid, head)
    trunk_ext = 180.0 - trunk_internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "trunk_extension_deg": trunk_ext.astype(float),
        }
    )
    return out


def compute_hip_abduction_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unilateral hip abduction (per side):
    angle(contralateral_hip - ipsilateral_hip - ipsilateral_ankle)
    Returns `left_hip_abduction_deg` and `right_hip_abduction_deg`.
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

    # left abduction: right_hip - left_hip - left_ankle
    left_internal = angle_abc_deg(r_hip, l_hip, l_ank)
    left_abd = 180.0 - left_internal

    # right abduction: left_hip - right_hip - right_ankle
    right_internal = angle_abc_deg(l_hip, r_hip, r_ank)
    right_abd = 180.0 - right_internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_hip_abduction_deg": left_abd.astype(float),
            "right_hip_abduction_deg": right_abd.astype(float),
        }
    )
    return out


def compute_bilateral_leg_straddle_angle(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Angle(pelvis_center): angle(left_ankle - pelvis_center - right_ankle)
    Returns `bilateral_leg_straddle_deg`.
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

    pelvis = (l_hip + r_hip) / 2.0

    internal = angle_abc_deg(l_ank, pelvis, r_ank)
    straddle = 180.0 - internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "bilateral_leg_straddle_deg": straddle.astype(float),
        }
    )
    return out


def compute_hip_extension_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unilateral hip extension (proxy): angle(shoulder_midpoint - hip - knee)
    Returns `left_hip_extension_deg` and `right_hip_extension_deg`.
    """
    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    left_sh = _xy("left_shoulder")
    right_sh = _xy("right_shoulder")
    sh_mid = (left_sh + right_sh) / 2.0

    l_hip = _xy("left_hip")
    r_hip = _xy("right_hip")
    l_knee = _xy("left_knee")
    r_knee = _xy("right_knee")

    left_internal = angle_abc_deg(sh_mid, l_hip, l_knee)
    left_ext = 180.0 - left_internal

    right_internal = angle_abc_deg(sh_mid, r_hip, r_knee)
    right_ext = 180.0 - right_internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_hip_extension_deg": left_ext.astype(float),
            "right_hip_extension_deg": right_ext.astype(float),
        }
    )
    return out


def compute_shoulder_flexion_angles(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shoulder flexion (per side): angle(torso_reference - shoulder - wrist)
    Here torso_reference is the hip midpoint.
    Returns `left_shoulder_flexion_deg` and `right_shoulder_flexion_deg`.
    """
    def _xy(prefix: str) -> np.ndarray:
        x = landmarks_df.get(f"{prefix}_x")
        y = landmarks_df.get(f"{prefix}_y")
        if x is None or y is None:
            return np.full((len(landmarks_df), 2), np.nan, dtype=np.float32)
        return np.column_stack([x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)])

    l_hip = _xy("left_hip")
    r_hip = _xy("right_hip")
    hip_mid = (l_hip + r_hip) / 2.0

    l_sh = _xy("left_shoulder")
    r_sh = _xy("right_shoulder")
    l_w = _xy("left_wrist")
    r_w = _xy("right_wrist")

    left_internal = angle_abc_deg(hip_mid, l_sh, l_w)
    left_flex = 180.0 - left_internal

    right_internal = angle_abc_deg(hip_mid, r_sh, r_w)
    right_flex = 180.0 - right_internal

    out = pd.DataFrame(
        {
            "frame_idx": landmarks_df["frame_idx"].to_numpy(dtype=int),
            "time_sec": landmarks_df["time_sec"].to_numpy(dtype=float),
            "left_shoulder_flexion_deg": left_flex.astype(float),
            "right_shoulder_flexion_deg": right_flex.astype(float),
        }
    )
    return out
