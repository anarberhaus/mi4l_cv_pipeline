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
