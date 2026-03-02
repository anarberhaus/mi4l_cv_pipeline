"""
summary_metrics.py – Extended summary metrics for the MI4L pipeline.

All functions are pure: they accept numpy arrays / pandas objects and return
scalar values.  No pose-specific logic lives here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pose metadata registry
# ---------------------------------------------------------------------------
POSE_METADATA: dict[str, dict[str, str]] = {
    "kneeling_knee_flexion": {
        "movement_name": "Kneeling Knee Flexion",
        "joint_name": "knee",
        "angle_type": "vector-reference",
    },
    "prone_trunk_extension": {
        "movement_name": "Prone Trunk Extension",
        "joint_name": "trunk",
        "angle_type": "vector-reference",
    },
    "standing_hip_abduction": {
        "movement_name": "Standing Hip Abduction",
        "joint_name": "hip",
        "angle_type": "vector-reference",
    },
    "bilateral_leg_straddle": {
        "movement_name": "Bilateral Leg Straddle",
        "joint_name": "hip",
        "angle_type": "vector-vector",
    },
    "unilateral_hip_extension": {
        "movement_name": "Unilateral Hip Extension",
        "joint_name": "hip",
        "angle_type": "vector-reference",
    },
    "shoulder_flexion": {
        "movement_name": "Shoulder Flexion",
        "joint_name": "shoulder",
        "angle_type": "vector-vector",
    },
    "shoulder_stick_pass_through": {
        "movement_name": "Shoulder Stick Pass-Through",
        "joint_name": "shoulder",
        "angle_type": "distance",
    },
}


def get_pose_metadata(pose_name: str) -> dict[str, str]:
    """Return metadata dict for *pose_name*, with safe fallback."""
    default = {
        "movement_name": pose_name,
        "joint_name": "unknown",
        "angle_type": "unknown",
    }
    return POSE_METADATA.get(pose_name, default)


# ---------------------------------------------------------------------------
# Core derived metric
# ---------------------------------------------------------------------------

def compute_assist_gap(
    arom_peak: float | None,
    prom_peak: float | None,
) -> float | None:
    """prom_peak − arom_peak.  Returns None if either input is missing."""
    if arom_peak is None or prom_peak is None:
        return None
    if not (np.isfinite(arom_peak) and np.isfinite(prom_peak)):
        return None
    return float(prom_peak - arom_peak)


# ---------------------------------------------------------------------------
# End-range quality helpers
# ---------------------------------------------------------------------------

_NEAR_PEAK_FRACTION = 0.02  # 2 % of peak value


def _near_peak_mask(
    angles: np.ndarray,
    peak_val: float,
) -> np.ndarray:
    """Boolean mask: True where angle is within 2 % of *peak_val*."""
    band = abs(peak_val) * _NEAR_PEAK_FRACTION
    # Ensure a minimum band width so very small peaks still work
    band = max(band, 0.5)  # at least 0.5 deg
    return np.abs(angles - peak_val) <= band


def compute_peak_hold_time_s(
    angles: np.ndarray,
    time_sec: np.ndarray,
    peak_val: float,
    valid_mask: np.ndarray,
) -> float | None:
    """Longest consecutive duration inside the near-peak band (seconds)."""
    if peak_val is None or not np.isfinite(peak_val):
        return None
    mask = _near_peak_mask(angles, peak_val) & valid_mask & np.isfinite(angles)
    if not mask.any():
        return None

    # Find contiguous runs of True
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    best_dur = 0.0
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        dur = float(time_sec[min(e - 1, len(time_sec) - 1)] - time_sec[s])
        if dur > best_dur:
            best_dur = dur
    return float(best_dur) if best_dur > 0 else None


def compute_peak_band_std_deg(
    angles: np.ndarray,
    peak_val: float,
    valid_mask: np.ndarray,
) -> float | None:
    """Std-dev of angle values that fall inside the near-peak band."""
    if peak_val is None or not np.isfinite(peak_val):
        return None
    mask = _near_peak_mask(angles, peak_val) & valid_mask & np.isfinite(angles)
    if mask.sum() < 2:
        return None
    return float(np.std(angles[mask], ddof=1))


def compute_time_to_peak_s(
    angles: np.ndarray,
    time_sec: np.ndarray,
    peak_val: float,
    valid_mask: np.ndarray,
) -> float | None:
    """Time from first valid frame to first entry into the near-peak band."""
    if peak_val is None or not np.isfinite(peak_val):
        return None
    mask = _near_peak_mask(angles, peak_val) & valid_mask & np.isfinite(angles)
    valid_any = valid_mask & np.isfinite(angles)
    if not mask.any() or not valid_any.any():
        return None
    first_valid_idx = int(np.where(valid_any)[0][0])
    first_peak_idx = int(np.where(mask)[0][0])
    dt = float(time_sec[first_peak_idx] - time_sec[first_valid_idx])
    return dt if dt >= 0 else None


# ---------------------------------------------------------------------------
# Motor control / signal quality
# ---------------------------------------------------------------------------

def compute_fit_metrics(
    angles: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float | None, float | None]:
    """Fit a degree-3 polynomial to valid angle samples.

    Returns (r2, rmse_deg).  Both None if too few valid points.
    """
    ok = valid_mask & np.isfinite(angles)
    n = int(ok.sum())
    if n < 5:
        return None, None

    x = np.arange(len(angles), dtype=float)[ok]
    y = angles[ok]

    try:
        coeffs = np.polyfit(x, y, deg=min(3, n - 1))
        y_hat = np.polyval(coeffs, x)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r2: float | None
    if ss_tot < 1e-12:
        r2 = 1.0  # constant signal → perfect fit
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    rmse = float(np.sqrt(ss_res / n))
    return r2, rmse


def compute_jerk_rms(
    angles: np.ndarray,
    time_sec: np.ndarray,
    valid_mask: np.ndarray,
) -> float | None:
    """RMS of the second derivative (acceleration) of the angle signal.

    Uses finite differences on the valid, finite portion of the signal.
    """
    ok = valid_mask & np.isfinite(angles)
    if ok.sum() < 4:
        return None

    # Work on contiguous valid data
    idx = np.where(ok)[0]
    t = time_sec[idx]
    a = angles[idx]

    dt = np.diff(t)
    if (dt <= 0).any():
        # Non-monotonic time – fall back to uniform spacing
        dt = np.ones(len(t) - 1)

    # First derivative (angular velocity)
    da = np.diff(a) / dt
    if len(da) < 2:
        return None
    # Second derivative (angular acceleration → jerk proxy)
    dt2 = (dt[:-1] + dt[1:]) / 2.0
    d2a = np.diff(da) / dt2

    rms = float(np.sqrt(np.mean(d2a ** 2)))
    return rms if np.isfinite(rms) else None


# ---------------------------------------------------------------------------
# Compensation metrics
# ---------------------------------------------------------------------------

def _midpoint(
    landmarks_df: pd.DataFrame,
    left_name: str,
    right_name: str,
    coord: str,
) -> pd.Series:
    """Average of left/right landmark coordinate (x or y)."""
    lc = f"{left_name}_{coord}"
    rc = f"{right_name}_{coord}"
    if lc not in landmarks_df.columns or rc not in landmarks_df.columns:
        return pd.Series(np.nan, index=landmarks_df.index)
    return (landmarks_df[lc].astype(float) + landmarks_df[rc].astype(float)) / 2.0


def compute_torso_angle_change_deg(
    landmarks_df: pd.DataFrame,
    frames_used: list[int],
) -> float | None:
    """Change in torso orientation from movement-window start to peak frame.

    Torso vector = shoulder_midpoint → hip_midpoint.
    """
    if not frames_used or len(frames_used) < 2:
        return None

    smx = _midpoint(landmarks_df, "left_shoulder", "right_shoulder", "x")
    smy = _midpoint(landmarks_df, "left_shoulder", "right_shoulder", "y")
    hmx = _midpoint(landmarks_df, "left_hip", "right_hip", "x")
    hmy = _midpoint(landmarks_df, "left_hip", "right_hip", "y")

    sorted_frames = sorted(frames_used)
    start_frame = sorted_frames[0]
    # Peak frame = median of the frames_used cluster
    peak_frame = sorted_frames[len(sorted_frames) // 2]

    def _torso_angle(frame_idx: int) -> float | None:
        rows = landmarks_df[landmarks_df["frame_idx"] == frame_idx]
        if rows.empty:
            return None
        i = rows.index[0]
        dx = float(hmx.loc[i]) - float(smx.loc[i])
        dy = float(hmy.loc[i]) - float(smy.loc[i])
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return None
        return float(np.degrees(np.arctan2(dx, dy)))

    a_start = _torso_angle(start_frame)
    a_peak = _torso_angle(peak_frame)
    if a_start is None or a_peak is None:
        return None
    return float(abs(a_peak - a_start))


def compute_pelvis_drift_norm(
    landmarks_df: pd.DataFrame,
    frames_used: list[int],
) -> float | None:
    """Max horizontal displacement of hip midpoint during movement window,
    normalised by bounding-box width.
    """
    if not frames_used or len(frames_used) < 2:
        return None

    hmx = _midpoint(landmarks_df, "left_hip", "right_hip", "x")

    # Restrict to frames in the movement window
    mask = landmarks_df["frame_idx"].isin(frames_used)
    x_vals = hmx[mask].to_numpy(dtype=float)
    x_vals = x_vals[np.isfinite(x_vals)]
    if len(x_vals) < 2:
        return None

    drift_norm = float(np.max(x_vals) - np.min(x_vals))

    # Normalise by median bbox width (normalised coordinates → width ≈ bbox_w_px/image_w)
    if "bbox_w_px" in landmarks_df.columns and "image_w" in landmarks_df.columns:
        bw = landmarks_df.loc[mask, "bbox_w_px"].astype(float)
        iw = landmarks_df.loc[mask, "image_w"].astype(float)
        norm_bw = (bw / iw).median()
        if np.isfinite(norm_bw) and norm_bw > 0.01:
            drift_norm = drift_norm / float(norm_bw)

    return float(drift_norm) if np.isfinite(drift_norm) else None


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

def compute_frames_valid_pct(valid_mask: np.ndarray) -> float:
    """Percentage of True values in *valid_mask*."""
    if len(valid_mask) == 0:
        return 0.0
    return float(np.mean(valid_mask.astype(float)) * 100.0)


def compute_avg_landmark_visibility(
    landmarks_df: pd.DataFrame,
    frames_used: list[int],
) -> float | None:
    """Mean landmark visibility score over the movement window."""
    if not frames_used:
        return None

    vis_cols = [c for c in landmarks_df.columns if c.endswith("_visibility")]
    if not vis_cols:
        return None

    mask = landmarks_df["frame_idx"].isin(frames_used)
    sub = landmarks_df.loc[mask, vis_cols].astype(float)
    if sub.empty:
        return None

    avg = float(sub.values.mean())
    return avg if np.isfinite(avg) else None


# ---------------------------------------------------------------------------
# Convenience: compute all extended metrics for one summary row
# ---------------------------------------------------------------------------

def compute_extended_summary(
    *,
    pose_name: str,
    side: str,
    arom_peak: float | None,
    prom_peak: float | None,
    angles: np.ndarray,
    time_sec: np.ndarray,
    valid_mask: np.ndarray,
    peak_val: float | None,
    landmarks_df: pd.DataFrame,
    frames_used: list[int],
) -> dict[str, Any]:
    """Return a dict of all extended metric columns for one summary row."""

    # Metadata
    meta = get_pose_metadata(pose_name)
    out: dict[str, Any] = {
        "movement_name": meta["movement_name"],
        "joint_name": meta["joint_name"],
        "side": side,
        "angle_type": meta["angle_type"],
    }

    # Core derived
    out["assist_gap"] = compute_assist_gap(arom_peak, prom_peak)

    # End-range quality
    pv = peak_val if peak_val is not None else arom_peak
    out["peak_hold_time_s"] = compute_peak_hold_time_s(angles, time_sec, pv, valid_mask)
    out["peak_band_std_deg"] = compute_peak_band_std_deg(angles, pv, valid_mask)
    out["time_to_peak_s"] = compute_time_to_peak_s(angles, time_sec, pv, valid_mask)

    # Motor control
    r2, rmse = compute_fit_metrics(angles, valid_mask)
    out["fit_r2"] = r2
    out["fit_rmse_deg"] = rmse
    out["jerk_rms"] = compute_jerk_rms(angles, time_sec, valid_mask)

    # Compensation
    out["torso_angle_change_deg"] = compute_torso_angle_change_deg(landmarks_df, frames_used)
    out["pelvis_drift_norm"] = compute_pelvis_drift_norm(landmarks_df, frames_used)

    # Reliability
    out["frames_valid_pct"] = compute_frames_valid_pct(valid_mask)
    out["avg_landmark_visibility"] = compute_avg_landmark_visibility(landmarks_df, frames_used)

    return out
