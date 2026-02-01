from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mi4l.metrics.arom_prom import smooth_angle_series


def compute_knee_visibility_qc(landmarks_df: pd.DataFrame, visibility_threshold: float) -> pd.DataFrame:
    def _vis(name: str) -> pd.Series:
        col = f"{name}_visibility"
        if col not in landmarks_df.columns:
            return pd.Series(np.nan, index=landmarks_df.index)
        return landmarks_df[col].astype(float)

    pose_detected = landmarks_df.get("pose_detected")
    if pose_detected is None:
        pose_detected = pd.Series(True, index=landmarks_df.index)

    lmin = pd.concat([_vis("left_hip"), _vis("left_knee"), _vis("left_ankle")], axis=1).min(axis=1, skipna=False)
    rmin = pd.concat([_vis("right_hip"), _vis("right_knee"), _vis("right_ankle")], axis=1).min(axis=1, skipna=False)

    left_valid = pose_detected.fillna(False) & (lmin >= float(visibility_threshold))
    right_valid = pose_detected.fillna(False) & (rmin >= float(visibility_threshold))

    return pd.DataFrame(
        {
            "left_knee_vis_min": lmin.astype(float),
            "right_knee_vis_min": rmin.astype(float),
            "left_knee_valid": left_valid.astype(bool),
            "right_knee_valid": right_valid.astype(bool),
        }
    )


def compute_subject_size_qc(
    landmarks_df: pd.DataFrame,
    min_bbox_height_px: int,
    require_subject_size: bool,
) -> pd.DataFrame:
    bbox_h_px = landmarks_df.get("bbox_h_px")
    if bbox_h_px is None:
        bbox_h_px = pd.Series(np.nan, index=landmarks_df.index)

    if not require_subject_size or min_bbox_height_px <= 0:
        ok = pd.Series(True, index=landmarks_df.index)
    else:
        ok = bbox_h_px.astype(float) >= float(min_bbox_height_px)

    return pd.DataFrame(
        {
            "bbox_h_px": bbox_h_px.astype(float),
            "subject_size_ok": ok.astype(bool),
        }
    )


def apply_derivative_qc(time_sec: pd.Series, angle_deg: pd.Series, max_deg_per_sec: float) -> pd.Series:
    """
    Marks frames as OK if local angular velocity <= max_deg_per_sec.
    If max_deg_per_sec <= 0, returns all True.
    """
    if max_deg_per_sec <= 0:
        return pd.Series(True, index=angle_deg.index)

    t = time_sec.to_numpy(dtype=float)
    a = angle_deg.to_numpy(dtype=float)

    # Handle NaNs: derivative QC should not make NaNs "valid"
    ok = np.ones_like(a, dtype=bool)
    ok[np.isnan(a)] = False

    if len(a) < 3:
        return pd.Series(ok, index=angle_deg.index)

    dt = np.diff(t)
    da = np.diff(a)

    with np.errstate(invalid="ignore", divide="ignore"):
        vel = np.abs(da / dt)

    # vel length N-1; mark both adjacent samples when a jump occurs
    bad_edges = np.zeros_like(a, dtype=bool)
    bad = vel > float(max_deg_per_sec)
    bad_idx = np.where(bad)[0]
    bad_edges[bad_idx] = True
    bad_edges[bad_idx + 1] = True

    ok = ok & (~bad_edges)
    return pd.Series(ok, index=angle_deg.index)


def compute_video_level_qc_flags(landmarks_df: pd.DataFrame, qc_cfg: dict[str, Any]) -> list[str]:
    flags: list[str] = []

    n = int(len(landmarks_df))
    if n <= 0:
        return ["empty_video"]

    is_clipped = landmarks_df.get("is_clipped")
    if is_clipped is not None:
        clipped_ratio = float(is_clipped.fillna(False).mean())
        max_clipped_ratio = float(qc_cfg.get("max_clipped_ratio", 1.0))
        if clipped_ratio > max_clipped_ratio:
            flags.append(f"too_many_clipped_frames:{clipped_ratio:.3f}")

    if bool(qc_cfg.get("require_subject_size", False)):
        bbox_h_px = landmarks_df.get("bbox_h_px")
        if bbox_h_px is not None:
            min_bbox_height_px = float(qc_cfg.get("min_bbox_height_px", 0))
            # robust summary: median bbox height
            med = float(np.nanmedian(bbox_h_px.to_numpy(dtype=float))) if n > 0 else float("nan")
            if np.isfinite(min_bbox_height_px) and min_bbox_height_px > 0 and (not np.isfinite(med) or med < min_bbox_height_px):
                flags.append(f"subject_too_small:median_bbox_h_px={med:.1f}")

    return flags
