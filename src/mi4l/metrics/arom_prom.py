from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustMaxResult:
    value_deg: float | None
    confidence: float
    n_total: int
    n_valid: int
    n_used: int
    flags: list[str]
    frames_used: list[int] = field(default_factory=list)


def smooth_angle_series(angle_deg: pd.Series, smoothing_cfg: dict[str, Any]) -> pd.Series:
    method = str(smoothing_cfg.get("method", "none")).lower().strip()
    interpolate_limit = smoothing_cfg.get("interpolate_limit", 0)
    try:
        interpolate_limit_int = int(interpolate_limit) if interpolate_limit is not None else 0
    except Exception:
        interpolate_limit_int = 0

    s = angle_deg.astype(float).copy()

    if interpolate_limit_int > 0:
        s = s.interpolate(limit=interpolate_limit_int, limit_direction="both")

    if method in ("none", "", "off"):
        return s

    if method == "median":
        w = int(smoothing_cfg.get("median_window", 7))
        w = max(1, w)
        if w % 2 == 0:
            w += 1
        return s.rolling(window=w, center=True, min_periods=max(1, w // 2)).median()

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter  # type: ignore
        except Exception as e:
            raise ImportError("SciPy is required for Savitzky–Golay smoothing (method: savgol).") from e

        w = int(smoothing_cfg.get("savgol_window", 11))
        w = max(3, w)
        if w % 2 == 0:
            w += 1
        p = int(smoothing_cfg.get("savgol_polyorder", 3))
        p = max(1, min(p, w - 1))

        x = s.to_numpy(dtype=float)
        nan_mask = np.isnan(x)
        # Fill NaNs for filtering, then restore NaNs
        if nan_mask.any():
            x_filled = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
        else:
            x_filled = x
        y = savgol_filter(x_filled, window_length=w, polyorder=p, mode="interp").astype(float)
        y[nan_mask] = np.nan
        return pd.Series(y, index=s.index)

    raise ValueError(f"Unknown smoothing method: {method}")


def estimate_robust_max(
    angle_deg: pd.Series,
    valid_mask: pd.Series,
    smoothing_cfg: dict[str, Any],
    robust_cfg: dict[str, Any],
    qc_cfg: dict[str, Any],
    direction: str = "max",  # "max" or "min"
) -> RobustMaxResult:
    flags: list[str] = []

    n_total = int(len(angle_deg))
    if n_total == 0:
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=0, n_valid=0, n_used=0, flags=["empty_series"])

    smoothed = smooth_angle_series(angle_deg, smoothing_cfg=smoothing_cfg)

    vm = valid_mask.fillna(False).to_numpy(dtype=bool)
    values = smoothed.to_numpy(dtype=float)
    ok = vm & np.isfinite(values)

    n_valid = int(ok.sum())
    min_valid_frames = int(qc_cfg.get("min_valid_frames", 1))
    min_valid_ratio = float(qc_cfg.get("min_valid_ratio", 0.0))

    if n_valid < min_valid_frames:
        flags.append(f"too_few_valid_frames:{n_valid}<{min_valid_frames}")

    valid_ratio = n_valid / n_total if n_total > 0 else 0.0
    if min_valid_ratio > 0 and valid_ratio < min_valid_ratio:
        flags.append(f"low_valid_ratio:{valid_ratio:.3f}<{min_valid_ratio:.3f}")

    if n_valid == 0:
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=0, flags=flags + ["no_valid_values"])

    topk_percent = float(robust_cfg.get("topk_percent", 0.05))
    topk_percent = min(max(topk_percent, 0.0), 1.0)
    min_topk_frames = int(robust_cfg.get("min_topk_frames", 5))
    min_topk_frames = max(1, min_topk_frames)

    k = int(np.ceil(topk_percent * n_valid))
    k = max(min_topk_frames, k)
    k = min(n_valid, k)

    # Filter out short spikes if min_hold_frames is set
    min_hold_frames = int(robust_cfg.get("min_hold_frames", 0))
    if min_hold_frames > 1:
        # We need to filter 'ok' to only include runs of True that are >= min_hold_frames
        # This is a simple erosion/opening-like operation on the boolean mask
        # but we need to respect the original index gaps?
        # Simpler: just look at the boolean array 'ok'.
        # If we have [T, T, F, T, T, T, F], and min_hold=3, we keep only the 3 Ts.
        
        # Identify runs
        padded = np.concatenate(([False], ok, [False]))
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        mask_filtered = np.zeros_like(ok, dtype=bool)
        for s, e in zip(starts, ends):
            if (e - s) >= min_hold_frames:
                mask_filtered[s:e] = True
        
        # Update ok mask
        ok = mask_filtered
        # Re-calc n_valid
        n_valid = int(ok.sum())
        if n_valid == 0:
             return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=0, n_used=0, flags=flags + ["no_valid_hold"])
        
        # Re-calc k based on new n_valid? 
        # Usually we want k to be based on the *filtered* valid count to pick the peak of the hold.
        k = int(np.ceil(topk_percent * n_valid))
        k = max(min_topk_frames, k)
        k = min(n_valid, k)

    v = values[ok]
    # original indices of the valid entries (relative to the input series index)
    valid_idx = np.nonzero(ok)[0]
    # Take top-K by value (ascending for min, descending for max)
    if direction == "min":
        order = np.argsort(v)  # ascending: smallest first
    else:
        order = np.argsort(v)[::-1]  # descending: largest first
    top_order = order[:k]
    v_top = v[top_order]
    est = float(np.nanmedian(v_top)) if v_top.size > 0 else float("nan")

    # Map back to original series index labels for the frames used
    frames_used = []
    try:
        idx_labels = angle_deg.index.to_numpy()
        frames_used = [int(idx_labels[valid_idx[i]]) for i in top_order]
    except Exception:
        frames_used = [int(valid_idx[i]) for i in top_order]

    if not np.isfinite(est):
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=k, flags=flags + ["est_nan"])

    # Confidence (simple, bounded): depends on validity and topk support
    topk_factor = min(1.0, k / float(min_topk_frames))
    valid_factor = 1.0
    if min_valid_ratio > 0:
        valid_factor = min(1.0, valid_ratio / float(min_valid_ratio))
    confidence = float(np.clip(topk_factor * valid_factor, 0.0, 1.0))

    if k < min_topk_frames:
        flags.append(f"topk_insufficient:{k}<{min_topk_frames}")

    return RobustMaxResult(
        value_deg=est,
        confidence=confidence,
        n_total=n_total,
        n_valid=n_valid,
        n_used=k,
        flags=flags,
        frames_used=frames_used,
    )


def estimate_stable_plateau(
    series: pd.Series,
    valid_mask: pd.Series,
    smoothing_cfg: dict[str, Any],
    plateau_cfg: dict[str, Any],
    qc_cfg: dict[str, Any],
) -> RobustMaxResult:
    """
    Find the most stable plateau in `series` that is also above a minimum value
    threshold. Intended for metrics like stick pass-through where the key value
    is a stable holding period (e.g. the starting grip width), not the global
    max or min.

    Algorithm:
      1. Smooth the series.
      2. Compute a rolling std over `stability_window_sec` seconds.
      3. Mark frames as "stable" where rolling_std < stability_threshold AND
         value > min_value_threshold AND valid_mask is True.
      4. Find the longest contiguous run of stable frames.
      5. Return the median of that run as the estimate.
    """
    flags: list[str] = []

    n_total = int(len(series))
    if n_total == 0:
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=0, n_valid=0, n_used=0, flags=["empty_series"])

    # --- Config ---
    stability_window_sec: float = float(plateau_cfg.get("stability_window_sec", 2.0))
    stability_threshold: float = float(plateau_cfg.get("stability_threshold", 0.15))
    min_value_threshold: float = float(plateau_cfg.get("min_value_threshold", 1.2))
    min_plateau_frames: int = int(plateau_cfg.get("min_plateau_frames", 10))

    # --- 1. Smooth ---
    smoothed = smooth_angle_series(series, smoothing_cfg=smoothing_cfg)

    vm = valid_mask.fillna(False).to_numpy(dtype=bool)
    values = smoothed.to_numpy(dtype=float)

    n_valid = int((vm & np.isfinite(values)).sum())
    min_valid_frames = int(qc_cfg.get("min_valid_frames", 1))
    if n_valid < min_valid_frames:
        flags.append(f"too_few_valid_frames:{n_valid}<{min_valid_frames}")

    if n_valid == 0:
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=0, flags=flags + ["no_valid_values"])

    # --- 2. Rolling std ---
    # Estimate fps from time column if available, else assume 30fps
    # We work in frame-space: convert stability_window_sec to frames
    # Use a simple heuristic: window = max(3, stability_window_sec * estimated_fps)
    # We can't know fps here, so we use a frame count directly from config
    stability_window_frames: int = int(plateau_cfg.get("stability_window_frames", 30))

    smoothed_series = pd.Series(values)
    rolling_std = smoothed_series.rolling(
        window=max(3, stability_window_frames),
        center=True,
        min_periods=max(1, stability_window_frames // 2),
    ).std().to_numpy(dtype=float)

    # --- 3. Stability + threshold mask ---
    stable = (
        vm
        & np.isfinite(values)
        & np.isfinite(rolling_std)
        & (rolling_std < stability_threshold)
        & (values > min_value_threshold)
    )

    # --- 4. Find longest contiguous run ---
    # Pad with False to detect edges
    padded = np.concatenate(([False], stable, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        flags.append("no_stable_plateau_found")
        # Fallback: use all valid frames above threshold
        fallback_ok = vm & np.isfinite(values) & (values > min_value_threshold)
        if fallback_ok.sum() == 0:
            return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=0, flags=flags + ["no_values_above_threshold"])
        fallback_vals = values[fallback_ok]
        est = float(np.nanmedian(fallback_vals))
        fallback_idx = np.nonzero(fallback_ok)[0]
        idx_labels = series.index.to_numpy()
        frames_used = [int(idx_labels[i]) for i in fallback_idx]
        confidence = 0.3  # low confidence for fallback
        return RobustMaxResult(value_deg=est, confidence=confidence, n_total=n_total, n_valid=n_valid, n_used=len(fallback_idx), flags=flags, frames_used=frames_used)

    # Pick the longest run
    run_lengths = ends - starts
    best_idx = int(np.argmax(run_lengths))
    best_start = starts[best_idx]
    best_end = ends[best_idx]
    best_run = np.arange(best_start, best_end)

    if len(best_run) < min_plateau_frames:
        flags.append(f"plateau_too_short:{len(best_run)}<{min_plateau_frames}")

    # --- 5. Median of the best run ---
    run_values = values[best_run]
    est = float(np.nanmedian(run_values))

    if not np.isfinite(est):
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=len(best_run), flags=flags + ["est_nan"])

    # Map back to original index labels
    idx_labels = series.index.to_numpy()
    try:
        frames_used = [int(idx_labels[i]) for i in best_run]
    except Exception:
        frames_used = [int(i) for i in best_run]

    # Confidence: based on plateau length relative to min_plateau_frames
    plateau_factor = min(1.0, len(best_run) / max(1, min_plateau_frames))
    valid_ratio = n_valid / n_total if n_total > 0 else 0.0
    min_valid_ratio = float(qc_cfg.get("min_valid_ratio", 0.0))
    valid_factor = min(1.0, valid_ratio / float(min_valid_ratio)) if min_valid_ratio > 0 else 1.0
    confidence = float(np.clip(plateau_factor * valid_factor, 0.0, 1.0))

    return RobustMaxResult(
        value_deg=est,
        confidence=confidence,
        n_total=n_total,
        n_valid=n_valid,
        n_used=len(best_run),
        flags=flags,
        frames_used=frames_used,
    )
