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


def trim_valid_mask_to_movement_window(
    series: pd.Series,
    valid_mask: pd.Series,
    cfg: "dict[str, Any]",
    smoothing_cfg: "dict[str, Any]",
    direction: str = "max",
) -> pd.Series:
    """
    Pre-filter: restrict valid_mask to the window of active movement by
    detecting onset and end via the derivative of the smoothed angle series.

    Algorithm
    ---------
    1. Smooth the series.
    2. Compute frame-to-frame derivative; smooth it with a rolling window.
    3. Find the candidate peak = argmax (direction="max") or argmin
       (direction="min") of the smoothed valid values.
    4. Scan BACKWARD from the peak: find the last frame whose |derivative|
       exceeds slope_threshold → that is the movement onset.
    5. Scan FORWARD from the peak: find the first frame after the peak whose
       |derivative| exceeds slope_threshold (the return phase) → movement end.
    6. Apply buffer_frames inward on both sides.
    7. Return a copy of valid_mask with False outside [onset, end].

    If onset/end cannot be determined the original valid_mask is returned
    unchanged (conservative fallback – no frames are incorrectly removed).
    """
    slope_threshold: float = float(cfg.get("slope_threshold", 2.0))
    buffer: int = int(cfg.get("buffer_frames", 5))

    smoothed = smooth_angle_series(series, smoothing_cfg=smoothing_cfg)
    vm = valid_mask.fillna(False).to_numpy(dtype=bool)
    values = smoothed.to_numpy(dtype=float)
    n = len(values)

    if n < 3:
        return valid_mask

    ok = vm & np.isfinite(values)
    if ok.sum() == 0:
        return valid_mask

    # Derivative + light smoothing
    deriv = np.zeros(n, dtype=float)
    deriv[:-1] = np.diff(values)
    deriv[~ok] = 0.0
    deriv_s = pd.Series(deriv).rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    # Peak frame
    valid_vals = np.where(ok, values, np.nan)
    if direction == "min":
        peak_idx = int(np.nanargmin(valid_vals))
    else:
        peak_idx = int(np.nanargmax(valid_vals))

    # Onset: last frame BEFORE peak where |deriv| >= slope_threshold
    pre_peak = np.abs(deriv_s[:peak_idx])
    strong_pre = np.where(pre_peak >= slope_threshold)[0]
    onset = int(strong_pre[0]) if strong_pre.size > 0 else 0

    # End: first frame AFTER peak where |deriv| >= slope_threshold (return phase)
    post_peak = np.abs(deriv_s[peak_idx:])
    strong_post = np.where(post_peak >= slope_threshold)[0]
    # Skip the peak frame itself (index 0 in the slice)
    valid_post = strong_post[strong_post > 0]
    end = (peak_idx + int(valid_post[0])) if valid_post.size > 0 else (n - 1)

    # Apply inward buffer
    onset = min(onset + buffer, peak_idx)
    end = max(end - buffer, peak_idx)

    if end <= onset:
        return valid_mask

    new_mask_arr = valid_mask.to_numpy(dtype=bool).copy()
    new_mask_arr[:onset] = False
    new_mask_arr[end + 1:] = False
    return pd.Series(new_mask_arr, index=valid_mask.index)


def trim_valid_mask_by_rolling_std(
    series: pd.Series,
    valid_mask: pd.Series,
    cfg: "dict[str, Any]",
) -> pd.Series:
    """
    Pre-filter: exclude frames where the local angle variability is too high.

    A rolling standard deviation is computed over the raw angle series.
    Any frame whose rolling_std exceeds max_rolling_std is masked out.
    This removes artifact / prep-phase frames (e.g. wild spikes before
    the participant settles into position) so that downstream peak
    detection anchors to the actual smooth movement.

    Typical calibration (30 fps):
      - Noisy prep / artifact region: rolling std >> 20 deg  -> excluded
      - Smooth movement plateau:       rolling std <  10 deg  -> kept
    """
    window: int = int(cfg.get("rolling_window_frames", 15))
    max_std: float = float(cfg.get("max_rolling_std", 20.0))

    rolling_std = (
        series.rolling(window=window, center=True, min_periods=1)
        .std()
        .fillna(float("inf"))
    )
    stable_mask = rolling_std <= max_std
    return valid_mask & stable_mask


def estimate_movement_window_fallback(
    values: np.ndarray,
    ok: np.ndarray,
    peak_cfg: "dict[str, Any]",
    n_total: int,
    n_valid_original: int,
    flags_in: "list[str]",
    angle_index: "Any",
    direction: str = "max",
) -> "RobustMaxResult | None":
    """
    Fallback peak estimator for AROM movements where the peak is never *held*
    for enough consecutive frames to pass the primary method.

    Algorithm
    ---------
    1. Compute frame-to-frame derivative of the smoothed angle series.
    2. Smooth the derivative (rolling mean, window=3) to suppress noise.
    3. Find movement start  = first frame where derivative >= +slope_threshold.
    4. Find movement end   = first frame *after* the peak where derivative <= -slope_threshold.
    5. Apply movement_buffer_frames inward on both sides.
    6. Within the trimmed window collect valid values, take top fallback_top_percent.
    7. Return median of those values with confidence=0.5.

    Returns None if the window cannot be determined (fallback inconclusive).
    """
    slope_threshold: float = float(peak_cfg.get("slope_threshold", 3.0))
    buffer: int = int(peak_cfg.get("movement_buffer_frames", 5))
    top_pct: float = float(peak_cfg.get("fallback_top_percent", 0.20))
    top_pct = min(max(top_pct, 0.01), 1.0)

    n = len(values)
    if n < 3:
        return None

    # 1. Derivative (forward difference, padded with 0 at end)
    deriv = np.zeros(n, dtype=float)
    deriv[:-1] = np.diff(values)
    # Zero out derivative at invalid frames so slope detection ignores them
    deriv[~ok] = 0.0

    # 2. Smooth derivative (rolling mean, window=3)
    deriv_s = pd.Series(deriv).rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    # 3. Peak location: argmax/argmin of value within valid frames
    valid_vals = np.where(ok, values, np.nan)
    if direction == "min":
        peak_idx = int(np.nanargmin(valid_vals))
    else:
        peak_idx = int(np.nanargmax(valid_vals))

    # 4a. Movement start: last strong positive slope BEFORE peak
    search_start = deriv_s[:peak_idx]
    pos_mask = search_start >= slope_threshold
    if not pos_mask.any():
        # No strong rise found – fallback inconclusive
        return None
    win_start = int(np.where(pos_mask)[0][0])   # first onset of rise

    # 4b. Movement end: first strong negative slope AFTER peak
    search_end = deriv_s[peak_idx:]
    if direction == "min":
        neg_mask = search_end >= slope_threshold   # recovery is positive for min direction
    else:
        neg_mask = search_end <= -slope_threshold
    if not neg_mask.any():
        # No strong drop found after peak – use end of series
        win_end = n - 1
    else:
        win_end = peak_idx + int(np.where(neg_mask)[0][0])

    # 5. Apply buffer
    win_start = win_start + buffer
    win_end = win_end - buffer

    # 6. Validate window
    if win_end - win_start < 3:
        return None

    window_ok = ok.copy()
    window_ok[:win_start] = False
    window_ok[win_end + 1:] = False

    window_vals = values[window_ok]
    if window_vals.size == 0:
        return None

    # 7. Top top_pct% by value
    k = max(1, int(np.ceil(top_pct * window_vals.size)))
    k = min(k, window_vals.size)
    if direction == "min":
        top_vals = np.sort(window_vals)[:k]
    else:
        top_vals = np.sort(window_vals)[-k:]

    est = float(np.nanmedian(top_vals))
    if not np.isfinite(est):
        return None

    # Map back to original index labels for frames_used
    window_idx = np.nonzero(window_ok)[0]
    # Only include the top-k frames in frames_used
    if direction == "min":
        top_positions = np.argsort(window_vals)[:k]
    else:
        top_positions = np.argsort(window_vals)[-k:]
    try:
        idx_labels = np.asarray(angle_index)
        frames_used = [int(idx_labels[window_idx[p]]) for p in top_positions]
    except Exception:
        frames_used = [int(window_idx[p]) for p in top_positions]

    n_window_valid = int(window_ok.sum())
    confidence = 0.5  # reduced confidence for fallback path

    return RobustMaxResult(
        value_deg=est,
        confidence=confidence,
        n_total=n_total,
        n_valid=n_valid_original,
        n_used=k,
        flags=list(flags_in) + ["fallback_movement_window"],
        frames_used=frames_used,
    )


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
            # ----------------------------------------------------------------
            # Fallback: movement-window strategy (AROM short-hold case)
            # ----------------------------------------------------------------
            peak_cfg = robust_cfg.get("_peak_detection", {})
            if bool(peak_cfg.get("fallback_enabled", False)):
                print(
                    "WARNING: Primary peak detection failed – using movement-window fallback"
                )
                ok_original = (vm & np.isfinite(values))
                fb = estimate_movement_window_fallback(
                    values=values,
                    ok=ok_original,
                    peak_cfg=peak_cfg,
                    n_total=n_total,
                    n_valid_original=int(ok_original.sum()),
                    flags_in=flags + ["no_valid_hold"],
                    angle_index=angle_deg.index,
                    direction=direction,
                )
                if fb is not None:
                    return fb

            # ----------------------------------------------------------------
            # Last resort: top-20% of ALL valid frames, no movement window
            # required.  Confidence=0.25 signals this is the weakest path.
            # ----------------------------------------------------------------
            ok_lr = vm & np.isfinite(values)
            if ok_lr.sum() > 0:
                top_pct = float(peak_cfg.get("fallback_top_percent", 0.20)) if peak_cfg else 0.20
                lr_vals = values[ok_lr]
                k_lr = max(1, int(np.ceil(top_pct * lr_vals.size)))
                k_lr = min(k_lr, lr_vals.size)
                if direction == "min":
                    top_lr = np.sort(lr_vals)[:k_lr]
                else:
                    top_lr = np.sort(lr_vals)[-k_lr:]
                est_lr = float(np.nanmedian(top_lr))
                if np.isfinite(est_lr):
                    return RobustMaxResult(
                        value_deg=est_lr,
                        confidence=0.25,
                        n_total=n_total,
                        n_valid=int(ok_lr.sum()),
                        n_used=k_lr,
                        flags=flags + ["no_valid_hold", "last_resort_fallback"],
                    )

            return RobustMaxResult(
                value_deg=None,
                confidence=0.0,
                n_total=n_total,
                n_valid=0,
                n_used=0,
                flags=flags + ["no_valid_hold"],
            )
        
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
    direction: str = "max",
) -> RobustMaxResult:
    """
    Find the most stable plateau in `series` within a value range determined by
    `direction`.

    direction="max" (default): looks for plateaus above `min_value_threshold`.
        Used when the metric is highest during the active movement (e.g. old
        wrist/shoulder ratio).
    direction="min": looks for plateaus below `max_value_threshold`. Used when
        the metric is lowest during the active movement (e.g. shoulder/wrist
        ratio, where a narrower grip = higher score but the exercise period
        produces the lowest values).

    Algorithm:
      1. Smooth the series.
      2. Compute a rolling std over `stability_window_frames` frames.
      3. Mark frames as "stable" where rolling_std < stability_threshold AND
         value is within the direction-appropriate range AND valid_mask is True.
      4. Find the longest contiguous run of stable frames.
      5. Return the median of that run as the estimate.
    """
    flags: list[str] = []

    n_total = int(len(series))
    if n_total == 0:
        return RobustMaxResult(value_deg=None, confidence=0.0, n_total=0, n_valid=0, n_used=0, flags=["empty_series"])

    # --- Config ---
    stability_threshold: float = float(plateau_cfg.get("stability_threshold", 0.15))
    min_plateau_frames: int = int(plateau_cfg.get("min_plateau_frames", 10))

    if direction == "min":
        max_value_threshold: float = float(plateau_cfg.get("max_value_threshold", 0.9))
        value_filter_label = f"no_values_below_threshold:{max_value_threshold}"
    else:
        min_value_threshold: float = float(plateau_cfg.get("min_value_threshold", 1.2))
        value_filter_label = f"no_values_above_threshold:{min_value_threshold}"

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
    stability_window_frames: int = int(plateau_cfg.get("stability_window_frames", 30))

    smoothed_series = pd.Series(values)
    rolling_std = smoothed_series.rolling(
        window=max(3, stability_window_frames),
        center=True,
        min_periods=max(1, stability_window_frames // 2),
    ).std().to_numpy(dtype=float)

    # --- 3. Stability + threshold mask ---
    if direction == "min":
        value_ok = values < max_value_threshold
    else:
        value_ok = values > min_value_threshold

    stable = (
        vm
        & np.isfinite(values)
        & np.isfinite(rolling_std)
        & (rolling_std < stability_threshold)
        & value_ok
    )

    # --- 4. Find longest contiguous run ---
    # Pad with False to detect edges
    padded = np.concatenate(([False], stable, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        flags.append("no_stable_plateau_found")
        fallback_ok = vm & np.isfinite(values) & value_ok
        if fallback_ok.sum() == 0:
            return RobustMaxResult(value_deg=None, confidence=0.0, n_total=n_total, n_valid=n_valid, n_used=0, flags=flags + [value_filter_label])
        fallback_vals = values[fallback_ok]
        est = float(np.nanmedian(fallback_vals))
        fallback_idx = np.nonzero(fallback_ok)[0]
        idx_labels = series.index.to_numpy()
        frames_used = [int(idx_labels[i]) for i in fallback_idx]
        confidence = 0.3
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
