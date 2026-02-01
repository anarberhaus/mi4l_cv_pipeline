from __future__ import annotations

from dataclasses import dataclass
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

    v = values[ok]
    # Take top-K by value
    order = np.argsort(v)[::-1]
    v_top = v[order[:k]]
    est = float(np.nanmedian(v_top)) if v_top.size > 0 else float("nan")

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
    )
