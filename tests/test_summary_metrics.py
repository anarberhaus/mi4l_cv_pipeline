"""Unit tests for mi4l.metrics.summary_metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mi4l.metrics.summary_metrics import (
    compute_assist_gap,
    compute_avg_landmark_visibility,
    compute_extended_summary,
    compute_fit_metrics,
    compute_frames_valid_pct,
    compute_jerk_rms,
    compute_peak_band_std_deg,
    compute_peak_hold_time_s,
    compute_pelvis_drift_norm,
    compute_time_to_peak_s,
    compute_torso_angle_change_deg,
    get_pose_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_landmarks_df(n: int = 100) -> pd.DataFrame:
    """Minimal landmarks DataFrame for testing compensation/reliability."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "left_shoulder_x": 0.4 + rng.normal(0, 0.001, n),
        "left_shoulder_y": 0.3 + rng.normal(0, 0.001, n),
        "right_shoulder_x": 0.6 + rng.normal(0, 0.001, n),
        "right_shoulder_y": 0.3 + rng.normal(0, 0.001, n),
        "left_hip_x": 0.45 + rng.normal(0, 0.001, n),
        "left_hip_y": 0.6 + rng.normal(0, 0.001, n),
        "right_hip_x": 0.55 + rng.normal(0, 0.001, n),
        "right_hip_y": 0.6 + rng.normal(0, 0.001, n),
        "left_shoulder_visibility": rng.uniform(0.8, 1.0, n),
        "right_shoulder_visibility": rng.uniform(0.8, 1.0, n),
        "left_hip_visibility": rng.uniform(0.7, 1.0, n),
        "right_hip_visibility": rng.uniform(0.7, 1.0, n),
        "bbox_w_px": np.full(n, 200.0),
        "image_w": np.full(n, 1920.0),
    })
    return df


# ---------------------------------------------------------------------------
# Pose metadata
# ---------------------------------------------------------------------------

class TestPoseMetadata:
    def test_known_pose(self):
        m = get_pose_metadata("kneeling_knee_flexion")
        assert m["movement_name"] == "Kneeling Knee Flexion"
        assert m["joint_name"] == "knee"
        assert m["angle_type"] == "vector-reference"

    def test_unknown_pose_falls_back(self):
        m = get_pose_metadata("nonexistent_pose")
        assert m["movement_name"] == "nonexistent_pose"
        assert m["joint_name"] == "unknown"


# ---------------------------------------------------------------------------
# Assist gap
# ---------------------------------------------------------------------------

class TestAssistGap:
    def test_normal(self):
        assert compute_assist_gap(60.0, 80.0) == pytest.approx(20.0)

    def test_negative_gap(self):
        assert compute_assist_gap(80.0, 60.0) == pytest.approx(-20.0)

    def test_none_inputs(self):
        assert compute_assist_gap(None, 80.0) is None
        assert compute_assist_gap(60.0, None) is None
        assert compute_assist_gap(None, None) is None

    def test_nan_input(self):
        assert compute_assist_gap(float("nan"), 80.0) is None


# ---------------------------------------------------------------------------
# End-range quality
# ---------------------------------------------------------------------------

class TestPeakHoldTime:
    def test_constant_signal(self):
        """Constant signal at peak → entire duration is hold time."""
        n = 100
        angles = np.full(n, 90.0)
        time_sec = np.linspace(0, 3.3, n)
        valid = np.ones(n, dtype=bool)
        result = compute_peak_hold_time_s(angles, time_sec, 90.0, valid)
        assert result is not None
        assert result == pytest.approx(3.3, abs=0.1)

    def test_no_peak_frames(self):
        """Signal far from peak → None."""
        angles = np.full(50, 10.0)
        time_sec = np.linspace(0, 1.0, 50)
        valid = np.ones(50, dtype=bool)
        result = compute_peak_hold_time_s(angles, time_sec, 90.0, valid)
        assert result is None

    def test_none_peak(self):
        assert compute_peak_hold_time_s(np.array([1.0]), np.array([0.0]), None, np.array([True])) is None


class TestPeakBandStd:
    def test_constant_in_band(self):
        """Constant signal → std ≈ 0 (but None since ddof=1 with identical values needs >1)."""
        n = 50
        angles = np.full(n, 100.0)
        valid = np.ones(n, dtype=bool)
        result = compute_peak_band_std_deg(angles, 100.0, valid)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_too_few_frames(self):
        angles = np.array([100.0])
        valid = np.array([True])
        assert compute_peak_band_std_deg(angles, 100.0, valid) is None


class TestTimeToPeak:
    def test_linear_ramp(self):
        """Linear ramp from 0 to 100 → time to peak band should be near the end."""
        n = 100
        angles = np.linspace(0, 100, n)
        time_sec = np.linspace(0, 10.0, n)
        valid = np.ones(n, dtype=bool)
        result = compute_time_to_peak_s(angles, time_sec, 100.0, valid)
        assert result is not None
        # Peak band at 2% of 100 = within 2 deg of 100 → first entry ~frame 98
        assert result > 9.0

    def test_none_peak(self):
        assert compute_time_to_peak_s(np.array([1.0]), np.array([0.0]), None, np.array([True])) is None


# ---------------------------------------------------------------------------
# Motor control
# ---------------------------------------------------------------------------

class TestFitMetrics:
    def test_perfect_polynomial(self):
        """Degree-3 signal → R² ≈ 1.0, RMSE ≈ 0."""
        x = np.arange(50, dtype=float)
        angles = 2 * x ** 3 - x ** 2 + 3 * x + 1
        valid = np.ones(50, dtype=bool)
        r2, rmse = compute_fit_metrics(angles, valid)
        assert r2 is not None
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert rmse == pytest.approx(0.0, abs=1e-3)

    def test_too_few_points(self):
        angles = np.array([1.0, 2.0])
        valid = np.array([True, True])
        r2, rmse = compute_fit_metrics(angles, valid)
        assert r2 is None
        assert rmse is None

    def test_constant_signal(self):
        """Constant signal → R² = 1.0 (perfect fit to constant)."""
        angles = np.full(20, 42.0)
        valid = np.ones(20, dtype=bool)
        r2, rmse = compute_fit_metrics(angles, valid)
        assert r2 == pytest.approx(1.0)


class TestJerkRms:
    def test_smooth_vs_noisy(self):
        """Smooth signal should have lower jerk than noisy."""
        n = 100
        t = np.linspace(0, 5, n)
        smooth_angles = np.sin(t) * 30
        noisy_angles = smooth_angles + np.random.default_rng(0).normal(0, 5, n)
        valid = np.ones(n, dtype=bool)
        jerk_smooth = compute_jerk_rms(smooth_angles, t, valid)
        jerk_noisy = compute_jerk_rms(noisy_angles, t, valid)
        assert jerk_smooth is not None
        assert jerk_noisy is not None
        assert jerk_smooth < jerk_noisy

    def test_too_few_points(self):
        assert compute_jerk_rms(np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([True, True])) is None


# ---------------------------------------------------------------------------
# Compensation
# ---------------------------------------------------------------------------

class TestTorsoAngleChange:
    def test_no_frames(self):
        df = _make_landmarks_df(50)
        assert compute_torso_angle_change_deg(df, []) is None

    def test_single_frame(self):
        df = _make_landmarks_df(50)
        assert compute_torso_angle_change_deg(df, [10]) is None

    def test_returns_finite(self):
        df = _make_landmarks_df(100)
        result = compute_torso_angle_change_deg(df, list(range(20, 60)))
        assert result is not None
        assert np.isfinite(result)


class TestPelvisDrift:
    def test_no_frames(self):
        df = _make_landmarks_df(50)
        assert compute_pelvis_drift_norm(df, []) is None

    def test_returns_finite(self):
        df = _make_landmarks_df(100)
        result = compute_pelvis_drift_norm(df, list(range(20, 60)))
        assert result is not None
        assert np.isfinite(result)
        assert result >= 0


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

class TestFramesValidPct:
    def test_all_valid(self):
        assert compute_frames_valid_pct(np.ones(100, dtype=bool)) == pytest.approx(100.0)

    def test_half_valid(self):
        mask = np.array([True, False] * 50, dtype=bool)
        assert compute_frames_valid_pct(mask) == pytest.approx(50.0)

    def test_empty(self):
        assert compute_frames_valid_pct(np.array([], dtype=bool)) == 0.0


class TestAvgLandmarkVisibility:
    def test_returns_finite(self):
        df = _make_landmarks_df(50)
        result = compute_avg_landmark_visibility(df, list(range(10, 30)))
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_no_frames(self):
        df = _make_landmarks_df(50)
        assert compute_avg_landmark_visibility(df, []) is None


# ---------------------------------------------------------------------------
# None / NaN input handling
# ---------------------------------------------------------------------------

class TestNoneHandling:
    def test_peak_hold_with_all_nan(self):
        angles = np.full(10, np.nan)
        time_sec = np.linspace(0, 1, 10)
        valid = np.zeros(10, dtype=bool)
        assert compute_peak_hold_time_s(angles, time_sec, 50.0, valid) is None

    def test_fit_with_all_invalid(self):
        angles = np.full(10, np.nan)
        valid = np.zeros(10, dtype=bool)
        r2, rmse = compute_fit_metrics(angles, valid)
        assert r2 is None
        assert rmse is None

    def test_jerk_with_empty(self):
        assert compute_jerk_rms(np.array([]), np.array([]), np.array([], dtype=bool)) is None


# ---------------------------------------------------------------------------
# Integration: compute_extended_summary
# ---------------------------------------------------------------------------

class TestExtendedSummary:
    def test_returns_all_keys(self):
        n = 100
        df = _make_landmarks_df(n)
        angles = np.linspace(0, 90, n)
        time_sec = np.linspace(0, 5, n)
        valid = np.ones(n, dtype=bool)
        result = compute_extended_summary(
            pose_name="kneeling_knee_flexion",
            side="right",
            arom_peak=60.0,
            prom_peak=80.0,
            angles=angles,
            time_sec=time_sec,
            valid_mask=valid,
            peak_val=60.0,
            landmarks_df=df,
            frames_used=list(range(40, 80)),
        )
        expected_keys = {
            "movement_name", "joint_name", "side", "angle_type",
            "assist_gap",
            "peak_hold_time_s", "peak_band_std_deg", "time_to_peak_s",
            "fit_r2", "fit_rmse_deg", "jerk_rms",
            "torso_angle_change_deg", "pelvis_drift_norm",
            "frames_valid_pct", "avg_landmark_visibility",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["movement_name"] == "Kneeling Knee Flexion"
        assert result["assist_gap"] == pytest.approx(20.0)
