from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mi4l.angles.joint_angles import compute_knee_flexion_angles
from mi4l.io.export import (
    save_angles_csv,
    save_config_used,
    save_landmarks_csv,
    save_summary_csv,
)
from mi4l.metrics.arom_prom import estimate_robust_max
from mi4l.metrics.mi4l import compute_mi4l
from mi4l.pose.mediapipe_pose import extract_mediapipe_pose_landmarks
from mi4l.qc.qc_rules import (
    apply_derivative_qc,
    compute_knee_visibility_qc,
    compute_subject_size_qc,
    compute_video_level_qc_flags,
)
from mi4l.utils.config import load_config
from mi4l.viz.plots import plot_knee_angles


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MI4L (Milestone 1: knee only) from paired AROM/PROM videos.")
    p.add_argument("--arom", required=True, type=str, help="Path to AROM video (mp4).")
    p.add_argument("--prom", required=True, type=str, help="Path to PROM video (mp4).")
    p.add_argument("--out", required=True, type=str, help="Output folder (e.g., results/run_001/).")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config (e.g., configs/default.yaml).")
    return p.parse_args(argv)


def _process_one_video(kind: str, video_path: Path, out_dir: Path, cfg: dict) -> dict:
    # 1) Landmarks
    landmarks_df, meta = extract_mediapipe_pose_landmarks(video_path=str(video_path), pose_cfg=cfg.get("pose", {}))

    if cfg.get("export", {}).get("save_landmarks_csv", True):
        save_landmarks_csv(landmarks_df, out_dir / f"landmarks_{kind}.csv")

    # 2) Angles (knee only)
    angles_df = compute_knee_flexion_angles(landmarks_df)

    # 3) Basic QC masks
    vis_qc = compute_knee_visibility_qc(
        landmarks_df,
        visibility_threshold=float(cfg.get("qc", {}).get("landmark_visibility_threshold", 0.5)),
    )
    size_qc = compute_subject_size_qc(
        landmarks_df,
        min_bbox_height_px=int(cfg.get("qc", {}).get("min_bbox_height_px", 0)),
        require_subject_size=bool(cfg.get("qc", {}).get("require_subject_size", False)),
    )

    exclude_clipped = bool(cfg.get("qc", {}).get("exclude_clipped_frames", False))
    clipped_ok = (~landmarks_df["is_clipped"].fillna(False)) if exclude_clipped else pd.Series(True, index=landmarks_df.index)

    base_left_valid = vis_qc["left_knee_valid"] & size_qc["subject_size_ok"] & clipped_ok
    base_right_valid = vis_qc["right_knee_valid"] & size_qc["subject_size_ok"] & clipped_ok

    # 4) Derivative QC (uses smoothed series)
    max_dps = float(cfg.get("qc", {}).get("derivative_deg_per_sec_max", 0.0))
    left_deriv_ok = apply_derivative_qc(
        time_sec=angles_df["time_sec"],
        angle_deg=angles_df["left_knee_flexion_deg"],
        max_deg_per_sec=max_dps,
    )
    right_deriv_ok = apply_derivative_qc(
        time_sec=angles_df["time_sec"],
        angle_deg=angles_df["right_knee_flexion_deg"],
        max_deg_per_sec=max_dps,
    )

    left_valid = base_left_valid & left_deriv_ok
    right_valid = base_right_valid & right_deriv_ok

    # Attach QC columns to angles
    angles_df = angles_df.copy()
    angles_df["left_knee_valid"] = left_valid
    angles_df["right_knee_valid"] = right_valid
    angles_df["left_knee_vis_min"] = vis_qc["left_knee_vis_min"]
    angles_df["right_knee_vis_min"] = vis_qc["right_knee_vis_min"]
    angles_df["subject_bbox_h_px"] = landmarks_df["bbox_h_px"]
    angles_df["is_clipped"] = landmarks_df["is_clipped"]

    if cfg.get("export", {}).get("save_angles_csv", True):
        save_angles_csv(angles_df, out_dir / f"angles_{kind}.csv")

    # Optional plots
    plots_cfg = cfg.get("export", {}).get("plots", {}) or {}
    if bool(plots_cfg.get("enabled", False)):
        dpi = int(plots_cfg.get("dpi", 150))
        plot_knee_angles(
            angles_df=angles_df,
            out_path=out_dir / f"plot_knee_{kind}.png",
            title=f"Knee flexion (deg) - {kind.upper()}",
            dpi=dpi,
        )

    # 5) Robust max (top-K median)
    smooth_cfg = cfg.get("smoothing", {}) or {}
    robust_cfg = cfg.get("robust_max", {}) or {}
    qc_cfg = cfg.get("qc", {}) or {}

    left_est = estimate_robust_max(
        angle_deg=angles_df["left_knee_flexion_deg"],
        valid_mask=left_valid,
        smoothing_cfg=smooth_cfg,
        robust_cfg=robust_cfg,
        qc_cfg=qc_cfg,
    )
    right_est = estimate_robust_max(
        angle_deg=angles_df["right_knee_flexion_deg"],
        valid_mask=right_valid,
        smoothing_cfg=smooth_cfg,
        robust_cfg=robust_cfg,
        qc_cfg=qc_cfg,
    )

    # 6) Video-level QC flags
    video_flags = compute_video_level_qc_flags(
        landmarks_df=landmarks_df,
        qc_cfg=qc_cfg,
    )

    return {
        "landmarks_df": landmarks_df,
        "angles_df": angles_df,
        "meta": meta,
        "left": left_est,
        "right": right_est,
        "video_flags": video_flags,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("export", {}).get("save_config_used", True)):
        save_config_used(cfg, out_dir / "config_used.yaml")

    arom_path = Path(args.arom)
    prom_path = Path(args.prom)

    arom_res = _process_one_video("arom", arom_path, out_dir, cfg)
    prom_res = _process_one_video("prom", prom_path, out_dir, cfg)

    # 7) MI4L (knee only)
    rows: list[dict] = []
    mi4l_cfg = cfg.get("mi4l", {}) or {}

    for side in ("left", "right"):
        arom_est = arom_res[side]
        prom_est = prom_res[side]

        mi4l_res = compute_mi4l(
            arom_deg=arom_est.value_deg,
            prom_deg=prom_est.value_deg,
            prom_min_deg=float(mi4l_cfg.get("prom_min_deg", 1.0)),
            prom_lt_arom_tolerance_deg=float(mi4l_cfg.get("prom_lt_arom_tolerance_deg", 2.0)),
            invalidate_if_prom_lt_arom=bool(mi4l_cfg.get("invalidate_if_prom_lt_arom", True)),
        )

        qc_flags = []
        qc_flags.extend(arom_res.get("video_flags", []))
        qc_flags.extend(prom_res.get("video_flags", []))
        qc_flags.extend([f"arom:{f}" for f in arom_est.flags])
        qc_flags.extend([f"prom:{f}" for f in prom_est.flags])
        qc_flags.extend([f"mi4l:{f}" for f in mi4l_res.flags])

        rows.append(
            {
                "joint": "knee_flexion",
                "side": side,
                "arom_deg": arom_est.value_deg,
                "prom_deg": prom_est.value_deg,
                "mi4l": mi4l_res.value,
                "arom_confidence": arom_est.confidence,
                "prom_confidence": prom_est.confidence,
                "mi4l_valid": mi4l_res.valid,
                "qc_flags": ";".join(sorted(set(qc_flags))),
            }
        )

    summary_df = pd.DataFrame(rows)
    if cfg.get("export", {}).get("save_summary_csv", True):
        save_summary_csv(summary_df, out_dir / "summary.csv")

    # Console summary (short)
    for _, r in summary_df.iterrows():
        side = str(r["side"]).upper()
        arom = r["arom_deg"]
        prom = r["prom_deg"]
        mi4l = r["mi4l"]
        valid = bool(r["mi4l_valid"])
        flags = str(r["qc_flags"])
        print(f"[{side}] AROM={arom!s} deg | PROM={prom!s} deg | MI4L={mi4l!s} | valid={valid} | flags={flags}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
