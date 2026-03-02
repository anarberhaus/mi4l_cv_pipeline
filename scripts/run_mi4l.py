from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from mi4l.angles.joint_angles import (
    compute_knee_flexion_angles,
    compute_trunk_extension_angles,
    compute_hip_abduction_angles,
    compute_bilateral_leg_straddle_angle,
    compute_hip_extension_angles,
    compute_shoulder_flexion_angles,
    compute_stick_pass_through_metrics,
)
from mi4l.io.export import (
    save_angles_csv,
    save_config_used,
    save_landmarks_csv,
    save_summary_csv,
)
from mi4l.metrics.arom_prom import estimate_robust_max, estimate_stable_plateau
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
from mi4l.viz.snapshot import save_snapshot


POSE_TO_METRIC_FN = {
    "kneeling_knee_flexion": compute_knee_flexion_angles,
    "prone_trunk_extension": compute_trunk_extension_angles,
    "standing_hip_abduction": compute_hip_abduction_angles,
    "bilateral_leg_straddle": compute_bilateral_leg_straddle_angle,
    "unilateral_hip_extension": compute_hip_extension_angles,
    "shoulder_flexion": compute_shoulder_flexion_angles,
    "shoulder_stick_pass_through": compute_stick_pass_through_metrics,
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MI4L (Milestone 1: knee only) from paired AROM/PROM videos.")
    p.add_argument("--arom", required=True, type=str, help="Path to AROM video (mp4).")
    p.add_argument("--prom", required=False, type=str, help="Path to PROM video (mp4, optional).")
    p.add_argument("--out", required=True, type=str, help="Output folder (e.g., results/run_001/).")
    p.add_argument("--pose", required=True, type=str, help="Pose name (maps to metric function).")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config (e.g., configs/default.yaml).")
    p.add_argument("--side", required=False, type=str, choices=["left", "right", "both"],
                   help="Override mi4l.side from config (left | right | both).")
    return p.parse_args(argv)


def _process_one_video(kind: str, video_path: Path, out_dir: Path, cfg: dict, active_sides: list[str], pose_name: str | None = None) -> dict:
    # 1) Landmarks
    landmarks_df, meta = extract_mediapipe_pose_landmarks(video_path=str(video_path), pose_cfg=cfg.get("pose", {}))

    if cfg.get("export", {}).get("save_landmarks_csv", True):
        save_landmarks_csv(landmarks_df, out_dir / f"landmarks_{kind}.csv")

    # 2) Angles (pose-specific)
    if pose_name is None:
        metric_fn = compute_knee_flexion_angles
    else:
        metric_fn = POSE_TO_METRIC_FN.get(pose_name)
        if metric_fn is None:
            raise ValueError(f"Unknown pose name: {pose_name}")

    angles_df = metric_fn(landmarks_df)

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

    # 4) Derivative QC (uses smoothed series) - apply to available angle columns
    max_dps = float(cfg.get("qc", {}).get("derivative_deg_per_sec_max", 0.0))

    # Utility: find side-specific angle series in angles_df
    def _find_side_series(side: str):
        for c in angles_df.columns:
            if c.startswith(f"{side}_") and c.endswith("_deg"):
                return c
        # fallback: single deg column
        deg_cols = [c for c in angles_df.columns if c.endswith("_deg")]
        if len(deg_cols) == 1:
            return deg_cols[0]
        return None

    left_col = _find_side_series("left")
    right_col = _find_side_series("right")

    # Detect bilateral/standard poses (single non-sided column)
    target_cols = [c for c in angles_df.columns if c.endswith("_deg") or c.endswith("_dist_norm")]
    is_bilateral = (
        len(target_cols) == 1 and 
        not target_cols[0].startswith("left_") and 
        not target_cols[0].startswith("right_")
    )
    bilateral_col = target_cols[0] if is_bilateral else None

    # default masks
    left_deriv_ok = pd.Series(True, index=angles_df.index)
    right_deriv_ok = pd.Series(True, index=angles_df.index)
    bilateral_deriv_ok = pd.Series(True, index=angles_df.index)

    if left_col is not None:
        left_deriv_ok = apply_derivative_qc(time_sec=angles_df["time_sec"], angle_deg=angles_df[left_col], max_deg_per_sec=max_dps)
    if right_col is not None:
        right_deriv_ok = apply_derivative_qc(time_sec=angles_df["time_sec"], angle_deg=angles_df[right_col], max_deg_per_sec=max_dps)
    if bilateral_col is not None:
        bilateral_deriv_ok = apply_derivative_qc(time_sec=angles_df["time_sec"], angle_deg=angles_df[bilateral_col], max_deg_per_sec=max_dps)

    left_valid = base_left_valid & left_deriv_ok
    right_valid = base_right_valid & right_deriv_ok
    # For bilateral, use combined visibility from both sides
    bilateral_valid = (base_left_valid | base_right_valid) & bilateral_deriv_ok

    # Attach QC columns to angles (keep existing knee-named visibility fields)
    angles_df = angles_df.copy()
    angles_df["left_knee_valid"] = left_valid
    angles_df["right_knee_valid"] = right_valid
    angles_df["left_knee_vis_min"] = vis_qc["left_knee_vis_min"]
    angles_df["right_knee_vis_min"] = vis_qc["right_knee_vis_min"]
    angles_df["subject_bbox_h_px"] = landmarks_df["bbox_h_px"]
    angles_df["is_clipped"] = landmarks_df["is_clipped"]

    if cfg.get("export", {}).get("save_angles_csv", True):
        save_angles_csv(angles_df, out_dir / f"angles_{kind}.csv")

    # 5) Robust max (top-K median)
    smooth_cfg = cfg.get("smoothing", {}) or {}
    robust_cfg = dict(cfg.get("robust_max", {}) or {})  # shallow copy – safe to mutate
    qc_cfg = cfg.get("qc", {}) or {}
    # Inject peak_detection sub-config so estimate_robust_max can read fallback params
    # without requiring a signature change.
    peak_detection_cfg = cfg.get("peak_detection", {}) or {}
    robust_cfg["_peak_detection"] = peak_detection_cfg

    def _compute_est_for_col(col_name, valid_mask=None):
        if col_name is None or col_name not in angles_df.columns:
            return None
        # Use provided valid_mask or try to find one
        if valid_mask is None:
            base = col_name.rsplit("_", 2)[0]
            valid_mask = angles_df.get(f"{base}_valid") if f"{base}_valid" in angles_df.columns else pd.Series(True, index=angles_df.index)
        # Distance metrics (e.g. stick pass-through) use stable plateau detection
        if col_name.endswith("_dist_norm"):
            plateau_cfg = cfg.get("stable_plateau", {}) or {}
            return estimate_stable_plateau(
                series=angles_df[col_name],
                valid_mask=valid_mask,
                smoothing_cfg=smooth_cfg,
                plateau_cfg=plateau_cfg,
                qc_cfg=qc_cfg,
            )
        # Angle metrics use topk robust max/min
        direction = "max"
        return estimate_robust_max(angle_deg=angles_df[col_name], valid_mask=valid_mask, smoothing_cfg=smooth_cfg, robust_cfg=robust_cfg, qc_cfg=qc_cfg, direction=direction)

    # Compute estimates based on pose type
    if is_bilateral:
        bilateral_est = _compute_est_for_col(bilateral_col, bilateral_valid)
        left_est = None
        right_est = None
    else:
        # Only compute estimates for active sides
        left_est = _compute_est_for_col(left_col) if "left" in active_sides else None
        right_est = _compute_est_for_col(right_col) if "right" in active_sides else None
        bilateral_est = None

    # 6) Video-level QC flags
    video_flags = compute_video_level_qc_flags(
        landmarks_df=landmarks_df,
        qc_cfg=qc_cfg,
    )

    # Optional plots (now that robust frames are known)
    plots_cfg = cfg.get("export", {}).get("plots", {}) or {}
    if bool(plots_cfg.get("enabled", False)):
        dpi = int(plots_cfg.get("dpi", 150))
        # build robust_frames mapping from estimated frames_used (if available)
        robust_frames = {}
        try:
            if is_bilateral:
                robust_frames_bilateral = list(getattr(bilateral_est, "frames_used", []) or []) if bilateral_est else []
                robust_frames = {"bilateral": robust_frames_bilateral}
            else:
                robust_frames_left = list(getattr(left_est, "frames_used", []) or []) if left_est else []
                robust_frames_right = list(getattr(right_est, "frames_used", []) or []) if right_est else []
                robust_frames = {"left": robust_frames_left, "right": robust_frames_right}
        except Exception:
            robust_frames = None

        # Determine which side to plot
        plot_side = None
        if is_bilateral:
            plot_side = "both"  # Bilateral poses plot as single series
        elif len(active_sides) == 1:
            plot_side = active_sides[0]
        
        plot_knee_angles(
            angles_df=angles_df,
            out_path=out_dir / f"plot_{pose_name}_{kind}.png",
            title=f"{pose_name} (deg) - {kind.upper()}",
            dpi=dpi,
            robust_frames=robust_frames,
            side=plot_side,
        )

    # Snapshots: export frame image at median of top-K used frames (if provided)
    save_snaps = bool(cfg.get("export", {}).get("save_snapshots", True))
    if save_snaps:
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        def _save_bilateral_snapshot(est):
            # Get metric base name from bilateral column (strip _deg or _dist_norm)
            metric_base = bilateral_col.replace("_deg", "").replace("_dist_norm", "") if bilateral_col else "bilateral"
            
            # Find frame to snapshot: use frames_used logic similar to side snapshot
            frames_used = list(getattr(est, "frames_used", []) or []) if est else []
            if not frames_used:
                print(f"DEBUG: No valid frames used for bilateral snapshot. Forcing snapshot at fallback frame.")
                if len(angles_df) > 0:
                    frame_idx = int(angles_df.iloc[len(angles_df) // 2]["frame_idx"])
                else:
                    return
            else:
                # Find the largest temporal cluster in frames_used to avoid stragglers/outliers
                frames_used = sorted(list(frames_used))
                clusters = []
                current_cluster = [frames_used[0]]
                for f in frames_used[1:]:
                    if f - current_cluster[-1] > 10:
                        clusters.append(current_cluster)
                        current_cluster = [f]
                    else:
                        current_cluster.append(f)
                clusters.append(current_cluster)
                
                # Pick largest cluster
                largest_cluster = max(clusters, key=len)
                
                # Pick median of the largest cluster
                try:
                    frame_idx = int(np.median(np.array(largest_cluster, dtype=float)))
                except Exception:
                    frame_idx = int(largest_cluster[len(largest_cluster) // 2])
            
            # Get landmarks for this frame
            lm_row = landmarks_df[landmarks_df["frame_idx"] == frame_idx]
            if lm_row.empty:
                return
            lm_row = lm_row.iloc[0].to_dict()
            
            # Dispatch landmark selection and snapshot mode based on pose
            if pose_name == "shoulder_stick_pass_through":
                # Show left_wrist -> right_wrist with shoulder midpoint as reference
                # Compute shoulder midpoint for context
                try:
                    lshx = lm_row.get("left_shoulder_x")
                    lshy = lm_row.get("left_shoulder_y")
                    rshx = lm_row.get("right_shoulder_x")
                    rshy = lm_row.get("right_shoulder_y")
                    if None not in (lshx, lshy, rshx, rshy):
                        lm_row["shoulder_midpoint_x"] = (float(lshx) + float(rshx)) / 2.0
                        lm_row["shoulder_midpoint_y"] = (float(lshy) + float(rshy)) / 2.0
                except Exception:
                    pass
                a_name = "left_wrist"
                b_name = "right_wrist"
                c_name = None
                snap_mode = "distance"
            else:
                # Compute ALL midpoints needed by bilateral poses
                try:
                    lhipx = lm_row.get("left_hip_x")
                    lhipy = lm_row.get("left_hip_y")
                    rhipx = lm_row.get("right_hip_x")
                    rhipy = lm_row.get("right_hip_y")
                    if None not in (lhipx, lhipy, rhipx, rhipy):
                        lm_row["hip_midpoint_x"]  = (float(lhipx) + float(rhipx)) / 2.0
                        lm_row["hip_midpoint_y"]  = (float(lhipy) + float(rhipy)) / 2.0
                        lm_row["pelvis_center_x"] = lm_row["hip_midpoint_x"]
                        lm_row["pelvis_center_y"] = lm_row["hip_midpoint_y"]
                except Exception:
                    pass
                try:
                    lshx = lm_row.get("left_shoulder_x")
                    lshy = lm_row.get("left_shoulder_y")
                    rshx = lm_row.get("right_shoulder_x")
                    rshy = lm_row.get("right_shoulder_y")
                    if None not in (lshx, lshy, rshx, rshy):
                        lm_row["shoulder_midpoint_x"] = (float(lshx) + float(rshx)) / 2.0
                        lm_row["shoulder_midpoint_y"] = (float(lshy) + float(rshy)) / 2.0
                except Exception:
                    pass
                a_name = "left_ankle"
                b_name = "pelvis_center"
                c_name = "right_ankle"
                snap_mode = "auto"
            
            # Output filename
            fname = f"{metric_base}_both_{kind}_max.png"
            out_path = snap_dir / fname
            
            # Value to show
            ang_val = getattr(est, "value_deg", None)
            save_snapshot(video_path=video_path, landmarks_row=lm_row, a_name=a_name, b_name=b_name, c_name=c_name, out_path=out_path, angle_deg=ang_val, angle_mode=snap_mode, pose_name=pose_name)

        def _save_side_snapshot(side: str, est):
            frames_used = list(getattr(est, "frames_used", []) or []) if est else []
            if not frames_used:
                print(f"DEBUG: No valid frames used for snapshot on {side}. Forcing snapshot at fallback frame.")
                if len(angles_df) > 0:
                    mid_label = angles_df.index[len(angles_df) // 2]
                else:
                    return
            else:
                # Find the largest temporal cluster in frames_used to avoid stragglers/outliers
                frames_used = sorted(list(frames_used))

                # Simple clustering: split if gap > 10 frames (approx 0.3s at 30fps)
                clusters = []
                current_cluster = [frames_used[0]]
                for f in frames_used[1:]:
                    if f - current_cluster[-1] > 10:
                        clusters.append(current_cluster)
                        current_cluster = [f]
                    else:
                        current_cluster.append(f)
                clusters.append(current_cluster)

                # Pick largest cluster
                largest_cluster = max(clusters, key=len)
                
                # Pick median of the largest cluster
                try:
                    mid_label = int(np.median(np.array(largest_cluster, dtype=float)))
                except Exception:
                    mid_label = int(largest_cluster[len(largest_cluster) // 2])

            # map label to angles_df index/position
            if mid_label in angles_df.index:
                ang_row = angles_df.loc[mid_label]
            else:
                # fallback: try to find row with frame_idx == mid_label
                # (This depends on if index is frame_idx or simple range)
                rows = angles_df[angles_df["frame_idx"] == mid_label]
                if len(rows) > 0:
                    ang_row = rows.iloc[0]
                else:
                    return

            video_frame_idx = int(ang_row["frame_idx"])
            lm_rows = landmarks_df[landmarks_df["frame_idx"] == video_frame_idx]
            if len(lm_rows) == 0:
                return
            lm_row = lm_rows.iloc[0].to_dict()

            # compute midpoints if required by metric
            try:
                # hip midpoint
                lhipx = lm_row.get("left_hip_x")
                lhipy = lm_row.get("left_hip_y")
                rhipx = lm_row.get("right_hip_x")
                rhipy = lm_row.get("right_hip_y")
                if None not in (lhipx, lhipy, rhipx, rhipy):
                    lm_row["hip_midpoint_x"] = (float(lhipx) + float(rhipx)) / 2.0
                    lm_row["hip_midpoint_y"] = (float(lhipy) + float(rhipy)) / 2.0
            except Exception:
                pass

            try:
                # shoulder midpoint
                lshx = lm_row.get("left_shoulder_x")
                lshy = lm_row.get("left_shoulder_y")
                rshx = lm_row.get("right_shoulder_x")
                rshy = lm_row.get("right_shoulder_y")
                if None not in (lshx, lshy, rshx, rshy):
                    lm_row["shoulder_midpoint_x"] = (float(lshx) + float(rshx)) / 2.0
                    lm_row["shoulder_midpoint_y"] = (float(lshy) + float(rshy)) / 2.0
            except Exception:
                pass

            try:
                # pelvis center (same as hip midpoint)
                if "hip_midpoint_x" in lm_row:
                    lm_row["pelvis_center_x"] = lm_row["hip_midpoint_x"]
                    lm_row["pelvis_center_y"] = lm_row["hip_midpoint_y"]
            except Exception:
                pass

            # Determine A,B,C names per pose and side
            metric_base = None
            # find a representative angle column
            deg_cols = [c for c in angles_df.columns if c.endswith("_deg")]
            if len(deg_cols) == 1:
                metric_base = deg_cols[0].replace("_deg", "")
            else:
                # prefer side-specific
                chosen = None
                for c in deg_cols:
                    if c.startswith(f"{side}_"):
                        chosen = c
                        break
                if chosen is None and deg_cols:
                    chosen = deg_cols[0]
                if chosen is not None:
                    metric_base = chosen.replace("_deg", "")

            if metric_base is None:
                metric_base = pose_name

            # map to A,B,C
            a_name = ""
            b_name = ""
            c_name = ""
            snap_mode = "auto"   # overridden per-pose where needed
            mb = metric_base
            if mb.startswith("knee") or "knee" in mb:
                # knee_flexion -> hip - knee - ankle
                a_name = f"{side}_hip"
                b_name = f"{side}_knee"
                c_name = f"{side}_ankle"
            elif "trunk_extension" in mb or "trunk" in mb:
                a_name = "hip_midpoint"
                b_name = "shoulder_midpoint"
                c_name = "nose"
            elif "hip_abduction" in mb or "abduction" in mb:
                # contralateral_hip - ipsilateral_hip - ipsilateral_ankle
                if side == "left":
                    a_name = "right_hip"
                    b_name = "left_hip"
                    c_name = "left_ankle"
                else:
                    a_name = "left_hip"
                    b_name = "right_hip"
                    c_name = "right_ankle"
            elif "straddle" in mb:
                a_name = "left_ankle"
                b_name = "pelvis_center"
                c_name = "right_ankle"
            elif "hip_extension" in mb or "hip_extension" in pose_name:
                a_name = "shoulder_midpoint"
                b_name = f"{side}_hip"
                c_name = f"{side}_knee"
            elif "shoulder_flexion" in mb or ("shoulder" in mb and "stick" not in mb):
                a_name = "hip_midpoint"
                b_name = f"{side}_shoulder"
                c_name = f"{side}_wrist"
                snap_mode = "internal"
            else:
                # fallback to left-side hip/knee/ankle
                a_name = f"{side}_hip"
                b_name = f"{side}_knee"
                c_name = f"{side}_ankle"

            # output filename
            fname = f"{metric_base}_{side}_{kind}_max.png"
            out_path = snap_dir / fname

            # angle value to show: use est.value_deg if present
            ang_val = getattr(est, "value_deg", None)
            save_snapshot(video_path=video_path, landmarks_row=lm_row, a_name=a_name, b_name=b_name, c_name=c_name, out_path=out_path, angle_deg=ang_val, angle_mode=snap_mode, pose_name=pose_name)

        # Call appropriate snapshot function based on pose type
        if is_bilateral:
            _save_bilateral_snapshot(bilateral_est)
        else:
            if "left" in active_sides:
                _save_side_snapshot("left", left_est)
            if "right" in active_sides:
                _save_side_snapshot("right", right_est)

    return {
        "landmarks_df": landmarks_df,
        "angles_df": angles_df,
        "meta": meta,
        "left": left_est,
        "right": right_est,
        "bilateral": bilateral_est,
        "is_bilateral": is_bilateral,
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
    prom_path = Path(args.prom) if args.prom else None

    mi4l_cfg = cfg.get("mi4l", {}) or {}
    # --side CLI flag overrides mi4l.side from config
    if args.side:
        mi4l_cfg["side"] = args.side
        cfg["mi4l"] = mi4l_cfg
    mode_side = str(mi4l_cfg.get("side", "both")).lower().strip()
    
    active_sides = []
    if mode_side == "left":
        active_sides = ["left"]
    elif mode_side == "right":
        active_sides = ["right"]
    else:
        active_sides = ["left", "right"]

    arom_res = _process_one_video("arom", arom_path, out_dir, cfg, active_sides=active_sides, pose_name=args.pose)
    
    prom_res = None
    if prom_path:
        prom_res = _process_one_video("prom", prom_path, out_dir, cfg, active_sides=active_sides, pose_name=args.pose)

    # 7) MI4L (knee only)
    rows: list[dict] = []
    
    # Handle bilateral vs sided poses differently
    if arom_res.get("is_bilateral", False):
        # Bilateral pose: single summary row
        arom_est = arom_res.get("bilateral")
        prom_est = prom_res.get("bilateral") if prom_res else None
        
        if arom_est is not None:
            mi4l_val = None
            mi4l_valid = False
            mi4l_flags = []
            
            if prom_est is not None:
                mi4l_res = compute_mi4l(
                    arom_deg=arom_est.value_deg,
                    prom_deg=prom_est.value_deg,
                    prom_min_deg=float(mi4l_cfg.get("prom_min_deg", 1.0)),
                    prom_lt_arom_tolerance_deg=float(mi4l_cfg.get("prom_lt_arom_tolerance_deg", 2.0)),
                    invalidate_if_prom_lt_arom=bool(mi4l_cfg.get("invalidate_if_prom_lt_arom", True)),
                )
                mi4l_val = mi4l_res.value
                mi4l_valid = mi4l_res.valid
                mi4l_flags = mi4l_res.flags
            
            qc_flags = []
            qc_flags.extend(arom_res.get("video_flags", []))
            if prom_res:
                qc_flags.extend(prom_res.get("video_flags", []))
            qc_flags.extend([f"arom:{f}" for f in arom_est.flags])
            if prom_est:
                qc_flags.extend([f"prom:{f}" for f in prom_est.flags])
            qc_flags.extend([f"mi4l:{f}" for f in mi4l_flags])
            
            rows.append({
                "joint": args.pose,
                "side": "both",
                "arom_deg": arom_est.value_deg,
                "prom_deg": prom_est.value_deg if prom_est else None,
                "mi4l": mi4l_val,
                "arom_confidence": arom_est.confidence,
                "prom_confidence": prom_est.confidence if prom_est else None,
                "mi4l_valid": mi4l_valid,
                "qc_flags": ";".join(sorted(set(qc_flags))),
            })
    else:
        # Sided pose: iterate over active sides
        for side in active_sides:
            arom_est = arom_res[side]
            prom_est = prom_res[side] if prom_res else None

            # If estimation failed or didn't run, handle gracefully
            if arom_est is None:
                continue

            mi4l_val = None
            mi4l_valid = False
            mi4l_flags = []

            if prom_est is not None:
                mi4l_res = compute_mi4l(
                    arom_deg=arom_est.value_deg,
                    prom_deg=prom_est.value_deg,
                    prom_min_deg=float(mi4l_cfg.get("prom_min_deg", 1.0)),
                    prom_lt_arom_tolerance_deg=float(mi4l_cfg.get("prom_lt_arom_tolerance_deg", 2.0)),
                    invalidate_if_prom_lt_arom=bool(mi4l_cfg.get("invalidate_if_prom_lt_arom", True)),
                )
                mi4l_val = mi4l_res.value
                mi4l_valid = mi4l_res.valid
                mi4l_flags = mi4l_res.flags

            qc_flags = []
            qc_flags.extend(arom_res.get("video_flags", []))
            if prom_res:
                qc_flags.extend(prom_res.get("video_flags", []))
            qc_flags.extend([f"arom:{f}" for f in arom_est.flags])
            if prom_est:
                qc_flags.extend([f"prom:{f}" for f in prom_est.flags])
            qc_flags.extend([f"mi4l:{f}" for f in mi4l_flags])

            rows.append({
                "joint": args.pose,
                "side": side,
                "arom_deg": arom_est.value_deg,
                "prom_deg": prom_est.value_deg if prom_est else None,
                "mi4l": mi4l_val,
                "arom_confidence": arom_est.confidence,
                "prom_confidence": prom_est.confidence if prom_est else None,
                "mi4l_valid": mi4l_valid,
                "qc_flags": ";".join(sorted(set(qc_flags))),
            })

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
