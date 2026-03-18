"""
audit_results.py – Plausibility audit of batch_run_001 pipeline results.

Reads every summary.csv under a batch results folder, checks each measurement
against pose-specific physiological thresholds, and reports any suspicious
cases.  Optionally re-runs every flagged case with snapshots + plots enabled
for visual inspection.

Usage
-----
    # Audit only (produces audit_suspicious.csv + console report)
    python scripts/audit_results.py --batch results/batch_run_001

    # Audit + re-run flagged cases with snapshots
    python scripts/audit_results.py --batch results/batch_run_001 --rerun \
        --data-root "H:\\My Drive\\participant videos" \
        --config configs/default.yaml

Output
------
    analysis/outputs/audit_suspicious.csv   — flagged rows
    results/audit_rerun/<participant>/<pose_side>/   — re-run results
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrapping so run_mi4l can be imported in-process for re-runs
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _p in (_SCRIPTS_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Physiological plausibility thresholds
# Calibrated to the pipeline's angle definitions (not clinical norms).
#
# kneeling_knee_flexion  : ankle-to-knee vs horizontal, folded to [0,90].
#                          Any value > 90 is an artifact from the pre-fix era.
# standing_hip_abduction : ankle-to-hip vs vertical; 90 = horizontal.
#                          PROM > 100 implies leg above horizontal under passive
#                          force, which is implausible for this standing test.
#                          assist_gap > 45 is also suspicious.
# bilateral_leg_straddle : between-leg angle; > 170 = near full split + assist.
# unilateral_hip_extension: assist_gap > 55 flags extreme passive gains.
# prone_trunk_extension   : measured against horizontal; > 80 is very high.
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, dict[str, float]] = {
    "Kneeling Knee Flexion":    {"arom_max": 90.0, "prom_max": 90.0},
    "Standing Hip Abduction":   {"prom_max": 100.0, "gap_max": 45.0},
    "Bilateral Leg Straddle":   {"prom_max": 170.0},
    "Unilateral Hip Extension": {"gap_max": 55.0},
    "Prone Trunk Extension":    {"arom_max": 80.0},
}

# Maps pipeline pose folder names back to the movement_name string in CSVs
POSE_FOLDER_TO_MOVEMENT: dict[str, str] = {
    "kneeling_knee_flexion":    "Kneeling Knee Flexion",
    "standing_hip_abduction":   "Standing Hip Abduction",
    "bilateral_leg_straddle":   "Bilateral Leg Straddle",
    "unilateral_hip_extension": "Unilateral Hip Extension",
    "prone_trunk_extension":    "Prone Trunk Extension",
    "shoulder_flexion":         "Shoulder Flexion",
    "shoulder_stick_pass_through": "Shoulder Stick Pass-Through",
}

# FILE_MAP mirrors batch_process.py: stem_lower -> (pose, side, kind)
FILE_MAP: dict[str, tuple[str, str, str]] = {
    "kneeflex_la":       ("kneeling_knee_flexion",       "left",  "arom"),
    "kneeflex_lp":       ("kneeling_knee_flexion",       "left",  "prom"),
    "kneeflex_ra":       ("kneeling_knee_flexion",       "right", "arom"),
    "kneeflex_rp":       ("kneeling_knee_flexion",       "right", "prom"),
    "hipabd_la":         ("standing_hip_abduction",      "left",  "arom"),
    "hipabd_lp":         ("standing_hip_abduction",      "left",  "prom"),
    "hipabd_ra":         ("standing_hip_abduction",      "right", "arom"),
    "hipabd_rp":         ("standing_hip_abduction",      "right", "prom"),
    "hipextension_la":   ("unilateral_hip_extension",    "both",  "arom"),
    "hipextension_ra":   ("unilateral_hip_extension",    "both",  "arom"),
    "frontsplit":        ("unilateral_hip_extension",    "both",  "prom"),
    "shoulder_la":       ("shoulder_flexion",            "left",  "arom"),
    "shoulder_lp":       ("shoulder_flexion",            "left",  "prom"),
    "shoulder_ra":       ("shoulder_flexion",            "right", "arom"),
    "shoulder_rp":       ("shoulder_flexion",            "right", "prom"),
    "shoulderextension": ("shoulder_stick_pass_through", "both",  "arom"),
    "sidesplit_a":       ("bilateral_leg_straddle",      "both",  "arom"),
    "sidesplit_p":       ("bilateral_leg_straddle",      "both",  "prom"),
    "trunk_a":           ("prone_trunk_extension",       "both",  "arom"),
    "trunk_p":           ("prone_trunk_extension",       "both",  "prom"),
}

VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv"}


# ---------------------------------------------------------------------------
# Step 1 — Load and audit
# ---------------------------------------------------------------------------

def _load_batch(batch_dir: Path) -> pd.DataFrame:
    """Collect all summary.csv files from a batch output tree."""
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(batch_dir.rglob("summary.csv")):
        # Folder structure: <batch>/<participant>/<pose_side>/summary.csv
        rel = csv_path.relative_to(batch_dir)
        parts = rel.parts  # (participant, pose_side, summary.csv)
        if len(parts) < 3:
            continue
        participant = parts[0]
        pose_side_folder = parts[1]

        df = pd.read_csv(csv_path)
        df.insert(0, "participant", participant)
        df.insert(1, "pose_folder", pose_side_folder)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {batch_dir}")
    return pd.concat(rows, ignore_index=True)


def _audit(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that exceed any plausibility threshold."""
    for col in ("arom_deg", "prom_deg", "assist_gap"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # Recompute assist_gap in case it is missing or NaN
    mask_gap = df["assist_gap"].isna() & df["arom_deg"].notna() & df["prom_deg"].notna()
    df.loc[mask_gap, "assist_gap"] = df.loc[mask_gap, "prom_deg"] - df.loc[mask_gap, "arom_deg"]

    flagged_rows: list[dict] = []

    for _, row in df.iterrows():
        movement = str(row.get("movement_name", ""))
        thresh = THRESHOLDS.get(movement)
        if thresh is None:
            continue

        reasons: list[str] = []

        arom = row["arom_deg"]
        prom = row["prom_deg"]
        gap  = row["assist_gap"]

        if "arom_max" in thresh and pd.notna(arom) and arom > thresh["arom_max"]:
            reasons.append(f"arom={arom:.1f} > max {thresh['arom_max']}")
        if "prom_max" in thresh and pd.notna(prom) and prom > thresh["prom_max"]:
            reasons.append(f"prom={prom:.1f} > max {thresh['prom_max']}")
        if "gap_max" in thresh and pd.notna(gap) and gap > thresh["gap_max"]:
            reasons.append(f"assist_gap={gap:.1f} > max {thresh['gap_max']}")

        if reasons:
            flagged_rows.append({
                "participant":  row["participant"],
                "pose_folder":  row["pose_folder"],
                "movement_name": movement,
                "side":         row.get("side", ""),
                "arom_deg":     arom,
                "prom_deg":     prom,
                "assist_gap":   gap,
                "mi4l_valid":   row.get("mi4l_valid", ""),
                "qc_flags":     row.get("qc_flags", ""),
                "flag_reason":  " | ".join(reasons),
            })

    return pd.DataFrame(flagged_rows)


def _print_report(flagged: pd.DataFrame) -> None:
    if flagged.empty:
        print("No suspicious cases found.")
        return

    print(f"\n{'='*78}")
    print(f"  PLAUSIBILITY AUDIT — {len(flagged)} suspicious case(s) found")
    print(f"{'='*78}")
    header = f"{'Participant':<26} {'Pose / Side':<38} {'Flag reason'}"
    print(header)
    print("-" * 78)
    for _, r in flagged.sort_values(["movement_name", "participant"]).iterrows():
        pose_side = f"{r['movement_name']} / {r['side']}"
        print(f"{r['participant']:<26} {pose_side:<38} {r['flag_reason']}")
    print(f"{'='*78}\n")


# ---------------------------------------------------------------------------
# Step 2 — Re-run flagged cases
# ---------------------------------------------------------------------------

def _find_videos(participant_dir: Path) -> dict[str, Path]:
    """Return {stem_lower: path} for all video files in a participant folder."""
    found: dict[str, Path] = {}
    for f in participant_dir.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            found[f.stem.lower()] = f
    return found


def _build_rerun_job(
    participant: str,
    pose: str,
    side: str,
    data_root: Path,
    pose_folder: str = "",
) -> Optional[dict]:
    """
    Return a dict with arom_path, prom_path, pose, side for a single job.
    Returns None if the AROM video cannot be found.
    """
    participant_dir = data_root / participant
    if not participant_dir.exists():
        print(f"  [WARN] Participant folder not found: {participant_dir}")
        return None

    videos = _find_videos(participant_dir)

    # Find AROM and PROM video stems for this (pose, side)
    arom_stem: Optional[str] = None
    prom_stem: Optional[str] = None

    for stem, (p, s, kind) in FILE_MAP.items():
        if p != pose:
            continue
        if kind == "arom":
            if s == side or (s == "both" and side == "both"):
                if stem in videos:
                    arom_stem = stem
        elif kind == "prom":
            if s == side or (s == "both" and side == "both"):
                if stem in videos:
                    prom_stem = stem

    if arom_stem is None:
        print(f"  [WARN] AROM video not found for {participant} / {pose} / {side}")
        return None

    return {
        "arom_path": videos[arom_stem],
        "prom_path": videos.get(prom_stem) if prom_stem else None,
        "pose":      pose,
        "side":      side,
    }


def _pose_name_to_internal(movement_name: str) -> Optional[str]:
    """Convert movement_name string back to internal pose key."""
    inv = {v: k for k, v in POSE_FOLDER_TO_MOVEMENT.items()}
    return inv.get(movement_name)


def _rerun_flagged(
    flagged: pd.DataFrame,
    data_root: Path,
    config_path: str,
    out_root: Path,
) -> None:
    import run_mi4l  # noqa: PLC0415

    print(f"\nRe-running {len(flagged)} flagged case(s) with snapshots + plots...\n")

    for _, row in flagged.iterrows():
        participant = str(row["participant"])
        movement    = str(row["movement_name"])
        side        = str(row["side"])
        pose        = _pose_name_to_internal(movement)

        if pose is None:
            print(f"  [SKIP] Cannot map movement name to pose key: {movement}")
            continue

        pose_folder = str(row.get("pose_folder", ""))
        job = _build_rerun_job(participant, pose, side, data_root, pose_folder=pose_folder)
        if job is None:
            continue

        out_dir = out_root / participant / f"{pose}_{side}"
        out_dir.mkdir(parents=True, exist_ok=True)

        argv = [
            "--arom",   str(job["arom_path"]),
            "--out",    str(out_dir),
            "--pose",   pose,
            "--config", config_path,
            "--side",   side,
        ]
        if job["prom_path"]:
            argv += ["--prom", str(job["prom_path"])]

        label = f"{participant} | {movement} / {side}"
        print(f"  Running: {label}")
        try:
            run_mi4l.main(argv)
            print(f"  Done:    {label}\n")
        except Exception:
            print(f"  ERROR:   {label}")
            traceback.print_exc()
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plausibility audit of MI4L batch results.")
    p.add_argument("--batch",     required=True,  type=str, help="Path to batch results folder (e.g. results/batch_run_001).")
    p.add_argument("--out-csv",   default="analysis/outputs/audit_suspicious.csv", type=str, help="Where to write the flagged-cases CSV.")
    p.add_argument("--rerun",     action="store_true", help="Re-run all flagged cases with snapshots + plots.")
    p.add_argument("--data-root", default=None, type=str, help="Root folder of participant video folders (required for --rerun).")
    p.add_argument("--config",    default="configs/default.yaml", type=str, help="Pipeline config YAML (default: configs/default.yaml).")
    p.add_argument("--rerun-out", default="results/audit_rerun", type=str, help="Output folder for re-run results.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    batch_dir = Path(args.batch)
    print(f"Loading summaries from: {batch_dir}")
    df = _load_batch(batch_dir)
    print(f"Loaded {len(df)} measurement rows across {df['participant'].nunique()} participants.\n")

    flagged = _audit(df)
    _print_report(flagged)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out_csv, index=False)
    print(f"Suspicious cases saved to: {out_csv}")

    if args.rerun:
        if not args.data_root:
            print("ERROR: --data-root is required when using --rerun.")
            return 1
        _rerun_flagged(
            flagged=flagged,
            data_root=Path(args.data_root),
            config_path=args.config,
            out_root=Path(args.rerun_out),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
