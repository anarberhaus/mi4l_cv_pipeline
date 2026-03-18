"""
batch_process.py – Run the MI4L pipeline for every participant in a data root.

Usage
-----
    python scripts/batch_process.py \
        --data-root "H:\\My Drive\\participant videos" \
        --out results/batch_run_001 \
        --config configs/default.yaml

Structure expected under --data-root
--------------------------------------
    <data-root>/
        ParticipantName1/
            kneeflex_la.mp4
            kneeflex_lp.mp4
            ...
        ParticipantName2/
            ...

Output
------
    <out>/
        ParticipantName1/
            kneeling_knee_flexion_left/   summary.csv, snapshots/, ...
            kneeling_knee_flexion_right/
            ...
        ParticipantName2/
            ...
        all_participants_summary.csv      merged CSV with a "participant" column
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Ensure the scripts folder and src folder are on the path so run_mi4l and
# mi4l package can be imported regardless of how this script is invoked.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _p in (_SCRIPTS_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Video filename stem → (pose_name, side, kind)
# kind is "arom" or "prom"
# ---------------------------------------------------------------------------
FILE_MAP: dict[str, tuple[str, str, str]] = {
    "kneeflex_la":       ("kneeling_knee_flexion",       "left",  "arom"),
    "kneeflex_lp":       ("kneeling_knee_flexion",       "left",  "prom"),
    "kneeflex_ra":       ("kneeling_knee_flexion",       "right", "arom"),
    "kneeflex_rp":       ("kneeling_knee_flexion",       "right", "prom"),
    "hipabd_la":         ("standing_hip_abduction",      "left",  "arom"),
    "hipabd_lp":         ("standing_hip_abduction",      "left",  "prom"),
    "hipabd_ra":         ("standing_hip_abduction",      "right", "arom"),
    "hipabd_rp":         ("standing_hip_abduction",      "right", "prom"),
    # Both la/ra AROM videos are side="both" — the calculation is bilateral
    # (angle between legs); whichever file exists is used.  If a participant
    # recorded both, hipextension_ra (alphabetically last) wins — that is fine.
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

# Poses that are AROM-only (no PROM expected — not an error if missing)
AROM_ONLY_POSES: set[str] = {"shoulder_stick_pass_through"}

# Video file extensions to recognise
VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Job:
    participant: str
    pose: str
    side: str
    arom_path: Path
    prom_path: Optional[Path]
    out_dir: Path


# ---------------------------------------------------------------------------
# Job assembly
# ---------------------------------------------------------------------------

def _find_videos(participant_dir: Path) -> dict[str, Path]:
    """Return {stem_lower: path} for all recognised video files in a folder."""
    found: dict[str, Path] = {}
    for f in participant_dir.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            found[f.stem.lower()] = f
    return found


def _assemble_jobs(participant_dir: Path, out_base: Path) -> list[Job]:
    """
    Scan one participant folder and return one Job per (pose, side) pair.
    """
    participant = participant_dir.name
    videos = _find_videos(participant_dir)

    # Build lookup: (pose, side, kind) → path
    lookup: dict[tuple[str, str, str], Path] = {}
    for stem, path in videos.items():
        if stem in FILE_MAP:
            key = FILE_MAP[stem]
            lookup[key] = path
        else:
            print(f"  [SKIP] Unrecognised file: {path.name}")

    if not lookup:
        print(f"  [WARN] No recognised videos in {participant_dir}")
        return []

    # Collect unique (pose, side) pairs that have an AROM video
    arom_pairs: set[tuple[str, str]] = {
        (pose, side)
        for (pose, side, kind), _ in lookup.items()
        if kind == "arom"
    }

    jobs: list[Job] = []
    for pose, side in sorted(arom_pairs):
        arom_path = lookup.get((pose, side, "arom"))
        if arom_path is None:
            continue

        # Resolve PROM: direct match (AROM is already side="both" for hip extension,
        # so this also finds frontsplit which is registered as side="both")
        prom_path = lookup.get((pose, side, "prom"))

        if prom_path is None and pose not in AROM_ONLY_POSES:
            print(f"  [WARN] {participant} | {pose} {side}: AROM found but no PROM — running AROM only")

        out_dir = out_base / participant / f"{pose}_{side}"

        jobs.append(Job(
            participant=participant,
            pose=pose,
            side=side,
            arom_path=arom_path,
            prom_path=prom_path,
            out_dir=out_dir,
        ))

    return jobs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_job(job: Job, config_path: str) -> Optional[pd.DataFrame]:
    """Call run_mi4l.main() in-process and return the summary DataFrame."""
    # Build argv as if called from the command line
    argv = [
        "--arom",   str(job.arom_path),
        "--out",    str(job.out_dir),
        "--pose",   job.pose,
        "--config", config_path,
        "--side",   job.side,
    ]
    if job.prom_path:
        argv += ["--prom", str(job.prom_path)]

    # Import here so mediapipe is only imported once (already loaded by the
    # first job; subsequent calls reuse the cached module)
    import run_mi4l  # noqa: PLC0415
    run_main = run_mi4l.main

    try:
        exit_code = run_main(argv)
        if exit_code != 0:
            print(f"  [ERROR] run_mi4l returned exit code {exit_code}")
            return None
    except Exception:
        print(f"  [ERROR] Exception while processing job:")
        traceback.print_exc()
        return None

    summary_path = job.out_dir / "summary.csv"
    if not summary_path.exists():
        print(f"  [WARN] No summary.csv produced at {summary_path}")
        return None

    df = pd.read_csv(summary_path)
    df.insert(0, "participant", job.participant)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-process all participant videos through the MI4L pipeline."
    )
    p.add_argument(
        "--data-root", required=True,
        help='Root folder containing one sub-folder per participant (e.g. "H:\\My Drive\\participant videos").',
    )
    p.add_argument(
        "--out", required=True,
        help="Output root folder (e.g. results/batch_run_001).",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--participants", nargs="*", default=None,
        help="Optional list of participant folder names to process. Processes all if omitted.",
    )
    p.add_argument(
        "--snapshots", action="store_true", default=False,
        help="Enable snapshots and plots in the batch output (disabled by default for speed).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    config_path = str(Path(args.config).resolve())

    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    # Build a batch-specific config that inherits all defaults but can
    # toggle snapshots/plots on or off.
    with open(config_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    batch_cfg = copy.deepcopy(base_cfg)
    if args.snapshots:
        batch_cfg.setdefault("export", {})["save_snapshots"] = True
        batch_cfg.setdefault("export", {}).setdefault("plots", {})["enabled"] = True
    else:
        batch_cfg.setdefault("export", {})["save_snapshots"] = False
        batch_cfg.setdefault("export", {}).setdefault("plots", {})["enabled"] = False
    batch_config_path = str(out_root / "_batch_config.yaml")
    with open(batch_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(batch_cfg, f, sort_keys=False)
    config_path = batch_config_path

    # Discover participant folders
    participant_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if args.participants:
        keep = set(args.participants)
        participant_dirs = [d for d in participant_dirs if d.name in keep]

    if not participant_dirs:
        print("[ERROR] No participant folders found.")
        return 1

    print(f"Found {len(participant_dirs)} participant(s): {[d.name for d in participant_dirs]}")

    # Assemble all jobs
    all_jobs: list[Job] = []
    for pdir in participant_dirs:
        print(f"\nScanning {pdir.name}...")
        jobs = _assemble_jobs(pdir, out_root)
        print(f"  -> {len(jobs)} job(s)")
        all_jobs.extend(jobs)

    if not all_jobs:
        print("[ERROR] No jobs to run.")
        return 1

    print(f"\nTotal jobs: {len(all_jobs)}\n{'-' * 60}")

    # Run all jobs, collect summaries
    all_summaries: list[pd.DataFrame] = []
    n_ok = 0
    n_fail = 0

    for i, job in enumerate(all_jobs, start=1):
        label = f"{job.participant} | {job.pose} {job.side}"
        print(f"\n[{i}/{len(all_jobs)}] {label}")
        print(f"  AROM: {job.arom_path.name}")
        if job.prom_path:
            print(f"  PROM: {job.prom_path.name}")

        df = _run_job(job, config_path)
        if df is not None:
            all_summaries.append(df)
            n_ok += 1
        else:
            n_fail += 1

    # Merge and write combined summary
    print(f"\n{'-' * 60}")
    print(f"Completed: {n_ok} ok, {n_fail} failed")

    if all_summaries:
        merged = pd.concat(all_summaries, ignore_index=True)
        merged_path = out_root / "all_participants_summary.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Merged summary written to: {merged_path}")
        print(f"  {len(merged)} rows, {len(merged.columns)} columns")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
