"""
run_all.py  –  Run the MI4L pipeline for all supported poses sequentially.

Usage (from project root):
    python scripts/run_all.py
    .venv\\Scripts\\python.exe scripts/run_all.py
    conda run -n <env> python scripts/run_all.py

The script auto-detects the Python interpreter that has the required packages
(pandas, mediapipe, etc.) installed. If auto-detection fails, set PYTHON_OVERRIDE
below to the full path of the correct interpreter.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pose definitions — map each pose to its actual video files and active side
# ---------------------------------------------------------------------------
POSES = [
    {
        "name": "kneeling_knee_flexion",
        "arom": "data/arom/kneeFlex_la.mp4",
        "prom": "data/prom/kneeFlex_lp.mp4",
        "out":  "results/kneeling_knee_flexion_left",
        "side": "left",
    },
    {
        "name": "kneeling_knee_flexion",
        "arom": "data/arom/knee_flex_ra.mp4",
        "prom": "data/prom/knee_flex_rp.mp4",
        "out":  "results/kneeling_knee_flexion_right",
        "side": "right",
    },
    {
        "name": "prone_trunk_extension",
        "arom": "data/arom/trunk_a.mp4",
        "prom": "data/prom/trunk_p.mp4",
        "out":  "results/prone_trunk_extension",
        "side": "both",
    },
    {
        "name": "standing_hip_abduction",
        "arom": "data/arom/legRaise_ra.mp4",
        "prom": "data/prom/legRaise_rp.mp4",
        "out":  "results/standing_hip_abduction",
        "side": "right",
    },
    {
        "name": "bilateral_leg_straddle",
        "arom": "data/arom/legSpread_arom.mp4",
        "prom": "data/prom/legSpread_prom.mp4",
        "out":  "results/bilateral_leg_straddle",
        "side": "both",
    },
    {
        "name": "unilateral_hip_extension",
        "arom": "data/arom/hip_extension_ra.mp4",
        "prom": "data/prom/hip_extension_rp.mp4",
        "out":  "results/unilateral_hip_extension",
        "side": "right",
    },
    {
        "name": "shoulder_flexion",
        "arom": "data/arom/shoulder_la.mp4",
        "prom": "data/prom/shoulder_prom.mp4",
        "out":  "results/shoulder_flexion",
        "side": "left",
    },
    {
        "name": "shoulder_stick_pass_through",
        "arom": "data/other/stickShoulder.mp4",
        "prom": None,       # no PROM video available
        "out":  "results/shoulder_stick_pass_through",
        "side": "both",
    },
]

CONFIG = "configs/default.yaml"
SCRIPT = "scripts/run_mi4l.py"

# ---------------------------------------------------------------------------
# Python interpreter resolution
# ---------------------------------------------------------------------------
# Set PYTHON_OVERRIDE to force a specific interpreter, or leave as None for auto-detect.
# PYTHON_OVERRIDE = r"C:\Users\alexn\anaconda3\envs\mi4l\python.exe"
PYTHON_OVERRIDE: str | None = None


def _find_python() -> str:
    """
    Return the first Python that has pandas, cv2, AND a compatible mediapipe
    (one exposing mp.solutions.pose). The system Python 3.12 has a newer
    mediapipe without mp.solutions, so we verify that API explicitly.
    """
    if PYTHON_OVERRIDE:
        return PYTHON_OVERRIDE

    import shutil
    candidates = [
        # Dedicated conda env for this project – confirmed working
        r"C:\Users\alexn\anaconda3\envs\mi4l\python.exe",
        sys.executable,  # current interpreter
    ]
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    # Probe explicitly checks mp.solutions.pose (dropped in mediapipe >= 0.10)
    check = (
        "import pandas, cv2; "
        "import mediapipe as mp; "
        "mp.solutions.pose.PoseLandmark; "
        "print('ok')"
    )
    for candidate in dict.fromkeys(candidates):  # deduplicate, preserve order
        if not Path(candidate).exists():
            continue
        try:
            res = subprocess.run(
                [candidate, "-c", check],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0 and "ok" in res.stdout:
                print(f"[INFO] Using Python: {candidate}")
                return candidate
        except Exception:
            continue

    print("[WARN] No suitable Python found – falling back to sys.executable")
    return sys.executable


PYTHON = _find_python()


def run_pose(pose: dict) -> bool:
    """Run a single pose. Returns True on success, False on failure/skip."""
    name      = pose["name"]
    arom_path = Path(pose["arom"])
    prom_path = Path(pose["prom"]) if pose["prom"] else None
    out_dir   = pose["out"]
    side      = pose.get("side", "both")

    if not arom_path.exists():
        print(f"  [SKIP] AROM not found: {arom_path}")
        return False

    cmd = [
        PYTHON, SCRIPT,
        "--arom",   str(arom_path),
        "--out",    out_dir,
        "--pose",   name,
        "--config", CONFIG,
        "--side",   side,
    ]

    if prom_path and prom_path.exists():
        cmd.extend(["--prom", str(prom_path)])
    elif prom_path:
        print(f"  [INFO] PROM not found ({prom_path}) – running AROM-only.")

    print(f"  CMD: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Exit code {exc.returncode}")
        return False
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False


def main() -> None:
    sep = "=" * 56
    print(f"\n{sep}")
    print("  MI4L Pipeline – Run All Poses")
    print(f"{sep}\n")

    results: list[tuple[str, str, str, bool]] = []

    for i, pose in enumerate(POSES, 1):
        label = f"{pose['name']} ({pose.get('side', 'both')})"
        print(f"[{i}/{len(POSES)}] {label}")
        ok = run_pose(pose)
        status = "DONE" if ok else "SKIP/ERROR"
        results.append((pose["name"], pose.get("side", "both"), pose["arom"], ok))
        print(f"  → {status}\n")

    # Summary table
    print(f"\n{sep}")
    print("  Summary")
    print(sep)
    for name, side, arom, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name:<38} [{side:<5}]  {Path(arom).name}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
