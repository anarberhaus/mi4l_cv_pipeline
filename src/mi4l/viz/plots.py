from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_knee_angles(angles_df: pd.DataFrame, out_path: str | Path, title: str, dpi: int = 150) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError("matplotlib is required for plots (export.plots.enabled=true).") from e

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    t = angles_df["time_sec"].to_numpy(dtype=float)

    plt.figure()
    if "left_knee_flexion_deg" in angles_df.columns:
        plt.plot(t, angles_df["left_knee_flexion_deg"].to_numpy(dtype=float), label="Left knee")
    if "right_knee_flexion_deg" in angles_df.columns:
        plt.plot(t, angles_df["right_knee_flexion_deg"].to_numpy(dtype=float), label="Right knee")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=int(dpi))
    plt.close()
