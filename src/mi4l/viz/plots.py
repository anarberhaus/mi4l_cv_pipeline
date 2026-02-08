from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_knee_angles(
    angles_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    dpi: int = 150,
    robust_frames: dict | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError("matplotlib is required for plots (export.plots.enabled=true).") from e

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    t = angles_df["time_sec"].to_numpy(dtype=float)
    plt.figure(figsize=(8, 4))
    # aesthetics
    left_color = "#1f77b4"
    right_color = "#ff7f0e"
    lw = 2.5

    if "left_knee_flexion_deg" in angles_df.columns:
        plt.plot(t, angles_df["left_knee_flexion_deg"].to_numpy(dtype=float), label="Left", color=left_color, linewidth=lw)
    if "right_knee_flexion_deg" in angles_df.columns:
        plt.plot(t, angles_df["right_knee_flexion_deg"].to_numpy(dtype=float), label="Right", color=right_color, linewidth=lw)

    # Shade robust frames if provided (expects dict with 'left'/'right' list of indices into angles_df)
    if robust_frames is not None and isinstance(robust_frames, dict):
        for side, idxs in robust_frames.items():
            if not idxs:
                continue
            # Map indices to times
            try:
                times = angles_df.loc[idxs, "time_sec"].to_numpy(dtype=float)
            except Exception:
                # fallback: if idxs are positions
                times = angles_df["time_sec"].iloc[idxs].to_numpy(dtype=float)
            if times.size > 0:
                t0 = float(times.min())
                t1 = float(times.max())
                plt.axvspan(t0, t1, alpha=0.15, color=(0.2, 0.2, 0.2))

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.legend(title="Side")
    plt.tight_layout()
    plt.savefig(p, dpi=int(dpi))
    plt.close()
