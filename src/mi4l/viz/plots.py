from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_knee_angles(
    angles_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    dpi: int = 150,
    robust_frames: dict | None = None,
    side: str | None = None,  # "left", "right", or None (both)
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
    single_color = "#2ca02c"  # green for non-sided metrics
    lw = 2.5

    # Auto-detect angle columns (ending with _deg, excluding time_sec)
    angle_cols = [col for col in angles_df.columns if col.endswith("_deg") and col != "time_sec"]
    
    # Separate into left, right, and non-sided columns
    left_cols = [col for col in angle_cols if col.startswith("left_")]
    right_cols = [col for col in angle_cols if col.startswith("right_")]
    other_cols = [col for col in angle_cols if col not in left_cols and col not in right_cols]
    
    # Determine which sides to plot
    plot_left = (side is None or side == "left") and len(left_cols) > 0
    plot_right = (side is None or side == "right") and len(right_cols) > 0
    plot_other = len(other_cols) > 0

    # Plot left side columns
    if plot_left:
        for col in left_cols:
            plt.plot(t, angles_df[col].to_numpy(dtype=float), label="Left", color=left_color, linewidth=lw)
    
    # Plot right side columns
    if plot_right:
        for col in right_cols:
            plt.plot(t, angles_df[col].to_numpy(dtype=float), label="Right", color=right_color, linewidth=lw)
    
    # Plot non-sided columns (e.g., trunk_extension_deg)
    if plot_other:
        for col in other_cols:
            # Extract a clean label from column name
            label = col.replace("_deg", "").replace("_", " ").title()
            plt.plot(t, angles_df[col].to_numpy(dtype=float), label=label, color=single_color, linewidth=lw)

    # Shade robust frames if provided (expects dict with 'left'/'right'/'bilateral'/'both' list of indices into angles_df)
    if robust_frames is not None and isinstance(robust_frames, dict):
        for side_key, idxs in robust_frames.items():
            # For bilateral poses, side_key might be 'bilateral' or 'both'
            # Only shade if this side is being plotted
            if side is not None:
                # If plotting a specific side, only shade that side's frames
                if side_key not in [side, "bilateral", "both"]:
                    continue
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
    
    # Only show legend if plotting multiple series or both sides
    if (plot_left and plot_right) or len(angle_cols) > 1:
        plt.legend(title="Side" if (plot_left or plot_right) else None)
    
    plt.tight_layout()
    plt.savefig(p, dpi=int(dpi))
    plt.close()
