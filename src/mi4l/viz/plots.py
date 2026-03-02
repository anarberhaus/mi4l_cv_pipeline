"""
plots.py – Angle / distance time-series visualisation for the MI4L pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------

# Human-readable labels for each pose and what unit/quantity is plotted
_POSE_PLOT_META: dict[str, dict[str, str]] = {
    "kneeling_knee_flexion":     {"label": "Knee Flexion Angle",          "unit": "deg"},
    "prone_trunk_extension":     {"label": "Trunk Extension Angle",        "unit": "deg"},
    "standing_hip_abduction":    {"label": "Hip Abduction Angle",          "unit": "deg"},
    "bilateral_leg_straddle":    {"label": "Leg Straddle Angle",           "unit": "deg"},
    "unilateral_hip_extension":  {"label": "Hip Extension Angle",          "unit": "deg"},
    "shoulder_flexion":          {"label": "Shoulder Flexion Angle",       "unit": "deg"},
    "shoulder_stick_pass_through": {"label": "Stick Pass-Through Width",   "unit": "norm"},
}

_KIND_LABELS: dict[str, str] = {
    "arom": "AROM (Active Range of Motion)",
    "prom": "PROM (Passive Range of Motion)",
}

_SIDE_LABELS: dict[str, str] = {
    "left":  "Left",
    "right": "Right",
    "both":  "Both Sides",
}


def _build_title(
    pose_name: str,
    kind: str,
    side: str | None,
) -> tuple[str, str]:
    """Return (main_title, subtitle) for the plot."""
    meta = _POSE_PLOT_META.get(pose_name, {"label": pose_name.replace("_", " ").title(), "unit": "deg"})
    label = meta["label"]

    kind_label = _KIND_LABELS.get(kind.lower(), kind.upper())
    if side and side not in ("both", None):
        side_str = _SIDE_LABELS.get(side, side.capitalize())
        subtitle = f"{kind_label}  ·  {side_str} Side"
    else:
        subtitle = kind_label

    return label, subtitle


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_knee_angles(
    angles_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "",           # legacy param – ignored when pose_name is given
    dpi: int = 150,
    robust_frames: dict | None = None,
    side: str | None = None,   # "left", "right", "both", or None
    pose_name: str | None = None,
    kind: str = "arom",        # "arom" or "prom"
) -> None:
    try:
        import matplotlib.pyplot as plt          # type: ignore
        import matplotlib.ticker as ticker       # type: ignore
    except Exception as e:
        raise ImportError("matplotlib is required for plots (export.plots.enabled=true).") from e

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # ── Colours & style ────────────────────────────────────────────────────
    left_color   = "#3A86FF"   # vivid blue
    right_color  = "#FF6B35"   # warm orange
    single_color = "#2D9E4A"   # forest green
    robust_fill  = "#BDBDBD"   # light grey for peak-hold region
    robust_alpha = 0.30

    lw = 2.2

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Reserve space at the top for title + legend header band
    # top=0.78 means the axes occupies up to 78% of figure height from bottom;
    # the remaining 22% on top is the header area used by title and legend.
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.09, right=0.98)

    # Subtle grid – horizontal only
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # Trim spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")

    t = angles_df["time_sec"].to_numpy(dtype=float)

    # ── Detect value columns ───────────────────────────────────────────────
    value_cols = [
        c for c in angles_df.columns
        if (c.endswith("_deg") or c.endswith("_dist_norm")) and c != "time_sec"
    ]
    left_cols  = [c for c in value_cols if c.startswith("left_")]
    right_cols = [c for c in value_cols if c.startswith("right_")]
    other_cols = [c for c in value_cols if c not in left_cols and c not in right_cols]

    plot_left  = (side is None or side == "left")  and len(left_cols)  > 0
    plot_right = (side is None or side == "right") and len(right_cols) > 0
    plot_other = len(other_cols) > 0

    # ── Robust-window shading ──────────────────────────────────────────────
    if robust_frames and isinstance(robust_frames, dict):
        for side_key, idxs in robust_frames.items():
            if side is not None and side_key not in (side, "bilateral", "both"):
                continue
            if not idxs:
                continue
            try:
                times = angles_df.loc[idxs, "time_sec"].to_numpy(dtype=float)
            except Exception:
                times = angles_df["time_sec"].iloc[idxs].to_numpy(dtype=float)
            if times.size > 0:
                ax.axvspan(
                    float(times.min()), float(times.max()),
                    alpha=robust_alpha, color=robust_fill,
                    label="Peak hold window", zorder=1,
                )

    # ── Angle / distance lines ─────────────────────────────────────────────
    legend_handles = []

    if plot_left:
        for col in left_cols:
            line, = ax.plot(
                t, angles_df[col].to_numpy(dtype=float),
                label="Left", color=left_color, linewidth=lw, zorder=3,
            )
            legend_handles.append(line)

    if plot_right:
        for col in right_cols:
            line, = ax.plot(
                t, angles_df[col].to_numpy(dtype=float),
                label="Right", color=right_color, linewidth=lw, zorder=3,
            )
            legend_handles.append(line)

    if plot_other:
        for col in other_cols:
            lbl = _SIDE_LABELS.get(side or "both", "Both Sides") if not plot_left and not plot_right else col.replace("_deg", "").replace("_dist_norm", "").replace("_", " ").title()
            line, = ax.plot(
                t, angles_df[col].to_numpy(dtype=float),
                label=lbl, color=single_color, linewidth=lw, zorder=3,
            )
            legend_handles.append(line)

    # ── Axis labels ────────────────────────────────────────────────────────
    has_dist = any(c.endswith("_dist_norm") for c in value_cols)
    has_deg  = any(c.endswith("_deg")       for c in value_cols)

    ax.set_xlabel("Time (s)", fontsize=12, labelpad=6)
    if has_dist and not has_deg:
        ax.set_ylabel("Width (shoulder-widths)", fontsize=12, labelpad=6)
    elif has_dist and has_deg:
        ax.set_ylabel("Value", fontsize=12, labelpad=6)
    else:
        ax.set_ylabel("Angle (°)", fontsize=12, labelpad=6)

    # ── Title (left-aligned in figure coords) ─────────────────────────────
    if pose_name:
        main_title, subtitle = _build_title(pose_name, kind, side)
    else:
        raw = title.replace("_", " ").replace("(deg)", "").replace(" -", " ·").strip()
        main_title = raw
        subtitle = ""

    # Use figure text so the title sits in the header band, not the axes box
    fig.text(
        0.09, 0.97, main_title,
        fontsize=15, fontweight="bold",
        ha="left", va="top",
        transform=fig.transFigure,
    )
    if subtitle:
        fig.text(
            0.09, 0.90, subtitle,
            fontsize=10, color="#555555",
            ha="left", va="top",
            transform=fig.transFigure,
        )

    # ── Legend – placed in the header band, right-aligned ─────────────────
    # Collect all artist handles/labels from the axes (lines + patch for the
    # robust window), then render a figure-level legend well above the plot.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles, labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.99),   # top-right corner of figure
            bbox_transform=fig.transFigure,
            fontsize=10,
            title="Side" if (plot_left or plot_right) else None,
            title_fontsize=10,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            frameon=True,
        )
        leg.get_frame().set_linewidth(0.8)

    # ── Save ───────────────────────────────────────────────────────────────
    fig.savefig(p, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
