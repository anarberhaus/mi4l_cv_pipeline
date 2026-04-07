"""
Thesis-Quality Plot Regeneration
MWI Pipeline -- Alex Narberhaus Piera -- Capstone Thesis 2026

Run from repo root:
    python analysis/plot_thesis_quality.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots_final"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = OUTPUT_DIR / "master_summary.csv"

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("Set2")
COLOR_GREEN = "#66c2a5"
COLOR_ORANGE = "#fc8d62"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

STICK_MOVEMENT = "Shoulder Stick Pass-Through"


def _short(name: str) -> str:
    return (name
            .replace("Kneeling ", "")
            .replace("Standing ", "")
            .replace("Prone ", "")
            .replace("Bilateral ", "")
            .replace("Unilateral ", "")
            .replace("Shoulder ", "S. "))


def _save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", name)


# ===================================================================
# Data loading
# ===================================================================

def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    # Fill missing grip_ratio_normalized from raw grip width
    mask = df["grip_ratio_normalized"].isna() & df["grip_width_shoulder_widths"].notna()
    df.loc[mask, "grip_ratio_normalized"] = 1.0 / df.loc[mask, "grip_width_shoulder_widths"]
    log.info("Loaded master_summary.csv: %d rows, %d participants",
             len(df), df["participant_id"].nunique())
    return df


# ===================================================================
# Plot 01 -- Success Rate
# ===================================================================

def plot_01(df: pd.DataFrame):
    rows = []
    for move, grp in df.groupby("movement_name"):
        total = len(grp)
        if move == STICK_MOVEMENT:
            valid = total
        else:
            valid = (grp["mi4l_valid"] == True).sum()
        rows.append({"pose": move, "valid": valid, "total": total,
                     "pct": valid / total * 100})
    perf = pd.DataFrame(rows).sort_values("pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_GREEN if p >= 80 else COLOR_ORANGE for p in perf["pct"]]
    bars = ax.barh(perf["pose"].apply(_short), perf["pct"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(80, ls="--", color="grey", lw=1.2)
    ax.text(80.5, -0.6, "80% threshold", color="grey", fontsize=10, va="top")

    for bar, (_, row) in zip(bars, perf.iterrows()):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{row['pct']:.1f}% ({row['valid']}/{row['total']})",
                va="center", fontsize=10)

    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 110)
    ax.set_title("Pipeline Success Rate per Pose")
    fig.tight_layout()
    _save(fig, "plot_01_success_rate.png")


# ===================================================================
# Plot 02 -- MWI Distribution
# ===================================================================

def plot_02(df: pd.DataFrame):
    valid = df[(df["movement_name"] != STICK_MOVEMENT) &
               (df["mi4l_valid"] == True)].copy()
    order = (valid.groupby("movement_name")["mi4l"]
             .median().sort_values().index.tolist())
    valid["pose_short"] = valid["movement_name"].apply(_short)
    order_short = [_short(m) for m in order]
    counts = valid.groupby("movement_name").size()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=valid, y="pose_short", x="mi4l", hue="pose_short",
                order=order_short, palette="Set2", width=0.5,
                legend=False, ax=ax)
    sns.stripplot(data=valid, y="pose_short", x="mi4l", order=order_short,
                  color="0.3", size=5, alpha=0.6, jitter=0.15, ax=ax)
    ax.axvline(0.5, ls="--", color="grey", lw=1.2)
    ax.text(0.505, -0.55, "MWI = 0.5", color="grey", fontsize=10, va="top")

    for i, move in enumerate(order):
        n = counts.get(move, 0)
        ax.text(ax.get_xlim()[1] * 0.97, i, f"n={n}", va="center",
                ha="right", fontsize=10, color="0.4")

    ax.set_xlabel("MWI Score (0 = full active control, 1 = no active control)")
    ax.set_ylabel("")
    ax.set_title("MWI Score Distribution per Pose (valid trials only)")
    fig.tight_layout()
    _save(fig, "plot_02_mwi_distribution.png")


# ===================================================================
# Plot 03 -- AROM / PROM Comparison
# ===================================================================

def plot_03(df: pd.DataFrame):
    valid = df[(df["movement_name"] != STICK_MOVEMENT) &
               (df["mi4l_valid"] == True)].copy()

    melted = valid.melt(
        id_vars=["movement_name"],
        value_vars=["arom_deg", "prom_deg"],
        var_name="measure", value_name="degrees",
    )
    melted["measure"] = melted["measure"].map({"arom_deg": "AROM", "prom_deg": "PROM"})
    melted["pose_short"] = melted["movement_name"].apply(_short)
    pose_order = sorted(valid["movement_name"].unique())
    pose_short_order = [_short(m) for m in pose_order]
    counts = valid.groupby("movement_name").size()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="pose_short", y="degrees", hue="measure",
                order=pose_short_order,
                palette={"AROM": COLOR_GREEN, "PROM": COLOR_ORANGE},
                width=0.6, ax=ax)

    for i, move in enumerate(pose_order):
        n = counts.get(move, 0)
        ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center",
                fontsize=9, color="0.4")

    ax.set_xlabel("")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("AROM and PROM Distributions per Pose")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", loc="upper left")

    fig.text(0.5, -0.02,
             "Note: angle values reflect pipeline-specific geometric definitions "
             "and are not directly comparable to standard clinical normative ranges.",
             ha="center", fontsize=9, style="italic", color="0.45",
             wrap=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    _save(fig, "plot_03_arom_prom_comparison.png")


# ===================================================================
# Plot 04 -- Confidence Scores
# ===================================================================

def plot_04(df: pd.DataFrame):
    rows = []
    for move, grp in df.groupby("movement_name"):
        is_stick = move == STICK_MOVEMENT
        if is_stick:
            valid = grp
        else:
            valid = grp[grp["mi4l_valid"] == True]
        rows.append({
            "pose": move,
            "mean_arom": valid["arom_confidence"].mean(),
            "std_arom": valid["arom_confidence"].std(),
            "mean_prom": valid["prom_confidence"].mean() if not is_stick else np.nan,
            "std_prom": valid["prom_confidence"].std() if not is_stick else np.nan,
            "n": len(valid),
        })
    cdf = pd.DataFrame(rows)
    cdf["pose_short"] = cdf["pose"].apply(_short)
    x = np.arange(len(cdf))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, cdf["mean_arom"], w, yerr=cdf["std_arom"],
           label="AROM", color=COLOR_GREEN, capsize=3, edgecolor="white")
    prom_mask = cdf["mean_prom"].notna()
    ax.bar(x[prom_mask] + w / 2, cdf.loc[prom_mask, "mean_prom"], w,
           yerr=cdf.loc[prom_mask, "std_prom"],
           label="PROM", color=COLOR_ORANGE, capsize=3, edgecolor="white")
    ax.axhline(0.7, ls="--", color="grey", lw=1.2)
    ax.text(len(cdf) - 0.5, 0.705, "Minimum threshold (0.7)",
            color="grey", fontsize=10, ha="right")

    for i, row in cdf.iterrows():
        ax.text(i, -0.06, f"n={row['n']}", ha="center", fontsize=9, color="0.4")

    all_one = (cdf["mean_arom"].round(4) == 1.0).all() and (
        cdf.loc[prom_mask, "mean_prom"].round(4) == 1.0).all()
    all_zero_std = (cdf["std_arom"].fillna(0).round(4) == 0).all() and (
        cdf.loc[prom_mask, "std_prom"].fillna(0).round(4) == 0).all()
    if all_one and all_zero_std:
        ax.text(0.5, 0.55, "All estimates produced by primary top-k estimator.\n"
                "Fallback not triggered.", transform=ax.transAxes,
                ha="center", fontsize=11, style="italic", color="0.35",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(cdf["pose_short"], rotation=20, ha="right")
    ax.set_ylabel("Confidence Score (0-1)")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Mean Confidence Scores per Pose")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "plot_04_confidence_scores.png")


# ===================================================================
# Plot 05 -- Frames Valid
# ===================================================================

def plot_05(df: pd.DataFrame):
    data = df.copy()
    data["pose_short"] = data["movement_name"].apply(_short)
    pose_order = sorted(data["pose_short"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=data, x="pose_short", y="frames_valid_pct",
                hue="pose_short", order=pose_order, palette="Set2",
                width=0.5, legend=False, ax=ax)
    ax.axhline(80, ls="--", color="grey", lw=1.2)
    ax.text(len(pose_order) - 0.5, 81, "80% reference", color="grey",
            fontsize=10, ha="right")

    outliers = data[data["frames_valid_pct"] < 50]
    for _, row in outliers.iterrows():
        idx = pose_order.index(row["pose_short"])
        ax.annotate(row["participant_id"],
                    xy=(idx, row["frames_valid_pct"]),
                    xytext=(idx + 0.3, row["frames_valid_pct"] - 2),
                    fontsize=8, color="red", arrowprops=dict(
                        arrowstyle="->", color="red", lw=0.8))

    counts = data.groupby("pose_short").size()
    for i, pose in enumerate(pose_order):
        ax.text(i, ax.get_ylim()[0] + 1, f"n={counts[pose]}",
                ha="center", fontsize=9, color="0.4")

    ax.set_xlabel("")
    ax.set_ylabel("Frames Valid (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Percentage of Frames Surviving Quality Control per Pose")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "plot_05_frames_valid.png")


# ===================================================================
# Plot 06 -- Stick Pass-Through
# ===================================================================

def plot_06(df: pd.DataFrame):
    stick = df[df["movement_name"] == STICK_MOVEMENT].copy()
    grip = stick["grip_ratio_normalized"].dropna()
    if grip.empty:
        log.warning("No grip_ratio_normalized data for plot_06")
        return

    mean_val = grip.mean()

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True)

    sns.histplot(grip, kde=True, bins=8, color=PALETTE[0], edgecolor="white",
                 ax=ax_hist)
    ax_hist.axvline(1.0, ls="--", color="grey", lw=1.2)
    ax_hist.text(1.005, ax_hist.get_ylim()[1] * 0.9,
                 "Shoulder-width grip\n(perfect)", color="grey", fontsize=9,
                 va="top")
    ax_hist.axvline(mean_val, ls="--", color=PALETTE[1], lw=1.4)
    ax_hist.text(mean_val + 0.01, ax_hist.get_ylim()[1] * 0.7,
                 f"Mean = {mean_val:.2f}", color=PALETTE[1], fontsize=10,
                 va="top")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Shoulder Stick Pass-Through: Grip Ratio Distribution")
    ax_hist.text(0.98, 0.95, f"n={len(grip)}", transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=10, color="0.4")

    sns.boxplot(x=grip, color=PALETTE[0], width=0.4, ax=ax_box)
    sns.stripplot(x=grip, color="0.3", size=6, alpha=0.7, jitter=0.12,
                  ax=ax_box)
    ax_box.set_xlabel("Grip Ratio (shoulder_dist / wrist_dist)")
    ax_box.set_ylabel("")

    fig.text(0.5, -0.01,
             "Higher ratio = narrower grip relative to shoulders = better mobility",
             ha="center", fontsize=9, style="italic", color="0.45")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    _save(fig, "plot_06_stick_passthrough.png")


# ===================================================================
# Plot 07 -- Assist Gap
# ===================================================================

def plot_07(df: pd.DataFrame):
    valid = df[(df["movement_name"] != STICK_MOVEMENT) &
               (df["mi4l_valid"] == True)].copy()
    valid["pose_short"] = valid["movement_name"].apply(_short)
    order = (valid.groupby("pose_short")["assist_gap"]
             .mean().sort_values(ascending=False).index.tolist())
    counts = valid.groupby("pose_short").size()
    means = valid.groupby("pose_short")["assist_gap"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=valid, x="pose_short", y="assist_gap", hue="pose_short",
                order=order, palette="Set2", width=0.5, legend=False, ax=ax)
    sns.stripplot(data=valid, x="pose_short", y="assist_gap", order=order,
                  color="0.3", size=5, alpha=0.6, jitter=0.15, ax=ax)
    ax.axhline(0, ls="--", color="grey", lw=1.2)
    ax.text(len(order) - 0.5, 1, "No gap", color="grey", fontsize=10,
            ha="right")

    for i, pose in enumerate(order):
        n = counts.get(pose, 0)
        m = means.get(pose, 0)
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"mean={m:.1f}\nn={n}", ha="center", fontsize=9, color="0.4")

    ax.set_xlabel("")
    ax.set_ylabel("Assist Gap (degrees)")
    ax.set_title("Assist Gap per Pose (PROM minus AROM in degrees)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "plot_07_assist_gap.png")


# ===================================================================
# Summary Table
# ===================================================================

def _fmt(mean, std):
    if pd.isna(mean):
        return "--"
    return f"{mean:.1f} +/- {std:.1f}"


def summary_table(df: pd.DataFrame):
    perf_rows = []
    for move, grp in df.groupby("movement_name"):
        is_stick = move == STICK_MOVEMENT
        total = len(grp)
        if is_stick:
            valid = grp
            valid_count = total
        else:
            valid = grp[grp["mi4l_valid"] == True]
            valid_count = len(valid)

        row = {
            "Pose": move,
            "N (valid trials)": valid_count,
            "Success Rate (%)": round(valid_count / total * 100, 1),
        }

        if is_stick:
            grip = valid["grip_ratio_normalized"].dropna()
            row["Mean AROM (deg) +/- SD"] = _fmt(grip.mean(), grip.std()) + " (grip ratio)"
            row["Mean PROM (deg) +/- SD"] = "--"
            row["Mean MWI +/- SD"] = "--"
            row["Mean Assist Gap (deg) +/- SD"] = "--"
        else:
            row["Mean AROM (deg) +/- SD"] = _fmt(valid["arom_deg"].mean(),
                                                   valid["arom_deg"].std())
            row["Mean PROM (deg) +/- SD"] = _fmt(valid["prom_deg"].mean(),
                                                   valid["prom_deg"].std())
            row["Mean MWI +/- SD"] = _fmt(valid["mi4l"].mean(),
                                           valid["mi4l"].std())
            row["Mean Assist Gap (deg) +/- SD"] = _fmt(valid["assist_gap"].mean(),
                                                        valid["assist_gap"].std())

        row["Mean Confidence"] = round(valid["arom_confidence"].mean(), 2)
        perf_rows.append(row)

    table = pd.DataFrame(perf_rows)
    table.to_csv(OUTPUT_DIR / "thesis_summary_table.csv", index=False)
    log.info("Saved thesis_summary_table.csv")
    print("\n-- Thesis Summary Table --")
    print(table.to_string(index=False))
    return table


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("Thesis-Quality Plot Regeneration")
    log.info("=" * 60)

    df = load_master()

    plot_01(df)
    plot_02(df)
    plot_03(df)
    plot_04(df)
    plot_05(df)
    plot_06(df)
    plot_07(df)
    summary_table(df)

    log.info("All outputs saved to %s", PLOT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
