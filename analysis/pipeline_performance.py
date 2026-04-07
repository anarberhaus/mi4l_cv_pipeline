"""
Pipeline Performance & Physiological Plausibility Analysis
MWI Pipeline — Alex Narberhaus Piera — Capstone Thesis 2026

Run from repo root:
    python analysis/pipeline_performance.py
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style (spec: 10x6 min, DPI 300, Set2, white bg, light grey grid) ─
PALETTE = sns.color_palette("Set2")
sns.set_theme(style="whitegrid", font_scale=1.15, rc={
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def _apply_spec_style(fig, ax):
    """Apply spec styling: remove top/right spines, light grey grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.4)


# ═════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════

STANDARD_COLS = [
    "movement_name", "joint_name", "angle_type", "side",
    "arom_deg", "prom_deg", "mi4l",
    "arom_confidence", "prom_confidence", "mi4l_valid",
    "qc_flags", "assist_gap",
    "peak_hold_time_s", "peak_band_std_deg", "time_to_peak_s",
    "fit_r2", "fit_rmse_deg", "jerk_rms",
    "torso_angle_change_deg", "pelvis_drift_norm",
    "frames_valid_pct", "avg_landmark_visibility",
]

STICK_COLS = [
    "movement_name", "joint_name", "angle_type", "side",
    "grip_width_shoulder_widths", "grip_ratio_normalized",
    "confidence", "plateau_duration_s", "qc_flags",
    "frames_valid_pct", "avg_landmark_visibility",
]


# Top-level names under results/ that must never be scanned (e.g. app/session exports)
SKIP_RESULT_ROOTS = frozenset({"assessments"})


def _iter_result_batch_dirs() -> list[Path]:
    """All direct subfolders of results/ to scan, excluding assessments and hidden dirs."""
    if not RESULTS_DIR.exists():
        return []
    out: list[Path] = []
    for p in sorted(RESULTS_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if p.name in SKIP_RESULT_ROOTS:
            log.info("Skipping results root (excluded): %s", p.name)
            continue
        out.append(p)
    return out


def collect_summaries() -> pd.DataFrame:
    """Scan each batch folder under results/ (except assessments/) for summary.csv files; combine into one df.

    No deduplication across batches — rows are tagged with batch_id.
    Missing batch roots are simply not present (no hard-coded batch list).
    """
    frames: list[pd.DataFrame] = []
    skipped = 0

    batch_dirs = _iter_result_batch_dirs()
    if not batch_dirs:
        log.error("No batch folders found under %s (after exclusions).", RESULTS_DIR)
        raise SystemExit(1)

    for batch_dir in batch_dirs:
        batch_id = batch_dir.name

        for csv_path in sorted(batch_dir.rglob("summary.csv")):
            if "assessments" in csv_path.parts:
                continue
            # Skip all_participants_summary.csv or other top-level CSVs
            rel = csv_path.relative_to(batch_dir)
            parts = rel.parts
            if len(parts) < 3:
                continue

            participant_id = parts[0]

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                log.warning("Could not read %s: %s", csv_path, exc)
                skipped += 1
                continue

            has_stick_schema = "grip_ratio_normalized" in df.columns

            if has_stick_schema:
                df = df.rename(columns={"confidence": "arom_confidence"})
                df["prom_deg"] = np.nan
                df["mi4l"] = np.nan
                df["prom_confidence"] = np.nan
                df["mi4l_valid"] = False
                df["assist_gap"] = np.nan
                df["arom_deg"] = np.nan
            else:
                df["grip_ratio_normalized"] = np.nan
                df["grip_width_shoulder_widths"] = np.nan
                is_stick_movement = (
                    df["movement_name"].str.contains("Stick Pass-Through", case=False, na=False)
                )
                if is_stick_movement.any():
                    raw_vals = df.loc[is_stick_movement, "arom_deg"]
                    normalised = raw_vals.where(raw_vals <= 1.0, other=np.nan)
                    df.loc[is_stick_movement, "grip_ratio_normalized"] = normalised
                    df.loc[is_stick_movement, "grip_width_shoulder_widths"] = raw_vals.where(
                        raw_vals > 1.0, other=np.nan
                    )

            df["participant_id"] = participant_id
            df["batch_id"] = batch_id
            frames.append(df)

            label = "stick" if has_stick_schema else "standard"
            log.info("  + %s  [%s]  %s (%s)", participant_id, df["movement_name"].iloc[0], batch_id, label)

    if not frames:
        log.error("No summary.csv files found in batch folders!")
        raise SystemExit(1)

    master = pd.concat(frames, ignore_index=True)

    n_participants = master["participant_id"].nunique()
    n_trials = len(master)
    log.info("Collected %d trials from %d participants (skipped %d)", n_trials, n_participants, skipped)

    print("\n-- Data Collection Summary --")
    print(f"Total participants: {n_participants}")
    print(f"Total trials: {n_trials}")
    print("By batch:")
    for bid, grp in master.groupby("batch_id"):
        print(f"  {bid}: {len(grp)} trials, {grp['participant_id'].nunique()} participants")
    print("By pose:")
    for move, grp in master.groupby("movement_name"):
        print(f"  {move}: {len(grp)} trials")

    return master


# ═════════════════════════════════════════════════════════════════════════
# 2. PIPELINE PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════

def compute_performance(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for move, grp in master.groupby("movement_name"):
        total = len(grp)

        is_stick = move == "Shoulder Stick Pass-Through"
        if is_stick:
            valid = grp  # stick doesn't use mi4l_valid
            invalid = grp.iloc[0:0]
        else:
            valid = grp[grp["mi4l_valid"] == True]
            invalid = grp[grp["mi4l_valid"] != True]

        valid_count = len(valid)
        success_pct = valid_count / total * 100 if total else 0

        def _flag_pct(subset, flag):
            if len(subset) == 0:
                return 0.0
            has_flag = subset["qc_flags"].fillna("").str.contains(flag)
            return has_flag.sum() / len(subset) * 100

        # Top failure reasons among invalid trials
        failure_reasons = ""
        if len(invalid) > 0:
            flags_exploded = (
                invalid["qc_flags"].dropna()
                .str.split(";")
                .explode()
                .str.strip()
            )
            flags_exploded = flags_exploded[flags_exploded != ""]
            if len(flags_exploded):
                top3 = flags_exploded.value_counts().head(3).index.tolist()
                failure_reasons = "; ".join(top3)

        rows.append({
            "movement_name": move,
            "total_trials": total,
            "valid_trials": valid_count,
            "success_rate_pct": round(success_pct, 1),
            "mean_arom_confidence": round(valid["arom_confidence"].mean(), 4),
            "std_arom_confidence": round(valid["arom_confidence"].std(), 4),
            "mean_prom_confidence": round(valid["prom_confidence"].mean(), 4) if not is_stick else np.nan,
            "std_prom_confidence": round(valid["prom_confidence"].std(), 4) if not is_stick else np.nan,
            "mean_frames_valid_pct": round(grp["frames_valid_pct"].mean(), 2),
            "fallback_used_pct": round(_flag_pct(grp, "fallback_movement_window"), 1),
            "top_failure_reasons": failure_reasons,
        })

    perf = pd.DataFrame(rows)
    log.info("Performance metrics computed for %d poses", len(perf))
    return perf


# ═════════════════════════════════════════════════════════════════════════
# 3. PHYSIOLOGICAL PLAUSIBILITY CHECK
# ═════════════════════════════════════════════════════════════════════════

def plausibility_check(master: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats only; no normative range comparison."""
    rows = []
    for move, grp in master.groupby("movement_name"):
        is_stick = move == "Shoulder Stick Pass-Through"

        if is_stick:
            valid = grp
        else:
            valid = grp[grp["mi4l_valid"] == True]

        if valid.empty:
            continue

        row = {"movement_name": move, "n_valid_trials": len(valid)}

        if is_stick:
            grip = valid["grip_ratio_normalized"].dropna()
            row["mean_grip_ratio"] = round(grip.mean(), 4) if len(grip) else np.nan
            row["std_grip_ratio"] = round(grip.std(), 4) if len(grip) else np.nan
            row["mean_arom_deg"] = np.nan
            row["std_arom_deg"] = np.nan
            row["mean_prom_deg"] = np.nan
            row["std_prom_deg"] = np.nan
        else:
            row["mean_grip_ratio"] = np.nan
            row["std_grip_ratio"] = np.nan
            row["mean_arom_deg"] = round(valid["arom_deg"].mean(), 2)
            row["std_arom_deg"] = round(valid["arom_deg"].std(), 2)
            row["mean_prom_deg"] = round(valid["prom_deg"].mean(), 2)
            row["std_prom_deg"] = round(valid["prom_deg"].std(), 2)

        rows.append(row)

    plaus = pd.DataFrame(rows)
    log.info("Plausibility check completed for %d poses", len(plaus))
    return plaus


# ═════════════════════════════════════════════════════════════════════════
# 4. MWI SCORE DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════

def mwi_distribution(master: pd.DataFrame) -> pd.DataFrame:
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    valid = angle_df[angle_df["mi4l_valid"] == True]

    rows = []
    for move, grp in valid.groupby("movement_name"):
        mwi = grp["mi4l"]
        ag = grp["assist_gap"]
        rows.append({
            "movement_name": move,
            "n": len(grp),
            "mwi_mean": round(mwi.mean(), 4),
            "mwi_std": round(mwi.std(), 4),
            "mwi_min": round(mwi.min(), 4),
            "mwi_max": round(mwi.max(), 4),
            "count_low": int((mwi < 0.2).sum()),
            "count_mid": int(((mwi >= 0.2) & (mwi <= 0.8)).sum()),
            "count_high": int((mwi > 0.8).sum()),
            "assist_gap_mean": round(ag.mean(), 2),
            "assist_gap_std": round(ag.std(), 2),
        })

    dist = pd.DataFrame(rows)
    log.info("MWI distribution computed for %d poses", len(dist))
    return dist


# ═════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════

def _short_name(name: str) -> str:
    return (name
            .replace("Kneeling ", "")
            .replace("Standing ", "")
            .replace("Prone ", "")
            .replace("Bilateral ", "")
            .replace("Unilateral ", "")
            .replace("Shoulder ", "S. "))


def _canonical_pose_order(master: pd.DataFrame) -> list[str]:
    """Sorted full movement names — use for x-axis category order across comparable plots."""
    return sorted(master["movement_name"].unique())


def participant_alias_map(master: pd.DataFrame) -> dict[str, str]:
    """Stable P1, P2, … labels for anonymising participant_id in figures."""
    ids = sorted(master["participant_id"].astype(str).unique())
    return {pid: f"P{i + 1}" for i, pid in enumerate(ids)}


def _sort_p_labels(labels: list[str]) -> list[str]:
    """Natural order P2 < P10 (string sort would put P10 before P2)."""

    def key(lab: str) -> tuple[int, str]:
        if lab.startswith("P") and lab[1:].isdigit():
            return (int(lab[1:]), lab)
        return (9999, lab)

    return sorted(labels, key=key)


def _xaxis_pose_labels(ax, rotation: int = 28):
    """Consistent tilted pose labels (readable, same across figures)."""
    ax.tick_params(axis="x", rotation=rotation)
    plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")


# AROM / PROM colours aligned with plot_03 (thesis consistency)
COLOR_AROM = "#4caf50"
COLOR_PROM = "#ff9800"


def plot_01_success_rate(perf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = perf.sort_values("success_rate_pct", ascending=True)
    colors = ["#4caf50" if v >= 80 else "#ff9800" for v in df["success_rate_pct"]]
    bars = ax.barh(df["movement_name"].apply(_short_name), df["success_rate_pct"], color=colors)
    for i, (bar, (_, row)) in enumerate(zip(bars, df.iterrows())):
        pct = row["success_rate_pct"]
        valid = int(row["valid_trials"])
        total = int(row["total_trials"])
        ax.text(pct + 1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}% ({valid}/{total})",
                va="center", fontsize=10)
    ax.axvline(80, ls="--", color="grey", lw=1.2, label="80% threshold")
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Pipeline success rate per pose")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    _apply_spec_style(fig, ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_01_success_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_01_success_rate.png")


def plot_02_mwi_distribution(master: pd.DataFrame, pose_order: list[str] | None = None):
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    valid = angle_df[angle_df["mi4l_valid"] == True].copy()
    if valid.empty:
        log.warning("No valid MWI data for plot_02")
        return

    # Same pose order as other per-pose plots (easier cross-figure comparison)
    if pose_order is None:
        pose_order = _canonical_pose_order(master)
    pose_order = [m for m in pose_order if m != "Shoulder Stick Pass-Through"]
    valid["pose_short"] = valid["movement_name"].apply(_short_name)
    order_short = [_short_name(m) for m in pose_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = sns.boxplot(data=valid, y="pose_short", x="mi4l", hue="pose_short",
                     order=order_short, palette="Set2", width=0.5, legend=False, ax=ax)
    sns.stripplot(data=valid, y="pose_short", x="mi4l", order=order_short,
                  color="0.3", size=5, alpha=0.6, jitter=0.15, ax=ax)
    for i, pose in enumerate(order_short):
        n = (valid["pose_short"] == pose).sum()
        ax.text(1.02, i, f"n={n}", va="center", fontsize=10, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, ls="--", color="grey", lw=1.2, label="MWI = 0.5")
    ax.set_xlabel("MWI score (0 = full active control, 1 = no active control)")
    ax.set_ylabel("")
    ax.set_title("MWI score distribution per pose (valid trials only)")
    ax.legend(loc="upper right")
    _apply_spec_style(fig, ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_02_mwi_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_02_mwi_distribution.png")


def plot_03_arom_prom(master: pd.DataFrame, pose_order: list[str] | None = None):
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    valid = angle_df[angle_df["mi4l_valid"] == True].copy()
    if valid.empty:
        log.warning("No valid angle data for plot_03")
        return

    melted = valid.melt(
        id_vars=["movement_name"],
        value_vars=["arom_deg", "prom_deg"],
        var_name="measure",
        value_name="degrees",
    )
    melted["measure"] = melted["measure"].map({"arom_deg": "AROM", "prom_deg": "PROM"})
    melted["pose_short"] = melted["movement_name"].apply(_short_name)
    if pose_order is None:
        pose_order = _canonical_pose_order(master)
    pose_order = [m for m in pose_order if m != "Shoulder Stick Pass-Through"]
    pose_order_short = [_short_name(m) for m in pose_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="pose_short", y="degrees", hue="measure",
                order=pose_order_short, palette={"AROM": COLOR_AROM, "PROM": COLOR_PROM},
                width=0.62, ax=ax, linewidth=1.0)

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.03
    for i, pose in enumerate(pose_order_short):
        sub = melted[melted["pose_short"] == pose]
        n = len(sub) // 2
        ax.text(i, ymin + pad, f"n={n}", ha="center", fontsize=10, va="bottom")

    ax.set_xlabel("")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("AROM and PROM distributions per pose")
    _xaxis_pose_labels(ax)
    ax.legend(title="", loc="upper right")
    fig.text(0.5, -0.06,
             "Note: angle values reflect pipeline-specific geometric definitions "
             "and are not directly comparable to standard clinical normative ranges.",
             ha="center", fontsize=10, wrap=True, transform=fig.transFigure)
    _apply_spec_style(fig, ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(PLOT_DIR / "plot_03_arom_prom_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_03_arom_prom_comparison.png")


def plot_04_confidence(perf: pd.DataFrame, pose_order: list[str] | None = None):
    # X-axis: pose names only (aggregated metrics); no participant labels to anonymise.
    df = perf.dropna(subset=["mean_arom_confidence"]).copy()
    if pose_order is not None:
        order_map = {m: i for i, m in enumerate(pose_order)}
        df["_k"] = df["movement_name"].map(lambda m: order_map.get(m, 999))
        df = df.sort_values("_k").drop(columns="_k")
    df["pose_short"] = df["movement_name"].apply(_short_name)
    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df["mean_arom_confidence"], w, yerr=df["std_arom_confidence"],
           label="AROM", color=COLOR_AROM, capsize=3, edgecolor="white", linewidth=0.5)
    prom_mask = df["mean_prom_confidence"].notna()
    ax.bar(x[prom_mask] + w / 2,
           df.loc[prom_mask, "mean_prom_confidence"], w,
           yerr=df.loc[prom_mask, "std_prom_confidence"],
           label="PROM", color=COLOR_PROM, capsize=3, edgecolor="white", linewidth=0.5)
    ax.axhline(0.7, ls="--", color="grey", lw=1.2, label="Minimum threshold (0.7)")
    if (df["mean_arom_confidence"] == 1.0).all() and (df["std_arom_confidence"].fillna(0) == 0).all():
        ax.text(0.5, 0.95, "All estimates produced by primary top-k estimator. Fallback not triggered.",
                ha="center", va="top", transform=ax.transAxes, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pose_short"])
    ax.set_ylabel("Confidence score (0–1)")
    ax.set_title("Mean confidence scores per pose")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    _apply_spec_style(fig, ax)
    _xaxis_pose_labels(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_04_confidence_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_04_confidence_scores.png")


def plot_05_frames_valid(master: pd.DataFrame, pose_order: list[str] | None = None):
    df = master.copy()
    df["pose_short"] = df["movement_name"].apply(_short_name)
    if pose_order is None:
        pose_order = _canonical_pose_order(master)
    pose_order_short = [_short_name(m) for m in pose_order]
    alias = participant_alias_map(master)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="pose_short", y="frames_valid_pct", hue="pose_short",
                order=pose_order_short, palette="Set2", width=0.52, legend=False, ax=ax)
    # Label each low-QC trial at its actual frames_valid_pct (not "worst per participant only"),
    # so e.g. P10 stays beside the ~25% point when that participant also has a 0% trial.
    bad = df[df["frames_valid_pct"] < 50].copy()
    if not bad.empty:
        bad = bad.drop_duplicates(
            subset=["pose_short", "participant_id", "frames_valid_pct"],
            keep="first",
        )
        by_anchor: dict[tuple[int, float], list[str]] = defaultdict(list)
        for _, row in bad.iterrows():
            idx = pose_order_short.index(row["pose_short"]) if row["pose_short"] in pose_order_short else 0
            y = round(float(row["frames_valid_pct"]), 4)
            label = alias.get(str(row["participant_id"]), str(row["participant_id"]))
            by_anchor[(idx, y)].append(label)
        for (idx, y), labels in sorted(by_anchor.items(), key=lambda t: (t[0][0], t[0][1])):
            uniq = _sort_p_labels(list(dict.fromkeys(labels)))
            # Several participants can share the same % (e.g. three at 0%) and sit on one marker;
            # one comma-separated label avoids staggered P1/P5/P10 overlapping as garbled text.
            text = ", ".join(uniq)
            ax.annotate(
                text,
                (idx, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.88,
                annotation_clip=False,
            )
    ax.axhline(80, ls="--", color="grey", lw=1.2, label="80% reference")
    ax.set_xlabel("")
    ax.set_ylabel("Frames valid (%)")
    ax.set_title("Percentage of frames surviving quality control per pose")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    _apply_spec_style(fig, ax)
    _xaxis_pose_labels(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_05_frames_valid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_05_frames_valid.png")


def plot_06_stick_passthrough(master: pd.DataFrame):
    stick = master[master["movement_name"] == "Shoulder Stick Pass-Through"]
    if "grip_width_shoulder_widths" in stick.columns:
        vals = stick["grip_width_shoulder_widths"].dropna()
    elif "grip_ratio_normalized" in stick.columns:
        vals = (1.0 / stick["grip_ratio_normalized"].replace(0, np.nan)).dropna()
    else:
        log.warning("No grip column found for plot_06")
        return

    if vals.empty:
        log.warning("No stick pass-through data for plot_06")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    ax1.hist(vals, bins=15, density=True, alpha=0.7, color=PALETTE[0], edgecolor="white")
    vals.plot.kde(ax=ax1, color="darkgreen", lw=2)
    mean_val = vals.mean()
    ax1.axvline(1.0, ls="--", color="grey", lw=1.2, label="Shoulder-width grip")  # noqa: PLC0415
    ax1.axvline(mean_val, ls="--", color="orange", lw=1.2, label=f"Mean ({mean_val:.2f})")
    ax1.set_xlabel("Grip Ratio (higher = better shoulder mobility)")
    ax1.set_ylabel("Density")
    ax1.set_title("Shoulder Stick Pass-Through: Grip Ratio Distribution")
    ax1.legend()
    ax1.set_xlim(0, None)
    _apply_spec_style(fig, ax1)

    ax2.boxplot([vals], tick_labels=["Stick Pass-Through"])
    ax2.set_ylabel("Grip Ratio")
    ax2.set_title(f"Grip Ratio (n={len(vals)})")
    ax2.axhline(1.0, ls="--", color="grey", alpha=0.6)
    _apply_spec_style(fig, ax2)

    fig.text(0.5, -0.02, "Ratio of 1.0 = grip exactly at shoulder width", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_06_stick_passthrough.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_06_stick_passthrough.png")


def plot_07_assist_gap(master: pd.DataFrame):
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    angle_df = angle_df.dropna(subset=["assist_gap"])

    if angle_df.empty:
        log.warning("No assist_gap data for plot_07")
        return

    order = angle_df.groupby("movement_name")["assist_gap"].mean().sort_values(ascending=False).index.tolist()
    angle_df["pose_short"] = angle_df["movement_name"].apply(_short_name)
    order_short = [_short_name(m) for m in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=angle_df, x="pose_short", y="assist_gap", hue="pose_short",
                order=order_short, palette="Set2", width=0.5, legend=False, ax=ax)
    sns.stripplot(data=angle_df, x="pose_short", y="assist_gap", order=order_short,
                  color="0.3", size=5, alpha=0.6, jitter=0.15, ax=ax)
    for i, pose in enumerate(order_short):
        sub = angle_df[angle_df["pose_short"] == pose]
        n = len(sub)
        mean_val = sub["assist_gap"].mean()
        ax.text(i, ax.get_ylim()[1] * 0.84, f"mean={mean_val:.1f}\nn={n}", ha="center", fontsize=10)
    ax.axhline(0, ls="--", color="grey", lw=1.2, label="No gap")
    ax.set_xlabel("")
    ax.set_ylabel("Assist gap (degrees)")
    ax.set_title("Assist gap per pose (PROM minus AROM)", pad=16)
    _apply_spec_style(fig, ax)
    _xaxis_pose_labels(ax)
    ax.legend(loc="upper right")
    fig.text(
        0.5, -0.01,
        "n = trials with assist_gap per pose (after QC). n differs across poses if videos are missing, "
        "trials fail QC, or a pose was not recorded for all participants.",
        ha="center", fontsize=9, wrap=True, transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(PLOT_DIR / "plot_07_assist_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_07_assist_gap.png")


def write_thesis_summary_table(perf: pd.DataFrame, plaus: pd.DataFrame, dist: pd.DataFrame, master: pd.DataFrame):
    """Build thesis_summary_table.csv with Pose, N, Success Rate, Mean ± SD columns."""
    rows = []
    for _, p_row in perf.iterrows():
        move = p_row["movement_name"]
        is_stick = move == "Shoulder Stick Pass-Through"

        pl = plaus[plaus["movement_name"] == move].iloc[0] if move in plaus["movement_name"].values else None

        row = {
            "Pose": move,
            "N valid trials": int(p_row["valid_trials"]),
            "Success Rate (%)": round(p_row["success_rate_pct"], 1),
        }

        if is_stick:
            if pl is not None and pd.notna(pl.get("mean_grip_ratio")):
                m, s = pl["mean_grip_ratio"], pl["std_grip_ratio"]
                row["Mean AROM ± SD"] = f"{m:.2f} ± {s:.2f}" if pd.notna(s) else f"{m:.2f}"
                row["Mean PROM ± SD"] = ""
            else:
                row["Mean AROM ± SD"] = ""
                row["Mean PROM ± SD"] = ""
            row["Mean MWI ± SD"] = ""
            row["Mean Assist Gap ± SD"] = ""
        else:
            if pl is not None:
                ma, sa = pl["mean_arom_deg"], pl["std_arom_deg"]
                mp, sp = pl["mean_prom_deg"], pl["std_prom_deg"]
                row["Mean AROM ± SD"] = f"{ma:.1f} ± {sa:.1f}" if pd.notna(sa) else f"{ma:.1f}"
                row["Mean PROM ± SD"] = f"{mp:.1f} ± {sp:.1f}" if pd.notna(sp) else f"{mp:.1f}"
            else:
                row["Mean AROM ± SD"] = ""
                row["Mean PROM ± SD"] = ""

            d = dist[dist["movement_name"] == move]
            if not d.empty:
                mw, sw = d["mwi_mean"].iloc[0], d["mwi_std"].iloc[0]
                ag_m, ag_s = d["assist_gap_mean"].iloc[0], d["assist_gap_std"].iloc[0]
                row["Mean MWI ± SD"] = f"{mw:.2f} ± {sw:.2f}" if pd.notna(sw) else f"{mw:.2f}"
                row["Mean Assist Gap ± SD"] = f"{ag_m:.1f} ± {ag_s:.1f}" if pd.notna(ag_s) else f"{ag_m:.1f}"
            else:
                row["Mean MWI ± SD"] = ""
                row["Mean Assist Gap ± SD"] = ""

        conf = p_row.get("mean_arom_confidence", np.nan)
        row["Mean Confidence"] = round(conf, 3) if pd.notna(conf) else ""

        rows.append(row)

    tbl = pd.DataFrame(rows)
    out = OUTPUT_DIR / "thesis_summary_table.csv"
    tbl.to_csv(out, index=False)
    log.info("Saved %s", out)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Pipeline Performance & Plausibility Analysis")
    log.info("=" * 60)

    # 1 — Collect
    master = collect_summaries()
    master.to_csv(OUTPUT_DIR / "master_summary.csv", index=False)
    log.info("Saved master_summary.csv  (%d rows)", len(master))

    # 2 — Performance
    perf = compute_performance(master)
    perf.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)
    log.info("Saved performance_metrics.csv")
    print("\n-- Performance Metrics --")
    print(perf.to_string(index=False))

    # 3 — Plausibility
    plaus = plausibility_check(master)
    plaus.to_csv(OUTPUT_DIR / "plausibility_check.csv", index=False)
    log.info("Saved plausibility_check.csv")
    print("\n-- Plausibility Check --")
    print(plaus.to_string(index=False))

    # 4 — MWI distribution
    dist = mwi_distribution(master)
    dist.to_csv(OUTPUT_DIR / "mwi_distribution.csv", index=False)
    log.info("Saved mwi_distribution.csv")
    print("\n-- MWI Distribution --")
    print(dist.to_string(index=False))

    # 5 — Plots (shared pose order on category axes for cross-figure comparison)
    pose_order = _canonical_pose_order(master)
    log.info("Generating plots...")
    plot_01_success_rate(perf)
    plot_02_mwi_distribution(master, pose_order=pose_order)
    plot_03_arom_prom(master, pose_order=pose_order)
    plot_04_confidence(perf, pose_order=pose_order)
    plot_05_frames_valid(master, pose_order=pose_order)
    plot_06_stick_passthrough(master)
    plot_07_assist_gap(master)

    write_thesis_summary_table(perf, plaus, dist, master)

    log.info("All outputs saved to %s", OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
