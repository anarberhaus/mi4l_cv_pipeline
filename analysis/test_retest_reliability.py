"""
Test-Retest Reliability Analysis
MWI Pipeline — Alex Narberhaus Piera — Capstone Thesis 2026

Run from repo root:
    python analysis/test_retest_reliability.py \
        --session1 results_session1/ \
        --session2 results_session2/
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAIRS = 5

sns.set_theme(style="whitegrid", font_scale=1.15, rc={
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})
PALETTE = sns.color_palette("Set2")


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _infer_participant_id(csv_path: Path, session_root: Path) -> str | None:
    rel = csv_path.relative_to(session_root)
    parts = rel.parts
    if parts[0].startswith("batch_run"):
        return parts[1] if len(parts) >= 4 else None
    if parts[0] == "assessments":
        return parts[1] if len(parts) >= 4 else None
    return parts[0] if len(parts) >= 3 else None


def load_session(session_dir: Path) -> pd.DataFrame:
    """Load all summary.csv files from a session directory."""
    frames = []
    for csv_path in sorted(session_dir.rglob("summary.csv")):
        pid = _infer_participant_id(csv_path, session_dir)
        if pid is None:
            log.info("  Skipping standalone: %s", csv_path.relative_to(session_dir))
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            log.warning("  Could not read %s: %s", csv_path, exc)
            continue

        is_stick = "grip_ratio_normalized" in df.columns
        if is_stick:
            continue

        is_stick_movement = df["movement_name"].str.contains(
            "Stick Pass-Through", case=False, na=False
        )
        df = df[~is_stick_movement]
        if df.empty:
            continue

        df["participant_id"] = pid
        df["_mtime"] = csv_path.parent.stat().st_mtime
        frames.append(df)
        log.info("  + %s  [%s]", pid, df["movement_name"].iloc[0])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined
        .sort_values("_mtime", ascending=False)
        .drop_duplicates(subset=["participant_id", "movement_name", "side"], keep="first")
        .drop(columns=["_mtime"])
        .reset_index(drop=True)
    )
    return combined


def icc_interpretation(icc_val: float) -> str:
    if pd.isna(icc_val):
        return "N/A"
    if icc_val < 0.50:
        return "poor"
    if icc_val < 0.75:
        return "moderate"
    if icc_val < 0.90:
        return "good"
    return "excellent"


# ═════════════════════════════════════════════════════════════════════════
# 1. LOAD & MATCH TRIALS
# ═════════════════════════════════════════════════════════════════════════

def match_trials(s1: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    """Match valid trials across sessions by participant + movement."""
    s1_valid = s1[s1["mi4l_valid"] == True].copy()
    s2_valid = s2[s2["mi4l_valid"] == True].copy()

    s1_valid["session"] = 1
    s2_valid["session"] = 2

    merge_keys = ["participant_id", "movement_name", "side"]
    matched = s1_valid.merge(s2_valid, on=merge_keys, suffixes=("_s1", "_s2"))

    log.info("Matched pairs per pose:")
    for move, grp in matched.groupby("movement_name"):
        log.info("  %s: %d pairs", move, len(grp))

    return matched


# ═════════════════════════════════════════════════════════════════════════
# 2. RELIABILITY METRICS
# ═════════════════════════════════════════════════════════════════════════

def compute_reliability(matched: pd.DataFrame) -> pd.DataFrame:
    metrics_all = []

    for move, grp in matched.groupby("movement_name"):
        if len(grp) < MIN_PAIRS:
            log.warning("  %s: only %d pairs (< %d), skipping", move, len(grp), MIN_PAIRS)
            continue

        for metric_col, label in [("arom_deg", "arom_deg"),
                                   ("prom_deg", "prom_deg"),
                                   ("mi4l", "mwi")]:
            col_s1 = f"{metric_col}_s1"
            col_s2 = f"{metric_col}_s2"

            if col_s1 not in grp.columns or col_s2 not in grp.columns:
                continue

            vals_s1 = grp[col_s1].dropna()
            vals_s2 = grp[col_s2].dropna()
            common_idx = vals_s1.index.intersection(vals_s2.index)
            if len(common_idx) < MIN_PAIRS:
                continue

            long_df = pd.DataFrame({
                "targets": list(grp.loc[common_idx, "participant_id"]) * 2,
                "raters": [1] * len(common_idx) + [2] * len(common_idx),
                "ratings": list(grp.loc[common_idx, col_s1]) + list(grp.loc[common_idx, col_s2]),
            })

            try:
                icc_table = pg.intraclass_corr(
                    data=long_df, targets="targets", raters="raters", ratings="ratings"
                )
                icc_row = icc_table[icc_table["Type"] == "ICC2"]
                icc_val = icc_row["ICC"].values[0]
                ci_low = icc_row["CI95%"].values[0][0]
                ci_high = icc_row["CI95%"].values[0][1]
            except Exception as exc:
                log.warning("  ICC failed for %s / %s: %s", move, label, exc)
                icc_val = ci_low = ci_high = np.nan

            pooled_std = np.sqrt(
                (grp.loc[common_idx, col_s1].var() + grp.loc[common_idx, col_s2].var()) / 2
            )
            sem = pooled_std * np.sqrt(1 - icc_val) if not pd.isna(icc_val) else np.nan
            mdc95 = sem * 1.96 * np.sqrt(2) if not pd.isna(sem) else np.nan

            metrics_all.append({
                "movement_name": move,
                "metric": label,
                "n_pairs": len(common_idx),
                "ICC2": round(icc_val, 4) if not pd.isna(icc_val) else np.nan,
                "ICC2_CI95_low": round(ci_low, 4) if not pd.isna(ci_low) else np.nan,
                "ICC2_CI95_high": round(ci_high, 4) if not pd.isna(ci_high) else np.nan,
                "SEM": round(sem, 4) if not pd.isna(sem) else np.nan,
                "MDC95": round(mdc95, 4) if not pd.isna(mdc95) else np.nan,
                "interpretation": icc_interpretation(icc_val),
            })

    return pd.DataFrame(metrics_all)


# ═════════════════════════════════════════════════════════════════════════
# 3. VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════

def _short_name(name: str) -> str:
    return (name
            .replace("Kneeling ", "")
            .replace("Standing ", "")
            .replace("Prone ", "")
            .replace("Bilateral ", "")
            .replace("Unilateral ", "")
            .replace("Shoulder ", "S. "))


def plot_bland_altman(matched: pd.DataFrame, reliability: pd.DataFrame):
    poses = matched["movement_name"].unique()
    n = len(poses)
    if n == 0:
        log.warning("No poses for Bland-Altman plot")
        return

    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)

    for idx, move in enumerate(sorted(poses)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        grp = matched[matched["movement_name"] == move]

        s1 = grp["mi4l_s1"]
        s2 = grp["mi4l_s2"]
        mean_mwi = (s1 + s2) / 2
        diff = s1 - s2
        mean_diff = diff.mean()
        sd_diff = diff.std()
        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff

        ax.scatter(mean_mwi, diff, s=40, alpha=0.7, color=PALETTE[0], edgecolors="k", lw=0.5)
        ax.axhline(mean_diff, color="k", lw=1.2)
        ax.axhline(loa_upper, color="grey", ls="--", lw=1)
        ax.axhline(loa_lower, color="grey", ls="--", lw=1)
        ax.set_xlabel("Mean MWI")
        ax.set_ylabel("$\\Delta$ MWI (S1 $-$ S2)")
        ax.set_title(_short_name(move), fontsize=11)

        icc_row = reliability[
            (reliability["movement_name"] == move) & (reliability["metric"] == "mwi")
        ]
        if not icc_row.empty:
            icc_val = icc_row["ICC2"].values[0]
            ax.annotate(f"ICC = {icc_val:.2f}", xy=(0.02, 0.95),
                        xycoords="axes fraction", fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Bland-Altman: MWI Test-Retest", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_06_bland_altman_mwi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_06_bland_altman_mwi.png")


def plot_session_scatter(matched: pd.DataFrame, reliability: pd.DataFrame):
    poses = matched["movement_name"].unique()
    n = len(poses)
    if n == 0:
        log.warning("No poses for session scatter plot")
        return

    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)

    for idx, move in enumerate(sorted(poses)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        grp = matched[matched["movement_name"] == move]

        s1 = grp["mi4l_s1"]
        s2 = grp["mi4l_s2"]
        ax.scatter(s1, s2, s=50, alpha=0.7, color=PALETTE[1], edgecolors="k", lw=0.5)

        lo = min(s1.min(), s2.min()) - 0.05
        hi = max(s1.max(), s2.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Identity")
        ax.set_xlabel("Session 1 MWI")
        ax.set_ylabel("Session 2 MWI")
        ax.set_title(_short_name(move), fontsize=11)

        if len(s1) >= 3:
            r_val, _ = pearsonr(s1, s2)
            icc_row = reliability[
                (reliability["movement_name"] == move) & (reliability["metric"] == "mwi")
            ]
            icc_val = icc_row["ICC2"].values[0] if not icc_row.empty else np.nan
            label = f"r = {r_val:.2f}"
            if not pd.isna(icc_val):
                label += f"\nICC = {icc_val:.2f}"
            ax.annotate(label, xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.legend(loc="lower right", fontsize=8)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Session 1 vs Session 2 MWI", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_07_session_scatter_mwi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_07_session_scatter_mwi.png")


# ═════════════════════════════════════════════════════════════════════════
# 4. CONSOLE SUMMARY
# ═════════════════════════════════════════════════════════════════════════

def print_summary(reliability: pd.DataFrame):
    if reliability.empty:
        print("\nNo reliability metrics to display.")
        return

    print("\n" + "=" * 80)
    print("TEST-RETEST RELIABILITY SUMMARY")
    print("=" * 80)

    display_cols = ["movement_name", "metric", "n_pairs", "ICC2",
                    "ICC2_CI95_low", "ICC2_CI95_high", "SEM", "MDC95", "interpretation"]
    available = [c for c in display_cols if c in reliability.columns]
    print(reliability[available].to_string(index=False))
    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test-Retest Reliability Analysis")
    parser.add_argument("--session1", required=True, type=Path,
                        help="Path to first session results directory")
    parser.add_argument("--session2", required=True, type=Path,
                        help="Path to second session results directory")
    args = parser.parse_args()

    session1_dir = Path(args.session1).resolve()
    session2_dir = Path(args.session2).resolve()

    if not session1_dir.exists():
        log.error("Session 1 directory not found: %s", session1_dir)
        raise SystemExit(1)
    if not session2_dir.exists():
        log.error("Session 2 directory not found: %s", session2_dir)
        raise SystemExit(1)

    log.info("=" * 60)
    log.info("Test-Retest Reliability Analysis")
    log.info("=" * 60)

    # 1 — Load sessions
    log.info("Loading session 1: %s", session1_dir)
    s1 = load_session(session1_dir)
    log.info("  -> %d valid trials from %d participants",
             len(s1), s1["participant_id"].nunique() if len(s1) else 0)

    log.info("Loading session 2: %s", session2_dir)
    s2 = load_session(session2_dir)
    log.info("  -> %d valid trials from %d participants",
             len(s2), s2["participant_id"].nunique() if len(s2) else 0)

    if s1.empty or s2.empty:
        log.error("One or both sessions contain no valid data.")
        raise SystemExit(1)

    # 2 — Match trials
    matched = match_trials(s1, s2)
    if matched.empty:
        log.error("No matched trial pairs found across sessions.")
        raise SystemExit(1)

    matched.to_csv(OUTPUT_DIR / "matched_pairs.csv", index=False)
    log.info("Saved matched_pairs.csv (%d pairs)", len(matched))

    # 3 — Reliability metrics
    reliability = compute_reliability(matched)
    if not reliability.empty:
        reliability.to_csv(OUTPUT_DIR / "reliability_metrics.csv", index=False)
        log.info("Saved reliability_metrics.csv")
    else:
        log.warning("No reliability metrics could be computed (need >= %d pairs)", MIN_PAIRS)

    # 4 — Plots
    plot_bland_altman(matched, reliability)
    plot_session_scatter(matched, reliability)

    # 5 — Console summary
    print_summary(reliability)

    log.info("All outputs saved to %s", OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
