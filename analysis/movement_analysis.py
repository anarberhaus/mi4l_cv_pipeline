"""
Movement data analysis (descriptive).

Figures plot_08–plot_11 are written to analysis/outputs/plots/ (with plot_01–07).
CSVs and key_observations.txt go to analysis/outputs/movement_analysis/.

Run from repo root (after pipeline_performance.py):
    python analysis/movement_analysis.py
"""

from __future__ import annotations

import logging
import warnings
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
MASTER_PATH = REPO_ROOT / "analysis" / "outputs" / "master_summary.csv"
# Tables / text: movement_analysis subfolder; figures: same plots/ as pipeline (one place for all PNGs)
DATA_DIR = REPO_ROOT / "analysis" / "outputs" / "movement_analysis"
PLOT_DIR = REPO_ROOT / "analysis" / "outputs" / "plots"

# Match pipeline_performance: AROM green, PROM orange (comparable across thesis figures)
PALETTE_AROM = "#4caf50"
PALETTE_PROM = "#ff9800"

sns.set_theme(style="whitegrid", font_scale=1.15, rc={
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def _apply_spec_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.4)


def _short_name(name: str) -> str:
    return (name
            .replace("Kneeling ", "")
            .replace("Standing ", "")
            .replace("Prone ", "")
            .replace("Bilateral ", "")
            .replace("Unilateral ", "")
            .replace("Shoulder ", "S. "))


def canonical_pose_order_short(master: pd.DataFrame) -> list[str]:
    """Alphabetical full movement name → short labels (aligns with pipeline plots)."""
    names = sorted(master["movement_name"].unique())
    return [_short_name(m) for m in names]


def trial_angles_path(batch_id: str, participant_id: str, movement_name: str, side: str) -> Path:
    base = movement_name.lower().replace(" ", "_").replace("-", "_")
    return RESULTS_DIR / batch_id / participant_id / f"{base}_{side}"


def pick_angle_column(df: pd.DataFrame, side: str) -> str | None:
    deg_cols = [c for c in df.columns if c.endswith("_deg")]
    if not deg_cols:
        return None
    if len(deg_cols) == 1:
        return deg_cols[0]
    if side == "left":
        for c in deg_cols:
            if c.startswith("left_"):
                return c
    elif side == "right":
        for c in deg_cols:
            if c.startswith("right_"):
                return c
    neutral = [c for c in deg_cols if not (c.startswith("left_") or c.startswith("right_"))]
    if neutral:
        return neutral[0]
    return None


def load_angle_series(path: Path, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    if "time_sec" not in df.columns:
        return None
    t = df["time_sec"].values.astype(float)
    col = pick_angle_column(df, side)
    if col is None:
        deg_cols = [c for c in df.columns if c.endswith("_deg")]
        left_c = next((c for c in deg_cols if c.startswith("left_")), None)
        right_c = next((c for c in deg_cols if c.startswith("right_")), None)
        if left_c and right_c:
            v = (df[left_c].values.astype(float) + df[right_c].values.astype(float)) / 2.0
        else:
            return None
    else:
        v = df[col].values.astype(float)
    return t, v


def load_stick_series(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log.warning("Could not read stick %s: %s", path, exc)
        return None
    if "time_sec" not in df.columns or "stick_pass_through_dist_norm" not in df.columns:
        return None
    return df["time_sec"].values.astype(float), df["stick_pass_through_dist_norm"].values.astype(float)


def compute_hold_metrics(time_sec: np.ndarray, values: np.ndarray) -> dict | None:
    ok = np.isfinite(time_sec) & np.isfinite(values)
    if ok.sum() < 2:
        return None
    t = time_sec[ok]
    v = values[ok]
    peak = float(np.nanmax(v))
    if not np.isfinite(peak):
        return None
    if peak >= 0:
        in_band = (v >= 0.95 * peak) & (v <= peak * 1.001)
    else:
        in_band = (v <= 0.95 * peak) & (v >= peak * 1.001)

    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, flag in enumerate(in_band):
        if flag:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len == 0:
        return None

    t_start_hold = float(t[best_start])
    t_end_hold = float(t[best_start + best_len - 1])
    peak_hold_duration_sec = max(0.0, t_end_hold - t_start_hold)

    first_in = int(np.argmax(in_band)) if in_band.any() else 0
    time_to_peak_sec = float(t[first_in] - t[0]) if in_band.any() else np.nan

    return {
        "peak_hold_duration_sec": peak_hold_duration_sec,
        "time_to_peak_sec": time_to_peak_sec,
    }


def collect_hold_trials(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    angle_df = angle_df[angle_df["mi4l_valid"] == True]

    for _, row in angle_df.iterrows():
        batch_id = row["batch_id"]
        pid = row["participant_id"]
        move = row["movement_name"]
        side = row["side"]
        base = trial_angles_path(batch_id, pid, move, side)
        for kind in ("arom", "prom"):
            path = base / f"angles_{kind}.csv"
            loaded = load_angle_series(path, side)
            if loaded is None:
                log.info("Skip missing angles: %s", path)
                continue
            t, v = loaded
            m = compute_hold_metrics(t, v)
            if m is None:
                continue
            rows.append({
                "movement_name": move,
                "participant_id": pid,
                "side": side,
                "kind": kind.upper(),
                **m,
            })
    return pd.DataFrame(rows)


def aggregate_hold_per_pose(trial_df: pd.DataFrame) -> pd.DataFrame:
    """One row per pose: pool AROM + PROM trials (spec: aggregate per pose)."""
    if trial_df.empty:
        return pd.DataFrame()
    agg_rows = []
    for move, grp in trial_df.groupby("movement_name"):
        ph = grp["peak_hold_duration_sec"]
        tt = grp["time_to_peak_sec"]
        agg_rows.append({
            "movement_name": move,
            "n_trials": len(grp),
            "mean_peak_hold_duration_sec": round(ph.mean(), 4),
            "std_peak_hold_duration_sec": round(ph.std(), 4) if len(grp) > 1 else 0.0,
            "mean_time_to_peak_sec": round(tt.mean(), 4),
            "std_time_to_peak_sec": round(tt.std(), 4) if len(grp) > 1 else 0.0,
            "shortest_hold_sec": round(ph.min(), 4),
            "longest_hold_sec": round(ph.max(), 4),
        })
    return pd.DataFrame(agg_rows)


def plot_08_hold_duration(trial_df: pd.DataFrame):
    if trial_df.empty:
        log.warning("No data for plot_08")
        return
    df = trial_df.copy()
    df["pose_short"] = df["movement_name"].apply(_short_name)
    order_moves = df.groupby("movement_name")["peak_hold_duration_sec"].mean().sort_values(ascending=False).index
    order_short = [_short_name(m) for m in order_moves]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="pose_short", y="peak_hold_duration_sec", order=order_short,
        color=sns.color_palette("Set2")[1], width=0.55, ax=ax,
    )
    sns.stripplot(
        data=df, x="pose_short", y="peak_hold_duration_sec", order=order_short,
        color="0.25", size=4, alpha=0.55, jitter=0.2, ax=ax,
    )
    for i, pose in enumerate(order_short):
        sub = df[df["pose_short"] == pose]
        mean_val = sub["peak_hold_duration_sec"].mean()
        n = len(sub)
        ax.text(i, ax.get_ylim()[1] * 0.97, f"mean={mean_val:.2f}\nn={n}", ha="center", fontsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("Hold Duration (seconds)")
    ax.set_title("Peak Hold Duration per Pose", pad=18)
    ax.tick_params(axis="x", rotation=28)
    plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")
    _apply_spec_style(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_DIR / "plot_08_hold_duration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_08_hold_duration.png")


def section2_variability(master: pd.DataFrame) -> pd.DataFrame:
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    angle_df = angle_df[angle_df["mi4l_valid"] == True]
    rows = []
    for move, grp in angle_df.groupby("movement_name"):
        arom = grp["arom_deg"].dropna()
        prom = grp["prom_deg"].dropna()
        cv_arom = (arom.std() / arom.mean() * 100) if len(arom) > 1 and arom.mean() != 0 else np.nan
        cv_prom = (prom.std() / prom.mean() * 100) if len(prom) > 1 and prom.mean() != 0 else np.nan
        rows.append({
            "movement_name": move,
            "cv_arom_pct": round(cv_arom, 2) if pd.notna(cv_arom) else np.nan,
            "cv_prom_pct": round(cv_prom, 2) if pd.notna(cv_prom) else np.nan,
            "range_arom_deg": round(arom.max() - arom.min(), 2) if len(arom) else np.nan,
            "range_prom_deg": round(prom.max() - prom.min(), 2) if len(prom) else np.nan,
            "n": len(grp),
        })
    return pd.DataFrame(rows)


def plot_09_variability(var_df: pd.DataFrame, pose_order_short: list[str]):
    if var_df.empty:
        return
    df = var_df.copy()
    df["pose_short"] = df["movement_name"].apply(_short_name)
    order = [p for p in pose_order_short if p in set(df["pose_short"])]
    melted = df.melt(
        id_vars=["pose_short"],
        value_vars=["cv_arom_pct", "cv_prom_pct"],
        var_name="measure",
        value_name="cv_pct",
    )
    melted["measure"] = melted["measure"].map({"cv_arom_pct": "AROM", "cv_prom_pct": "PROM"})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted, x="pose_short", y="cv_pct", hue="measure",
        order=order, hue_order=["AROM", "PROM"],
        palette={"AROM": PALETTE_AROM, "PROM": PALETTE_PROM}, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("CV (%)")
    ax.set_title(
        "Participant Variability per Pose (Coefficient of Variation)\n"
        "CV = (std / mean) × 100 across participants per pose",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=28)
    plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation_mode="anchor")
    ax.legend(title="", loc="upper right")
    _apply_spec_style(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_09_variability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_09_variability.png")


def section3_active_control(master: pd.DataFrame) -> pd.DataFrame:
    angle_df = master[master["movement_name"] != "Shoulder Stick Pass-Through"]
    angle_df = angle_df[angle_df["mi4l_valid"] == True]
    rows = []
    for move, grp in angle_df.groupby("movement_name"):
        ag = grp["assist_gap"]
        prom = grp["prom_deg"]
        rel = (ag / prom.replace(0, np.nan) * 100.0)
        rows.append({
            "movement_name": move,
            "mean_assist_gap_deg": round(ag.mean(), 2),
            "std_assist_gap_deg": round(ag.std(), 2),
            "mean_relative_gap_pct": round(rel.mean(), 2),
            "std_relative_gap_pct": round(rel.std(), 2),
            "n": len(grp),
        })
    return pd.DataFrame(rows)


def plot_10_active_control(diff_df: pd.DataFrame, pose_order_short: list[str]):
    if diff_df.empty:
        return
    df = diff_df.copy()
    df["pose_short"] = df["movement_name"].apply(_short_name)
    # Same categorical order as other “per pose” bar-style comparisons (readability)
    order = [p for p in pose_order_short if p in set(df["pose_short"])]
    df = df.set_index("pose_short").loc[order].reset_index()

    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))
    xvals = df["mean_relative_gap_pct"].values
    xerr = df["std_relative_gap_pct"].values
    ax.barh(y_pos, xvals, xerr=xerr, color=sns.color_palette("Set2")[2], capsize=3, alpha=0.9)
    xmax = max((xvals + np.nan_to_num(xerr, nan=0)).max(), 1) * 1.12
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            min(row["mean_relative_gap_pct"] + 0.6, xmax * 0.98),
            i,
            f"{row['mean_relative_gap_pct']:.1f}%",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["pose_short"])
    ax.set_xlabel("Assist Gap as % of PROM (higher = harder to actively control)")
    ax.set_title("Active Control Difficulty per Pose", pad=16)
    ax.set_xlim(0, xmax)
    ax.axvline(0, color="grey", ls="--", lw=1)
    _apply_spec_style(ax)
    fig.text(
        0.5, -0.02,
        "n = valid trials per pose (after QC). n differs across poses if videos are missing, "
        "trials fail QC, or a pose was not recorded for all participants.",
        ha="center", fontsize=9, wrap=True, transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(PLOT_DIR / "plot_10_active_control_difficulty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_10_active_control_difficulty.png")


def resample_to_grid(t_norm: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    if len(t_norm) < 2:
        return np.full(n, np.nan)
    tq = np.linspace(0, 1, n)
    return np.interp(tq, t_norm, y, left=np.nan, right=np.nan)


def _trial_duration_seconds(t: np.ndarray) -> float | None:
    ok = np.isfinite(t)
    if ok.sum() < 2:
        return None
    tt = t[ok]
    return float(tt[-1] - tt[0])


def _max_duration_pose(sub: pd.DataFrame, move: str, is_stick: bool) -> float:
    """Longest trial duration (s) among all AROM/PROM (or stick AROM) series for this pose."""
    T_max = 0.0
    for _, row in sub.iterrows():
        batch_id = row["batch_id"]
        pid = row["participant_id"]
        side = row["side"]
        base = trial_angles_path(batch_id, pid, move, side)
        if is_stick:
            p = base / "angles_arom.csv"
            loaded = load_stick_series(p) if p.exists() else None
            if loaded is None:
                continue
            t, _ = loaded
            d = _trial_duration_seconds(t)
            if d is not None:
                T_max = max(T_max, d)
        else:
            for kind in ("arom", "prom"):
                p = base / f"angles_{kind}.csv"
                loaded = load_angle_series(p, side)
                if loaded is None:
                    continue
                t, _ = loaded
                d = _trial_duration_seconds(t)
                if d is not None:
                    T_max = max(T_max, d)
    return T_max


def _resample_absolute_time(t: np.ndarray, y: np.ndarray, T_max: float, grid: np.ndarray) -> np.ndarray:
    """Map time to (t - t0) / T_max so shorter trials end before x=1; interpolate y onto grid."""
    if T_max <= 1e-12:
        return np.full(len(grid), np.nan)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    t0, y0 = t[ok], y[ok]
    t_scaled = (t0 - t0[0]) / T_max
    t_end = t_scaled[-1]
    out = np.full(len(grid), np.nan)
    mask = (grid >= t_scaled[0] - 1e-12) & (grid <= t_end + 1e-12)
    out[mask] = np.interp(grid[mask], t_scaled, y0)
    return out


def _curves_with_finite_values(curves: list[np.ndarray], min_finite: int = 8) -> list[np.ndarray]:
    """Drop trajectories that are mostly NaN (avoids empty-slice warnings in mean/std)."""
    out = []
    for c in curves:
        if np.isfinite(c).sum() >= min_finite:
            out.append(c)
    return out


def section4_profiles(master: pd.DataFrame) -> None:
    poses = sorted(master["movement_name"].unique())
    n_poses = len(poses)
    ncols = 2
    nrows = int(np.ceil(n_poses / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    n_grid = 100
    grid = np.linspace(0, 1, n_grid)

    for idx, move in enumerate(poses):
        ax = axes_flat[idx]
        is_stick = move == "Shoulder Stick Pass-Through"
        if is_stick:
            sub = master[master["movement_name"] == move]
        else:
            sub = master[(master["movement_name"] == move) & (master["mi4l_valid"] == True)]

        T_max = _max_duration_pose(sub, move, is_stick)
        if T_max <= 0:
            log.warning("No duration data for time-series profile: %s", move)
            T_max = 1.0

        arom_curves: list[np.ndarray] = []
        prom_curves: list[np.ndarray] = []

        for _, row in sub.iterrows():
            batch_id = row["batch_id"]
            pid = row["participant_id"]
            side = row["side"]
            base = trial_angles_path(batch_id, pid, move, side)

            if is_stick:
                loaded = load_stick_series(base / "angles_arom.csv")
                if loaded is None:
                    log.info("Skip stick profile: %s", base / "angles_arom.csv")
                    continue
                t, v = loaded
                arom_curves.append(_resample_absolute_time(t, v, T_max, grid))
            else:
                for kind, bucket in (("arom", arom_curves), ("prom", prom_curves)):
                    path = base / f"angles_{kind}.csv"
                    loaded = load_angle_series(path, side)
                    if loaded is None:
                        continue
                    t, y = loaded
                    bucket.append(_resample_absolute_time(t, y, T_max, grid))

        arom_use = _curves_with_finite_values(arom_curves)
        prom_use = _curves_with_finite_values(prom_curves) if not is_stick else []

        if arom_use:
            mat = np.vstack(arom_use)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_a = np.nanmean(mat, axis=0)
                ddof = 0 if mat.shape[0] < 2 else 1
                std_a = np.nanstd(mat, axis=0, ddof=ddof)
            mean_a = np.nan_to_num(mean_a, nan=np.nan)
            std_a = np.nan_to_num(std_a, nan=0.0)
            lbl = "Grip (norm.)" if is_stick else "AROM"
            ax.plot(grid, mean_a, color=PALETTE_AROM, lw=2, label=lbl)
            ax.fill_between(grid, mean_a - std_a, mean_a + std_a, color=PALETTE_AROM, alpha=0.22)
        if prom_use and not is_stick:
            matp = np.vstack(prom_use)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_p = np.nanmean(matp, axis=0)
                ddof = 0 if matp.shape[0] < 2 else 1
                std_p = np.nanstd(matp, axis=0, ddof=ddof)
            mean_p = np.nan_to_num(mean_p, nan=np.nan)
            std_p = np.nan_to_num(std_p, nan=0.0)
            ax.plot(grid, mean_p, color=PALETTE_PROM, lw=2, label="PROM")
            ax.fill_between(grid, mean_p - std_p, mean_p + std_p, color=PALETTE_PROM, alpha=0.22)

        y_label = "Grip ratio (norm.)" if is_stick else "Angle (degrees)"
        ax.set_ylabel(y_label)
        ax.set_title(_short_name(move))
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "1"])
        ax.set_xlabel("Normalised time")
        _apply_spec_style(ax)

    for j in range(len(poses), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)

    fig.suptitle("Average Movement Profiles per Pose (AROM vs PROM)", fontsize=14, y=1.01)
    fig.supxlabel(
        "Normalised time: 0 = movement start; 1 = duration of the longest trial in this pose "
        "(shorter trials end before 1, so AROM vs PROM timing differences are visible).",
        fontsize=10,
        y=0.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(PLOT_DIR / "plot_11_time_series_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot_11_time_series_profiles.png")


def write_key_observations(
    hold_agg: pd.DataFrame,
    var_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    master: pd.DataFrame,
) -> None:
    lines: list[str] = []
    angle_valid = master[
        (master["movement_name"] != "Shoulder Stick Pass-Through")
        & (master["mi4l_valid"] == True)
    ]

    if not hold_agg.empty:
        ph = hold_agg.set_index("movement_name")["mean_peak_hold_duration_sec"]
        lines.append(
            f"The pose with the longest mean peak hold duration is {ph.idxmax()} "
            f"({ph.max():.2f} s on average across AROM/PROM trials)."
        )
        lines.append(
            f"The pose with the shortest mean peak hold duration is {ph.idxmin()} "
            f"({ph.min():.2f} s on average)."
        )

    if not var_df.empty and var_df["cv_arom_pct"].notna().any():
        v = var_df.set_index("movement_name")["cv_arom_pct"].dropna()
        lines.append(f"The pose with the highest participant CV for AROM is {v.idxmax()} ({v.max():.1f}%).")
        lines.append(f"The pose with the lowest participant CV for AROM is {v.idxmin()} ({v.min():.1f}%).")

    if not diff_df.empty:
        d = diff_df.set_index("movement_name")
        lines.append(
            f"The pose with the largest mean assist gap (degrees) is {d['mean_assist_gap_deg'].idxmax()} "
            f"({d['mean_assist_gap_deg'].max():.1f}°)."
        )
        lines.append(
            f"The pose with the smallest mean assist gap (degrees) is {d['mean_assist_gap_deg'].idxmin()} "
            f"({d['mean_assist_gap_deg'].min():.1f}°)."
        )
        lines.append(
            f"The pose with the largest mean relative assist gap (% of PROM) is {d['mean_relative_gap_pct'].idxmax()} "
            f"({d['mean_relative_gap_pct'].max():.1f}%)."
        )
        lines.append(
            f"The pose with the smallest mean relative assist gap (% of PROM) is {d['mean_relative_gap_pct'].idxmin()} "
            f"({d['mean_relative_gap_pct'].min():.1f}%)."
        )

    n_part = master["participant_id"].nunique()
    lines.append(f"The dataset contains {n_part} distinct participants.")
    lines.append(f"There are {len(angle_valid)} valid angle trials across all poses (excluding stick pass-through).")

    out = DATA_DIR / "key_observations.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved key_observations.txt")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_PATH.exists():
        log.error("Missing %s — run analysis/pipeline_performance.py first.", MASTER_PATH)
        return 1

    master = pd.read_csv(MASTER_PATH)
    pose_order_short = canonical_pose_order_short(master)

    log.info("Section 1: hold duration…")
    trial_holds = collect_hold_trials(master)
    hold_agg = aggregate_hold_per_pose(trial_holds)
    hold_agg.to_csv(DATA_DIR / "hold_duration_per_pose.csv", index=False)
    plot_08_hold_duration(trial_holds)

    log.info("Section 2: variability…")
    var_df = section2_variability(master)
    var_df.to_csv(DATA_DIR / "variability_per_pose.csv", index=False)
    plot_09_variability(var_df, pose_order_short)

    log.info("Section 3: active control difficulty…")
    diff_df = section3_active_control(master)
    diff_df.to_csv(DATA_DIR / "active_control_difficulty.csv", index=False)
    plot_10_active_control(diff_df, pose_order_short)

    log.info("Section 4: time series profiles…")
    section4_profiles(master)

    write_key_observations(hold_agg, var_df, diff_df, master)

    log.info("Done. Plots → %s | tables → %s", PLOT_DIR, DATA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
