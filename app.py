"""
MI4L Movement Analysis — Streamlit UI
======================================
Professional front-end for the MI4L CV pipeline.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project packages are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mi4l.metrics.summary_metrics import POSE_METADATA  # noqa: E402


# ---------------------------------------------------------------------------
# Python interpreter resolution (mirrors scripts/run_all.py)
# ---------------------------------------------------------------------------
@st.cache_resource
def _find_pipeline_python() -> str:
    """Find a Python interpreter that has mediapipe with mp.solutions.pose."""
    import shutil as _shutil

    candidates = [
        # Dedicated conda env – confirmed working
        r"C:\Users\alexn\anaconda3\envs\mi4l\python.exe",
        sys.executable,
    ]
    for name in ("python", "python3"):
        found = _shutil.which(name)
        if found:
            candidates.append(found)

    check = (
        "import pandas, cv2; "
        "import mediapipe as mp; "
        "mp.solutions.pose.PoseLandmark; "
        "print('ok')"
    )
    for candidate in dict.fromkeys(candidates):
        if not Path(candidate).exists():
            continue
        try:
            res = subprocess.run(
                [candidate, "-c", check],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0 and "ok" in res.stdout:
                return candidate
        except Exception:
            continue

    return sys.executable  # fallback

# ---------------------------------------------------------------------------
# Pose registry — extends POSE_METADATA with UI-specific fields
# ---------------------------------------------------------------------------
BILATERAL_POSES = {"prone_trunk_extension", "bilateral_leg_straddle", "shoulder_stick_pass_through"}

POSE_KEYS = list(POSE_METADATA.keys())  # deterministic order

# Pose illustration images (generated, consistent branding)
_ASSETS_DIR = _PROJECT_ROOT / "assets" / "poses"
POSE_IMAGES = {
    "kneeling_knee_flexion":        _ASSETS_DIR / "lunge.png",
    "prone_trunk_extension":        _ASSETS_DIR / "cobra.png",
    "standing_hip_abduction":       _ASSETS_DIR / "hand_to_toe.png",
    "bilateral_leg_straddle":       _ASSETS_DIR / "side_splits.png",
    "unilateral_hip_extension":     _ASSETS_DIR / "front_splits.png",
    "shoulder_flexion":             _ASSETS_DIR / "shoulder_flexion.png",
    "shoulder_stick_pass_through":  _ASSETS_DIR / "shoulder_extension.png",
}

# Official display names from the MI4L poses reference (poses.pdf)
POSE_DISPLAY_NAMES = {
    "kneeling_knee_flexion":        "Lunge Pose",
    "prone_trunk_extension":        "Cobra Pose",
    "standing_hip_abduction":       "Extended Hand to Big Toe Pose",
    "bilateral_leg_straddle":       "Side Splits",
    "unilateral_hip_extension":     "Front Splits",
    "shoulder_flexion":             "Shoulder Flexion",
    "shoulder_stick_pass_through":  "Shoulder Extension",
}

def _display_name(pose_key: str) -> str:
    """Return the official display name for a pose, with fallback to POSE_METADATA."""
    return POSE_DISPLAY_NAMES.get(pose_key, POSE_METADATA.get(pose_key, {}).get("movement_name", pose_key))

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MI4L Movement Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Typography ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Outfit', sans-serif !important; }

/* ── Accent colours ───────────────────────────────────────── */
:root {
    --accent-teal: #2dd4bf;
    --accent-amber: #f59e0b;
    --accent-coral: #fb7185;
    --bg-card: linear-gradient(145deg, #1a1f2e 0%, #0d1117 100%);
    --border-subtle: rgba(45,212,191,0.15);
    --border-hover: rgba(45,212,191,0.5);
    --text-primary: #f0f4f8;
    --text-secondary: #8899a6;
}

/* ── Global background tint ───────────────────────────────── */
.stApp {
    background-color: #030303 !important;
    background-image: 
        radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(244,63,94,0.05) 0%, transparent 60%) !important;
}

/* ── Hero Geometric Animations ────────────────────────────── */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(var(--rot)); }
    50% { transform: translateY(15px) rotate(var(--rot)); }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.elegant-shape {
    position: absolute;
    border-radius: 9999px;
    backdrop-filter: blur(2px);
    border: 2px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(255, 255, 255, 0.1);
    animation: float 12s infinite ease-in-out;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

.elegant-shape::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 9999px;
    background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.2), transparent 70%);
}

.hero-fade-up {
    animation: fadeUp 1s cubic-bezier(0.25, 0.4, 0.25, 1) forwards;
    opacity: 0;
}
.hero-delay-0 { animation-delay: 0.5s; }
.hero-delay-1 { animation-delay: 0.7s; }
.hero-delay-2 { animation-delay: 0.9s; }


/* ── Unified Pose Cards (Container mapping) ─────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative !important;
    background-color: #0d1117 !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 0 !important;
    overflow: hidden !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 4px 24px rgba(45,212,191,0.12), 0 0 0 1px rgba(45,212,191,0.2) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    gap: 0 !important; /* Remove gap between image and text */
}

/* Base image styling inside cards */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] {
    margin-bottom: -10px !important;
}

/* Invisible overlay button to make entire card clickable */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10 !important;
    opacity: 0 !important; /* Fully invisible but clickable */
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button {
    width: 100% !important;
    height: 100% !important;
    cursor: pointer !important;
}

/* Custom internal card text */
.card-info {
    padding: 0.5rem 1rem 1.5rem 1rem;
    text-align: center;
}
.card-info h4 {
    margin: 0 0 0.3rem 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
}
.card-info p {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* ── Primary action button (Run Analysis) ─────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 50%, #0d9488 100%) !important;
    color: #042f2e !important;
    font-weight: 700 !important;
    border: none !important;
    min-height: auto !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(45,212,191,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent-teal) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Section headers ──────────────────────────────────────── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Info / success / error boxes ─────────────────────────── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Progress bars ────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-amber)) !important;
    border-radius: 8px !important;
}

/* ── Expander styling ─────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ── Download buttons ─────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--accent-teal) !important;
    min-height: auto !important;
    font-size: 0.85rem !important;
}
.stDownloadButton > button:hover {
    background: rgba(45,212,191,0.08) !important;
    border-color: var(--border-hover) !important;
}

/* ── Video uploader area ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border-color: var(--border-subtle) !important;
}

/* ── Hide default Streamlit UI chrome ─────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "screen":       "landing",
    "selected_pose": None,
    "selected_side": "right",
    "arom_file":     None,
    "prom_file":     None,
    "results_dir":   None,
    "run_error":     None,
    "run_stdout":    None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _go(screen: str, **extra):
    st.session_state.screen = screen
    for k, v in extra.items():
        st.session_state[k] = v


# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 1 — LANDING                                        ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_landing():
    st.markdown("")
    # Animated geometric hero
    st.markdown(
        """<div style="position: relative; width: 100%; min-height: 48vh; display: flex; align-items: center; justify-content: center; overflow: hidden; margin-top: -2rem; margin-bottom: 2rem;">
<!-- Floating Shapes Background -->
<div style="position: absolute; inset: 0; overflow: hidden; pointer-events: none;">
<div class="elegant-shape" style="--rot: 12deg; width: 600px; height: 140px; background: linear-gradient(to right, rgba(99,102,241,0.15), transparent); left: -5%; top: 20%;"></div>
<div class="elegant-shape" style="--rot: -15deg; width: 500px; height: 120px; background: linear-gradient(to right, rgba(244,63,94,0.15), transparent); right: 0%; top: 75%; animation-delay: 0.5s;"></div>
<div class="elegant-shape" style="--rot: 20deg; width: 200px; height: 60px; background: linear-gradient(to right, rgba(245,158,11,0.15), transparent); right: 20%; top: 15%; animation-delay: 0.6s;"></div>
</div>
<!-- Text Content -->
<div style="position: relative; z-index: 10; text-align: center; max-width: 800px; padding: 0 1rem;">
<div class="hero-fade-up hero-delay-0" style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 9999px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 2rem;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: rgba(244,63,94,0.8); margin-top: auto; margin-bottom: auto;"></span>
<span style="font-size: 0.875rem; color: rgba(255,255,255,0.6); letter-spacing: 0.025em;">MI4L Clinical Tool</span>
</div>
<h1 class="hero-fade-up hero-delay-1" style="font-size: clamp(2.5rem, 5vw, 4.5rem); font-weight: 700; line-height: 1.1; margin-bottom: 1.5rem; letter-spacing: -0.025em; text-align: center;">
<span style="background: linear-gradient(to bottom, #ffffff, rgba(255,255,255,0.8)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Elevate Your</span>
<br/>
<span style="background: linear-gradient(to right, #a5b4fc, rgba(255,255,255,0.9), #fda4af); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Movement Vision</span>
</h1>
<p class="hero-fade-up hero-delay-2" style="font-size: clamp(1rem, 2vw, 1.15rem); color: rgba(255,255,255,0.4); font-weight: 300; letter-spacing: 0.025em; line-height: 1.6; max-width: 500px; margin: 0 auto; text-align: center;">
Biomechanical range-of-motion assessment powered by computer vision. Select a pose to begin your analysis.
</p>
</div>
</div>""", 
        unsafe_allow_html=True
    )

    # ── Pose grid (4 + 3) ───────────────────────────────────
    row1_keys = POSE_KEYS[:4]
    row2_keys = POSE_KEYS[4:]

    def _pose_row(keys):
        cols = st.columns(len(keys), gap="medium")
        for col, key in zip(cols, keys):
            meta = POSE_METADATA[key]
            display_name = _display_name(key)
            bilateral = key in BILATERAL_POSES
            side_label = "Bilateral" if bilateral else "Unilateral"
            with col:
                with st.container(border=True):
                    # Show pose illustration (takes up top half of card)
                    img_path = POSE_IMAGES.get(key)
                    if img_path and img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                        
                    # Custom styled text matching the new design
                    st.markdown(
                        f"""
                        <div class="card-info">
                            <h4>{display_name}</h4>
                            <p>{meta['joint_name'].title()} · {side_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                        
                    # Invisible overlay button to capture clicks for the entire card
                    if st.button("Select", key=f"sel_{key}", use_container_width=True):
                        default_side = "both" if bilateral else "right"
                        _go("upload", selected_pose=key, selected_side=default_side)
                        st.rerun()

    _pose_row(row1_keys)
    st.markdown("")  # gap
    # Centre the second row of 3
    pad_l, centre, pad_r = st.columns([0.5, 3, 0.5])
    with centre:
        _pose_row(row2_keys)


# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 2 — UPLOAD                                         ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_upload():
    pose_key = st.session_state.selected_pose
    meta = POSE_METADATA.get(pose_key, {})
    bilateral = pose_key in BILATERAL_POSES

    # Back button
    if st.button("← Back to poses"):
        _go("landing")
        st.rerun()

    st.markdown(
        f"<h2 style='margin-bottom:0.2rem;'>{_display_name(pose_key)}</h2>",
        unsafe_allow_html=True,
    )
    tag_col1, tag_col2, tag_col3 = st.columns(3)
    tag_col1.markdown(f"**Joint:** {meta.get('joint_name', '—').title()}")
    tag_col2.markdown(f"**Angle type:** {meta.get('angle_type', '—')}")
    tag_col3.markdown(f"**Laterality:** {'Bilateral' if bilateral else 'Unilateral'}")

    st.divider()

    # ── Side selector (unilateral only) ─────────────────────
    if not bilateral:
        side = st.radio(
            "Side",
            ["Left", "Right"],
            index=0 if st.session_state.selected_side == "left" else 1,
            horizontal=True,
        )
        st.session_state.selected_side = side.lower()
    else:
        st.session_state.selected_side = "both"
        st.info("This is a bilateral pose — both sides are analysed automatically.")

    st.markdown("")

    # ── File uploaders ──────────────────────────────────────
    is_stick_pass = pose_key == "shoulder_stick_pass_through"

    if is_stick_pass:
        # Shoulder extension uses a single video (no AROM/PROM split)
        st.markdown("##### Video *")
        arom = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "avi"],
            key="upload_arom",
            label_visibility="collapsed",
        )
        prom = None  # not applicable
        if arom:
            st.video(arom)
    else:
        col_a, col_p = st.columns(2)

        with col_a:
            st.markdown("##### AROM Video *")
            arom = st.file_uploader(
                "Upload AROM video",
                type=["mp4", "mov", "avi"],
                key="upload_arom",
                label_visibility="collapsed",
            )
            if arom:
                st.video(arom)

        with col_p:
            st.markdown("##### PROM Video *")
            prom = st.file_uploader(
                "Upload PROM video",
                type=["mp4", "mov", "avi"],
                key="upload_prom",
                label_visibility="collapsed",
            )
            if prom:
                st.video(prom)

    st.markdown("")

    # ── Run button ──────────────────────────────────────────
    if is_stick_pass:
        run_disabled = arom is None
    else:
        run_disabled = arom is None or prom is None
    if st.button(
        "🚀  Run Analysis",
        use_container_width=True,
        disabled=run_disabled,
        type="primary",
    ):
        st.session_state.arom_file = arom
        st.session_state.prom_file = prom
        _go("processing")
        st.rerun()


# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 3 — PROCESSING  (animated progress bars)           ║
# ╚═════════════════════════════════════════════════════════════╝

# Steps with their relative time weights (how much of the total
# bar animation time each step takes).
_STEPS = [
    ("🔍", "Extracting pose landmarks",                    5),
    ("📐", "Computing joint angles",                       2),
    ("📊", "Detecting movement window & computing metrics", 2),
    ("📈", "Generating plots & snapshots",                  1),
]
_TOTAL_WEIGHT = sum(w for _, _, w in _STEPS)

# Total time for ALL progress bars combined (seconds).
# The real pipeline usually finishes in <30s; bars finish in ~20s,
# and a holding spinner covers any remaining time.
_BAR_TOTAL_S = 20.0

# Tick interval (seconds) — smooth animation.
_TICK_INTERVAL = 0.08


def _render_processing():
    pose_key = st.session_state.selected_pose
    meta = POSE_METADATA.get(pose_key, {})
    side = st.session_state.selected_side
    has_prom = st.session_state.prom_file is not None

    st.markdown(
        f"<h2 style='margin-bottom:0.3rem;'>Analysing: {_display_name(pose_key)}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Side: **{side.title()}**")

    # ── Write uploaded files to temp dir ────────────────────
    tmp_root = Path(tempfile.mkdtemp(prefix="mi4l_ui_"))
    input_dir = tmp_root / "input"
    input_dir.mkdir()
    out_dir = tmp_root / "output"
    out_dir.mkdir()

    arom_path = input_dir / "arom_video.mp4"
    arom_path.write_bytes(st.session_state.arom_file.getvalue())

    prom_path = None
    if has_prom:
        prom_path = input_dir / "prom_video.mp4"
        prom_path.write_bytes(st.session_state.prom_file.getvalue())

    # ── Resolve the correct Python interpreter ──────────────
    pipeline_python = _find_pipeline_python()
    script_path = str(_PROJECT_ROOT / "scripts" / "run_mi4l.py")
    config_path = str(_PROJECT_ROOT / "configs" / "default.yaml")

    cmd = [
        pipeline_python, script_path,
        "--arom", str(arom_path),
        "--out", str(out_dir),
        "--pose", pose_key,
        "--side", side,
        "--config", config_path,
    ]
    if prom_path:
        cmd.extend(["--prom", str(prom_path)])

    # ── Launch pipeline in a background thread ──────────────
    result_holder: dict = {}

    def _run_pipeline():
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(_PROJECT_ROOT),
            )
            result_holder["result"] = res
        except Exception as exc:
            result_holder["error"] = exc

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    # ── Build step UI: label placeholder + progress bar ─────
    step_labels = []
    step_bars = []
    for icon, label, _ in _STEPS:
        lbl = st.empty()   # placeholder for the label text
        lbl.markdown(f"⏳  **{label}**")
        bar = st.progress(0)
        step_labels.append(lbl)
        step_bars.append(bar)

    pipeline_done = False
    t_start = time.time()

    # ── Animate bars (~20s total) ───────────────────────────
    for step_idx, (icon, label, weight) in enumerate(_STEPS):
        bar = step_bars[step_idx]
        lbl = step_labels[step_idx]

        # Mark this step as active with spinner icon
        lbl.markdown(f"⏳  **{label}** …")

        step_fraction = weight / _TOTAL_WEIGHT
        step_budget_s = _BAR_TOTAL_S * step_fraction
        total_ticks = max(int(step_budget_s / _TICK_INTERVAL), 5)

        progress = 0
        for tick in range(total_ticks):
            if not thread.is_alive():
                pipeline_done = True

            if pipeline_done:
                # Fast-forward remaining bars
                remaining = 100 - progress
                chunk = max(remaining // 3, 5)
                progress = min(progress + chunk, 100)
                bar.progress(progress)
                if progress >= 100:
                    break
                time.sleep(0.03)
            else:
                target = int((tick + 1) / total_ticks * 100)
                progress = min(target, 100)
                bar.progress(progress)
                time.sleep(_TICK_INTERVAL)

        bar.progress(100)
        # Mark step as done
        lbl.markdown(f"✅  **{label}**")

    # ── If pipeline is still running, show holding spinner ───
    if thread.is_alive():
        with st.spinner("Finalising analysis…"):
            while thread.is_alive():
                time.sleep(0.2)

    thread.join(timeout=5)

    # ── Handle result ───────────────────────────────────────
    if "error" in result_holder:
        exc = result_holder["error"]
        st.session_state.run_error = str(exc)
        st.session_state.run_stdout = str(exc)
        st.error(f"Pipeline error:\n```\n{exc}\n```")
        if st.button("← Back to Upload"):
            _go("upload")
            st.rerun()
        return

    result = result_holder.get("result")
    if result is None:
        st.error("Pipeline did not produce a result.")
        if st.button("← Back to Upload"):
            _go("upload")
            st.rerun()
        return

    combined_output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        st.session_state.run_error = combined_output
        st.session_state.run_stdout = combined_output
        st.error(f"Pipeline exited with code {result.returncode}")
        with st.expander("Full output", expanded=True):
            st.code(combined_output, language="text")
        if st.button("← Back to Upload"):
            _go("upload")
            st.rerun()
        return

    # Success!
    elapsed = time.time() - t_start
    st.success(f"✅  Analysis complete in {elapsed:.0f}s")
    st.session_state.results_dir = str(out_dir)
    st.session_state.run_error = None
    st.session_state.run_stdout = combined_output

    _go("results")
    st.rerun()


# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 4 — RESULTS                                        ║
# ╚═════════════════════════════════════════════════════════════╝
def _fmt(val, suffix="", decimals=1):
    """Format a metric value for display."""
    if val is None or (isinstance(val, float) and not pd.notna(val)):
        return "—"
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _render_results():
    out_dir = Path(st.session_state.results_dir)
    pose_key = st.session_state.selected_pose
    meta = POSE_METADATA.get(pose_key, {})

    # ── Header ──────────────────────────────────────────────
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown(
            f"<h2>Results — {_display_name(pose_key)}</h2>",
            unsafe_allow_html=True,
        )
    with head_right:
        if st.button("🏠  New Analysis"):
            _go("landing")
            st.rerun()

    # ── Load summary ────────────────────────────────────────
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        st.error("summary.csv not found in output directory.")
        return
    df = pd.read_csv(summary_path)

    if df.empty:
        st.warning("No results were produced. Check QC flags or video quality.")
        return

    row = df.iloc[0]  # single-row summary (per the pipeline design)

    # ── Console output (collapsed) ──────────────────────────
    with st.expander("Pipeline console output", expanded=False):
        st.code(st.session_state.run_stdout or "(no output)", language="text")

    # ── CORE METRICS ────────────────────────────────────────
    st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AROM Peak", _fmt(row.get("arom_deg"), "°"))
    c2.metric("PROM Peak", _fmt(row.get("prom_deg"), "°"))
    c3.metric("Assist Gap", _fmt(row.get("assist_gap"), "°"))
    mi4l_val = row.get("mi4l")
    mi4l_display = _fmt(mi4l_val, decimals=3) if pd.notna(mi4l_val) else "—"
    mi4l_valid = row.get("mi4l_valid", False)
    c4.metric("MI4L Score", mi4l_display, delta="Valid" if mi4l_valid else "Invalid",
              delta_color="normal" if mi4l_valid else "inverse")

    # ── QUALITY METRICS ─────────────────────────────────────
    st.markdown('<div class="section-header">End-Range Quality</div>', unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Peak Hold Time", _fmt(row.get("peak_hold_time_s"), " s", 2))
    q2.metric("Peak Stability", _fmt(row.get("peak_band_std_deg"), "°", 2))
    q3.metric("Time to Peak", _fmt(row.get("time_to_peak_s"), " s", 2))
    q4.metric("Jerk RMS", _fmt(row.get("jerk_rms"), decimals=2))
    q5.metric("Fit R²", _fmt(row.get("fit_r2"), decimals=3))

    # ── COMPENSATION METRICS ────────────────────────────────
    st.markdown('<div class="section-header">Compensation</div>', unsafe_allow_html=True)
    comp1, comp2, comp3 = st.columns(3)
    comp1.metric("Torso Angle Change", _fmt(row.get("torso_angle_change_deg"), "°", 2))
    comp2.metric("Pelvis Drift", _fmt(row.get("pelvis_drift_norm"), decimals=3))
    comp3.metric("Frames Valid", _fmt(row.get("frames_valid_pct"), "%", 1))

    # ── QC FLAGS ────────────────────────────────────────────
    flags = str(row.get("qc_flags", ""))
    if flags and flags != "nan":
        with st.expander("⚠️  QC Flags", expanded=False):
            for f in flags.split(";"):
                if f.strip():
                    st.markdown(f"- `{f.strip()}`")

    # ── VISUALISATIONS ──────────────────────────────────────
    st.markdown('<div class="section-header">Visualisations</div>', unsafe_allow_html=True)

    # Plots
    plot_files = sorted(out_dir.glob("plot_*.png"))
    if plot_files:
        plot_cols = st.columns(min(len(plot_files), 2))
        for i, pf in enumerate(plot_files):
            with plot_cols[i % len(plot_cols)]:
                st.image(str(pf), caption=pf.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("No plot images were generated.")

    # Snapshots
    snap_dir = out_dir / "snapshots"
    if snap_dir.exists():
        snap_files = sorted(snap_dir.glob("*.png"))
        if snap_files:
            st.markdown("**Snapshots**")
            snap_cols = st.columns(min(len(snap_files), 3))
            for i, sf in enumerate(snap_files):
                with snap_cols[i % len(snap_cols)]:
                    st.image(str(sf), caption=sf.stem.replace("_", " ").title(), use_container_width=True)

    # ── EXPORT / DOWNLOADS ──────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)

    # Summary CSV
    with d1:
        if summary_path.exists():
            st.download_button(
                "📋 Summary CSV",
                data=summary_path.read_bytes(),
                file_name="summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Full angle CSVs
    angle_csvs = sorted(out_dir.glob("angles_*.csv"))
    with d2:
        if angle_csvs:
            for ac in angle_csvs:
                st.download_button(
                    f"📄 {ac.stem.replace('_', ' ').title()}",
                    data=ac.read_bytes(),
                    file_name=ac.name,
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{ac.name}",
                )

    # Plot downloads
    with d3:
        if plot_files:
            for pf in plot_files:
                st.download_button(
                    f"📈 {pf.stem.replace('_', ' ').title()}",
                    data=pf.read_bytes(),
                    file_name=pf.name,
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{pf.name}",
                )

    # Snapshots download
    with d4:
        if snap_dir.exists():
            snap_files_all = sorted(snap_dir.glob("*.png"))
            for sf in snap_files_all:
                st.download_button(
                    f"🖼️ {sf.stem.replace('_', ' ').title()}",
                    data=sf.read_bytes(),
                    file_name=sf.name,
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_snap_{sf.name}",
                )


# ╔═════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                    ║
# ╚═════════════════════════════════════════════════════════════╝
_SCREENS = {
    "landing":    _render_landing,
    "upload":     _render_upload,
    "processing": _render_processing,
    "results":    _render_results,
}

_SCREENS.get(st.session_state.screen, _render_landing)()
