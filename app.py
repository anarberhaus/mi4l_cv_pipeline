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
    """Find a Python interpreter that has the required dependencies."""
    import shutil as _shutil

    # On Streamlit Cloud and most local environments, the current interpreter 
    # (sys.executable) already has all requirements from requirements.txt.
    candidates = [sys.executable]
    
    for name in ("python3", "python"):
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
        try:
            # Skip if candidate is actually a path that doesn't exist
            if "/" in candidate or "\\" in candidate:
                if not Path(candidate).exists():
                    continue
            
            res = subprocess.run(
                [candidate, "-c", check],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0 and "ok" in res.stdout:
                return candidate
        except Exception:
            continue

    return sys.executable  # fallback to current process

# ---------------------------------------------------------------------------
# Pose registry — extends POSE_METADATA with UI-specific fields
# ---------------------------------------------------------------------------
BILATERAL_POSES = {"prone_trunk_extension", "bilateral_leg_straddle", "shoulder_stick_pass_through"}

POSE_KEYS = list(POSE_METADATA.keys())  # deterministic order

# Pose illustration images (generated, consistent branding)
_ASSETS_DIR = _PROJECT_ROOT / "assets" / "poses"
POSE_IMAGES = {
    "kneeling_knee_flexion":        _ASSETS_DIR / "lunge_dimmed.png",
    "prone_trunk_extension":        _ASSETS_DIR / "cobra_dimmed.png",
    "standing_hip_abduction":       _ASSETS_DIR / "hand_to_toe_dimmed.png",
    "bilateral_leg_straddle":       _ASSETS_DIR / "side_splits_dimmed.png",
    "unilateral_hip_extension":     _ASSETS_DIR / "front_splits_dimmed.png",
    "shoulder_flexion":             _ASSETS_DIR / "shoulder_flexion_dimmed.png",
    "shoulder_stick_pass_through":  _ASSETS_DIR / "shoulder_extension_dimmed.png",
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
    page_title="MWI Mobility Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Auth Initialization (Required before any UI rendering)
# ---------------------------------------------------------------------------
from mi4l.supabase_client import get_supabase_client
try:
    supabase = get_supabase_client()
except Exception as e:
    st.error(f"Supabase Configuration Error: {e}")
    st.stop()

import extra_streamlit_components as stx
cookie_manager = stx.CookieManager()

if "auth_initialized" not in st.session_state:
    st.session_state.auth_initialized = False
if "user" not in st.session_state:
    st.session_state.user = None
if "history_analysis_data" not in st.session_state:
    st.session_state.history_analysis_data = {}
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

if not st.session_state.auth_initialized:
    # 1) Wait for CookieManager mount cycle
    cookies = cookie_manager.get_all()
    if len(cookies) == 0 and not st.session_state.get("_cookie_checked", False):
        st.session_state._cookie_checked = True
        st.stop()

    # 2) Try native session restoration (Strictly as requested)
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.user = session.user
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
        else:
            st.session_state.user = None
    except Exception:
        st.session_state.user = None
                
    st.session_state.auth_initialized = True
    st.rerun()

# 4) Re-sync Supabase client on every rerun using tokens in session state
if st.session_state.user and st.session_state.access_token and st.session_state.refresh_token:
    try:
        supabase.auth.set_session(st.session_state.access_token, st.session_state.refresh_token)
    except Exception as e:
        print(f"DEBUG: Re-sync Error: {e}")

st.markdown("""
<style>
/* ── Typography ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Outfit', sans-serif !important; }

/* ── Accent colours ───────────────────────────────────────── */
:root {
    --accent-teal: #6366f1; /* Now Indigo */
    --accent-amber: #f43f5e; /* Now Rose */
    --accent-coral: #fb7185;
    --bg-card: linear-gradient(145deg, #1a1f2e 0%, #0d1117 100%);
    --border-subtle: rgba(99,102,241,0.15);
    --border-hover: rgba(99,102,241,0.5);
    --text-primary: #f0f4f8;
    --text-secondary: #8899a6;
}

/* ── Global background tint ───────────────────────────────── */
.stApp {
    background-color: #0d1117 !important;
}

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
    box-shadow: 0 4px 24px rgba(99,102,241,0.12), 0 0 0 1px rgba(99,102,241,0.2) !important;
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
    padding: 0.5rem 1rem 1.0rem 1rem;
    text-align: center;
    height: 90px; /* Force locked height so cards align perfectly */
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.card-info h4 {
    margin: 0 0 0.4rem 0;
    font-size: 1.0rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
    line-height: 1.2;
}
.card-info p {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* ── Primary action button (Run Analysis) ─────────────────── */
.stButton > button[kind="primary"] {
    background: var(--accent-teal) !important;
    color: #ffffff !important;
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
    "full_poses_data": {},
    "full_results_dir": None,
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
    # Clean, static hero section
    st.markdown(
        """<div style="width: 100%; display: flex; align-items: center; justify-content: center; margin-top: 1rem; margin-bottom: 3rem;">
<div style="text-align: center; max-width: 800px; padding: 0 1rem;">
<div style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 9999px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1.5rem;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-teal); margin-top: auto; margin-bottom: auto;"></span>
<span style="font-size: 0.875rem; color: rgba(255,255,255,0.6); letter-spacing: 0.025em;">MI4L Clinical Tool</span>
</div>
<h1 style="font-size: clamp(2rem, 4vw, 3.5rem); font-weight: 700; line-height: 1.1; margin-bottom: 1rem; letter-spacing: -0.025em; color: var(--text-primary);">
MWI Mobility Analysis
</h1>
<p style="font-size: clamp(1rem, 2vw, 1.15rem); color: var(--text-secondary); font-weight: 300; letter-spacing: 0.025em; line-height: 1.6; max-width: 600px; margin: 0 auto;">
Analyze joint mobility, calculate Active vs Passive Range of Motion, and generate an MI4L score.
</p>
</div>
</div>""", 
        unsafe_allow_html=True
    )

    # ── Full Assessment Entry Card ──────────────────────────────
    st.markdown('<div class="section-header">🏆 Full MWI Assessment</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-info" style="height: auto; padding: 1.5rem; text-align: left; align-items: flex-start;">
                <h4 style="font-size: 1.25rem;">Full MWI Assessment</h4>
                <p style="font-size: 0.95rem; margin-top: 0.5rem;">Upload all the videos at once, to calculate your complete MWI result and analysis of each individual pose at once</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Select Full Assessment", key="sel_full_assessment", use_container_width=True):
            _go("full_upload")
            st.rerun()
            
    st.markdown('<div class="section-header">🎯 Single Pose Analysis</div>', unsafe_allow_html=True)

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

    # ── Render History Log ───────────────────────────────────
    _render_history_log()
    
# ╔═════════════════════════════════════════════════════════════╗
# ║  HISTORY LOG COMPONENT                                      ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_history_log():
    from datetime import datetime
    import pandas as pd
    
    st.markdown('<div class="section-header" style="margin-top: 3rem;">🕒 Analysis History Log</div>', unsafe_allow_html=True)
    
    if not st.session_state.get("user"):
        st.info("Please log in to view your analysis history.")
        return
        
    user_id = st.session_state.user.id
    try:
        response = supabase.table("analyses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        analyses = response.data
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return
        
    if not analyses:
        st.info("No past analyses found.")
        return
        
    for row in analyses:
        analysis_id = row["id"]
        a_type = row["assessment_type"]
        # Parse Supabase timestamptz correctly
        dt = pd.to_datetime(row["created_at"])
        
        with st.container(border=True):
            cols = st.columns([4, 1], vertical_alignment="center")
            
            with cols[0]:
                if a_type == "full_mwi":
                    icon = "🏆"
                    name = "Full MWI Assessment"
                else:
                    icon = "🎯"
                    pose_key = row.get("pose_name", "Unknown")
                    name = _display_name(pose_key)
                    
                st.markdown(f"<h4 style='margin:0; font-size:1.1rem;'>{icon} {name}</h4>", unsafe_allow_html=True)
                st.caption(f"{dt.strftime('%d %b %Y, %H:%M:%S')} • Cloud ID: {analysis_id[:8]}")
                
            with cols[1]:
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("View", key=f"hist_{analysis_id}", use_container_width=True):
                        # Cache the history data in session state for the specific results page routers to use
                        st.session_state.history_analysis_id = analysis_id
                        st.session_state.history_analysis_data = row
                        
                        if a_type == "full_mwi":
                            _go("full_results")
                        else:
                            st.session_state.selected_pose = row.get("pose_name")
                            _go("results")
                        st.rerun()
                
                with btn_cols[1]:
                    if st.button("🗑️", key=f"del_trigger_{analysis_id}", use_container_width=True, help="Delete this analysis"):
                        st.session_state[f"confirm_delete_{analysis_id}"] = True
                
                # Confirmation logic
                
                if st.session_state.get(f"confirm_delete_{analysis_id}"):

                    try:
                        # 1️⃣ Delete entire storage folder by prefix
                        def delete_storage_prefix(bucket: str, prefix: str, page_size: int = 200):
                            storage = supabase.storage.from_(bucket)

                            files = []
                            stack = [prefix]

                            while stack:
                                p = stack.pop()
                                offset = 0

                                while True:
                                    items = storage.list(p, {"limit": page_size, "offset": offset})
                                    if not items:
                                        break

                                    for it in items:
                                        name = it["name"]
                                        full = f"{p}/{name}"

                                        # Heuristic: files have metadata/created_at/id; folders usually don't.
                                        if it.get("metadata") is not None or it.get("id") is not None:
                                            files.append(full)
                                        else:
                                            stack.append(full)

                                    offset += page_size

                            if files:
                                storage.remove(files)

                        # usage inside delete
                        analysis_prefix = f"users/{user_id}/{analysis_id}"
                        delete_storage_prefix("analysis-results", analysis_prefix)  

                        # 2️⃣ Delete DB row
                        supabase.table("analyses") \
                            .delete() \
                            .eq("id", analysis_id) \
                            .eq("user_id", user_id) \
                            .execute()

                        st.toast("Analysis deleted.")
                        del st.session_state[f"confirm_delete_{analysis_id}"]
                        st.rerun()

                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                        del st.session_state[f"confirm_delete_{analysis_id}"]
                        st.rerun()


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
            import os
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_PROJECT_ROOT / "src")
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(_PROJECT_ROOT),
                env=env,
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

    # ── Supabase Upload ─────────────────────────────────────
    import uuid
    analysis_id = str(uuid.uuid4())
    user_id = st.session_state.user.id
    
    summary_json = {}
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if not df.empty:
                row_dict = df.iloc[0].to_dict()
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                summary_json = row_dict
        except Exception as e:
            print(f"JSON Parse Error: {e}")

    csv_files = []
    # upload all csvs
    for csv_file in out_dir.glob("*.csv"):
        storage_path = f"users/{user_id}/{analysis_id}/{csv_file.name}"
        try:
            with open(csv_file, "rb") as f:
                supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "text/csv"})
            csv_files.append({"name": csv_file.name, "path": storage_path})
        except Exception as e:
            print(f"Storage Upload Error ({csv_file.name}): {e}")

    snapshot_urls = []
    timeplot_urls = []
    
    # upload snapshots
    snap_dir = out_dir / "snapshots"
    if snap_dir.exists() and snap_dir.is_dir():
        for snap_file in snap_dir.glob("*.png"):
            storage_path = f"users/{user_id}/{analysis_id}/snapshots/{snap_file.name}"
            try:
                with open(snap_file, "rb") as f:
                    supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "image/png"})
                url = supabase.storage.from_("analysis-results").create_signed_url(storage_path, 31536000)["signedURL"] # 1 year
                snapshot_urls.append({"name": snap_file.name, "url": url, "path": storage_path})
            except Exception as e:
                print(f"Snapshot Upload Error ({snap_file.name}): {e}")
                
    # upload plots
    plot_files = sorted(out_dir.glob("plot_*.png"))
    for plot_file in plot_files:
        storage_path = f"users/{user_id}/{analysis_id}/plots/{plot_file.name}"
        try:
            with open(plot_file, "rb") as f:
                supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "image/png"})
            url = supabase.storage.from_("analysis-results").create_signed_url(storage_path, 31536000)["signedURL"] # 1 year
            timeplot_urls.append({"name": plot_file.name, "url": url, "path": storage_path})
        except Exception as e:
            print(f"Plot Upload Error ({plot_file.name}): {e}")

    try:
        analysis_data = {
            "id": analysis_id,
            "user_id": user_id,
            "assessment_type": "pose_single",
            "pose_name": pose_key,
            "summary_json": summary_json,
            "csv_files": csv_files,
            "snapshot_urls": snapshot_urls,
            "timeplot_urls": timeplot_urls,
            "created_at": pd.Timestamp.now().isoformat()
        }
        res = supabase.table("analyses").insert(analysis_data).execute()
        print(f"DEBUG: Supabase Insert Result (Single): {res}")
        st.session_state.history_analysis_id = analysis_id
        st.session_state.history_analysis_data = analysis_data
    except Exception as e:
        print(f"DEBUG: Supabase Insert Error (Single): {e}")
        st.warning(f"Failed to save to cloud history: {e}")

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
    pose_key = st.session_state.selected_pose
    meta = POSE_METADATA.get(pose_key, {})
    is_hist = st.session_state.get("history_analysis_id") is not None
    
    if is_hist:
        hist_data = st.session_state.history_analysis_data
        row = hist_data.get("summary_json", {})
        csv_files = hist_data.get("csv_files", [])
        out_dir = Path("historical_run")
    else:
        out_dir = Path(st.session_state.results_dir)
        summary_path = out_dir / "summary.csv"
        if not summary_path.exists():
            st.error("summary.csv not found in output directory.")
            return
        df = pd.read_csv(summary_path)
    
        if df.empty:
            st.warning("No results were produced. Check QC flags or video quality.")
            return
    
        row = df.iloc[0]  # single-row summary (per the pipeline design)

    # ── Header ──────────────────────────────────────────────
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown(
            f"<h2>Results — {_display_name(pose_key)}</h2>",
            unsafe_allow_html=True,
        )
    with head_right:
        if st.button("🏠  New Analysis"):
            st.session_state.history_analysis_id = None
            st.session_state.history_analysis_data = None
            _go("landing")
            st.rerun()

    # ── Console output (collapsed) ──────────────────────────
    if not is_hist:
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

    if is_hist:
        snap_urls = hist_data.get("snapshot_urls", [])
        tp_urls = hist_data.get("timeplot_urls", [])
        
        if tp_urls:
            cols = st.columns(min(len(tp_urls), 2))
            for i, item in enumerate(tp_urls):
                with cols[i % len(cols)]:
                    st.image(item["url"], caption=item["name"].replace("_", " ").title(), use_container_width=True)
        else:
            st.info("No plot images found for this record.")
            
        if snap_urls:
            st.markdown("**Snapshots**")
            cols = st.columns(min(len(snap_urls), 3))
            for i, item in enumerate(snap_urls):
                with cols[i % len(cols)]:
                    st.image(item["url"], caption=item["name"].replace("_", " ").title(), use_container_width=True)
        else:
            st.info("No snapshots found for this record.")
    else:
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
    
    if is_hist:
        st.markdown("📥 Download CSVs directly from Cloud Storage:")
        for cf in csv_files:
            try:
                res = supabase.storage.from_("analysis-results").create_signed_url(cf["path"], 3600)
                url = res.get("signedURL")
                if url:
                    st.markdown(f"- [**Download {cf['name']}**]({url})")
            except Exception as e:
                st.error(f"Failed to load {cf['name']}: {e}")
    else:
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
# ║  SCREEN 5 — FULL UPLOAD                                     ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_full_upload():
    if st.button("← Back to poses"):
        _go("landing")
        st.rerun()

    st.markdown("## 🏆 Full MWI Assessment")
    st.markdown("Upload your AROM and PROM videos for all supported poses to generate a complete scorecard.")

    data = st.session_state.full_poses_data
    total_poses = len(POSE_KEYS)
    uploaded_poses = sum(1 for k in POSE_KEYS if data.get(k, {}).get("arom") is not None)

    st.markdown(
        """
        <style>
        /* Target the first major column to act as a sticky sidebar */
        [data-testid="column"]:first-of-type {
            position: sticky;
            top: 4rem;
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 10;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    side_col, main_col = st.columns([1, 2.5], gap="large")

    with side_col:
        st.markdown("### 📊 Assessment Progress", unsafe_allow_html=True)
        st.progress(uploaded_poses / total_poses if total_poses > 0 else 0)
        st.markdown(f"**{uploaded_poses} of {total_poses} poses uploaded**")
        st.divider()
        if st.button("🚀 Begin Full Analysis", use_container_width=True, type="primary", disabled=uploaded_poses == 0):
            _go("full_processing")
            st.rerun()

    with main_col:
        for pose_key in POSE_KEYS:
            bilateral = pose_key in BILATERAL_POSES
            meta = POSE_METADATA.get(pose_key, {})
            
            with st.container(border=True):
                if pose_key not in data:
                    data[pose_key] = {"arom": None, "prom": None, "side": "both" if bilateral else "right"}
                    
                img_col, ctrl_col = st.columns([1, 2.5], gap="large")
                
                with img_col:
                    img_path = POSE_IMAGES.get(pose_key)
                    if img_path and img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    st.markdown(f"#### {_display_name(pose_key)}")
                    st.caption(f"{meta.get('joint_name', '').title()} · {'Bilateral' if bilateral else 'Unilateral'}")
                    
                with ctrl_col:
                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
                    if not bilateral:
                        side = st.radio("Target Side", ["Left", "Right"], key=f"side_{pose_key}", horizontal=True, 
                                        index=0 if data[pose_key]["side"] == "left" else 1)
                        data[pose_key]["side"] = side.lower()
                    
                    is_stick_pass = pose_key == "shoulder_stick_pass_through"
                    
                    if is_stick_pass:
                        arom = st.file_uploader("Upload Video *", type=["mp4", "mov", "avi"], key=f"arom_{pose_key}")
                        data[pose_key]["arom"] = arom
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            arom = st.file_uploader("AROM Video *", type=["mp4", "mov", "avi"], key=f"arom_{pose_key}")
                            data[pose_key]["arom"] = arom
                        with c2:
                            prom = st.file_uploader("PROM Video *", type=["mp4", "mov", "avi"], key=f"prom_{pose_key}")
                            data[pose_key]["prom"] = prom

# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 6 — FULL PROCESSING                                 ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_full_processing():
    # ── 1. Render background UI but blur it out ─────────────
    st.markdown(
        """
        <style>
        /* Blur the main app elements slightly */
        .stApp > header, .main .block-container {
            filter: blur(5px) brightness(0.5);
            pointer-events: none;
            user-select: none;
        }
        
        /* The foreground processing modal */
        #processing-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 999999;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid rgba(99,102,241,0.4);
            border-radius: 20px;
            padding: 4rem 3rem;
            width: 90%;
            max-width: 650px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(99,102,241,0.2);
            backdrop-filter: blur(16px);
            text-align: center;
        }
        
        #processing-modal h2 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.75rem;
            color: #f0f4f8;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }
        
        #processing-modal p.subtext {
            color: rgba(255,255,255,0.6);
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        </style>
        
        <div id="processing-modal">
            <h2>⚙️ Pipeline Processing</h2>
            <p class="subtext">Analyzing biomechanics across all poses...</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ── 2. Render actual progress containers using Streamlit logic ────
    # They will visually hover on the blurred background because Streamlit appends them after the style tags
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assessments_dir = _PROJECT_ROOT / "results" / "assessments"
    assessments_dir.mkdir(parents=True, exist_ok=True)
    out_dir = assessments_dir / f"{timestamp}_Session"
    out_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.full_results_dir = str(out_dir)

    tmp_root = Path(tempfile.mkdtemp(prefix="mi4l_full_"))
    
    pipeline_python = _find_pipeline_python()
    script_path = str(_PROJECT_ROOT / "scripts" / "run_mi4l.py")
    config_path = str(_PROJECT_ROOT / "configs" / "default.yaml")

    data = st.session_state.full_poses_data
    poses_to_run = [k for k in POSE_KEYS if data.get(k, {}).get("arom") is not None]
    
    # Empty placeholders that act as our modal content
    modal_container = st.container()
    with modal_container:
        st.markdown(
            """
            <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -10%); z-index: 9999999; width: 100%; max-width: 550px; text-align: center; padding: 0 2rem;">
            """, unsafe_allow_html=True
        )
        status_text = st.empty()
        progress_bar = st.progress(0)
        action_button = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        
    all_summary_data = []

    for i, pose_key in enumerate(poses_to_run):
        pose_data = data[pose_key]
        status_text.markdown(f"<h3 style='color: white; font-size: 1.25rem; font-weight: 500; margin-bottom: 1rem;'>Processing {_display_name(pose_key)} ({i+1}/{len(poses_to_run)})...</h3>", unsafe_allow_html=True)
        
        pose_tmp = tmp_root / pose_key
        pose_tmp.mkdir(parents=True, exist_ok=True)
        pose_out = out_dir / pose_key
        pose_out.mkdir(parents=True, exist_ok=True)
        
        arom_file = pose_data["arom"]
        prom_file = pose_data["prom"]
        
        arom_path = pose_tmp / "arom.mp4"
        arom_path.write_bytes(arom_file.getvalue())
        
        cmd = [
            pipeline_python, script_path,
            "--arom", str(arom_path),
            "--out", str(pose_out),
            "--pose", pose_key,
            "--side", pose_data["side"],
            "--config", config_path,
        ]
        
        if prom_file is not None:
            prom_path = pose_tmp / "prom.mp4"
            prom_path.write_bytes(prom_file.getvalue())
            cmd.extend(["--prom", str(prom_path)])
            
        try:
            import os
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_PROJECT_ROOT / "src")
            res = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(_PROJECT_ROOT), 
                timeout=600,
                env=env
            )
            summary_csv = pose_out / "summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                if not df.empty:
                    row_dict = df.iloc[0].to_dict()
                    row_dict["pose_key"] = pose_key
                    all_summary_data.append(row_dict)
                    
        except Exception as e:
            st.error(f"Error processing {pose_key}: {e}")
            
        progress_bar.progress((i + 1) / len(poses_to_run))
        
    status_text.markdown("<h3 style='color: #10b981;'>✅ Processing Complete!</h3>", unsafe_allow_html=True)
    
    import json
    import uuid
    
    # ── Supabase Upload ─────────────────────────────────────
    analysis_id = str(uuid.uuid4())
    user_id = st.session_state.user.id
    csv_files = []
    
    # Clean NaNs in summary data for JSON DB insertion
    db_summary_data = []
    for d in all_summary_data:
        clean_d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
        db_summary_data.append(clean_d)
    
    snapshot_urls = []
    timeplot_urls = []
    
    for pose_dir_item in out_dir.iterdir():
        if pose_dir_item.is_dir():
            pose_key = pose_dir_item.name
            
            # upload all csvs
            for csv_file in pose_dir_item.glob("*.csv"):
                storage_path = f"users/{user_id}/{analysis_id}/{pose_key}/{csv_file.name}"
                try:
                    with open(csv_file, "rb") as f:
                        supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "text/csv"})
                    csv_files.append({"name": f"{pose_key}_{csv_file.name}", "path": storage_path, "pose_key": pose_key})
                except Exception as e:
                    print(f"Storage Upload Error ({csv_file.name}): {e}")
            
            # upload snapshots
            pose_snap_dir = pose_dir_item / "snapshots"
            if pose_snap_dir.exists() and pose_snap_dir.is_dir():
                for snap_file in pose_snap_dir.glob("*.png"):
                    storage_path = f"users/{user_id}/{analysis_id}/{pose_key}/snapshots/{snap_file.name}"
                    try:
                        with open(snap_file, "rb") as f:
                            supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "image/png"})
                        url = supabase.storage.from_("analysis-results").create_signed_url(storage_path, 31536000)["signedURL"]
                        snapshot_urls.append({"name": snap_file.name, "url": url, "path": storage_path, "pose_key": pose_key})
                    except Exception as e:
                        print(f"Snapshot Upload Error ({snap_file.name}): {e}")
            
            # upload plots
            pose_plot_files = sorted(pose_dir_item.glob("plot_*.png"))
            for plot_file in pose_plot_files:
                storage_path = f"users/{user_id}/{analysis_id}/{pose_key}/plots/{plot_file.name}"
                try:
                    with open(plot_file, "rb") as f:
                        supabase.storage.from_("analysis-results").upload(file=f, path=storage_path, file_options={"content-type": "image/png"})
                    url = supabase.storage.from_("analysis-results").create_signed_url(storage_path, 31536000)["signedURL"]
                    timeplot_urls.append({"name": plot_file.name, "url": url, "path": storage_path, "pose_key": pose_key})
                except Exception as e:
                    print(f"Plot Upload Error ({plot_file.name}): {e}")
                    
    try:
        analysis_data = {
            "id": analysis_id,
            "user_id": user_id,
            "assessment_type": "full_mwi",
            "pose_name": None,
            "summary_json": db_summary_data,
            "csv_files": csv_files,
            "snapshot_urls": snapshot_urls,
            "timeplot_urls": timeplot_urls,
            "created_at": pd.Timestamp.now().isoformat()
        }
        res = supabase.table("analyses").insert(analysis_data).execute()
        print(f"DEBUG: Supabase Insert Result (Full): {res}")
        st.session_state.history_analysis_id = analysis_id
        st.session_state.history_analysis_data = analysis_data
    except Exception as e:
        print(f"DEBUG: Supabase Insert Error (Full): {e}")
        st.warning(f"Failed to save to cloud history: {e}")
    
    with open(out_dir / "master_summary.json", "w") as f:
        json.dump(all_summary_data, f, indent=4)
        
    with action_button.container():
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("View Mobility Report Card", type="primary", use_container_width=True):
            _go("full_results")
            st.rerun()
        _go("full_results")
        st.rerun()

# ╔═════════════════════════════════════════════════════════════╗
# ║  SCREEN 7 — FULL RESULTS                                    ║
# ╚═════════════════════════════════════════════════════════════╝
def _render_full_results():
    if st.button("🏠 New Assessment"):
        st.session_state.history_analysis_id = None
        st.session_state.history_analysis_data = None
        _go("landing")
        st.rerun()

    is_hist = st.session_state.get("history_analysis_id") is not None
    if is_hist:
        hist_data = st.session_state.history_analysis_data
        all_data = hist_data.get("summary_json", [])
        csv_files = hist_data.get("csv_files", [])
        dt_str = pd.to_datetime(hist_data['created_at']).strftime('%d %b %Y, %H:%M:%S')
        sess_name = f"Historical Analysis ({dt_str})"
        out_dir = Path("historical_run")
    else:
        out_dir = Path(st.session_state.full_results_dir)
        json_path = out_dir / "master_summary.json"
        
        if not json_path.exists():
            st.error("No summary data found for this session.")
            return
            
        import json
        with open(json_path, "r") as f:
            all_data = json.load(f)
            
        if not all_data:
            st.warning("No valid results were generated.")
            return
            
        sess_name = out_dir.name
        csv_files = []
        
    st.markdown("## 🏆 Mobility Report Card")
    st.caption(f"Session: {sess_name}")
    
    valid_mi4ls = [d["mi4l"] for d in all_data if d.get("mi4l_valid", str(d.get("mi4l_valid")) == "True") and pd.notna(d.get("mi4l"))]
    global_mi4l = sum(valid_mi4ls) / len(valid_mi4ls) if valid_mi4ls else None
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Poses Analyzed", len(all_data))
    
    mi4l_display = f"{global_mi4l:.3f}" if global_mi4l is not None else "—"
    c2.metric("Global MI4L Index", mi4l_display, "Average Body Mobility", delta_color="normal")
    
    st.markdown("### 🌡️ Body Heatmap (Stiffness vs. Mobility)")
    st.markdown("Green indicates high mobility (MI4L >= 0.20 or good range), Red indicates stiffness.")
    
    import plotly.graph_objects as go
    
    joints = []
    scores = []
    colors = []
    
    for d in all_data:
        j_name = d.get("joint_name", "Unknown").title()
        pk = d["pose_key"]
        val = d.get("mi4l")
        if pd.isna(val) or not d.get("mi4l_valid", str(d.get("mi4l_valid")) == "True"):
            val = 0.0 # Default if invalid just for color mapping
        
        joints.append(f"{_display_name(pk)}")
        scores.append(val)
        
        if val < 0.1:
            colors.append("#ef4444") # Red
        elif val < 0.2:
            colors.append("#f59e0b") # Amber
        else:
            colors.append("#10b981") # Green
            
    fig = go.Figure(data=[go.Bar(
        x=joints,
        y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition='auto',
    )])
    fig.update_layout(title_text='Joint Mobility (MI4L Score)', yaxis_title='MI4L Score', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("## 📋 Detailed Pose Reports")
    
    for d in all_data:
        pose_key = d["pose_key"]
        name = _display_name(pose_key)
        pose_dir = out_dir / pose_key
        
        st.markdown(f"### ❖ {name}")
        
        # ── CORE METRICS ────────────────────────────────────────
        st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AROM Peak", _fmt(d.get("arom_deg"), "°"))
        c2.metric("PROM Peak", _fmt(d.get("prom_deg"), "°"))
        c3.metric("Assist Gap", _fmt(d.get("assist_gap"), "°"))
        c_mi4l = d.get("mi4l")
        mi4l_display_pose = _fmt(c_mi4l, decimals=3) if pd.notna(c_mi4l) else "—"
        mi4l_valid_pose = d.get("mi4l_valid", str(d.get("mi4l_valid")) == "True")
        c4.metric("MI4L Score", mi4l_display_pose, delta="Valid" if mi4l_valid_pose else "Invalid",
                  delta_color="normal" if mi4l_valid_pose else "inverse")
                  
        # ── QUALITY METRICS ─────────────────────────────────────
        st.markdown('<div class="section-header">End-Range Quality</div>', unsafe_allow_html=True)
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Peak Hold Time", _fmt(d.get("peak_hold_time_s"), " s", 2))
        q2.metric("Peak Stability", _fmt(d.get("peak_band_std_deg"), "°", 2))
        q3.metric("Time to Peak", _fmt(d.get("time_to_peak_s"), " s", 2))
        q4.metric("Jerk RMS", _fmt(d.get("jerk_rms"), decimals=2))
        q5.metric("Fit R²", _fmt(d.get("fit_r2"), decimals=3))
        
        # ── COMPENSATION METRICS ────────────────────────────────
        st.markdown('<div class="section-header">Compensation</div>', unsafe_allow_html=True)
        comp1, comp2, comp3 = st.columns(3)
        comp1.metric("Torso Angle Change", _fmt(d.get("torso_angle_change_deg"), "°", 2))
        comp2.metric("Pelvis Drift", _fmt(d.get("pelvis_drift_norm"), decimals=3))
        comp3.metric("Frames Valid", _fmt(d.get("frames_valid_pct"), "%", 1))
        
        # ── QC FLAGS ────────────────────────────────────────────
        flags = str(d.get("qc_flags", ""))
        if flags and flags != "nan":
            with st.expander("⚠️  QC Flags", expanded=False):
                for f in flags.split(";"):
                    if f.strip():
                        st.markdown(f"- `{f.strip()}`")

        # ── VISUALISATIONS ──────────────────────────────────────
        st.markdown('<div class="section-header">Visualisations</div>', unsafe_allow_html=True)

        if is_hist:
            pose_snaps = [s for s in hist_data.get("snapshot_urls", []) if s.get("pose_key") == pose_key]
            pose_plots = [p for p in hist_data.get("timeplot_urls", []) if p.get("pose_key") == pose_key]
            
            if pose_plots:
                cols = st.columns(min(len(pose_plots), 2))
                for i, item in enumerate(pose_plots):
                    with cols[i % len(cols)]:
                        st.image(item["url"], caption=item["name"].replace("_", " ").title(), use_container_width=True)
            else:
                st.info("No plot images found for this pose.")
            
            if pose_snaps:
                st.markdown("**Snapshots**")
                cols = st.columns(min(len(pose_snaps), 3))
                for i, item in enumerate(pose_snaps):
                    with cols[i % len(cols)]:
                        st.image(item["url"], caption=item["name"].replace("_", " ").title(), use_container_width=True)
            else:
                st.info("No snapshots found for this pose.")
        else:
            # Plots
            plot_files = sorted(pose_dir.glob("plot_*.png"))
            if plot_files:
                plot_cols = st.columns(min(len(plot_files), 2))
                for i, pf in enumerate(plot_files):
                    with plot_cols[i % len(plot_cols)]:
                        st.image(str(pf), caption=pf.stem.replace("_", " ").title(), use_container_width=True)
            else:
                st.info("No plot images were generated.")
    
            # Snapshots
            snap_dir = pose_dir / "snapshots"
            if snap_dir.exists():
                snap_files = sorted(snap_dir.glob("*.png"))
                if snap_files:
                    st.markdown("**Snapshots**")
                    snap_cols = st.columns(min(len(snap_files), 3))
                    for i, sf in enumerate(snap_files):
                        with snap_cols[i % len(snap_cols)]:
                            st.image(str(sf), caption=sf.stem.replace("_", " ").title(), use_container_width=True)

        # ── EXPORT / DOWNLOADS ──────────────────────────────────
        # ── EXPORT / DOWNLOADS ──────────────────────────────────
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        
        if is_hist:
            pose_csvs = [cf for cf in csv_files if cf.get("pose_key") == pose_key]
            if pose_csvs:
                st.markdown("📥 Download CSVs directly from Cloud Storage:")
                for cf in pose_csvs:
                    try:
                        res = supabase.storage.from_("analysis-results").create_signed_url(cf["path"], 3600)
                        url = res.get("signedURL")
                        if url:
                            st.markdown(f"- [**Download {cf['name']}**]({url})")
                    except Exception as e:
                        pass
        else:
            d1, d2, d3, d4 = st.columns(4)
    
            # Summary CSV
            summary_path = pose_dir / "summary.csv"
            with d1:
                if summary_path.exists():
                    st.download_button(
                        "📋 Summary CSV",
                        data=summary_path.read_bytes(),
                        file_name=f"{pose_key}_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_sum_{pose_key}"
                    )
    
            # Full angle CSVs
            angle_csvs = sorted(pose_dir.glob("angles_*.csv"))
            with d2:
                if angle_csvs:
                    for ac in angle_csvs:
                        st.download_button(
                            f"📄 {ac.stem.replace('_', ' ').title()}",
                            data=ac.read_bytes(),
                            file_name=f"{pose_key}_{ac.name}",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"dl_ang_{pose_key}_{ac.name}",
                        )
    
            # Plot downloads
            with d3:
                if plot_files:
                    for pf in plot_files:
                        st.download_button(
                            f"📈 {pf.stem.replace('_', ' ').title()}",
                            data=pf.read_bytes(),
                            file_name=f"{pose_key}_{pf.name}",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl_plt_{pose_key}_{pf.name}",
                        )
    
            # Snapshots download
            with d4:
                if snap_dir.exists():
                    snap_files_all = sorted(snap_dir.glob("*.png"))
                    for sf in snap_files_all:
                        st.download_button(
                            f"🖼️ {sf.stem.replace('_', ' ').title()}",
                            data=sf.read_bytes(),
                            file_name=f"{pose_key}_{sf.name}",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl_snp_{pose_key}_{sf.name}",
                        )
        
        st.write("")
        st.markdown("---")


# ╔═════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                    ║
# ╚═════════════════════════════════════════════════════════════╝
_SCREENS = {
    "landing":    _render_landing,
    "upload":     _render_upload,
    "processing": _render_processing,
    "results":    _render_results,
    "full_upload": _render_full_upload,
    "full_processing": _render_full_processing,
    "full_results": _render_full_results,
}

def _render_auth():
    st.markdown("<h2 style='text-align: center; margin-top: 3rem;'>MWI Mobility Analysis Login</h2>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.container(border=True):
                st.info("Sign up or log in to sync your mobility history to the cloud.")
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                st.write("")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Login", use_container_width=True, key="login_button"):
                        try:
                            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = res.user
                            st.session_state.access_token = res.session.access_token
                            st.session_state.refresh_token = res.session.refresh_token
                            cookie_manager.set("access_token", res.session.access_token, max_age=86400 * 30, key="set_access")
                            cookie_manager.set("refresh_token", res.session.refresh_token, max_age=86400 * 30, key="set_refresh")
                            time.sleep(0.5) # allow cookies to set
                            st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")
                with col_btn2:
                    if st.button("Sign Up", use_container_width=True, key="signup_button"):
                        try:
                            res = supabase.auth.sign_up({"email": email, "password": password})
                            st.success("Signup successful! You can now log in.")
                        except Exception as e:
                            st.error(f"Signup failed: {e}")

if not st.session_state.get("auth_initialized"):
    st.stop()

if not st.session_state.get("user"):
    _render_auth()
    st.stop()

# Render logout button in sidebar
with st.sidebar:
    if st.button("Log Out", key="logout_button"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        cookie_manager.delete("access_token", key="del_access")
        cookie_manager.delete("refresh_token", key="del_refresh")
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        time.sleep(0.5)
        st.rerun()
        
_SCREENS.get(st.session_state.screen, _render_landing)()
