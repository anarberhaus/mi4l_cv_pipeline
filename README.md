# MWI CV Pipeline

Computer-vision pipeline that estimates joint angles from RGB video and computes **AROM**, **PROM**, and the **Mobility Index for Longevity (MI4L)** score.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Setup](#setup)
3. [Quick start](#quick-start)
4. [Running a single pose](#running-a-single-pose)
5. [Running all poses at once](#running-all-poses-at-once)
6. [Aggregate analysis (thesis figures)](#aggregate-analysis-thesis-figures)
7. [Supported poses](#supported-poses)
8. [Outputs](#outputs)
9. [Configuration](#configuration-configsdefaultyaml)
10. [Tests](#tests)
11. [How it works](#how-it-works)

---

## Project structure

```
mwi_cv_pipeline/
├── src/mi4l/
│   ├── angles/       # Joint angle computation (per pose)
│   ├── io/           # Landmark / CSV / config I/O
│   ├── metrics/      # AROM/PROM estimation, MI4L, summary metrics
│   ├── pose/         # MediaPipe landmark extraction
│   ├── qc/           # Quality-control rules and flags
│   ├── supabase_client.py # Supabase integration logic
│   ├── utils/        # Config loading, helpers
│   └── viz/          # Plots and snapshot export
├── scripts/
│   ├── run_mi4l.py   # Main CLI: run ONE pose
│   ├── run_all.py    # Convenience script: run ALL poses
│   ├── batch_process.py   # Batch folder processing (optional)
│   └── audit_results.py   # Inspect / audit result folders (optional)
├── analysis/
│   ├── pipeline_performance.py  # Aggregate metrics + plots from results/ (run second)
│   ├── movement_analysis.py     # Movement / time-series figures (run after pipeline_performance)
│   └── outputs/                   # Generated CSVs and plots (gitignored by default)
├── configs/
│   └── default.yaml  # Default config (thresholds, params)
├── data/             # Your video files (gitignored)
│   ├── arom/         # AROM video files go here
│   ├── prom/         # PROM video files go here
│   └── other/        # Videos that are neither arom nor prom
├── results/          # Local pipeline outputs (gitignored)
├── tests/            # Unit tests
├── app.py            # Streamlit GUI for analysis and history tracking
├── Dockerfile        # Docker setup for Render / Linux deployments
├── requirements.txt  # Python dependencies (pip)
└── packages.txt      # OS-level dependencies (apt-get)
```

---

## Setup

> **Note on MediaPipe Compatibility:**
> This repository strictly pins `mediapipe==0.10.14` and relies on Python 3.10. MediaPipe versions >= 0.10.30 have removed the Legacy Solutions API, which this pipeline is built upon. 

You need a Supabase project to use the Streamlit interface. Copy `.env.example` to `.env` and fill in your values (or create `.env` manually):
```env
SUPABASE_URL="your-project-url"
SUPABASE_KEY="your-anon-key"
```

Keep `.env` and any other secrets **out of version control** (they are listed in `.gitignore`). Local video folders (`data/`), pipeline outputs (`results/`), and generated analysis exports (`analysis/outputs/`) are also ignored by default so participant media and aggregated tables are not pushed to GitHub by mistake.

### Option A: Local Conda (Windows/Mac)

```bash
# 1. Create a dedicated environment with Python 3.10
conda create -n mi4l python=3.10 -y
conda activate mi4l

# 2. Install dependencies via pip (inside the activated env)
pip install -r requirements.txt

# 3. Install the pipeline package itself in editable mode
pip install -e .
```

After this, always activate your environment before running:
```bash
conda activate mi4l
```

### Option B: plain virtual environment (venv)

If you prefer not to use conda, make sure you are on **Python 3.10 or 3.11** first, then:

```bash
# Create and activate venv
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Verify the installation

```bash
python -c "import mediapipe as mp; mp.solutions.pose.PoseLandmark; print('All good!')"
```

If this prints `All good!` you are set. If you get an `AttributeError` regarding `solutions`, you are not on `mediapipe==0.10.14`.

### Option C: Docker (Server Deployment)

For deploying to services like Render, we use the provided `Dockerfile` built on `python:3.10-slim`.
This automatically installs the required Linux graphics drivers (`libgl1`, `libsm6`, `libxext6`) needed to initialize MediaPipe's C++ backends.
```bash
docker build -t mi4l_app .
docker run -p 8501:8501 --env-file .env mi4l_app
```

---

## Quick start

**All commands are run from the project root directory.**

### Launching the Web Interface
The recommended way to use the pipeline is via the Streamlit GUI. It includes Supabase authentication, session persistence, and a dark-mode video processing UI.

```bash
# Ensure your .env file exists with Supabase credentials
conda activate mi4l
streamlit run app.py
```

### Running CLI Commands
For headless batch processing:

```bash
# Run a single pose (example: right knee flexion)
python scripts/run_mi4l.py \
  --arom  data/arom/knee_flex_ra.mp4 \
  --prom  data/prom/knee_flex_rp.mp4 \
  --out   results/kneeling_knee_flexion_right \
  --pose  kneeling_knee_flexion \
  --side  right \
  --config configs/default.yaml

# Run all poses defined in run_all.py at once
python scripts/run_all.py
```

Results are saved to the folder specified by `--out` (or to `results/<pose>/` when using `run_all.py`). When using the GUI, results are automatically uploaded to your Supabase project.

---

## Aggregate analysis (thesis figures)

After you have one or more **batch folders** under `results/` (each containing per-participant `summary.csv` trees), you can build combined tables and publication-style plots.

**Prerequisites:** `results/` with pipeline outputs.

From the **repository root**:

```bash
conda activate mi4l   # or your venv
python analysis/pipeline_performance.py
python analysis/movement_analysis.py
```

- `pipeline_performance.py` writes aggregated CSVs and plots (e.g. success rate, MWI distribution, QC frames) under `analysis/outputs/`.
- `movement_analysis.py` adds movement-focused summaries and additional plots (e.g. variability, hold duration, time-series profiles).

Regenerate these whenever you add or change batches under `results/`.

---

## Running a single pose

Use `scripts/run_mi4l.py` when you want to process one specific pose.

### Syntax

```bash
python scripts/run_mi4l.py \
  --arom   <path/to/arom_video.mp4> \
  --prom   <path/to/prom_video.mp4> \   # optional; omit for AROM-only
  --out    <output_folder/> \
  --pose   <pose_name> \
  --side   <left|right|both> \
  --config configs/default.yaml
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--arom` | ✅ | Path to the AROM video file |
| `--prom` | ❌ | Path to the PROM video file. Omit to run AROM-only (MI4L will not be computed) |
| `--out` | ✅ | Output directory (created automatically if it does not exist) |
| `--pose` | ✅ | Pose name (see [Supported poses](#supported-poses)) |
| `--side` | ✅ | `left`, `right`, or `both` |
| `--config` | ✅ | Path to YAML config file (use `configs/default.yaml`) |

### Examples

**Knee flexion, right side, AROM + PROM:**
```bash
python scripts/run_mi4l.py \
  --arom  data/arom/knee_flex_ra.mp4 \
  --prom  data/prom/knee_flex_rp.mp4 \
  --out   results/kneeling_knee_flexion_right \
  --pose  kneeling_knee_flexion \
  --side  right \
  --config configs/default.yaml
```

**Shoulder flexion, AROM only, left side:**
```bash
python scripts/run_mi4l.py \
  --arom  data/arom/shoulder_la.mp4 \
  --out   results/shoulder_flexion \
  --pose  shoulder_flexion \
  --side  left \
  --config configs/default.yaml
```

**Bilateral leg straddle, both sides:**
```bash
python scripts/run_mi4l.py \
  --arom  data/arom/legSpread_arom.mp4 \
  --prom  data/prom/legSpread_prom.mp4 \
  --out   results/bilateral_leg_straddle \
  --pose  bilateral_leg_straddle \
  --side  both \
  --config configs/default.yaml
```

---

## Running all poses at once

`scripts/run_all.py` loops over every pose defined in its `POSES` list and calls `run_mi4l.py` for each one sequentially.

```bash
python scripts/run_all.py
```

### What it does

1. Auto-detects a suitable Python interpreter (one that has mediapipe with `mp.solutions.pose` available). If your conda environment is active, it will be found automatically.
2. Runs each pose in sequence. Poses whose AROM video file is not found on disk are skipped with a `[SKIP]` message.
3. Prints a summary table at the end showing which poses succeeded.

### Customising the pose list

Open `scripts/run_all.py` and edit the `POSES` list near the top. Each entry maps a pose name to its video files, output folder, and side:

```python
{
    "name": "kneeling_knee_flexion",
    "arom": "data/arom/knee_flex_ra.mp4",
    "prom": "data/prom/knee_flex_rp.mp4",   # set to None for AROM-only
    "out":  "results/kneeling_knee_flexion_right",
    "side": "right",
},
```

If `run_all.py` picks the wrong Python interpreter, set `PYTHON_OVERRIDE` at the top of the file to the full path of your interpreter:

```python
# scripts/run_all.py, line ~88
PYTHON_OVERRIDE = r"C:\Users\you\anaconda3\envs\mi4l\python.exe"
```

---

## Supported poses

| `--pose` value | Movement | `--side` |
|---|---|---|
| `kneeling_knee_flexion` | Knee Flexion | `left` or `right` |
| `prone_trunk_extension` | Trunk / Lumbar Extension | `both` |
| `standing_hip_abduction` | Hip Abduction | `left` or `right` |
| `bilateral_leg_straddle` | Bilateral Leg Straddle | `both` |
| `unilateral_hip_extension` | Hip Extension (one leg) | `left` or `right` |
| `shoulder_flexion` | Shoulder Flexion | `left` or `right` |
| `shoulder_stick_pass_through` | Shoulder Stick Pass-Through | `both` |

---

## Outputs

All output files are written to the folder specified by `--out`:

| File | Description |
|---|---|
| `summary.csv` | One-row-per-side summary with ROM scores, biomechanical metrics, and data quality (see column reference below) |
| `angles_arom.csv` | Per-frame angle values for the AROM video |
| `angles_prom.csv` | Per-frame angle values for the PROM video |
| `landmarks_arom.csv` | Raw MediaPipe landmark coordinates (AROM) |
| `landmarks_prom.csv` | Raw MediaPipe landmark coordinates (PROM) |
| `plot_<pose>_arom.png` | Angle-over-time plot (AROM) |
| `plot_<pose>_prom.png` | Angle-over-time plot (PROM) |
| `snapshots/` | Best-frame annotated images for each side |
| `config_used.yaml` | Snapshot of the config used for this run |

### `summary.csv` column reference

| Column | Type | Description |
|---|---|---|
| **Metadata** | | |
| `movement_name` | string | Human-readable pose name (e.g. "Kneeling Knee Flexion") |
| `joint_name` | string | Anatomical joint assessed (knee, hip, trunk, shoulder) |
| `angle_type` | string | Measurement geometry: `vector-reference`, `vector-vector`, or `distance` |
| `side` | string | Body side: `left`, `right`, or `both` |
| **Core ROM metrics** | | |
| `arom_deg` | float | Active Range of Motion peak (degrees), top-k mean |
| `prom_deg` | float | Passive Range of Motion peak (degrees), top-k mean |
| `mi4l` | float | Mobility Index for Longevity: `(PROM − AROM) / PROM` |
| `arom_confidence` | float | Estimation confidence for AROM (0 to 1) |
| `prom_confidence` | float | Estimation confidence for PROM (0 to 1) |
| `mi4l_valid` | bool | Whether the MI4L score passed all validation checks |
| `qc_flags` | string | Semicolon-separated quality control flags |
| **Derived metric** | | |
| `assist_gap` | float | `prom_deg − arom_deg`: passive capacity beyond active control |
| **End-range quality** | | |
| `peak_hold_time_s` | float | Longest consecutive time (s) within 2% of peak value |
| `peak_band_std_deg` | float | Std deviation (°) of angles within the near-peak band |
| `time_to_peak_s` | float | Time (s) from movement start to first entry into near-peak band |
| **Motor control** | | |
| `fit_r2` | float | R² of a degree-3 polynomial fit to the angle time-series |
| `fit_rmse_deg` | float | RMSE (°) of that polynomial fit |
| `jerk_rms` | float | RMS of the second derivative of the angle signal |
| **Compensation** | | |
| `torso_angle_change_deg` | float | Change in torso orientation (°) from movement start to peak |
| `pelvis_drift_norm` | float | Normalised horizontal displacement of hip midpoint during movement |
| **Reliability** | | |
| `frames_valid_pct` | float | Percentage of frames with valid pose detection in the movement window |
| `avg_landmark_visibility` | float | Mean landmark visibility score during the movement window |

---

## Configuration (`configs/default.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `pose` | `model_complexity` | `1` | MediaPipe model complexity; 0 = fastest, 2 = most accurate |
| `pose` | `min_detection_confidence` | `0.5` | Minimum detection confidence |
| `pose` | `frame_stride` | `1` | Process every N-th frame (increase to speed up at cost of resolution) |
| `qc` | `landmark_visibility_threshold` | `0.5` | Min MediaPipe visibility score to treat a frame as valid |
| `qc` | `min_bbox_height_px` | `100` | Minimum subject height in pixels (smaller = subject too far away) |
| `qc` | `derivative_deg_per_sec_max` | `600` | Max angle change rate: frames exceeding this are rejected |
| `smoothing` | `method` | `median` | Smoothing method: `none`, `median`, or `savgol` |
| `robust_max` | `topk_percent` | `0.10` | Fraction of frames used for the top-K peak estimate |
| `mi4l` | `side` | `left` | Default side when `--side` is not passed via CLI |
| `export` | `save_snapshots` | `true` | Save annotated best-frame images |
| `export > plots` | `enabled` | `true` | Generate angle-over-time plots |

---

## Tests

Unit tests live under `tests/`. With dependencies installed:

```bash
pytest tests/
```

---

## How it works

1. **Landmark extraction**: MediaPipe Pose detects 33 body landmarks per frame from the RGB video.
2. **Angle computation**: Pose-specific geometry converts landmarks into joint angles (or normalised distances for the stick pass-through).
3. **Quality control**: Frames are filtered by landmark visibility, subject size, clipping, and angle derivative limits. Failing frames are excluded from estimation.
4. **AROM / PROM estimation**: The top-K median of valid frames in the detected movement window gives a robust peak angle. A short-hold fallback handles movements that don't sustain the peak for long.
5. **MI4L computation**: `MI4L = (PROM − AROM) / PROM`, clamped to [0, 1] and validated.
6. **Extended metrics**: Additional biomechanical metrics (end-range quality, motor control, compensation, reliability) are computed from the same movement window.
7. **Export**: Summary CSV, per-frame angle CSVs, angle-over-time plots, annotated snapshots, and a copy of the config are written to the output folder.
