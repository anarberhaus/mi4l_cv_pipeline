# MI4L CV Pipeline

Computer-vision pipeline that estimates joint angles from RGB video and computes **AROM**, **PROM**, and the **Mobility Index for Longevity (MI4L)** score.

---

## Project structure

```
mi4l_cv_pipeline/
├── src/mi4l/
│   ├── angles/       # Joint angle computation (all poses)
│   ├── io/           # Landmark / CSV / config I/O
│   ├── metrics/      # AROM/PROM estimation & MI4L computation
│   ├── pose/         # MediaPipe landmark extraction
│   ├── qc/           # Quality-control rules & flags
│   ├── utils/        # Config loading, helpers
│   └── viz/          # Plots & snapshot export
├── scripts/
│   └── run_mi4l.py   # Main CLI entry point
├── configs/
│   └── default.yaml  # Default config (thresholds, params)
├── data/             # Local test videos (gitignored)
│   ├── arom/         # Recommended folder for AROM videos
│   └── prom/         # Recommended folder for PROM videos
├── results/          # Pipeline outputs (gitignored)
```

---

## Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install the package in editable mode
pip install -U pip
pip install opencv-python mediapipe pandas numpy scipy
pip install -e .
```

---

## CLI usage

All runs go through `scripts/run_mi4l.py`.

### Basic syntax

```bash
python scripts/run_mi4l.py \
  --arom  "data/arom/video_name.mp4" \
  --prom  "data/prom/video_name.mp4" \
  --out   "results/run_001/" \
  --pose  <POSE_NAME> \
  --config configs/default.yaml
```

`--prom` is **optional** — omit it to run AROM-only (no MI4L score will be computed).

### Supported poses

| `--pose` value | Assessment |
|---|---|
| `kneeling_knee_flexion` | Knee flexion |
| `prone_trunk_extension` | Trunk / lumbar extension |
| `standing_hip_abduction` | Hip abduction |
| `bilateral_leg_straddle` | Bilateral leg straddle |
| `unilateral_hip_extension` | Hip extension (one leg) |
| `shoulder_flexion` | Shoulder flexion |
| `shoulder_stick_pass_through` | Shoulder stick pass-through mobility |

### Example commands

**Knee flexion (both sides, AROM + PROM):**
```bash
python scripts/run_mi4l.py \
  --arom  "data/arom/knee_arom.mp4" \
  --prom  "data/prom/knee_prom.mp4" \
  --out   "results/knee_flexion/" \
  --pose  kneeling_knee_flexion \
  --config configs/default.yaml
```

**Shoulder flexion (AROM only, left side):**
```bash
python scripts/run_mi4l.py \
  --arom  "data/arom/shoulder_arom.mp4" \
  --out   "results/shoulder_flexion/" \
  --pose  shoulder_flexion \
  --config configs/default.yaml
```
> Set `mi4l.side: left` in `configs/default.yaml` to restrict to one side.

**Shoulder stick pass-through:**
```bash
python scripts/run_mi4l.py \
  --arom  "data/arom/stick_arom.mp4" \
  --prom  "data/prom/stick_prom.mp4" \
  --out   "results/stick_pass_through/" \
  --pose  shoulder_stick_pass_through \
  --config configs/default.yaml
```

**Hip extension (right side only):**
```bash
python scripts/run_mi4l.py \
  --arom  "data/arom/hip_ext_arom.mp4" \
  --out   "results/hip_extension/" \
  --pose  unilateral_hip_extension \
  --config configs/default.yaml
```

---

## Configuration (`configs/default.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `pose` | `model_complexity` | `1` | MediaPipe model complexity (0–2) |
| `pose` | `min_detection_confidence` | `0.5` | Minimum detection confidence |
| `pose` | `frame_stride` | `1` | Process every N-th frame |
| `qc` | `landmark_visibility_threshold` | `0.5` | Min MediaPipe visibility score |
| `qc` | `min_bbox_height_px` | `150` | Min subject height in pixels |
| `qc` | `derivative_deg_per_sec_max` | `600` | Max angle change rate (deg/s) |
| `smoothing` | `method` | `median` | `none` / `median` / `savgol` |
| `robust_max` | `topk_percent` | `0.10` | Top-K % of frames for AROM/PROM |
| `mi4l` | `side` | `left` | `both` / `left` / `right` |
| `export` | `save_snapshots` | `true` | Save best-frame snapshot images |
| `export` → `plots` | `enabled` | `true` | Generate angle-over-time plots |

---

## Outputs

All outputs are written to the folder specified by `--out`:

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
| `config_used.yaml` | Snapshot of the config used for the run |

### `summary.csv` column reference

| Column | Type | Description |
|---|---|---|
| **Metadata** | | |
| `movement_name` | string | Human-readable pose name (e.g. "Kneeling Knee Flexion") |
| `joint_name` | string | Anatomical joint being assessed (knee, hip, trunk, shoulder) |
| `angle_type` | string | Measurement geometry: `vector-reference`, `vector-vector`, or `distance` |
| `side` | string | Body side: `left`, `right`, or `both` |
| **Core ROM metrics** | | |
| `arom_deg` | float | Active Range of Motion peak (degrees), top-k mean |
| `prom_deg` | float | Passive Range of Motion peak (degrees), top-k mean |
| `mi4l` | float | Mobility Index for Longevity: `(PROM − AROM) / PROM` |
| `arom_confidence` | float | Estimation confidence for AROM (0–1) |
| `prom_confidence` | float | Estimation confidence for PROM (0–1) |
| `mi4l_valid` | bool | Whether the MI4L score passed all validation checks |
| `qc_flags` | string | Semicolon-separated quality control flags |
| **Derived metric** | | |
| `assist_gap` | float | `prom_deg − arom_deg` — passive capacity beyond active control |
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
| `frames_valid_pct` | float | Percentage of frames with valid pose detection inside the movement window |
| `avg_landmark_visibility` | float | Mean landmark visibility score during the movement window |

---

## How it works

1. **Landmark extraction** — MediaPipe Pose extracts 33 body landmarks per frame.
2. **Angle computation** — pose-specific geometry converts landmarks to joint angles (or normalised distances for stick pass-through).
3. **Quality control** — frames are filtered by landmark visibility, subject size, clipping, and angle derivative limits.
4. **AROM/PROM estimation** — the top-K median of valid frames gives a robust peak angle.
5. **MI4L computation** — `MI4L = AROM / PROM` (clamped and validated).
6. **Export** — CSVs, plots, and snapshots are written to the output folder.
