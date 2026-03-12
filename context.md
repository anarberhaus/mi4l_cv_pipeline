# Project Context: MWI Computer Vision Pipeline

This document provides a high-level overview of the MI4L CV Pipeline's architecture, directory structure, and execution flow to serve as a comprehensive knowledge base for developers working on the repository.

## 1. System Overview

The **MWI CV Pipeline** is a specialized computer vision application built to estimate human joint angles directly from RGB video. Its primary purpose is to compute **AROM** (Active Range of Motion) and **PROM** (Passive Range of Motion), from which it derives the **MWI score**: `(PROM - AROM) / PROM`.

The application logic provides two main modes of interaction:
1. **Command-Line Interface (CLI)**: For batch processing, automation, and headless execution via the Python scripts in `scripts/`.
2. **Graphical User Interface (GUI)**: A modern, reactive web application built with Streamlit (`app.py`), which functions as a professional front-end wrapping the underlying pipeline processing logic.

---

## 2. Directory Architecture

The repository separates the underlying backend logic from the execution scripts, configuration, and UI.

### `/src/mi4l/` (Backend Library)
This represents the core engine of the system.
- `pose/`: Wrapping MediaPipe's pose estimation model to extract the raw 33 3D landmarks for each frame.
- `angles/`: Geometric math modules for converting 3D landmarks into 2D/3D joint angles based on specific anatomical definitions (e.g. tracking vector-vector angles or vertical reference deviations).
- `qc/`: Quality Control system that filters bad frames based on bounding box size, visibility confidence, clipping, or unnatural angular velocity (jerk).
- `metrics/`: Core business logic that converts time-series angles into final metrics (AROM/PROM peak computation handling top-K median smoothing) and derives structural evaluations (MWI calculation). 
- `viz/`: Snapshot image generation (drawing lines/wedges onto detected poses) and time-series matplotlib plotting. 
- `io/`: Data reading/writing utilities for video metadata and metric exports (CSVs).

### Execution & UIs
- `app.py`: The root Streamlit application. Provides a rich visual interface, including **Supabase authentication** for user sessions, cloud persistence for analysis history, global dark-mode styling, and a complete multi-pose assessment flow. It utilizes `subprocess` to trigger the underlying pipeline engine safely.
- `scripts/run_mi4l.py`: The single-pose CLI entrypoint. Takes raw arguments for AROM/PROM paths, pose configuration, and side, saving artifacts to the `results/` folder.
- `scripts/run_all.py` & `scripts/run_all_poses.bat`: Automation wrappers that sequentially execute `.mp4` tests against all predefined pose settings.
- `scripts/debug_landmarks.py`: Diagnostic utility for verifying precise MediaPipe outputs on specific frames.

### Configuration, Data & Deployment
- `configs/default.yaml`: Global parameter settings for the pipeline (smoothing models, frame-stride, minimal confidence bounds, etc).
- `Dockerfile` & `packages.txt`: Deployment configurations defining OS-level libraries (`libgl1`, `libglib2.0-0`) required for MediaPipe and OpenCV on Linux servers (e.g., Render, Streamlit Cloud).
- `data/` & `results/`: Local untracked storage for raw inputs and pipeline artifacts.

---

## 3. Data Flow

1. **Input Generation**: User passes an AROM video (and optionally a PROM video) either via CLI `--arom` arguments or by uploading in the Streamlit `app.py`. The UI supports both single-pose analysis and batch-upload generation.
2. **Pose Extraction**: Frame-by-frame, MediaPipe extracts landmarks.
3. **Filtering**: The `qc/` engine flags frames where the person isn't visible, is cut off by the edge, or is moving impossibly fast (bad tracker jumps).
4. **Angle Geometry**: Valid frames yield angle integers based on the chosen body pose logic (flexion, abduction, etc.).
5. **Peak Calculation**: Time-series smoothing resolves the highest reliable hold-angle across the set for both AROM (active flex) and PROM (assisted flex).
6. **Output Export**: The `metrics/` logic derives secondary insights (jerk RMS, torso compensation drift), scores the MWI capacity gap, and `viz/` exports annotated freeze-frames (`snapshots/`), time-series line graphs, and CSVs.

## 4. Maintenance Notes
- **Git Tracking**: We explicitly avoid tracking `.mp4` video files, `__pycache__`, and the generated `results/` folder to prevent severe repository bloat.
- **Environment**: Due to strict dependency conflicts on certain Operating Systems with `mediapipe`, execution relies exclusively on a clean Conda / Venv shell targeting Python 3.10 or 3.11.
