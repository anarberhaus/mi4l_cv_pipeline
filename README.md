# MI4L CV Pipeline

Computer-vision pipeline to estimate joint angles from RGB video and (later) compute AROM, PROM, and the Mobility Index for Longevity (MI4L).

## Current status
✅ Video I/O working (reads FPS, frame count, resolution, duration)  
🚧 Next: pose landmarks (MediaPipe/OpenPose) → joint angles → QC → MI4L

## Project structure
- `src/mi4l/` : core library code
- `scripts/` : runnable scripts / entry points
- `configs/` : configuration files (thresholds, params)
- `data/` : local test videos (gitignored)
- `results/` : outputs (gitignored)

## Setup
Create a virtual environment (recommended), then install dependencies:

```bash
python -m pip install -U pip
python -m pip install opencv-python
python -m pip install -e .

#python scripts/run_mi4l.py --video "data/test.mp4"
