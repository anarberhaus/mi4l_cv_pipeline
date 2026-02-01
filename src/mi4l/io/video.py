from pathlib import Path
import cv2

def get_video_info(video_path: str) -> dict:
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {p}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_s = frame_count / fps if fps else None
    return {
        "path": str(p),
        "fps": fps,
        "frame_count": frame_count,
        "resolution": (w, h),
        "duration_s": duration_s,
    }
