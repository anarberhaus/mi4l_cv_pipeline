from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count_est: int
    width: int
    height: int


def _get_capture_meta(cap: cv2.VideoCapture) -> VideoMeta:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return VideoMeta(fps=fps, frame_count_est=frame_count, width=width, height=height)


def _mp_landmark_names() -> list[str]:
    import mediapipe as mp  # local import

    # PoseLandmark is an IntEnum; iterate yields all members
    return [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]


def extract_mediapipe_pose_landmarks(video_path: str, pose_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    import mediapipe as mp  # local import

    frame_stride = int(pose_cfg.get("frame_stride", 1) or 1)
    frame_stride = max(frame_stride, 1)

    model_complexity = int(pose_cfg.get("model_complexity", 1))
    static_image_mode = bool(pose_cfg.get("static_image_mode", False))
    smooth_landmarks = bool(pose_cfg.get("smooth_landmarks", True))
    enable_segmentation = bool(pose_cfg.get("enable_segmentation", False))
    min_detection_confidence = float(pose_cfg.get("min_detection_confidence", 0.5))
    min_tracking_confidence = float(pose_cfg.get("min_tracking_confidence", 0.5))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    meta = _get_capture_meta(cap)
    names = _mp_landmark_names()

    # Pre-build columns for speed and consistent ordering
    lm_cols: list[str] = []
    for name in names:
        lm_cols.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])

    rows: list[dict[str, Any]] = []

    with mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_idx = 0
        kept_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            time_sec = float(frame_idx) / float(meta.fps)
            row: dict[str, Any] = {
                "frame_idx": int(frame_idx),
                "time_sec": time_sec,
                "pose_detected": results.pose_landmarks is not None,
                "image_w": meta.width,
                "image_h": meta.height,
            }

            if results.pose_landmarks is None:
                for c in lm_cols:
                    row[c] = np.nan
                row["bbox_xmin"] = np.nan
                row["bbox_ymin"] = np.nan
                row["bbox_xmax"] = np.nan
                row["bbox_ymax"] = np.nan
                row["bbox_h_px"] = np.nan
                row["bbox_w_px"] = np.nan
                row["bbox_area_norm"] = np.nan
                row["is_clipped"] = False
            else:
                lms = results.pose_landmarks.landmark
                xs = np.empty(len(lms), dtype=np.float32)
                ys = np.empty(len(lms), dtype=np.float32)
                vis = np.empty(len(lms), dtype=np.float32)

                for i, (name, lm) in enumerate(zip(names, lms, strict=False)):
                    row[f"{name}_x"] = float(lm.x)
                    row[f"{name}_y"] = float(lm.y)
                    row[f"{name}_z"] = float(lm.z)
                    row[f"{name}_visibility"] = float(lm.visibility)

                    xs[i] = lm.x
                    ys[i] = lm.y
                    vis[i] = lm.visibility

                # bbox from landmarks; use all points (including low visibility) to avoid empty bbox
                xmin = float(np.nanmin(xs))
                ymin = float(np.nanmin(ys))
                xmax = float(np.nanmax(xs))
                ymax = float(np.nanmax(ys))

                row["bbox_xmin"] = xmin
                row["bbox_ymin"] = ymin
                row["bbox_xmax"] = xmax
                row["bbox_ymax"] = ymax
                row["bbox_h_px"] = (ymax - ymin) * float(meta.height)
                row["bbox_w_px"] = (xmax - xmin) * float(meta.width)
                row["bbox_area_norm"] = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))

                # Conservative clipped detection: bbox near edges or coords outside [0,1]
                edge_margin = 0.02
                clipped = (
                    (xmin <= edge_margin)
                    or (ymin <= edge_margin)
                    or (xmax >= (1.0 - edge_margin))
                    or (ymax >= (1.0 - edge_margin))
                    or (xmin < 0.0)
                    or (ymin < 0.0)
                    or (xmax > 1.0)
                    or (ymax > 1.0)
                )
                row["is_clipped"] = bool(clipped)

            rows.append(row)
            kept_idx += 1
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    # Ensure landmark columns exist even if first frames had no detections
    for c in lm_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Order columns
    base_cols = [
        "frame_idx",
        "time_sec",
        "pose_detected",
        "image_w",
        "image_h",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_h_px",
        "bbox_w_px",
        "bbox_area_norm",
        "is_clipped",
    ]
    ordered_cols = base_cols + lm_cols
    df = df.reindex(columns=[c for c in ordered_cols if c in df.columns])

    meta_out = {
        "fps": meta.fps,
        "frame_count_est": meta.frame_count_est,
        "width": meta.width,
        "height": meta.height,
        "frame_stride": frame_stride,
        "processed_frames": int(len(df)),
        "video_path": video_path,
    }
    return df, meta_out
