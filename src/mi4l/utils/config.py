from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "pose": {
        "backend": "mediapipe",
        "model_complexity": 1,
        "static_image_mode": False,
        "smooth_landmarks": True,
        "enable_segmentation": False,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "frame_stride": 1,
    },
    "angles": {"joints": ["knee_flexion"]},
    "qc": {
        "landmark_visibility_threshold": 0.5,
        "min_valid_ratio": 0.3,
        "min_valid_frames": 15,
        "min_bbox_height_px": 200,
        "require_subject_size": True,
        "edge_margin": 0.02,
        "exclude_clipped_frames": False,
        "max_clipped_ratio": 0.35,
        "derivative_deg_per_sec_max": 600.0,
    },
    "smoothing": {
        "method": "median",  # none | median | savgol
        "interpolate_limit": 5,
        "median_window": 7,
        "savgol_window": 11,
        "savgol_polyorder": 3,
    },
    "robust_max": {
        "method": "topk_median",
        "topk_percent": 0.05,
        "min_topk_frames": 5,
    },
    "mi4l": {
        "prom_min_deg": 1.0,
        "prom_lt_arom_tolerance_deg": 2.0,
        "invalidate_if_prom_lt_arom": True,
    },
    "export": {
        "save_landmarks_csv": True,
        "save_angles_csv": True,
        "save_summary_csv": True,
        "save_config_used": True,
        "plots": {"enabled": True, "dpi": 150},
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    stack: list[tuple[dict[str, Any], dict[str, Any]]] = [(out, override)]
    while stack:
        dst, src = stack.pop()
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                stack.append((dst[k], v))  # type: ignore[index]
            else:
                dst[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data: dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
    return _deep_merge(DEFAULT_CONFIG, data)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)