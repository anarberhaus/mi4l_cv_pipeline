from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mi4l.utils.config import save_config


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_landmarks_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    _ensure_parent(p)
    df.to_csv(p, index=False)


def save_angles_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    _ensure_parent(p)
    df.to_csv(p, index=False)


def save_summary_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    _ensure_parent(p)
    df.to_csv(p, index=False)


def save_config_used(cfg: dict[str, Any], path: str | Path) -> None:
    save_config(cfg, path)
