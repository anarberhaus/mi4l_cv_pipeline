from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
    p = Path(path)
    _ensure_parent(p)
    import yaml
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

