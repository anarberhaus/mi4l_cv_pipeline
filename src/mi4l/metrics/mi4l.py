from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mi4LResult:
    value: float | None
    valid: bool
    flags: list[str]


def compute_mi4l(
    arom_deg: float | None,
    prom_deg: float | None,
    prom_min_deg: float,
    prom_lt_arom_tolerance_deg: float,
    invalidate_if_prom_lt_arom: bool,
) -> Mi4LResult:
    flags: list[str] = []

    if arom_deg is None or prom_deg is None:
        return Mi4LResult(value=None, valid=False, flags=["missing_arom_or_prom"])

    if not np.isfinite(arom_deg) or not np.isfinite(prom_deg):
        return Mi4LResult(value=None, valid=False, flags=["nonfinite_arom_or_prom"])

    if prom_deg <= float(prom_min_deg):
        return Mi4LResult(value=None, valid=False, flags=[f"prom_too_small:{prom_deg:.3f}<={prom_min_deg:.3f}"])

    if prom_deg + float(prom_lt_arom_tolerance_deg) < arom_deg:
        flags.append(f"prom_lt_arom:{prom_deg:.3f}+{prom_lt_arom_tolerance_deg:.3f}<{arom_deg:.3f}")
        if invalidate_if_prom_lt_arom:
            return Mi4LResult(value=None, valid=False, flags=flags)

    mi4l = (prom_deg - arom_deg) / prom_deg
    if not np.isfinite(mi4l):
        return Mi4LResult(value=None, valid=False, flags=flags + ["mi4l_nonfinite"])

    return Mi4LResult(value=float(mi4l), valid=True, flags=flags)
