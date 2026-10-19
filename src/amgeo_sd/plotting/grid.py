"""Turn echo tables into regular arrays for ``pcolormesh``."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def grid_time_gate(
    df: pd.DataFrame, param: str, nrang: int, gap_factor: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``param`` per (time, range gate) for one beam.

    Returns ``X`` (matplotlib date numbers, one per sounding), ``Y`` (gates ``0..nrang-1``) and
    ``Z`` of shape ``(len(X), nrang)`` with NaN where there is no echo. A NaN column is inserted wherever
    two soundings are more than ``gap_factor`` median spacings apart, so data gaps are not smeared over.
    """
    t = mdates.date2num(df["time"].values)
    ut, ti = np.unique(t, return_inverse=True)
    gates = df["slist"].values.astype(int)
    keep = (gates >= 0) & (gates < nrang)
    total, count = np.zeros((len(ut), nrang)), np.zeros((len(ut), nrang))
    np.add.at(total, (ti[keep], gates[keep]), df[param].values[keep].astype(float))
    np.add.at(count, (ti[keep], gates[keep]), np.isfinite(df[param].values[keep]))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(count > 0, total / count, np.nan)
    if len(ut) > 1:
        step = np.diff(ut)
        gaps = np.flatnonzero(step > gap_factor * np.median(step))
        if gaps.size:
            ut = np.insert(ut, gaps + 1, ut[gaps] + np.median(step))
            z = np.insert(z, gaps + 1, np.nan, axis=0)
    return ut, np.arange(nrang), z
