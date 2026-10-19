"""Scan-mode estimation (replaces ``PrintFitRec.beam_summary`` of the legacy ``tools/fit_records.py``)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from loguru import logger

from amgeo_sd.models import Beam


def estimate_scan_info(beams: Sequence[Beam]) -> dict:
    """Scan period, themis special beam and scan mode from a stretch of beams.

    Returns ``{"s_time": minutes, "t_beam": beam number or -1, "s_mode": "normal" | "themis"}``.

    The scan period is the median spacing of scan-start beams per channel (largest over channels),
    rounded as ``int((seconds + 10) / 60)``. Where the period exceeds one minute the beam sounded
    most often within the first period is taken as the themis special beam.

    Unlike the legacy code, which differenced only the last two scan starts of beams *with echoes*,
    all beams are used, so a scan-start beam without scatter cannot double the estimated period.
    """
    default = {"s_time": 1, "t_beam": -1, "s_mode": "normal"}
    if not beams:
        return default
    periods = []
    for channel in sorted({b.channel for b in beams}):
        starts = [b.time for b in beams if b.scan == 1 and b.channel == channel]
        starts = sorted(set(starts))
        if len(starts) >= 2:
            periods.append(float(np.median(np.diff(starts).astype("timedelta64[us]").astype(float) / 1e6)))
    if not periods:
        logger.warning("fewer than two scan starts found; assuming 1-min scans")
        return default
    s_time = int((max(periods) + 10) / 60.0)
    t_beam = -1
    if s_time > 1:
        first = min(b.time for b in beams)
        counts = Counter(b.bmnum for b in beams if (b.time - first).total_seconds() < s_time * 60)
        t_beam = counts.most_common(1)[0][0]
    return {"s_time": s_time, "t_beam": t_beam, "s_mode": "themis" if t_beam >= 0 else "normal"}
