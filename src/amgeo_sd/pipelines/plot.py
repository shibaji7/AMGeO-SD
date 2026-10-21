"""RTI and field-of-view figures from fitacf or fitacf++ files, driven by the ``[plot]`` settings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from amgeo_sd.config import Settings
from amgeo_sd.io import load_frame
from amgeo_sd.plotting import plot_fan, plot_rti, plot_rti_compare


def _label(meta: dict) -> str:
    return " ".join(x for x in (meta["kind"], meta["fit_version"]) if x)


def run_plot(kind: str, settings: Settings) -> str:
    """Draw one figure and return its path.

    Parameters
    ----------
    kind : ``"rti"`` or ``"fov"``
        RTI of ``plot.beam`` (``plot.params`` panels; with ``plot.compare`` the same beam from several files stacked),
        or a map of the scan nearest ``plot.time`` (one map per ``plot.params`` entry).
    settings : Settings
        Uses the ``[plot]`` table (inputs, beam, params, time, coords, ...) and ``[geometry]``; every key is in the
        [parameter reference](../PARAMETERS.md).

    Returns
    -------
    str
        Path of the figure written, ``<plot.out>/<name>.<plot.format>``.

    Raises
    ------
    ValueError
        For an unknown ``kind``, or when the radar code cannot be determined.
    RuntimeError
        When the inputs hold no echoes in the time range.
    """
    if kind not in ("rti", "fov"):
        raise ValueError(f"unknown plot kind {kind!r}; use 'rti' or 'fov'")
    p = settings.plot
    df, meta = load_frame(p.inputs, p.start, p.end)
    if df.empty:
        raise RuntimeError("no echoes in the selected inputs / time range")
    rad = p.rad or meta["rad"]
    if not rad:
        raise ValueError("radar code unknown: set [plot] rad (the file names carry none)")
    label = _label(meta)
    t0 = pd.Timestamp(df["time"].min()).to_pydatetime()
    out_dir = Path(p.out)
    if kind == "rti":
        stem = (
            p.name
            or f"rti_{rad}_b{p.beam:02d}_{t0:%Y%m%d_%H%M}_{'-'.join(p.params if not p.compare else p.params[:1])}"
        )
        path = out_dir / f"{stem}.{p.format}"
        if p.compare:
            frames = {label or "data": df}
            for src in p.compare:
                other, m = load_frame([src], p.start, p.end)
                name = _label(m) or Path(src).name
                frames[name if name not in frames else f"{name} ({Path(src).name})"] = other
            plot_rti_compare(frames, p.beam, p.params[0], path, rad=rad, flag=p.flag or None, dpi=p.dpi)
        else:
            plot_rti(df, p.beam, p.params, path, rad=rad, kind=label, dpi=p.dpi)
    else:
        when = p.time or t0
        stem = p.name or f"fov_{rad}_{when:%Y%m%d_%H%M}_{'-'.join(p.params)}_{p.coords}"
        path = out_dir / f"{stem}.{p.format}"
        plot_fan(
            df,
            rad,
            when,
            p.params,
            path,
            coords=p.coords,
            projection=p.projection,
            kind=label,
            dpi=p.dpi,
            max_gate=p.max_gate or None,
            terminator=p.terminator,
            geometry=settings.geometry.backend,
            altitude=settings.geometry.altitude,
        )
    import matplotlib.pyplot as plt

    plt.close("all")
    logger.info(f"wrote {path}")
    return str(path)
