"""Radar hardware lookup and range-cell geolocation for the whole field of view."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pydarn

from amgeo_sd.geo import geopack
from amgeo_sd.geo.fov import CalcFov

#: Altitude [km] at which range cells are located and azimuths are evaluated (legacy value).
FOV_ALTITUDE = 300.0


def hardware(rad: str, date: dt.datetime | None = None) -> SimpleNamespace:
    """Hardware parameters of radar ``rad`` valid on ``date`` (default: now).

    Wraps ``pydarn.read_hdw_file`` (pydarn >= 4, hdw files bundled with pydarn) into the flat
    object :class:`~amgeo_sd.geo.fov.CalcFov` expects. ``boresight`` is the physical boresight;
    like the legacy code, the electronic beam offset is not applied.
    """
    h = pydarn.read_hdw_file(rad, date or dt.datetime.now(dt.UTC).replace(tzinfo=None))
    return SimpleNamespace(
        rad=rad,
        beams=h.beams,
        gates=h.gates,
        beam_separation=h.beam_separation,
        rx_rise_time=h.rx_rise_time,
        geographic=h.geographic,
        boresight=h.boresight.physical,
    )


def radar_fov(
    rad: str,
    date: dt.datetime | None = None,
    frang: float = 180.0,
    rsep: float = 45.0,
    altitude: float = FOV_ALTITUDE,
    azm: bool = True,
) -> dict[str, np.ndarray]:
    """Geolocate the corners of every range cell of radar ``rad``.

    Parameters
    ----------
    date : datetime, optional
        Selects the hardware file epoch (radars are occasionally re-surveyed).
    frang, rsep : float
        First-range [km] and range separation [km] of the experiment. The legacy code always
        used 180 / 45, which is also the default here; pass the values of the data if different.
    altitude : float
        Cell altitude [km] passed to the standard virtual-height model.
    azm : bool
        Also compute the geographic azimuth of the ray between consecutive gate corners.

    Returns
    -------
    dict with ``beams`` (0..nbeams), ``gates`` (0..ngates) and arrays of shape
    ``(nbeams + 1, ngates + 1)``: ``glat``/``glon`` (cell corners, deg) and, if ``azm``,
    ``gazm`` (deg, NaN in the last row/column).
    """
    hdw = hardware(rad, date)
    fov = CalcFov(frang=frang, rsep=rsep, hdw=hdw, altitude=altitude)
    out = {
        "beams": np.arange(hdw.beams + 1),
        "gates": np.arange(hdw.gates + 1),
        "glat": fov.latFull,
        "glon": fov.lonFull,
    }
    if azm:
        gazm = np.full((hdw.beams + 1, hdw.gates + 1), np.nan)
        for b in range(hdw.beams):
            for g in range(hdw.gates):
                d = geopack.calcDistPnt(
                    fov.latFull[b, g],
                    fov.lonFull[b, g],
                    altitude,
                    distLat=fov.latFull[b, g + 1],
                    distLon=fov.lonFull[b, g + 1],
                    distAlt=altitude,
                )
                gazm[b, g] = d["az"]
        out["gazm"] = gazm
    return out
