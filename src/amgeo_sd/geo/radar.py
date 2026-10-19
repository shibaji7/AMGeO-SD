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
#: Geometry implementations selectable with ``[geometry] backend`` / ``--geometry``.
BACKENDS = ("legacy", "pydarn")


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
    backend: str = "legacy",
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
        Cell altitude [km]: ``legacy`` passes it to the standard virtual-height model, ``pydarn`` uses it as
        the fixed cell height.
    azm : bool
        Also compute the geographic azimuth of the ray between consecutive gate corners.
    backend : ``"legacy"`` or ``"pydarn"``
        ``legacy``: the original ``CalcFov`` (standard virtual-height model, ``model='IS'``), which reproduces
        the old results exactly. ``pydarn``: pydarn 4's ``gate2geographic_location`` at a fixed cell height; it
        applies the electronic beam offset, and it reads the radar's *current* hardware file whatever ``date``
        is. The two differ by 0.1-0.6 deg in latitude and up to ~3 deg in longitude at far gates.

    Returns
    -------
    dict with ``beams`` (0..nbeams), ``gates`` (0..ngates) and arrays of shape
    ``(nbeams + 1, ngates + 1)``: ``glat``/``glon`` (cell corners, deg) and, if ``azm``,
    ``gazm`` (deg, NaN in the last row/column).
    """
    if backend not in BACKENDS:
        raise ValueError(f"geometry backend must be one of {BACKENDS}, not {backend!r}")
    hdw = hardware(rad, date)
    if backend == "legacy":
        fov = CalcFov(frang=frang, rsep=rsep, hdw=hdw, altitude=altitude)
        lat, lon = fov.latFull, fov.lonFull
    else:
        lat, lon = _pydarn_corners(rad, date, hdw, frang, rsep, altitude)
    out = {"beams": np.arange(hdw.beams + 1), "gates": np.arange(hdw.gates + 1), "glat": lat, "glon": lon}
    if azm:
        gazm = np.full((hdw.beams + 1, hdw.gates + 1), np.nan)
        for b in range(hdw.beams):
            for g in range(hdw.gates):
                if backend == "legacy":
                    d = geopack.calcDistPnt(
                        lat[b, g], lon[b, g], altitude, distLat=lat[b, g + 1], distLon=lon[b, g + 1], distAlt=altitude
                    )
                    gazm[b, g] = d["az"]
                else:
                    gazm[b, g] = pydarn.calculate_azimuth(
                        lat[b, g], lon[b, g], altitude, lat[b, g + 1], lon[b, g + 1], altitude
                    )
        out["gazm"] = gazm
    return out


def _pydarn_corners(rad: str, date, hdw, frang: float, rsep: float, altitude: float):
    """Cell-corner latitude / longitude from pydarn 4 (fixed ``altitude``)."""
    from pydarn.utils.coordinates import gate2geographic_location

    stid = pydarn.RadarID(pydarn.read_hdw_file(rad, date or dt.datetime.now(dt.UTC).replace(tzinfo=None)).stid)
    lat = np.empty((hdw.beams + 1, hdw.gates + 1))
    lon = np.empty_like(lat)
    for b in range(hdw.beams + 1):
        for g in range(hdw.gates + 1):
            lat[b, g], lon[b, g] = gate2geographic_location(
                stid=stid, beam=b, range_gate=g, height=altitude, rsep=rsep, frang=frang
            )
    return lat, lon
