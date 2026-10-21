"""Write fitacf++ products: an xarray/NetCDF4 dataset plus the run's ``params.toml`` and ``script.cmd``.

Layout follows the legacy ``utils.to_xarray`` so existing readers keep working::

    dims   scan_num, sound (beam slot within a scan), beams (nbeams+1), gates (ngates+1)
    geometry   gdlat, gdlon, gdazm   (beams, gates)   range-cell corner geolocation
    per beam   bmnum, noise.sky, tfreq, scan, nrang, frang, rsep, time, mppul, intt.us, intt.sc   (scan_num, sound)
    per echo   v, w_l, p_l, gflg + algorithm outputs (gflg_conv, gflg_kde, v_mad, clabel)
               (scan_num, sound, gates), NaN where there is no echo
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from amgeo_sd import __version__
from amgeo_sd.models import Scan

__all__ = ["output_dir", "save_run_metadata", "scans_to_dataset", "write_netcdf"]

#: (variable, long_name, units, description) of the per-echo fields stored on ``Beam`` itself.
ECHO_FIELDS = (
    ("v", "Velocity", "m/s", "LOS Doppler velocity"),
    ("w_l", "Width", "m/s", "LOS spectral width"),
    ("p_l", "Power", "dB", "Backscatter power"),
    ("gflg", "Ground scatter flag", "bool", "Ground scatter flag (0 ionospheric, 1 ground)"),
)
#: Per-echo fields kept in ``Beam.extra``.
EXTRA_FIELDS = (
    (
        "gflg_conv",
        "Ground scatter flag",
        "",
        "Ground scatter flag from the weighted-neighbour fraction (0 IS, 1 GS, 2 undetermined)",
    ),
    (
        "gflg_kde",
        "Ground scatter flag",
        "",
        "Ground scatter flag from the Beta/KDE estimate (0 IS, 1 GS, 2 undetermined)",
    ),
    (
        "v_mad",
        "Velocity median absolute deviation (MAD)",
        "m/s",
        "Median absolute deviation after applying the median filter",
    ),
    ("clabel", "Cluster label", "", "DBSCAN cluster label in (beam, gate) space, -1 is noise"),
    (
        "cluster_tag",
        "Cluster tag",
        "",
        "DBSCAN cluster label of the echo within its beam (-1 is noise); unique only per beam",
    ),
    (
        "gflg_ribiero",
        "Ground scatter flag (Ribiero)",
        "",
        "Cluster ground scatter flag: -1 unclassified, 0 ionospheric, 1 ground",
    ),
)
#: Per-beam scalars: (variable, Beam attribute, dtype, long_name, units, description).
BEAM_FIELDS = (
    ("bmnum", "bmnum", "i4", "Sounding beam", "1", "Sounding beam number"),
    ("noise.sky", "noise_sky", "f8", "Sky noise", "1", "Background sky noise"),
    ("tfreq", "tfreq", "f8", "Frequency", "kHz", "Transmit signal frequency"),
    ("scan", "scan", "i4", "Scan flag", "1", "Scan flag indicates start of a scan"),
    ("nrang", "nrang", "i4", "Range number", "1", "Number of range gates in each beam sounding"),
    ("frang", "frang", "f8", "First range", "km", "Distance to the first range gate"),
    ("rsep", "rsep", "f8", "Range separation", "km", "Range gate separation"),
    ("mppul", "mppul", "i4", "Pulse counts", "1", "Number of pulses"),
    ("intt.us", "intt_us", "i4", "Int. Time micro-sec", "micro-seconds", "Integration time in micro-seconds"),
    ("intt.sc", "intt_sc", "i4", "Int. Time sec", "seconds", "Integration time in seconds"),
)


def output_dir(base: str | os.PathLike, sim_id: str, rad: str) -> Path:
    """``<base>/<sim_id>/<rad>/`` (created), the legacy output layout.

    Parameters
    ----------
    base : str or path
        Base output directory (``[run] out``).
    sim_id : str
        Run name (``[run] sim_id``).
    rad : str
        Radar code.

    Returns
    -------
    Path
        The directory, created if needed.
    """
    path = Path(base) / sim_id / rad
    path.mkdir(parents=True, exist_ok=True)
    return path


def scans_to_dataset(scans: Sequence[Scan], fov: dict, attrs: dict | None = None) -> xr.Dataset:
    """Pack filtered scans and range-cell geometry into the fitacf++ dataset.

    Parameters
    ----------
    scans : sequence of Scan
        Output of a filter; ``Beam.extra`` entries listed in :data:`EXTRA_FIELDS` are written when present.
    fov : dict
        Result of :func:`amgeo_sd.geo.radar_fov` (needs ``glat``, ``glon``, ``beams``, ``gates``; ``gazm`` optional).
    attrs : dict, optional
        Global attributes; values are made NetCDF safe (bool -> int, datetime -> ISO string, None -> "").

    Returns
    -------
    xarray.Dataset
        Dimensions ``scan_num``, ``sound`` (beam slot within a scan), ``beams`` (nbeams + 1) and ``gates`` (ngates + 1).
        ``gdlat``, ``gdlon``, ``gdazm`` (beams, gates) hold the range-cell corner geolocation; per-beam variables
        (``bmnum``, ``tfreq``, ``noise.sky``, ``scan``, ``nrang``, ``frang``, ``rsep``, ``time``, ``mppul``, ``intt.us``,
        ``intt.sc``) have shape (scan_num, sound); per-echo variables (``v``, ``w_l``, ``p_l``, ``gflg`` and the extras
        ``gflg_conv``, ``gflg_kde``, ``v_mad``, ``clabel``, ``cluster_tag``, ``gflg_ribiero`` that the beams carry) have
        shape (scan_num, sound, gates), NaN where there is no echo.

    Raises
    ------
    ValueError
        When ``scans`` is empty or an echo lies outside the radar's range gates.
    """
    if not scans:
        raise ValueError("no scans to export")
    n_scan, n_slot, n_gate = len(scans), max(len(s.beams) for s in scans), len(fov["gates"])
    for s in scans:
        for b in s.beams:
            if b.n_echoes and b.slist.max() >= n_gate:
                raise ValueError(f"range gate {b.slist.max()} outside the radar's {n_gate - 1} gates")

    def per_beam(attr: str, dtype: str) -> np.ndarray:
        fill = -1 if dtype.startswith("i") else np.nan
        out = np.full((n_scan, n_slot), fill, dtype=dtype)
        for i, s in enumerate(scans):
            for j, b in enumerate(s.beams):
                out[i, j] = getattr(b, attr)
        return out

    def per_echo(get) -> np.ndarray:
        out = np.full((n_scan, n_slot, n_gate), np.nan)
        for i, s in enumerate(scans):
            for j, b in enumerate(s.beams):
                if b.n_echoes:
                    out[i, j, b.slist] = get(b)
        return out

    times = np.full((n_scan, n_slot), np.datetime64("NaT", "us"))
    for i, s in enumerate(scans):
        for j, b in enumerate(s.beams):
            times[i, j] = np.datetime64(b.time, "us")

    sc = ("scan_num", "sound")
    data = {}
    for name, geo, long_name in (
        ("gdlat", "glat", "Latitude"),
        ("gdlon", "glon", "Longitude"),
        ("gdazm", "gazm", "Ray Azimuth"),
    ):
        if geo in fov:
            data[name] = (
                ("beams", "gates"),
                fov[geo],
                dict(
                    long_name=long_name,
                    units="degree",
                    description=f"Geographic {long_name.lower()} at each range cell corner",
                ),
            )
    for name, attr, dtype, long_name, units, desc in BEAM_FIELDS:
        data[name] = (sc, per_beam(attr, dtype), dict(long_name=long_name, units=units, description=desc))
    data["time"] = (sc, times, dict(long_name="Date time", description="Beam sounding datetime"))
    gate_dims = ("scan_num", "sound", "gates")
    for name, long_name, units, desc in ECHO_FIELDS:
        data[name] = (
            gate_dims,
            per_echo(lambda b, n=name: getattr(b, n)),
            dict(long_name=long_name, units=units, description=desc),
        )
    for name, long_name, units, desc in EXTRA_FIELDS:
        if any(name in b.extra for s in scans for b in s.beams):
            data[name] = (
                gate_dims,
                per_echo(lambda b, n=name: b.extra[n] if n in b.extra else np.full(b.n_echoes, np.nan)),
                dict(long_name=long_name, units=units, description=desc),
            )

    coords = {
        "beams": ("beams", fov["beams"], dict(long_name="Beam number", description="Radar beams")),
        "gates": ("gates", fov["gates"], dict(long_name="Gate number", description="Range gate")),
        "sound": (
            "sound",
            np.arange(n_slot),
            dict(long_name="Beam sound seq", description="Radar beam sound sequence"),
        ),
        "scan_num": ("scan_num", np.arange(n_scan), dict(long_name="Scan number", description="Number of scan")),
        "scan_stime": (
            "scan_num",
            np.array([np.datetime64(s.stime, "us") for s in scans]),
            dict(long_name="Scan start time"),
        ),
        "scan_etime": (
            "scan_num",
            np.array([np.datetime64(s.etime, "us") for s in scans]),
            dict(long_name="Scan end time"),
        ),
    }
    global_attrs = {
        "source": "AMGeO - SD data processing (amgeo_sd)",
        "software_version": __version__,
        "history": f"Created {dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}",
    }
    global_attrs.update(_netcdf_safe(attrs or {}))
    return xr.Dataset(data_vars=data, coords=coords, attrs=global_attrs)


def _netcdf_safe(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, (list, tuple)):
            v = "-".join(map(str, v))
        elif isinstance(v, (dt.date, dt.datetime)):
            v = v.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(v, os.PathLike):
            v = str(v)
        elif v is None:
            v = ""
        out[k] = v
    return out


def write_netcdf(ds: xr.Dataset, path: str | os.PathLike, complevel: int = 4) -> Path:
    """Write ``ds`` as NetCDF4 with zlib compression on every numeric variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset from :func:`scans_to_dataset`.
    path : str or path
        Output file (``<rad>_<start>_<end>.pp.nc``); parent directories are created.
    complevel : int
        zlib compression level, 1 (fast) to 9 (small).

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {k: {"zlib": True, "complevel": complevel} for k, v in ds.data_vars.items() if v.dtype.kind in "fiu"}
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return path


def save_run_metadata(argv: Sequence[str], params_toml: str, out_dir: str | os.PathLike) -> None:
    """Write ``script.cmd`` (the command line) and ``params.toml`` (annotated parameters) into ``out_dir``.

    Parameters
    ----------
    argv : sequence of str
        The command line of the run (``sys.argv``).
    params_toml : str
        Text of ``params.toml``, normally from :func:`amgeo_sd.config.dump_settings_toml`.
    out_dir : str or path
        Existing output directory.
    """
    out_dir = Path(out_dir)
    (out_dir / "script.cmd").write_text("python " + " ".join(argv) + "\n")
    (out_dir / "params.toml").write_text(params_toml)
