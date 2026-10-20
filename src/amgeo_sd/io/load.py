"""Load fitacf or fitacf++ data from any supported file into scans / one DataFrame.

Supported inputs (chosen by file name):

* ``*.fitacf``, ``*.fitacf3``, ``*.fitacf.bz2``, ``*.fitacf.gz`` - SuperDARN fitacf 2.0 / 2.5 / 3.0
* ``*.fitacf.pp.bz2`` (earlier name ``*.fitacf.pl.bz2``) - fitacf++ DMAP written by :mod:`amgeo_sd.io.writer`
* ``*.nc`` - fitacf++ NetCDF written by :mod:`amgeo_sd.export`

Directories are searched recursively. All sources give the same table (see
:func:`amgeo_sd.io.scans_to_dataframe`), so plotting code does not care where the data came from.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from amgeo_sd.io.fitacf import (
    FITACFPP_SUFFIX,
    LEGACY_FITACFPP_SUFFIXES,
    beams_to_scans,
    read_fitacf_file,
    records_to_beams,
    scans_to_dataframe,
)
from amgeo_sd.models import Beam, Scan

_RAD_IN_NAME = re.compile(r"\.([a-z]{3})\.(?:[a-z]\.)?fitacf")
#: Per-echo variables of a fitacf++ NetCDF that map onto ``Beam`` attributes (everything else 3-D is an extra).
_NC_STANDARD = {"v", "w_l", "p_l", "gflg"}


def is_netcdf(path: str | os.PathLike) -> bool:
    """True for fitacf++ NetCDF files (by name)."""
    return str(path).endswith(".nc")


def expand_paths(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Files as given; directories replaced by the fitacf / fitacf++ / NetCDF files below them (sorted)."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and (".fitacf" in f.name or f.suffix == ".nc")))
        else:
            out.append(p)
    return out


def read_fitacfpp_nc(path: str | os.PathLike) -> tuple[list[Scan], dict]:
    """Read a fitacf++ NetCDF into scans (one :class:`Scan` per ``scan_num``) and its global attributes."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    extras = [n for n, v in ds.data_vars.items() if v.dims == ("scan_num", "sound", "gates") and n not in _NC_STANDARD]
    bm, times = ds["bmnum"].values, ds["time"].values
    scans: list[Scan] = []
    for i in range(ds.sizes["scan_num"]):
        beams = []
        for j in range(ds.sizes["sound"]):
            if bm[i, j] < 0 or np.isnat(times[i, j]):
                continue
            gates = np.flatnonzero(np.isfinite(ds["v"].values[i, j]))

            def take(name, i=i, j=j, gates=gates):
                return ds[name].values[i, j, gates]

            beams.append(
                Beam(
                    time=pd.Timestamp(times[i, j]).to_pydatetime(),
                    bmnum=int(bm[i, j]),
                    tfreq=float(ds["tfreq"].values[i, j]),
                    noise_sky=float(ds["noise.sky"].values[i, j]),
                    nrang=int(ds["nrang"].values[i, j]),
                    mppul=int(ds["mppul"].values[i, j]),
                    frang=float(ds["frang"].values[i, j])
                    if "frang" in ds
                    else np.nan,  # absent in files from before 0.1.0
                    rsep=float(ds["rsep"].values[i, j]) if "rsep" in ds else np.nan,
                    intt_sc=int(ds["intt.sc"].values[i, j]),
                    intt_us=int(ds["intt.us"].values[i, j]),
                    scan=int(ds["scan"].values[i, j]),
                    slist=gates.astype(int),
                    v=take("v"),
                    v_e=np.full(len(gates), np.nan),
                    w_l=take("w_l"),
                    p_l=take("p_l"),
                    gflg=np.nan_to_num(take("gflg"), nan=-1).astype(np.int8),
                    extra={n: take(n) for n in extras},
                )
            )
        scan = Scan(beams=beams, s_mode=str(ds.attrs.get("s_mode", "normal")))
        scans.append(scan.update())
    return [s for s in scans if s.beams], dict(ds.attrs)


def load_scans(
    paths: Iterable[str | os.PathLike], start: dt.datetime | None = None, end: dt.datetime | None = None
) -> tuple[list[Scan], dict]:
    """Scans from fitacf / fitacf++ files, restricted to beams with ``start <= time <= end``.

    Returns ``(scans, meta)``; ``meta`` has ``kind`` (``"fitacf"`` or ``"fitacf++"``), ``rad`` (from the
    attributes or the file name, else ``None``), ``fit_version`` and ``files``.
    """
    files = expand_paths(paths)
    if not files:
        raise FileNotFoundError("no fitacf / fitacf++ files found")
    meta: dict = {"files": [str(f) for f in files], "rad": None, "kind": "fitacf", "fit_version": ""}
    nc = [f for f in files if is_netcdf(f)]
    if nc and len(nc) != len(files):
        raise ValueError("cannot mix NetCDF and fitacf files in one load")
    if nc:
        scans = []
        for f in nc:
            sc, attrs = read_fitacfpp_nc(f)
            scans.extend(sc)
            meta["rad"] = meta["rad"] or attrs.get("rad")
        meta["kind"] = "fitacf++"
        scans = [
            Scan(
                beams=[b for b in s.beams if (start is None or b.time >= start) and (end is None or b.time <= end)],
                s_mode=s.s_mode,
            ).update()
            for s in scans
        ]
        return [s for s in scans if s.beams], meta
    beams: list[Beam] = []
    for f in files:
        beams.extend(records_to_beams(read_fitacf_file(f), start, end))
        m = _RAD_IN_NAME.search(f.name)
        meta["rad"] = meta["rad"] or (m.group(1) if m else None)
        if f.name.endswith((FITACFPP_SUFFIX, *LEGACY_FITACFPP_SUFFIXES)):
            meta["kind"] = "fitacf++"
    beams.sort(key=lambda b: b.time)
    versions = sorted({b.fit_version for b in beams if b.fit_version})
    meta["fit_version"] = "/".join(versions)
    return beams_to_scans(beams), meta


def load_frame(
    paths: Iterable[str | os.PathLike], start: dt.datetime | None = None, end: dt.datetime | None = None
) -> tuple[pd.DataFrame, dict]:
    """:func:`load_scans` flattened to one row per echo (``scans_to_dataframe``), extras included."""
    scans, meta = load_scans(paths, start, end)
    extra = sorted(set.intersection(*[set(b.extra) for s in scans for b in s.beams if b.n_echoes] or [set()]))
    df = scans_to_dataframe(scans, extra=extra)
    meta["nrang"] = int(df["nrang"].max()) if len(df) else 0
    meta["extra"] = extra
    return df, meta
