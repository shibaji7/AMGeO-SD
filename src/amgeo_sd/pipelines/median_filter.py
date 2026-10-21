"""Median-filter model: fitacf files -> filtered scans -> ``*.pp.nc`` (legacy ``fitacf_amgeo.py``)."""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Sequence

from loguru import logger

from amgeo_sd import __version__, export
from amgeo_sd.config import Geometry, Input, Median, Output, Run, Settings, dump_settings_toml
from amgeo_sd.filters import MedianFilter
from amgeo_sd.geo import radar_fov
from amgeo_sd.io import FitacfReader, fitacfpp_name, scans_to_records, write_fitacfpp
from amgeo_sd.io.summary import estimate_scan_info


def run_median_filter(
    rad: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    files: Sequence[str | os.PathLike] | None = None,
    root: str | os.PathLike | None = None,
    ftype: str = "fitacf",
    thresh: float = 0.6,
    pth: float = 0.25,
    lth: float = 0.2,
    uth: float = 0.8,
    kde: bool = True,
    cluster: bool = True,
    eps: float = 2,
    min_samples: int = 5,
    n_jobs: int = 1,
    out_base: str | os.PathLike = "data/outputs",
    sim_id: str = "L100",
    save: bool = True,
    write_netcdf: bool = True,
    write_fitacf: bool = True,
    extra_fields: bool = False,
    input_paths: Sequence[str | os.PathLike] | None = None,
    geometry: str = "legacy",
    altitude: float = 300.0,
    edge_gates: str = "legacy",
    argv: Sequence[str] | None = None,
) -> dict:
    """Run the median-filter model for one radar and time window.

    Input is either ``files`` or the archive ``root`` (default ``$AMGEO_SD_DATA``); ``input_paths`` only
    changes what is recorded in ``params.toml`` (the directories the files were expanded from). Scans whose start
    lies in ``[start, end]`` are filtered; the neighbouring scans needed by the 3-scan window are read
    from just outside it. See :class:`amgeo_sd.filters.MedianFilter` for the algorithm. Every filter parameter has
    the meaning and allowed values given in the [parameter reference](../PARAMETERS.md) (`[median]` table).

    Parameters
    ----------
    rad : str
        Three-letter radar code (``"bks"``).
    start, end : datetime
        Time window (UTC, naive). Scans starting inside it are written.
    files : sequence of path, optional
        Explicit fitacf files (not directories; expand them with :func:`amgeo_sd.io.expand_paths`). When ``None`` the
        files are looked up under ``root``.
    root : path, optional
        Archive root (default ``$AMGEO_SD_DATA``) with the layout ``<year>/<ftype>/<rad>/<YYYYMMDD>.*``.
    ftype : str
        ``"fitacf"`` or ``"fitacf3"``; used for the archive lookup and in messages.
    thresh : float
        Share of the neighbour weight that must hold an echo for a cell to survive, 0 to 1.
    pth : float
        Beta-distribution cut of ``gflg_kde``.
    lth, uth : float
        Lower and upper probability bounds of ``gflg_conv`` / ``gflg_kde``: at or below ``lth`` ionospheric, at or
        above ``uth`` ground, between them undetermined (2).
    kde : bool
        Compute ``gflg_kde`` (slow; NaN when ``False``).
    cluster : bool
        Compute the per-scan DBSCAN label ``clabel``.
    eps, min_samples : float, int
        DBSCAN radius and minimum size of ``clabel``.
    n_jobs : int
        Worker processes of the filter (needs an ``if __name__ == "__main__":`` guard in the calling script when > 1).
    out_base : path
        Output base directory; results go to ``<out_base>/<sim_id>/<rad>/``.
    sim_id : str
        Name of the run, the second directory level.
    save : bool
        Write files. ``False`` only returns the filtered scans.
    write_netcdf : bool
        Write ``<rad>_<start>_<end>.pp.nc`` (only when ``save``).
    write_fitacf : bool
        Write ``<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pp.bz2`` (only when ``save``).
    extra_fields : bool
        Store the non-standard fields (``v_mad``, ``gflg_conv``, ``gflg_kde``, ``clabel``, ``amgeo.model``, ...) in that file.
        Off by default so that ``pydarnio.read_fitacf`` and other fitacf readers can open it; the ``.pp.nc`` always has them.
    input_paths : sequence of path, optional
        What to record as ``[input] paths`` in ``params.toml`` (e.g. the directories ``files`` came from).
    geometry : ``"legacy"`` or ``"pydarn"``
        Field-of-view implementation stored in the NetCDF file, see :func:`amgeo_sd.geo.radar_fov`.
    altitude : float
        Cell height [km] of the geolocation.
    edge_gates : ``"legacy"`` or ``"missing"``
        How out-of-range gates count as neighbours, see :class:`amgeo_sd.filters.MedianFilter`.
    argv : sequence of str, optional
        Command line recorded in ``script.cmd`` / ``params.toml``; ``sys.argv`` when ``None``.

    Returns
    -------
    dict
        ``scan_info``, ``n_scans``, ``n_echoes``, ``scans`` (the filtered :class:`~amgeo_sd.models.Scan` list) and,
        when ``save``, ``out_dir`` plus ``nc_file`` and ``fitacf_file`` (the files written).

    Raises
    ------
    RuntimeError
        When there is no data at ``start``, fewer than three scans, or (with ``save``) nothing survives the filter.

    Examples
    --------
    >>> out = run_median_filter("bks", datetime(2025, 2, 8, 3), datetime(2025, 2, 8, 4), files=files, kde=False)
    >>> out["nc_file"]
    'data/outputs/L100/bks/bks_202502080300_202502080400.pp.nc'
    """
    logger.info(f"median filter: {rad} {start} -> {end}")
    probe = FitacfReader(rad, start, start + dt.timedelta(minutes=5), ftype, files, root).beams()
    if not probe:
        raise RuntimeError(f"no {ftype} data for {rad} at {start}")
    scan_info = estimate_scan_info(probe)
    s_time = dt.timedelta(minutes=scan_info["s_time"])
    logger.info(f"scan info: {scan_info}")

    reader = FitacfReader(rad, start - 2 * s_time, end + 3 * s_time, ftype, files, root)
    scans = reader.scans(scan_info["s_mode"])
    if len(scans) < 3:
        raise RuntimeError(f"need at least 3 scans, found {len(scans)}")
    flt = MedianFilter(
        thresh=thresh,
        pbnd=(lth, uth),
        pth=pth,
        kde=kde,
        cluster=cluster,
        eps=eps,
        min_samples=min_samples,
        edge_gates=edge_gates,
    )
    filtered = [s for s in flt.filter_scans(scans, n_jobs=n_jobs) if s.beams and start <= s.stime <= end]
    n_echoes = sum(b.n_echoes for s in filtered for b in s.beams)
    logger.info(f"{len(filtered)} filtered scans, {n_echoes} echoes")

    params = dict(
        rad=rad,
        start=start,
        end=end,
        ftype=ftype,
        thresh=thresh,
        pth=pth,
        lth=lth,
        uth=uth,
        kde=kde,
        cluster=cluster,
        eps=eps,
        min_samples=min_samples,
        edge_gates=edge_gates,
        sim_id=sim_id,
        **scan_info,
    )  # NetCDF attrs
    out = {"scan_info": scan_info, "n_scans": len(filtered), "n_echoes": n_echoes, "scans": filtered}
    if save:
        if not filtered:
            raise RuntimeError("no scans survived the filter; nothing to save")
        out_dir = export.output_dir(out_base, sim_id, rad)
        out["out_dir"] = str(out_dir)
        command = "python " + " ".join(argv if argv is not None else sys.argv)
        if write_netcdf:
            first = probe[0]
            fov = radar_fov(rad, start, frang=first.frang, rsep=first.rsep, altitude=altitude, backend=geometry)
            attrs = dict(
                params,
                type="auroral",
                geometry_backend=geometry,
                geometry_altitude=altitude,
                gflg_type="us_m",
                gflg_type_desc="Generated by Median Filtering algo.",
                description=(
                    f"fitacf++: weighted 3x3x3 median filter, threshold {thresh}, edge_gates {edge_gates}; ground scatter flags "
                    f"gflg_conv (bounds {lth}/{uth}) and gflg_kde (Beta cut {pth}) are recommended, "
                    "gflg is the legacy v-w rule and labels almost everything ground scatter"
                ),
                recommended_flags="gflg_kde, gflg_conv",
            )
            ds = export.scans_to_dataset(filtered, fov, attrs)
            name = f"{rad}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.pp.nc"
            out["nc_file"] = str(export.write_netcdf(ds, out_dir / name))
            logger.info(f"wrote {out['nc_file']}")
        if write_fitacf:
            records = scans_to_records(filtered, reader.records(), command, extra_fields=extra_fields)
            out["fitacf_file"] = str(write_fitacfpp(out_dir / fitacfpp_name(rad, start), records))
            logger.info(f"wrote {out['fitacf_file']} ({len(records)} records)")
        settings = Settings(
            run=Run(rad, start, end, sim_id, str(out_base), n_jobs),
            input=Input(
                [str(f) for f in (input_paths if input_paths is not None else files or [])],
                str(root) if root else None,
                ftype,
            ),
            median=Median(thresh, pth, lth, uth, kde, cluster, eps, min_samples, edge_gates),
            geometry=Geometry(geometry, altitude),
            output=Output(write_netcdf, write_fitacf, extra_fields),
        )
        provenance = {
            "command": command,
            "software_version": __version__,
            "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        result = {
            "scan_period_min": scan_info["s_time"],
            "themis_beam": scan_info["t_beam"],
            "scan_mode": scan_info["s_mode"],
            "n_scans": len(filtered),
            "n_echoes": n_echoes,
            **{k: v for k, v in out.items() if k in ("nc_file", "fitacf_file")},
        }
        export.save_run_metadata(
            argv if argv is not None else sys.argv,
            dump_settings_toml(settings, "AMGeO-SD median-filter run", provenance, result),
            out_dir,
        )
    return out


def run_from_settings(settings: Settings, argv: Sequence[str] | None = None) -> dict:
    """:func:`run_median_filter` with the parameters of a :class:`~amgeo_sd.config.Settings`.

    Parameters
    ----------
    settings : Settings
        Parameters from :func:`amgeo_sd.config.load_settings`; uses the `[run]`, `[input]`, `[median]`, `[geometry]` and
        `[output]` tables. Directories in `[input] paths` are expanded to files.
    argv : sequence of str, optional
        Command line recorded in the output; ``sys.argv`` when ``None``.

    Returns
    -------
    dict
        As :func:`run_median_filter` (with ``save=True``).
    """
    from amgeo_sd.cli import expand_fitacf_paths

    r, i, m, o = settings.run, settings.input, settings.median, settings.output
    return run_median_filter(
        r.rad,
        r.start,
        r.end,
        files=expand_fitacf_paths(i.paths) if i.paths else None,
        root=i.root or None,
        ftype=i.ftype,
        thresh=m.thresh,
        pth=m.pth,
        lth=m.lth,
        uth=m.uth,
        kde=m.kde,
        cluster=m.cluster,
        eps=m.eps,
        min_samples=m.min_samples,
        edge_gates=m.edge_gates,
        n_jobs=r.jobs,
        out_base=r.out,
        sim_id=r.sim_id,
        save=True,
        geometry=settings.geometry.backend,
        altitude=settings.geometry.altitude,
        write_netcdf=o.netcdf,
        write_fitacf=o.fitacf,
        extra_fields=o.extra_fields,
        input_paths=i.paths,
        argv=argv,
    )
