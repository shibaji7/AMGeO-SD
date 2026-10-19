"""Command line interface: ``python -m amgeo_sd <command>`` (normally driven by ``make``).

Commands are added as the legacy workflows are ported; see ``docs/PORTING.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from dateutil import parser as dateparser
from loguru import logger

from amgeo_sd.io import FitacfReader, beams_to_scans

_SUFFIXES = (".fitacf", ".fitacf3", ".bz2", ".gz")


def expand_fitacf_paths(paths: list[str]) -> list[Path]:
    """Files as given; directories replaced by the fitacf files below them (sorted)."""
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(
            sorted(f for f in p.rglob("*") if ".fitacf" in f.name and f.suffix in _SUFFIXES) if p.is_dir() else [p]
        )
    return files


def cmd_info(args: argparse.Namespace) -> int:
    """Print a one-block summary (beams, scans, echoes) for fitacf files."""
    files = expand_fitacf_paths(args.paths)
    if not files:
        print("no fitacf files found")
        return 1
    reader = FitacfReader("-", dt.datetime.min, dt.datetime.max, files=files)
    beams = reader.beams()
    scans = beams_to_scans(beams)
    echoes = sum(b.n_echoes for b in beams)
    gs = sum(int((b.gflg == 1).sum()) for b in beams)
    periods = np.diff([s.stime for s in scans]).astype("timedelta64[s]").astype(float) if len(scans) > 1 else [np.nan]
    print(f"files          : {len(files)}")
    print(f"time span      : {beams[0].time:%Y-%m-%d %H:%M:%S} .. {beams[-1].time:%Y-%m-%d %H:%M:%S} UT")
    print(f"beams          : {len(beams)} ({sum(b.n_echoes == 0 for b in beams)} without scatter)")
    print(
        f"scans          : {len(scans)} (beams/scan {min(len(s.beams) for s in scans)}-{max(len(s.beams) for s in scans)},"
        f" median period {np.median(periods):.0f} s)"
    )
    print(
        f"beam numbers   : {min(b.bmnum for b in beams)}-{max(b.bmnum for b in beams)}; range gates {max(b.nrang for b in beams)}"
    )
    print(f"tfreq [kHz]    : {min(b.tfreq for b in beams):.0f}-{max(b.tfreq for b in beams):.0f}")
    print(f"echoes         : {echoes} ({100 * gs / max(echoes, 1):.1f}% flagged ground scatter by fitacf)")
    return 0


def cmd_median(args: argparse.Namespace) -> int:
    """Median-filter model: fitacf -> fitacf++ (.nc and .fitacf.pl.bz2). Parameters: --config TOML, then flags."""
    from amgeo_sd.config import load_settings
    from amgeo_sd.pipelines import run_from_settings

    overrides = {
        "run": dict(rad=args.rad, start=args.start, end=args.end, sim_id=args.sim_id, out=args.out, jobs=args.jobs),
        "input": dict(paths=args.input, root=args.root, ftype=args.ftype),
        "median": dict(
            thresh=args.thresh,
            pth=args.pth,
            lth=args.lth,
            uth=args.uth,
            kde=args.kde,
            cluster=args.cluster,
            eps=args.eps,
            min_samples=args.min_samples,
        ),
        "output": dict(netcdf=args.netcdf, fitacf=args.fitacf),
    }
    settings = load_settings(args.config, overrides)
    if args.print_config:
        import json

        print(json.dumps(settings.to_dict(), indent=2, default=str))
        return 0
    out = run_from_settings(settings, argv=["-m", "amgeo_sd", *sys.argv[1:]])
    print(f"scan period    : {out['scan_info']['s_time']} min ({out['scan_info']['s_mode']} mode)")
    print(f"filtered scans : {out['n_scans']}  echoes: {out['n_echoes']}")
    for key in ("nc_file", "fitacf_file"):
        if key in out:
            print(f"{key:<15}: {out[key]}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Figures from fitacf or fitacf++ files (``plot rti`` / ``plot fov``). Parameters: --config TOML, then flags."""
    import matplotlib

    matplotlib.use("Agg")
    from amgeo_sd.config import load_settings
    from amgeo_sd.pipelines import run_plot

    overrides = {
        "plot": dict(
            inputs=args.input,
            compare=args.compare,
            rad=args.rad,
            beam=args.beam,
            start=args.start,
            end=args.end,
            time=args.time,
            params=args.params,
            flag=args.flag,
            coords=args.coords,
            projection=args.projection,
            max_gate=args.max_gate,
            terminator=args.terminator,
            out=args.out,
            name=args.name,
            format=args.format,
            dpi=args.dpi,
        )
    }
    settings = load_settings(args.config, overrides, require="plot")
    if args.print_config:
        import json

        print(json.dumps(settings.to_dict()["plot"], indent=2, default=str))
        return 0
    print(run_plot(args.kind, settings))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``amgeo_sd`` command line: ``info``, ``median`` and ``plot`` sub-commands."""
    parser = argparse.ArgumentParser(prog="amgeo_sd", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("info", help=cmd_info.__doc__)
    p.add_argument("paths", nargs="+", help="fitacf files or directories containing them")
    p.set_defaults(func=cmd_info)

    m = sub.add_parser("median", help=cmd_median.__doc__)
    m.add_argument("-c", "--config", help="TOML parameter file (see configs/default.toml); flags below override it")
    m.add_argument("--print-config", action="store_true", help="print the resolved parameters and exit")
    m.add_argument("-r", "--rad", help="radar code, e.g. bks")
    m.add_argument("-s", "--start", type=dateparser.parse, help="start time (UT), e.g. 2025-02-08T03:00")
    m.add_argument("-e", "--end", type=dateparser.parse, help="end time (UT)")
    m.add_argument(
        "-i", "--input", nargs="+", help="fitacf files or directories (default: archive lookup under --root)"
    )
    m.add_argument("--root", help="archive root laid out <year>/<ftype>/<rad>/ (default $AMGEO_SD_DATA)")
    m.add_argument("--ftype", help="fitacf or fitacf3 (archive lookup only)")
    m.add_argument("-th", "--thresh", type=float, help="neighbour-weight threshold (default 0.7)")
    m.add_argument("-pth", "--pth", type=float, help="Beta cut for the KDE flag (default 0.25)")
    m.add_argument("-lth", "--lth", type=float, help="lower probability bound: ionospheric (default 0.2)")
    m.add_argument("-uth", "--uth", type=float, help="upper probability bound: ground (default 0.8)")
    m.add_argument("--eps", type=float, help="DBSCAN eps in (beam, gate) units (default 2)")
    m.add_argument("-ms", "--min-samples", type=int, help="DBSCAN min samples (default 5)")
    flag = argparse.BooleanOptionalAction
    m.add_argument("--kde", action=flag, default=None, help="compute the Beta/KDE flag (default on; slow part)")
    m.add_argument("--cluster", action=flag, default=None, help="DBSCAN labelling (default on)")
    m.add_argument("--netcdf", action=flag, default=None, help="write the .fitacf++.nc file (default on)")
    m.add_argument("--fitacf", action=flag, default=None, help="write the .fitacf.pl.bz2 file (default on)")
    m.add_argument("-j", "--jobs", type=int, help="worker processes (default 1)")
    m.add_argument("-o", "--out", help="output base dir (default data/outputs)")
    m.add_argument("-sid", "--sim-id", help="run id; output goes to <out>/<sim-id>/<rad>/ (default L100)")
    m.set_defaults(func=cmd_median)

    pl = sub.add_parser("plot", help=cmd_plot.__doc__)
    pl.add_argument("kind", choices=["rti", "fov"], help="rti: range-time plot of one beam; fov: map of one scan")
    pl.add_argument("-c", "--config", help="TOML parameter file ([plot] table); flags below override it")
    pl.add_argument("--print-config", action="store_true", help="print the resolved [plot] parameters and exit")
    pl.add_argument(
        "-i", "--input", nargs="+", help="fitacf / fitacf++ files or directories (.nc, .fitacf.pl.bz2, .fitacf[.bz2])"
    )
    pl.add_argument("--compare", nargs="+", help="rti: further inputs stacked below the first")
    pl.add_argument("-r", "--rad", help="radar code (default: from the files)")
    pl.add_argument("-b", "--beam", type=int, help="rti: beam number")
    pl.add_argument("-s", "--start", type=dateparser.parse, help="only data from this time (UT)")
    pl.add_argument("-e", "--end", type=dateparser.parse, help="only data up to this time (UT)")
    pl.add_argument("-t", "--time", type=dateparser.parse, help="fov: map of the scan nearest this time (UT)")
    pl.add_argument(
        "-p", "--params", nargs="+", help="v w_l p_l v_mad, flags gflg gflg_conv gflg_kde gflg_ribiero, cluster_tag"
    )
    pl.add_argument("--flag", help="rti compare: flag column for the classification panels ('' = none)")
    pl.add_argument("--coords", choices=["geo", "aacgmv2"], help="fov: coordinate frame")
    pl.add_argument("--projection", choices=["auto", "polar", "ortho"], help="fov: map projection")
    pl.add_argument("--max-gate", type=int, help="fov: gates drawn (0 = all)")
    pl.add_argument(
        "--terminator", action=argparse.BooleanOptionalAction, default=None, help="fov: shade the night side"
    )
    pl.add_argument("-o", "--out", help="output directory (default data/figures)")
    pl.add_argument("--name", help="file name without extension (default: generated)")
    pl.add_argument("--format", choices=["png", "pdf", "svg"], help="figure format (default png)")
    pl.add_argument("--dpi", type=int, help="resolution of raster output (default 150)")
    pl.set_defaults(func=cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    logger.remove()
    logger.enable("amgeo_sd")
    logger.add(sys.stderr, level="INFO")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
