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
    """Median-filter model: fitacf -> fitacf++ (.nc and .fitacf.pp.bz2). Parameters: --config TOML, then flags."""
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
            edge_gates=args.edge_gates,
        ),
        "geometry": dict(backend=args.geometry, altitude=args.altitude),
        "output": dict(netcdf=args.netcdf, fitacf=args.fitacf, extra_fields=args.extra_fields),
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


def cmd_dbscan(args: argparse.Namespace) -> int:
    """DBSCAN / GMM model: label raw echoes with clusters and the Ribiero flag -> fitacf++. Parameters: --config TOML, then flags."""
    from amgeo_sd.config import load_settings
    from amgeo_sd.pipelines import run_dbscan_from_settings

    overrides = {
        "run": dict(rad=args.rad, start=args.start, end=args.end, sim_id=args.sim_id, out=args.out),
        "input": dict(paths=args.input, root=args.root, ftype=args.ftype),
        "dbscan": dict(
            eps=args.eps,
            eps_g=args.eps_g,
            eps_s=args.eps_s,
            min_samples=args.min_samples,
            tb_mult=args.tb_mult,
            velocity_threshold=args.velocity_threshold,
            min_gate=args.min_gate,
            gmm=args.gmm,
            n_clusters=args.n_clusters,
            cov=args.cov,
            n_data_th=args.n_data_th,
        ),
        "geometry": dict(backend=args.geometry, altitude=args.altitude),
        "output": dict(netcdf=args.netcdf, fitacf=args.fitacf, extra_fields=args.extra_fields),
    }
    settings = load_settings(args.config, overrides, require="dbscan")
    if args.print_config:
        import json

        print(json.dumps(settings.to_dict(), indent=2, default=str))
        return 0
    out = run_dbscan_from_settings(settings, argv=["-m", "amgeo_sd", *sys.argv[1:]])
    print(f"scan period    : {out['scan_info']['s_time']} min ({out['scan_info']['s_mode']} mode)")
    print(f"scans / echoes : {out['n_scans']} / {out['n_echoes']}")
    print(
        f"clusters       : {out['n_clusters']}  ({100 * out['fraction_noise']:.1f}% noise, {100 * out['fraction_ground']:.1f}% ground scatter)"
    )
    for key in ("nc_file", "fitacf_file"):
        if key in out:
            print(f"{key:<15}: {out[key]}")
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    """Print the parameter reference (docs/PARAMETERS.md) generated from the settings classes."""
    from amgeo_sd.params_doc import parameters_markdown

    sys.stdout.write(parameters_markdown())
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    """Write the Python API reference (docs/API.md and docs/api/*.md) generated from the docstrings."""
    from amgeo_sd import api_doc

    gaps = api_doc.problems()
    if gaps:
        print("\n".join(gaps), file=sys.stderr)
        return 1
    for path in api_doc.write(args.out):
        print(path)
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """Write an annotated TOML parameter file with every setting at its default (to edit, then pass with --config)."""
    from amgeo_sd.config import Settings, dump_settings_toml

    text = dump_settings_toml(
        Settings(), "AMGeO-SD parameters (defaults); see docs/PARAMETERS.md", tables=tuple(Settings().to_dict())
    )
    if args.out == "-":
        sys.stdout.write(text)
        return 0
    path = Path(args.out)
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to overwrite", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(path)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Figures from fitacf or fitacf++ files (``plot rti`` / ``plot fov``). Parameters: --config TOML, then flags."""
    import matplotlib

    matplotlib.use("Agg")
    from amgeo_sd.config import load_settings
    from amgeo_sd.pipelines import run_plot

    overrides = {
        "geometry": dict(backend=args.geometry, altitude=args.altitude),
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
        ),
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
    m.add_argument(
        "-th", "--thresh", type=float, help="neighbour-weight threshold (default 0.6; the legacy code used 0.7)"
    )
    m.add_argument(
        "--edge-gates",
        choices=["legacy", "missing"],
        help="gates outside the radar range count as empty (legacy, default) or are not counted (missing)",
    )
    m.add_argument("-pth", "--pth", type=float, help="Beta cut for the KDE flag (default 0.25)")
    m.add_argument("-lth", "--lth", type=float, help="lower probability bound: ionospheric (default 0.2)")
    m.add_argument("-uth", "--uth", type=float, help="upper probability bound: ground (default 0.8)")
    m.add_argument("--eps", type=float, help="DBSCAN eps in (beam, gate) units (default 2)")
    m.add_argument("-ms", "--min-samples", type=int, help="DBSCAN min samples (default 5)")
    flag = argparse.BooleanOptionalAction
    m.add_argument("--kde", action=flag, default=None, help="compute the Beta/KDE flag (default on; slow part)")
    m.add_argument("--cluster", action=flag, default=None, help="DBSCAN labelling (default on)")
    m.add_argument("--netcdf", action=flag, default=None, help="write the .pp.nc file (default on)")
    m.add_argument("--fitacf", action=flag, default=None, help="write the .fitacf.pp.bz2 file (default on)")
    m.add_argument(
        "--extra-fields",
        action=flag,
        default=None,
        help="store non-standard fields (v_mad, gflg_kde, ...) in the .fitacf.pp.bz2; "
        "off (default) keeps it readable by pydarnio.read_fitacf",
    )
    m.add_argument(
        "--geometry",
        choices=["legacy", "pydarn"],
        help="range-cell geolocation: original CalcFov or pydarn 4 (default legacy)",
    )
    m.add_argument("--altitude", type=float, help="cell altitude in km (default 300)")
    m.add_argument("-j", "--jobs", type=int, help="worker processes (default 1)")
    m.add_argument("-o", "--out", help="output base dir (default data/outputs)")
    m.add_argument("-sid", "--sim-id", help="run id; output goes to <out>/<sim-id>/<rad>/ (default L100)")
    m.set_defaults(func=cmd_median)

    d = sub.add_parser("dbscan", help=cmd_dbscan.__doc__)
    d.add_argument("-c", "--config", help="TOML parameter file (see configs/default.toml); flags below override it")
    d.add_argument("--print-config", action="store_true", help="print the resolved parameters and exit")
    d.add_argument("-r", "--rad", help="radar code, e.g. bks")
    d.add_argument("-s", "--start", type=dateparser.parse, help="start time (UT), e.g. 2025-02-08T03:00")
    d.add_argument("-e", "--end", type=dateparser.parse, help="end time (UT)")
    d.add_argument(
        "-i", "--input", nargs="+", help="fitacf files or directories (default: archive lookup under --root)"
    )
    d.add_argument("--root", help="archive root laid out <year>/<ftype>/<rad>/ (default $AMGEO_SD_DATA)")
    d.add_argument("--ftype", help="fitacf or fitacf3 (archive lookup only)")
    d.add_argument("-eps", "--eps", type=float, help="DBSCAN radius (default 2; the legacy code used 1)")
    d.add_argument("-eps_g", "--eps-g", type=float, help="range-gate scale (default 1)")
    d.add_argument("-eps_s", "--eps-s", type=float, help="scan-number scale (default 1)")
    d.add_argument("-ms", "--min-samples", type=int, help="DBSCAN min samples (default 8; the legacy code used 5)")
    d.add_argument("-tm", "--tb-mult", type=int, help="min-samples multiplier on the themis beam (default 10)")
    d.add_argument("-vth", "--velocity-threshold", type=float, help="Ribiero velocity threshold in m/s (default 25)")
    d.add_argument("--min-gate", type=int, help="gates below this stay unclassified (default 7)")
    d.add_argument(
        "-gmm", "--gmm", help="Gaussian-mixture step: A all beams, T themis beam, or beam list like 3,7 (default none)"
    )
    d.add_argument("-nc", "--n-clusters", type=int, help="mixture components (default 5)")
    d.add_argument(
        "-cov", "--cov", choices=["full", "tied", "diag", "spherical"], help="mixture covariance type (default full)"
    )
    d.add_argument("-gth", "--n-data-th", type=int, help="beams with fewer echoes skip the mixture (default 100)")
    d.add_argument("--geometry", choices=["legacy", "pydarn"], help="range-cell geolocation (default legacy)")
    d.add_argument("--altitude", type=float, help="cell altitude in km (default 300)")
    d.add_argument(
        "--netcdf", action=argparse.BooleanOptionalAction, default=None, help="write the .pp.nc file (default on)"
    )
    d.add_argument(
        "--fitacf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write the .fitacf.pp.bz2 file (default on)",
    )
    d.add_argument(
        "--extra-fields",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="store non-standard fields (cluster_tag, gflg_ribiero) in the .fitacf.pp.bz2; "
        "off (default) keeps it readable by pydarnio.read_fitacf",
    )
    d.add_argument("-o", "--out", help="output base dir (default data/outputs)")
    d.add_argument("-sid", "--sim-id", help="run id; output goes to <out>/<sim-id>/<rad>/ (default L100)")
    d.set_defaults(func=cmd_dbscan)

    sub.add_parser("params", help=cmd_params.__doc__).set_defaults(func=cmd_params)

    ic = sub.add_parser("init-config", help=cmd_init_config.__doc__)
    ic.add_argument(
        "-o", "--out", default="amgeo_sd.toml", help="file to write, or - for stdout (default amgeo_sd.toml)"
    )
    ic.add_argument("--force", action="store_true", help="overwrite an existing file")
    ic.set_defaults(func=cmd_init_config)

    api = sub.add_parser("api", help=cmd_api.__doc__)
    api.add_argument(
        "-o", "--out", default="docs", help="the docs directory to write API.md and api/ into (default docs)"
    )
    api.set_defaults(func=cmd_api)

    pl = sub.add_parser("plot", help=cmd_plot.__doc__)
    pl.add_argument("kind", choices=["rti", "fov"], help="rti: range-time plot of one beam; fov: map of one scan")
    pl.add_argument("-c", "--config", help="TOML parameter file ([plot] table); flags below override it")
    pl.add_argument("--print-config", action="store_true", help="print the resolved [plot] parameters and exit")
    pl.add_argument(
        "-i", "--input", nargs="+", help="fitacf / fitacf++ files or directories (.nc, .fitacf.pp.bz2, .fitacf[.bz2])"
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
    pl.add_argument("--geometry", choices=["legacy", "pydarn"], help="fov: range-cell geolocation (default legacy)")
    pl.add_argument("--altitude", type=float, help="fov: cell altitude in km (default 300)")
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
