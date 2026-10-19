"""Command line interface: ``python -m amgeo_sd <command>`` (normally driven by ``make``).

Commands are added as the legacy workflows are ported; see ``docs/PORTING.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from amgeo_sd.io import FitacfReader, beams_to_scans

_SUFFIXES = (".fitacf", ".fitacf3", ".bz2", ".gz")


def _expand(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(f for f in p.rglob("*") if ".fitacf" in f.name and f.suffix in _SUFFIXES) if p.is_dir() else [p])
    return files


def cmd_info(args: argparse.Namespace) -> int:
    """Print a one-block summary (beams, scans, echoes) for fitacf files."""
    files = _expand(args.paths)
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
    print(f"scans          : {len(scans)} (beams/scan {min(len(s.beams) for s in scans)}-{max(len(s.beams) for s in scans)},"
          f" median period {np.median(periods):.0f} s)")
    print(f"beam numbers   : {min(b.bmnum for b in beams)}-{max(b.bmnum for b in beams)}; range gates {max(b.nrang for b in beams)}")
    print(f"tfreq [kHz]    : {min(b.tfreq for b in beams):.0f}-{max(b.tfreq for b in beams):.0f}")
    print(f"echoes         : {echoes} ({100 * gs / max(echoes, 1):.1f}% flagged ground scatter by fitacf)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amgeo_sd", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("info", help=cmd_info.__doc__)
    p.add_argument("paths", nargs="+", help="fitacf files or directories containing them")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.enable("amgeo_sd")
    logger.add(sys.stderr, level="INFO")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
