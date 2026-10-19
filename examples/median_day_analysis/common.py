"""Arguments and file lookup shared by the analysis scripts."""

import argparse
import datetime as dt
import glob
from pathlib import Path

from dateutil import parser as dateparser

KEYS = ["time", "bmnum", "slist"]


def parse_args(description: str, argv=None, extra=None) -> argparse.Namespace:
    """Common command line: radar, the folder holding the model outputs, the raw data and the time window."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--rad", required=True, help="radar code, e.g. bks")
    ap.add_argument(
        "--work",
        required=True,
        help="OUT of `make median` / `make dbscan`: holds <median-sim>/<rad>/ and <dbscan-sim>/<rad>/",
    )
    ap.add_argument("--data", help="raw fitacf files or directory (default tests/data/<rad>_<day without dashes>)")
    ap.add_argument("--day", default="2025-02-08", help="UT day analysed (default 2025-02-08)")
    ap.add_argument("--start", type=dateparser.parse, help="window start (default: start of --day)")
    ap.add_argument("--end", type=dateparser.parse, help="window end (default: end of --day)")
    ap.add_argument("--median-sim", default="day", help="sim_id of the median-filter run (default day)")
    ap.add_argument("--dbscan-sim", default="day-dbscan", help="sim_id of the DBSCAN run (default day-dbscan)")
    if extra:
        extra(ap)
    return ap.parse_args(argv)


def window(args) -> tuple[dt.datetime, dt.datetime]:
    """Analysis window: ``--start`` / ``--end`` or the whole ``--day``."""
    day = dateparser.parse(args.day)
    return args.start or day, args.end or day.replace(hour=23, minute=59, second=59)


def data_files(args) -> list[Path]:
    """Raw fitacf files of the analysis (sorted)."""
    root = Path(args.data or f"tests/data/{args.rad}_{args.day.replace('-', '')}")
    files = sorted(root.rglob("*.fitacf*")) if root.is_dir() else [root]
    if not files:
        raise SystemExit(f"no fitacf files under {root}")
    return files


def nc_file(args, sim: str) -> str:
    """The fitacf++ NetCDF of one run."""
    found = sorted(glob.glob(f"{args.work}/{sim}/{args.rad}/*.nc"))
    if not found:
        raise SystemExit(
            f"no .nc under {args.work}/{sim}/{args.rad}/ - run `make median` / `make dbscan` first (see README.md)"
        )
    return found[0]
