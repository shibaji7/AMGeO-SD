"""Read SuperDARN ``fitacf`` / ``fitacf3`` files into beams, scans and DataFrames.

Replaces the legacy ``sd/get_fit_data.py`` (``FetchData``). Main differences:

* decoding uses ``pydarnio.read_fitacf`` (pydarnio >= 2), not ``SDarnRead``;
* file discovery parses the *file name* (not the full path) and takes the data
  root from an argument or the ``AMGEO_SD_DATA`` environment variable instead of
  a cwd-relative ``config/conf.ini``;
* ``.bz2``, ``.gz`` and plain files are all accepted.

Typical use::

    reader = FitacfReader("bks", start, end, files=["20150317.0200.00.bks.a.fitacf.bz2"])
    scans = reader.scans()
    df = scans_to_dataframe(scans)
"""

from __future__ import annotations

import bz2
import datetime as dt
import gzip
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pydarnio
from loguru import logger

from amgeo_sd.models import GATE_FIELDS, Beam, Scan

#: Environment variable holding the archive root, used when ``root`` is not given.
DATA_ROOT_ENV = "AMGEO_SD_DATA"
#: Default layout below the root: ``<year>/<ftype>/<rad>/<YYYYMMDD>.*``.
DEFAULT_LAYOUT = "{year}/{ftype}/{rad}/{date}.*{ftype}*"
#: Each archive file spans two hours.
FILE_SPAN = dt.timedelta(hours=2)

_FNAME_RE = re.compile(r"^(?P<date>\d{8})\.(?P<hm>\d{4})\.(?P<sc>\d{2})\.")

# fitacf record key -> Beam attribute, for scalar header values.
_SCALARS = {
    "bmnum": "bmnum", "channel": "channel", "cp": "cp", "tfreq": "tfreq",
    "noise.sky": "noise_sky", "nrang": "nrang", "rsep": "rsep", "frang": "frang",
    "smsep": "smsep", "lagfr": "lagfr", "mppul": "mppul",
    "intt.sc": "intt_sc", "intt.us": "intt_us",
}
_INT_ATTRS = {"bmnum", "channel", "cp", "nrang", "mppul", "intt_sc", "intt_us"}


def _open_bytes(path: Path) -> bytes:
    """Return the decompressed bytes of ``path`` (``.bz2``, ``.gz`` or plain)."""
    opener = {".bz2": bz2.open, ".gz": gzip.open}.get(path.suffix, open)
    with opener(path, "rb") as fp:
        return fp.read()


def read_fitacf_file(path: str | os.PathLike) -> list[dict]:
    """Decode every record of one fitacf file.

    Corrupted trailing data is tolerated: valid records are returned and a
    warning is logged with the byte offset where decoding stopped.
    """
    path = Path(path)
    records, bad_byte = pydarnio.read_fitacf(_open_bytes(path), mode="lax")
    if bad_byte is not None:
        logger.warning(f"{path.name}: corrupt data from byte {bad_byte}; kept {len(records)} records")
    return records


def _record_time(rec: dict) -> dt.datetime:
    return dt.datetime(
        int(rec["time.yr"]), int(rec["time.mo"]), int(rec["time.dy"]),
        int(rec["time.hr"]), int(rec["time.mt"]), int(rec["time.sc"]), int(rec["time.us"]),
    )


def _per_echo(rec: dict, key: str, n: int, dtype, fill) -> np.ndarray:
    """Per-echo array ``key`` of length ``n``; ``fill`` if the key is absent."""
    if key not in rec:
        return np.full(n, fill, dtype=dtype)
    arr = np.atleast_1d(np.asarray(rec[key])).astype(dtype)
    if arr.shape != (n,):
        raise ValueError(f"record field {key!r} has length {arr.size}, expected {n} (len(slist))")
    return arr


def record_to_beam(rec: dict) -> Beam:
    """Convert one decoded fitacf record into a :class:`Beam`."""
    kwargs: dict = {}
    for key, attr in _SCALARS.items():
        if key in rec:
            kwargs[attr] = int(rec[key]) if attr in _INT_ATTRS else float(rec[key])
    # abnormal scan starts are flagged -1 in fitacf; treat every non-zero as "new scan"
    kwargs["scan"] = int(rec.get("scan", 0) != 0)
    slist = np.atleast_1d(np.asarray(rec["slist"])).astype(int) if "slist" in rec else np.empty(0, dtype=int)
    n = len(slist)
    return Beam(
        time=_record_time(rec),
        slist=slist,
        v=_per_echo(rec, "v", n, float, np.nan),
        v_e=_per_echo(rec, "v_e", n, float, np.nan),
        w_l=_per_echo(rec, "w_l", n, float, np.nan),
        p_l=_per_echo(rec, "p_l", n, float, np.nan),
        gflg=_per_echo(rec, "gflg", n, np.int8, -1),
        **kwargs,
    )


def records_to_beams(
    records: Iterable[dict], start: dt.datetime | None = None, end: dt.datetime | None = None
) -> list[Beam]:
    """Convert records to beams, keeping those with ``start <= time <= end``.

    Beams with no scatter are kept: they carry the scan-start flag that
    :func:`beams_to_scans` relies on.
    """
    beams = []
    for rec in records:
        beam = record_to_beam(rec)
        if (start is None or beam.time >= start) and (end is None or beam.time <= end):
            beams.append(beam)
    return beams


def beams_to_scans(beams: Sequence[Beam], s_mode: str = "normal") -> list[Scan]:
    """Group time-ordered beams into scans using the scan-start flag.

    A new scan opens at every beam with ``scan == 1`` whose time differs from the
    previous beam's (dual-channel radars repeat the flag at the same time).
    """
    if not beams:
        return []
    scans, current = [], Scan(beams=[beams[0]], s_mode=s_mode)
    for prev, beam in zip(beams[:-1], beams[1:]):
        if beam.scan == 1 and beam.time != prev.time:
            scans.append(current.update())
            current = Scan(s_mode=s_mode)
        current.beams.append(beam)
    scans.append(current.update())
    return scans


_BEAM_COLUMNS = (
    "time", "bmnum", "channel", "cp", "tfreq", "noise_sky", "scan", "nrang", "rsep", "frang",
    "smsep", "lagfr", "mppul", "intt_sc", "intt_us",
)


def beams_to_dataframe(beams: Iterable[Beam], extra: Sequence[str] = ()) -> pd.DataFrame:
    """One row per echo, with the beam metadata repeated on each row.

    ``extra`` names additional per-echo arrays stored in ``Beam.extra`` to include.
    """
    frames = []
    for b in beams:
        n = b.n_echoes
        if n == 0:
            continue
        cols = {c: np.full(n, getattr(b, c)) for c in _BEAM_COLUMNS if c != "time"}
        cols["time"] = np.full(n, np.datetime64(b.time, "us"))
        cols.update({f: getattr(b, f) for f in GATE_FIELDS})
        cols.update({k: b.extra[k] for k in extra})
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame(columns=list(_BEAM_COLUMNS) + list(GATE_FIELDS) + list(extra))
    df = pd.concat(frames, ignore_index=True)
    return df[list(_BEAM_COLUMNS) + list(GATE_FIELDS) + list(extra)]


def scans_to_dataframe(scans: Iterable[Scan], extra: Sequence[str] = (), start_scnum: int = 0) -> pd.DataFrame:
    """Like :func:`beams_to_dataframe` plus ``scnum`` (scan counter) and ``sbnum``
    (beam index inside the scan)."""
    frames = []
    for i, scan in enumerate(scans):
        for j, beam in enumerate(scan.beams):
            df = beams_to_dataframe([beam], extra)
            df["scnum"], df["sbnum"] = i + start_scnum, j
            frames.append(df)
    if not frames:
        return beams_to_dataframe([], extra).assign(scnum=pd.Series(dtype=int), sbnum=pd.Series(dtype=int))
    return pd.concat(frames, ignore_index=True)


def find_fitacf_files(
    rad: str,
    start: dt.datetime,
    end: dt.datetime,
    ftype: str = "fitacf",
    root: str | os.PathLike | None = None,
    layout: str = DEFAULT_LAYOUT,
) -> list[Path]:
    """Locate archive files holding data between ``start`` and ``end``.

    Parameters
    ----------
    root : path, optional
        Archive root; defaults to ``$AMGEO_SD_DATA``.
    layout : str
        Glob below ``root`` with ``{year}``, ``{ftype}``, ``{rad}``, ``{date}`` fields.

    A file named ``YYYYMMDD.HHMM.SS.*`` is selected when its two-hour span
    overlaps ``[start, end]``. The day before ``start`` is searched too because
    a file can begin before midnight.
    """
    root = root if root is not None else os.environ.get(DATA_ROOT_ENV)
    if root is None:
        raise ValueError(f"pass root= or set ${DATA_ROOT_ENV}")
    root = Path(root)
    found: set[Path] = set()
    day = (start - dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        pattern = layout.format(year=day.year, ftype=ftype, rad=rad, date=day.strftime("%Y%m%d"))
        for path in root.glob(pattern):
            t0 = _file_start(path)
            if t0 is not None and t0 <= end and t0 + FILE_SPAN > start:
                found.add(path)
        day += dt.timedelta(days=1)
    return sorted(found)


def _file_start(path: Path) -> dt.datetime | None:
    m = _FNAME_RE.match(path.name)
    if m is None:
        logger.warning(f"skipping {path}: name does not start with YYYYMMDD.HHMM.SS.")
        return None
    return dt.datetime.strptime(m["date"] + m["hm"] + m["sc"], "%Y%m%d%H%M%S")


class FitacfReader:
    """Load fitacf data for one radar and time window.

    Parameters
    ----------
    rad : str
        Three-letter radar code (``"bks"``).
    start, end : datetime
        Inclusive time window (UTC, naive).
    ftype : str
        ``"fitacf"`` or ``"fitacf3"``; only used for file discovery.
    files : sequence of path, optional
        Explicit files to read; skips discovery.
    root : path, optional
        Archive root for discovery (default ``$AMGEO_SD_DATA``).
    """

    def __init__(
        self,
        rad: str,
        start: dt.datetime,
        end: dt.datetime,
        ftype: str = "fitacf",
        files: Sequence[str | os.PathLike] | None = None,
        root: str | os.PathLike | None = None,
    ):
        self.rad, self.start, self.end, self.ftype = rad, start, end, ftype
        self.files = (
            sorted(Path(f) for f in files) if files is not None
            else find_fitacf_files(rad, start, end, ftype, root)
        )
        if not self.files:
            logger.warning(f"no {ftype} files for {rad} between {start} and {end}")

    def beams(self) -> list[Beam]:
        """All beams in the window, time-ordered."""
        beams: list[Beam] = []
        for f in self.files:
            logger.info(f"reading {f}")
            beams.extend(records_to_beams(read_fitacf_file(f), self.start, self.end))
        beams.sort(key=lambda b: b.time)
        return beams

    def scans(self, s_mode: str = "normal") -> list[Scan]:
        """All scans in the window (see :func:`beams_to_scans`)."""
        return beams_to_scans(self.beams(), s_mode)
