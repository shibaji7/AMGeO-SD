"""Write fitacf++ as a DMAP file: ``<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pl.bz2``.

Each surviving beam of a filtered scan becomes one record built from the *original* fitacf record
of the same sounding, so every standard fitacf field (header, ``pwr0``, ``ltab``, ...) is kept and
tools that read DMAP and look only for the fitacf fields keep working. Then

* ``slist, v, w_l, p_l, gflg`` are replaced by the filtered values and ``v_e`` is NaN (the median
  has no fit error);
* every other per-echo fitacf array (``qflg``, ``p_s``, ``elv``, ``x_*``, ...) is carried over for
  gates that held an echo in the original sounding and is NaN (floats) or 0 (flags/ints) for cells
  the filter added;
* the arrays ``v_mad`` (float32), ``gflg_conv``, ``gflg_kde`` (int8, -1 = not computed) and
  ``clabel`` (int16) are added, plus the strings ``amgeo.model`` and ``amgeo.version``;
* the scan flag is re-set to 1 on the earliest kept beam of every scan and 0 elsewhere, so scans
  survive the removal of beams without echoes;
* ``origin.command`` / ``origin.time`` record the processing.

``pydarnio.read_fitacf`` refuses the extra fields; :func:`amgeo_sd.io.read_fitacf_file` falls back
to generic DMAP, and ``pydarnio.read_dmap`` reads the files directly.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pydarnio

from amgeo_sd import __version__
from amgeo_sd.io.fitacf import FITACFPP_SUFFIX, _record_time
from amgeo_sd.models import Beam, Scan

#: fitacf arrays indexed by range gate rather than by echo; copied unchanged.
_PER_GATE = {"pwr0"}
_EXTRA_DTYPES = {
    "v_mad": np.float32,
    "gflg_conv": np.int8,
    "gflg_kde": np.int8,
    "clabel": np.int16,
    "cluster_tag": np.int16,
    "gflg_ribiero": np.int8,
}


def fitacfpp_name(rad: str, start: dt.datetime) -> str:
    """``YYYYMMDD.HHMM.SS.<rad>.fitacf.pl.bz2``: the archive naming with the fitacf++ suffix."""
    return f"{start:%Y%m%d.%H%M.%S}.{rad}{FITACFPP_SUFFIX}"


def _fill(dtype: np.dtype) -> float | int:
    return np.nan if dtype.kind == "f" else 0


def beam_to_record(beam: Beam, template: dict) -> dict:
    """Build the fitacf++ record of ``beam`` from the original ``template`` record."""
    rec = dict(template)
    old_slist = np.atleast_1d(np.asarray(template["slist"])).astype(int) if "slist" in template else np.empty(0, int)
    where = {int(g): i for i, g in enumerate(old_slist)}
    src = np.array([where.get(int(g), -1) for g in beam.slist], dtype=int)
    n = len(beam.slist)

    for key, val in template.items():
        if key == "slist" or key in _PER_GATE or not isinstance(val, np.ndarray):
            continue
        if val.ndim == 1 and val.shape == old_slist.shape:
            new = np.full(n, _fill(val.dtype), dtype=val.dtype)
            ok = src >= 0
            new[ok] = val[src[ok]]
            rec[key] = new
    like = {k: template[k].dtype for k in ("slist", "v", "v_e", "w_l", "p_l", "gflg") if k in template}
    rec["slist"] = beam.slist.astype(like.get("slist", np.int16))
    for key, arr in (("v", beam.v), ("v_e", beam.v_e), ("w_l", beam.w_l), ("p_l", beam.p_l)):
        rec[key] = np.asarray(arr).astype(like.get(key, np.float32))
    rec["gflg"] = beam.gflg.astype(like.get("gflg", np.int8))
    for key, dtype in _EXTRA_DTYPES.items():
        if key in beam.extra:
            arr = np.asarray(beam.extra[key], dtype=float)
            if np.issubdtype(dtype, np.integer):
                arr = np.where(np.isnan(arr), -1, arr)
            rec[key] = arr.astype(dtype)
    return rec


def scans_to_records(scans: Sequence[Scan], source_records: Sequence[dict], command: str = "") -> list[dict]:
    """fitacf++ records for all beams of ``scans``, in time order.

    ``source_records`` are the raw records the scans were derived from (see
    :meth:`amgeo_sd.io.FitacfReader.records`); a beam without a matching record raises ``KeyError``.
    """
    index = {(_record_time(r), int(r["bmnum"]), int(r.get("channel", 0))): r for r in source_records}
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for scan in scans:
        beams = sorted((b for b in scan.beams if b.n_echoes), key=lambda b: b.time)
        for i, b in enumerate(beams):
            try:
                tmpl = index[(b.time, b.bmnum, b.channel)]
            except KeyError:
                raise KeyError(f"no source record for beam {b.bmnum} at {b.time}") from None
            rec = beam_to_record(b, tmpl)
            scan_dtype = np.asarray(tmpl.get("scan", 0)).dtype
            rec["scan"] = scan_dtype.type(1 if i == 0 else 0)
            rec["origin.command"] = command or "amgeo_sd"
            rec["origin.time"] = now
            rec["amgeo.model"] = "median"
            rec["amgeo.version"] = __version__
            out.append(rec)
    out.sort(key=_record_time)
    return out


def write_fitacfpp(path: str | os.PathLike, records: Sequence[dict]) -> Path:
    """Write ``records`` as a bz2-compressed DMAP stream (generic writer: allows the extra fields).

    ``pydarnio.write_dmap`` *appends* to an existing file, so the data go to a temporary file that
    replaces ``path`` atomically; re-running a pipeline therefore never duplicates records.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.unlink(missing_ok=True)
    pydarnio.write_dmap(list(records), str(tmp), bz2=True)
    os.replace(tmp, path)
    return path
