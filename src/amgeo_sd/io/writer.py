"""Write fitacf++ as a DMAP file: ``<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pp.bz2``.

Each surviving beam of a filtered scan becomes one record built from the *original* fitacf record
of the same sounding, so every standard fitacf field (header, ``pwr0``, ``ltab``, ...) is kept and
tools that read DMAP and look only for the fitacf fields keep working. Then

* ``slist, v, w_l, p_l, gflg`` are replaced by the filtered values; ``v_e`` (velocity error) is set to
  ``v_mad`` (a robust local spread, similar in size to the fit errors) so it is never NaN, and ``qflg`` is
  1 for cells the filter added (original cells keep their flag), so tools that select on the quality flag show them;
* every other per-echo fitacf array (``qflg``, ``p_s``, ``elv``, ``x_*``, ...) is carried over for
  gates that held an echo in the original sounding and is NaN (floats) or 0 (flags/ints) for cells
  the filter added;
* only when ``extra_fields`` is on (default off): the arrays ``v_mad`` (float32), ``gflg_conv``, ``gflg_kde``
  (int8, -1 = not computed), ``clabel``, ``cluster_tag`` (int16) and ``gflg_ribiero`` (int8), plus the strings
  ``amgeo.model`` and ``amgeo.version`` are added (see :data:`EXTRA_RECORD_FIELDS`);
* the scan flag is re-set to 1 on the earliest kept beam of every scan and 0 elsewhere, so scans
  survive the removal of beams without echoes;
* ``origin.command`` / ``origin.time`` record the processing.

With the default ``extra_fields=False`` the file holds standard fitacf fields only, so ``pydarnio.read_fitacf``, pydarn and
other fitacf readers open it. ``pydarnio.read_fitacf`` refuses the extra fields: with ``extra_fields=True`` read the file with
:func:`amgeo_sd.io.read_fitacf_file` (falls back to generic DMAP) or ``pydarnio.read_dmap``.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pydarnio
from loguru import logger

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


#: The non-standard DMAP fields written when ``extra_fields`` is on (all other fields are standard fitacf fields).
EXTRA_RECORD_FIELDS = (*_EXTRA_DTYPES, "amgeo.model", "amgeo.version")


def fitacfpp_name(rad: str, start: dt.datetime) -> str:
    """``YYYYMMDD.HHMM.SS.<rad>.fitacf.pp.bz2``: the archive naming with the fitacf++ suffix.

    Parameters
    ----------
    rad : str
        Three-letter radar code.
    start : datetime
        Time stamp of the file, normally the first record.

    Returns
    -------
    str
        The file name (no directory).
    """
    return f"{start:%Y%m%d.%H%M.%S}.{rad}{FITACFPP_SUFFIX}"


def _fill(dtype: np.dtype) -> float | int:
    return np.nan if dtype.kind == "f" else 0


def _without_extras(record: dict, extra_fields: bool) -> dict:
    """Copy of ``record``, without the non-standard fields unless ``extra_fields``."""
    return {k: v for k, v in record.items() if extra_fields or k not in EXTRA_RECORD_FIELDS}


def beam_to_record(beam: Beam, template: dict, extra_fields: bool = False) -> dict:
    """Build the fitacf++ record of ``beam`` from the original ``template`` record.

    Parameters
    ----------
    beam : Beam
        The processed beam (its ``slist``, ``v``, ``w_l``, ``p_l``, ``gflg`` and ``extra`` arrays replace the template's).
    template : dict
        The raw record of the same sounding; all its other fields are kept.
    extra_fields : bool
        Also store the non-standard arrays of :data:`EXTRA_RECORD_FIELDS` that the beam carries. ``False`` leaves a
        record of standard fitacf fields only (any such field already in the template is removed).

    Returns
    -------
    dict
        A DMAP record with every field of the template. ``slist``, ``v``, ``w_l``, ``p_l`` and ``gflg`` are the beam's;
        ``v_e`` is its ``v_mad`` when present; ``qflg`` is 1 for cells the filter added; other per-echo arrays are
        carried over for gates that held an echo in the template (NaN or 0 for added cells); ``v_mad``, ``gflg_conv``,
        ``gflg_kde``, ``clabel``, ``cluster_tag`` and ``gflg_ribiero`` are added when ``extra_fields`` is on and the
        beam has them.
    """
    rec = _without_extras(template, extra_fields)
    old_slist = np.atleast_1d(np.asarray(template["slist"])).astype(int) if "slist" in template else np.empty(0, int)
    where = {int(g): i for i, g in enumerate(old_slist)}
    src = np.array([where.get(int(g), -1) for g in beam.slist], dtype=int)
    n = len(beam.slist)

    for key, val in template.items():
        if key == "slist" or key in _PER_GATE or key not in rec or not isinstance(val, np.ndarray):
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
    if "v_mad" in beam.extra:
        rec["v_e"] = np.asarray(beam.extra["v_mad"]).astype(like.get("v_e", np.float32))
    if "qflg" in rec:  # cells the filter added have no original quality flag: mark them good
        rec["qflg"] = np.where(src >= 0, rec["qflg"], 1).astype(template["qflg"].dtype)
    for key, dtype in _EXTRA_DTYPES.items():
        if extra_fields and key in beam.extra:
            arr = np.asarray(beam.extra[key], dtype=float)
            if np.issubdtype(dtype, np.integer):
                arr = np.where(np.isnan(arr), -1, arr)
            rec[key] = arr.astype(dtype)
    return rec


def scans_to_records(
    scans: Sequence[Scan],
    source_records: Sequence[dict],
    command: str = "",
    model: str = "median",
    keep_empty: bool = False,
    extra_fields: bool = False,
) -> list[dict]:
    """fitacf++ records for the beams of ``scans``, in time order.

    By default beams without echoes are dropped and the scan flag is re-set on the first kept beam of every scan
    (filters that remove beams); with ``keep_empty`` every beam is written and the original scan flags are kept
    (models that only label the raw echoes).

    Parameters
    ----------
    scans : sequence of Scan
        Processed scans.
    source_records : sequence of dict
        The raw records the scans were derived from (:meth:`FitacfReader.records`); a beam without a matching record
        (same time, beam number and channel) raises ``KeyError``.
    command : str
        Text for ``origin.command`` (the command line of the run); ``"amgeo_sd"`` when empty.
    model : str
        Stored in ``amgeo.model``: ``"median"`` or ``"dbscan"``.
    keep_empty : bool
        Write beams without echoes too and keep the original scan flags.
    extra_fields : bool
        Store the non-standard fields (:data:`EXTRA_RECORD_FIELDS`: ``v_mad``, ``gflg_conv``, ``gflg_kde``, ``clabel``,
        ``cluster_tag``, ``gflg_ribiero``, ``amgeo.model``, ``amgeo.version``). Off by default, so that
        ``pydarnio.read_fitacf`` and other fitacf readers can open the file.

    Returns
    -------
    list of dict
        One DMAP record per written beam, sorted by time.
    """
    index = {(_record_time(r), int(r["bmnum"]), int(r.get("channel", 0))): r for r in source_records}
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for scan in scans:
        beams = sorted((b for b in scan.beams if keep_empty or b.n_echoes), key=lambda b: b.time)
        for i, b in enumerate(beams):
            try:
                tmpl = index[(b.time, b.bmnum, b.channel)]
            except KeyError:
                raise KeyError(f"no source record for beam {b.bmnum} at {b.time}") from None
            rec = beam_to_record(b, tmpl, extra_fields) if b.n_echoes else _without_extras(tmpl, extra_fields)
            if not keep_empty:
                scan_dtype = np.asarray(tmpl.get("scan", 0)).dtype
                rec["scan"] = scan_dtype.type(1 if i == 0 else 0)
            rec["origin.command"] = command or "amgeo_sd"
            rec["origin.time"] = now
            if extra_fields:
                rec["amgeo.model"] = model
                rec["amgeo.version"] = __version__
            out.append(rec)
    out.sort(key=_record_time)
    return out


def write_fitacfpp(path: str | os.PathLike, records: Sequence[dict]) -> Path:
    """Write ``records`` as a bz2-compressed DMAP stream.

    Records of standard fitacf fields only (``extra_fields`` off) go through ``pydarnio.write_fitacf``, which stores every
    field with its fitacf data type, so ``pydarnio.read_fitacf``, pydarn and other fitacf readers open the file. If a
    record is not valid fitacf (e.g. a field is missing) a warning is logged and the generic writer is used instead. Records
    with the non-standard fields (:data:`EXTRA_RECORD_FIELDS`) always use ``pydarnio.write_dmap``, which infers the data
    types: read such a file with :func:`amgeo_sd.io.read_fitacf_file` or ``pydarnio.read_dmap``.

    ``pydarnio`` *appends* to an existing file, so the data go to a temporary file that
    replaces ``path`` atomically; re-running a pipeline therefore never duplicates records.

    Parameters
    ----------
    path : str or path
        Output file (use :func:`fitacfpp_name`); parent directories are created.
    records : sequence of dict
        Records from :func:`scans_to_records`.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.unlink(missing_ok=True)
    records = list(records)
    standard = not any(key in EXTRA_RECORD_FIELDS for rec in records for key in rec)
    written = False
    if standard:
        try:
            pydarnio.write_fitacf(records, str(tmp), bz2=True)
            written = True
        except Exception as exc:  # a record that is not valid fitacf
            logger.warning(f"{path.name}: not valid fitacf ({str(exc)[:120]}); written as generic DMAP")
            tmp.unlink(missing_ok=True)
    if not written:
        pydarnio.write_dmap(records, str(tmp), bz2=True)
    os.replace(tmp, path)
    return path
