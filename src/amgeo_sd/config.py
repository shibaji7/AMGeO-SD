"""Run parameters from TOML files.

Precedence, lowest to highest: built-in defaults < TOML file < command-line overrides. Unknown
keys are errors, so a typo in a config file cannot silently fall back to a default.

A config file has one table per topic; see ``configs/default.toml`` for every key with comments::

    [run]      rad, start, end, sim_id, out, jobs
    [input]    paths, root, ftype
    [median]   thresh, pth, lth, uth, kde, cluster, eps, min_samples
    [output]   netcdf, fitacf
    [plot]     inputs, compare, rad, beam, start, end, time, params, flag, coords, projection, ...
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from dateutil import parser as dateparser


@dataclass
class Run:
    """``[run]``: what to process and where results go."""

    rad: str | None = None
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    sim_id: str = "L100"
    out: str = "data/outputs"
    jobs: int = 1


@dataclass
class Input:
    """``[input]``: where the fitacf data come from."""

    paths: list[str] = field(default_factory=list)  # fitacf files or directories; empty = archive lookup
    root: str | None = None  # archive root (default $AMGEO_SD_DATA)
    ftype: str = "fitacf"


@dataclass
class Median:
    """``[median]``: parameters of the median-filter model."""

    thresh: float = 0.7
    pth: float = 0.25
    lth: float = 0.2
    uth: float = 0.8
    kde: bool = True
    cluster: bool = True
    eps: float = 2.0
    min_samples: int = 5


@dataclass
class Output:
    """``[output]``: which products to write."""

    netcdf: bool = True  # <rad>_<start>_<end>.fitacf++.nc
    fitacf: bool = True  # <YYYYMMDD.HHMM.SS>.<rad>.fitacf.pl.bz2


@dataclass
class Plot:
    """``[plot]``: figures from fitacf or fitacf++ data."""

    inputs: list[str] = field(
        default_factory=list
    )  # fitacf / fitacf++ files or directories (.nc, .fitacf.pl.bz2, .fitacf[.bz2])
    compare: list[str] = field(default_factory=list)  # RTI only: more inputs, stacked below the first for comparison
    rad: str | None = None  # default: taken from the files
    beam: int = 0  # RTI beam
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    time: dt.datetime | None = None  # FoV: map of the scan nearest to this time
    params: list[str] = field(default_factory=lambda: ["v", "w_l", "p_l", "gflg"])
    flag: str = "gflg"  # RTI compare: classification panel under each dataset ("" = none)
    coords: str = "geo"  # FoV: "geo" or "aacgmv2"
    projection: str = "auto"  # FoV: "auto", "polar" or "ortho"
    max_gate: int = 0  # FoV: gates drawn (0 = all in the data)
    terminator: bool = False  # FoV: shade the night side (geo only)
    out: str = "data/figures"
    name: str = ""  # file name without extension ("" = generated)
    format: str = "png"
    dpi: int = 150


@dataclass
class Settings:
    """All parameter tables; build with :func:`load_settings`."""

    run: Run = field(default_factory=Run)
    input: Input = field(default_factory=Input)
    median: Median = field(default_factory=Median)
    output: Output = field(default_factory=Output)
    plot: Plot = field(default_factory=Plot)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Plain nested dict, one entry per table."""
        return asdict(self)

    def validate(self, require: str = "median") -> Settings:
        """Check the values; ``require`` is the workflow the settings are for: ``"median"`` or ``"plot"``."""
        if require == "plot":
            if not self.plot.inputs:
                raise ValueError("[plot] inputs is empty")
            if self.plot.coords not in ("geo", "aacgmv2"):
                raise ValueError("[plot] coords must be 'geo' or 'aacgmv2'")
            if self.plot.projection not in ("auto", "polar", "ortho"):
                raise ValueError("[plot] projection must be auto, polar or ortho")
            return self
        missing = [k for k in ("rad", "start", "end") if getattr(self.run, k) is None]
        if missing:
            raise ValueError(f"missing required setting(s) in [run]: {', '.join(missing)}")
        if self.run.end < self.run.start:
            raise ValueError("[run] end is before start")
        if not 0 < self.median.thresh <= 1:
            raise ValueError("[median] thresh must be in (0, 1]")
        if not 0 <= self.median.lth < self.median.uth <= 1:
            raise ValueError("[median] need 0 <= lth < uth <= 1")
        if not (self.output.netcdf or self.output.fitacf):
            raise ValueError("[output] enable at least one of netcdf / fitacf")
        return self


def _coerce(name: str, value: Any, current: Any) -> Any:
    if name in ("start", "end", "time") and value is not None:
        if isinstance(value, dt.datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, dt.date):
            return dt.datetime(value.year, value.month, value.day)
        return dateparser.parse(str(value))
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [str(v) for v in ([value] if isinstance(value, str) else value)]
    return value


#: Tables written for information only (by :func:`dump_settings_toml`); ignored when a file is loaded.
INFO_TABLES = ("provenance", "result")


def _apply(settings: Settings, table: str, values: dict[str, Any], origin: str) -> None:
    if table in INFO_TABLES:
        return
    section = getattr(settings, table, None)
    if section is None or table.startswith("_"):
        raise ValueError(f"{origin}: unknown table [{table}]; valid: {', '.join(f.name for f in fields(settings))}")
    valid = {f.name for f in fields(section)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"{origin}: unknown key(s) in [{table}]: {', '.join(sorted(unknown))}; valid: {', '.join(sorted(valid))}"
        )
    for key, value in values.items():
        setattr(section, key, _coerce(key, value, getattr(section, key)))


def load_settings(
    path: str | Path | None = None, overrides: dict[str, dict[str, Any]] | None = None, require: str = "median"
) -> Settings:
    """Defaults, then the TOML ``path``, then ``overrides`` (``{"run": {"rad": "bks"}, ...}``; None values skipped).

    Relative ``input.paths``, ``input.root`` and ``run.out`` in a file are kept as written (relative
    to the working directory, as for command-line values).
    """
    settings = Settings()
    if path is not None:
        with open(path, "rb") as fp:
            data = tomllib.load(fp)
        for table, values in data.items():
            if not isinstance(values, dict):
                raise ValueError(f"{path}: top-level key {table!r} must be a [table]")
            _apply(settings, table, values, str(path))
    for table, values in (overrides or {}).items():
        _apply(settings, table, {k: v for k, v in values.items() if v is not None}, "command line")
    return settings.validate(require)


# ----------------------------------------------------------------------------- annotated output
TABLE_DOCS = {
    "run": "What to process and where results go.",
    "input": "Where the fitacf data come from.",
    "median": "Median-filter model parameters.",
    "output": "Which products to write.",
    "plot": "Figures from fitacf or fitacf++ data.",
}
FIELD_DOCS = {
    "run": {
        "rad": "three-letter radar code",
        "start": "UT; scans starting in [start, end] are written",
        "end": "UT",
        "sim_id": "run id: results go to <out>/<sim_id>/<rad>/",
        "out": "output base directory",
        "jobs": "worker processes for the filter",
    },
    "input": {
        "paths": "fitacf files or directories (fitacf 2.0/2.5/3.0, .bz2/.gz/plain); empty = use the archive",
        "root": "archive root laid out <year>/<ftype>/<rad>/; empty = $AMGEO_SD_DATA",
        "ftype": '"fitacf" or "fitacf3" (archive lookup only)',
    },
    "median": {
        "thresh": "min fraction of the 3x3x3 neighbour weight that must hold an echo",
        "pth": "Beta-distribution cut for the KDE ground-scatter flag",
        "lth": "P(ground) <= lth  -> ionospheric (0)",
        "uth": "P(ground) >= uth  -> ground (1); in between -> undetermined (2)",
        "kde": "compute gflg_kde (the slow Beta fit); false -> NaN",
        "cluster": "DBSCAN labels per scan -> clabel",
        "eps": "DBSCAN radius in (beam, gate) units",
        "min_samples": "DBSCAN core-point size",
    },
    "output": {
        "netcdf": "<rad>_<start>_<end>.fitacf++.nc",
        "fitacf": "<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pl.bz2 (DMAP with extra fields)",
    },
    "plot": {
        "inputs": "fitacf / fitacf++ files or directories (.nc, .fitacf.pl.bz2, .fitacf[.bz2|.gz])",
        "compare": "RTI: more inputs stacked below the first one",
        "rad": "radar code; empty = from the files",
        "beam": "RTI beam number",
        "start": "UT, only data from here (optional)",
        "end": "UT, only data up to here (optional)",
        "time": "UT, FoV map of the scan nearest to this time",
        "params": "RTI panels / FoV maps: v w_l p_l v_mad, flags gflg gflg_conv gflg_kde gflg_ribiero, cluster_tag",
        "flag": "RTI compare: classification panel for each dataset; empty = none",
        "coords": '"geo" or "aacgmv2"',
        "projection": '"auto", "polar" or "ortho"',
        "max_gate": "FoV: number of gates drawn; 0 = all in the data",
        "terminator": "FoV: shade the night side (geo coordinates only)",
        "out": "output directory",
        "name": "file name without extension; empty = generated from radar, time and parameters",
        "format": "png, pdf or svg",
        "dpi": "resolution of raster output",
    },
}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None).isoformat()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value))  # a JSON string is a valid TOML basic string


_LONG = 44  # values longer than this get their comment on the line above instead of beside them


def _table(name: str, rows: list[tuple[str, Any, str]], header: str = "") -> list[str]:
    """Lines of one table: ``key = value   # comment`` with aligned columns; unset values are commented out."""
    lines = ([f"# {header}"] if header else []) + [f"[{name}]"]
    width = max((len(k) for k, _, _ in rows), default=0)
    rendered = [(k, _toml_value(v) if v is not None else None, c) for k, v, c in rows]
    vwidth = max((len(v) for _, v, _ in rendered if v is not None and len(v) <= _LONG), default=0)
    for k, v, c in rendered:
        if v is None:
            lines.append(f"# {k.ljust(width)} =  (unset)" + (f"   # {c}" if c else ""))
        elif len(v) > _LONG:
            lines += ([f"# {k}: {c}"] if c else []) + [f"{k.ljust(width)} = {v}"]
        else:
            lines.append((f"{k.ljust(width)} = {v.ljust(vwidth)}" + (f"   # {c}" if c else "")).rstrip())
    return lines + [""]


def dump_settings_toml(
    settings: Settings,
    title: str = "AMGeO-SD run parameters",
    provenance: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    tables: Sequence[str] = ("run", "input", "median", "output"),
) -> str:
    """Annotated TOML of ``settings``: same layout as the input files, so it can be fed back with ``--config``.

    ``provenance`` (command, software version, time) and ``result`` (output files, counts) are appended as
    the informational tables ``[provenance]`` and ``[result]``, which :func:`load_settings` ignores.
    """
    lines = [
        f"# {title}",
        "# Re-run with:  make median CONFIG=<this file>   (or: python -m amgeo_sd median -c <this file>)",
        "",
    ]
    for table, values in ((t, v) for t, v in settings.to_dict().items() if t in tables):
        docs = FIELD_DOCS[table]
        lines += _table(table, [(k, v, docs.get(k, "")) for k, v in values.items()], TABLE_DOCS[table])
    for name, header, values in (
        ("provenance", "How this file was produced (informational).", provenance),
        ("result", "What the run produced (informational).", result),
    ):
        if values:
            lines += _table(name, [(k, v, "") for k, v in values.items()], header)
    return "\n".join(lines).rstrip() + "\n"
