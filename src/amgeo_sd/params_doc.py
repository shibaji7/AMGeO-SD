"""Generate ``docs/PARAMETERS.md`` from the settings classes in :mod:`amgeo_sd.config`.

Keys, types and defaults come from the dataclasses and the one-line descriptions from ``FIELD_DOCS``, so the table cannot drift
from the code; ``PARAM_META`` adds what the dataclasses do not know (allowed values, the command-line flag, the ``make``
variable). ``make docs`` rewrites the file; ``tests/test_docs.py`` fails when it is stale or names a flag or variable that does
not exist.
"""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import MISSING, fields

from amgeo_sd.config import FIELD_DOCS, TABLE_DOCS, Settings, _toml_value

#: Which commands read each table.
TABLE_USED_BY = {
    "run": "`median`, `dbscan`",
    "input": "`median`, `dbscan`",
    "median": "`median`",
    "dbscan": "`dbscan`",
    "geometry": "`median`, `dbscan`, `plot fov`",
    "output": "`median`, `dbscan`",
    "plot": "`plot rti`, `plot fov`",
}

#: (allowed values / unit, command-line flag, make variable) per key; ``-`` = none.
PARAM_META: dict[tuple[str, str], tuple[str, str, str]] = {
    ("run", "rad"): ("three-letter radar code, e.g. `bks`, `gbr`; required", "`-r`, `--rad`", "`RAD`"),
    ("run", "start"): ("UT; TOML datetime or ISO string; required", "`-s`, `--start`", "`START`"),
    ("run", "end"): ("UT, not before `start`; required", "`-e`, `--end`", "`END`"),
    ("run", "sim_id"): ("any folder name", "`-sid`, `--sim-id`", "`SID`"),
    ("run", "out"): ("directory (created); results in `<out>/<sim_id>/<rad>/`", "`-o`, `--out`", "`OUT`"),
    ("run", "jobs"): ("whole number >= 1; `median` only", "`-j`, `--jobs`", "`JOBS`"),
    ("input", "paths"): (
        "files or directories (searched recursively); empty = archive lookup",
        "`-i`, `--input`",
        "`INPUT`",
    ),
    ("input", "root"): ("directory laid out `<year>/<ftype>/<rad>/<YYYYMMDD>.*`", "`--root`", "-"),
    ("input", "ftype"): ("`fitacf` or `fitacf3`; archive lookup only", "`--ftype`", "-"),
    ("median", "thresh"): ("(0, 1]", "`-th`, `--thresh`", "`THRESH`"),
    ("median", "pth"): ("(0, 1)", "`-pth`, `--pth`", "`PTH`"),
    ("median", "lth"): ("0 <= lth < uth <= 1", "`-lth`, `--lth`", "`LTH`"),
    ("median", "uth"): ("0 <= lth < uth <= 1", "`-uth`, `--uth`", "`UTH`"),
    ("median", "kde"): ("`true` / `false`", "`--kde`, `--no-kde`", "-"),
    ("median", "cluster"): ("`true` / `false`", "`--cluster`, `--no-cluster`", "-"),
    ("median", "eps"): ("> 0, in (beam, gate) units", "`--eps`", "-"),
    ("median", "min_samples"): ("whole number >= 1", "`-ms`, `--min-samples`", "-"),
    ("median", "edge_gates"): ("`legacy` or `missing`", "`--edge-gates`", "-"),
    ("dbscan", "eps"): ("> 0, in scaled (gate, scan) units", "`-eps`, `--eps`", "-"),
    ("dbscan", "eps_g"): ("> 0", "`-eps_g`, `--eps-g`", "-"),
    ("dbscan", "eps_s"): ("> 0", "`-eps_s`, `--eps-s`", "-"),
    ("dbscan", "min_samples"): ("whole number >= 1", "`-ms`, `--min-samples`", "-"),
    ("dbscan", "tb_mult"): ("whole number >= 1", "`-tm`, `--tb-mult`", "-"),
    ("dbscan", "velocity_threshold"): ("m/s, > 0", "`-vth`, `--velocity-threshold`", "-"),
    ("dbscan", "min_gate"): ("range gate number >= 0", "`--min-gate`", "-"),
    ("dbscan", "gmm"): ('`""`, `"A"`, `"T"` or a beam list such as `"3,7"`', "`-gmm`, `--gmm`", "`GMM`"),
    ("dbscan", "n_clusters"): ("whole number >= 1", "`-nc`, `--n-clusters`", "-"),
    ("dbscan", "cov"): ("`full`, `tied`, `diag`, `spherical`", "`-cov`, `--cov`", "-"),
    ("dbscan", "n_data_th"): ("whole number >= 1", "`-gth`, `--n-data-th`", "-"),
    ("geometry", "backend"): ("`legacy` or `pydarn`", "`--geometry`", "`GEOMETRY`"),
    ("geometry", "altitude"): ("km, in (0, 2000)", "`--altitude`", "-"),
    ("output", "netcdf"): ("`true` / `false`; at least one output must be on", "`--netcdf`, `--no-netcdf`", "-"),
    ("output", "fitacf"): ("`true` / `false`; at least one output must be on", "`--fitacf`, `--no-fitacf`", "-"),
    ("plot", "inputs"): (
        "files or directories of one kind (all `.nc` or all fitacf-type files)",
        "`-i`, `--input`",
        "`INPUT`",
    ),
    ("plot", "compare"): ("same kinds as `inputs`; RTI only", "`--compare`", "`COMPARE`"),
    ("plot", "rad"): ("three-letter radar code", "`-r`, `--rad`", "`RAD`"),
    ("plot", "beam"): ("beam number", "`-b`, `--beam`", "`BEAM`"),
    ("plot", "start"): ("UT", "`-s`, `--start`", "`START`"),
    ("plot", "end"): ("UT", "`-e`, `--end`", "`END`"),
    ("plot", "time"): ("UT; FoV only", "`-t`, `--time`", "`TIME`"),
    ("plot", "params"): (
        "`v w_l p_l v_mad`, flags `gflg gflg_conv gflg_kde gflg_ribiero`, `cluster_tag`",
        "`-p`, `--params`",
        "`PARAMS`",
    ),
    ("plot", "flag"): ('a flag column or `""`; RTI compare only', "`--flag`", "-"),
    ("plot", "coords"): ("`geo` or `aacgmv2`; FoV only", "`--coords`", "`COORDS`"),
    ("plot", "projection"): ("`auto`, `polar`, `ortho`; FoV only", "`--projection`", "-"),
    ("plot", "max_gate"): ("whole number; FoV only", "`--max-gate`", "-"),
    ("plot", "terminator"): ("`true` / `false`; geo coordinates only", "`--terminator`, `--no-terminator`", "-"),
    ("plot", "out"): ("directory (created)", "`-o`, `--out`", "`FIGDIR`"),
    ("plot", "name"): ("file name without extension", "`--name`", "-"),
    ("plot", "format"): ("`png`, `pdf`, `svg`", "`--format`", "-"),
    ("plot", "dpi"): ("whole number", "`--dpi`", "-"),
}

#: Default text for keys whose dataclass default is ``None``.
NONE_DEFAULTS = {
    ("run", "rad"): "required",
    ("run", "start"): "required",
    ("run", "end"): "required",
    ("input", "root"): "`$AMGEO_SD_DATA`",
    ("plot", "rad"): "from the files",
    ("plot", "start"): "all data",
    ("plot", "end"): "all data",
    ("plot", "time"): "first scan",
}

#: Make variables: name -> (meaning, targets that use it). ``-`` entries in ``PARAM_META`` point here.
MAKE_VARIABLES = {
    "CONDA_ENV": ("conda environment every command runs in (default `amgeo`)", "all"),
    "CONFIG": ("TOML parameter file passed as `-c`", "`median`, `dbscan`, `plot-rti`, `plot-fov`, `config-show`"),
    "RAD": ("radar code", "`median`, `dbscan`, `plot-*`, `analyse-day`"),
    "START": ("start time `YYYY-MM-DDTHH:MM` (UT)", "`median`, `dbscan`, `plot-*`"),
    "END": ("end time (UT)", "`median`, `dbscan`, `plot-*`"),
    "INPUT": (
        "input files or directories (`-i`); for `plot-*` the data to plot",
        "`median`, `dbscan`, `plot-*`, `analyse-day`",
    ),
    "OUT": ("output base directory (`-o`)", "`median`, `dbscan`, `out-clean`"),
    "SID": ("run id (`-sid`)", "`median`, `dbscan`"),
    "JOBS": ("worker processes (`-j`)", "`median`"),
    "THRESH": ("median threshold (`-th`)", "`median`"),
    "PTH": ("Beta cut (`-pth`)", "`median`"),
    "LTH": ("lower probability bound (`-lth`)", "`median`"),
    "UTH": ("upper probability bound (`-uth`)", "`median`"),
    "GMM": ("Gaussian-mixture beams (`-gmm`)", "`dbscan`"),
    "GEOMETRY": ("`legacy` or `pydarn` (`--geometry`)", "`median`, `dbscan`, `plot-*`"),
    "EXTRA": (
        'any further command-line flags, e.g. `EXTRA="--no-kde --edge-gates missing"`',
        "`median`, `dbscan`, `plot-*`, `config-show`",
    ),
    "COMPARE": ("second dataset stacked in the RTI (`--compare`)", "`plot-rti`"),
    "BEAM": ("beam number (`-b`); for `analyse-day` the beam of the figures", "`plot-rti`, `analyse-day`"),
    "TIME": ("FoV time (`-t`)", "`plot-fov`"),
    "PARAMS": ("plotted parameters (`-p`)", "`plot-rti`, `plot-fov`"),
    "COORDS": ("`geo` or `aacgmv2` (`--coords`)", "`plot-fov`"),
    "FIGDIR": ("figure directory (`-o`)", "`plot-rti`, `plot-fov`"),
    "DAY": ("UT day analysed, default `2025-02-08`", "`analyse-day`"),
    "WORK": ("analysis working directory, default `data/analysis`", "`analyse-day`"),
    "DATA": ("radar-day folders summarised, default all of `tests/data/*/`", "`data-info`, `data-clean`"),
}

OUTPUT_FILES = [
    (
        "`<rad>_<start>_<end>.pp.nc`",
        "`median`, `dbscan`",
        "fitacf++ NetCDF4 (`[output] netcdf`); `<start>`/`<end>` as `YYYYMMDDHHMM`",
    ),
    (
        "`<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pp.bz2`",
        "`median`, `dbscan`",
        "fitacf++ as bz2 DMAP (`[output] fitacf`); time = `start`. Files named `.fitacf.pl.bz2` by earlier versions are still read",
    ),
    (
        "`params.toml`",
        "`median`, `dbscan`",
        "all parameters, annotated, plus `[provenance]` and `[result]`; valid as `--config`",
    ),
    ("`script.cmd`", "`median`, `dbscan`", "the command line"),
    (
        "`<name>.png` / `.pdf` / `.svg`",
        "`plot`",
        "default name `rti_<rad>_b<beam>_<date>_<time>_<params>` or `fov_<rad>_<date>_<time>_<params>_<coords>`",
    ),
]


def _md(text: str) -> str:
    """Make free text safe inside a Markdown table cell (``<out>`` would otherwise be read as an HTML tag)."""
    return html.escape(text, quote=False).replace("|", "\\|")


def _type_name(annotation: str) -> str:
    return (
        annotation.replace("dt.datetime", "datetime")
        .replace("list[str]", "list of str")
        .replace(" | None", "")
        .replace("None | ", "")
    )


def _default(settings: Settings, table: str, f) -> str:
    if f.default is not MISSING:
        value = f.default
    elif f.default_factory is not MISSING:
        value = f.default_factory()
    else:  # pragma: no cover - every settings field has a default
        value = None
    if value is None:
        return NONE_DEFAULTS.get((table, f.name), "none")
    if isinstance(value, (dt.datetime, bool, int, float, str, list, tuple)):
        text = _toml_value(value)
        return f"`{text}`"
    return f"`{value}`"  # pragma: no cover


def parameters_markdown() -> str:
    """The full text of ``docs/PARAMETERS.md``."""
    settings = Settings()
    out = [
        "<!-- Generated by `make docs` (amgeo_sd.params_doc); do not edit by hand. -->",
        "# Parameter reference",
        "",
        "Every setting of `amgeo_sd`, with type, default, allowed values and how to set it. "
        "Back to the [README](../README.md) or the [usage guide](USAGE.md); the Python functions are in the [API reference](API.md); "
        "example folders: [`configs/`](../configs/README.md), [`examples/`](../examples/median_day_analysis/README.md), "
        "[`tests/`](../tests/README.md), [`tests/data/`](../tests/data/README.md), [`_bkp/`](../_bkp/README.md).",
        "",
        "## How parameters are set",
        "",
        "Precedence, lowest to highest: **built-in default < TOML file < command-line flag / `make` variable**. Unknown keys and values outside the "
        "allowed range are errors, so a typo cannot silently fall back to a default.",
        "",
        "| Way | Example |",
        "|---|---|",
        "| TOML file | `make median CONFIG=configs/median_bks_demo.toml` (see [`configs/default.toml`](../configs/default.toml), every key with a comment) |",
        "| command line | `python -m amgeo_sd median -c cfg.toml --thresh 0.5 --no-kde` |",
        "| `make` variable | `make median RAD=bks START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/bks_20250208 THRESH=0.5` |",
        "| check what will run | `make config-show CONFIG=cfg.toml THRESH=0.5` (or `python -m amgeo_sd median --print-config ...`) |",
        "",
        "Every run writes an annotated `params.toml` next to its outputs; it is itself a valid config (`make median CONFIG=<run>/params.toml` repeats the run).",
        "",
    ]
    for table in settings.to_dict():
        meta_rows = []
        for f in fields(getattr(settings, table)):
            allowed, cli, make = PARAM_META[(table, f.name)]
            meta_rows.append(
                f"| `{f.name}` | {_type_name(str(f.type))} | {_default(settings, table, f)} | {allowed} | {_md(FIELD_DOCS[table][f.name])} | {cli} | {make} |"
            )
        out += [
            f"## `[{table}]`",
            "",
            f"{TABLE_DOCS[table]} Used by {TABLE_USED_BY[table]}.",
            "",
            "| Key | Type | Default | Allowed | Description | Command line | `make` |",
            "|---|---|---|---|---|---|---|",
            *meta_rows,
            "",
        ]
    out += [
        "## `make` variables",
        "",
        "Variables are given on the command line (`make median RAD=bks ...`); an empty variable is not passed on, so the TOML file or the default applies.",
        "",
        "| Variable | Meaning | Used by |",
        "|---|---|---|",
        *[f"| `{name}` | {meaning} | {used} |" for name, (meaning, used) in MAKE_VARIABLES.items()],
        "",
        "## Output files",
        "",
        "| File | Written by | Content |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in OUTPUT_FILES],
        "",
        "Results go to `<out>/<sim_id>/<rad>/`. The meaning of the variables inside the files is in the [usage guide](USAGE.md).",
        "",
        "## Plot parameter names",
        "",
        "| Name | Kind | Notes |",
        "|---|---|---|",
        "| `v`, `w_l`, `p_l` | numeric | line-of-sight velocity [m/s], spectral width [m/s], power [dB] |",
        "| `v_mad` | numeric | median filter output: local spread of the velocity [m/s] |",
        "| `gflg` | flag | fitacf flag for raw data; for the median model the legacy v-w rule (not recommended) |",
        "| `gflg_conv`, `gflg_kde` | flag | median model; recommended |",
        "| `gflg_ribiero` | flag | DBSCAN model: -1 unclassified, 0 ionospheric, 1 ground |",
        "| `cluster_tag` | categorical | DBSCAN model cluster label, unique within a beam only |",
        "",
    ]
    return "\n".join(out)
