# Using amgeo_sd

Everything is driven from `make` (run `make help`), from the TOML parameter files in `configs/`, or from Python.
What is not ported yet is listed in [PORTING.md](PORTING.md).

## 1. Make targets

| Target | What it does |
|---|---|
| `make env` / `env-update` / `env-remove` / `install` | conda env `amgeo` (Python 3.12) from `environment.yml`; editable install of the package |
| `make data-unpack` / `data-info` / `data-clean` | unzip `tests/data/archives/*.zip`; summarise beams / scans / echoes per radar-day; delete unpacked folders that can be restored |
| `make median` / `median-demo` | median-filter model: fitacf -> fitacf++ (`.nc` and `.fitacf.pl.bz2`) |
| `make plot-rti` / `plot-fov` / `plot-demo` | RTI and field-of-view figures from fitacf or fitacf++ files |
| `make config-show` | print the resolved median parameters without running |
| `make test` / `test-unit` / `test-real` / `test-cov` | test suite (real-data tests skip when `tests/data` is empty) |
| `make lint` / `format` | ruff checks / formatting |
| `make out-clean` / `clean` | delete pipeline results (`OUT`, default `data/outputs`) / caches and build files |

Variables (`CONFIG RAD START END INPUT OUT SID JOBS THRESH PTH LTH UTH EXTRA COMPARE BEAM TIME PARAMS COORDS FIGDIR`) are
passed as `make median RAD=bks START=2025-02-08T03:00 ...`; anything else goes in `EXTRA="--no-kde"`.

## 2. Parameters: TOML files

Precedence: built-in defaults < TOML file < command-line / make variables. Unknown keys are errors.
[`configs/default.toml`](../configs/default.toml) lists **every** parameter with its default and a comment:

```toml
[run]     rad, start, end, sim_id, out, jobs
[input]   paths, root, ftype
[median]  thresh, pth, lth, uth, kde, cluster, eps, min_samples
[output]  netcdf, fitacf
[plot]    inputs, compare, rad, beam, start, end, time, params, flag, coords, projection, max_gate, terminator, out, name, format, dpi
```

Every run writes an **annotated `params.toml`** (all parameters with comments, plus `[provenance]` and `[result]`) and
`script.cmd` next to its outputs. `params.toml` is itself a valid config: `make median CONFIG=<run>/params.toml` repeats the run.
`make config-show CONFIG=... JOBS=8` shows what a combination resolves to.

## 3. Median-filter model (fitacf -> fitacf++)

```bash
make median CONFIG=configs/median_bks_demo.toml                      # 1 h of bks from tests/data
make median RAD=gbr START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/gbr_20250208 JOBS=4
make median CONFIG=configs/median_gbr_demo.toml EXTRA="--no-kde"   # skip the slow Beta fit
```

Input: fitacf 2.0 / 2.5 / 3.0 files (`.bz2`, `.gz` or plain), given as files/directories (`INPUT`, `[input] paths`) or looked up in
an archive `<root>/<year>/<ftype>/<rad>/<YYYYMMDD>.*` (`[input] root` or `$AMGEO_SD_DATA`).

Algorithm (`amgeo_sd.filters.MedianFilter`): for every cell of the middle one of three consecutive scans, the 27 neighbours
(scan -1..+1, beam -1..+1, gate -1..+1) are weighted by `[[1,2,1],[2,3,2],[1,2,1]] / [[2,3,2],[3,5,3],[2,3,2]] / ...`. The cell
survives if occupied-neighbour weight / existing-neighbour weight >= `thresh`; survivors get weighted medians of `v w_l p_l`, `v_mad`
and the flags

| variable | meaning |
|---|---|
| `gflg` | 0 ionospheric if median `w_l > -3 v + 90`, else 1 ground |
| `gflg_conv` | weighted fraction of ground-scatter neighbours: <= `lth` 0, >= `uth` 1, else 2 (undetermined) |
| `gflg_kde` | Beta-distribution fit of the neighbour flags, `1 - cdf(pth)` mapped like `gflg_conv` |
| `clabel` | per-scan DBSCAN label in (beam, gate) space, -1 = noise |

Speed: about 0.15 ms per echo with KDE; an hour of bks takes ~10 s.

### Output files (in `<out>/<sim_id>/<rad>/`)

| file | content |
|---|---|
| `<rad>_<start>_<end>.fitacf++.nc` | NetCDF4: `gdlat gdlon gdazm` (beams x gates); per beam `bmnum noise.sky tfreq scan nrang frang rsep time mppul intt.us intt.sc` (scan_num x sound); per echo `v w_l p_l gflg gflg_conv gflg_kde v_mad clabel` (scan_num x sound x gates, NaN = no echo) |
| `<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pl.bz2` | fitacf++ as DMAP, one record per surviving beam (see below) |
| `params.toml`, `script.cmd` | annotated parameters and the command line |

**`.fitacf.pl.bz2`**: a bz2 DMAP stream. Each record is the original fitacf record (all standard fields: header values, `pwr0`,
`ltab`, ...) with `slist v w_l p_l gflg` replaced by the filtered values and `v_e` NaN. Other per-echo fitacf arrays (`qflg`, `p_s`, `elv`, ...)
keep their value where the gate had an echo and are NaN / 0 where the filter added a cell. Added arrays: `v_mad`, `gflg_conv`, `gflg_kde`
(-1 = not computed), `clabel`; strings `amgeo.model`, `amgeo.version`; `origin.command` / `origin.time` record the processing.
The scan flag is re-set on the first kept beam of every scan. Because of the extra fields, `pydarnio.read_fitacf` rejects these files;
read them with `amgeo_sd.io.read_fitacf_file` (falls back to generic DMAP) or `pydarnio.read_dmap`. The raw-data archive lookup (`root`) skips `*.fitacf.pl.bz2`, so filtered and raw data are never mixed; the loaders accept them when given explicitly or by directory.

### Differences from the legacy output

* scans with start time in `[start, end]` are written; the legacy run emitted the neighbouring scans and left the last two as empty placeholders;
* the KDE jitter uses a seeded generator per scan (reproducible, independent of `jobs`); a failed Beta fit gives 2 instead of an exception;
* filtered beams with no surviving echo are dropped (as after the legacy DBSCAN regrouping); `v_e` is NaN;
* `clabel` is also written; `scan_stime` / `scan_etime` are coordinates of `scan_num` rather than separate dimensions; `frang`, `rsep` are stored;
* the scan period is the median spacing of all scan-start beams (the legacy code differenced the last two starts among beams with echoes);
* range-cell geometry uses the data's `frang` / `rsep` (legacy: always 180 / 45 km; identical for bks and gbr);
* the radar geometry is the legacy `CalcFov` (standard virtual-height model), which differs by 20-25 km from pydarn 4's fixed-300 km field of view.

## 4. Plots (fitacf and fitacf++)

All plots read any supported input through `amgeo_sd.io.load_frame`: raw fitacf 2.0 / 2.5 / 3.0, `*.fitacf.pl.bz2`, or `*.fitacf++.nc`
(files or directories), and produce the same echo table, so a raw and a filtered dataset can be drawn identically.

```bash
make plot-rti INPUT=tests/data/bks_20250208 COMPARE=data/outputs/demo/bks/bks_202502080300_202502080400.fitacf++.nc \
     BEAM=12 START=2025-02-08T03:00 END=2025-02-08T04:00 PARAMS=v          # raw above, fitacf++ below (+ gflg panels)
make plot-rti INPUT=data/outputs/demo/bks/*.fitacf++.nc BEAM=12 PARAMS="v w_l p_l gflg gflg_conv v_mad"
make plot-fov INPUT=data/outputs/demo/bks/20250208.0300.00.bks.fitacf.pl.bz2 TIME=2025-02-08T03:30 PARAMS="v gflg_conv" COORDS=aacgmv2
make plot-demo                                                              # runs the demo filter, then all of the above
```

* **RTI** (`plot rti`): one panel per parameter for one beam. Numeric: `v w_l p_l v_mad` (or `[plot]` ranges via the API); flags: `gflg gflg_conv gflg_kde gflg_ribiero`
  (IS red, GS blue, US black, undetermined green); `cluster_tag`. With `compare`, the same beam of each input is stacked on identical axes and colour scale.
* **FoV** (`plot fov`): the scan nearest `time`, each echo drawn as its range cell on a cartopy map; ground scatter (`gflg == 1`) is gray, flag
  parameters use the IS / GS classes. `coords` is `geo` or `aacgmv2` (cells, radar and coastlines are converted; 300 km for data, 0 km for coastlines).
  `projection` is `auto` (orthographic around the field of view, polar stereographic above 70 deg), `polar` or `ortho`.
* Figures go to `data/figures/` (`FIGDIR`); `[plot] format` is `png`, `pdf` or `svg`.

Python:

```python
import datetime as dt
from amgeo_sd.io import load_frame
from amgeo_sd.plotting import plot_rti, plot_rti_compare, plot_fan

raw, _ = load_frame(["tests/data/bks_20250208"], dt.datetime(2025, 2, 8, 3), dt.datetime(2025, 2, 8, 4))
pp, meta = load_frame(["data/outputs/demo/bks/bks_202502080300_202502080400.fitacf++.nc"])
plot_rti_compare({"fitacf 2.5": raw, "fitacf++": pp}, beam=12, param="v", path="rti.png", rad="bks", flag="gflg")
plot_fan(pp, "bks", dt.datetime(2025, 2, 8, 3, 30), params=("v", "gflg_conv"), path="fan.png")
```

Legacy -> new names: `RangeTimeIntervalPlot.addParamPlot / addCluster / addGSIS` -> `add_param / add_clusters / add_gsis`;
`FanPlots.plot_fov / plot_layfovs / add_instrument` are unchanged; the `sdcarto` GeoAxes class is replaced by functions inside `FanPlots`.

## 5. Python API

```python
from amgeo_sd.io import FitacfReader, load_scans, scans_to_dataframe, write_fitacfpp, scans_to_records
from amgeo_sd.filters import MedianFilter
from amgeo_sd.geo import radar_fov
from amgeo_sd.pipelines import run_median_filter, run_plot
from amgeo_sd.config import load_settings
```

* `FitacfReader(rad, start, end, files=... | root=...)`: `.records()` (raw dicts), `.beams()`, `.scans()`. `load_scans` / `load_frame` accept any supported file.
* `Beam` (one sounding: header attributes, per-echo arrays `slist v v_e w_l p_l gflg`, `extra` dict, `fit_version`) and `Scan` live in `amgeo_sd.models`.
* `MedianFilter(...).filter_scans(scans, n_jobs=4)` returns the filtered scans (each interior scan once).
* `radar_fov("bks", date)` returns cell-corner `glat glon gazm`; `hardware("bks", date)` the pydarn hardware record.
* `load_settings(path, overrides)` builds the validated `Settings`; `dump_settings_toml` writes the annotated TOML.
* Column names where a dot was awkward: `noise.sky -> noise_sky`, `intt.sc -> intt_sc`, `intt.us -> intt_us` (NetCDF keeps the dotted names).
