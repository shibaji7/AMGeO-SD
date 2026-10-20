# Using amgeo_sd

Everything is driven from `make` (run `make help`), from the TOML parameter files in `configs/`, or from Python.
Every parameter is described in the **[parameter reference](PARAMETERS.md)**; what is not ported yet is listed in [PORTING.md](PORTING.md).
Back to the [README](../README.md) or the [documentation index](README.md).

## 1. Make targets

| Target | What it does |
|---|---|
| `make env` / `env-update` / `env-remove` / `install` | conda env `amgeo` (Python 3.12) from `environment.yml`; editable install of the package |
| `make data-unpack` / `data-info` / `data-clean` | unzip `tests/data/archives/*.zip`; summarise beams / scans / echoes per radar-day; delete unpacked folders that can be restored |
| `make median` / `median-demo` | median-filter model: fitacf -> fitacf++ (`.nc` and `.fitacf.pp.bz2`) |
| `make analyse-day RAD=bks` | whole-day comparison of the median filter and the DBSCAN model: runs both, prints retention / flag tables, draws figures (`examples/median_day_analysis/`) |
| `make dbscan` / `dbscan-demo` | DBSCAN / Gaussian-mixture model: raw echoes labelled with clusters and the Ribiero scatter flag -> fitacf++ |
| `make plot-rti` / `plot-fov` / `plot-demo` | RTI and field-of-view figures from fitacf or fitacf++ files |
| `make config-show` | print the resolved median parameters without running |
| `make test` / `test-unit` / `test-real` / `test-cov` | test suite (real-data tests skip when `tests/data` is empty) |
| `make lint` / `format` | ruff checks / formatting |
| `make docs` | regenerate [`PARAMETERS.md`](PARAMETERS.md) from the settings classes |
| `make out-clean` / `clean` | delete pipeline results (`OUT`, default `data/outputs`) / caches and build files |

Variables (`CONFIG RAD START END INPUT OUT SID JOBS THRESH PTH LTH UTH GMM GEOMETRY EXTRA COMPARE BEAM TIME PARAMS COORDS FIGDIR DAY WORK`) are
passed as `make median RAD=bks START=2025-02-08T03:00 ...`; anything else goes in `EXTRA="--no-kde"`. What each variable means and which
targets use it is in the [parameter reference](PARAMETERS.md#make-variables).

## 2. Parameters: TOML files

Precedence: built-in defaults < TOML file < command-line / make variables. Unknown keys are errors.
[`configs/default.toml`](../configs/default.toml) lists **every** parameter with its default and a comment, and the
[parameter reference](PARAMETERS.md) gives type, allowed values, flag and make variable for each (the files in
[`configs/`](../configs/README.md) are explained there). The tables and keys:

```toml
[run]     rad, start, end, sim_id, out, jobs
[input]   paths, root, ftype
[median]  thresh, pth, lth, uth, kde, cluster, eps, min_samples, edge_gates
[dbscan]  eps, eps_g, eps_s, min_samples, tb_mult, velocity_threshold, min_gate, gmm, n_clusters, cov, n_data_th
[geometry] backend, altitude
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
survives if occupied-neighbour weight / existing-neighbour weight >= `thresh` (default **0.6**; the legacy code used 0.7); survivors get weighted
medians of `v w_l p_l`, `v_mad` and the flags

| variable | meaning |
|---|---|
| `gflg_kde` **(recommended)** | Beta-distribution fit of the neighbour flags, `1 - cdf(pth)` mapped like `gflg_conv` |
| `gflg_conv` **(recommended)** | weighted fraction of ground-scatter neighbours: <= `lth` 0, >= `uth` 1, else 2 (undetermined) |
| `gflg` *(legacy rule, not recommended)* | 0 ionospheric if median `w_l > -3 v + 90`, else 1 ground |
| `clabel` | per-scan DBSCAN label in (beam, gate) space, -1 = noise |

**Which flag to use.** `gflg_kde` and `gflg_conv` (the neighbours' own fitacf flags, pooled) and, from the DBSCAN model, `gflg_ribiero` agree with physics:
on a whole day of bks and gbr 0.0-0.5% of the echoes they call ground scatter move faster than 50 m/s. The legacy `gflg` rule labels 96% of the bks echoes
ground scatter, and on gbr 18.7% of its "ground scatter" moves faster than 50 m/s (with the default DBSCAN settings, 36.4% of the kept gbr echoes and 18.1% of the bks ones
are ground for this rule but ionospheric for the Ribiero rule; 53% / 41% of them move faster than 25 m/s). It is still written for compatibility and the NetCDF attribute `recommended_flags` says `gflg_kde, gflg_conv`.
`gflg_conv` leaves 28% (bks) to 39% (gbr) of the echoes undetermined (2); `gflg_kde` decides more of them.

**Threshold.** Proxy measured on a whole day with the DBSCAN clusters as "structure" (not ground truth), `thresh` 0.5 / 0.6 / 0.7:

| | noise removed | large clusters kept | medium clusters kept |
|---|---|---|---|
| bks | 87.8% / 96.1% / 99.4% | 95.6% / 91.4% / 83.6% | 84.8% / 71.2% / 47.8% |
| gbr | 86.8% / 95.8% / 99.3% | 92.4% / 85.3% / 71.3% | 86.2% / 74.8% / 55.7% |

0.7 removes nearly all isolated echoes but cuts thin and patchy structure (dense bands get holes, ionospheric fringes disappear); 0.6 is the default compromise,
0.5 keeps patchy ionospheric scatter at the price of more isolated echoes. Reproduce with `make analyse-day RAD=bks`.

**Range edges (`edge_gates`).** The legacy rule counts the neighbours below gate 0 and above the last gate as empty cells (`edge_gates = "legacy"`, the default, so
results match the old code). A cell at the first or last gate can then reach at most 40/55 = 73% of the neighbour weight, so with the 0.7 threshold only a perfectly
filled neighbourhood survives: on bks 17% of the structure echoes at gate 0 and 2% at the last gate were kept, against 33-88% at their neighbours. `edge_gates = "missing"`
leaves such neighbours out of the weight, like missing beams, which changes only gates 0 and `nrang - 1` (interior gates are identical; tested). Use
`[median] edge_gates = "missing"`, `--edge-gates missing` or `EXTRA="--edge-gates missing"`.

Speed: about 0.15 ms per echo with KDE; an hour of bks takes ~10 s.

### Output files (in `<out>/<sim_id>/<rad>/`)

| file | content |
|---|---|
| `<rad>_<start>_<end>.pp.nc` | NetCDF4: `gdlat gdlon gdazm` (beams x gates); per beam `bmnum noise.sky tfreq scan nrang frang rsep time mppul intt.us intt.sc` (scan_num x sound); per echo `v w_l p_l gflg gflg_conv gflg_kde v_mad clabel` (scan_num x sound x gates, NaN = no echo) |
| `<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pp.bz2` | fitacf++ as DMAP, one record per surviving beam (see below) |
| `params.toml`, `script.cmd` | annotated parameters and the command line |

**`.fitacf.pp.bz2`**: a bz2 DMAP stream. Each record is the original fitacf record (all standard fields: header values, `pwr0`,
`ltab`, ...) with `slist v w_l p_l gflg` replaced by the filtered values, `v_e` set to `v_mad` (the weighted median absolute deviation of the
neighbouring velocities, in m/s: a robust local spread, not a fit error; on bks it has a median of 3.8 m/s against 4.0 m/s for the original fit errors)
and `qflg` = 1 for every kept cell. Other per-echo fitacf arrays (`p_s`, `p_l_e`, `elv`, ...)
keep their value where the gate had an echo and are NaN / 0 where the filter added a cell (about 4% of cells). Added arrays: `v_mad`, `gflg_conv`, `gflg_kde`
(-1 = not computed), `clabel`; strings `amgeo.model`, `amgeo.version`; `origin.command` / `origin.time` record the processing.
The scan flag is re-set on the first kept beam of every scan. Because of the extra fields, `pydarnio.read_fitacf` rejects these files;
read them with `amgeo_sd.io.read_fitacf_file` (falls back to generic DMAP) or `pydarnio.read_dmap`. The raw-data archive lookup (`root`) skips `*.fitacf.pp.bz2`, so filtered and raw data are never mixed; the loaders accept them when given explicitly or by directory.

### Differences from the legacy output

* scans with start time in `[start, end]` are written; the legacy run emitted the neighbouring scans and left the last two as empty placeholders;
* the default `thresh` is 0.6 instead of 0.7 and `edge_gates` can treat out-of-range gates as missing (see above); the legacy behaviour is `thresh = 0.7`, `edge_gates = "legacy"`;
* the KDE jitter uses a seeded generator per scan (reproducible, independent of `jobs`); a failed Beta fit gives 2 instead of an exception;
* filtered beams with no surviving echo are dropped (as after the legacy DBSCAN regrouping); `v_e` is NaN in memory and in the `.nc`, `v_mad` in the `.fitacf.pp.bz2`;
* `clabel` is also written; `scan_stime` / `scan_etime` are coordinates of `scan_num` rather than separate dimensions; `frang`, `rsep` are stored;
* the scan period is the median spacing of all scan-start beams (the legacy code differenced the last two starts among beams with echoes);
* range-cell geometry uses the data's `frang` / `rsep` (legacy: always 180 / 45 km; identical for bks and gbr);
* the radar geometry is the legacy `CalcFov` (standard virtual-height model), which differs by 20-25 km from pydarn 4's fixed-300 km field of view.

## 4. DBSCAN / Gaussian-mixture model (raw echoes -> clusters -> fitacf++)

```bash
make dbscan CONFIG=configs/dbscan_bks_demo.toml                       # 1 h of bks, plain DBSCAN
make dbscan CONFIG=configs/dbscan_gbr_demo.toml                       # 1 h of gbr, with the Gaussian-mixture step on every beam
make dbscan RAD=bks START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/bks_20250208 GMM=T EXTRA="-eps 1.5 -ms 4"
make dbscan-demo                                                      # both demos into data/outputs/demo-dbscan/
```

Reads the scans in `[start, end]` and **labels** the raw echoes; nothing is removed or changed. For every (channel, beam):

1. DBSCAN over the echoes of all scans in the plane (range gate / `eps_g`, scan number / `eps_s`) with radius `eps` (default 2) and `min_samples`
   (default 8; `min_samples * tb_mult` on the themis special beam of themis-mode radars) gives `cluster_tag` (-1 = noise).
   The legacy code used `eps = 1`, `min_samples = 5`; see the differences below for why the defaults changed.
   **Tags are unique only within a beam**, so always read them together with `bmnum`.
2. Optional (`gmm`): the beam's velocities are first split into `n_clusters` Gaussian-mixture components (on Box-Cox transformed `|v|`, beams
   with fewer than `n_data_th` echoes stay one component) and DBSCAN runs inside each component. `gmm` is `""` (off), `"A"` (all beams),
   `"T"` (themis beam) or a list such as `"3,7"`. The mixture fragments clusters: on the gbr hour, 20.0% of echoes are noise without it and 60.6% with it.
3. Each cluster is classified by the Ribiero et al. (2011) rule: with R = (echoes faster than `velocity_threshold`) / (echoes slower or equal),
   a cluster is ionospheric when R exceeds a limit that depends on how long it lasts (>14 h: 0.15, >3 h: 0.2, >2 h: 0.33, >1 h: 0.475, shorter: 0.5;
   Burrell et al. 2018 for the short ones), otherwise ground scatter. Echoes in noise and in gates below `min_gate` stay unclassified: `gflg_ribiero`
   is -1 unclassified, 0 ionospheric, 1 ground.

Outputs (`<out>/<sim_id>/<rad>/`): `<rad>_<start>_<end>.pp.nc` with `v w_l p_l gflg` unchanged plus `cluster_tag` and `gflg_ribiero`; a
`.fitacf.pp.bz2` that keeps **every original record** (also beams without echoes, original scan flags and `qflg` / `v_e`) with the two extra arrays and
`amgeo.model = "dbscan"`; an annotated `params.toml` (`[dbscan]` table, result counts) and `script.cmd`. Plot it with
`make plot-rti INPUT=<nc> BEAM=12 PARAMS="v cluster_tag gflg_ribiero"`.

Differences from the legacy code: the defaults are `eps = 2`, `min_samples = 8` (legacy 1 and 5). With the legacy values patchy but continuous structure
falls apart into dozens of 3-12 echo fragments that the Ribiero rule classifies one by one, usually as ground scatter because they are slow. On bks beam 12 this
made cluster 201 (10:57-12:20 UT, directly after the clearly ionospheric cluster 173 on the same gates) and the string of fragments between cluster 23 and 76
(04:00-04:50 UT) ground scatter; with the new defaults 96% / 78% of those echoes are ionospheric, while the long afternoon band stays 99% ground scatter. Whole day, all
beams: noise 22.2% -> 9.7%, clusters 11,077 -> 2,548, ionospheric 15.7% -> 19.4%, unclassified 25.2% -> 15.0%. This is a judgement about continuity, there is no ground
truth; a bridge that is much sparser than about 65% occupied still fragments (raise `eps` / lower `min_samples`, or `-eps 1 -ms 5` restores the legacy behaviour). Lowering
`velocity_threshold` (e.g. 15 m/s) also turns cluster 201 ionospheric but does not link the fragments, and below about 12 m/s it starts to flip the ground-scatter band. Also:
the scaled features are used for clustering only (the legacy code overwrote the `slist` / `scnum` columns with them);
with the mixture, labels of different components no longer collide; a beam with fewer than `n_data_th` echoes no longer crashes; the per-beam
RTI / fan images and the hard-coded 2015 fan times of the old run are replaced by `make plot-rti` / `plot-fov`.

## 5. Geometry

`[geometry] backend` (`--geometry`, `make ... GEOMETRY=`) selects how range cells are located, for the NetCDF `gdlat gdlon gdazm` and the maps:

| backend | |
|---|---|
| `legacy` (default) | the original `CalcFov`: standard virtual-height model. Reproduces the old results exactly (tested against arrays made by the old code) |
| `pydarn` | pydarn 4's `gate2geographic_location` at the fixed height `[geometry] altitude` (default 300 km), electronic beam offset applied. Uses the radar's *current* hardware file, whatever the data date. |

They differ by 0.1-0.6 deg in latitude and up to ~3 deg in longitude at far gates (about 20-25 km). The choice never changes the echoes, only where they are placed;
it is recorded in the NetCDF attributes (`geometry_backend`, `geometry_altitude`) and in `params.toml`.

## 6. Plots (fitacf and fitacf++)

All plots read any supported input through `amgeo_sd.io.load_frame`: raw fitacf 2.0 / 2.5 / 3.0, `*.fitacf.pp.bz2`, or `*.pp.nc`
(files or directories), and produce the same echo table, so a raw and a filtered dataset can be drawn identically.

```bash
make plot-rti INPUT=tests/data/bks_20250208 COMPARE=data/outputs/demo/bks/bks_202502080300_202502080400.pp.nc \
     BEAM=12 START=2025-02-08T03:00 END=2025-02-08T04:00 PARAMS=v          # raw above, fitacf++ below (+ gflg panels)
make plot-rti INPUT=data/outputs/demo/bks/*.pp.nc BEAM=12 PARAMS="v w_l p_l gflg gflg_conv v_mad"
make plot-fov INPUT=data/outputs/demo/bks/20250208.0300.00.bks.fitacf.pp.bz2 TIME=2025-02-08T03:30 PARAMS="v gflg_conv" COORDS=aacgmv2
make plot-demo                                                              # runs the demo filter, then all of the above
```

* **RTI** (`plot rti`): one panel per parameter for one beam. Numeric: `v w_l p_l v_mad` (or `[plot]` ranges via the API); flags: `gflg gflg_conv gflg_kde gflg_ribiero`
  (IS red, GS blue, US black, undetermined green); `cluster_tag`. With `compare`, the same beam of each input is stacked on identical axes and colour scale.
* **FoV** (`plot fov`): the scan nearest `time`, each echo drawn as its range cell on a cartopy map; ground scatter (`gflg == 1`) is gray, flag
  parameters use the IS / GS classes. `coords` is `geo` or `aacgmv2`; `[geometry]` selects the cell geolocation (cells, radar and coastlines are converted; 300 km for data, 0 km for coastlines).
  `projection` is `auto` (orthographic around the field of view, polar stereographic above 70 deg), `polar` or `ortho`.
* Figures go to `data/figures/` (`FIGDIR`); `[plot] format` is `png`, `pdf` or `svg`.

Python:

```python
import datetime as dt
from amgeo_sd.io import load_frame
from amgeo_sd.plotting import plot_rti, plot_rti_compare, plot_fan

raw, _ = load_frame(["tests/data/bks_20250208"], dt.datetime(2025, 2, 8, 3), dt.datetime(2025, 2, 8, 4))
pp, meta = load_frame(["data/outputs/demo/bks/bks_202502080300_202502080400.pp.nc"])
plot_rti_compare({"fitacf 2.5": raw, "fitacf++": pp}, beam=12, param="v", path="rti.png", rad="bks", flag="gflg")
plot_fan(pp, "bks", dt.datetime(2025, 2, 8, 3, 30), params=("v", "gflg_conv"), path="fan.png")
```

Legacy -> new names: `RangeTimeIntervalPlot.addParamPlot / addCluster / addGSIS` -> `add_param / add_clusters / add_gsis`;
`FanPlots.plot_fov / plot_layfovs / add_instrument` are unchanged; the `sdcarto` GeoAxes class is replaced by functions inside `FanPlots`.

## 7. Python API

```python
from amgeo_sd.io import FitacfReader, load_scans, scans_to_dataframe, write_fitacfpp, scans_to_records
from amgeo_sd.filters import MedianFilter
from amgeo_sd.geo import radar_fov
from amgeo_sd.pipelines import run_dbscan_model, run_median_filter, run_plot
from amgeo_sd.clustering import cluster_scans
from amgeo_sd.config import load_settings
```

* `FitacfReader(rad, start, end, files=... | root=...)`: `.records()` (raw dicts), `.beams()`, `.scans()`. `load_scans` / `load_frame` accept any supported file.
* `Beam` (one sounding: header attributes, per-echo arrays `slist v v_e w_l p_l gflg`, `extra` dict, `fit_version`) and `Scan` live in `amgeo_sd.models`.
* `MedianFilter(...).filter_scans(scans, n_jobs=4)` returns the filtered scans (each interior scan once).
* `cluster_scans(scans, scan_info, Dbscan(...))` adds `cluster_tag` / `gflg_ribiero` to the echoes in place.
* `radar_fov("bks", date, backend="legacy" | "pydarn")` returns cell-corner `glat glon gazm`; `hardware("bks", date)` the pydarn hardware record.
* `load_settings(path, overrides)` builds the validated `Settings`; `dump_settings_toml` writes the annotated TOML.
* Column names where a dot was awkward: `noise.sky -> noise_sky`, `intt.sc -> intt_sc`, `intt.us -> intt_us` (NetCDF keeps the dotted names).
