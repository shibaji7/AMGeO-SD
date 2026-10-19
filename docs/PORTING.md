# Porting map: legacy `_bkp/` -> `amgeo_sd`

Legacy code is frozen in [`_bkp/`](../_bkp/README.md). Every legacy feature gets a new home, a
`make` target, and a test on the real data in `tests/data` (bks = mid-latitude, gbr = high-latitude,
2025-02-08). Status: **done**, **todo**.

## 1. Data model and I/O

| Legacy | New | Status |
|---|---|---|
| `sd/get_fit_data.py` `Gate`, `Beam`, `Scan` | `amgeo_sd/models.py` (`Beam`, `Scan` dataclasses; `Gate` dropped, was only used inside the median filter) | done |
| `FetchData` (file glob, `fetch_data`, `convert_to_pandas`, `scans_to_pandas`) | `amgeo_sd/io/fitacf.py` (`FitacfReader`, `find_fitacf_files`, `beams_to_dataframe`, `scans_to_dataframe`) | done |
| `FetchData.pandas_to_beams`, `pandas_to_scans` (DataFrame back to scans; merges the first two scans when `len(s0)+len(s1)==len(s2)`) | `io/fitacf.py` `dataframe_to_scans` | todo |
| `tools/fit_records.py` `PrintFitRec.beam_summary` (scan period, themis beam, `s_mode`) | `io/summary.py` `scan_info()`; `make data-info` already prints the basics | todo |
| `PrintFitRec.print_fit_record` (text dump with geo/mag coordinates) | `io/summary.py` `write_fit_record()` | todo |
| `config/conf.ini` (`SD_DATA`, `BASE_DIR`, figure limits), `utils.get_config` | `$AMGEO_SD_DATA` / `--root` for input; output dir as an argument (`OUT`) | partly (input done) |
| `config/radar.csv`, `utils.get_radar` | `amgeo_sd/data/radars.csv` + `radars.get_radar()` | todo |

## 2. Geometry (davitpy-derived)

| Legacy | New | Notes |
|---|---|---|
| `tools/geoPack.py` | `amgeo_sd/geo/geopack.py` | port as-is, then compare with pydarn/aacgmv2 results |
| `tools/rad_fov.py` `CalcFov` | `amgeo_sd/geo/fov.py` | needs `pydarn.read_hdw_file(abbrv, date)` (signature changed in pydarn 4) |
| `tools/model_vheight.py` | `amgeo_sd/geo/vheight.py` | |
| `utils.geolocate_radar_fov`, `fitacf_geopos.py` | `geo/fov.py` + `make geopos` | |
| `clustering_algo.get_mean_sza` | `geo/sun.py` | pysolar 0.13 |

## 3. Algorithms

| Legacy | New | Status |
|---|---|---|
| `sd/boxcar_filter.py` `Filter` (3x3x3 weighted median, conv/KDE ground-scatter flags, per-scan DBSCAN) | `amgeo_sd/filters/median.py` | todo |
| `sd/analysis.py` `Simulate` (+ `fitacf_amgeo.py`) | `amgeo_sd/pipelines/median_filter.py` -> `make median` | todo |
| `clustering_algo.BeamGateTimeFilter`, `BeamGateFilter`, `ScatterTypeDetection` (+ `fitacf_amgeo_cluster.py`) | `amgeo_sd/clustering/beam_gate.py`, `scatter_type.py` -> `make cluster` | todo |
| `clustering_algo.DBScan` (GMM + DBSCAN, Ribiero threshold) (+ `fitacf_amgeo_dbcluster.py`) | `amgeo_sd/clustering/dbscan.py` -> `make dbscan` | todo |
| `utils.ribiero_gs_flg`, `_run_riberio_threshold*`, `smooth`, `get_gridded_parameters` | `amgeo_sd/scatter.py`, `amgeo_sd/grid.py` | todo |
| `Beam.gs_estimation` (Sundeen / Blanchard criteria) | `amgeo_sd/scatter.py` | todo |

## 4. Output

| Legacy | New |
|---|---|
| `utils.to_xarray`, `Simulate.to_netcdf`, `BeamGateTimeFilter.repackage_to_netcdf`, `save_cmd` | `amgeo_sd/export.py` (xarray -> NetCDF4, params.json, script.cmd) |
| `utils.build_base_folder_structure`, `os.system("mkdir/gzip/rm -rf")` | `pathlib` + `gzip` module |

## 5. Plotting (four overlapping copies today)

`get_cluster_cmap` exists 4x, RTI plot classes 3x (`plotlib`, `fan_rti`, `sd_plots`, plus `rti.py`),
`MasterPlotter` 2x, `FanPlots` 2x. Merge into one `amgeo_sd/plotting/` package:

| Legacy | New |
|---|---|
| `sd/fan_rti.py`, `sd/plotlib.py` (`FanPlots`, `RangeTimeIntervalPlot`, `FoV`) , `sd/sd_plots.py`, `rti.py` | `plotting/rti.py`, `plotting/fan.py` -> `make plot-rti`, `make plot-fov` |
| `tools/sd_carto.py` `SDCarto` (needs `descartes`, unmaintained) | `plotting/carto.py`, patches via matplotlib/shapely directly |
| `sd/scan_cluster_plots.py`, `plotlib.InvestigativePlots`, `MovieMaker`, beam-gate/cluster statistics plots | `plotting/cluster.py`, `plotting/movie.py` (`cv2` only for the movie) |
| `sd/plot_processed_data.py`, `fitacf_plots.py`, `fitacf_utils.py` | `pipelines/plot_outputs.py` |
| `fitacf_polar.py` (velocity histogram by range), `SuperDARN_3D_Scatter_Volume_Plots.ipynb` | `examples/` |

## 6. Dropped / archive only

| Legacy | Why |
|---|---|
| `extra_codes/geoloc.py` | needs davitpy on Python 2.7; replaced by `geo/fov.py` |
| `extra_codes/job.py`, `simulate.py` | import `Process2Movie` / `printFitRec`, which do not exist in `analysis.py`; the new CLI covers them |
| `extra_codes/bs_analysis.py` | SAIS backscatter study using `spacepy`; port later to `examples/` if still wanted |
| `amgeo.yml` | py3.7 lock file; replaced by `environment.yml` |

## 7. Known defects to fix while porting

* `utils.smooth`: `numpy.ones` (undefined name) and `eval()` for the window.
* `BeamGateFilter.doFilter`: `DataFrame.append` (removed in pandas 2); `stats.median_absolute_deviation` (removed in SciPy 1.9; use `median_abs_deviation`).
* `get_gridded_parameters`: positional `pivot(x, y)` is keyword-only in pandas 2.
* `rti.py` runs the whole analysis at import time; `fitacf_amgeo_dbcluster` hard-codes 2015-03-17 FoV times.
* Shell-outs (`os.system("mkdir -p")`, `gzip`, `rm -rf {out_dir}*`) and cwd-relative paths.
* `plt.style.use(["science", "ieee"])` needs SciencePlots (not installed).
* `datetime.utcfromtimestamp` (deprecated), `ColorbarBase`, `descartes`.

## 8. Keeping behaviour identical

To prove the port preserves results, run the **legacy** pipeline once on the test radar-days in a
py3.7 env built from `_bkp/amgeo.yml` and save its NetCDF/CSV outputs as golden files; each ported
stage is then compared against them. This needs the old env; see the open question in the hand-off.
