# Porting: what is left

Legacy code is frozen in [`_bkp/`](../_bkp/README.md). This file lists only the legacy features that have **no
replacement yet**; everything already ported is documented in [USAGE.md](USAGE.md) and the module docstrings.
Every remaining item gets a new home under `src/amgeo_sd/`, a `make` target where it is a workflow, a TOML
section, and a test on the real data in `tests/data` (bks = mid-latitude, gbr = high-latitude, 2025-02-08).

## 1. Beam-gate-time clustering model (the last fitacf++ workflow)

| Legacy | New home | Notes |
|---|---|---|
| `sd/clustering_algo.py` `BeamGateTimeFilter`, `BeamGateFilter`, `ScatterTypeDetection` (+ `fitacf_amgeo_cluster.py`): beam-gate DBSCAN per 30-min window, KDE ionospheric/ground classification, cluster tracking in time | `clustering/beam_gate.py`, `clustering/scatter_type.py` -> `make cluster` | needs `utils.smooth` (fix the `numpy.ones` / `eval` bug) |
| `Beam.gs_estimation` (Sundeen / Blanchard 2009 criteria) | `scatter.py` | |
| `BeamGateTimeFilter.repackage_to_netcdf`, `save_data`, per-cluster `.h5`, `beam_gate_clusters.csv`, `beam_gate_time_tracker.json` | `export.py` additions | the `.nc` / `.fitacf.pp.bz2` writers already handle `cluster_tag`, `gflg_ribiero` (used by the DBSCAN model) |

This model gets its own TOML table (like `[median]` and `[dbscan]`), a `params.toml` in the output folder, and
`.nc` + `.fitacf.pp.bz2` outputs through the existing writers; `make cluster`.

## 2. Plotting tools that depend on the clustering models

The RTI and field-of-view plots (all three legacy RTI classes and both fan classes merged) are done. These legacy
plot tools consume the beam-gate-time model's cluster records and still need porting under `src/amgeo_sd/plotting/`:

| Legacy | New home |
|---|---|
| `sd/plotlib.py`: `InvestigativePlots`, `MovieMaker` (+ `.avi` via OpenCV) | `plotting/investigate.py`, `plotting/movie.py` |
| `sd/plotlib.py`: `to_midlatitude_gate_summary`, `beam_gate_boundary_tracker`, `beam_gate_boundary_plots`, `cluster_stats`, `general_stats`, `individal_cluster_stats`, `rand_jitter` | `plotting/cluster_stats.py` |
| `sd/scan_cluster_plots.py`: `MasterPlotter`, `ClusterScanMap` (scan scatter maps with size / colour legends, quivers) | `plotting/scan_map.py` |
| `sd/plot_processed_data.py` `PlotProcessedData`; `fitacf_plots.py` (re-plot a finished run from its parameter file) | `pipelines/plot_outputs.py`: with `params.toml` instead of `params.json` |
| `fitacf_utils.py` `plot_rti` over `.nc.gz` outputs | covered by `make plot-rti`; add reading of gzipped `.nc` |
| `SuperDARN_3D_Scatter_Volume_Plots.ipynb`, `fitacf_polar.py` (velocity histogram by range) | `examples/` |

## 3. Smaller pieces

| Legacy | New home |
|---|---|
| `FetchData.pandas_to_beams`, `pandas_to_scans` (DataFrame back to scans; merges the first two scans when `len(s0)+len(s1)==len(s2)`) | `io/fitacf.py` `dataframe_to_scans` |
| `PrintFitRec.print_fit_record` (text dump of every record with geographic and AACGM coordinates) | `io/summary.py` `write_fit_record` |
| `fitacf_geopos.py`: AACGM latitude/longitude and azimuth of every range cell at a given time | `geo/radar.py` -> `make geopos` |
| `clustering_algo.get_mean_sza` (solar zenith angle of a cell, pysolar) | `geo/sun.py` |
| `config/radar.csv`, `utils.get_radar` (mid / high latitude radar type) | `amgeo_sd/data/radars.csv` + `radars.get_radar()` |
| `extra_codes/bs_analysis.py` (sub-auroral backscatter study, needs `spacepy`) | `examples/` if still wanted |
| `utils.get_gridded_parameters` | replaced by `plotting/grid.py`; delete from the list once the clustering plots are ported |

## 4. Verification still open

* **fitacf 3.0** is read by the same code as fitacf 2.5 (tested with synthetic fitacf 2.0 / 2.5 / 3.0 records), but
  there is no fitacf3 file in `tests/data`; add one radar-day to test it on real data.
* **Reference outputs.** The median and DBSCAN models are checked against literal translations of the legacy logic and the
  geometry against arrays produced by the legacy code (`tests/golden`). For the beam-gate-time model, run the legacy
  pipeline once in a Python 3.7 environment built from `_bkp/amgeo.yml` on the bks / gbr test days, store its
  `.nc` outputs as golden files and compare stage by stage. This needs the old environment.
* **Archive layout.** `AMGEO_SD_DATA` lookups assume `<year>/<ftype>/<rad>/<YYYYMMDD>.*`; untested against a real archive.

## 5. Defects in the legacy code that the port must not carry over

* `utils.smooth`: undefined `numpy.ones`, window chosen with `eval()`.
* `BeamGateFilter.doFilter`: `DataFrame.append` (removed in pandas 2); `stats.median_absolute_deviation` (removed in SciPy 1.9; use `median_abs_deviation`).
* `rti.py` runs its whole analysis at import time.
* Shell-outs (`os.system("mkdir -p")`, `gzip`, `rm -rf {out_dir}*`) and cwd-relative paths in the clustering code.
* `plt.style.use(["science", "ieee"])` needs SciencePlots (not installed); `descartes` is unmaintained.
* `sd_carto.get_aacgm_geom` was never defined, so AACGM coastlines never worked; the new FoV plot converts them itself.
