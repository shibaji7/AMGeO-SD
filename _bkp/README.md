# _bkp: original AMGeO-SD code (frozen)

Verbatim copy of the 2020-2022 Python 3.7 code base (upstream commit `5063195`, branch `master` on GitHub), moved here
with `git mv` so history is preserved. **Do not edit.** It is the reference for the port; what is still to be moved is listed in
[`docs/PORTING.md`](../docs/PORTING.md). Back to the [README](../README.md).

| Legacy | Role | Replaced by |
|---|---|---|
| `fitacf_amgeo.py`, `sd/analysis.py`, `sd/boxcar_filter.py` | median-filter model | `make median` (`amgeo_sd.filters`, `pipelines/median_filter.py`) |
| `fitacf_amgeo_dbcluster.py`, `DBScan` in `sd/clustering_algo.py` | DBSCAN / GMM model | `make dbscan` (`amgeo_sd.clustering`) |
| `fitacf_amgeo_cluster.py`, `BeamGateTimeFilter` in `sd/clustering_algo.py` | beam-gate-time clustering | **not ported yet** |
| `sd/get_fit_data.py`, `tools/fit_records.py` | reading fitacf, scan mode | `amgeo_sd.io` |
| `tools/rad_fov.py`, `geoPack.py`, `model_vheight.py` | radar geometry | `amgeo_sd.geo` (copied unchanged) |
| `sd/fan_rti.py`, `sd_plots.py`, `plotlib.py`, `rti.py`, `tools/sd_carto.py` | plots | `amgeo_sd.plotting` (RTI, field of view); the rest not ported yet |
| `extra_codes/` | old drivers (call functions that no longer exist) | dropped |

It does not run in the `amgeo` (Python 3.12) environment: it expects `sys.path` hacks, a cwd-relative `config/`, `/sd-data`,
`pydarnio.SDarnRead`, and pandas < 2 / scipy < 1.9. Rebuild the old environment from `amgeo.yml` only if you need to regenerate
reference outputs.
