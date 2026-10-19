# Using amgeo_sd

Two parts: what works **today** (the fitacf reader), and what the **original code did** and how each
piece will be reproduced (see [PORTING.md](PORTING.md) for status per function).

## 1. What works today

### Make targets

```bash
make help         # list all targets with descriptions
make env          # create conda env "amgeo" (python 3.12) + install the package
make data-unpack  # unzip tests/data/archives/*.zip -> tests/data/<radar>_<date>/
make data-info    # beams / scans / echoes for every radar-day folder (or DATA="dir1 dir2")
make test         # all tests;  test-unit = in-memory only;  test-real = real bks/gbr files
make test-cov     # coverage report
make clean        # caches, coverage, build files
make data-clean   # delete unpacked data folders (only those restorable from archives/)
```

### Python API

```python
import datetime as dt
from amgeo_sd.io import FitacfReader, scans_to_dataframe

# explicit files ...
reader = FitacfReader("bks", dt.datetime(2025, 2, 8, 3), dt.datetime(2025, 2, 8, 4),
                      files=["tests/data/bks_20250208/20250208.0200.00.bks.a.fitacf.bz2"])
# ... or archive lookup: root=<dir> or $AMGEO_SD_DATA, laid out <year>/<ftype>/<rad>/<YYYYMMDD>.*
beams = reader.beams()          # list[amgeo_sd.models.Beam], time ordered
scans = reader.scans()          # list[amgeo_sd.models.Scan] grouped by the scan flag
df = scans_to_dataframe(scans)  # one row per echo; columns time, bmnum, slist, v, v_e, w_l, p_l, gflg, scnum, sbnum, ...
```

* A `Beam` holds one sounding (header values as attributes, per-echo numpy arrays `slist v v_e w_l p_l gflg`,
  and an `extra` dict for arrays that later algorithms add, e.g. `v_mad`, cluster labels).
* A `Scan` holds the beams of one sweep plus `stime`, `etime`, mean `freq`/`nsky`.
* `.bz2`, `.gz` and plain files are all read; corrupt trailing records are dropped with a warning.
* Column names differ from the legacy ones where a dot was awkward: `noise.sky -> noise_sky`,
  `intt.sc -> intt_sc`, `intt.us -> intt_us`.

## 2. The original workflows (not yet ported)

All three start from fitacf files and end with an xarray/NetCDF4 `*.fitacf++.nc` file under
`data/outputs/<sim_id>/<rad>/` together with `params.json` and `script.cmd` (the exact command line).
Outputs add flags to the fitacf variables `v w_l p_l gflg` and geolocation (`gdlat gdlon gdazm`).

### A. Median-filter model (`fitacf_amgeo.py`, `Simulate`, `Filter`)

1. `fit_records.beam_summary`: scan period, themis beam, scan mode from the first minutes of data.
2. Load scans, drop repeated beams, slide a 3-scan x 3-beam x 3-gate **weighted median** (weights
   `[[1,2,1],[2,3,2],[1,2,1]] / [[2,3,2],[3,5,3],[2,3,2]] / ...`); a gate survives if the weight of occupied
   cells / total weight >= `thresh` (default 0.7).
3. Per surviving gate: median `v`, `w_l`, `p_l`, `v_mad`, and three ground-scatter flags:
   `gflg` (ionospheric if median `w_l > -3 v + 90`, else ground), `gflg_conv` (weighted fraction of GS neighbours vs. `lth`/`uth`,
   2 = undecided) and `gflg_kde` (beta-distribution fit, cut at `pth`).
4. Per-scan DBSCAN on (beam, gate) gives `clabel`; save `.nc`.

Options: `-r -s -e -th -pth -lth -uth -gct -sid -ftype`. Planned: `make median RAD=bks START=... END=...`.

### B. Beam-gate-time clustering for mid-latitude radars (`fitacf_amgeo_cluster.py`, `BeamGateTimeFilter`)

1. Cut the period into `dur`-minute windows (default 30); in each, run model A, then per beam a DBSCAN
   over range gates to find scatter bands (`eps`, `min_samples`), smooth/merge them across beams and
   classify each band as ionospheric / ground scatter with a KDE-based probability (`ScatterTypeDetection`).
2. Track the resulting beam-gate boxes from window to window (`beam_gate_time_tracker.json`).
3. For every tracked cluster apply the Ribiero et al. (2011) rule (velocity threshold 25 m/s, duration-dependent
   high/low-velocity ratio) -> `cluster_tag`, `ribiero_gflg`; write per-cluster `.h5` and the merged `.nc`.
4. Diagnostic figures per window (`beamgate/figs/...`) and an optional `.avi` movie.

### C. Scan-by-scan DBSCAN/GMM (`fitacf_amgeo_dbcluster.py`, `DBScan`)

For each channel and beam, cluster echoes in (range gate / `eps_g`, scan number / `eps_s`) with DBSCAN
(`eps`, `min_samples`; `min_samples x tb_mult` on the themis beam), optionally after a Gaussian-mixture split
on Box-Cox-transformed velocity (`-gmm bmnum|T|A`, `-nc`, `-cov`, `-gth`), then apply the Ribiero rule
(`-vth`) -> `gflg_ribiero`. Writes RTI and fan-plot images and the `.nc`.

### Plots (`fitacf_plots.py`, `fitacf_utils.py`, `sd/*plot*.py`, `rti.py`)

* **RTI** (range-time-intensity, one beam): velocity, cluster labels, GS/IS flag.
* **Fan / FoV maps** (cartopy, geographic or AACGM): echoes coloured by velocity with a GS mask, radar FoV, terminator.
* **Cluster scan maps**, beam-gate boundary / cluster statistics, and `.avi` movies.
* `fitacf_utils.py -o rti|fov` plots from stored `.nc.gz` outputs; `fitacf_plots.py` re-plots from `params.json`.

### Planned make interface

`make median`, `make cluster`, `make dbscan`, `make plot-rti`, `make plot-fov`, `make geopos`,
each taking `RAD START END SID OUT` and the algorithm parameters above; added as each stage is ported and
verified against the legacy outputs.
