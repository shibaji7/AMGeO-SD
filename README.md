# AMGeO-SD

**Scatter classification for SuperDARN radar data ("fitacf++")**, for the AMGeO project.

SuperDARN (Super Dual Auroral Radar Network) HF radars record line-of-sight Doppler velocity, spectral width and
power of backscatter from the ionosphere and from the ground. Raw `fitacf` files mix the two and contain a lot of
isolated noise echoes. AMGeO-SD reads `fitacf` files, **removes noise and separates ionospheric from ground scatter**,
and writes the result as **fitacf++**: a NetCDF file and a fitacf-compatible DMAP file that downstream tools can use
in place of the raw data. It also draws range-time (RTI) and field-of-view maps of both raw and fitacf++ data.

The package is a modernisation (Python 3.12, tests on real data, one `make` entry point) of research code written
in 2020-2022; the original is frozen in [`_bkp/`](_bkp/README.md) and what is still to be moved is listed in
[`docs/PORTING.md`](docs/PORTING.md).

```text
 fitacf 2.0 / 2.5 / 3.0  --+--> median-filter model --+--> <rad>_<start>_<end>.pp.nc             (NetCDF4)
 (.bz2 .gz or plain)       |    (clean + flag)        +--> <date>.<rad>.fitacf.pp.bz2           (DMAP)
                           +--> DBSCAN / GMM model ---+--> params.toml, script.cmd              (provenance)
                                (cluster + flag)                      |
                                                                      +--> RTI and field-of-view figures
```

## Status

| Part | State |
|---|---|
| fitacf reader (2.0, 2.5, 3.0; `.bz2`, `.gz`, plain), scans, DataFrames | done, tested on real bks / gbr data (fitacf 3.0 only on synthetic records) |
| median-filter model | done, equals a literal copy of the legacy algorithm on real data |
| DBSCAN / Gaussian-mixture model + Ribiero cluster rule | done, equals a literal copy of the legacy logic on real data |
| fitacf++ outputs (`.pp.nc`, `.fitacf.pp.bz2`) | done |
| RTI and field-of-view plots, geographic or AACGM | done |
| radar geometry: original `CalcFov` or pydarn 4 | done, switchable |
| TOML parameter files, generated parameter reference | done |
| beam-gate-time clustering model and the plots that need it | not ported yet, see [`docs/PORTING.md`](docs/PORTING.md) |

## Quick start

You need conda and the data (see [Data](#data)). Everything is run through `make`; `make help` lists all targets.

```bash
make env            # conda env "amgeo" (Python 3.12) + editable install
make data-unpack    # unzip the test data in tests/data/archives/ (bks and gbr, 2025-02-08)
make data-info      # beams / scans / echoes per radar-day, a quick check that the data load

make median-demo    # median-filter model, 1 hour of bks and gbr  -> data/outputs/demo/
make dbscan-demo    # DBSCAN / GMM model, same hours              -> data/outputs/demo-dbscan/
make plot-demo      # RTI + field-of-view figures, raw vs fitacf++ -> data/figures/demo/

make test           # full test suite
```

Your own run, for any radar and time range:

```bash
make median RAD=bks START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/bks_20250208
make plot-rti INPUT=data/outputs/L100/bks/bks_202502080300_202502080400.pp.nc BEAM=12 PARAMS="v w_l gflg_kde"
```

or with a parameter file (every key documented in [`configs/default.toml`](configs/default.toml)):

```bash
make median CONFIG=configs/median_bks_demo.toml THRESH=0.5     # file first, then make/command-line values override
```

## The two models

| | median-filter model (`make median`) | DBSCAN / GMM model (`make dbscan`) |
|---|---|---|
| idea | an echo must be supported by its neighbours in time (3 scans), beam and range gate | echoes of one beam that are close in (gate, scan) form a cluster; a cluster is ionospheric or ground by its speed and lifetime |
| echoes | **removes** isolated echoes, replaces values by neighbourhood medians, adds a few filled cells | keeps **all** raw echoes, only labels them |
| classification | `gflg_kde`, `gflg_conv` (neighbour flags), plus the legacy `gflg` (not recommended) | `gflg_ribiero` (Ribiero et al. 2011), `cluster_tag` |
| main parameters | `thresh` (default 0.6), `edge_gates` | `eps` (2), `min_samples` (8), `gmm` |
| outputs keep | filtered `v w_l p_l` and `v_mad`; `v_e` = `v_mad` in the DMAP | original records unchanged plus two extra arrays |
| details | [usage guide, section 3](docs/USAGE.md) | [usage guide, section 4](docs/USAGE.md) |

Which flag to trust, how the default thresholds were chosen (whole-day tests on bks and gbr) and what changed
against the legacy code are explained in the usage guide.

## Outputs

Results go to `<out>/<sim_id>/<rad>/` (default `data/outputs/L100/<rad>/`):

| File | Content |
|---|---|
| `<rad>_<start>_<end>.pp.nc` | NetCDF4: geometry `gdlat gdlon gdazm`, per-beam metadata, per-echo `v w_l p_l gflg ...` on a (scan, beam slot, gate) grid, NaN = no echo |
| `<YYYYMMDD.HHMM.SS>.<rad>.fitacf.pp.bz2` | the same data as a bz2 DMAP file, one record per beam, all original fitacf fields kept |
| `params.toml` | every parameter of the run, annotated, plus provenance and result counts; can be fed back with `CONFIG=` |
| `script.cmd` | the exact command line |

`.pp` stands for "plus plus". Files an earlier version wrote as `.fitacf.pl.bz2` are still read.
`pydarnio.read_fitacf` rejects the extra fields of `.fitacf.pp.bz2`; read them with `amgeo_sd.io.read_fitacf_file`
(or `pydarnio.read_dmap`).

## Plots

`make plot-rti` (range-time of one beam) and `make plot-fov` (map of one scan) read any raw or fitacf++ file, so
a raw and a filtered dataset can be drawn identically and stacked for comparison. Parameters: `v w_l p_l v_mad`, the
flags `gflg gflg_conv gflg_kde gflg_ribiero` and `cluster_tag`; maps in geographic or AACGM coordinates.
See [usage guide, section 6](docs/USAGE.md).

## Configuration

Settings come from three places, later ones winning: **built-in default < TOML file < command-line flag / make variable**.
Unknown keys and out-of-range values are errors. The complete list, with type, default, allowed values, the matching
command-line flag and make variable, is the [parameter reference](docs/PARAMETERS.md).

## Repository layout

```text
src/amgeo_sd/          the package
  io/                    fitacf reader (2.0/2.5/3.0), fitacf++ writer and loader, scan-mode estimate
  filters/               median filter
  clustering/            DBSCAN / Gaussian-mixture model;  scatter.py: Ribiero cluster rule
  geo/                   radar field of view (legacy CalcFov, pydarn 4), coordinates
  plotting/              RTI and field-of-view plots
  pipelines/             median, dbscan and plot workflows;  export.py: NetCDF + params.toml
  config.py              TOML parameters;  params_doc.py: generates docs/PARAMETERS.md;  cli.py: python -m amgeo_sd
configs/               TOML parameter files (configs/README.md)
docs/                  documentation (docs/README.md is the index)
examples/              whole-day analysis of the two models
tests/                 pytest suite on real data (tests/README.md);  tests/data/ is git-ignored
_bkp/                  the original code, frozen
Makefile               the entry point for everything above
```

## Data

Real test data are fitacf 2.5 files of Blackstone (`bks`, mid-latitude, 24 beams) and Goose Bay (`gbr`, high-latitude,
16 beams) for 2025-02-08, kept in `tests/data/` (git-ignored, see [`tests/data/README.md`](tests/data/README.md)). Tests
that need them skip when they are absent. Set `AMGEO_SD_DATA` to an archive laid out as
`<year>/<ftype>/<rad>/<YYYYMMDD>.*` to select data by time range instead of by path.

SuperDARN data are subject to the SuperDARN data usage policy (shipped as `data_usage_policy.txt` with the data):
acknowledge SuperDARN in publications and presentations and consult an experienced SuperDARN researcher before
presenting results; see <https://superdarn.vt.edu/data-acknowledgement/>.

## Development

```bash
make test         # everything (about 4 minutes); make test-unit is the fast subset without real data
make lint         # ruff: errors, import order, missing docstrings
make format       # ruff format
make docs         # regenerate docs/PARAMETERS.md after changing a setting (a test fails when it is stale)
```

Work happens on `main`. Ported legacy modules (`geo/geopack.py`, `geo/fov.py`, `geo/vheight.py`) keep their original
style and are excluded from formatting.

## Documentation

| Document | For |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | how to run each model and plot, what the outputs contain, how the choices were made |
| [`docs/PARAMETERS.md`](docs/PARAMETERS.md) | every parameter: type, default, allowed values, flag, make variable |
| [`docs/PORTING.md`](docs/PORTING.md) | what of the legacy code is still to be ported |
| [`configs/README.md`](configs/README.md) | the TOML files and how to write your own |
| [`examples/median_day_analysis/README.md`](examples/median_day_analysis/README.md) | the whole-day comparison of the two models |
| [`tests/README.md`](tests/README.md) | how the tests are organised |
| [`docs/README.md`](docs/README.md) | index of all of the above |

## License

MIT, see [LICENSE](LICENSE) (copyright 2020 shibaji7).
