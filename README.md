# AMGeO-SD

SuperDARN high-confidence scatter classification (ionospheric vs. ground scatter, "fitacf++") for AMGeO.
The 2020-2022 research code is being modernised into the `amgeo_sd` package (Python 3.12, src layout).

> **Status:** revamp in progress. Ported and tested on real bks / gbr data: fitacf reader (2.0 / 2.5 / 3.0), radar geometry,
> the median-filter model with fitacf++ output (`.nc` and `.fitacf.pl.bz2`), RTI and field-of-view plots, TOML parameter files.
> Still to move: the two clustering models and their plots, see [`docs/PORTING.md`](docs/PORTING.md).
> The original code is untouched in [`_bkp/`](_bkp/README.md).

## Quick start

Everything runs through `make`; `make help` lists all targets, [`docs/USAGE.md`](docs/USAGE.md) explains them.

```bash
make env          # create conda env "amgeo" (python 3.12) and install the package
make data-unpack  # unzip tests/data/archives/*.zip (git-ignored real data)
make median-demo  # median-filter model on 1 h of bks and gbr -> data/outputs/demo/
make plot-demo    # RTI and field-of-view figures, raw fitacf vs fitacf++ -> data/figures/demo/
make test         # unit tests + tests on the real bks / gbr files
```

Parameters live in TOML files ([`configs/default.toml`](configs/default.toml) lists all of them); every run writes an annotated
`params.toml` that can be fed back with `CONFIG=`.

## Layout

```
src/amgeo_sd/   package: models, config, io/, geo/, filters/, plotting/, pipelines/, export, cli
configs/        TOML parameter files (default.toml documents every key)
tests/          pytest suite; tests/data/ holds real fitacf files (git-ignored, see tests/data/README.md)
docs/           USAGE.md (how to use), PORTING.md (what is left to port)
_bkp/           frozen original code (reference only)
Makefile        single entry point for env, data and test commands
```

## Data

fitacf / fitacf3 files (`YYYYMMDD.HHMM.SS.<rad>[.a].fitacf[.bz2|.gz]`), read with `pydarnio>=2`.
Set `AMGEO_SD_DATA` to an archive root laid out as `<year>/<ftype>/<rad>/` for date-range lookups.
Please follow the SuperDARN data usage policy (`data_usage_policy.txt` shipped with the data).

## License

MIT, see [LICENSE](LICENSE).
