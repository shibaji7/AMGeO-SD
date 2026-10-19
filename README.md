# AMGeO-SD

SuperDARN high-confidence scatter classification (ionospheric vs. ground scatter, "fitacf++") for AMGeO.
The 2020-2022 research code is being modernised into the `amgeo_sd` package (Python 3.12, src layout).

> **Status:** revamp in progress. The fitacf reader is ported; filters, clustering, geometry and
> plotting are still being moved. The original code is untouched in [`_bkp/`](_bkp/README.md) and the
> full plan is in [`docs/PORTING.md`](docs/PORTING.md).

## Quick start

Everything runs through `make`; `make help` lists all targets.

```bash
make env          # create conda env "amgeo" (python 3.12) and install the package
make data-unpack  # unzip tests/data/archives/*.zip (git-ignored real data)
make data-info    # beams / scans / echoes per radar-day
make test         # unit tests + tests on the real bks / gbr files
```

## Layout

```
src/amgeo_sd/   package (models.py, io/fitacf.py, cli.py)
tests/          pytest suite; tests/data/ holds real fitacf files (git-ignored, see tests/data/README.md)
docs/           PORTING.md - legacy -> new module map and defects to fix
_bkp/           frozen original code (reference only)
Makefile        single entry point for env, data and test commands
```

## Data

fitacf / fitacf3 files (`YYYYMMDD.HHMM.SS.<rad>[.a].fitacf[.bz2|.gz]`), read with `pydarnio>=2`.
Set `AMGEO_SD_DATA` to an archive root laid out as `<year>/<ftype>/<rad>/` for date-range lookups.
Please follow the SuperDARN data usage policy (`data_usage_policy.txt` shipped with the data).

## License

MIT, see [LICENSE](LICENSE).
