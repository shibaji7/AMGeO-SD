# Tests

`make test` runs everything (about 4 minutes); `make test-unit` runs only tests that need no real data (seconds);
`make test-real` only those on the real files. `make test-cov` adds a coverage report. Back to the [README](../README.md).

## Layout

| File | Covers |
|---|---|
| `test_fitacf_io.py`, `test_writer_and_load.py` | fitacf reader (2.0 / 2.5 / 3.0 records), scans, DataFrames, file discovery, the `.fitacf.pp.bz2` writer and the multi-format loader |
| `test_geo.py` | radar geometry: legacy `CalcFov` against arrays made by the legacy code (`golden/`), pydarn backend |
| `test_median_filter.py`, `test_pipeline_median.py` | median filter (against a literal copy of the legacy loops), NetCDF export, scan-mode estimate, pipeline outputs |
| `test_dbscan_model.py` | Ribiero rule, DBSCAN / GMM clustering (against a literal copy of the legacy logic), pipeline outputs |
| `test_plotting.py` | RTI and field-of-view plots, plot CLI |
| `test_config.py`, `test_docs.py` | TOML loading and validation; generated parameter reference, CLI flags, make variables and Markdown links stay consistent |
| `test_examples.py` | the whole-day analysis scripts and `make analyse-day` on a 30-minute window |
| `reference_median.py`, `reference_dbscan.py` | slow translations of the legacy algorithms used as oracles |
| `golden/` | arrays computed once with the legacy code (field of view of bks and gbr) |
| `data/` | real fitacf files, git-ignored, see [`data/README.md`](data/README.md) |

## Markers and data

* `@pytest.mark.realdata`: needs the files in `tests/data/`; such tests skip (not fail) when they are missing, via the
  `fitacf_files` fixture in `conftest.py`.
* Tests build their inputs from small synthetic records where they can, and from 10-30 minute windows of the real files otherwise,
  so no test needs a full day.
* The oracle tests are the main correctness evidence: the vectorised median filter and the clustering code must give the same
  numbers as the straightforward, slow translation of the original code on real data.

## Adding a test

Put it next to the closest existing file, use the real data through the `fitacf_files` fixture and mark it `realdata`, and keep
tolerances explicit (`np.testing.assert_allclose(..., atol=...)`). Run `make lint` and `make format` before committing.
