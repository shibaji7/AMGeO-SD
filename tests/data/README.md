# Test data (not tracked by git)

Real SuperDARN fitacf files used by the `realdata` tests and by the demos. Everything in this folder except this file
and `.gitkeep` is git-ignored (large files, and redistribution is governed by the SuperDARN data policy). Back to the
[README](../../README.md) or the [tests README](../README.md).

## Layout

One folder per radar-day, named `<rad>_<YYYYMMDD>`:

```text
tests/data/bks_20250208/    Blackstone, mid-latitude, 24 beams, 1 min scans, 12 two-hourly files, fitacf 2.5
tests/data/gbr_20250208/    Goose Bay, high-latitude, 16 beams, 1 min scans, 12 two-hourly files, fitacf 2.5
tests/data/archives/        the original .zip downloads (bks_20250208_fitacf.zip, ...)
```

File names follow the SuperDARN archive: `YYYYMMDD.HHMM.SS.<rad>[.a].fitacf.bz2`, each file spanning two hours.

## Set up and restore

```bash
make data-unpack     # unzip every archive in tests/data/archives/ into tests/data/<name>/
make data-info       # beams / scans / echoes of every unpacked folder
make data-clean      # delete unpacked folders that can be restored from the archives (archives are kept)
```

To add a radar-day, put `<rad>_<YYYYMMDD>_fitacf.zip` into `tests/data/archives/` and run `make data-unpack`, or unpack the
files into `tests/data/<rad>_<YYYYMMDD>/` yourself. A day of fitacf 3.0 data would be a valuable addition: the reader handles
it, but it is only tested on synthetic records so far.

Tests that need these files skip automatically when they are absent. Use of the data is subject to the
`data_usage_policy.txt` shipped inside each folder (acknowledge SuperDARN, consult a SuperDARN researcher before
presenting results).
