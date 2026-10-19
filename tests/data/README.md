# Test data (not tracked by git)

Real SuperDARN fitacf files used by the `realdata` tests. Everything here except this
file and `.gitkeep` is git-ignored. Layout, one folder per radar-day:

    tests/data/bks_20250208/   Blackstone (mid-latitude), 24 beams, 1 min scans
    tests/data/gbr_20250208/   Goose Bay (high-latitude), 16 beams, 1 min scans
    tests/data/archives/       original .zip downloads

To restore: `make data-unpack` (unzips every archive in `tests/data/archives/` into
`tests/data/<name>/`). Use of the data is subject to `data_usage_policy.txt` inside each folder.
Tests that need these files skip automatically when they are absent.
