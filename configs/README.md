# Parameter files

TOML files that fill in the [parameters](../docs/PARAMETERS.md) of a run. Use one with `make <target> CONFIG=configs/<file>.toml`
or `python -m amgeo_sd <command> -c <file>`; flags and make variables given as well override the file
(default < file < flag). Back to the [README](../README.md).

| File | What it is |
|---|---|
| [`default.toml`](default.toml) | **every** parameter with its default and a comment; copy it and delete what you do not change |
| [`median_bks_demo.toml`](median_bks_demo.toml), [`median_gbr_demo.toml`](median_gbr_demo.toml) | median-filter model, 1 hour of bks / gbr from `tests/data` (`make median-demo`) |
| [`dbscan_bks_demo.toml`](dbscan_bks_demo.toml), [`dbscan_gbr_demo.toml`](dbscan_gbr_demo.toml) | DBSCAN model, same hours; the gbr one also turns the Gaussian-mixture step on (`make dbscan-demo`) |
| [`plot_rti_demo.toml`](plot_rti_demo.toml), [`plot_fov_demo.toml`](plot_fov_demo.toml) | an RTI (raw vs fitacf++) and a field-of-view figure of the demo output (`make plot-demo`) |

## Write your own

```toml
# my_run.toml: only what differs from the defaults
[run]
rad    = "bks"
start  = 2025-02-08T00:00:00      # TOML datetimes need no quotes
end    = 2025-02-08T23:59:59
sim_id = "day"

[input]
paths = ["tests/data/bks_20250208"]

[median]
thresh = 0.5
```

```bash
make config-show CONFIG=my_run.toml     # print what will actually run
make median CONFIG=my_run.toml
```

A run's own `params.toml` (in its output folder) is a complete, annotated config of that run, with extra `[provenance]` and
`[result]` tables that are ignored when it is read back, so it can be used to repeat or adjust the run.
