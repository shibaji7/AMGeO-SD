# Median-filter vs DBSCAN: whole-day analysis

Back to the [README](../../README.md) · [usage guide](../../docs/USAGE.md) · [parameter reference](../../docs/PARAMETERS.md).

Scripts used to judge the median-filter model on a full day of one radar (bks, gbr in `tests/data`), using the
DBSCAN model's clusters as a *proxy* for "structure vs isolated noise" (DBSCAN is itself a heuristic, so this
measures agreement between two algorithms, not truth).

One command does everything for one radar (about 3 minutes):

```bash
make analyse-day RAD=bks            # or RAD=gbr; DAY=2025-02-08 is the default
```

It runs `make median` and `make dbscan` for the whole day into `data/analysis/<rad>/` (git-ignored), then the scripts below,
writing their text output next to the data and the figures to `data/analysis/<rad>/figures/`.

| Script | Output |
|---|---|
| `analysis_day.py` | `retention.txt`: share of echoes kept by DBSCAN cluster class, fitacf flag, hour, gate block and beam; threshold sweep -> `sweep_<rad>.csv` |
| `analysis_more.py` | `edges_and_flags.txt`: retention at the first / last range gate, bias against ionospheric scatter, velocity change, agreement of the scatter flags |
| `analysis_flags.py` | `flag_physics.txt`: for every flag, how fast its "ground scatter" echoes are (should be slow) and how slow its "ionospheric" ones |
| `plot_day.py` | `rti_<rad>_b<beam>_raw_vs_median.png`, `rti_<rad>_b<beam>_dbscan.png`, `threshold_tradeoff_<rad>.png` |
| `common.py` | shared arguments (`--rad --work --data --day --start --end --median-sim --dbscan-sim`) |

By hand (from the project root), for any window that both models have run over:

```bash
make median RAD=bks START=2025-02-08T00:00 END=2025-02-08T23:59:59 INPUT=tests/data/bks_20250208 SID=day OUT=data/analysis/bks EXTRA="--no-fitacf"
make dbscan RAD=bks START=2025-02-08T00:00 END=2025-02-08T23:59:59 INPUT=tests/data/bks_20250208 SID=day-dbscan OUT=data/analysis/bks EXTRA="--no-fitacf"
python examples/median_day_analysis/analysis_day.py --rad bks --work data/analysis/bks --thresholds 0.5,0.6,0.7
python examples/median_day_analysis/plot_day.py --rad bks --work data/analysis/bks --beam 12
```

Findings for the 2025-02-08 data, and what they led to, are in `docs/USAGE.md` (section on the DBSCAN model) and `docs/PORTING.md`.
