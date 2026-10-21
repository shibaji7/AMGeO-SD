# Documentation index

Start with the [project README](../README.md) for what AMGeO-SD does and the quick start. The rest, in the order you will
probably need it:

| Document | Read it to | Generated? |
|---|---|---|
| [`USAGE.md`](USAGE.md) | run the median-filter and DBSCAN models and the plots, understand every output variable, and see why the defaults are what they are (whole-day tests on bks and gbr) | no |
| [`PARAMETERS.md`](PARAMETERS.md) | look up one parameter: type, default, allowed values, TOML key, command-line flag, make variable; plus all make variables and output file names | **yes**, `make docs` (a test fails when it is stale) |
| [`API.md`](API.md) and [`api/`](api/models.md) | call the package from Python: every public class and function with its parameters (type, default, meaning), returns and errors, plus runnable workflows | **yes**, `make docs` (from the docstrings; tests fail when stale or when a parameter is undocumented) |
| [`PORTING.md`](PORTING.md) | see which parts of the original (legacy) code are not ported yet, and known defects the port must not copy | no |
| [`../configs/README.md`](../configs/README.md) | pick or write a TOML parameter file | no |
| [`../examples/median_day_analysis/README.md`](../examples/median_day_analysis/README.md) | repeat the whole-day comparison of the two models (`make analyse-day`) | no |
| [`../tests/README.md`](../tests/README.md) | run or extend the tests, and understand the reference ("oracle") tests | no |
| [`../tests/data/README.md`](../tests/data/README.md) | set up the real test data | no |
| [`../_bkp/README.md`](../_bkp/README.md) | find the original code | no |

Conventions used throughout: times are UT; `rad` is the three-letter radar code (`bks`, `gbr`, ...); "fitacf++" is the product,
`.pp` its file-name marker; a "scan" is one sweep of all beams (1 or 2 minutes); an "echo" is one range gate with scatter in one beam
sounding.
