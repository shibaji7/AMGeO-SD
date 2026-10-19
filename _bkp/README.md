# _bkp: original AMGeO-SD code (frozen)

Verbatim copy of the 2020-2022 Python 3.7 code base (upstream commit `5063195`), moved here
with `git mv` so history is preserved. **Do not edit.** It is the reference for the port;
every function is mapped to its new home in [`docs/PORTING.md`](../docs/PORTING.md).

It does not run in the `amgeo` (py3.12) env: it expects `sys.path` hacks, cwd-relative
`config/`, `/sd-data`, `pydarnio.SDarnRead`, and pandas<2 / scipy<1.9. Rebuild the old env
from `amgeo.yml` only if you need to regenerate reference outputs.
