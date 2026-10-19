"""Figures of the analysis: raw vs median-filtered RTI, DBSCAN clusters and flags, and the threshold trade-off.

Run from the project root, see README.md:
    python examples/median_day_analysis/plot_day.py --rad bks --work data/analysis/bks
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amgeo_sd.config import load_settings  # noqa: E402
from amgeo_sd.io import load_frame  # noqa: E402
from amgeo_sd.plotting import plot_rti, plot_rti_compare  # noqa: E402

from common import data_files, nc_file, parse_args, window  # noqa: E402


def tradeoff_figure(sweep: pd.DataFrame, path: Path, rad: str, default_thresh: float) -> None:
    """Retention of DBSCAN-defined structure, removal of DBSCAN noise and added cells against the median threshold."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    th = sweep["thresh"]
    ax.plot(th, sweep["large_kept%"], "o-", label="kept: large clusters (>=500 echoes)")
    ax.plot(th, sweep["medium_kept%"], "o-", label="kept: medium (50-499)")
    ax.plot(th, sweep["small_kept%"], "o-", label="kept: small (<50)")
    ax.plot(th, sweep["noise_removed%"], "s--", c="k", label="removed: DBSCAN noise")
    ax.plot(th, sweep["new%of_out"], "^:", c="gray", label="new cells, % of output")
    ax.axvline(default_thresh, c="r", lw=0.8)
    ax.set(xlabel="median-filter threshold", ylabel="%", title=f"{rad.upper()}: threshold trade-off")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    def extra(ap):
        ap.add_argument("--beam", type=int, help="beam of the RTI figures (default: the beam with most echoes)")
        ap.add_argument("--out", help="figure directory (default <work>/figures)")
        ap.add_argument("--dpi", type=int, default=110)

    args = parse_args(__doc__.splitlines()[0], argv, extra)
    start, end = window(args)
    out = Path(args.out or f"{args.work}/figures")
    out.mkdir(parents=True, exist_ok=True)
    raw, _ = load_frame(data_files(args), start, end)
    med_nc, db_nc = nc_file(args, args.median_sim), nc_file(args, args.dbscan_sim)
    med, _ = load_frame([med_nc])
    db, _ = load_frame([db_nc])
    beam = args.beam if args.beam is not None else int(raw.bmnum.value_counts().idxmax())
    thresh = load_settings(Path(med_nc).parent / "params.toml").median.thresh
    made = []

    p = out / f"rti_{args.rad}_b{beam:02d}_raw_vs_median.png"
    plot_rti_compare(
        {"raw fitacf": raw, f"median filter, thresh {thresh}": med},
        beam,
        "v",
        p,
        rad=args.rad,
        flag="gflg",
        dpi=args.dpi,
    )
    made.append(p)
    p = out / f"rti_{args.rad}_b{beam:02d}_dbscan.png"
    plot_rti(db, beam, ("v", "cluster_tag", "gflg_ribiero"), p, rad=args.rad, kind="DBSCAN model", dpi=args.dpi)
    made.append(p)
    sweep_csv = Path(args.work) / f"sweep_{args.rad}.csv"
    if sweep_csv.exists():
        p = out / f"threshold_tradeoff_{args.rad}.png"
        tradeoff_figure(pd.read_csv(sweep_csv), p, args.rad, thresh)
        made.append(p)
    plt.close("all")
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
