"""Median-filter model against the raw echoes and the DBSCAN labels: retention tables and threshold sweep.

Run from the project root, see README.md:
    python examples/median_day_analysis/analysis_day.py --rad bks --work data/analysis/bks
"""

import pandas as pd

from amgeo_sd.filters import MedianFilter
from amgeo_sd.io import FitacfReader, load_frame, scans_to_dataframe

from common import KEYS, data_files, nc_file, parse_args, window

CLASSES = ["noise", "small(<50)", "medium(50-499)", "large(>=500)"]


def kept_mask(raw, filt):
    """True for the raw echoes that also exist (same time, beam, gate) in ``filt``."""
    m = raw[KEYS].merge(filt[KEYS].assign(k=1), on=KEYS, how="left")
    return m["k"].fillna(0).to_numpy().astype(bool)


def pct(s):
    return (100 * s).round(1)


def rate(mask, sel):
    """Percentage of ``mask`` that is True among the entries selected by ``sel`` (NaN when none are selected)."""
    return 100 * mask[sel].mean() if sel.any() else float("nan")


def main(argv=None):
    def extra(ap):
        ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8", help="median thresholds of the sweep")
        ap.add_argument("--jobs", type=int, default=4, help="worker processes for the sweep")

    args = parse_args(__doc__.splitlines()[0], argv, extra)
    start, end = window(args)
    scans = FitacfReader(args.rad, start, end, files=data_files(args)).scans()
    raw = scans_to_dataframe(scans)
    med, _ = load_frame([nc_file(args, args.median_sim)])
    db, _ = load_frame([nc_file(args, args.dbscan_sim)])
    print(
        f"== {args.rad}: raw {len(raw)} echoes, {raw.bmnum.nunique()} beams, {raw.scnum.max() + 1} scans, {start} -> {end}"
    )

    # DBSCAN class and cluster size of every raw echo: a proxy for structure vs isolated noise (not ground truth)
    sizes = db.groupby(["bmnum", "cluster_tag"]).size().rename("csize").reset_index()
    db = db.merge(sizes, on=["bmnum", "cluster_tag"], how="left")
    db.loc[db.cluster_tag < 0, "csize"] = 0
    db["cclass"] = pd.cut(db.csize, [-1, 0, 49, 499, 1e9], labels=CLASSES)
    raw = raw.merge(db[KEYS + ["cclass", "gflg_ribiero"]], on=KEYS, how="left")
    raw["kept"] = kept_mask(raw, med)
    raw["hour"] = pd.to_datetime(raw.time).dt.hour
    added = len(med) - raw.kept.sum()
    print(
        f"median filter: keeps {raw.kept.mean() * 100:.1f}% of raw echoes; output {len(med)} cells, {added} ({100 * added / len(med):.1f}%) are NEW cells"
    )

    print("\n-- retention by DBSCAN cluster class")
    t = raw.groupby("cclass", observed=True).agg(echoes=("kept", "size"), kept=("kept", "mean"))
    t["kept"], t["share"] = pct(t.kept), pct(t.echoes / len(raw))
    print(t.to_string())
    print("\n-- retention by fitacf ground-scatter flag (0 = ionospheric, 1 = ground)")
    t = raw.groupby("gflg").kept.agg(["size", "mean"])
    t["mean"] = pct(t["mean"])
    print(t.to_string())
    print("\n-- retention by hour (UT), % ; raw echoes per hour")
    h = raw.groupby("hour").agg(raw=("kept", "size"), kept=("kept", "mean"))
    h["kept"] = pct(h.kept)
    print(h.T.to_string())
    print("\n-- retention by gate block, %")
    raw["gb"] = (raw.slist // 10) * 10
    g = raw.groupby("gb").agg(n=("kept", "size"), kept=("kept", "mean"))
    g["kept"] = pct(g.kept)
    print(g.T.to_string())
    print("\n-- retention within each cluster class by gate block, %")
    print(
        raw.pivot_table(index="cclass", columns="gb", values="kept", aggfunc="mean", observed=True)
        .mul(100)
        .round(0)
        .to_string()
    )
    print("\n-- per beam")
    nsc = raw.scnum.nunique()
    rows = {}
    for b, d in raw.groupby("bmnum"):
        rows[b] = {
            "kept%": 100 * d.kept.mean(),
            "noise_share%": 100 * (d.cclass == "noise").mean(),
            "struct_kept%": 100 * d.kept[d.cclass.isin(CLASSES[2:])].mean(),
            "noise_kept%": 100 * d.kept[d.cclass == "noise"].mean(),
            "scans_w_echo%": 100 * d.scnum.nunique() / nsc,
            "echoes/scan": len(d) / nsc,
        }
    print(pd.DataFrame(rows).round(1).to_string())

    print("\n-- threshold sweep (no KDE, no clustering)")
    cls = raw.cclass.to_numpy()
    out = []
    for th in (float(x) for x in args.thresholds.split(",")):
        f = scans_to_dataframe(MedianFilter(thresh=th, kde=False, cluster=False).filter_scans(scans, n_jobs=args.jobs))
        km = kept_mask(raw, f)
        out.append(
            {
                "thresh": th,
                "kept%": 100 * km.mean(),
                "noise_removed%": 100 - rate(km, cls == CLASSES[0]),
                "small_kept%": rate(km, cls == CLASSES[1]),
                "medium_kept%": rate(km, cls == CLASSES[2]),
                "large_kept%": rate(km, cls == CLASSES[3]),
                "new_cells": int(len(f) - km.sum()),
                "new%of_out": 100 * (len(f) - km.sum()) / len(f),
            }
        )
    sweep = pd.DataFrame(out).round(1)
    print(sweep.to_string(index=False))
    sweep.to_csv(f"{args.work}/sweep_{args.rad}.csv", index=False)


if __name__ == "__main__":
    main()
