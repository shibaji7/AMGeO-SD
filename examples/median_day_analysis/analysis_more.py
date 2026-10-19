"""Edge gates, ionospheric-scatter bias, velocity change and flag agreement of the median filter.

Run from the project root, see README.md:
    python examples/median_day_analysis/analysis_more.py --rad bks --work data/analysis/bks
"""

import pandas as pd

from amgeo_sd.io import FitacfReader, load_frame, scans_to_dataframe

from common import KEYS, data_files, nc_file, parse_args, window


def main(argv=None):
    args = parse_args(__doc__.splitlines()[0], argv)
    start, end = window(args)
    raw = scans_to_dataframe(FitacfReader(args.rad, start, end, files=data_files(args)).scans())
    med, _ = load_frame([nc_file(args, args.median_sim)])
    db, _ = load_frame([nc_file(args, args.dbscan_sim)])
    sizes = db.groupby(["bmnum", "cluster_tag"]).size().rename("csize").reset_index()
    db = db.merge(sizes, on=["bmnum", "cluster_tag"], how="left")
    db["struct"] = (db.cluster_tag >= 0) & (db.csize >= 50)
    raw = raw.merge(db[KEYS + ["struct", "gflg_ribiero"]], on=KEYS, how="left")
    med_cols = ["v", "w_l", "gflg", "gflg_conv", "gflg_kde", "v_mad"]
    both = raw.merge(med[KEYS + med_cols].rename(columns=lambda c: c if c in KEYS else c + "_m"), on=KEYS, how="left")
    both["kept"] = both["v_m"].notna()
    nrang = int(raw.nrang.max())
    print(f"== {args.rad}  (nrang {nrang})")

    s = both[both.struct]
    print("\n-- retention of STRUCTURE echoes (cluster >= 50 echoes) at the range edges, % by gate")
    g = s.groupby("slist").kept.mean().mul(100).round(0)
    print(
        "gates 0-4:",
        g.reindex(range(5)).tolist(),
        "| gates",
        nrang - 5,
        "-",
        nrang - 1,
        ":",
        g.reindex(range(nrang - 5, nrang)).tolist(),
    )
    print(f"interior gates 10-{nrang - 15} mean: {g.loc[10 : nrang - 15].mean():.0f}%")

    print("\n-- retention of structure echoes by fitacf flag, %")
    print(s.groupby("gflg").kept.agg(["size", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(1)).to_string())
    print("   by Ribiero cluster flag:")
    print(
        s.groupby("gflg_ribiero")
        .kept.agg(["size", "mean"])
        .assign(mean=lambda d: (100 * d["mean"]).round(1))
        .to_string()
    )

    k = both[both.kept]
    dv = (k.v_m - k.v).abs()
    print(
        f"\n-- velocity change on kept echoes |v_median - v_raw|: median {dv.median():.1f}  mean {dv.mean():.1f}  "
        f"90% {dv.quantile(0.9):.1f}  >10 m/s {100 * (dv > 10).mean():.1f}%  >25 m/s {100 * (dv > 25).mean():.1f}%"
    )
    print(
        f"   v_mad of kept echoes: median {k.v_mad_m.median():.1f}  90% {k.v_mad_m.quantile(0.9):.1f} m/s; raw fit error v_e median {k.v_e.median():.1f} m/s"
    )

    print("\n-- flag agreement on the echoes kept by the median filter (% of kept echoes)")
    k = k.assign(fitacf=k.gflg, med_rule=k.gflg_m, ribiero=k.gflg_ribiero)
    for a, b in (("fitacf", "med_rule"), ("fitacf", "ribiero"), ("med_rule", "ribiero")):
        print(f"\n{a} (rows) vs {b} (columns)\n{pd.crosstab(k[a], k[b], normalize=True).mul(100).round(1).to_string()}")
    print("\ngflg_conv (rows) vs ribiero (columns):")
    print(pd.crosstab(k.gflg_conv_m, k.gflg_ribiero, normalize=True).mul(100).round(1).to_string())
    for name, col in (
        ("gflg_conv", k.gflg_conv_m),
        ("gflg_kde", k.gflg_kde_m),
        ("fitacf gflg", k.gflg),
        ("ribiero", k.gflg_ribiero),
    ):
        print(f"share of kept echoes by {name:<12}:", (100 * col.value_counts(normalize=True)).round(1).to_dict())


if __name__ == "__main__":
    main()
