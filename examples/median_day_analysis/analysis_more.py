"""Edge gates, ionospheric-scatter bias, velocity change and flag agreement (run from the project root)."""

import glob
import sys

import numpy as np
import pandas as pd

from amgeo_sd.io import load_frame

KEYS = ["time", "bmnum", "slist"]


def main():
    rad, AN = sys.argv[1], sys.argv[2]
    raw, _ = load_frame([f"tests/data/{rad}_20250208"])
    med, _ = load_frame([glob.glob(f"{AN}/day/{rad}/*.nc")[0]])
    db, _ = load_frame([glob.glob(f"{AN}/day-dbscan/{rad}/*.nc")[0]])
    sizes = db.groupby(["bmnum", "cluster_tag"]).size().rename("csize").reset_index()
    db = db.merge(sizes, on=["bmnum", "cluster_tag"], how="left")
    db["struct"] = (db.cluster_tag >= 0) & (db.csize >= 50)
    raw = raw.merge(db[KEYS + ["struct", "gflg_ribiero", "cluster_tag"]], on=KEYS, how="left")
    both = raw.merge(med[KEYS + ["v", "w_l", "p_l", "gflg", "gflg_conv", "gflg_kde", "v_mad"]].rename(columns=lambda c: c if c in KEYS else c + "_m"), on=KEYS, how="left")
    both["kept"] = both["v_m"].notna()
    nrang = int(raw.nrang.max())
    print(f"== {rad}  (nrang {nrang})")

    s = both[both.struct]
    print("\n-- retention of STRUCTURE echoes (cluster >= 50 echoes) by gate at the range edges, %")
    g = s.groupby("slist").kept.mean().mul(100).round(0)
    print("gates 0-4:", g.reindex(range(0, 5)).tolist(), "| gates", nrang - 5, "-", nrang - 1, ":", g.reindex(range(nrang - 5, nrang)).tolist())
    print("interior gates 10-60 mean: %.0f%%" % g.loc[10:60].mean())

    print("\n-- retention of structure echoes by fitacf flag (does the filter prefer ground scatter?), %")
    print(s.groupby("gflg").kept.agg(["size", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(1)).to_string())
    print("   by Ribiero cluster flag:")
    print(s.groupby("gflg_ribiero").kept.agg(["size", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(1)).to_string())

    k = both[both.kept]
    dv = (k.v_m - k.v).abs()
    print(f"\n-- velocity change on kept echoes: |v_median - v_raw|  median {dv.median():.1f}  mean {dv.mean():.1f}  90% {dv.quantile(.9):.1f}  >10 m/s {100 * (dv > 10).mean():.1f}%  >25 m/s {100 * (dv > 25).mean():.1f}%")
    print(f"   v_mad of kept echoes: median {k.v_mad_m.median():.1f}  90% {k.v_mad_m.quantile(.9):.1f} m/s; raw fit error v_e median {k.v_e.median():.1f} m/s")

    print("\n-- flag agreement on the echoes kept by the median filter (rows = flag A, columns = flag B), counts in %")
    k = k.assign(fitacf=k.gflg, med_rule=k.gflg_m, ribiero=k.gflg_ribiero)

    def table(a, b):
        t = pd.crosstab(k[a], k[b], normalize=True).mul(100).round(1)
        print(f"\n{a}  vs  {b}\n{t.to_string()}")

    table("fitacf", "med_rule")
    table("fitacf", "ribiero")
    table("med_rule", "ribiero")
    print("\ngflg_conv (median neighbourhood fraction) vs ribiero:")
    print(pd.crosstab(k.gflg_conv_m, k.gflg_ribiero, normalize=True).mul(100).round(1).to_string())
    print("\nshare of kept echoes by gflg_conv  :", (100 * k.gflg_conv_m.value_counts(normalize=True)).round(1).to_dict())
    print("share of kept echoes by gflg_kde   :", (100 * k.gflg_kde_m.value_counts(normalize=True)).round(1).to_dict())
    print("share of kept echoes by fitacf gflg:", (100 * k.gflg.value_counts(normalize=True)).round(1).to_dict(), "| ribiero:", (100 * k.gflg_ribiero.value_counts(normalize=True)).round(1).to_dict())


if __name__ == "__main__":
    main()
