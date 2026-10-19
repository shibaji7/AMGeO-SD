"""Physical consistency of the scatter flags: ground scatter should be slow (|v| small), ionospheric scatter should move."""

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
    k = med.merge(db[KEYS + ["gflg_ribiero", "cluster_tag"]], on=KEYS, how="inner")
    k = k.merge(raw[KEYS + ["gflg", "v"]].rename(columns={"gflg": "gflg_fitacf", "v": "v_raw"}), on=KEYS, how="left")
    print(f"== {rad}: {len(k)} echoes kept by the median filter that are also in the DBSCAN output")

    def stats(name, col, gs=1, is_=0):
        g, i = k[k[col] == gs], k[k[col] == is_]
        print(
            f"{name:<22} GS: n={len(g):7d} median|v|={g.v.abs().median():5.1f} >50 m/s={100 * (g.v.abs() > 50).mean():5.1f}%   "
            f"IS: n={len(i):7d} median|v|={i.v.abs().median():5.1f} <10 m/s={100 * (i.v.abs() < 10).mean():5.1f}%  median w={i.w_l.median():5.1f}"
        )

    print("median |v| uses the filtered velocity; raw v for the fitacf flag")
    k_raw = k.assign(v=k.v_raw)
    k_save = k
    k = k_raw
    stats("fitacf gflg (raw v)", "gflg_fitacf")
    k = k_save
    stats("median rule gflg", "gflg")
    stats("gflg_conv (0/1 only)", "gflg_conv")
    stats("gflg_kde (0/1 only)", "gflg_kde")
    stats("Ribiero cluster flag", "gflg_ribiero")
    u = k[k.gflg_conv == 2]
    print(f"gflg_conv undetermined (2): n={len(u)}  median|v|={u.v.abs().median():.1f}  Ribiero says IS {100 * (u.gflg_ribiero == 0).mean():.0f}% GS {100 * (u.gflg_ribiero == 1).mean():.0f}% unclassified {100 * (u.gflg_ribiero == -1).mean():.0f}%")
    both = k[(k.gflg == 1) & (k.gflg_ribiero == 0)]
    print(f"\nmedian rule says GS but Ribiero says IS: n={len(both)} ({100 * len(both) / len(k):.1f}% of kept): median|v|={both.v.abs().median():.1f}, >25 m/s: {100 * (both.v.abs() > 25).mean():.0f}%, median w_l={both.w_l.median():.1f}")
    both = k[(k.gflg == 1) & (k.gflg_ribiero == 1)]
    print(f"both say GS: n={len(both)}: median|v|={both.v.abs().median():.1f}, >25 m/s: {100 * (both.v.abs() > 25).mean():.0f}%, median w_l={both.w_l.median():.1f}")
    both = k[(k.gflg == 0) & (k.gflg_ribiero == 0)]
    print(f"both say IS: n={len(both)}: median|v|={both.v.abs().median():.1f}, <10 m/s: {100 * (both.v.abs() < 10).mean():.0f}%, median w_l={both.w_l.median():.1f}")


if __name__ == "__main__":
    main()
