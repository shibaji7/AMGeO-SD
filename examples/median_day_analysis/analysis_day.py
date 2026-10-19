"""Whole-day analysis of the median-filter model against the raw echoes and the DBSCAN labels.

usage: python analysis_day.py <rad> <dir with day/<rad> and day-dbscan/<rad> outputs>   (run from the project root)
"""

import datetime as dt
import glob
import sys

import numpy as np
import pandas as pd

from amgeo_sd.filters import MedianFilter
from amgeo_sd.io import FitacfReader, load_frame, scans_to_dataframe

KEYS = ["time", "bmnum", "slist"]


def kept_mask(raw, filt):
    m = raw[KEYS].merge(filt[KEYS].assign(k=1), on=KEYS, how="left")
    return m["k"].fillna(0).to_numpy().astype(bool)


def main():
    rad, AN = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(f"tests/data/{rad}_20250208/*.bz2"))
    start, end = dt.datetime(2025, 2, 8), dt.datetime(2025, 2, 8, 23, 59, 59)
    scans = FitacfReader(rad, start, end, files=files).scans()
    raw = scans_to_dataframe(scans)
    med, _ = load_frame([glob.glob(f"{AN}/day/{rad}/*.nc")[0]])
    db, _ = load_frame([glob.glob(f"{AN}/day-dbscan/{rad}/*.nc")[0]])
    print(f"== {rad}: raw {len(raw)} echoes, {raw.bmnum.nunique()} beams, {raw.scnum.max() + 1} scans")

    # DBSCAN class and cluster size of every raw echo (proxy for structure vs isolated noise)
    sizes = db.groupby(["bmnum", "cluster_tag"]).size().rename("csize").reset_index()
    db = db.merge(sizes, on=["bmnum", "cluster_tag"], how="left")
    db.loc[db.cluster_tag < 0, "csize"] = 0
    db["cclass"] = pd.cut(db.csize, [-1, 0, 49, 499, 1e9], labels=["noise", "small(<50)", "medium(50-499)", "large(>=500)"])
    raw = raw.merge(db[KEYS + ["cclass", "gflg_ribiero"]], on=KEYS, how="left")
    raw["kept"] = kept_mask(raw, med)
    raw["hour"] = pd.to_datetime(raw.time).dt.hour
    added = len(med) - raw.kept.sum()
    print(f"median (thresh 0.7): keeps {raw.kept.mean() * 100:.1f}% of raw echoes; output {len(med)} cells, {added} ({100 * added / len(med):.1f}%) are NEW cells")

    pct = lambda s: (100 * s).round(1)  # noqa: E731
    print("\n-- retention by DBSCAN cluster class")
    t = raw.groupby("cclass", observed=True).agg(echoes=("kept", "size"), kept=("kept", "mean"))
    t["kept"] = pct(t.kept)
    t["share"] = pct(t.echoes / len(raw))
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
    print("\n-- retention within each cluster class by gate block, % (shows the gate effect is not just noise)")
    print(raw.pivot_table(index="cclass", columns="gb", values="kept", aggfunc="mean", observed=True).mul(100).round(0).to_string())
    print("\n-- per beam")
    nsc = raw.scnum.nunique()
    rows = {}
    for b, d in raw.groupby("bmnum"):
        rows[b] = {
            "kept%": 100 * d.kept.mean(),
            "noise_share%": 100 * (d.cclass == "noise").mean(),
            "struct_kept%": 100 * d.kept[d.cclass.isin(["medium(50-499)", "large(>=500)"])].mean(),
            "noise_kept%": 100 * d.kept[d.cclass == "noise"].mean(),
            "scans_w_echo%": 100 * d.scnum.nunique() / nsc,
            "echoes/scan": len(d) / nsc,
        }
    print(pd.DataFrame(rows).round(1).to_string())

    print("\n-- threshold sweep (no KDE, no clustering)")
    out_rows = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        f = scans_to_dataframe(MedianFilter(thresh=th, kde=False, cluster=False).filter_scans(scans, n_jobs=4))
        km = kept_mask(raw, f)
        c = raw.cclass.to_numpy()
        out_rows.append({
            "thresh": th, "kept%": 100 * km.mean(), "noise_removed%": 100 * (~km[c == "noise"]).mean(),
            "small_kept%": 100 * km[c == "small(<50)"].mean(), "medium_kept%": 100 * km[c == "medium(50-499)"].mean(),
            "large_kept%": 100 * km[c == "large(>=500)"].mean(), "new_cells": len(f) - km.sum(),
            "new%of_out": 100 * (len(f) - km.sum()) / len(f),
        })
    print(pd.DataFrame(out_rows).round(1).to_string(index=False))
    raw.drop(columns=["time"]).to_pickle(f"{AN}/raw_{rad}.pkl")


if __name__ == "__main__":
    main()
