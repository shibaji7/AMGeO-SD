"""Physical consistency of the scatter flags: ground scatter should be slow, ionospheric scatter should move.

Run from the project root, see README.md:
    python examples/median_day_analysis/analysis_flags.py --rad bks --work data/analysis/bks
"""

from amgeo_sd.io import FitacfReader, load_frame, scans_to_dataframe

from common import KEYS, data_files, nc_file, parse_args, window


def main(argv=None):
    args = parse_args(__doc__.splitlines()[0], argv)
    start, end = window(args)
    raw = scans_to_dataframe(FitacfReader(args.rad, start, end, files=data_files(args)).scans())
    med, _ = load_frame([nc_file(args, args.median_sim)])
    db, _ = load_frame([nc_file(args, args.dbscan_sim)])
    k = med.merge(db[KEYS + ["gflg_ribiero", "cluster_tag"]], on=KEYS, how="inner")
    k = k.merge(raw[KEYS + ["gflg", "v"]].rename(columns={"gflg": "gflg_fitacf", "v": "v_raw"}), on=KEYS, how="left")
    print(f"== {args.rad}: {len(k)} echoes kept by the median filter that are also in the DBSCAN output")

    def stats(name, data, col, speed):
        g, i = data[data[col] == 1], data[data[col] == 0]
        print(
            f"{name:<22} GS: n={len(g):7d} median|v|={g[speed].abs().median():5.1f} >50 m/s={100 * (g[speed].abs() > 50).mean():5.1f}%   "
            f"IS: n={len(i):7d} median|v|={i[speed].abs().median():5.1f} <10 m/s={100 * (i[speed].abs() < 10).mean():5.1f}%  median w={i.w_l.median():5.1f}"
        )

    print("speed = filtered velocity (raw velocity for the fitacf flag)")
    stats("fitacf gflg (raw v)", k, "gflg_fitacf", "v_raw")
    stats("median rule gflg", k, "gflg", "v")
    stats("gflg_conv (0/1 only)", k, "gflg_conv", "v")
    stats("gflg_kde (0/1 only)", k, "gflg_kde", "v")
    stats("Ribiero cluster flag", k, "gflg_ribiero", "v")
    u = k[k.gflg_conv == 2]
    print(
        f"gflg_conv undetermined (2): n={len(u)}  median|v|={u.v.abs().median():.1f}  Ribiero says IS {100 * (u.gflg_ribiero == 0).mean():.0f}% "
        f"GS {100 * (u.gflg_ribiero == 1).mean():.0f}% unclassified {100 * (u.gflg_ribiero == -1).mean():.0f}%"
    )
    for label, sel, test in (
        ("median rule says GS but Ribiero says IS", (k.gflg == 1) & (k.gflg_ribiero == 0), "fast"),
        ("both say GS", (k.gflg == 1) & (k.gflg_ribiero == 1), "fast"),
        ("both say IS", (k.gflg == 0) & (k.gflg_ribiero == 0), "slow"),
    ):
        d = k[sel]
        extra = (
            f">25 m/s: {100 * (d.v.abs() > 25).mean():.0f}%"
            if test == "fast"
            else f"<10 m/s: {100 * (d.v.abs() < 10).mean():.0f}%"
        )
        print(
            f"\n{label}: n={len(d)} ({100 * len(d) / len(k):.1f}% of kept): median|v|={d.v.abs().median():.1f}, {extra}, median w_l={d.w_l.median():.1f}"
        )


if __name__ == "__main__":
    main()
