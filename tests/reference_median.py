"""Literal translation of the legacy ``Filter.doFilter`` loops (``_bkp/sd/boxcar_filter.py``).

Slow on purpose: it is the oracle the vectorised :class:`amgeo_sd.filters.MedianFilter` is
tested against. Only the container types (``Beam`` dataclass instead of ad-hoc attributes) and the
random generator (shared with the new code so that the KDE jitter matches) differ from the original.
"""

import datetime as dt

import numpy as np

from amgeo_sd.filters.median import DEFAULT_WEIGHTS, conv_flag, kde_flag

_EPOCH = dt.datetime(1970, 1, 1)


def reference_filter(scans, thresh=0.7, pbnd=(0.2, 0.8), pth=0.25, seed=0, kde=True):
    """Return {bmnum: dict(slist, v, w_l, p_l, v_mad, gflg, gflg_conv, gflg_kde)} for scans[1]."""
    w = DEFAULT_WEIGHTS
    rng = np.random.default_rng([seed, int((scans[1].stime - _EPOCH).total_seconds() * 1e6)])
    result = {}
    for b in scans[1].beams:
        bmnum = b.bmnum
        rec = {k: [] for k in ("slist", "v", "w_l", "p_l", "v_mad", "gflg", "gflg_conv", "gflg_kde")}
        for r in range(0, b.nrang):
            box = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]
            for j in range(0, 3):
                for k in range(-1, 2):
                    for n in range(-1, 2):
                        s = scans[j]
                        tbm = None
                        for bm in s.beams:
                            if bm.bmnum == bmnum + k:
                                tbm = bm
                        if tbm is None:
                            continue
                        if r + n in tbm.slist:
                            ind = np.array(tbm.slist).tolist().index(r + n)
                            box[j][k + 1][n + 1] = (tbm.v[ind], tbm.w_l[ind], tbm.p_l[ind], tbm.gflg[ind])
                        else:
                            box[j][k + 1][n + 1] = 0
            pts = 0.0
            tot = 0.0
            v, w_l, p_l, gfx = [], [], [], []
            for j in range(0, 3):
                for k in range(0, 3):
                    for n in range(0, 3):
                        bx = box[j][k][n]
                        if bx is None:
                            continue
                        wt = w[j][k][n]
                        tot += wt
                        if not (isinstance(bx, int) and bx == 0):
                            pts += wt
                            for _ in range(0, wt):
                                v.append(bx[0])
                                w_l.append(bx[1])
                                p_l.append(bx[2])
                                gfx.append(bx[3])
            if pts / tot >= thresh:
                rec["slist"].append(r)
                rec["v"].append(np.median(v))
                rec["w_l"].append(np.median(w_l))
                rec["p_l"].append(np.median(p_l))
                rec["v_mad"].append(np.median(np.abs(np.array(v) - np.median(v))))
                rec["gflg"].append(0 if np.median(w_l) > -3.0 * np.median(v) + 90.0 else 1)
                rec["gflg_conv"].append(conv_flag(np.nansum(gfx), tot, pbnd))
                rec["gflg_kde"].append(kde_flag(np.array(gfx), pbnd, pth, rng) if kde else np.nan)
        if rec["slist"]:
            result[bmnum] = {k: np.array(x) for k, x in rec.items()}
    return result
