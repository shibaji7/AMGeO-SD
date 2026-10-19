"""Literal translation of the legacy DBSCAN / GMM / Ribiero code (``_bkp/sd/clustering_algo.py`` ``DBScan`` and
``_bkp/sd/utils.py``), kept slow and unchanged in logic as the oracle for :mod:`amgeo_sd.clustering`.

Legacy defects are *not* reproduced where they crash (GMM on beams with fewer than ``n_data_th`` echoes) and the
GMM function returns the raw component labels so the tests can compare each step on its own.
"""

import numpy as np
from matplotlib.dates import date2num
from scipy.stats import boxcox
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def ribiero_gs_flg(vel, time, vth):
    """Legacy ``utils.ribiero_gs_flg``; ``time`` in days, ordered."""
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > vth)
    low = np.sum(np.abs(vel) <= vth)
    if low == 0:
        R = 1.0
    else:
        R = high / low
    if L > 14.0:
        return not (R > 0.15)
    elif L > 3:
        return not (R > 0.2)
    elif L > 2:
        return not (R > 0.33)
    elif L > 1:
        return not (R > 0.475)
    else:
        return not (R > 0.5)


def run_riberio_threshold_on_rad(u, vth=25.0):
    """Legacy ``utils._run_riberio_threshold_on_rad``."""
    df = u.copy()
    clust_flag = np.array(df.cluster_tag)
    vel = np.hstack(np.abs(df.v))
    t = np.hstack(df.time.apply(lambda x: date2num(x)))
    clust_flag[np.array(df.slist) < 7] = -1
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1:
            gs_flg[clust_mask] = -1
        else:
            gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask], vth)
    df["gflg_ribiero"] = gs_flg
    return df


def run_dbscan(_o, p, themis):
    """Legacy ``DBScan.run_dbscan`` (without overwriting the caller's columns)."""
    slist, scnum = _o.slist / p["eps_g"], _o.scnum / p["eps_s"]
    min_samples = p["min_samples"] * p["tb_mult"] if themis else p["min_samples"]
    ds = DBSCAN(eps=p["eps"], min_samples=min_samples).fit(np.column_stack([slist, scnum]))
    return np.copy(ds.labels_)


def gmm_components(_o, p):
    """Legacy ``gmm_on_existing_cluster`` + ``get_gmm_box_cox`` for feature ``v`` (data large enough)."""
    data = _o.copy()
    vals = np.copy(data["v"])
    vals[np.abs(vals) <= 1e-5] = 1e-5
    vals = boxcox(np.abs(vals))[0]
    vals[np.isnan(vals)] = 0.0
    vals[np.isinf(vals)] = 0.0
    data["v"] = vals
    estimator = GaussianMixture(
        n_components=p["n_clusters"],
        covariance_type=p["cov"],
        max_iter=100,
        reg_covar=1e-3,
        random_state=0,
        n_init=5,
        init_params="kmeans",
    )
    estimator.fit(data[["v"]])
    return estimator.predict(data[["v"]])


def legacy_gmm_dbscan(_o, p, themis):
    """Legacy ``run_gmm_dbscan`` including its constant label offset (labels may collide between components)."""
    comp = gmm_components(_o, p)
    labels = np.copy(comp)
    db_labels = np.zeros_like(labels) * np.nan
    dclust = np.max(labels) + 1
    for ct in np.unique(labels):
        ls = run_dbscan(_o[comp == ct], p, themis)
        ls[ls >= 0] += dclust
        db_labels[labels == ct] = ls
    return db_labels


def reference_cluster(frame, p, scan_info, gmm_beams=()):
    """(cluster_tag, gflg_ribiero) for every row of ``frame`` as ``DBScan.run_2D_cluster_bybeam`` produced them."""
    tag = np.full(len(frame), np.nan)
    flag = np.full(len(frame), np.nan)
    for ch in np.unique(frame.channel):
        fr = frame[frame.channel == ch]
        for b in np.unique(fr.bmnum):
            _o = fr[fr.bmnum == b].copy()
            themis = scan_info["s_mode"] != "normal" and scan_info["t_beam"] == b
            _o["cluster_tag"] = legacy_gmm_dbscan(_o, p, themis) if b in gmm_beams else run_dbscan(_o, p, themis)
            _o = run_riberio_threshold_on_rad(_o, vth=p["velocity_threshold"])
            tag[_o.index] = _o["cluster_tag"]
            flag[_o.index] = _o["gflg_ribiero"]
    return tag, flag
