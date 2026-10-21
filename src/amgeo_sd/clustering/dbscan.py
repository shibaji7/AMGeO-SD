"""DBSCAN / Gaussian-mixture clustering model (legacy ``DBScan`` in ``sd/clustering_algo.py``, ``fitacf_amgeo_dbcluster.py``).

For every (channel, beam) the echoes of all scans are clustered in the plane (range gate / ``eps_g``, scan number / ``eps_s``)
with DBSCAN (``eps``, ``min_samples``; ``min_samples * tb_mult`` on the themis special beam). Optionally the velocities of a
beam are first split into ``n_clusters`` Gaussian-mixture components (on Box-Cox transformed ``|v|``) and DBSCAN is run inside each
component. Each cluster is then classified as ionospheric or ground scatter by the Ribiero rule
(:func:`amgeo_sd.scatter.ribiero_flags`). Results are stored per echo in ``Beam.extra``:

* ``cluster_tag``: DBSCAN label, ``-1`` = noise. **Labels are only unique within one beam** (and, with GMM, are made
  unique across components).
* ``gflg_ribiero``: ``-1`` unclassified (noise, or gates below ``min_gate``), ``0`` ionospheric, ``1`` ground.

No echo is removed or modified; the model only labels the raw fitacf echoes.

Differences from the legacy code, all deliberate:

* the defaults are ``eps = 2`` and ``min_samples = 8`` (legacy: 1 and 5). With the legacy values patchy but physically continuous
  structure falls apart into dozens of 3-12 echo fragments that the Ribiero rule then classifies one by one (usually as ground
  scatter because slow); the larger radius links the fragments to their parent cluster;
* the scaled (gate / ``eps_g``, scan / ``eps_s``) features are used for clustering only, the legacy code overwrote the
  ``slist`` and ``scnum`` columns with them (no effect for the default ``eps_g = eps_s = 1``);
* with GMM, cluster labels of different components no longer collide (the legacy code added the same offset to every
  component, merging their clusters in the Ribiero step);
* with GMM, a beam with fewer than ``n_data_th`` echoes is clustered as a single component (the legacy code raised an
  ``AttributeError`` there).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from amgeo_sd.config import Dbscan
from amgeo_sd.io.fitacf import scans_to_dataframe
from amgeo_sd.models import Scan
from amgeo_sd.scatter import ribiero_flags


def parse_gmm_beams(spec: str, beams: Sequence[int], t_beam: int = -1) -> set[int]:
    """Beams that get the Gaussian-mixture step: ``""`` none, ``"A"`` all, ``"T"`` the themis beam, ``"3,7"`` a list.

    Parameters
    ----------
    spec : str
        The ``[dbscan] gmm`` setting.
    beams : sequence of int
        Beam numbers present in the data (used by ``"A"``).
    t_beam : int
        Themis special beam number, ``-1`` if none (used by ``"T"``).

    Returns
    -------
    set of int
        The selected beam numbers.
    """
    spec = spec.strip()
    if not spec:
        return set()
    if "A" in spec.upper():
        return {int(b) for b in beams}
    if "T" in spec.upper():
        return {t_beam} if t_beam >= 0 else set()
    return {int(x) for x in spec.split(",") if x.strip()}


def box_cox_abs(values: np.ndarray) -> np.ndarray:
    """Box-Cox transform of ``|values|`` (floor 1e-5); non-finite results and constant input become 0.

    Parameters
    ----------
    values : ndarray
        Velocities [m/s]; only their magnitude is used.

    Returns
    -------
    ndarray
        The transformed values, same length as ``values``.
    """
    x = np.abs(np.asarray(values, dtype=float))
    x[x <= 1e-5] = 1e-5
    try:
        y = boxcox(x)[0]
    except ValueError:  # constant data
        y = np.zeros_like(x)
    y[~np.isfinite(y)] = 0.0
    return y


def gmm_components(velocity: np.ndarray, p: Dbscan) -> np.ndarray:
    """Gaussian-mixture component of every echo (all 0 when the beam has fewer than ``n_data_th`` echoes).

    Parameters
    ----------
    velocity : ndarray
        Velocities [m/s] of the echoes of one beam; the mixture is fitted to the Box-Cox transform of their magnitude.
    p : Dbscan
        Settings; uses ``n_clusters`` (components), ``cov`` (covariance type) and ``n_data_th`` (minimum number of
        echoes for the fit).

    Returns
    -------
    ndarray of int
        Component number (0 to ``n_clusters - 1``) of every echo.
    """
    if len(velocity) < p.n_data_th:
        return np.zeros(len(velocity), dtype=int)
    x = box_cox_abs(velocity).reshape(-1, 1)
    gm = GaussianMixture(
        n_components=p.n_clusters,
        covariance_type=p.cov,
        max_iter=100,
        reg_covar=1e-3,
        random_state=0,
        n_init=5,
        init_params="kmeans",
    )
    return gm.fit(x).predict(x)


def _dbscan(gate: np.ndarray, scan: np.ndarray, p: Dbscan, themis: bool) -> np.ndarray:
    if len(gate) == 0:
        return np.empty(0, dtype=int)
    min_samples = p.min_samples * p.tb_mult if themis else p.min_samples
    features = np.column_stack([gate / p.eps_g, scan / p.eps_s])
    return DBSCAN(eps=p.eps, min_samples=min_samples).fit(features).labels_


def cluster_beam(df: pd.DataFrame, p: Dbscan, use_gmm: bool = False, themis: bool = False) -> np.ndarray:
    """Cluster labels of the echoes of one beam (``df`` has ``slist``, ``scnum`` and, for GMM, ``v``).

    Parameters
    ----------
    df : pandas.DataFrame
        Echoes of one beam and channel with the columns ``slist`` (range gate), ``scnum`` (scan number) and, when
        ``use_gmm``, ``v`` (velocity).
    p : Dbscan
        Settings; uses ``eps``, ``eps_g``, ``eps_s``, ``min_samples``, ``tb_mult`` and, for GMM, ``n_clusters``,
        ``cov`` and ``n_data_th``.
    use_gmm : bool
        Split the echoes into velocity components first and cluster inside each.
    themis : bool
        The beam is the themis special beam: ``min_samples`` is multiplied by ``tb_mult`` because it is sounded
        more often.

    Returns
    -------
    ndarray of int
        One label per row of ``df``; ``-1`` is noise. With GMM the labels of different components do not collide.
    """
    gate, scan = df["slist"].to_numpy(float), df["scnum"].to_numpy(float)
    if not use_gmm:
        return _dbscan(gate, scan, p, themis)
    comp = gmm_components(df["v"].to_numpy(float), p)
    labels = np.full(len(df), -1, dtype=int)
    offset = int(comp.max()) + 1 if len(comp) else 0
    for c in np.unique(comp):
        m = comp == c
        sub = _dbscan(gate[m], scan[m], p, themis)
        found = sub >= 0
        sub[found] += offset
        offset += int(sub.max()) + 1 if found.any() else 0
        labels[m] = sub
    return labels


def cluster_scans(scans: Sequence[Scan], scan_info: dict, p: Dbscan) -> Sequence[Scan]:
    """Add ``cluster_tag`` and ``gflg_ribiero`` to the echoes of ``scans`` (modified in place and returned).

    Each (channel, beam) is clustered on its own in the plane (range gate / ``eps_g``, scan number / ``eps_s``), optionally
    after splitting its velocities into Gaussian-mixture components; the cluster's speed and lifetime then decide
    ionospheric or ground scatter (:func:`~amgeo_sd.scatter.ribiero_flags`). Cluster labels are unique within a beam only.
    No echo is removed or changed.

    Parameters
    ----------
    scans : sequence of Scan
        Raw (unfiltered) scans; ``Beam.extra`` is modified in place.
    scan_info : dict
        Result of :func:`amgeo_sd.io.estimate_scan_info` (``s_mode``, ``t_beam``); selects the themis special beam.
    p : Dbscan
        Settings; uses ``eps``, ``eps_g``, ``eps_s``, ``min_samples``, ``tb_mult``, ``gmm``, ``n_clusters``, ``cov``,
        ``n_data_th``, ``velocity_threshold`` and ``min_gate``. See the [parameter reference](../PARAMETERS.md).

    Returns
    -------
    sequence of Scan
        The same scans, each beam with ``cluster_tag`` (-1 noise) and ``gflg_ribiero`` (-1 unclassified, 0
        ionospheric, 1 ground) in ``extra``.
    """
    frame = scans_to_dataframe(scans)
    tag = np.full(len(frame), -1, dtype=int)
    flag = np.full(len(frame), -1, dtype=np.int8)
    themis_mode = scan_info["s_mode"] != "normal"
    gmm_beams = parse_gmm_beams(p.gmm, np.unique(frame["bmnum"]) if len(frame) else [], scan_info["t_beam"])
    for _, chan in frame.groupby("channel"):
        for b, rows in chan.groupby("bmnum"):
            pos = rows.index.to_numpy()
            labels = cluster_beam(rows, p, use_gmm=b in gmm_beams, themis=themis_mode and b == scan_info["t_beam"])
            tag[pos] = labels
            flag[pos] = ribiero_flags(
                labels,
                rows["v"].to_numpy(),
                rows["time"].to_numpy(),
                rows["slist"].to_numpy(),
                p.velocity_threshold,
                p.min_gate,
            )
    groups = frame.groupby(["scnum", "sbnum"]).indices if len(frame) else {}
    for i, scan in enumerate(scans):
        for j, beam in enumerate(scan.beams):
            idx = groups.get((i, j))
            if idx is None:
                beam.extra["cluster_tag"] = np.empty(0)
                beam.extra["gflg_ribiero"] = np.empty(0)
            else:
                beam.extra["cluster_tag"] = tag[idx].astype(float)
                beam.extra["gflg_ribiero"] = flag[idx].astype(float)
    return scans
