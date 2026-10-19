"""Weighted 3-scan x 3-beam x 3-gate median filter (the "boxcar" model of fitacf++).

Port of the legacy ``sd/boxcar_filter.py`` (``Filter``). For every range cell of the *middle*
scan of three consecutive scans, the 27 neighbouring cells (previous/current/next scan, beam-1..+1,
gate-1..+1) are weighted with ``DEFAULT_WEIGHTS``. The cell survives if the weight of neighbours that
hold an echo, divided by the weight of all neighbours that exist, is at least ``thresh``. Survivors get

* ``v``, ``w_l``, ``p_l``: weighted medians (each neighbour value repeated ``weight`` times),
* ``v_mad``: median absolute deviation of ``v``,
* ``gflg``: ``0`` (ionospheric) if ``w_l > -3 v + 90`` else ``1`` (ground),
* ``gflg_conv``: weighted fraction of ground-scatter neighbours (over *all* existing neighbour
  weight) mapped to ``0`` (<= ``lth``), ``1`` (>= ``uth``) or ``2`` (undetermined),
* ``gflg_kde``: the neighbours' flags, jittered into (0, 1), are fitted with a Beta distribution and
  ``1 - cdf(pth)`` is mapped the same way,
* ``clabel``: DBSCAN label of the cell in (beam, gate) space within the scan (-1 = noise).

The legacy implementation looped over every cell in Python; this one builds dense arrays so that the
neighbourhood sums are a few array shifts. It is verified against a literal translation of the legacy
loops in ``tests/reference_median.py``.

Differences from the legacy code, all deliberate:

* the KDE jitter uses a seeded ``numpy.random.Generator`` per scan instead of the global state, so
  results do not depend on process scheduling;
* a failed Beta fit gives ``gflg_kde = 2`` (undetermined) instead of raising;
* ``v_e`` of filtered beams is NaN (the legacy code left it empty).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
from collections.abc import Sequence

import numpy as np
from loguru import logger
from scipy.stats import beta
from sklearn.cluster import DBSCAN

from amgeo_sd.models import Scan

_EPOCH = dt.datetime(1970, 1, 1)

#: Weights over (scan, beam, gate); the centre cell has weight 5.
DEFAULT_WEIGHTS = np.array(
    [
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
        [[2, 3, 2], [3, 5, 3], [2, 3, 2]],
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
    ]
)


def discard_repeating_beams(scan: Scan) -> Scan:
    """Keep one beam per beam number and drop beams without echoes.

    Beams are ordered by beam number; when a beam number is sounded more than once in a scan
    (e.g. a themis special beam or two channels) the first one in file order is kept.
    """
    out = Scan(s_mode=scan.s_mode, stime=scan.stime, etime=scan.etime)
    seen: set[int] = set()
    for beam in sorted(scan.beams, key=lambda b: b.bmnum):
        if beam.bmnum not in seen and beam.n_echoes > 0:
            out.beams.append(beam)
            seen.add(beam.bmnum)
    if out.beams:
        out.update()
    return out


def _flag_from_probability(p: float, pbnd: Sequence[float]) -> float:
    if p <= pbnd[0]:
        return 0.0
    if p >= pbnd[1]:
        return 1.0
    return 2.0


def kde_flag(gfx: np.ndarray, pbnd: Sequence[float], pth: float, rng: np.random.Generator) -> float:
    """Ground-scatter flag from the Beta-distribution fit of neighbouring flags.

    ``gfx`` are the 0/1 flags of the (weight-repeated) neighbours. Zeros are replaced by uniform
    draws in [0.01, 0.05] and ones by draws in [0.95, 0.99], a Beta(a, b) with fixed ``loc=0,
    scale=1`` is fitted, and ``1 - cdf(pth)`` (probability of ground scatter) is mapped to
    0 / 1 / 2 with ``pbnd``.
    """
    x = np.asarray(gfx, dtype=float).copy()
    zeros, ones = x == 0, x == 1
    x[zeros] = rng.uniform(low=0.01, high=0.05, size=zeros.sum())
    x[ones] = rng.uniform(low=0.95, high=0.99, size=ones.sum())
    try:
        a, b, _, _ = beta.fit(x, floc=0.0, fscale=1.0)
        p = 1.0 - beta.cdf(pth, a, b, loc=0.0, scale=1.0)
    except Exception as exc:  # degenerate samples can make the optimiser fail
        logger.debug(f"beta fit failed ({exc}); flag set to 2")
        return 2.0
    if not np.isfinite(p):
        return 2.0
    return _flag_from_probability(p, pbnd)


def conv_flag(weighted_ground: float, total_weight: float, pbnd: Sequence[float]) -> float:
    """Conventional flag: ground-scatter weight fraction mapped to 0 / 1 / 2."""
    return _flag_from_probability(weighted_ground / total_weight, pbnd)


def cluster_scan(scan: Scan, eps: float = 2, min_samples: int = 5) -> Scan:
    """Label echoes of one scan with DBSCAN in (beam number, range gate) space.

    The labels are stored per beam in ``beam.extra["clabel"]`` (``-1`` is noise). The scan is
    modified in place and returned.
    """
    points = [np.column_stack([np.full(b.n_echoes, b.bmnum), b.slist]) for b in scan.beams]
    if not points or sum(len(p) for p in points) == 0:
        for b in scan.beams:
            b.extra["clabel"] = np.empty(0, dtype=int)
        return scan
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(np.vstack(points)).labels_
    start = 0
    for b in scan.beams:
        b.extra["clabel"] = labels[start : start + b.n_echoes]
        start += b.n_echoes
    return scan


class MedianFilter:
    """Weighted median filter over three consecutive scans.

    Parameters
    ----------
    thresh : float
        Minimum fraction of neighbour weight that must hold an echo (default 0.7).
    weights : ndarray, shape (3, 3, 3)
        Weights over (scan, beam, gate); integers.
    pbnd : (float, float)
        Lower/upper probability bounds for ionospheric / ground scatter (legacy ``lth``/``uth``).
    pth : float
        Beta-distribution cut for the KDE flag.
    kde : bool
        Compute ``gflg_kde``. The Beta fit dominates the run time; if ``False`` the flag is NaN.
    cluster : bool
        Run DBSCAN (``eps``, ``min_samples``) on each filtered scan to fill ``clabel``.
    seed : int
        Seed for the KDE jitter.
    """

    def __init__(
        self,
        thresh: float = 0.7,
        weights: np.ndarray | None = None,
        pbnd: Sequence[float] = (1.0 / 5.0, 4.0 / 5.0),
        pth: float = 0.25,
        kde: bool = True,
        cluster: bool = True,
        eps: float = 2,
        min_samples: int = 5,
        seed: int = 0,
    ):
        self.thresh = thresh
        self.weights = np.asarray(DEFAULT_WEIGHTS if weights is None else weights, dtype=int)
        if self.weights.shape != (3, 3, 3):
            raise ValueError("weights must have shape (3, 3, 3)")
        self.pbnd, self.pth, self.kde = tuple(pbnd), pth, kde
        self.cluster, self.eps, self.min_samples, self.seed = cluster, eps, min_samples, seed
        # index of the originating neighbour (0..26) for every weight-repeated sample
        self._rep = np.repeat(np.arange(27), self.weights.ravel())

    # ------------------------------------------------------------------ one window
    def filter_window(self, scans: Sequence[Scan]) -> Scan:
        """Filter the middle one of three consecutive scans.

        The scans must already have unique beams (see :func:`discard_repeating_beams`).
        Returns a new :class:`~amgeo_sd.models.Scan`; beams without surviving echoes are omitted.
        """
        if len(scans) != 3:
            raise ValueError("filter_window needs exactly 3 scans")
        centre = scans[1]
        out = Scan(s_mode=centre.s_mode, stime=centre.stime, etime=centre.etime)
        if not centre.beams:
            return out
        nb = 1 + max(b.bmnum for s in scans for b in s.beams)
        ng = max(int(b.slist.max()) + 1 for s in scans for b in s.beams)
        ng = max(ng, max(b.nrang for b in centre.beams))
        fields, present, exists = self._dense(scans, nb, ng)
        pts, tot = self._neighbour_weights(present, exists)

        rng = np.random.default_rng(
            [self.seed, int((centre.stime - _EPOCH).total_seconds() * 1e6) if centre.stime else 0]
        )
        w_flat = self.weights.ravel()
        for beam in sorted(centre.beams, key=lambda b: b.bmnum):
            b = beam.bmnum
            gates = np.arange(beam.nrang)
            keep = gates[(pts[b, : beam.nrang] / tot[b]) >= self.thresh]
            if keep.size == 0:
                continue
            nbr = {k: self._gather(fields[k], b, keep) for k in ("v", "w_l", "p_l", "gflg")}
            v, w_l, p_l = (np.nanmedian(nbr[k][:, self._rep], axis=1) for k in ("v", "w_l", "p_l"))
            v_rep = nbr["v"][:, self._rep]
            v_mad = np.nanmedian(np.abs(v_rep - v[:, None]), axis=1)
            gflg = np.where(w_l > -3.0 * v + 90.0, 0, 1).astype(np.int8)
            g = nbr["gflg"]  # (N, 27), NaN where the neighbour has no echo
            ground_weight = np.nansum(g * w_flat, axis=1)
            conv = np.array([conv_flag(gw, tot[b], self.pbnd) for gw in ground_weight])
            if self.kde:
                g_rep = g[:, self._rep]
                kde = np.array([kde_flag(row[~np.isnan(row)], self.pbnd, self.pth, rng) for row in g_rep])
            else:
                kde = np.full(keep.size, np.nan)
            out.beams.append(
                dataclasses.replace(
                    beam,
                    slist=keep.astype(int),
                    v=v,
                    w_l=w_l,
                    p_l=p_l,
                    v_e=np.full(keep.size, np.nan),
                    gflg=gflg,
                    extra={"v_mad": v_mad, "gflg_conv": conv, "gflg_kde": kde},
                )
            )
        if out.beams:
            out.update()
            if self.cluster:
                cluster_scan(out, self.eps, self.min_samples)
        return out

    # --------------------------------------------------------------- whole sequence
    def filter_scans(self, scans: Sequence[Scan], n_jobs: int = 1) -> list[Scan]:
        """Filter every interior scan of a time-ordered scan list.

        ``scans[0]`` and ``scans[-1]`` only serve as neighbours; the result has
        ``len(scans) - 2`` scans. Beams are de-duplicated first. ``n_jobs > 1`` uses processes.
        """
        scans = [discard_repeating_beams(s) for s in scans]
        windows = [scans[i - 1 : i + 2] for i in range(1, len(scans) - 1)]
        if n_jobs > 1 and len(windows) > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            # "spawn": forking a process that already runs threads (BLAS, pytest) can deadlock
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                return list(pool.map(self.filter_window, windows, chunksize=max(1, len(windows) // (4 * n_jobs))))
        return [self.filter_window(w) for w in windows]

    # ------------------------------------------------------------------- internals
    @staticmethod
    def _dense(scans: Sequence[Scan], nb: int, ng: int):
        fields = {k: np.full((3, nb, ng), np.nan) for k in ("v", "w_l", "p_l", "gflg")}
        present = np.zeros((3, nb, ng), dtype=bool)
        exists = np.zeros((3, nb), dtype=bool)
        for j, scan in enumerate(scans):
            for beam in scan.beams:
                exists[j, beam.bmnum] = True
                present[j, beam.bmnum, beam.slist] = True
                for k in fields:
                    fields[k][j, beam.bmnum, beam.slist] = getattr(beam, k)
        return fields, present, exists

    def _neighbour_weights(self, present: np.ndarray, exists: np.ndarray):
        """Weight of occupied neighbours (``pts[b, g]``) and of existing ones (``tot[b]``)."""
        _, nb, ng = present.shape
        padded = np.pad(present, ((0, 0), (1, 1), (1, 1)))
        pts = np.zeros((nb, ng))
        for j, k, n in itertools.product(range(3), repeat=3):
            pts += self.weights[j, k, n] * padded[j, k : k + nb, n : n + ng]
        padded_exists = np.pad(exists, ((0, 0), (1, 1)))
        tot = np.zeros(nb)
        for j, k in itertools.product(range(3), repeat=2):
            tot += self.weights[j, k].sum() * padded_exists[j, k : k + nb]
        return pts, tot

    @staticmethod
    def _gather(field: np.ndarray, b: int, gates: np.ndarray) -> np.ndarray:
        """(N, 27) neighbour values of cells (b, gates) in the order scan, beam, gate."""
        padded = np.pad(field, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        cols = [padded[j, b + k, gates + n] for j, k, n in itertools.product(range(3), repeat=3)]
        return np.column_stack(cols)
