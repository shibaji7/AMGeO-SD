"""Core data containers: one radar sounding (:class:`Beam`) and one scan (:class:`Scan`).

A SuperDARN *beam* is a single integration period in one look direction. A *scan*
is the sequence of beams that sweeps the full field of view (typically 1 or 2 min,
3 min for some modes). Per-gate arrays (``slist``, ``v``, ...) only hold range
gates that contain scatter, so they all share one length: the number of echoes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

#: Per-echo arrays; each has length ``len(beam.slist)``.
GATE_FIELDS = ("slist", "v", "v_e", "w_l", "p_l", "gflg")


def _empty_float() -> np.ndarray:
    return np.empty(0, dtype=float)


@dataclass
class Beam:
    """One beam sounding with its scatter echoes.

    Attributes
    ----------
    time : datetime.datetime
        Start time of the integration (UTC, naive).
    bmnum, channel, cp, tfreq, nrang, rsep, frang, smsep, lagfr, mppul :
        Radar/experiment metadata as stored in the fitacf record.
    noise_sky : float
        Sky noise level.
    intt_sc, intt_us : int
        Integration time, seconds and microseconds.
    fit_version : str
        Fit algorithm revision from the file, e.g. ``"2.5"`` (fitacf) or ``"3.0"`` (fitacf3); ``""`` if absent.
    scan : int
        ``1`` if this beam starts a scan, else ``0``. Fitacf marks abnormal scan
        starts with ``-1``; these are folded into ``1``.
    slist : ndarray of int
        Range gates with scatter.
    v, v_e, w_l, p_l : ndarray of float
        Line-of-sight velocity [m/s], its error, spectral width [m/s] and power [dB].
    gflg : ndarray of int8
        Ground-scatter flag from the fit (``1`` ground, ``0`` ionospheric,
        ``-1`` if absent in the file).
    extra : dict
        Free-form per-echo arrays added by downstream algorithms (``v_mad``,
        cluster labels, ...). Each must have length ``len(slist)``.
    """

    time: dt.datetime
    bmnum: int
    channel: int = 0
    cp: int = 0
    tfreq: float = np.nan
    noise_sky: float = np.nan
    nrang: int = 0
    rsep: float = np.nan
    frang: float = np.nan
    smsep: float = np.nan
    lagfr: float = np.nan
    mppul: int = 0
    intt_sc: int = 0
    intt_us: int = 0
    scan: int = 0
    fit_version: str = ""
    slist: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    v: np.ndarray = field(default_factory=_empty_float)
    v_e: np.ndarray = field(default_factory=_empty_float)
    w_l: np.ndarray = field(default_factory=_empty_float)
    p_l: np.ndarray = field(default_factory=_empty_float)
    gflg: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_echoes(self) -> int:
        """Number of range gates with scatter."""
        return len(self.slist)


@dataclass
class Scan:
    """An ordered group of beams forming one scan.

    ``stime``/``etime`` are the earliest/latest beam times, and ``freq``/``nsky``
    the mean transmit frequency and sky noise; all are refreshed by :meth:`update`.
    """

    beams: list[Beam] = field(default_factory=list)
    s_mode: str = "normal"
    stime: dt.datetime | None = None
    etime: dt.datetime | None = None
    freq: float = np.nan
    nsky: float = np.nan

    def update(self) -> Scan:
        """Refresh ``stime``, ``etime`` and mean ``freq``/``nsky`` from the beams."""
        if self.beams:
            self.stime = min(b.time for b in self.beams)
            self.etime = max(b.time for b in self.beams)
            self.freq = float(np.mean([b.tfreq for b in self.beams]))
            self.nsky = float(np.mean([b.noise_sky for b in self.beams]))
        return self
