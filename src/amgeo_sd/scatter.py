"""Ground-/ionospheric-scatter rules applied to clusters of echoes.

The Ribiero et al. (2011) rule classifies a whole *cluster* from how long it lives and how many of its echoes
move faster than a velocity threshold (port of the legacy ``utils.ribiero_gs_flg`` and
``_run_riberio_threshold_on_rad``, with the Burrell et al. 2018 extension for clusters shorter than an hour).
"""

from __future__ import annotations

import numpy as np

__all__ = ["ribiero_flags", "ribiero_gs_flag"]

#: (minimum duration in hours, maximum high/low velocity ratio for ionospheric scatter), checked in order.
#: Clusters shorter than an hour use 0.5 (Burrell et al., 2018); longer than 14 h use 0.15 (an addition of the
#: legacy code, 0.2 in the original paper).
RIBIERO_RATIOS = ((14.0, 0.15), (3.0, 0.2), (2.0, 0.33), (1.0, 0.475), (-np.inf, 0.5))


def ribiero_gs_flag(abs_velocity: np.ndarray, duration_hours: float, vth: float) -> bool:
    """True (ground scatter) or False (ionospheric) for one cluster.

    ``R`` is the number of echoes faster than ``vth`` over the number at or below it (``1`` if none are slow);
    the cluster is ionospheric when ``R`` exceeds the limit belonging to its duration, see ``RIBIERO_RATIOS``.

    Parameters
    ----------
    abs_velocity : ndarray
        Speeds ``|v|`` [m/s] of the cluster's echoes.
    duration_hours : float
        Time span between the first and last echo of the cluster [h].
    vth : float
        Speed [m/s] that separates "fast" from "slow" echoes.

    Returns
    -------
    bool
        ``True`` for ground scatter, ``False`` for ionospheric scatter.
    """
    high = int(np.sum(abs_velocity > vth))
    low = int(np.sum(abs_velocity <= vth))
    ratio = 1.0 if low == 0 else high / low
    for min_hours, limit in RIBIERO_RATIOS:
        if duration_hours > min_hours:
            return not ratio > limit
    return True  # unreachable: the last entry accepts every duration


def ribiero_flags(
    cluster: np.ndarray, velocity: np.ndarray, time: np.ndarray, slist: np.ndarray, vth: float = 25.0, min_gate: int = 7
) -> np.ndarray:
    """Per-echo flag from per-echo cluster labels: ``-1`` unclassified, ``0`` ionospheric, ``1`` ground.

    Echoes with label ``-1`` (DBSCAN noise) and echoes in gates below ``min_gate`` (default 7) are unclassified.
    All echoes of a cluster get the cluster's flag; its duration is the time span of its echoes.
    Pass one beam (and channel) at a time: labels are only unique within a beam.

    Parameters
    ----------
    cluster : ndarray of int
        Cluster label of every echo; ``-1`` is noise.
    velocity : ndarray
        Line-of-sight velocity [m/s] of every echo.
    time : ndarray of datetime64
        Time of every echo.
    slist : ndarray of int
        Range gate of every echo.
    vth : float
        Speed [m/s] separating fast from slow echoes (``[dbscan] velocity_threshold``).
    min_gate : int
        Echoes in gates below this are not classified (``[dbscan] min_gate``).

    Returns
    -------
    ndarray of int8
        One flag per echo: -1 unclassified, 0 ionospheric, 1 ground.
    """
    labels = np.asarray(cluster).copy()
    labels[np.asarray(slist) < min_gate] = -1
    speed, time = np.abs(np.asarray(velocity, dtype=float)), np.asarray(time)
    out = np.zeros(len(labels), dtype=np.int8)
    for lab in np.unique(labels):
        mask = labels == lab
        if lab == -1:
            out[mask] = -1
        else:
            hours = (time[mask].max() - time[mask].min()) / np.timedelta64(1, "h")
            out[mask] = ribiero_gs_flag(speed[mask], float(hours), vth)
    return out
