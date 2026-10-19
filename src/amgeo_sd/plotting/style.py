"""Shared plot style: parameter ranges and colour maps (from legacy ``fan_rti`` / ``sd_plots``)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """Display range of a per-echo parameter: ``vmin..vmax`` in steps of ``step`` (discrete colour bins)."""

    label: str
    vmin: float
    vmax: float
    step: float
    cmap: str

    @property
    def bounds(self) -> list[float]:
        """Colour-bin edges ``vmin, vmin+step, ..., vmax``."""
        return [float(b) for b in np.arange(self.vmin, self.vmax + self.step / 2, self.step)]


#: Colour scales; velocity follows the SuperDARN convention (red = away from the radar, blue = towards).
PARAMS: dict[str, ParamSpec] = {
    "v": ParamSpec("Velocity [m/s]", -100, 100, 25, "jet_r"),
    "w_l": ParamSpec("Spectral width [m/s]", 0, 200, 20, "viridis"),
    "p_l": ParamSpec("Power [dB]", 0, 30, 5, "plasma"),
    "v_mad": ParamSpec("MAD of velocity [m/s]", 0, 60, 10, "magma"),
}

#: Ground-scatter flag classes: value -> (colour, label). 2 = undetermined for the conv / KDE flags (SAIS in the legacy RTI).
FLAG_CLASSES = {-1: ("black", "US"), 0: ("red", "IS"), 1: ("blue", "GS"), 2: ("green", "undetermined")}


def apply_style() -> None:
    """Small, consistent defaults (serif tick labels as in the legacy figures)."""
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.default": "default",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 100,
        }
    )


def get_cluster_cmap(n_clusters: int, plot_noise: bool = False) -> mcolors.ListedColormap:
    """Qualitative colour map with ``n_clusters`` shuffled colours (same shuffle as the legacy code);
    the first entry is black when ``plot_noise`` so label -1 (noise) is drawn black."""
    base = plt.cm.gist_rainbow
    colors = [base(i) for i in range(base.N)]
    while len(colors) < n_clusters:
        colors.extend(base(i) for i in range(base.N))
    colors = np.array(colors)
    order = np.arange(len(colors))
    random.Random(10).shuffle(order)
    colors = colors[order]
    if plot_noise:
        colors[0] = (0, 0, 0, 1.0)
    return mcolors.ListedColormap(colors, name="cluster")
