"""Plotting for fitacf and fitacf++ data: RTI (``rti``) and field-of-view / fan maps (``fov``)."""

from amgeo_sd.plotting.fov import FanPlots, plot_fan, select_scan
from amgeo_sd.plotting.rti import RangeTimeIntervalPlot, plot_rti, plot_rti_compare
from amgeo_sd.plotting.style import apply_style, get_cluster_cmap

__all__ = [
    "FanPlots",
    "apply_style",
    "get_cluster_cmap",
    "RangeTimeIntervalPlot",
    "plot_fan",
    "plot_rti",
    "plot_rti_compare",
    "select_scan",
]
