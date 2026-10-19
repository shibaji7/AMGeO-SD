"""Scatter filters."""

from amgeo_sd.filters.median import (
    DEFAULT_WEIGHTS,
    EDGE_GATES,
    MedianFilter,
    cluster_scan,
    discard_repeating_beams,
    kde_flag,
)

__all__ = ["DEFAULT_WEIGHTS", "EDGE_GATES", "MedianFilter", "cluster_scan", "discard_repeating_beams", "kde_flag"]
