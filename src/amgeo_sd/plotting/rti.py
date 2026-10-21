"""Range-time-intensity (RTI) plots for fitacf and fitacf++ tables.

Port of the legacy ``RangeTimeIntervalPlot`` (``sd/fan_rti.py``, ``sd/sd_plots.py``, ``sd/plotlib.py``) and of the
comparison script ``rti.py``. Input is the echo table of :func:`amgeo_sd.io.load_frame` (one row per echo with
``time, bmnum, slist`` and the parameters), whatever file format it came from.

Legacy -> new names: ``addParamPlot`` -> :meth:`add_param`, ``addCluster`` -> :meth:`add_clusters`,
``addGSIS`` -> :meth:`add_gsis`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from amgeo_sd.plotting.grid import grid_time_gate
from amgeo_sd.plotting.style import FLAG_CLASSES, PARAMS, apply_style, get_cluster_cmap


class RangeTimeIntervalPlot:
    """A figure with stacked RTI panels for one beam.

    Parameters
    ----------
    nrang : int
        Number of range gates shown (y axis 0..nrang).
    xlim : (datetime, datetime), optional
        Time axis limits (UT). Defaults to the span of the first table added.
    fig_title : str
        Figure title, drawn at the top left.
    num_subplots : int
        Number of panels; add them with the ``add_*`` methods.
    figsize : (float, float), optional
        Figure size in inches; default width 9, height 2.6 per panel.
    dpi : int
        Resolution of the figure on screen.

    Examples
    --------
    >>> rti = RangeTimeIntervalPlot(nrang=75, num_subplots=2)
    >>> rti.add_param(frame, beam=12, title="(a) velocity", param="v")
    >>> rti.add_gsis(frame, beam=12, title="(b) flag", flag="gflg_kde", xlabel="Time [UT]")
    >>> rti.save("rti.png")
    """

    def __init__(
        self, nrang: int = 75, xlim=None, fig_title: str = "", num_subplots: int = 3, figsize=None, dpi: int = 100
    ):
        apply_style()
        self.nrang, self.num_subplots = nrang, num_subplots
        self.xlim = None if xlim is None else (mdates.date2num(xlim[0]), mdates.date2num(xlim[1]))
        self._created = 0
        self.fig = plt.figure(figsize=figsize or (9, 2.6 * num_subplots), dpi=dpi)
        self.fig.suptitle(fig_title, x=0.08, y=0.995, ha="left", fontweight="bold", fontsize=13)
        self.fig.subplots_adjust(left=0.08, right=0.86, top=0.93 if fig_title else 0.97, bottom=0.07, hspace=0.45)

    # ----------------------------------------------------------------- helpers
    def _axis(self, df: pd.DataFrame, xlabel: str, ylabel: str, title: str):
        self._created += 1
        if self._created > self.num_subplots:
            raise ValueError(f"figure was created with {self.num_subplots} panels")
        ax = self.fig.add_subplot(self.num_subplots, 1, self._created)
        if self.xlim is None:
            t = mdates.date2num(df["time"].values)
            self.xlim = (float(t.min()), float(t.max()))
        ax.set_xlim(self.xlim)
        ax.set_ylim(0, self.nrang)
        span = self.xlim[1] - self.xlim[0]
        if span <= 1.0:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            loc = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        return ax

    def _colorbar(self, ax, mappable, label: str, ticks=None):
        pos = ax.get_position()
        cax = self.fig.add_axes([pos.x1 + 0.015, pos.y0, 0.014, pos.height])
        cb = self.fig.colorbar(mappable, cax=cax, ticks=ticks)
        cb.set_label(label)
        return cb

    @staticmethod
    def _beam(df: pd.DataFrame, beam: int) -> pd.DataFrame:
        d = df[df["bmnum"] == beam]
        if d.empty:
            raise ValueError(f"no echoes for beam {beam}")
        return d

    # ----------------------------------------------------------------- panels
    def add_param(
        self,
        df: pd.DataFrame,
        beam: int,
        title: str = "",
        param: str = "v",
        vmin=None,
        vmax=None,
        step=None,
        cmap: str | None = None,
        label: str | None = None,
        xlabel: str = "",
        ylabel: str = "Range gate",
    ):
        """Colour-coded ``param`` (``v``, ``w_l``, ``p_l``, ``v_mad`` or any numeric column) versus time and gate.

        Parameters
        ----------
        df : pandas.DataFrame
            Echo table (:func:`amgeo_sd.io.load_frame`) with ``time``, ``bmnum``, ``slist`` and ``param``.
        beam : int
            Beam number to draw; ``ValueError`` when it has no echoes.
        title : str
            Panel title (drawn at the left).
        param : str
            Column to colour by. ``v``, ``w_l``, ``p_l`` and ``v_mad`` have preset ranges and colour maps; any other
            numeric column needs ``vmin`` and ``vmax``.
        vmin, vmax : float, optional
            Colour range. Defaults: velocity -100..100 m/s, width 0..200 m/s, power 0..30 dB, v_mad 0..60 m/s.
        step : float, optional
            Width of a colour bin; default per parameter, or (``vmax`` - ``vmin``) / 8 for other columns.
        cmap : str, optional
            Matplotlib colour map name; default per parameter (``jet_r``, ``viridis``, ``plasma``, ``magma``).
        label : str, optional
            Colour-bar label; default the parameter's name and unit.
        xlabel : str
            x-axis label.
        ylabel : str
            y-axis label.

        Returns
        -------
        matplotlib.axes.Axes
            The new panel.
        """
        spec = PARAMS.get(param)
        if spec is None and (vmin is None or vmax is None):
            raise ValueError(f"unknown parameter {param!r}: give vmin, vmax and step or use one of {sorted(PARAMS)}")
        vmin, vmax = (spec.vmin if vmin is None else vmin), (spec.vmax if vmax is None else vmax)
        step = spec.step if step is None else (step or (vmax - vmin) / 8)
        bounds = np.arange(vmin, vmax + step / 2, step)
        cm = plt.get_cmap(cmap or (spec.cmap if spec else "viridis"))
        norm = mcolors.BoundaryNorm(bounds, cm.N, extend="both")
        d = self._beam(df, beam)
        ax = self._axis(df, xlabel, ylabel, title)
        x, y, z = grid_time_gate(d, param, self.nrang)
        qm = ax.pcolormesh(x, y, np.ma.masked_invalid(z.T), cmap=cm, norm=norm, shading="nearest", rasterized=True)
        self._colorbar(ax, qm, label or (spec.label if spec else param), ticks=bounds[:: max(1, len(bounds) // 9)])
        return ax

    def add_gsis(
        self,
        df: pd.DataFrame,
        beam: int,
        title: str = "",
        flag: str = "gflg",
        xlabel: str = "",
        ylabel: str = "Range gate",
    ):
        """Ionospheric / ground scatter classification from the flag column ``flag``.

        IS (ionospheric) is red, GS (ground) blue, US (unclassified, -1) black and undetermined (2) green.

        Parameters
        ----------
        df : pandas.DataFrame
            Echo table with ``time``, ``bmnum``, ``slist`` and the flag column.
        beam : int
            Beam number to draw.
        title : str
            Panel title.
        flag : str
            Flag column: ``gflg`` (fit / legacy median rule), ``gflg_conv``, ``gflg_kde`` (median model) or
            ``gflg_ribiero`` (DBSCAN model).
        xlabel : str
            x-axis label.
        ylabel : str
            y-axis label.

        Returns
        -------
        matplotlib.axes.Axes
            The new panel.
        """
        d = self._beam(df, beam)
        present = sorted({int(v) for v in d[flag].dropna().unique() if int(v) in FLAG_CLASSES})
        if not present:
            raise ValueError(f"column {flag!r} has no valid flag values")
        lo, hi = present[0], present[-1]
        classes = list(range(lo, hi + 1))
        cmap = mcolors.ListedColormap([FLAG_CLASSES[c][0] for c in classes])
        norm = mcolors.BoundaryNorm(np.arange(lo, hi + 2) - 0.5, cmap.N)
        ax = self._axis(df, xlabel, ylabel, title)
        x, y, z = grid_time_gate(d, flag, self.nrang)
        ax.pcolormesh(x, y, np.ma.masked_invalid(z.T), cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.legend(
            handles=[mpatches.Patch(color=FLAG_CLASSES[c][0], label=FLAG_CLASSES[c][1]) for c in present],
            loc="lower right",
            fontsize=7,
        )
        return ax

    def add_clusters(
        self,
        df: pd.DataFrame,
        beam: int,
        title: str = "",
        column: str = "cluster_tag",
        label_clusters: bool = True,
        xlabel: str = "",
        ylabel: str = "Range gate",
    ):
        """Cluster membership (``column``: ``cluster_tag`` from the clustering models); noise (-1) is black.

        ``clabel`` of the median model is a *per-scan* DBSCAN label, not continuous in time, so it is rarely
        meaningful here.

        Parameters
        ----------
        df : pandas.DataFrame
            Echo table with ``time``, ``bmnum``, ``slist`` and the label column.
        beam : int
            Beam number to draw.
        title : str
            Panel title.
        column : str
            Label column; ``ValueError`` when absent (cluster plots need the output of the DBSCAN model).
        label_clusters : bool
            Write the cluster number on large clusters (more than 250 echoes, or any cluster when there are fewer
            than 50).
        xlabel : str
            x-axis label.
        ylabel : str
            y-axis label.

        Returns
        -------
        matplotlib.axes.Axes
            The new panel.
        """
        if column not in df:
            raise ValueError(f"column {column!r} not in the data; cluster plots need output of a clustering model")
        d = self._beam(df, beam)
        labels = np.sort(d[column].dropna().unique())
        index = {lab: i for i, lab in enumerate(labels)}
        d = d.assign(_idx=d[column].map(index))
        cmap = get_cluster_cmap(len(labels), plot_noise=bool(len(labels) and labels[0] == -1))
        norm = mcolors.BoundaryNorm(np.arange(len(labels) + 1) - 0.5, cmap.N)
        ax = self._axis(df, xlabel, ylabel, title)
        x, y, z = grid_time_gate(d, "_idx", self.nrang)
        ax.pcolormesh(x, y, np.ma.masked_invalid(z.T), cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        if label_clusters:
            t = mdates.date2num(d["time"].values)
            for lab in labels:
                if lab == -1:
                    continue
                m = (d[column] == lab).values
                if m.sum() > 250 or (len(labels) < 50 and m.sum() > 0):
                    ax.text(
                        np.median(t[m]), np.median(d["slist"].values[m]), str(int(lab)), fontsize=8, fontweight="bold"
                    )
        return ax

    # ----------------------------------------------------------------- output
    def save(self, path: str | os.PathLike, dpi: int | None = None) -> str:
        """Save the figure (format from the file extension) and return the path.

        Parameters
        ----------
        path : str or path
            Output file (``.png``, ``.pdf``, ...); the directory is created.
        dpi : int, optional
            Resolution of the saved file; the figure's own resolution when ``None``.

        Returns
        -------
        str
            The path written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"saved RTI figure {path}")
        return str(path)

    def close(self) -> None:
        """Release the figure."""
        self.fig.clf()
        plt.close(self.fig)


def plot_rti(
    df: pd.DataFrame,
    beam: int,
    params: Sequence[str] = ("v", "w_l", "p_l", "gflg"),
    path: str | os.PathLike | None = None,
    rad: str = "",
    nrang: int | None = None,
    xlim=None,
    kind: str = "",
    dpi: int = 120,
) -> RangeTimeIntervalPlot:
    """One figure with a panel per entry of ``params`` for ``beam``.

    Entries are numeric columns (``v``, ``w_l``, ``p_l``, ``v_mad``), flag columns (``gflg``, ``gflg_conv``,
    ``gflg_kde``, ``gflg_ribiero``) or ``cluster_tag``.

    Parameters
    ----------
    df : pandas.DataFrame
        Echo table from :func:`amgeo_sd.io.load_frame` (raw fitacf or fitacf++, same columns).
    beam : int
        Beam number to draw.
    params : sequence of str
        One panel per entry, top to bottom.
    path : str or path, optional
        Save the figure here (the format follows the extension); not saved when ``None``.
    rad : str
        Radar code for the title.
    nrang : int, optional
        Number of range gates on the y axis; the largest ``nrang`` of the data when ``None``.
    xlim : (datetime, datetime), optional
        Time limits; the span of the data when ``None``.
    kind : str
        Description of the data (e.g. ``"fitacf 2.5"``, ``"fitacf++"``) appended to the title.
    dpi : int
        Resolution.

    Returns
    -------
    RangeTimeIntervalPlot
        The figure object (``.fig`` is the matplotlib figure); call ``.close()`` when done.
    """
    nrang = nrang or int(df["nrang"].max())
    t0 = pd.Timestamp(df["time"].min())
    title = f"{rad.upper()} beam {beam}, {t0:%Y-%m-%d}" + (f" ({kind})" if kind else "")
    rti = RangeTimeIntervalPlot(nrang, xlim or (df["time"].min(), df["time"].max()), title, len(params), dpi=dpi)
    for i, p in enumerate(params):
        xl = "Time [UT]" if i == len(params) - 1 else ""
        label = f"({chr(97 + i)}) {p}"
        if p.startswith("gflg"):
            rti.add_gsis(df, beam, label, flag=p, xlabel=xl)
        elif p == "cluster_tag":
            rti.add_clusters(df, beam, label, xlabel=xl)
        else:
            rti.add_param(df, beam, label, param=p, xlabel=xl)
    if path:
        rti.save(path)
    return rti


def plot_rti_compare(
    frames: Mapping[str, pd.DataFrame],
    beam: int,
    param: str = "v",
    path: str | os.PathLike | None = None,
    rad: str = "",
    nrang: int | None = None,
    flag: str | None = None,
    dpi: int = 120,
) -> RangeTimeIntervalPlot:
    """Stack the same beam from several tables (e.g. raw fitacf above fitacf++), identical axes and colour scale.

    With ``flag`` (a flag column such as ``gflg``) each dataset also gets a classification panel.

    Parameters
    ----------
    frames : mapping of str to pandas.DataFrame
        Panel label (e.g. ``"raw fitacf"``) to echo table; drawn in this order.
    beam : int
        Beam number to draw.
    param : str
        Numeric column drawn for every dataset, with one shared colour scale.
    path : str or path, optional
        Save the figure here; not saved when ``None``.
    rad : str
        Radar code for the title.
    nrang : int, optional
        Number of range gates on the y axis; the largest ``nrang`` of the tables when ``None``.
    flag : str, optional
        Flag column; adds a classification panel under each dataset.
    dpi : int
        Resolution.

    Returns
    -------
    RangeTimeIntervalPlot
        The figure object; call ``.close()`` when done.
    """
    nrang = nrang or max(int(f["nrang"].max()) for f in frames.values())
    tmin = min(f["time"].min() for f in frames.values())
    tmax = max(f["time"].max() for f in frames.values())
    per = 2 if flag else 1
    n = per * len(frames)
    rti = RangeTimeIntervalPlot(
        nrang, (tmin, tmax), f"{rad.upper()} beam {beam}, {pd.Timestamp(tmin):%Y-%m-%d}", n, dpi=dpi
    )
    k = 0
    for name, df in frames.items():
        k += 1
        rti.add_param(df, beam, f"({chr(96 + k)}) {name}", param=param, xlabel="Time [UT]" if k == n else "")
        if flag:
            k += 1
            rti.add_gsis(df, beam, f"({chr(96 + k)}) {name}: {flag}", flag=flag, xlabel="Time [UT]" if k == n else "")
    if path:
        rti.save(path)
    return rti
