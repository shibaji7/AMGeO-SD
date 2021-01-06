#!/usr/bin/env python

"""plot_lib.py: module is dedicated to plot and create the movies."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
pd.set_option("mode.chained_assignment", None)
import datetime as dt

font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["mathtext.default"] = "default"

import glob
import os
import utils
from loguru import logger

class FanPlots(object):
    """Class creates Fov-Fan plots of the images."""
    
    def __init__(self, figsize=(5,5), nrows=1, ncols=1, dpi=120):
        logger.info("Fan plot initialization")
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.nrows = nrows
        self.ncols = ncols
        self.self._num_subplots_created = 0
        return
    
    def _add_axis(self, date, rad, coords="geo", map_projection=None, bounds=[]):
        self._num_subplots_created += 1
        logger.info(f"Adding axes to fan plot : {self._num_subplots_created}")
        pos = int(str(self.nrows) + str(self.ncols) + str(self._num_subplots_created))
        ax = self.fig.add_subplot(pos, projection="sdcarto",\
                map_projection = map_projection,\
                coords=coords, plot_date=date)
        ax.coastlines()
        ax.overlay_radar()
        ax.overlay_fov()
        ax.grid_on(draw_labels=[False, False, True, True])
        ax.enum(bounds=bounds, text_coord=True, dtype=self.dtype)
        self.ax = ax
        return ax
    
    def plot_data(self, df):
        """ Plot data - overlay on top of map """"
        return
    
    def save(self, filepath):
        logger.info(f"Save FoV figure to : {filepath}")
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
    
CLUSTER_CMAP = plt.cm.gist_rainbow

def get_cluster_cmap(n_clusters, plot_noise=False):
    cmap = CLUSTER_CMAP
    cmaplist = [cmap(i) for i in range(cmap.N)]
    while len(cmaplist) < n_clusters:
        cmaplist.extend([cmap(i) for i in range(cmap.N)])
    cmaplist = np.array(cmaplist)
    r = np.array(range(len(cmaplist)))
    random.seed(10)
    random.shuffle(r)
    cmaplist = cmaplist[r]
    if plot_noise:
        cmaplist[0] = (0, 0, 0, 1.0)    # black for noise
    rand_cmap = cmap.from_list("Cluster cmap", cmaplist, len(cmaplist))
    return rand_cmap
    
class RangeTimeIntervalPlot(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """
    
    def __init__(self, nrang, unique_times, fig_title="", num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=100) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15)
        mpl.rcParams.update({"font.size": 10})
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, p_step=25, xlabel="Time UT", zparam="v",
                    label="Velocity [m/s]"):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = plt.cm.jet
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        self._add_colorbar(self.fig, ax, bounds, cmap, label=label)
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        return
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax
    
    def _add_colorbar(self, fig, ax, bounds, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return
    
    def addCluster(self, df, beam, title, xlabel="", label_clusters=True):
        # add new axis
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        unique_labs = np.sort(np.unique(df.labels))
        for i, j in zip(range(len(unique_labs)), unique_labs):
            if j > 0:
                df["labels"]=np.where(df["labels"]==j, i, df["labels"])
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam="labels")
        flags = df.labels
        if -1 in flags:
            cmap = get_cluster_cmap(len(np.unique(flags)), plot_noise=True)       # black for noise
        else:
            cmap = get_cluster_cmap(len(np.unique(flags)), plot_noise=False)

        # Lower bound for cmap is inclusive, upper bound is non-inclusive
        bounds = list(range( len(np.unique(flags))+2 ))    # need (max_cluster+1) to be the upper bound
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_title(title,  loc="left", fontdict={"fontweight": "bold"})
        if label_clusters:
            num_flags = len(np.unique(flags))
            for f in np.unique(flags):
                flag_mask = Z.T==f
                g = Y[flag_mask].astype(int)
                t_c = X[flag_mask]
                # Only label clusters large enough to be distinguishable on RTI map,
                # OR label all clusters if there are few
                if (len(t_c) > 250 or
                   (num_flags < 50 and len(t_c) > 0)) \
                   and f != -1:
                    m = int(len(t_c) / 2)  # Time is sorted, so this is roughly the index of the median time
                    ax.text(t_c[m], g[m], str(int(f)), fontdict={"size": 8, "fontweight": "bold"})  # Label cluster #
        return
    
    def addGSIS(self, df, beam, title, xlabel="", zparam="gflg_0", clusters=None, label_clusters=False):
        # add new axis
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam,)
        flags = np.array(df[zparam]).astype(int)
        if -1 in flags and 2 in flags:                     # contains noise flag
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                (1.0, 0.0, 0.0, 1.0),     # blue
                (0.0, 0.0, 1.0, 1.0),     # red
                (0.0, 1.0, 0.0, 1.0)])    # green
            bounds = [-1, 0, 1, 2, 3]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US"), mpatches.Patch(color="green", label="SAIS")]
        elif -1 in flags and 2 not in flags:
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                                              (1.0, 0.0, 0.0, 1.0),     # blue
                                              (0.0, 0.0, 1.0, 1.0)])    # red
            bounds = [-1, 0, 1, 2]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US")]
        else:
            cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0),  # blue
                                              (0.0, 0.0, 1.0, 1.0)])  # red
            bounds = [0, 1, 2]          # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS")]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_title(title,  loc="left", fontdict={"fontweight": "bold"})
        ax.legend(handles=handles, loc=4)
        if label_clusters:
            flags = df.labels
            num_flags = len(np.unique(flags))
            X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam="labels")
            for f in np.unique(flags):
                flag_mask = Z.T==f
                g = Y[flag_mask].astype(int)
                t_c = X[flag_mask]
                # Only label clusters large enough to be distinguishable on RTI map,
                # OR label all clusters if there are few
                if (len(t_c) > 250 or
                   (num_flags < 100 and len(t_c) > 0)) \
                   and f != -1:
                    tct = ""
                    m = int(len(t_c) / 2)  # Time is sorted, so this is roughly the index of the median time
                    if clusters[beam][int(f)]["type"] == "IS": tct = "%.1f IS"%((1-clusters[beam][int(f)]["auc"])*100)
                    if clusters[beam][int(f)]["type"] == "GS": tct = "%.1f GS"%(clusters[beam][int(f)]["auc"]*100)
                    ax.text(t_c[m], g[m], tct, fontdict={"size": 8, "fontweight": "bold", "color":"gold"})  # Label cluster #
        return
    
    def save(self, filepath):
        logger.info(f"Save RTI figure to : {filepath}")
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return