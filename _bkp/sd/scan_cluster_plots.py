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
import matplotlib as mpl
from matplotlib.dates import DateFormatter, num2date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import random

import os
from loguru import logger
import numpy as np
import pandas as pd
pd.set_option("mode.chained_assignment", None)
import datetime as dt

import utils
from get_fit_data import FetchData

#splot.style("spacepy_altgrid")
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["mathtext.default"] = "default"

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

class MasterPlotter(object):
    """Class for all the plotting"""

    def __init__(self, ylim=[0,70], xlim=[-2,18]):
        self.ylim = ylim
        self.xlim = xlim
        return

    def set_axes(self, ax):
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        
        ax.tick_params(which="both", width=2)
        ax.tick_params(which="major", length=7)
        ax.tick_params(which="minor", length=4)
        return ax

    def set_point_legend(self, ax,
            keys={"oc": {"size":5, "color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"size":5, "color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"size":5, "color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}},
            leg_keys=("g-s","i-s","u-s"), sp=1, loc="upper left", ncol=1, fontsize=6, frameon=True, is_unknown=False):
        oc = ax.scatter([], [], s=keys["oc"]["size"], color=keys["oc"]["color"], marker=keys["oc"]["marker"],
                facecolors=keys["oc"]["facecolors"], edgecolors=keys["oc"]["edgecolors"], alpha=keys["oc"]["alpha"], lw=keys["oc"]["lw"])
        cc = ax.scatter([], [], s=keys["cc"]["size"], color=keys["cc"]["color"], marker=keys["cc"]["marker"],
                facecolors=keys["cc"]["facecolors"], edgecolors=keys["cc"]["edgecolors"], alpha=keys["cc"]["alpha"], lw=keys["cc"]["lw"])
        if is_unknown:
            un = ax.scatter([], [], s=keys["un"]["size"], color=keys["un"]["color"], marker=keys["un"]["marker"],
                    facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"], alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
            leg = ax.legend((oc,cc,un), leg_keys, scatterpoints=sp, loc=loc, ncol=ncol, fontsize=fontsize, frameon=frameon)
        else: leg = ax.legend((oc,cc), leg_keys, scatterpoints=sp, loc=loc, ncol=ncol, fontsize=fontsize, frameon=frameon)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        return leg

    def set_size_legend(self, ax, maxx, keys={"l0": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":.5, "lw":.1},
        "l1": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
        "l2": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1}},
        leg_keys=("150", "75", "10"), sp=1, loc="upper right", ncol=1, fontsize=6, frameon=True, leg=None, c=15):
        l0 = ax.scatter([], [], s=c*float(leg_keys[0])/maxx, color=keys["l0"]["color"], marker=keys["l0"]["marker"],
                facecolors=keys["l0"]["facecolors"], edgecolors=keys["l0"]["edgecolors"], alpha=keys["l0"]["alpha"], lw=keys["l0"]["lw"])
        l1 = ax.scatter([], [], s=c*float(leg_keys[1])/maxx, color=keys["l1"]["color"], marker=keys["l1"]["marker"],
                facecolors=keys["l1"]["facecolors"], edgecolors=keys["l1"]["edgecolors"], alpha=keys["l1"]["alpha"], lw=keys["l1"]["lw"])
        l2 = ax.scatter([], [], s=c*float(leg_keys[2])/maxx, color=keys["l2"]["color"], marker=keys["l2"]["marker"],
                facecolors=keys["l2"]["facecolors"], edgecolors=keys["l2"]["edgecolors"], alpha=keys["l2"]["alpha"], lw=keys["l2"]["lw"])
        ax.legend((l0,l1,l2), leg_keys, scatterpoints=sp, loc=loc, ncol=ncol, fontsize=fontsize, frameon=frameon)
        if leg is not None: ax.add_artist(leg)
        return

    def draw_specific_points(self, ax, o, u, cast="gflg",
            keys={"oc": {"color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":0., "lw":.1},
                "cc": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}, c=15):
        zp = cast
        if "gsflg_" in cast: zp = "gsflg"
        u = c * np.abs(u) / np.max(np.abs(u))
        Xgs,Ygs,ugs = utils.get_gridded_parameters(o, "bmnum", "slist", zparam=zp)
        xoc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==1.),Xgs)
        yoc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==1.),Ygs)
        uoc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==1.),u.T)
        ax.scatter(xoc, yoc, s=uoc, color=keys["oc"]["color"], marker=keys["oc"]["marker"],
                facecolors=keys["oc"]["facecolors"], edgecolors=keys["oc"]["edgecolors"],
                alpha=keys["oc"]["alpha"], lw=keys["oc"]["lw"])
        xcc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==0.),Xgs)
        ycc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==0.),Ygs)
        ucc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==0.),u.T)
        ax.scatter(xcc, ycc, s=ucc, color=keys["cc"]["color"], marker=keys["cc"]["marker"],
                facecolors=keys["cc"]["facecolors"], edgecolors=keys["cc"]["edgecolors"],
                alpha=keys["cc"]["alpha"], lw=keys["cc"]["lw"])
        xun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==2),Xgs)
        yun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==2),Ygs)
        uun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==2),u.T)
        ax.scatter(xun, yun, s=uun, color=keys["un"]["color"], marker=keys["un"]["marker"],
                facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"],
                alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
        return

    def plot_quiver(self, ax, o, p="v", qv={"xx":0.75, "yy": 0.98, "v":1000, "labelpos":"S", "color":"r", "labelcolor":"r"}):
        X,Y,u = utils.get_gridded_parameters(o, "bmnum", "slist",zparam=p)
        v = np.zeros_like(u)
        q = ax.quiver(X, Y, u.T, v.T, scale=1e4, color=self.base_color)
        ax.quiverkey(q, qv["xx"], qv["yy"], qv["v"], r"$%s$ m/s"%str(qv["v"]), labelpos=qv["labelpos"],
                fontproperties={"size": font["size"], "weight": "bold"},
                color=qv["color"], labelcolor=qv["labelcolor"])
        return

    def draw_scatter(self, ax, o, label, cast="gflg", zparam="v_mad",
            keys={"oc": {"color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}, c=15):
        zp = cast
        if "gsflg_" in cast: zp = "gsflg"
        ax = self.set_axes(ax)
        ax.text(label["xx"], label["yy"], label["text"],
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes,fontdict=fonttext)
        X,Y,u = utils.get_gridded_parameters(o, "bmnum", "slist", zparam=zparam)
        keys["oc"]["color"], keys["oc"]["edgecolors"] = self.base_color, self.base_color
        keys["cc"]["color"], keys["cc"]["facecolors"], keys["cc"]["edgecolors"] = self.base_color, self.base_color, self.base_color
        keys["un"]["color"], keys["un"]["edgecolors"] = self.base_color, self.base_color
        self.draw_specific_points(ax, o, u, cast=cast, keys=keys, c=c)
        return np.max(np.abs(u))

class ClusterScanMap(MasterPlotter):
    """Class that plots all the cluster investigative figures"""
    
    def __init__(self, scan, e, rad, sim_id, keywords={}, folder="data/outputs/{sim_id}/{rad}/",
            xlim=[-2,18], ylim=[0,70]):
        super().__init__(ylim, xlim)
        self.scan = scan
        self.rad = rad
        self.e = e
        self.sim_id = sim_id
        self.folder = folder.format(rad=self.rad, sim_id=self.sim_id)
        if not os.path.exists(self.folder): 
            logger.info("Create folder - mkdir {folder}".format(folder=self.folder))
            os.system("mkdir -p {folder}".format(folder=self.folder))
        for p in keywords.keys():
            setattr(self, p, keywords[p])
            
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang", "time", "intt.sc", "intt.us", "mppul"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "v_e", "v_mad", "gflg_kde", "gflg_conv", "clabel"]
        io = FetchData(None, None)
        self.o = io.scans_to_pandas([self.scan], s_params, v_params)
        self.clusters = self.o.clabel.unique()
        self.cmap = get_cluster_cmap(len(self.clusters), plot_noise=True) if -1 in self.clusters else\
        get_cluster_cmap(len(self.clusters), plot_noise=False)
        return
    
    def _plot_med_filt(self):
        """
        Plot median filterd data.
        """
        C = 10
        fonttext["size"] = 6
        fig = plt.figure(figsize=(4,4),dpi=150)
        ax = fig.add_subplot(111)
        
        for i, c in enumerate(self.clusters):
            self.base_color = self.cmap(i)
            o = self.o[self.o.clabel==c]
            mx = self.draw_scatter(ax, o, label={"xx":0.8, "yy":1.02, "text":""}, zparam="v", cast=self.gs, c=C)
            self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax, is_unknown=True), c=C)
            font["size"] = 5
            self.plot_quiver(ax, o, qv={"xx":0.8, "yy": 1.02, "v":1000,
                "labelpos":"N", "color":"r", "labelcolor":"r"})
        ax.text(0.95,1.05,r"$P^{LoS}_{dB}$",horizontalalignment="center", verticalalignment="center", fontdict=fonttext, 
                transform=ax.transAxes)
        font["size"] = 10
        fonttext["color"] = "green"
        ax.text(0.35, 1.05, "Date=%s, Radar=%s, GSf=%d, \nFreq=%.1f MHz, N_sky=%.1f"%(self.e.strftime("%Y-%m-%d"),self.rad.upper(),
                                                                                      self.gflg_type, self.o.tfreq.mean(), 
                                                                                      self.o["noise.sky"].mean()), 
                horizontalalignment="center", verticalalignment="center", fontdict=fonttext, transform=ax.transAxes)
        fonttext["color"] = "blue"
        ax.set_xlabel("Beams",fontdict=font)
        ax.set_ylabel("Gates",fontdict=font)
        fonttext["size"] = 10
        ax.text(0.5, 0.9, "%s UT"%self.e.strftime("%H:%M"), horizontalalignment="center", verticalalignment="center", 
                fontdict=fonttext, transform=ax.transAxes)
        fonttext["size"] = 6
        fonttext["color"] = "green"
        ax.text(1.05,.2,r"Med-Filt($\tau$=%.2f)"%self.thresh,horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        if self.gs=="gsflg_kde":
            ax.text(1.05, 0.75, r"KDE$(p_{th}=%.1f,q_{th}=[%.2f,%.2f])$"%(self.pth, self.pbnd[0], self.pbnd[1]),
                    horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, fontdict=fonttext, rotation=90)
        else:
            ax.text(1.05, 0.75, r"CONV$(q_{th}=[%.2f,%.2f])$"%(self.pbnd[0], self.pbnd[1]),
                    horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, fontdict=fonttext, rotation=90)
        fonttext["color"] = "blue"
        fig.savefig("{folder}med_filt_{dn}.png".format(folder=self.folder, dn=self.e.strftime("%Y-%m-%d-%H-%M")), bbox_inches="tight")
        plt.close()
        return