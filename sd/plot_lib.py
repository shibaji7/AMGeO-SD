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
import spacepy.plot as splot
import seaborn as sns
import matplotlib.colors as mcolors

import numpy as np

splot.style("spacepy_altgrid")
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["mathtext.default"] = "default"


import utils

class InvestigativePlots(object):
    """Class that plots all the investigative figures"""

    def __init__(self, figure_name, e, rad, data, **keywords):
        """
        initialze all the parameters
        figure_name: Name of the investigation "1plot", "4plot", "5plot"
        e: datetime of the middle scan [0]
        data: SuperDARN radar scans. Data are in the following order - [-1,0,+1,filt]
        **keywords: other keywords for drawing
        """
        self.figure_name = figure_name
        self.data = data
        self.rad = rad
        self.e = e
        for p in keywords.keys():
            setattr(self, p, keywords[p])
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

    def set_point_legend(self, ax, ylim=[0,70], xlim=[-2,18],
            keys={"oc": {"size":5, "color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"size":5, "color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"size":5, "color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}},
            leg_keys=("g-s","i-s","u-n"), sp=1, loc="upper left", ncol=1, fontsize=6, frameon=True, is_unknown=False):
        oc = ax.scatter([], [], s=keys["oc"]["size"], color=keys["oc"]["color"], marker=keys["oc"]["marker"],
                facecolors=keys["oc"]["facecolors"], edgecolors=keys["oc"]["edgecolors"], alpha=keys["oc"]["alpha"], lw=keys["oc"]["lw"])
        cc = ax.scatter([], [], s=keys["cc"]["size"], color=keys["cc"]["color"], marker=keys["cc"]["marker"],
                facecolors=keys["cc"]["facecolors"], edgecolors=keys["cc"]["edgecolors"], alpha=keys["cc"]["alpha"], lw=keys["cc"]["lw"])
        if is_unknown:
            un = ax.scatter([], [], s=keys["un"]["size"], color=keys["un"]["color"], marker=keys["un"]["marker"],
                    facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"], alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
            leg = ax.legend((oc,cc,un), leg_keys, scatterpoints=sp, loc=loc, ncol=ncol, fontsize=fontsize, frameon=frameon)
        else: leg = ax.legend((oc,cc), leg_keys, scatterpoints=sp, loc=loc, ncol=ncol, fontsize=fontsize, frameon=frameon)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
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

    def draw_generic_points(self, ax, scan, cast="gflg",
            keys={"oc": {"size":5, "color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"size":5, "color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"size":5, "color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}):
        Xgs,Ygs,ugs = utils.get_gridded_parameters(utils.parse_parameter(scan, p=cast), zparam=cast)
        xoc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==1.),Xgs)
        yoc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==1.),Ygs)
        ax.scatter(xoc, yoc, s=keys["oc"]["size"], color=keys["oc"]["color"], marker=keys["oc"]["marker"],
                facecolors=keys["oc"]["facecolors"], edgecolors=keys["oc"]["edgecolors"],
                alpha=keys["un"]["alpha"], lw=keys["oc"]["lw"])
        xcc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==0.),Xgs)
        ycc = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==0.),Ygs)
        ax.scatter(xcc, ycc, s=keys["cc"]["size"], color=keys["cc"]["color"], marker=keys["cc"]["marker"],
                facecolors=keys["cc"]["facecolors"], edgecolors=keys["cc"]["edgecolors"],
                alpha=keys["cc"]["alpha"], lw=keys["cc"]["lw"])
        xun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==-1),Xgs)
        yun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==-1),Ygs)
        ax.scatter(xun, yun, s=keys["un"]["size"], color=keys["un"]["color"], marker=keys["un"]["marker"],
                facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"],
                alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
        return

    def draw_specific_points(self, ax, scan, u, cast="gflg",
            keys={"oc": {"color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":0., "lw":.1},
                "cc": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}, c=15):
        u = c * np.abs(u) / np.max(np.abs(u))
        Xgs,Ygs,ugs = utils.get_gridded_parameters(utils.parse_parameter(scan, p=cast), zparam=cast)
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
        xun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==-1),Xgs)
        yun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==-1),Ygs)
        uun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==-1),u.T)
        ax.scatter(xun, yun, s=uun, color=keys["un"]["color"], marker=keys["un"]["marker"],
                facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"],
                alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
        return

    def draw_quiver(self, ax, scan, label, cast="gflg", zparam="velocity",
            keys={"oc": {"size":5, "color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":0., "lw":.1},
                "cc": {"size":5, "color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"size":5, "color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}},
            qv={"xx":0.85, "yy":0.98, "v":1000, "labelpos":"S", "color":"r", "labelcolor":"r"}):
        ax = self.set_axes(ax)
        ax.text(label["xx"], label["yy"], label["text"],
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(scan, p=zparam),zparam=zparam)
        v = np.zeros_like(u)
        q = ax.quiver(X, Y, u.T, v.T, scale=1e4)
        ax.quiverkey(q, qv["xx"], qv["yy"], qv["v"], r"$%s$ m/s"%str(qv["v"]), labelpos=qv["labelpos"],
                fontproperties={"size": font["size"], "weight": "bold"},
                color=qv["color"], labelcolor=qv["labelcolor"])
        self.draw_generic_points(ax, scan, cast=cast, keys=keys)
        return
    
    def draw_scatter(self, ax, scan, label, cast="gflg", zparam="v_mad",
            keys={"oc": {"color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}, c=15):
        ax = self.set_axes(ax)
        ax.text(label["xx"], label["yy"], label["text"],
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes,fontdict=fonttext)
        X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(scan, p=zparam), zparam=zparam)
        self.draw_specific_points(ax, scan, u, cast=cast, keys=keys, c=c)
        return np.max(np.abs(u))

    def draw(self, figure_name=None):
        """
        Draw the figures for the scan
        """
        if figure_name is not None: self.figure_name = figure_name
        if self.figure_name == "1plot":
            C = 20
            dur = int((self.data[1].stime-self.data[0].stime).seconds/60.)
            fig, ax = plt.subplots(figsize=(6, 4), sharex="all", sharey="all", nrows=1, ncols=1, dpi=100)
            mx = self.draw_scatter(ax, self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$power^{LoS}[dB]$"}, zparam="power", c=C)
            self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax), c=C)
            fig.text(0.5, 0.01, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
            ax.text(1.02,0.5,"Date=%s, Rad=%s, Scan Dur=%d"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(),dur),
                    fontdict={"color":"blue","size":7}, ha="center", va="center", transform=ax.transAxes,
                    rotation=90)
            X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(self.data[1], p="velocity"),zparam="velocity")
            v = np.zeros_like(u)
            q = ax.quiver(X, Y, u.T, v.T)
            ax.quiverkey(q, 0.75, 0.98, 1000, r"$1000$ m/s", labelpos="S",
                    fontproperties={"size": font["size"], "weight": "bold"}, color="r", labelcolor="r")
            #fig.savefig("data/sim/{rad}/{date}.1plot.png".format(rad=self.rad, date=self.e.strftime("%y%m%d%H%M")),bbox_inches="tight")
            fig.savefig("1plot.png".format(rad=self.rad, date=self.e.strftime("%y%m%d%H%M")),bbox_inches="tight")
        elif self.figure_name == "4plot":
            dur = int((self.data[1].stime-self.data[0].stime).seconds/60.)
            fig, axes = plt.subplots(figsize=(6, 6), sharex="all", sharey="all", nrows=2, ncols=2, dpi=100)
            fig.subplots_adjust(wspace=0.1,hspace=0.1)
            fig.text(0.5, 0.06, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
            axes[0,1].text(1.05,0.5,"Date=%s, Rad=%s, \nScan Dur=%s"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(), dur),
                    fontdict={"color":"blue","size":7}, ha="center", va="center",rotation="vertical", transform=axes[0,1].transAxes)
            
            C = 30
            self.draw_quiver(axes[0,0], self.data[1], label={"xx":.9, "yy":1.05, "text":r"$v^{LoS}[m/s]$"}, zparam="velocity")
            self.set_point_legend(axes[0,0])
            mx = self.draw_scatter(axes[0,1], self.data[1], 
                    label={"xx":0.75, "yy":1.05, "text":r"$v_{error}^{LoS}[m/s]$"}, zparam="v_e", c=C)
            self.set_size_legend(axes[0,1], mx, leg_keys=("90", "45", "10"), leg=self.set_point_legend(axes[0,1]), c=C)
            mx = self.draw_scatter(axes[1,0], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$power^{LoS}[dB]$"}, zparam="power", c=C)
            self.set_size_legend(axes[1,0], mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(axes[1,0]), c=C)
            mx = self.draw_scatter(axes[1,1], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$width^{LoS}[m/s]$"}, zparam="width", c=C)
            self.set_size_legend(axes[1,1], mx, leg_keys=("200", "100", "10"), leg=self.set_point_legend(axes[1,1]), c=C)
            fig.savefig("4plot.png",bbox_inches="tight")
        elif self.figure_name == "5plot":

        return
