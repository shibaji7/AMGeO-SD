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

import numpy as np
import pandas as pd
pd.set_option("mode.chained_assignment", None)
import datetime as dt

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

import glob
import os
import utils
from loguru import logger

gs_map = {-1: "fitacf.def", 0: "Sundeen", 1: "Blanchard", 2: "Blanchard.2009", 3: "Chakraborty.2020"}

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

    def draw_generic_points(self, ax, scan, cast="gflg",
            keys={"oc": {"size":5, "color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1},
                "cc": {"size":5, "color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"size":5, "color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}):
        zp = cast
        if "gsflg_" in cast: zp = "gsflg"
        Xgs,Ygs,ugs = utils.get_gridded_parameters(utils.parse_parameter(scan, p=cast), zparam=zp)
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
        xun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==2),Xgs)
        yun = np.ma.masked_where(np.logical_not(ugs.T.astype(float)==2),Ygs)
        ax.scatter(xun, yun, s=keys["un"]["size"], color=keys["un"]["color"], marker=keys["un"]["marker"],
                facecolors=keys["un"]["facecolors"], edgecolors=keys["un"]["edgecolors"],
                alpha=keys["un"]["alpha"], lw=keys["un"]["lw"])
        return

    def draw_specific_points(self, ax, scan, u, cast="gflg",
            keys={"oc": {"color":"k", "marker":"o", "facecolors":"None", "edgecolors":"k", "alpha":0., "lw":.1},
                "cc": {"color":"k", "marker":"o", "facecolors":"k", "edgecolors":"k", "alpha":0.5, "lw":.1},
                "un": {"color":"k", "marker":r"$\oslash$", "facecolors":"None", "edgecolors":"k", "alpha":1., "lw":.1}}, c=15):
        zp = cast
        if "gsflg_" in cast: zp = "gsflg"
        u = c * np.abs(u) / np.max(np.abs(u))
        Xgs,Ygs,ugs = utils.get_gridded_parameters(utils.parse_parameter(scan, p=cast), zparam=zp)
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

    def plot_quiver(self, ax, scan, p="v", qv={"xx":0.75, "yy": 0.98, "v":1000, "labelpos":"S", "color":"r", "labelcolor":"r"}):
        X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(scan, p=p),zparam=p)
        v = np.zeros_like(u)
        q = ax.quiver(X, Y, u.T, v.T, scale=1e4)
        ax.quiverkey(q, qv["xx"], qv["yy"], qv["v"], r"$%s$ m/s"%str(qv["v"]), labelpos=qv["labelpos"],
                fontproperties={"size": font["size"], "weight": "bold"},
                color=qv["color"], labelcolor=qv["labelcolor"])
        return

    def draw_quiver(self, ax, scan, label, cast="gflg", zparam="v",
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
        zp = cast
        if "gsflg_" in cast: zp = "gsflg"
        ax = self.set_axes(ax)
        ax.text(label["xx"], label["yy"], label["text"],
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes,fontdict=fonttext)
        X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(scan, p=zparam), zparam=zparam)
        self.draw_specific_points(ax, scan, u, cast=cast, keys=keys, c=c)
        return np.max(np.abs(u))
    


class InvestigativePlots(MasterPlotter):
    """Class that plots all the investigative figures"""

    def __init__(self, figure_name, e, rad, sim_id, data, keywords={}, folder="data/outputs/{sim_id}/{rad}/",
            xlim=[-2,18], ylim=[0,70]):
        """
        initialze all the parameters
        figure_name: Name of the investigation "1plot", "4plot", "5plot"
        e: datetime of the middle scan [0]
        folder: name of the folder to store figures
        data: SuperDARN radar scans. Data are in the following temporal order - [-1,0,+1,filt]
        **keywords: other keywords for drawing
        """
        super().__init__(ylim, xlim)
        self.figure_name = figure_name
        self.data = data
        self.rad = rad
        self.e = e
        self.sim_id = sim_id
        self.folder = folder.format(rad=self.rad, sim_id=self.sim_id)
        if not os.path.exists(self.folder): 
            logger.info("Create folder - mkdir {folder}".format(folder=self.folder))
            os.system("mkdir -p {folder}".format(folder=self.folder))
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        return

    def draw(self, figure_name=None):
        """
        Draw the figures for the scan
        """
        cast = "gflg"
        if self.gflg_type > 0: cast = "gsflg_" + str(self.gflg_type)
        if figure_name is not None: self.figure_name = figure_name
        if self.figure_name == "1plot":
            C = 20
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            fig, ax = plt.subplots(figsize=(6, 4), sharex="all", sharey="all", nrows=1, ncols=1, dpi=150)
            mx = self.draw_scatter(ax, self.data[1], label={"xx":0.75, "yy":1.03, "text":r"$power^{LoS}[dB]$"}, 
                                   zparam="p_l", c=C, cast=cast)
            self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax), c=C)
            fig.text(0.5, 0.01, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
            fig.suptitle("Date=%s, Rad=%s, Scan Dur=%d min"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(),dur), size=12,
                                        y=0.98, fontdict={"color":"darkblue"})
            ax.text(0.2, 1.03, "GS Flag=%s"%(gs_map[self.gflg_type]), fontdict={"color":"blue","size":7}, 
                    ha="center", va="center", transform=ax.transAxes)
            if self.cid is not None: ax.text(0.1, 0.1, "C:%02d"%self.cid, ha="center", va="center", 
                                             transform=ax.transAxes, fontdict={"color":"blue","size":7})
            X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(self.data[1], p="v"),zparam="v")
            v = np.zeros_like(u)
            q = ax.quiver(X, Y, u.T, v.T)
            ax.quiverkey(q, 0.75, 0.98, 1000, r"$1000$ m/s", labelpos="S",
                    fontproperties={"size": font["size"], "weight": "bold"}, color="r", labelcolor="r")
            fname = "{folder}{date}.1plot.png".format(folder=self.folder, rad=self.rad, date=self.e.strftime("%Y%m%d%H%M"))
            fig.savefig(fname, bbox_inches="tight")
        elif self.figure_name == "4plot":
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            fig, axes = plt.subplots(figsize=(6, 6), sharex="all", sharey="all", nrows=2, ncols=2, dpi=150)
            fig.subplots_adjust(wspace=0.1,hspace=0.1)
            fig.text(0.5, 0.06, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
             
            fig.suptitle("Date=%s, Rad=%s, Scan Dur=%d min"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(),dur), size=12,
                                        y=0.94, fontdict={"color":"darkblue"})
            C = 30
            self.draw_quiver(axes[0,0], self.data[1], label={"xx":.9, "yy":1.05, "text":r"$v^{LoS}[m/s]$"}, zparam="v", cast=cast)
            axes[0,0].text(0.2, 1.05, "GS Flag=%s"%(gs_map[self.gflg_type]), fontdict={"color":"blue","size":7},
                                        ha="center", va="center", transform=axes[0,0].transAxes)
            self.set_point_legend(axes[0,0])
            mx = self.draw_scatter(axes[0,1], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$v_{error}^{LoS}[m/s]$"}, 
                    zparam="v_e", c=C, cast=cast)
            self.set_size_legend(axes[0,1], mx, leg_keys=("90", "45", "10"), leg=self.set_point_legend(axes[0,1]), c=C)
            mx = self.draw_scatter(axes[1,0], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$power^{LoS}[dB]$"}, zparam="p_l", 
                    c=C, cast=cast)
            self.set_size_legend(axes[1,0], mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(axes[1,0]), c=C)
            mx = self.draw_scatter(axes[1,1], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$width^{LoS}[m/s]$"}, 
                    zparam="w_l", c=C, cast=cast)
            self.set_size_legend(axes[1,1], mx, leg_keys=("200", "100", "10"), leg=self.set_point_legend(axes[1,1]), c=C)
            ax = axes[1,1]
            if self.cid is not None: ax.text(0.1, 0.1, "C:%02d"%self.cid, ha="center", va="center", 
                                             transform=ax.transAxes, fontdict={"color":"blue","size":7})
            fname = "{folder}{date}.4plot.png".format(folder=self.folder, rad=self.rad, date=self.e.strftime("%Y%m%d%H%M"))
            fig.savefig(fname, bbox_inches="tight")
        elif self.figure_name == "5plot":
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            scan_times = [self.e - dt.timedelta(minutes=dur), self.e, self.e + dt.timedelta(minutes=dur)]
            fig = plt.figure(figsize=(8,5),dpi=180)
            fig.subplots_adjust(wspace=0.1,hspace=0.1)
            fig.text(0.5, 0.06, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
            fig.suptitle("Date=%s, Radar=%s"%(self.e.strftime("%Y-%m-%d"),self.rad.upper()),size=12,
                    y=0.94, fontdict={"color":"darkblue"})

            font["size"] = 7
            fonttext["size"] = 7
            
            ax = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
            self.draw_quiver(ax, self.data[0], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[0].strftime("%H:%M")}, cast=cast)
            self.set_point_legend(ax)
            ax.set_xticklabels([])
            ax = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
            self.draw_quiver(ax, self.data[1], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[1].strftime("%H:%M")}, cast=cast)
            self.set_point_legend(ax)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
            self.draw_quiver(ax, self.data[2], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[2].strftime("%H:%M")}, cast=cast)
            self.set_point_legend(ax)
            ax.text(1.05, .5, "GS Flag=%s"%(gs_map[self.gflg_type]), fontdict={"color":"blue","size":7},
                                        ha="center", va="center", transform=ax.transAxes, rotation=90)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax = plt.subplot2grid(shape=(2,6), loc=(1,2), colspan=2)
            self.draw_quiver(ax, self.data[3], label={"xx":0.75, "yy":0.1, "text":r"Med-Filt($\tau$=%.2f)"%self.thresh}, cast=self.gs)
            self.set_point_legend(ax, is_unknown=True)
            ax = plt.subplot2grid(shape=(2,6), loc=(1,4), colspan=2)
            mx = self.draw_scatter(ax, self.data[3], 
                    label={"xx":0.75, "yy":0.1, "text":r"MAD[Med-Filt($\tau$=%.2f)]"%self.thresh}, c=20, cast=self.gs)
            if self.gs=="gflg_kde":
                fonttext["color"] = "darkgreen"
                ax.text(1.05, 0.6, r"KDE$(p_{th}=%.1f,[%.2f,%.2f])$"%(self.pth, self.pbnd[0], self.pbnd[1]), 
                        horizontalalignment="center", verticalalignment="center",
                        transform=ax.transAxes, fontdict=fonttext, rotation=90)
                fonttext["color"] = "blue"
            elif self.gs=="gflg_conv":
                fonttext["color"] = "darkgreen"
                ax.text(1.05, 0.6, r"CONV$([%.2f,%.2f])$"%(self.pbnd[0], self.pbnd[1]),
                        horizontalalignment="center", verticalalignment="center",
                        transform=ax.transAxes, fontdict=fonttext, rotation=90)
                fonttext["color"] = "blue"
            self.set_size_legend(ax, mx, leg_keys=("150", "75", "25"), leg=self.set_point_legend(ax, is_unknown=True), c=20)
            ax.set_yticklabels([])
            if self.cid is not None: ax.text(0.1, 0.1, "C:%02d"%self.cid, ha="center", va="center", 
                                             transform=ax.transAxes, fontdict={"color":"blue","size":7})
            fname = "{folder}{date}.5plot.png".format(folder=self.folder, rad=self.rad, date=self.e.strftime("%Y%m%d%H%M"))
            fig.savefig(fname, bbox_inches="tight")
        return fname


class MovieMaker(MasterPlotter):
    """Class creates movie out of the images."""

    def __init__(self, rad, dates, scans, figure_name, sim_id, keywords={}, folder="data/outputs/{sim_id}/{rad}/",
            xlim=[-2,18], ylim=[0,70]):
        """
        initialze all the parameters
        rad: Radar code
        figure_name: Name of the investigation "raw", "med_filt"
        scans: Scan data list
        dates: datetime of the scans
        folder: Folder name of the dataset
        data: SuperDARN radar scans. Data are in the following order - [-1,0,+1,filt]
        **keywords: other keywords for models
        """
        super().__init__(ylim, xlim)
        self.rad = rad
        self.dates = dates
        self.scans = scans
        self.figure_name = figure_name
        self.sim_id = sim_id
        self.folder = folder.format(rad=self.rad, sim_id=self.sim_id)
        if not os.path.exists(self.folder):
            logger.info("Create folder - mkdir {folder}".format(folder=self.folder))
            os.system("mkdir -p {folder}".format(folder=self.folder))
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        return

    def _plot_raw(self):
        """
        Plot the scan.
        """
        fonttext["size"] = 10
        C = 10
        fig = plt.figure(figsize=(3,3),dpi=150)
        ax = fig.add_subplot(111)
        mx = self.draw_scatter(ax, self.scan, label={"xx":0.5, "yy":0.9, "text":"%s UT"%self.e.strftime("%H:%M")}, zparam="p_l", c=C)
        fonttext["size"] = 6
        self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax), c=C)
        font["size"] = 5
        self.plot_quiver(ax, self.scan, qv={"xx":0.8, "yy": 1.02, "v":1000,
            "labelpos":"N", "color":"r", "labelcolor":"r"})
        ax.text(0.95,1.05,r"$P^{LoS}_{dB}$",horizontalalignment="center", verticalalignment="center", fontdict=fonttext, transform=ax.transAxes)
        font["size"] = 10
        fonttext["color"] = "green"
        ax.set_xlabel("Beams",fontdict=font)
        ax.set_ylabel("Gates",fontdict=font)
        ax.text(0.35, 1.05, "Date=%s, Radar=%s, GSf=%d, \nFreq=%.1f MHz, N_sky=%.1f"%(self.e.strftime("%Y-%m-%d"),self.rad.upper(),
            self.gflg_type, self.scan.f,self.scan.nsky), horizontalalignment="center", 
            verticalalignment="center", fontdict=fonttext, transform=ax.transAxes)
        fonttext["color"] = "blue"
        fig.savefig("{folder}raw_{id}.png".format(folder=self.folder, id="%04d"%self._ix), bbox_inches="tight")
        plt.close()
        return

    def _plot_med_filt(self):
        """
        Plot median filterd data.
        """
        C = 10
        fonttext["size"] = 6
        fig = plt.figure(figsize=(3,3),dpi=150)
        ax = fig.add_subplot(111)
        mx = self.draw_scatter(ax, self.scan, label={"xx":0.8, "yy":1.02, "text":""}, zparam=self.zparam, cast=self.gs, c=C)
        self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax, is_unknown=True), c=C)
        font["size"] = 5
        self.plot_quiver(ax, self.scan, qv={"xx":0.8, "yy": 1.02, "v":1000,
            "labelpos":"N", "color":"r", "labelcolor":"r"})
        ax.text(0.95,1.05,r"$P^{LoS}_{dB}$",horizontalalignment="center", verticalalignment="center", fontdict=fonttext, transform=ax.transAxes)
        font["size"] = 10
        fonttext["color"] = "green"
        ax.text(0.35, 1.05, "Date=%s, Radar=%s, GSf=%d, \nFreq=%.1f MHz, N_sky=%.1f"%(self.e.strftime("%Y-%m-%d"),self.rad.upper(),
            self.gflg_type, self.scan.f,self.scan.nsky), horizontalalignment="center", 
            verticalalignment="center", fontdict=fonttext, transform=ax.transAxes)
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
        fig.savefig("{folder}med_filt_{gs}_{id}.png".format(folder=self.folder, id="%04d"%self._ix, gs=self.gs), bbox_inches="tight")
        plt.close()
        return

    def _create_movie(self):
        if self.figure_name == "raw": files = glob.glob("{folder}/{reg}".format(folder=self.folder, reg=self.figure_name+"*.png"))
        else: files = glob.glob("{folder}{reg}".format(folder=self.folder, reg=self.figure_name+"_"+self.gs+"*.png"))
        files.sort()
        for i, name in enumerate(files):
            nname = "_".join(name.split("_")[:-1] + ["{id}.png".format(id="%04d"%i)])
            os.system("mv {name} {fname}".format(name=name, folder=self.folder, fname=nname))
        if self.figure_name == "raw":
            reg = "raw_%04d.png"
            fname = "{rad}_{figure_name}.mp4".format(rad=self.rad, figure_name=self.figure_name)
        elif self.figure_name == "med_filt":
            reg = "med_filt_{gs}_%04d.png".format(gs=self.gs)
            fname = "{rad}_{figure_name}_{gs}.mp4".format(rad=self.rad, figure_name=self.figure_name, gs=self.gs)
        cmd = "ffmpeg -r 1 -i {folder}{reg} -c:v libx264 -vf 'scale=1420:-2,fps=3,format=yuv420p' {folder}/{fname}".format(reg=reg, fname=fname, folder=self.folder)
        logger.info("Running movie making script - ", cmd)
        os.system(cmd)
        return fname

    def exe(self, keywords={}):
        """
        Execute all the plots.
        **keywords: other keywords for models
        """
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        for self._ix, self.e, self.scan in zip(range(len(self.dates)), self.dates, self.scans):
            if self.figure_name == "raw": self._plot_raw()
            elif self.figure_name == "med_filt": self._plot_med_filt()
        fname = self._create_movie()
        return self.folder + fname

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def to_midlatitude_gate_summary(rad, df, gate_lims, names, smooth, fname, sb):
    """ Plot gate distribution summary """
    fig, axes = plt.subplots(figsize=(5,5), nrows=4, ncols=1, sharey="row", sharex=True, dpi=150)
    attrs = ["p_l"]
    beams = sb[1]
    scans = sb[0]
    count = sb[2]
    xpt = 100./scans
    labels = ["Power (dB)"]
    for j, attr, lab in zip(range(1), attrs, labels):
        ax = axes[j]
        ax.scatter(rand_jitter(df.slist), rand_jitter(df[attr]), color="r", s=1)
        ax.grid(color="gray", linestyle="--", linewidth=0.3)
        ax.set_ylabel(lab, fontdict=font)
        ax.set_xlim(0,110)
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax = axes[-3]
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.scatter(df.groupby(["slist"]).count().reset_index()["slist"], xpt*df.groupby(["slist"]).count().reset_index()["p_l"], color="k", s=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.set_ylabel("%Count", fontdict=font)
    ax.set_xlim(0,110)
    fonttext["color"] = "k"
    ax = axes[-2]
    ax.scatter(smooth[0], xpt*smooth[1], color="k", s=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlim(0,110)
    ax.set_ylabel("<%Count>", fontdict=font)
    ax = axes[-1]
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.set_xlim(0,110)
    ax.set_xlabel("Gate", fontdict=font)
    ax.set_ylabel(r"$d_{<\%Count>}$", fontdict=font)
    ds = pd.DataFrame()
    ds["x"], ds["y"] = smooth[0], smooth[1]
    for k in gate_lims.keys():
        l, u = gate_lims[k][0], gate_lims[k][1]
        du = ds[(ds.x>=l) & (ds.x<=u)]
        p = np.append(np.diff(du.y), [0])
        p[np.argmax(du.y)] = 0
        ax.scatter(du.x, xpt*p, color="k", s=3)
    ax.axhline(0, color="b", lw=0.4, ls="--")
    ax.scatter(smooth[0], smooth[1], color="r", s=1, alpha=.6)
    ax.scatter(ds.x, ds.y, color="b", s=0.6, alpha=.5)
    fonttext["size"] = 8
    for k, n in zip(gate_lims.keys(), names.keys()):
        if k >= 0:
            for j in range(len(axes)):
                ax = axes[j]
                ax.axvline(x=gate_lims[k][0], color="b", lw=0.6, ls="--")
                ax.axvline(x=gate_lims[k][1], color="darkgreen", lw=0.6, ls=":")
                #ax.text(np.mean(gate_lims[k])/110, 0.7, names[n],
                #    horizontalalignment="center", verticalalignment="center",
                #    transform=ax.transAxes, fontdict=fonttext)
    fig.suptitle("Rad-%s, %s [%s-%s] UT"%(rad, df.time.tolist()[0].strftime("%Y-%m-%d"), 
        df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M")) + "\n" + 
        r"$Beams=%s, N_{sounds}=%d, N_{gates}=%d$"%(beams, scans, 110), size=12)
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
    fig, axes = plt.subplots(figsize=(21,12), nrows=4, ncols=len(gate_lims.keys()), sharey=True, dpi=150)
    fonttext["size"], L = 5, len(gate_lims.keys())
    for i, k in enumerate(gate_lims.keys()):
        ax = axes[0, i] if L > 1 else axes[0]
        dx = df[(df.slist>=gate_lims[k][0]) & (df.slist<=gate_lims[k][1])]
        bins = list(range(-100,100,1))
        ax.hist(dx.v, bins=bins, histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Velocity (m/s)", horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{v},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.v), np.median(dx.v),np.nanquantile(np.abs(dx.v), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[1, i] if L > 1 else axes[1]
        dx = df[(df.slist>=gate_lims[k][0]) & (df.slist<=gate_lims[k][1])]
        bins = list(range(-1000,1000,1))
        ax.hist(dx.v, bins=bins, histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Velocity (m/s)", horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{v},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.v), np.median(dx.v),np.nanquantile(np.abs(dx.v), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[2, i] if L > 1 else axes[2]
        ax.hist(dx.w_l, bins=range(0,100,1), histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Width (m/s)", horizontalalignment="center", verticalalignment="center",
                                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{w},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.w_l), np.median(dx.w_l), np.nanquantile(np.abs(dx.w_l), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[3, i] if L > 1 else axes[3]
        ax.scatter(dx.v, np.abs(dx.w_l), s=2)
        ax.set_xlim(-100,100)
        ax.set_ylim(0,100)
        if i==0: ax.set_xlabel("Velocity (m/s)", fontdict=font)
        ax.set_ylabel("Width (m/s)", fontdict=font)
    #fig.savefig(fname.replace("png", "pdf"), bbox_inches="tight")
    return

def beam_gate_boundary_tracker(recs, curves, glim, blim, title, fname):
    """ Beam gate boundary plots showing the distribution of clusters """
    fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1, dpi=180)
    ax.set_ylabel("Gates", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xlim(blim[0]-1, blim[1] + 2)
    ax.set_ylim(glim[0], glim[1])
    for b in range(blim[0], blim[1] + 1):
        ax.axvline(b, lw=0.3, color="gray", ls="--")
    ax.axvline(b+1, lw=0.3, color="gray", ls="--")
    fonttext["size"] = 6
    for rec in recs:
        if curves is not None:
            curve = curves[rec["cluster"]]
            if len(curve) > 0: 
                if curve["curve"]=="parabola": func = lambda x, ac, bc: ac*np.sqrt(x)-bc
                elif curve["curve"]=="line": func = lambda x, ac, bc: ac + bc*x
                beams = curve["beams"].tolist() + [np.max(curve["beams"])+1]
                ax.plot(beams, func(beams, *curve["p_ubs"]), "b--", lw=0.5)
                ax.plot(beams, func(beams, *curve["p_lbs"]), "g--", lw=0.5)
                ax.plot(beams, 0.5*(func(beams, *curve["p_ubs"])+func(beams, *curve["p_lbs"])), "k-", lw=0.5)
        p = plt.Rectangle((rec["beam_low"], rec["gate_low"]), rec["beam_high"]-rec["beam_low"]+1,
                          rec["gate_high"]-rec["gate_low"]+1, fill=False, ls="--", lw=0.5, ec="r")
        ax.add_patch(p)
        ax.scatter([rec["mean_beam"]+0.5],[rec["mean_gate"]+0.5],s=3,color="r")
        ax.text(rec["mean_beam"]+1, rec["mean_gate"]+2, "C: %02d"%int(rec["cluster"]),
                    ha="center", va="center",fontdict=fonttext)
    ax.set_title(title)
    fonttext["size"] = 10
    ax.set_xticks(np.arange(blim[0], blim[1] + 1) + 0.5)
    ax.set_xticklabels(np.arange(blim[0], blim[1] + 1))
    fig.savefig(fname, bbox_inches="tight")
    return

def beam_gate_boundary_plots(boundaries, clusters, clust_idf, glim, blim, title, fname, gflg_type=-1):
    """ Beam gate boundary plots showing the distribution of clusters """
    GS_CASES = ["Sudden [2004]", "Blanchard [2006]", "Blanchard [2009]"]
    if gflg_type>=0: case = GS_CASES[gflg_type]
    fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1, dpi=180)
    ax.set_ylabel("Gates", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xlim(blim[0]-1, blim[1] + 2)
    ax.set_ylim(glim[0], glim[1])
    for b in range(blim[0], blim[1] + 1):
        ax.axvline(b, lw=0.3, color="gray", ls="--")
        boundary = boundaries[b]
        for bnd in boundary:
            ax.plot([b, b+1], [bnd["lb"], bnd["lb"]], ls="--", color="b", lw=0.5)
            ax.plot([b, b+1], [bnd["ub"], bnd["ub"]], ls="--", color="g", lw=0.5)
            #ax.scatter([b+0.5], [bnd["peak"]], marker="*", color="k", s=3)
    fonttext["size"] = 6
    for x in clusters.keys():
        C = clusters[x]
        for _c in C: 
            if clust_idf is None: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, "%02d"%int(x),
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
            else: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, clust_idf[x],
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
    ax.axvline(b+1, lw=0.3, color="gray", ls="--")
    ax.set_title(title)
    fonttext["size"] = 10
    if gflg_type>=0: ax.text(1.05, 0.5, case, ha="center", va="center", fontdict=fonttext, transform=ax.transAxes, rotation=90)
    ax.set_xticks(np.arange(blim[0], blim[1] + 1) + 0.5)
    ax.set_xticklabels(np.arange(blim[0], blim[1] + 1))
    fig.savefig(fname, bbox_inches="tight")
    return

def cluster_stats(df, cluster, fname, title):
    fig, axes = plt.subplots(figsize=(4,8), nrows=3, ncols=1, dpi=120, sharey=True)
    v, w, p = [], [], []
    for c in cluster:
        v.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].v.tolist())
        w.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].w_l.tolist())
        p.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].p_l.tolist())
    ax = axes[0]
    v, w, p = np.array(v), np.array(w), np.array(p)
    v[v<-1000] = -1000
    v[v>1000] = 1000
    l, u = np.quantile(v,0.1), np.quantile(v,0.9)
    ax.axvline(np.sign(l)*np.log10(abs(l)), ls="--", lw=1., color="r")
    ax.axvline(np.sign(u)*np.log10(abs(u)), ls="--", lw=1., color="r")
    ax.hist(np.sign(v)*np.log10(abs(v)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"V, $ms^{-1}$")
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])
    ax.text(0.7, 0.7, r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(v[(v>l) & (v<u)]), np.median(v[(v>l) & (v<u)])) + "\n"\
            + r"$V_{max}$=%.1f, $V_{min}$=%.1f"%(np.max(v[(v>l) & (v<u)]), np.min(v[(v>l) & (v<u)])) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(v[(v>l) & (v<u)]),len(v[(v>l) & (v<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax = axes[1]
    w[w>100]=100
    w[w<-100] = 100
    l, u = np.quantile(w,0.1), np.quantile(w,0.9)
    ax.axvline(l, ls="--", lw=1., color="r")
    ax.axvline(u, ls="--", lw=1., color="r")
    ax.hist(w, bins=range(-100,100,1), histtype="step", density=False)
    ax.set_xlabel(r"W, $ms^{-1}$")
    ax.set_xlim([-100,100])
    ax.set_ylabel("Counts")
    ax.text(0.75, 0.8, r"$W_{\mu}$=%.1f, $\hat{W}$=%.1f"%(np.mean(w[(w>l) & (w<u)]), np.median(w[(w>l) & (w<u)])) + "\n"\
            + r"$W_{max}$=%.1f, $W_{min}$=%.1f"%(np.max(w[(w>l) & (w<u)]), np.min(w[(w>l) & (w<u)])) + "\n"\
            + r"$W_{\sigma}$=%.1f, n=%d"%(np.std(w[(w>l) & (w<u)]),len(w[(w>l) & (w<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax = axes[2]
    p[p>30]=30
    l, u = np.quantile(p,0.1), np.quantile(p,0.9)
    ax.axvline(l, ls="--", lw=1., color="r")
    ax.axvline(u, ls="--", lw=1., color="r")
    ax.hist(p, bins=range(0,30,1), histtype="step", density=False)
    ax.set_xlabel(r"P, $dB$")
    ax.set_xlim([0,30])
    ax.set_ylabel("Counts")
    ax.text(0.75, 0.8, r"$P_{\mu}$=%.1f, $\hat{P}$=%.1f"%(np.mean(p[(p>l) & (p<u)]), np.median(p[(p>l) & (p<u)])) + "\n"\
            + r"$P_{max}$=%.1f, $P_{min}$=%.1f"%(np.max(p[(p>l) & (p<u)]), np.min(p[(p>l) & (p<u)])) + "\n"\
            + r"$P_{\sigma}$=%.1f, n=%d"%(np.std(p[(p>l) & (p<u)]),len(p[(p>l) & (p<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(fname, bbox_inches="tight")
    return

def general_stats(g_stats, fname):
    fig, axes = plt.subplots(figsize=(6,10), nrows=4, ncols=1, dpi=120, sharex=True)
    ax = axes[0]
    width=0.2
    df = pd.DataFrame.from_records(list(g_stats.values()))
    ax.bar(df.bmnum-width, df.sound, width=0.3, color="r", label="S")
    ax.set_ylabel(r"$N_{sounds}$", fontdict=font)
    ax.legend(loc=2)
    ax = ax.twinx()
    ax.bar(df.bmnum+width, df.echo, width=0.3, color="b", label="E")
    ax.set_ylabel(r"$N_{echo}$", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax.legend(loc=1)

    ax = axes[1]
    ax.errorbar(df.bmnum, df.v, yerr=df.v_mad, color="r", elinewidth=2.5, ecolor="r", fmt="o", ls="None")
    ax.set_ylim(-20, 40)
    ax.set_ylabel(r"$V_{med}$", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax = axes[2]
    ax.errorbar(df.bmnum, df.w, yerr=df.w_mad, color="b", elinewidth=1.5, ecolor="b", fmt="o", ls="None")
    ax.set_ylim(-20, 40)
    ax.set_ylabel(r"$W_{med}$", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax = axes[3]
    ax.errorbar(df.bmnum, df.p, yerr=df.p_mad, color="k", elinewidth=0.5, ecolor="k", fmt="o", ls="None")
    ax.set_ylim(-20, 40)
    ax.set_ylabel(r"$P_{med}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(df.bmnum)

    fig.subplots_adjust(hspace=0.1)
    fig.savefig(fname, bbox_inches="tight")
    return


def individal_cluster_stats(cluster, df, fname, title):
    fig = plt.figure(figsize=(8,12), dpi=120)
    V = []
    vbbox, vbox = {}, []
    vecho, vsound, echo, sound = {}, {}, [], []
    beams = []
    for c in cluster:
        v = np.array(df[(df.slist>=c["lb"]) & (df.slist<=c["ub"]) & (df.bmnum==c["bmnum"])].v)
        V.extend(v.tolist())
        v[v<-1000] = -1000
        v[v>1000] = 1000
        if c["bmnum"] not in vbbox.keys(): 
            vbbox[c["bmnum"]] = v.tolist()
            vecho[c["bmnum"]] = c["echo"]
            vsound[c["bmnum"]] = c["sound"]
        else: 
            vbbox[c["bmnum"]].extend(v.tolist())
            vecho[c["bmnum"]] += c["echo"]
            #vsound[c["bmnum"]] += c["sound"]
    beams = sorted(vbbox.keys())
    avbox = []
    for b in beams:
        if b!=15: avbox.append(vbbox[b])
        vbox.append(vbbox[b])
        echo.append(vecho[b])
        sound.append(vsound[b])
    from scipy import stats
    pval = -1.
    if len(vbox) > 1: 
        H, pval = stats.f_oneway(*avbox)
    ax = plt.subplot2grid((4, 2), (0, 1), colspan=1)
    V = np.array(V)
    V[V<-1000] = -1000
    V[V>1000] = 1000
    l, u = np.quantile(V,0.05), np.quantile(V,0.95)
    ax.axvline(np.sign(l)*np.log10(abs(l)), ls="--", lw=1., color="r")
    ax.axvline(np.sign(u)*np.log10(abs(u)), ls="--", lw=1., color="r")
    ax.hist(np.sign(V)*np.log10(abs(V)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.text(0.7, 0.8, r"$V_{min}$=%.1f, $V_{max}$=%.1f"%(np.min(V[(V>l) & (V<u)]), np.max(V[(V>l) & (V<u)])) + "\n"\
            + r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(V[(V>l) & (V<u)]), np.median(V[(V>l) & (V<u)])) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(V[(V>l) & (V<u)]),len(V[(V>l) & (V<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_xlabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_yticklabels([])
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])

    ax = plt.subplot2grid((4, 2), (0, 0), colspan=1)
    ax.hist(np.sign(V)*np.log10(abs(V)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.text(0.7, 0.8, r"$V_{min}$=%.1f, $V_{max}$=%.1f"%(np.min(V), np.max(V)) + "\n"\
            + r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(V), np.median(V)) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(V),len(V)),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_ylabel("Counts", fontdict=font)
    ax.set_xlabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])
    
    ax = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    ax.boxplot(vbox, flierprops = dict(marker="o", markerfacecolor="none", markersize=0.8, linestyle="none"))
    ax.set_ylim(-100,100)
    ax.set_xlabel(r"Beams", fontdict=font)
    ax.set_ylabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_xticklabels(beams)
    ax.text(1.05,0.5, "p-val=%.2f"%pval, horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict={"size":10}, rotation=90)
    ax.axhline(0, ls="--", lw=0.5, color="k")
    fig.suptitle(title, y=0.92)

    ax = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax.boxplot(vbox, flierprops = dict(marker="o", markerfacecolor="none", markersize=0.8, linestyle="none"),
            showbox=False, showcaps=False)
    ax.set_xticklabels(beams)
    ax.axhline(0, ls="--", lw=0.5, color="k")
    ax.set_xlabel(r"Beams", fontdict=font)
    ax.set_ylabel(r"V, $ms^{-1}$", fontdict=font)

    ax = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    width=0.2
    ax.bar(np.array(beams)-width, sound, width=0.3, color="r", label="S")
    ax.set_ylabel(r"$N_{sounds}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.legend(loc=2)
    ax = ax.twinx()
    ax.bar(np.array(beams)+width, echo, width=0.3, color="b", label="E")
    ax.set_ylabel(r"$N_{echo}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(beams)
    ax.legend(loc=1)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig(fname, bbox_inches="tight")
    return

class FoV(object):
    import sys
    sys.path.append("../darn_tools/")
    sys.path.append("../darn_tools/utils/")
    import plot_utils
    
    def __init__(self, rad, dates):
        self.rad = rad
        self.dates = dates
        return
    
    def box(self, dlist, dtype="fitacf", fname= None, gflg_key="gflg"):
        import cartopy
        self.dtype = dtype
        fig = plt.figure(figsize=(12, 5), dpi=180)
        for i in range(3):
            ax = fig.add_subplot(131+i, projection="sdfovcarto",\
                                 coords="geo", rad=self.rad, plot_date=self.dates[i])
            ax.coastlines(lw=0.1, alpha=0.5)
            ax.overlay_radar()
            ax.overlay_fov(maxGate=50)
            if i != 0: 
                ax.grid_on(draw_labels=[False, False, False, False])
                ax.enum(bounds=[-110, -70, 45, 85],dtype=self.dtype)
            else:
                ax.grid_on(draw_labels=[False, False, True, True])
                ax.enum(bounds=[-110, -70, 45, 85], text_coord=True,dtype=self.dtype)
            if i==2: ax.overlay_fitacfp_radar_data(dlist[i], add_colorbar=True, gflg_key=gflg_key,
                                                   colorbar_label=r"Velocity, $ms^{-1}$",
                                                   p_max=500, p_min=-500, p_step=50)
            else: ax.overlay_fitacfp_radar_data(dlist[i], add_colorbar=False, gflg_key = gflg_key,
                                               p_max=500, p_min=-500, p_step=50)
        fig.subplots_adjust(wspace=0.2, hspace=0.1)
        if fname is None: fig.savefig("test.png", bbox_inches="tight")
        else: fig.savefig(fname, bbox_inches="tight")
        return

class FanPlots(object):
    """Class creates Fov-Fan plots of the images."""
    
    def __init__(self, figsize=(5,5), nrows=1, ncols=1, dpi=120, ):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.nrows = nrows
        self.ncols = ncols
        self.self._num_subplots_created = 0
        return
    
    def _add_axis(self, date, rad, coords="geo", map_projection=None, bounds=[]):
        self._num_subplots_created += 1
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
        """ Plot data - overlay on top of map """
        return
    
    def save(self, filepath):
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
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam, rounding=False)
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
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam="labels", rounding=False)
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
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam, rounding=False)
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
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return