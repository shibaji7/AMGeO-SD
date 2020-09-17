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
import datetime as dt

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

import glob
import os
import utils

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
        ax = self.set_axes(ax)
        ax.text(label["xx"], label["yy"], label["text"],
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes,fontdict=fonttext)
        X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(scan, p=zparam), zparam=zparam)
        self.draw_specific_points(ax, scan, u, cast=cast, keys=keys, c=c)
        return np.max(np.abs(u))


class InvestigativePlots(MasterPlotter):
    """Class that plots all the investigative figures"""

    def __init__(self, figure_name, e, rad, sim_id, data, keywords={}, folder="data/outputs/{rad}/{sim_id}/",
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
            print("\n Create folder - mkdir {folder}".format(folder=self.folder))
            os.system("mkdir -p {folder}".format(folder=self.folder))
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        return

    def draw(self, figure_name=None):
        """
        Draw the figures for the scan
        """
        if figure_name is not None: self.figure_name = figure_name
        if self.figure_name == "1plot":
            C = 20
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            fig, ax = plt.subplots(figsize=(6, 4), sharex="all", sharey="all", nrows=1, ncols=1, dpi=100)
            mx = self.draw_scatter(ax, self.data[1], label={"xx":0.75, "yy":1.03, "text":r"$power^{LoS}[dB]$"}, zparam="p_l", c=C)
            self.set_size_legend(ax, mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(ax), c=C)
            fig.text(0.5, 0.01, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
            fig.suptitle("Date=%s, Rad=%s, Scan Dur=%d min"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(),dur), size=12,
                                        y=0.98, fontdict={"color":"darkblue"})
            ax.text(0.2, 1.03, "GS Flag=%s"%(gs_map[self.gflg_type]), fontdict={"color":"blue","size":7}, 
                    ha="center", va="center", transform=ax.transAxes)
            X,Y,u = utils.get_gridded_parameters(utils.parse_parameter(self.data[1], p="v"),zparam="v")
            v = np.zeros_like(u)
            q = ax.quiver(X, Y, u.T, v.T)
            ax.quiverkey(q, 0.75, 0.98, 1000, r"$1000$ m/s", labelpos="S",
                    fontproperties={"size": font["size"], "weight": "bold"}, color="r", labelcolor="r")
            fig.savefig("{folder}{date}.1plot.png".format(folder=self.folder,
                rad=self.rad, date=self.e.strftime("%Y%m%d%H%M")),bbox_inches="tight")
        elif self.figure_name == "4plot":
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            fig, axes = plt.subplots(figsize=(6, 6), sharex="all", sharey="all", nrows=2, ncols=2, dpi=100)
            fig.subplots_adjust(wspace=0.1,hspace=0.1)
            fig.text(0.5, 0.06, "Beams", ha="center",fontdict={"color":"blue"})
            fig.text(0.06, 0.5, "Gates", va="center", rotation="vertical",fontdict={"color":"blue"})
             
            fig.suptitle("Date=%s, Rad=%s, Scan Dur=%d min"%(self.e.strftime("%Y-%m-%d %H:%M"),self.rad.upper(),dur), size=12,
                                        y=0.94, fontdict={"color":"darkblue"})
            C = 30
            self.draw_quiver(axes[0,0], self.data[1], label={"xx":.9, "yy":1.05, "text":r"$v^{LoS}[m/s]$"}, zparam="v")
            axes[0,0].text(0.2, 1.05, "GS Flag=%s"%(gs_map[self.gflg_type]), fontdict={"color":"blue","size":7},
                                        ha="center", va="center", transform=axes[0,0].transAxes)
            self.set_point_legend(axes[0,0])
            mx = self.draw_scatter(axes[0,1], self.data[1], 
                    label={"xx":0.75, "yy":1.05, "text":r"$v_{error}^{LoS}[m/s]$"}, zparam="v_e", c=C)
            self.set_size_legend(axes[0,1], mx, leg_keys=("90", "45", "10"), leg=self.set_point_legend(axes[0,1]), c=C)
            mx = self.draw_scatter(axes[1,0], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$power^{LoS}[dB]$"}, zparam="p_l", c=C)
            self.set_size_legend(axes[1,0], mx, leg_keys=("30", "15", "5"), leg=self.set_point_legend(axes[1,0]), c=C)
            mx = self.draw_scatter(axes[1,1], self.data[1], label={"xx":0.75, "yy":1.05, "text":r"$width^{LoS}[m/s]$"}, zparam="w_l", c=C)
            self.set_size_legend(axes[1,1], mx, leg_keys=("200", "100", "10"), leg=self.set_point_legend(axes[1,1]), c=C)
            fig.savefig("{folder}{date}.4plot.png".format(folder=self.folder,
                                rad=self.rad, date=self.e.strftime("%Y%m%d%H%M")),bbox_inches="tight")
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
            self.draw_quiver(ax, self.data[0], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[0].strftime("%H:%M")}, cast="gflg")
            self.set_point_legend(ax)
            ax.set_xticklabels([])
            ax = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
            self.draw_quiver(ax, self.data[1], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[1].strftime("%H:%M")}, cast="gflg")
            self.set_point_legend(ax)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
            self.draw_quiver(ax, self.data[2], label={"xx":0.2, "yy":1.05, "text":"UT=%s"%scan_times[2].strftime("%H:%M")}, cast="gflg")
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
            fig.savefig("{folder}{date}.5plot.png".format(folder=self.folder,
                                rad=self.rad, date=self.e.strftime("%Y%m%d%H%M")),bbox_inches="tight")
        return


class MovieMaker(MasterPlotter):
    """Class creates movie out of the images."""

    def __init__(self, rad, dates, scans, figure_name, sim_id, keywords={}, folder="data/outputs/{rad}/{sim_id}/",
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
            print("\n Create folder - mkdir {folder}".format(folder=self.folder))
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
        fig = plt.figure(figsize=(3,3),dpi=100)
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
        fig = plt.figure(figsize=(3,3),dpi=100)
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
        print("\n Running movie making script - ", cmd)
        os.system(cmd)
        return

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
        self._create_movie()
        return
