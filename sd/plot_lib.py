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
#import spacepy.plot as splot
#import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
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
        cast = "gflg"
        if self.gflg_type > 0: cast = "gsflg_" + str(self.gflg_type)
        if figure_name is not None: self.figure_name = figure_name
        if self.figure_name == "1plot":
            C = 20
            dur = int((self.data[1].stime.replace(microsecond=0)-self.data[0].stime.replace(microsecond=0)).seconds/60.)
            fig, ax = plt.subplots(figsize=(6, 4), sharex="all", sharey="all", nrows=1, ncols=1, dpi=100)
            mx = self.draw_scatter(ax, self.data[1], label={"xx":0.75, "yy":1.03, "text":r"$power^{LoS}[dB]$"}, zparam="p_l", c=C, cast=cast)
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
    fonttext["size"] = 5
    for i, k in enumerate(gate_lims.keys()):
        ax = axes[0, i]
        dx = df[(df.slist>=gate_lims[k][0]) & (df.slist<=gate_lims[k][1])]
        bins = list(range(-100,100,1))
        ax.hist(dx.v, bins=bins, histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Velocity (m/s)", horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{v},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.v), np.median(dx.v),np.nanquantile(np.abs(dx.v), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[1, i]
        dx = df[(df.slist>=gate_lims[k][0]) & (df.slist<=gate_lims[k][1])]
        bins = list(range(-1000,1000,1))
        ax.hist(dx.v, bins=bins, histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Velocity (m/s)", horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{v},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.v), np.median(dx.v),np.nanquantile(np.abs(dx.v), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[2, i]
        ax.hist(dx.w_l, bins=range(0,100,1), histtype="step", density=False)
        if i==0: ax.set_ylabel("Density", fontdict=font)
        if i==len(gate_lims.keys())-1: ax.text(1.05, 0.5, "Width (m/s)", horizontalalignment="center", verticalalignment="center",
                                transform=ax.transAxes, fontdict=fonttext, rotation=90)
        stat = "\n" + r"$\mu,\hat{w},CI(90)$=%.1f,%.1f,%.1f"%(np.mean(dx.w_l), np.median(dx.w_l), np.nanquantile(np.abs(dx.w_l), 0.9))
        ax.text(0.6, 0.7, str(k) + stat, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        ax = axes[3, i]
        ax.scatter(dx.v, np.abs(dx.w_l), s=2)
        ax.set_xlim(-100,100)
        ax.set_ylim(0,100)
        if i==0: ax.set_xlabel("Velocity (m/s)", fontdict=font)
        ax.set_ylabel("Width (m/s)", fontdict=font)
    fig.savefig(fname.replace("png", "pdf"), bbox_inches="tight")
    return

def beam_gate_boundary_plots(boundaries, clusters, clust_idf, glim, blim, title, fname):
    """ Beam gate boundary plots showing the distribution of clusters """
    fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1, dpi=240)
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
    for x in clusters.keys():
        C = clusters[x]
        for _c in C: 
            if clust_idf is None: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, "%02d"%int(x),
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
            else: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, clust_idf[x],
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
    ax.axvline(b+1, lw=0.3, color="gray", ls="--")
    ax.set_title(title)
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
    fig, axes = plt.subplots(figsize=(6,5), nrows=2, ncols=1, dpi=120, sharex=True)
    ax = axes[0]
    width=0.2
    df = pd.DataFrame.from_records(list(g_stats.values()))
    ax.bar(df.bmnum-width, df.sound, width=0.3, color="r", label="S")
    ax.set_ylabel(r"$N_{sounds}$", fontdict=font)
    ax.legend(loc=2)
    ax = ax.twinx()
    ax.bar(df.bmnum+width, df.echo, width=0.3, color="b", label="E")
    ax.set_ylabel(r"$N_{echo}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax.legend(loc=1)

    ax = axes[1]
    ax.errorbar(df.bmnum, df.v, yerr=df.v_mad, color="r", elinewidth=2.5, ecolor="r", fmt="o", ls="None", label="V")
    ax.errorbar(df.bmnum, df.w, yerr=df.w_mad, color="b", elinewidth=1.5, ecolor="b", fmt="o", ls="None", label="W")
    ax.errorbar(df.bmnum, df.p, yerr=df.p_mad, color="k", elinewidth=0.5, ecolor="k", fmt="o", ls="None", label="P")
    ax.set_ylim(-20, 40)
    ax.set_ylabel(r"$V_{med},W_{med},P_{med}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax.legend(loc=1)

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
        print(H,pval)
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
