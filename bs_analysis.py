#!/usr/bin/env python

"""bs_analsys.py: module is dedicated to analyze bacscatter for sub-auroral IS drifts."""

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
import spacepy.plot as splot
import matplotlib.colors as mcolors

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
import os
from netCDF4 import Dataset, date2num, num2date
import pandas as pd
import numpy as np
import glob



def slow_filter(df, params=["bm","slist"]):
    """
    Do filter for sub-auroral scatter
    """
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=4, min_samples=60).fit(df[params].values)
    df["labels"] = clustering.labels_
    return df

def sais_is_scatter(v,w):
    print(w.shape, v.shape, (0.7*(v+5)**2).shape)
    sctr = w-(50-(0.7*(v+5)**2)) > 0
    return sctr

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

rad = "bks"
sim_id = "L100"

datfname = glob.glob("data/outputs/{rad}/{sim_id}/*.gz".format(rad=rad, sim_id=sim_id))[0]
ev = datfname.split("/")[-1]
dtu = (ev.split("_")[-1]).split(".")[0]
start, end = (ev.split("_")[1]).split(".")[1], (ev.split("_")[-1]).split(".")[1]
os.system("gzip -d " + datfname)
ds = Dataset(datfname.replace(".gz",""))
os.system("gzip " + datfname.replace(".gz", ""))

df = pd.DataFrame()
df["slist"] = ds.variables["slist"][:].ravel()
df["v"] = ds.variables["v"][:].ravel()
df["w"] = ds.variables["w_l"][:].ravel()
df["p"] = ds.variables["p_l"][:].ravel()
df["bm"] = np.array(ds.variables["bmnum"][:].ravel().tolist()*110)

fig, axes = plt.subplots(figsize=(15,9), nrows=3, ncols=5, sharey="row", dpi=150)
slist_ranges = [(0,10), (10,30), (30,60), (60,70), (70,110)]
d = False
for i, rng in enumerate(slist_ranges):
    u = df[(df.slist >= rng[0]) & (df.slist < rng[1])]
    ax = axes[0, i]
    bins = np.arange(-50,50,5); bins[0], bins[-1] = -9999, 9999
    ax.hist(u.v, bins=bins, density=d, color="r", alpha=0.4)
    ax.text(0.75, 0.8, r"$\mu$=%.1f, $\bar{\mu}$=%.1f"%(np.mean(u.v), np.median(u.v)) + "\n"\
            + r"M=%.1f, m=%.1f"%(np.max(u.v), np.min(u.v)) + "\n"\
            + r"$\sigma$=%.1f, n=%d"%(np.std(u.v),len(u)), 
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, 
            bbox=dict(facecolor="gray", alpha=0.5), fontdict={"size":8})
    ax.set_xlim(-50,50)
    ax.set_xlabel("Velocity (m/s)", fontdict=font)
    ax.set_title( "Gate - Range(%d,%d)"%rng, fontdict=fonttext)
    ax = axes[1, i]
    bins = np.arange(0,100,5)
    ax.hist(u.w, bins=bins, density=d, color="b", alpha=0.4)
    ax.set_xlim(0,60)
    ax.text(0.75, 0.8, r"$\mu$=%.1f, $\bar{\mu}$=%.1f"%(np.mean(u.w), np.median(u.w)) + "\n"\
            + r"M=%.1f, m=%.1f"%(np.max(u.w), np.max(u.w)) + "\n"\
            + r"$\sigma$=%.1f, n=%d"%(np.std(u.w),len(u)),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
            bbox=dict(facecolor="gray", alpha=0.5), fontdict={"size":8})
    ax.set_xlabel("Width (m/s)", fontdict=font)
    ax = axes[2, i]
    bins = np.arange(0,30,3)
    ax.hist(u.p, bins=bins, density=d, color="k", alpha=0.4)
    ax.text(0.75, 0.8, r"$\mu$=%.1f, $\bar{\mu}$=%.1f"%(np.mean(u.p), np.median(u.p)) + "\n"\
            + r"M=%.1f, m=%.1f"%(np.max(u.p), np.max(u.p)) + "\n"\
            + r"$\sigma$=%.1f, n=%d"%(np.std(u.p),len(u)),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,                                                                bbox=dict(facecolor="gray", alpha=0.5), fontdict={"size":8})
    ax.set_xlim(0,30)
    ax.set_xlabel("Power (dB)", fontdict=font)
axes[0,0].set_ylabel("Counts", fontdict=font)
axes[1,0].set_ylabel("Counts", fontdict=font)
axes[2,0].set_ylabel("Counts", fontdict=font)
fig.suptitle("Date- %s, Dur- (%s, %s) UT, Rad- %s"%(dtu,start,end,rad.upper()))
fig.savefig("data/outputs/{rad}/{sim_id}/histogram.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()

## 2D Scatter Plots
fig, ax = plt.subplots(figsize=(3,3), nrows=1, ncols=1, sharey="row", dpi=120)
colors = ["blue", "red", "green", "black", "magenta"]
colors = ["red", "red", "blue", "red", "blue"]
for i, (rng, col) in enumerate(zip(slist_ranges, colors)):
    u = df[(df.slist >= rng[0]) & (df.slist < rng[1])]
    ax.scatter(u.v, u.w, s=2, c=col, alpha=0.7)
    #ax.scatter(u.v, u.w, s=2, c=col, label="Range(%d,%d)"%rng)
v = np.linspace(-30,30,100)
w = 40-(0.2*(v+5)**2)
ax.plot(v,w,lw=1.2,color="gray")
#ax.legend(scatterpoints=3, ncol=1, fontsize=10, frameon=True)
ax.grid(True)
ax.set_xlim(-50,50)
ax.set_ylim(0,50)
ax.set_ylabel("Width [m/s]", fontdict={"size":10})
ax.set_xlabel("Velocity [m/s]", fontdict={"size":10})
#ax.set_title("Date- %s, Dur- (%s, %s) UT, Rad- %s"%(dtu,start,end,rad.upper()))
fig.savefig("data/outputs/{rad}/{sim_id}/sctr2D.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()

## Velocity gate scatter plot
vlim = 200
fig, ax = plt.subplots(figsize=(5,5), nrows=1, ncols=1, sharey="row", dpi=150)
for i, (rng, col) in enumerate(zip(slist_ranges, colors)):
    u = df[(df.slist >= rng[0]) & (df.slist < rng[1])]
    slist, v = np.array(u.slist), np.array(u.v)
    v[v>vlim] = vlim-1; v[v<-vlim] = -vlim+1
    ax.scatter(rand_jitter(slist), rand_jitter(v), c=col, s=1, label="Range(%d,%d)"%rng)
ax.set_ylim(-vlim,vlim)
ax.set_title("Date-%s, Dur- (%s, %s), Rad- %s"%(dtu,start,end,rad.upper()))
ax.legend(scatterpoints=3, ncol=1, fontsize=10, frameon=True)
ax.grid(True)
ax.set_xlabel("Gate")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Date- %s, Dur- (%s, %s) UT, Rad- %s"%(dtu,start,end,rad.upper()))
fig.savefig("data/outputs/{rad}/{sim_id}/gatesctr.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()


vlim = 200
fig, ax = plt.subplots(figsize=(5,5), nrows=1, ncols=1, sharey="row", dpi=150)
for i, (rng, col) in enumerate(zip(slist_ranges, colors)):
    u = df[(df.slist >= rng[0]) & (df.slist < rng[1])]
    slist, v, bm = np.array(u.slist), np.array(u.v), np.mod(u.bm,4)
    v[v>vlim] = vlim-1; v[v<-vlim] = -vlim+1
    ax.scatter(rand_jitter(slist), rand_jitter(v), c=bm, s=1)
ax.set_ylim(-vlim,vlim)
ax.set_title("Date-%s, Dur- (%s, %s), Rad- %s"%(dtu,start,end,rad.upper()))
ax.grid(True)
ax.set_xlabel("Gate")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Date- %s, Dur- (%s, %s) UT, Rad- %s"%(dtu,start,end,rad.upper()))
fig.savefig("data/outputs/{rad}/{sim_id}/bmgatesctr.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()

## Velocity gate scatter plot
df = df.dropna()
fig, axes = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharey="row", dpi=150)
ax = axes[0]
ax.scatter(rand_jitter(df.slist), rand_jitter(df.p), color="r", s=1)
ax.set_title("Date-%s, Dur- (%s, %s), Rad- %s"%(dtu,start,end,rad.upper()))
ax.grid(True)
ax.set_ylabel("Power [dB]")
ax = axes[1]
ax.scatter(df.groupby(["slist"]).count().reset_index()["slist"], df.groupby(["slist"]).count().reset_index()["p"], color="k", s=3)
ax.grid(True)
ax.set_xlabel("Gate")
ax.set_ylabel("Count")
fig.savefig("data/outputs/{rad}/{sim_id}/gatedistribution.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()

## Filter and detection of SAIS
df = slow_filter(df)
fig, axes = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharey="row", dpi=150)
ax = axes[0]
df = df[df.labels!=-1]
v = np.array(df.v)
v[v>vlim] = vlim-1; v[v<-vlim] = -vlim+1
ax.scatter(rand_jitter(df.slist), rand_jitter(v), c=np.array(df.labels), s=1)
ax.set_ylim(-vlim,vlim)
ax.set_title("Date-%s, Dur- (%s, %s), Rad- %s"%(dtu,start,end,rad.upper()))
ax.grid(True)
ax.set_ylabel("Velocity [m/s]")
ax = axes[1]
sais = sais_is_scatter(np.array(df.v), np.array(df.w))
ax.scatter(rand_jitter(df.slist)[sais], rand_jitter(v)[sais]-100, c="r", s=1, label="IS + SAIS + Others")
ax.scatter(rand_jitter(df.slist)[~sais], rand_jitter(v)[~sais], c="k", s=1, label="GS", alpha=.5)
ax.set_ylim(-vlim,vlim)
ax.legend(scatterpoints=3, ncol=1, fontsize=10, frameon=True)
ax.grid(True)
ax.set_ylabel("Velocity [m/s]")
ax.set_xlabel("Gate")
fig.savefig("data/outputs/{rad}/{sim_id}/filtered.png".format(rad=rad, sim_id=sim_id), bbox_inches="tight")
plt.close()
