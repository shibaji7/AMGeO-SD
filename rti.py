import sys
sys.path.append("sd/")
sys.path.append("tools/")

import datetime as dt
from get_fit_data import FetchData
from boxcar_filter import Filter
import utils
import multiprocessing as mp
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import numpy as np

class RTI(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """
    
    def __init__(self, nrang, fig_title="", num_subplots=4):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=180) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15)
        matplotlib.rcParams.update({"font.size": 10})
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, p_step=25, xlabel="Time UT", zparam="v",
                    label="Velocity [m/s]"):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="mdates", yparam="slist", zparam=zparam, rounding=False)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = plt.cm.jet_r
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([mdates.date2num(dt.datetime(2014,2,20)),
                     mdates.date2num(dt.datetime(2014,2,21))])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm, snap=True)
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
    
    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

beam = 13
p0 = mp.Pool(8)
drange = [dt.datetime(2014,2,19,23,50), dt.datetime(2014,2,21,1)]
io = FetchData("bks", drange, ftype="fitacf3")
_, scan = io.fetch_data(by="scan")
df0 = io.scans_to_pandas(scan)
df0["mdates"] = df0.time.apply(lambda x: mdates.date2num(x))
dfs = []
taus = [0.2, 0.4]

for th in taus:
    fil = Filter(thresh=th, pbnd=[0.1, 0.9], pth=0.5)
    scans, fscans, kscans = [], [], []
    for s in scan:
        scans.append(fil._discard_repeting_beams(s))
    for i in range(1, len(scans)-2):
        sc = scans[i-1:i+2]
        fscans.append(sc)
    partial_filter = partial(fil.doFilter, gflg_type=-1, do_cluster=True)
    for s in p0.map(partial_filter, fscans):
        kscans.append(s)
    u = io.scans_to_pandas(kscans)
    u["mdates"] = u.time.apply(lambda x: mdates.date2num(x))
    dfs.append(u)

rti = RTI(100, "", num_subplots=len(taus)+1)
rti.addParamPlot(df0, beam, f"(a) 20 Feb 2014, BKS, Beam: {beam}")
labels = ["(b)", "(c)"]
for i in range(len(taus)):
    rti.addParamPlot(dfs[i], beam, r"%s $\tau=%.1f$"%(labels[i], taus[i]))
rti.save("data/med-filt.png")
rti.close()