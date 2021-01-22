#!/usr/bin/env python

"""clustering_algo.py: module is dedicated to run all clusteing analysis and classification."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
import pandas as pd
from scipy import stats, ndimage
from loguru import logger
import traceback

from sklearn.cluster import DBSCAN

import utils
from boxcar_filter import Filter

from plotlib import *

class BeamGate(object):
    """ Class to filter middle latitude radars """

    def __init__(self, rad, scans, eps=2, min_samples=10, _dict_={}):
        """
        initialize variables
        
        rad: Radar code
        scans: List of scans
        eps: Radius of DBSCAN
        min_samples: min samplese of DBSCAN
        """
        logger.info("BeamGate Cluster")
        self.rad = rad
        filt = Filter()
        self.scans = [filt._discard_repeting_beams(s) for s in scans]
        self.eps = eps
        self.min_samples = min_samples
        self.boundaries = {}
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self.fig_folder = utils.get_config("BASE_DIR") + self.sim_id + "/" + self.rad +\
                    "/figs/%s_%s/"%(self.start.strftime("%Y-%m-%d-%H-%M"), self.end.strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.fig_folder): os.system("mkdir -p " + self.fig_folder)
        return

    def extract_gatelims(self, df):
        """
        Extract gate limits for individual clusters
        """
        glims = {}
        for l in set(df.labels):
            if l >= 0:
                u = df[df.labels==l]
                if len(u) > 0: glims[l] = [np.min(u.slist) + 1, np.max(u.slist) - 1]
        return glims

    def filter_by_dbscan(self, df, bm):
        """
        Do filter by dbscan name
        """
        du, sb = df[df.bmnum==bm], [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l), 
                "sound":sb[0], "echo":len(du), "v_mad": stats.median_absolute_deviation(du.v), 
                "p_mad": stats.median_absolute_deviation(du.p_l), "w_mad": stats.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all":
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else:
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        l = du.groupby("time")
        logger.info(f" Beam Analysis:  {bm}, {len(l)}")
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        eco, Eco = utils.smooth(eco, self.window), utils.smooth(Eco, self.window)
        glims, labels = {}, []
        ds = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(du[["slist"]].values)
        du["labels"] = ds.labels_
        names = {}
        for j, r in enumerate(set(ds.labels_)):
            x = du[du.labels==r]
            glims[r] = [np.min(x.slist), np.max(x.slist)]
            names[r] = "C"+str(j)
            if r >= 0: self.boundaries[bm].append({"peak": Rng[np.min(x.slist) + Eco[np.min(x.slist):np.max(x.slist)].argmax()],
                "ub": np.max(x.slist), "lb": np.min(x.slist), "value": np.max(eco)*xpt, "bmnum": bm, "echo": len(x), "sound": sb[0]})
        logger.info(f" Individual clster detected: {set(du.labels)}")
        fig_file = self.fig_folder + "gate_{bm}_summary.png".format(bm="%02d"%bm)
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco), fig_file, sb)
        return

    def filter_by_SMF(self, df, bm, method=None):
        """
        Filter by Simple Minded Filter
        method: np/ndimage
        """
        def local_minima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True): 
            """Find all local maxima of the array, separated by at least min_distance."""
            array = np.asarray(array)
            cval = 0 
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest" 
            else: mode = "constant" 
            cval = array.max()+1 
            min_points = array == ndimage.minimum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
            troughs = [indices[min_points] for indices in np.indices(array.shape)][0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] <= np.max(du.slist) - min_distance: troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        def local_maxima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True):
            array = np.asarray(array)
            cval = 0
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest"
            else: mode = "constant"
            cval = array.max()+1
            max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval)
            peaks = [indices[max_points] for indices in np.indices(array.shape)][0]
            return peaks

        def local_minima_np(array):
            """ Local minima by numpy stats """
            troughs = signal.argrelmin(eco, order=self.order)[0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] != np.max(du.slist): troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        if bm == "all": du = df.copy()
        else: du = df[df.bmnum==bm]
        l = len(du.groupby("time"))
        logger.info(f" Beam Analysis:  {bm}, {l}")
        sb = [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l),
                "sound":sb[0], "echo":len(du), "v_mad": stats.median_absolute_deviation(du.v),
                "p_mad": stats.median_absolute_deviation(du.p_l), "w_mad": stats.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all": 
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else: 
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        glims, labels = {}, []
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        eco, Eco = utils.smooth(eco, self.window), utils.smooth(Eco, self.window)
        du["labels"] = [np.nan] * len(du)
        names = {}
        if method == "np": troughs = local_minima_np(np.array(eco))
        else: troughs = local_minima_ndimage(np.array(eco), min_distance=5)
        peaks = local_maxima_ndimage(np.array(eco), min_distance=5)
        logger.info(f" Gate bounds (T,P): {troughs}, {peaks}")
        peaks = np.append(peaks, np.median(troughs[-2:]))
        #if len(peaks) + 1 < len(troughs): troughs = troughs[:len(peaks) + 1]
        logger.info(f" Gate bounds adjustment (T,P): {troughs}, {peaks}")
        for r in range(len(troughs)-1):
            glims[r] = [troughs[r], troughs[r+1]]
            du["labels"] = np.where((du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r]), r, du["labels"])
            names[r] = "C" + str(r)
            if r >= 0: 
                self.boundaries[bm].append({"peak": peaks[r], "ub": troughs[r+1], "lb": troughs[r], 
                                            "value": np.max(eco)*xpt, "bmnum": bm, 
                                            "echo": len(du[(du["slist"]<=troughs[r+1]) 
                                                           & (du["slist"]>=troughs[r])]), "sound": sb[0]})

        du["labels"] =  np.where(np.isnan(du["labels"]), -1, du["labels"])
        logger.info(f" Individual clster detected:  {set(du.labels)}")
        fig_file = self.fig_folder + "gate_{bm}_summary.png".format(bm="%02d"%bm)
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco), fig_file, sb)
        return

    def doFilter(self, io, window=11, order=1, beams=range(4,24), themis=15):
        """
        Do filter for sub-auroral scatter
        """
        self.gen_summary = {}
        self.clust_idf = {}
        self.order = order
        self.window = window
        df = pd.DataFrame()
        for i, fsc in enumerate(self.scans):
            dx = io.convert_to_pandas(fsc.beams, v_params=["p_l", "v", "w_l", "slist"])
            df = df.append(dx)
        logger.info(f" Beam range -  {np.min(df.bmnum)}, {np.max(df.bmnum)}")
        if beams == None or len(beams) == 0: 
            bm = "all"
            self.boundaries[bm] = []
            self.filter_by_SMF(df, bm)
        else:
            for bm in beams:
                self.boundaries[bm] = []
                self.gen_summary[bm] = {}
                if bm==themis: 
                    try:
                        #self.filter_by_SMF(df, bm)
                        self.filter_by_dbscan(df, bm)
                    except:
                        logger.error(f"SMF failed for this themis beam {themis}")
                        traceback.print_exc()
                        self.filter_by_dbscan(df, bm)
                else: self.filter_by_dbscan(df, bm)
        title = "Date: %s [%s-%s] UT | %s"%(df.time.tolist()[0].strftime("%Y-%m-%d"),
                df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M"), self.rad.upper())
        self.gc = []
        for x in self.boundaries.keys():
            for j, m in enumerate(self.boundaries[x]):
                self.gc.append(m)
        self.ubeam, self.lbeam = np.max(df.bmnum), np.min(df.bmnum)
        self.sma_bgspace()
        self.probabilistic_cluster_identification(df)
        fname = self.fig_folder + "gate_boundary.png"
        beam_gate_boundary_plots(self.boundaries, self.clusters, None, 
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname)
        fname = self.fig_folder + "gate_boundary_id.png"
        beam_gate_boundary_plots(self.boundaries, self.clusters, self.clust_idf,
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname, gflg_type=self.gflg_type)
        fname = self.fig_folder + "general_stats.png"
        general_stats(self.gen_summary, fname=fname)
        for c in self.clusters.keys():
            try:
                cluster = self.clusters[c]
                fname = self.fig_folder + "clust_%02d_stats.png"%c
                cluster_stats(df, cluster, fname, title+" | Cluster# %02d"%c)
            except:
                logger.error(f"Error in cluster stats, Cluster #{c}")
            #individal_cluster_stats(self.clusters[c], df, self.fig_folder + "ind_clust_%02d_stats.png"%c, 
            #        title+" | Cluster# %02d"%c+"\n"+"Cluster ID: _%s_"%self.clust_idf[c].upper())
        return

    def cluster_identification(self, df, qntl=[0.05,0.95]):
        """ Idenitify the cluster based on Idenitification """
        for c in self.clusters.keys():
            V = []
            for cls in self.clusters[c]:
                v = np.array(df[(df.slist>=cls["lb"]) & (df.slist<=cls["ub"]) & (df.bmnum==cls["bmnum"])].v)
                V.extend(v.tolist())
            V = np.array(V)
            l, u = np.quantile(V,qntl[0]), np.quantile(V,qntl[1])
            Vmin, Vmax = np.min(V[(V>l) & (V<u)]), np.max(V[(V>l) & (V<u)])
            Vmean, Vmed = np.mean(V[(V>l) & (V<u)]), np.median(V[(V>l) & (V<u)])
            Vsig = np.std(V[(V>l) & (V<u)])
            self.clust_idf[c] = "us"
            if Vmin > -20 and Vmax < 20: self.clust_idf[c] = "gs"
            elif (Vmin > -50 and Vmax < 50) and (Vmed-Vsig < -20 or Vmed+Vsig > 20): self.clust_idf[c] = "sais"
        return
    
    def probabilistic_cluster_identification(self, df):
        """ Idenitify the cluster based on Idenitification """
        df["cluster_id"] = [np.nan] * len(df)
        prob_dctor = ScatterTypeDetection()
        for c in self.clusters.keys():
            for cls in self.clusters[c]:
                df["cluster_id"] = np.where((df.slist<=cls["lb"]) & (df.slist<=cls["ub"]) & (df.bmnum==cls["bmnum"]),
                                            c, df.cluster_id)
            du = df[df["cluster_id"]==c]
            prob_dctor.copy(du)
            du, prob_clusters = prob_dctor.run(thresh=[self.lth, self.uth], pth=self.pth)
            auc = prob_clusters[self.gflg_type][c]["auc"]
            if prob_clusters[self.gflg_type][c]["type"] == "IS": auc = 1-auc
            if prob_clusters[self.gflg_type][c]["type"] == "US": self.clust_idf[c] = "us"
            else: self.clust_idf[c] = "%.1f"%(auc*100) + "%" + prob_clusters[self.gflg_type][c]["type"].lower()
        return

    def sma_bgspace(self):
        """
        Simple minded algorithm in B.G space
        """
        def range_comp(x, y, pcnt=0.7):
            _cx = False
            insc = set(x).intersection(y)
            if len(x) < len(y) and len(insc) >= len(x)*pcnt: _cx = True
            if len(x) > len(y) and len(insc) >= len(y)*pcnt: _cx = True
            return _cx
        def find_adjucent(lst, mxx):
            mxl = []
            for l in lst:
                if l["peak"] >= mxx["lb"] and l["peak"] <= mxx["ub"]: mxl.append(l)
                elif mxx["peak"] >= l["lb"] and mxx["peak"] <= l["ub"]: mxl.append(l)
                elif range_comp(range(l["lb"], l["ub"]+1), range(mxx["lb"], mxx["ub"]+1)): mxl.append(l)
            return mxl
        def nested_cluster_find(bm, mx, j, case=-1):
            if bm < self.lbeam and bm > self.ubeam: return
            else:
                if (case == -1 and bm >= self.lbeam) or (case == 1 and bm <= self.ubeam):
                    mxl = find_adjucent(self.boundaries[bm], mx)
                    for m in mxl:
                        if m in self.gc:
                            del self.gc[self.gc.index(m)]
                            self.clusters[j].append(m)
                            nested_cluster_find(m["bmnum"] + case, m, j, case)
                            nested_cluster_find(m["bmnum"] + (-1*case), m, j, (-1*case))
                return
        self.clusters = {}
        j = 0
        while len(self.gc) > 0:
            self.clusters[j] = []
            mx = max(self.gc, key=lambda x:x["value"])
            self.clusters[j].append(mx)
            if mx in self.gc: del self.gc[self.gc.index(mx)]
            nested_cluster_find(mx["bmnum"] - 1, mx, j, case=-1)
            nested_cluster_find(mx["bmnum"] + 1, mx, j, case=1)
            j += 1
        return
    
class TimeFilter(object):
    """ Class to time filter middle latitude radars """
    
    def __init__(self, df, beam=7, tw=15, eps=2, min_samples=10, _dict_={}):
        logger.info("Time Cluster by Beam")
        self.df = df[df.bmnum==beam]
        self.tw = tw
        self.eps = eps
        self.min_samples = min_samples
        self.boundaries = {}
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        return
    
    def run_codes(self):
        start = self.df.time.tolist()[0]
        end = start + dt.timedelta(minutes=self.tw)
        k, j = 0, 0
        labels = []
        time_index = []
        while start <= self.df.time.tolist()[-1]:
            u = self.df[(self.df.time>=start) & (self.df.time<=end)]
            ds = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(u[["slist"]].values)
            labs = ds.labels_
            labs[labs>=0] = labs[labs>=0] + k
            labels.extend(labs.tolist())
            start = end
            end = start + dt.timedelta(minutes=self.tw)
            k += len(set(labs[labs>=0]))
            time_index.extend([j]*len(labs))
            j += 1
        self.df["gate_labels"] = labels
        self.df["labels"] = labels#[-1]*len(self.df)
        self.df["time_index"] = time_index
        
        K = len(self.df)
        for ti in np.unique(time_index):
            u = self.df[self.df.time_index==ti]
            self.boundaries[ti] = []
            for ix in np.unique(u.gate_labels):
                du = u[u.gate_labels==ix]
                if ix >= 0 and len(du)/len(u) > 0.5:
                    self.boundaries[ti].append({"peak": du.slist.mean(), "ub": du.slist.max(), "lb": du.slist.min(),
                                                "value":len(du)/len(u), "time_index": ti, "gc": ix})
            logger.info(f"{ti}, {np.unique(u.gate_labels)}, {len(self.boundaries[ti])}")
        
        self.ltime, self.utime = np.min(time_index), np.max(time_index)
        self.gc = []
        self.clust_idf = {}
        for ti in np.unique(time_index):
            clust = self.boundaries[ti]
            for cl in clust:
                self.gc.append(cl)
        self.sma_bgspace()
        self.probabilistic_cluster_identification()
        return
    
    def sma_bgspace(self):
        """
        Simple minded algorithm in B.G space
        """
        def range_comp(x, y, pcnt=0.7):
            _cx = False
            insc = set(x).intersection(y)
            if len(x) < len(y) and len(insc) >= len(x)*pcnt: _cx = True
            if len(x) > len(y) and len(insc) >= len(y)*pcnt: _cx = True
            return _cx
        def find_adjucent(lst, mxx):
            mxl = []
            for l in lst:
                if l["peak"] >= mxx["lb"] and l["peak"] <= mxx["ub"]: mxl.append(l)
                elif mxx["peak"] >= l["lb"] and mxx["peak"] <= l["ub"]: mxl.append(l)
                elif range_comp(range(l["lb"], l["ub"]+1), range(mxx["lb"], mxx["ub"]+1)): mxl.append(l)
            return mxl
        def nested_cluster_find(ti, mx, j, case=-1):
            if ti < self.ltime and ti > self.utime: return
            else:
                if (case == -1 and ti >= self.ltime) or (case == 1 and ti <= self.utime):
                    mxl = find_adjucent(self.boundaries[ti], mx)
                    for m in mxl:
                        if m in self.gc:
                            del self.gc[self.gc.index(m)]
                            self.clusters[j].append(m)
                            nested_cluster_find(m["time_index"] + case, m, j, case)
                            nested_cluster_find(m["time_index"] + (-1*case), m, j, (-1*case))
                return
        self.clusters = {}
        j = 0
        while len(self.gc) > 0:
            self.clusters[j] = []
            mx = max(self.gc, key=lambda x:x["value"])
            self.clusters[j].append(mx)
            if mx in self.gc: del self.gc[self.gc.index(mx)]
            nested_cluster_find(mx["time_index"] - 1, mx, j, case=-1)
            nested_cluster_find(mx["time_index"] + 1, mx, j, case=1)
            j += 1
        
        for c in self.clusters.keys():
            clust = self.clusters[c]
            for cl in clust:
                logger.info(f"{c}, {cl}")
                self.df["labels"] = np.where((self.df.slist<=cl["ub"]) & (self.df.slist>=cl["lb"]) & 
                                             (self.df.time_index==cl["time_index"]) & (self.df.gate_labels==cl["gc"])
                                             , c, self.df["labels"])
        return
    
    def probabilistic_cluster_identification(self):
        """ Idenitify the cluster based on Idenitification """
        self.df["cluster_id"] = [np.nan] * len(self.df)
        prob_dctor = ScatterTypeDetection()
        for c in np.unique(self.df["labels"]):
            if c >= 0:
                du = df[df["labels"]==c]
                prob_dctor.copy(du)
                du, prob_clusters = prob_dctor.run(thresh=[self.lth, self.uth], pth=self.pth)
                auc = prob_clusters[self.gflg_type][c]["auc"]
                if prob_clusters[self.gflg_type][c]["type"] == "IS": auc = 1-auc
                if prob_clusters[self.gflg_type][c]["type"] == "US": self.clust_idf[c] = "us"
                else: self.clust_idf[c] = "%.1f"%(auc*100) + "%" + prob_clusters[self.gflg_type][c]["type"].lower()
        return
    
class ScatterTypeDetection(object):
    """ Detecting scatter type """

    def __init__(self):
        """ kind: 0- individual, 2- KDE by grouping """
        return
    
    def copy(self, df):
        self.df = df.fillna(0)
        return

    def run(self, cases=[0,1,2], thresh=[1./3.,2./3.], pth=0.5):
        self.pth = pth
        self.thresh = thresh
        self.clusters = {}
        for cs in cases:
            self.case = cs
            self.clusters[self.case] = {}
            for kind in range(2):
                if kind == 0: 
                    self.indp()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                if kind == 1: 
                    self.kde()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                    self.df["proba_%d"%(cs)] = self.proba
        return self.df.copy(), self.clusters

    def kde_beam(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        self.proba = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            self.clusters[self.case][bm] = {}
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams) 
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                    if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                    if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                    ## KDE part
                    f[f==0.] = np.random.uniform(0.001, 0.01, size=len(f[f==0.]))
                    f[f==1.] = np.random.uniform(0.95, 0.99, size=len(f[f==1.]))
                    try:
                        a, b, loc, scale = beta.fit(f, floc=0., fscale=1.)
                        auc = 1 - beta.cdf(self.pth, a, b, loc=0., scale=1.)
                        if np.isnan(auc): auc = np.mean(f)
                        #print(" KDE Estimating AUC: %.2f"%auc, c)
                    except:
                        auc = np.mean(f)
                        #print(" Estimating AUC: %.2f"%auc, c)
                    if auc < self.thresh[0]: gflg, ty = 0., "IS"
                    elif auc >= self.thresh[1]: gflg, ty = 1., "GS"
                    else: gflg, ty = -1, "US"
                    self.gs_flg[clust_mask] = gflg
                    self.proba[clust_mask] = f
                    self.clusters[self.case][bm][c] = {"auc": auc, "type": ty}
        return
    
    def kde(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        self.proba = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1: self.gs_flg[clust_mask] = -1
            else:
                v, w = vel[clust_mask], wid[clust_mask]
                if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                ## KDE part
                f[f==0.] = np.random.uniform(0.001, 0.01, size=len(f[f==0.]))
                f[f==1.] = np.random.uniform(0.95, 0.99, size=len(f[f==1.]))
                try:
                    a, b, loc, scale = beta.fit(f, floc=0., fscale=1.)
                    auc = 1 - beta.cdf(self.pth, a, b, loc=0., scale=1.)
                    if np.isnan(auc): auc = np.mean(f)
                    #print(" KDE Estimating AUC: %.2f"%auc, c)
                except:
                    auc = np.mean(f)
                    #print(" Estimating AUC: %.2f"%auc, c)
                if auc < self.thresh[0]: gflg, ty = 0., "IS"
                elif auc >= self.thresh[1]: gflg, ty = 1., "GS"
                else: gflg, ty = -1, "US"
                self.gs_flg[clust_mask] = gflg
                self.proba[clust_mask] = f
                self.clusters[self.case][c] = {"auc": auc, "type": ty}
        return

    def indp(self):
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams)
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    gflg = np.zeros(len(v))
                    if self.case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
                    if self.case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
                    if self.case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
                    self.gs_flg[clust_mask] = gflg
        return