#!/usr/bin/env python

"""boxcar_filter.py: module is dedicated to run all analysis and filtering."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import copy
import numpy as np
from scipy import stats as st
from scipy import signal
from scipy.stats import beta
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN

from get_sd_data import Gate, Beam, Scan
from plot_lib import *
import utils


def create_gaussian_weights(mu, sigma, _kernel=3, base_w=5):
    """
    Method used to create gaussian weights
    mu: n 1D list of all mean values
    sigma: n 1D list of sigma matrix
    """
    _k = (_kernel-1)/2
    _kNd = np.zeros((3,3,3))
    for i in range(_kernel):
        for j in range(_kernel):
            for k in range(_kernel):
                _kNd[i,j,k] = np.exp(-((float(i-_k)**2/(2*sigma[0]**2)) + (float(j-_k)**2/(2*sigma[1]**2)) + (float(k-_k)**2/(2*sigma[2]**2))))
    _kNd = np.floor(_kNd * base_w).astype(int)
    return _kNd

class Filter(object):
    """Class to filter data - Boxcar median filter."""

    def __init__(self, thresh=.7, w=None, pbnd=[1./5., 4./5.], pth=0.25, kde_plot_point=(6,20), verbose=False):
        """
        initialize variables

        thresh: Threshold of the weight matrix
        w: Weight matrix
        pbnd: Lower and upper bounds of IS / GS probability
        pth: Probability of the threshold
        kde_plot_point: Plot the distribution for a point with (beam, gate)
        """
        self.thresh = thresh
        if w is None: w = np.array([[[1,2,1],[2,3,2],[1,2,1]],
                                    [[2,3,2],[3,5,3],[2,3,2]],
                                    [[1,2,1],[2,3,2],[1,2,1]]])
        self.w = w
        self.pbnd = pbnd
        self.pth = pth
        self.kde_plot_point = kde_plot_point
        self.verbose = verbose
        return

    def _discard_repeting_beams(self, scan, ch=True):
        """
        Discard all more than one repeting beams
        scan: SuperDARN scan
        """
        oscan = Scan(scan.stime, scan.etime, scan.stype)
        if ch: scan.beams = sorted(scan.beams, key=lambda bm: (bm.bmnum))
        else: scan.beams = sorted(scan.beams, key=lambda bm: (bm.bmnum, bm.time))
        bmnums = []
        for bm in scan.beams:
            if bm.bmnum not in bmnums:
                if hasattr(bm, "slist") and len(getattr(bm, "slist")) > 0:
                    oscan.beams.append(bm)
                    bmnums.append(bm.bmnum)
        oscan.update_time()
        oscan.beams = sorted(oscan.beams, key=lambda bm: bm.bmnum)
        return oscan


    def doFilter(self, i_scans, comb=False, plot_kde=False, gflg_type=-1):
        """
        Median filter based on the weight given by matrix (3X3X3) w, and threshold based on thresh
    
        i_scans: 3 consecutive radar scans
        comb: combine beams
        plot_kde: plot KDE estimate for the range cell if true
        gflg_type: Type of gflag used in this study [-1, 0, 1, 2] -1 is default other numbers are listed in
                    Evan's presentation.
        """
        if comb: 
            scans = []
            for s in i_scans:
                scans.append(self._discard_repeting_beams(s))
        else: scans = i_scans

        self.scans = scans
        w, pbnd, pth, kde_plot_point = self.w, self.pbnd, self.pth, self.kde_plot_point
        oscan = Scan(scans[1].stime, scans[1].etime, scans[1].stype)
        if w is None: w = np.array([[[1,2,1],[2,3,2],[1,2,1]],
                                [[2,3,2],[3,5,3],[2,3,2]],
                                [[1,2,1],[2,3,2],[1,2,1]]])
        l_bmnum, r_bmnum = scans[1].beams[0].bmnum, scans[1].beams[-1].bmnum
    
        for b in scans[1].beams:
            bmnum = b.bmnum
            beam = Beam()
            beam.copy(b)
    
            for key in beam.__dict__.keys():
                if type(getattr(beam, key)) == np.ndarray: setattr(beam, key, [])

            setattr(beam, "v_mad", [])
            setattr(beam, "gflg_conv", [])
            setattr(beam, "gflg_kde", [])
    
            for r in range(0,b.nrang):
                box = [[[None for j in range(3)] for k in range(3)] for n in range(3)]
    
                for j in range(0,3):# iterate through time
                    for k in range(-1,2):# iterate through beam
                        for n in range(-1,2):# iterate through gate
                            # get the scan we are working on
                            s = scans[j]
                            if s == None: continue
                            # get the beam we are working on
                            if s == None: continue
                            # get the beam we are working on
                            tbm = None
                            for bm in s.beams:
                                if bm.bmnum == bmnum + k: tbm = bm
                            if tbm == None: continue
                            # check if target gate number is in the beam
                            if r+n in tbm.slist:
                                ind = np.array(tbm.slist).tolist().index(r + n)
                                box[j][k+1][n+1] = Gate(tbm, ind, gflg_type=gflg_type)
                            else: box[j][k+1][n+1] = 0
                pts = 0.0
                tot = 0.0
                v,w_l,p_l,gfx = list(), list(), list(), list()
        
                for j in range(0,3):# iterate through time
                    for k in range(0,3):# iterate through beam
                        for n in range(0,3):# iterate through gate
                            bx = box[j][k][n]
                            if bx == None: continue
                            wt = w[j][k][n]
                            tot += wt
                            if bx != 0:
                                pts += wt
                                for m in range(0, wt):
                                    v.append(bx.v)
                                    w_l.append(bx.w_l)
                                    p_l.append(bx.p_l)
                                    gfx.append(bx.gflg)
                if pts / tot >= self.thresh:# check if we meet the threshold
                    beam.slist.append(r)
                    beam.v.append(np.median(v))
                    beam.w_l.append(np.median(w_l))
                    beam.p_l.append(np.median(p_l))
                    beam.v_mad.append(np.median(np.abs(np.array(v)-np.median(v))))
                    
                    # Re-evaluate the groundscatter flag using old method
                    gflg = 0 if np.median(w_l) > -3.0 * np.median(v) + 90.0 else 1
                    beam.gflg.append(gflg)
                    
                    # Re-evaluate the groundscatter flag using weight function
                    #if bmnum == 12: print(bmnum,"-",r,":(0,1)->",len(gfx)-np.count_nonzero(gfx),np.count_nonzero(gfx),np.nansum(gfx) / tot)
                    gflg = np.nansum(gfx) / tot
                    if np.nansum(gfx) / tot <= pbnd[0]: gflg=0.
                    elif np.nansum(gfx) / tot >= pbnd[1]: gflg=1.
                    else: gflg=2.
                    beam.gflg_conv.append(gflg)
    
                    # KDE estimation using scipy Beta(a, b, loc=0, scale=1)
                    _gfx = np.array(gfx).astype(float)
                    _gfx[_gfx==0], _gfx[_gfx==1]= np.random.uniform(low=0.01, high=0.05, size=len(_gfx[_gfx==0])), np.random.uniform(low=0.95, high=0.99, size=len(_gfx[_gfx==1]))
                    a, b, loc, scale = beta.fit(_gfx, floc=0., fscale=1.)
                    if kde_plot_point is not None:
                        if (kde_plot_point[0] == bmnum) and (kde_plot_point[1] == r):
                            #if plot_kde: plot_kde_distribution(a, b, loc, scale, pth, pbnd)
                            pass
                    gflg = 1 - beta.cdf(pth, a, b, loc=0., scale=1.)
                    if gflg <= pbnd[0]: gflg=0.
                    elif gflg >= pbnd[1]: gflg=1.
                    else: gflg=2.
                    beam.gflg_kde.append(gflg)
            oscan.beams.append(beam)
    
        oscan.update_time()
        oscan._estimat_skills(verbose=self.verbose)
        sorted(oscan.beams, key=lambda bm: bm.bmnum)
        return oscan

class BoxCarGate(object):
    """Class to filter data - Boxcar median filter 1D."""

    def __init__(self, thresh=.1, w=None, verbose=True):
        """
        initialize variables

        thresh: Threshold of the weight matrix
        w: Weight matrix
        """
        self.thresh = thresh
        if w is None: w = np.array([1,3,1])
        self.w = w
        self.verbose = verbose
        return

    def _discard_repeting_beams(self, scan, ch=True):
        """
        Discard all more than one repeting beams
        scan: SuperDARN scan
        """
        oscan = Scan(scan.stime, scan.etime, scan.stype)
        if ch: scan.beams = sorted(scan.beams, key=lambda bm: (bm.bmnum))
        else: scan.beams = sorted(scan.beams, key=lambda bm: (bm.bmnum, bm.time))
        bmnums = []
        for bm in scan.beams:
            if bm.bmnum not in bmnums:
                if hasattr(bm, "slist") and len(getattr(bm, "slist")) > 0:
                    oscan.beams.append(bm)
                    bmnums.append(bm.bmnum)
        oscan.update_time()
        oscan.beams = sorted(oscan.beams, key=lambda bm: bm.bmnum)
        return oscan


    def doFilter(self, i_scan, gflg_type=-1):
        """
        Median filter based on the weight given by matrix (3X3X3) w, and threshold based on thresh
    
        i_scan: 1 radar scan
        gflg_type: Type of gflag used in this study [-1, 0, 1, 2] -1 is default other numbers are listed in
                    Evan's presentation.
        """
        self.scan = i_scan
        w = self.w
        oscan = Scan(scan.stime, scan.etime, scan.stype)
    
        for b in self.scan.beams:
            bmnum = b.bmnum
            beam = Beam()
            beam.copy(b)
    
            for key in beam.__dict__.keys():
                if type(getattr(beam, key)) == np.ndarray: setattr(beam, key, [])

            for r in range(0,b.nrang):
                box = [None for n in range(3)]
                for n in range(-1,2):# iterate through gate
                    if r+n in b.slist:
                        ind = np.array(b.slist).tolist().index(r + n)
                        box[n+1] = Gate(b, ind, gflg_type=gflg_type)
                    else: box[n+1] = None
                pts = 0.0
                tot = 0.0
                v,w_l,p_l = list(), list(), list()
        
                for n in range(0,3):# iterate through gate
                    bx = box[n]
                    wt = w[n]
                    tot += wt
                    if bx is not None:
                        pts += wt
                        for m in range(0, wt):
                            v.append(bx.v)
                            w_l.append(bx.w_l)
                            p_l.append(bx.p_l)
                if pts / tot >= self.thresh:# check if we meet the threshold
                    beam.slist.append(r)
                    beam.v.append(np.median(v))
                    beam.w_l.append(np.median(w_l))
                    beam.p_l.append(np.median(p_l))
                    beam.v_mad.append(np.median(np.abs(np.array(v)-np.median(v))))
            oscan.beams.append(beam)
    
        oscan.update_time()
        sorted(oscan.beams, key=lambda bm: bm.bmnum)
        return oscan

class MiddleLatFilter(object):
    """ Class to filter middle latitude radars """

    def __init__(self, rad, scans, eps=2, min_samples=10):
        """
        initialize variables
        
        rad: Radar code
        scans: List of scans
        eps: Radius of DBSCAN
        min_samples: min samplese of DBSCAN
        """
        self.rad = rad
        self.scans = scans
        self.eps = eps
        self.min_samples = min_samples
        self.boundaries = {}
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
                "sound":sb[0], "echo":len(du), "v_mad": st.median_absolute_deviation(du.v), 
                "p_mad": st.median_absolute_deviation(du.p_l), "w_mad": st.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all":
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else:
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        print(" Beam Analysis: ", bm, len(du.groupby("time")))
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
        print(" Individual clster detected: ", set(du.labels))
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco),
                "data/outputs/{r}/gate_{bm}_summary.png".format(r=self.rad, bm=bm), sb)
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
        print(" Beam Analysis: ", bm, len(du.groupby("time")))
        sb = [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l),
                "sound":sb[0], "echo":len(du), "v_mad": st.median_absolute_deviation(du.v),
                "p_mad": st.median_absolute_deviation(du.p_l), "w_mad": st.median_absolute_deviation(du.w_l)}
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
        print(" Gate bounds: ", troughs, peaks)
        peaks = np.append(peaks, np.median(troughs[-2:]))
        for r in range(len(troughs)-1):
            glims[r] = [troughs[r], troughs[r+1]]
            du["labels"] = np.where((du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r]), r, du["labels"])
            names[r] = "C" + str(r)
            if r >= 0: self.boundaries[bm].append({"peak": peaks[r],
                "ub": troughs[r+1], "lb": troughs[r], "value": np.max(eco)*xpt, "bmnum": bm, 
                "echo": len(du[(du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r])]), "sound": sb[0]})

        du["labels"] =  np.where(np.isnan(du["labels"]), -1, du["labels"])
        print(" Individual clster detected: ", set(du.labels))
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco),
                "data/outputs/{r}/gate_{bm}_summary.png".format(r=self.rad, bm=bm), sb)
        return

    def doFilter(self, io, window=11, order=1, beams=range(4,24)):
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
        print(" Beam range - ", np.min(df.bmnum), np.max(df.bmnum))
        if beams == None or len(beams) == 0: 
            bm = "all"
            self.boundaries[bm] = []
            self.filter_by_SMF(df, bm)
        else:
            for bm in beams:
                self.boundaries[bm] = []
                self.gen_summary[bm] = {}
                if bm==15: self.filter_by_SMF(df, bm)
                else: self.filter_by_dbscan(df, bm)
        title = "Date: %s [%s-%s] UT | %s"%(df.time.tolist()[0].strftime("%Y-%m-%d"),
                df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M"), self.rad.upper())
        fname = "data/outputs/%s/gate_boundary.png"%(self.rad)
        self.gc = []
        for x in self.boundaries.keys():
            for j, m in enumerate(self.boundaries[x]):
                self.gc.append(m)
        self.ubeam, self.lbeam = np.max(df.bmnum), np.min(df.bmnum)
        self.sma_bgspace()
        self.cluster_identification(df)
        beam_gate_boundary_plots(self.boundaries, self.clusters, None, 
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname)
        beam_gate_boundary_plots(self.boundaries, self.clusters, self.clust_idf,
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname.replace("gate_boundary", "gate_boundary_id"))
        general_stats(self.gen_summary, fname="data/outputs/%s/general_stats.png"%(self.rad))
        for c in self.clusters.keys():
            cluster = self.clusters[c]
            fname = "data/outputs/%s/clust_%02d_stats.png"%(self.rad, c)
            cluster_stats(df, cluster, fname, title+" | Cluster# %02d"%c)
            individal_cluster_stats(self.clusters[c], df, "data/outputs/%s/ind_clust_%02d_stats.png"%(self.rad, c), 
                    title+" | Cluster# %02d"%c+"\n"+"Cluster ID: _%s_"%self.clust_idf[c].upper())
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




if __name__ == "__main__":
    create_gaussian_weights(mu=[0,0,0], sigma=[3,3,3], base_w=7)
    filter = Filter()
