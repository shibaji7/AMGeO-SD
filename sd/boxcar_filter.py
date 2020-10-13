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

    def __init__(self, rad, scans, thresh=0.2):
        """
        initialize variables
        
        rad: Radar code
        scans: List of scans
        thresh: Threshold of the weight matrix
        """
        self.rad = rad
        self.scans = scans
        self.thresh = thresh
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
    
    def doFilter(self, io, window=7, order=1, beams=[]):
        """
        Do filter for sub-auroral scatter
        """
        from sklearn.cluster import DBSCAN
        df = pd.DataFrame()
        sb = [np.nan, np.nan]
        for i, fsc in enumerate(self.scans):
            dx = io.convert_to_pandas(fsc.beams, v_params=["p_l", "v", "w_l", "slist"])
            df = df.append(dx)
        if beams == None or len(beams) == 0: 
            bm = "all"
            sb[0], sb[1] = len(self.scans), np.max(df.bmnum) - np.min(df.bmnum) +1
            rng, eco = np.array(df.groupby(["slist"]).count().reset_index()["slist"]),\
                    np.array(df.groupby(["slist"]).count().reset_index()["p_l"])
            glims, labels = {}, []
            eco = utils.smooth(eco, window)
            troughs = signal.argrelmin(eco, order=order)[0]
            df["labels"] = [np.nan] * len(df)
            names = {}
            troughs = np.insert(troughs, 0, 0)
            troughs = np.append(troughs, [np.max(df.slist)])
            troughs[-2] = troughs[-2] + 1
            for r in range(len(troughs)-1):
                glims[r] = [troughs[r], troughs[r+1]]
                df["labels"] = np.where((df["slist"]<=troughs[r+1]) & (df["slist"]>=troughs[r]), r, df["labels"])
                names[r] = "C" + str(r)
            print(" Individual clster detected: ", set(df.labels))
            to_midlatitude_gate_summary(self.rad, df, glims, names, (rng,eco),
                    "data/outputs/{r}/gate_{bm}_summary.png".format(r=self.rad, bm=bm), sb)
        else:
            for bm in beams:
                sb[0], sb[1] = len(self.scans), 1
                print(" Beam Analysis: ", bm)
                du = pd.DataFrame()
                du = df[df.bmnum==bm]
                rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                        np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
                eco = utils.smooth(eco, window)
                glims, labels = {}, []
                ds = DBSCAN(eps=2, min_samples=10).fit(du[["slist"]].values)
                du["labels"] = ds.labels_
                names = {}
                for j, r in enumerate(set(ds.labels_)):
                    x = du[du.labels==r]
                    glims[r] = [np.min(x.slist), np.max(x.slist)]
                    names[r] = "C"+str(j)
                print(" Individual clster detected: ", set(du.labels))
                to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco),
                        "data/outputs/{r}/gate_{bm}_summary.png".format(r=self.rad, bm=bm), sb)
        return





if __name__ == "__main__":
    create_gaussian_weights(mu=[0,0,0], sigma=[3,3,3], base_w=7)
    filter = Filter()
