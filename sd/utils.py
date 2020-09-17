#!/usr/bin/env python

"""utils.py: module is dedicated to utility functions for the project."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import netCDF4
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

def get_gridded_parameters(q, xparam="beam", yparam="slist", zparam="v", r=0):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    plotParamDF[xparam] = np.round(plotParamDF[xparam].tolist(), r)
    plotParamDF[yparam] = np.round(plotParamDF[yparam].tolist(), r)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

def parse_parameter(scan, p="v", velo=5000.):
    """
    Parse the data from radar scan.
    """
    param,times,beams,slist = [], [], [], []
    for b in scan.beams:
        if b.slist is not None:
            c = len(b.slist)
            times.extend([b.time] * c)
            beams.extend([b.bmnum] * c)
            if p == "v" and b.v is not None: param.extend(b.v)
            if p == "v_mad" and b.v_mad is not None: param.extend(b.v_mad)
            if p == "p_l" and b.p_l is not None: param.extend(b.p_l)
            if p == "v_e" and b.v_e is not None: param.extend(b.v_e)
            if p == "w_l" and b.w_l is not None: param.extend(b.w_l)
            if p == "gflg" and b.gflg is not None: param.extend(b.gflg)
            if p == "gflg_conv" and b.gflg_conv is not None: param.extend(b.gflg_conv)
            if p == "gflg_kde" and b.gflg_kde is not None: param.extend(b.gflg_kde)
            slist.extend(b.slist)
    if len(times) > len(param): 
        import itertools
        l, pv = len(times)-len(param), [param[-1]]
        param.extend(list(itertools.chain.from_iterable(itertools.repeat(x, l) for x in pv)))
    _d = {"times":times, p:param, "beam":beams, "slist":slist}
    _d = pd.DataFrame.from_records(_d)
    if p == "v" and velo is not None: _d = _d[np.abs(_d.v) < velo]
    return _d


class Skills(object):
    """
    Internal Validation class that computes model skills.
    
    Internal validation methods make it possible to establish
    the quality of the clustering structure without having access
    to external information (i.e. they are based on the information
    provided by data used as input to the clustering algorithm).
    """

    def __init__(self, X, labels, name, verbose=False):
        """
        Initialize the parameters.
        X: List of n_features-dimensional data points. Each row corresponds to a single data point.
        lables: Predicted labels for each sample.
        """
        self.X = X
        self.labels = labels
        self.name = name
        self.clusters = set(self.labels)
        self.n_clusters = len(self.clusters)
        self.n_samples, _ = X.shape
        self.verbose = verbose
        if self.n_clusters > 1: self._compute()
        else: 
            if verbose: print("\n No skills are estimated as only one class is detected")
        return

    def _compute(self):
        """
        Compute all different skills scores
        - Davies Bouldin Score (dbscore)
        - Calinski Harabasz Score (chscore)
        - Silhouette Score (siscore)
        - Ball Hall Score (bhscore)
        - Hartigan Score (hscore)
        - Xu Score (xuscore)
        """
        self.dbscore = davies_bouldin_score(self.X, self.labels)
        self.chscore = calinski_harabasz_score(self.X, self.labels)
        self.siscore = silhouette_score(self.X, self.labels)
        self._errors()
        self.bhscore = self._ball_hall_score()
        self.hscore = self._hartigan_score()
        self.xuscore = self._xu_score()
        if self.verbose:
            print("\n Estimated Skills for: ", self.name)
            print(" Davies Bouldin Score- ",self.dbscore)
            print(" Calinski Harabasz Score- ",self.chscore)
            print(" Silhouette Score- ",self.siscore)
            print(" Ball-Hall Score- ",self.bhscore)
            print(" Hartigan Score- ",self.hscore)
            print(" Xu Score- ",self.xuscore)
            print(" Skill Estimation Done.")
        return

    def _errors(self):
        """
        Estimate SSE and SSB of the model.
        """
        sse, ssb = 0., 0.
        mean = np.mean(self.X, axis=0)
        for k in self.clusters:
            _x = self.X[self.labels == k]
            mean_k = np.mean(_x, axis=0)
            ssb += len(_x) * np.sum((mean_k - mean) ** 2)
            sse += np.sum((_x - mean_k) ** 2)
        self.sse, self.ssb = sse, ssb
        return

    def _ball_hall_score(self):
        """
        The Ball-Hall index is a dispersion measure based on the quadratic
        distances of the cluster points with respect to their centroid.
        """
        n_clusters = len(set(self.labels))
        return self.sse / n_clusters
    
    def _hartigan_score(self):
        """
        The Hartigan index is based on the logarithmic relationship between
        the sum of squares within the cluster and the sum of squares between clusters.
        """
        return np.log(self.ssb/self.sse)
    
    def _xu_score(self):
       """
       The Xu coefficient takes into account the dimensionality D of the data,
       the number N of data examples, and the sum of squared errors SSEM form M clusters.
       """
       n_clusters = len(set(self.labels))
       return np.log(n_clusters) + self.X.shape[1] * np.log2(np.sqrt(self.sse/(self.X.shape[1]*self.X.shape[0]**2)))

def get_radar(stn):
    _o = pd.read_csv("radar.csv")
    _o = _o[_o.rad==stn]
    lat, lon, rad_type = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180, _o["rad_type"].tolist()[0]
    return lat, lon, rad_type

def get_geolocate_range_cells(rad):
    """
    Fetch geolocate range cells
    rad: Radar code
    """
    fname = "data/sim/{rad}.geolocate.data.nc".format(rad=rad)
    gzfname = "data/sim/{rad}.geolocate.data.nc.gz".format(rad=rad)
    os.system("gzip -d " + gzfname)
    rootgrp = netCDF4.Dataset(fname)
    os.system("gzip " + fname)
    return rootgrp

def save_cmd(args, f):
    """ Save command to file """
    cmd = "python " + " ".join(args)
    with open(f + "script.cmd", "w") as o: o.write(cmd)
    return
