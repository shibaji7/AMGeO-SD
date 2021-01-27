#!/usr/bin/env python

"""utils.py: module is dedicated to utility functions for the project."""

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
pd.set_option("mode.chained_assignment", None)
import configparser
import datetime as dt
from loguru import logger
import json

def smooth(x, window_len=51, window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def get_gridded_parameters(q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=True):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    else:
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
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
    if "gsflg_" in p:
        gsflg_ind = int(p.split("_")[-1])
        p = "gsflg"
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
            if p == "gsflg" and b.gsflg is not None and b.gsflg[gsflg_ind] is not None: param.extend(b.gsflg[gsflg_ind])
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

def get_radar(stn):
    _o = pd.read_csv("config/radar.csv")
    _o = _o[_o.rad==stn]
    lat, lon, rad_type = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180, _o["rad_type"].tolist()[0]
    return lat, lon, rad_type

def save_cmd(argv, args, out_dir):
    """ Save command to file """
    def default(o):
        if isinstance(o, (dt.date, dt.datetime)):
            return o.isoformat()
    logger.info("Save command line arguments")
    cmd = "python " + " ".join(argv)
    with open(out_dir + "script.cmd", "w") as o: o.write(cmd)
    with open(out_dir + "params.json", "w") as o: o.write(json.dumps(args, sort_keys=True, indent=4, default=default))
    return

def build_base_folder_structure(rad, sim_id):
    logger.info("Create base folder stucture to save outfiles")
    folder = get_config("BASE_DIR") + "{sim_id}/{rad}/".format(sim_id=sim_id, rad=rad)
    if not os.path.exists(folder): os.system("mkdir -p " + folder)
    return folder

def get_config(key, section="amgeo"):
    config = configparser.ConfigParser()
    config.read("config/conf.ini")
    val = config[section][key]
    return val