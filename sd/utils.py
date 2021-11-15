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

import sys
sys.path.extend(["py/"])

import os
import numpy as np
import pandas as pd
pd.set_option("mode.chained_assignment", None)
import configparser
import datetime as dt
from loguru import logger
import json

from matplotlib.dates import date2num
import xarray

from get_fit_data import Scan

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

def ribiero_gs_flg(vel, time, vth):
    #logger.info(" Velocity threshold: %.1f"%vth)
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > vth)
    low = np.sum(np.abs(vel) <= vth)
    if low == 0: R = 1.0  # TODO hmm... this works right?
    else: R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        # Addition by us
        if R > 0.15: return False    # IS
        else: return True     # GS
        # Classic Ribiero 2011
        #return True  # GS
    elif L > 3:
        if R > 0.2: return False
        else: return True
    elif L > 2:
        if R > 0.33: return False
        else: return True
    elif L > 1:
        if R > 0.475: return False
        else: return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5: return False
        else: return True
    # Classic Ribiero 2011
    # else:
    #    return False

def _run_riberio_threshold(u, beam):
    df = u[u.bmnum==beam]
    clust_flag = np.array(df.labels); gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v)); t = np.hstack(df.time)
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1: gs_flg[clust_mask] = -1
        else: gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask], vth=25.)
    df["ribiero_gflg"] = gs_flg
    return df

def _run_riberio_threshold_on_rad(u, vth=25., flag="gflg_ribiero"):
    df = u.copy()
    clust_flag = np.array(df.cluster_tag)
    gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v))
    t = np.hstack(df.time.apply(lambda x: date2num(x)))
    clust_flag[np.array(df.slist) < 7] = -1
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1: gs_flg[clust_mask] = -1
        else: gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask], vth)
    df[flag] = gs_flg
    return df

def geolocate_radar_fov(rad, scan_dates, height=300, azm=False, mag=False):
    """ Geolocate each range cell """
    import pydarn
    import rad_fov
    import geoPack
    import aacgmv2
    hdw = pydarn.read_hdw_file(rad)
    fov = rad_fov.CalcFov(hdw=hdw, altitude=300.)
    lat_full = fov.latFull
    lon_full = fov.lonFull
    o = {"glat": lat_full, "glon": lon_full, 
         "beams": np.arange(hdw.beams+1), 
         "gates": np.arange(hdw.gates+1), 
         "scans": np.arange(len(scan_dates))}
    if azm:
        gazm_full = np.zeros((hdw.beams+1, hdw.gates+1))*np.nan
        for b in range(hdw.beams):
            for g in range(hdw.gates):
                do = geoPack.calcDistPnt(lat_full[b,g], lon_full[b,g], 300, 
                                         distLat=lat_full[b, g + 1], 
                                         distLon=lon_full[b, g + 1], distAlt=300)
                gazm_full[b,g] = do["az"]
        o["gazm"] = gazm_full
    # get aacgm_coords
    if mag: 
        mlat_full, mlon_full = np.zeros((len(scan_dates), hdw.beams+1, hdw.gates+1))*np.nan,\
            np.zeros((len(scan_dates), hdw.beams+1, hdw.gates+1))*np.nan
        for b in range(hdw.beams+1):
            for i, d in enumerate(scan_dates):
                mlat_full[i, b, :], mlon_full[i, b, :], _ = aacgmv2.convert_latlon_arr(lat_full[b], lon_full[b], height, d, "G2A")
        o["mlat"], o["mlon"] =  mlat_full, mlon_full        
        if azm:
            mazm_full = np.zeros((len(scan_dates), hdw.beams+1, hdw.gates+1))*np.nan
            for b in range(hdw.beams):
                for g in range(hdw.gates):
                    for i, _ in enumerate(scan_dates):
                        do = geoPack.calcDistPnt(mlat_full[b,g], mlon_full[b,g], height, distLat=mlat_full[b,g+1],
                                                 distLon=mlon_full[b,g+1], distAlt=height)
                        mazm_full[i,b,g] = do["az"]
            o["mazm"] = mazm_full
    o["scan_dates"] = [np.datetime64(d) for d in scan_dates]
    return o

def to_xarray(scans, fov, bmax, atrs={"description": ""}):
    """
    Convert data to Xarray
    """
    logger.info(f"Max beam depth {bmax}")
    
    def fetch_1D_param(key, tm=False):
        val = np.empty((fov["scans"].max()+1, bmax), dtype="datetime64[us]") if tm else\
                np.zeros((fov["scans"].max()+1, bmax))*np.nan
        for i, s in enumerate(scans):
            if tm: val[i,:] = [np.datetime64("nat")]*bmax
            for j, b in enumerate(s.beams):
                val[i, j] = np.datetime64(getattr(b, key)) if tm else getattr(b, key)
        return val
    
    def fetch_2D_param(key):
        val = np.zeros((fov["scans"].max()+1, bmax, fov["gates"].max()+1))*np.nan
        for i, s in enumerate(scans):
            for j, b in enumerate(s.beams):
                val[i, j, np.array(getattr(b, "slist")).astype(int)] = getattr(b, key)
        return val
    
    def set_param(va, key, shape, val, lname="", u="", dsc=""):
        atrs = dict(long_name = lname, units = u, description = dsc) if u else\
            dict(long_name = lname, description = dsc)
        va[key] = (shape, val, atrs)
        return va
    
    dv, cords = dict(), dict()
    # Setting coordinates
    cords = set_param(cords, "beams", "beams", fov["beams"], "Beam number", "1", "Radar beams")
    cords = set_param(cords, "sound", "sound", range(bmax), "Beam sound seq", "1", "Radar beam sound sequence")
    cords = set_param(cords, "gates", "gates", fov["gates"], "Gate number", "1", "Range gate")
    cords = set_param(cords, "scan_num", "scan_num", fov["scans"], "Scan number", "1", "Number of scan")
    # Setting 1D variables
    dv = set_param(dv, "gdlat", ["beams", "gates"], fov["glat"], "Latitude", "degree", "Geographic Latitudes at each range cell")
    dv = set_param(dv, "gdlon", ["beams", "gates"], fov["glon"], "Longitude", "degree", "Geographic Latitudes at each range cell")
    dv = set_param(dv, "gdazm", ["beams", "gates"], fov["gazm"], "Ray Azimuth", "degree", "Geographic Ray Azimuth at each range cell")
    #dv = set_param(dv, "scan_dates", ["scan_num"], fov["scan_dates"], "Date time", None, 
    #               "Scan start datetime")
    dv = set_param(dv, "bmnum", ["scan_num", "sound"], fetch_1D_param("bmnum").astype(int), "Sounding beam", 
                   "1", "Sounding beam number")
    dv = set_param(dv, "noise.sky", ["scan_num", "sound"], fetch_1D_param("noise.sky"), "Sky noise", 
                   "1", "Background sky noise")
    dv = set_param(dv, "tfreq", ["scan_num", "sound"], fetch_1D_param("tfreq"), "Frequency", 
                   "kHz", "Transmit signal frequency")
    dv = set_param(dv, "scan", ["scan_num", "sound"], fetch_1D_param("scan"), "Scan flag", 
                   "1", "Scan flag indicates start of a scan")
    dv = set_param(dv, "nrang", ["scan_num", "sound"], fetch_1D_param("nrang"), "Range number", 
                   "1", "Number of range gates in each beam sounding")
    dv = set_param(dv, "time", ["scan_num", "sound"], fetch_1D_param("time", True), "Date time", None,
                   "Beam sounding datetime")
    dv = set_param(dv, "mppul", ["scan_num", "sound"], fetch_1D_param("mppul"), "Pulse counts", 
                   "1", "Number of pulses")
    dv = set_param(dv, "intt.us", ["scan_num", "sound"], fetch_1D_param("intt.us"), "Int. Time micro-sec", 
                   "micro-seconds", "Integration time in micro-seconds")
    dv = set_param(dv, "intt.sc", ["scan_num", "sound"], fetch_1D_param("intt.sc"), "Int. Time sec", 
                   "seconds", "Integration time in seconds")
    # Setting 2D variables
    dv = set_param(dv, "v", ["scan_num", "sound", "gates"], fetch_2D_param("v"), "Velocity", 
                   "m/s", "LOS Doppler velocity")    
    dv = set_param(dv, "w_l", ["scan_num", "sound", "gates"], fetch_2D_param("w_l"), "Width", 
                   "m/s", "LOS Spectral width")
    dv = set_param(dv, "p_l", ["scan_num", "sound", "gates"], fetch_2D_param("p_l"), "Power", 
                   "dB", "Backscatter power")
    dv = set_param(dv, "p_l", ["scan_num", "sound", "gates"], fetch_2D_param("p_l"), "Power", 
                   "dB", "Backscatter power")
    dv = set_param(dv, "gflg", ["scan_num", "sound", "gates"], fetch_2D_param("gflg").astype(bool), "Ground scatter flag", 
                   "bool", "Ground scatter flag (determined using method described in fitacf method) (0/1)")
    dv = set_param(dv, "gflg_ribiero", ["scan_num", "sound", "gates"], fetch_2D_param("gflg_ribiero").astype(bool), 
                   "Modified ground scatter flag", "bool", 
                   "Ground scatter flag (determined using method described in clustering method) (0/1)")
    ds = xarray.Dataset(
        data_vars = dv,
        coords = cords,
        attrs = atrs,
    )
    return ds