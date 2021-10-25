#!/usr/bin/env python

"""fitacf_utils.py: module is dedicated to produce fitacf++ plots - anciliary plots."""

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
import datetime as dt
import argparse
from dateutil import parser as prs
from loguru import logger
import glob
from netCDF4 import Dataset, num2date
import pandas as pd
import matplotlib.dates as mdates

import sys
sys.path.append("sd/")
sys.path.append("tools/")
from fan_rti import RangeTimeIntervalPlot as RTI

def to_frame(nc):
    """
    Convert netCDF4 dataset to pandas frames
    """
    s_params = ["time", "bmnum", "noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul", "scnum"]
    v_params = ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad", "cluster_tag", "ribiero_gflg"]
    _dict_ = {k: [] for k in s_params + v_params}
    tparam = {"units":nc.variables["time"].units, "calendar":nc.variables["time"].calendar,
                "only_use_cftime_datetimes":False}
    for i in range(nc.variables["slist"].shape[0]):
        sl = nc.variables["slist"][:][i,:]
        idx = np.isnan(sl)
        L = len(sl[~idx])
        for k in s_params:
            _dict_[k].extend(L*[nc.variables[k][i]])
        for k in v_params:
            _dict_[k].extend(nc.variables[k][i,~idx])
    o = pd.DataFrame.from_dict(_dict_)
    time = o.time.apply(lambda x: num2date(x, tparam["units"], tparam["calendar"],
                                           only_use_cftime_datetimes=tparam["only_use_cftime_datetimes"])).tolist()
    time = np.array([x._to_real_datetime() for x in time]).astype("datetime64[ns]")
    time = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in time]
    o["dates"] = time
    o["mdates"] = o.dates.apply(lambda x: mdates.date2num(x)).tolist()
    o = o.sort_values(by=["dates"])
    return o 

def plot_rti(args):
    logger.info(f"Run RTI data-plot ...")
    ncfiles, frames = [], []
    tmpltfiles = "data/outputs/%s/%s/*.nc.gz"
    for sid in args.sim_id_list:
        files = glob.glob(tmpltfiles%(sid, args.rad))
        ncfiles.extend(files)
    for ncf in ncfiles:
        os.system("gzip -d "+ncf)
        nc = Dataset(ncf.replace(".gz", ""))
        os.system("gzip "+ncf.replace(".gz", ""))
        o = to_frame(nc)
        frames.append(o)
    if len(frames)>0: 
        title = "%s(%02d), %s"%(args.rad.upper(), args.bmnum, args.start.strftime("%Y-%m-%d"))
        frame = pd.concat(frames)
        rti = RTI(100, unique_times=np.unique(frame.mdates), fig_title="")
        rti.addParamPlot(frame, args.bmnum, title, xlabel="")
        rti.addCluster(frame, args.bmnum, "")
        rti.addGSIS(frame, args.bmnum, "", xlabel="Time, [UT]")
        for f in ncfiles:
            rti.save(f.replace(".nc.gz", ".png"))
    return

def plot_fov(args):
    logger.info(f"Run FoV data-plot ...")
    return

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="bks", help="SuperDARN radar code (default bks)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,16), help="Start date (default 2015-03-17)", 
            type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,17,17), help="End date (default 2015-03-17T12)", 
            type=prs.parse)
    parser.add_argument("-o", "--operation", default="rti", help="Plot RTI plots")
    parser.add_argument("-b", "--bmnum", default=15, help="Beam number")
    parser.add_argument("-sids", "--sim_ids", default="200,201", help="Simulation IDs")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    args = parser.parse_args()
    args.sim_id_list = args.sim_ids.split(",")
    logger.info(f"Simulation run using fitacf_utils.__main__")
    _dic_ = {}
    if args.verbose:
        logger.info("Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
            _dic_[k] = vars(args)[k]
    if args.operation in ["rti", "RTI"]: plot_rti(args)
    if args.operation in ["fov", "FOV"]: plot_fov(args)
    pass