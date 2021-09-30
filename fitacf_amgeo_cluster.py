#!/usr/bin/env python

"""fitacf_amgeo_cluster.py: module is dedicated to produce clustering data files."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import numpy as np
np.random.seed(0)
import sys
sys.path.append("sd/")
sys.path.append("tools/")
import datetime as dt
import argparse
from dateutil import parser as prs
from loguru import logger
import json

import utils
from fit_records import fetch_print_fit_rec
from get_fit_data import FetchData
from clustering_algo import BeamGateTimeFilter

def run_fitacf_amgeo_clustering(rad, start, end, gflg_cast, _dict_):
    """
    Method is dedicated to run fitacf++ simulation
    
    Input parameters
    ----------------
    rad <str>: Radar code
    start <datetime>: Start datetime object
    end <datetime>: End datetime object
    dofilter <bool>: If do median filter or not
    save <bool>: If save data or not
    gflg_type <int>: Flag type [-1, 0, 1, 2]
    gflg_cast <str>: Use to estimate IS/GS cluster type ["gflg_conv", "gflg_kde"]
    clear <bool>: If save clear outpur directory
    verbose <bool>: If save write on console
    thresh <float>: Median filter threshold (0,1)
    pth <float>: Probability of KDE function (0,1)
    lth <float>: Lower probability threshold for IS (0,1)
    uth <float>: Upper probability threshold for GS (0,1)
    out_dir <str>: Output directory
    ftype <str>: Fitacf type
    
    Return parameters
    -----------------
    _dict_ {
            folder_location <list(str)>: folder location to that generates 
           }
    scan_info {
                scan_details_parameters 
              }
    """
    gflg_cast_types = ["gflg_conv", "gflg_kde"]
    logger.info(f"run fitacf amgeo simulation {rad}")
    if gflg_cast not in gflg_cast_types: 
        logger.error(f"gflg_cast has to be in {gflg_cast_types}")
        raise Exception(f"gflg_cast has to be in {gflg_cast_types}")
    
    bgtf = BeamGateTimeFilter(rad, _dict_)
    scan_info = bgtf.run_bgc_algo().run_time_track_algo()
    bgtf.save_data()
    bgtf.repackage_to_netcdf()
    o = bgtf.out
    return o, scan_info

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="bks", help="SuperDARN radar code (default bks)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,17), help="Start date (default 2015-03-17)", 
            type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,17,12), help="End date (default 2015-03-17T12)", 
            type=prs.parse)
    parser.add_argument("-du", "--dur", default=30, help="Duration of the time window in min (30 mins)")
    parser.add_argument("-f", "--dofilter", action="store_false", help="Do filtering (default True)")
    parser.add_argument("-sv", "--save", action="store_false", help="Save data to file (defult True)")
    parser.add_argument("-gft", "--gflg_type", type=int, default=1, help="Ground scatter flag estimation type (default 2)")
    parser.add_argument("-gct", "--gflg_cast", default="gflg_conv", help="Ground scatter flag final plot type (gflg_conv)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-th", "--thresh", type=float, default=.7, help="Threshold value for filtering (default 0.7)")
    parser.add_argument("-pth", "--pth", type=float, default=.5, help="Probability threshold for KDE (default 0.5)")
    parser.add_argument("-lth", "--lth", type=float, default=1./3., help="Probability cut-off for IS")
    parser.add_argument("-uth", "--uth", type=float, default=2./3., help="Probability cut-off for GS")
    parser.add_argument("-sid", "--sim_id", default="L101", help="Simulation ID, need to store data into this folder (default L100)")
    parser.add_argument("-ftype", "--ftype", default="fitacf", help="FitACF file type (futacf, fitacf3)")
    parser.add_argument("-eps", "--eps", default=2, type=int, help="Epsilon for DBSCAN")
    parser.add_argument("-ms", "--min_samples", default=10, type=int, help="Min Samples for DBSCAN")
    parser.add_argument("-tw", "--tw", default=15, help="Duration of the time window for Time-Gate clustering in min (30 mins)")
    parser.add_argument("-cf", "--check_file", action="store_false", help="Check BGC file (default True)")
    args = parser.parse_args()
    logger.info(f"Simulation run using fitacf_amgeo_cluster.__main__")
    _dic_ = {}
    if args.verbose:
        logger.info("Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
            _dic_[k] = vars(args)[k]
    out_dir = utils.build_base_folder_structure(args.rad, args.sim_id)
    _o, scan_info = run_fitacf_amgeo_clustering(args.rad, args.start, args.end, args.gflg_cast, _dic_)
    logger.info(f"Simulation output from fitacf_amgeo_cluster.__main__\n{json.dumps(_o, sort_keys=True, indent=4)}")
    _dic_["out_dir"] = out_dir
    _dic_.update(scan_info)
    _dic_.update(_o)
    _dic_["gflg_type"] = -1
    _dic_["start"] = args.start
    _dic_["__func__"] = "fitacf_amgeo_cluster"
    utils.save_cmd(sys.argv, _dic_, out_dir)
    pass