#!/usr/bin/env python

"""fitacf_amgeo.py: module is dedicated to produce fitacf++ datafiles."""

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
from analysis import Simulate

def run_fitacf_amgeo_simulation(rad, start, end, gflg_cast, _dict_):
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
    scan_info = fetch_print_fit_rec(rad, start, start + dt.timedelta(minutes=5), file_type=_dict_["ftype"])
    sim = Simulate(rad, [start, end], scan_info, _dict_["verbose"], _dict_)
    return sim.out, scan_info

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,17), help="Start date (default 2015-03-17T03)", 
            type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,18), help="End date (default 2015-03-17T03:02)", 
            type=prs.parse)
    parser.add_argument("-f", "--dofilter", action="store_false", help="Do filtering (default True)")
    parser.add_argument("-sv", "--save", action="store_false", help="Save data to file (defult True)")
    parser.add_argument("-gft", "--gflg_type", type=int, default=-1, help="Ground scatter flag estimation type (default -1)")
    parser.add_argument("-gct", "--gflg_cast", default="gflg_conv", help="Ground scatter flag final plot type (gflg_conv)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-th", "--thresh", type=float, default=.7, help="Threshold value for filtering (default 0.7)")
    parser.add_argument("-pth", "--pth", type=float, default=.25, help="Probability threshold for KDE (default 0.25)")
    parser.add_argument("-lth", "--lth", type=float, default=.2, help="Probability cut-off for IS (default 0.2)")
    parser.add_argument("-uth", "--uth", type=float, default=.8, help="Probability cut-off for GS (default 0.8)")
    parser.add_argument("-sid", "--sim_id", default="L100", help="Simulation ID, need to store data into this folder (default L100)")
    parser.add_argument("-ftype", "--ftype", default="fitacf", help="FitACF file type (futacf, fitacf3)")
    args = parser.parse_args()
    logger.info(f"Simulation run using fitacf_amgeo.__main__")
    _dic_ = {}
    if args.verbose:
        logger.info("Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
            _dic_[k] = vars(args)[k]
    out_dir = utils.build_base_folder_structure(args.rad, args.sim_id)
    _o, scan_info = run_fitacf_amgeo_simulation(args.rad, args.start, args.end, args.gflg_cast, _dic_)
    logger.info(f"Simulation output from fitacf_amgeo.__main__\n{json.dumps(_o, sort_keys=True, indent=4)}")
    _dic_["out_dir"] = out_dir
    _dic_.update(scan_info)
    _dic_.update(_o)
    _dic_["__func__"] = "fitacf_amgeo"
    utils.save_cmd(sys.argv, _dic_, out_dir)
    pass