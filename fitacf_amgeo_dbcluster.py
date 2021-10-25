#!/usr/bin/env python

"""fitacf_amgeo_dbcluster.py: module is dedicated to produce clustering data files."""

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
from clustering_algo import DBScan

def run_fitacf_amgeo_clustering(args):
    """
    Method is dedicated to run fitacf++ simulation
    
    Return parameters
    -----------------
    _dict_ {
            folder_location <list(str)>: folder location to that generates 
           }
    scan_info {
                scan_details_parameters 
              }
    """
    
    db = DBScan(args)
    return

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="bks", help="SuperDARN radar code (default bks)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,16), help="Start date (default 2015-03-16)", 
            type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,16,1), help="End date (default 2015-03-17)", 
            type=prs.parse)
    parser.add_argument("-f", "--dofilter", action="store_false", help="Do filtering (default True)")
    parser.add_argument("-sv", "--save", action="store_false", help="Save data to file (defult True)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-tm", "--tb_mult", type=int, default=10, help="Themis beam min sample DBSCAN multiplier")
    parser.add_argument("-th", "--thresh", type=float, default=.7, help="Threshold value for filtering (default 0.7)")
    parser.add_argument("-sid", "--sim_id", default="101", help="Simulation ID, need to store data into this folder (default L100)")
    parser.add_argument("-ftype", "--ftype", default="fitacf", help="FitACF file type (futacf, fitacf3)")
    parser.add_argument("-eps", "--eps", default=1, type=int, help="Epsilon for DBSCAN")
    parser.add_argument("-eps_g", "--eps_g", default=1, type=int, help="Epsilon for Range Gate")
    parser.add_argument("-eps_s", "--eps_s", default=1, type=int, help="Epsilon for Scan")
    parser.add_argument("-ms", "--min_samples", default=5, type=int, help="Min Samples for DBSCAN")
    args = parser.parse_args()
    logger.info(f"Simulation run using fitacf_amgeo_dbcluster.__main__")
    _dic_ = {}
    if args.verbose:
        logger.info("Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
            _dic_[k] = vars(args)[k]
    out_dir = utils.build_base_folder_structure(args.rad, args.sim_id)
    run_fitacf_amgeo_clustering(args)
    #_o, scan_info = run_fitacf_amgeo_clustering(args.rad, args.start, args.end, args.gflg_cast, _dic_)
    #logger.info(f"Simulation output from fitacf_amgeo_dbcluster.__main__\n{json.dumps(_o, sort_keys=True, indent=4)}")
    #_dic_["out_dir"] = out_dir
    #_dic_.update(scan_info)
    #_dic_.update(_o)
    #_dic_["gflg_type"] = -1
    #_dic_["start"] = args.start
    #_dic_["__func__"] = "fitacf_amgeo_cluster"
    utils.save_cmd(sys.argv, _dic_, out_dir)
    pass