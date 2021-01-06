#!/usr/bin/env python

"""fitacf_plots.py: module is dedicated to produce plots for fitacf++ datafiles."""

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

from plot_processed_data import PlotProcessedData

def run_fitacf_amgeo_plotting(param_file, rad, start, end, sim_id, verbose):
    """
    Method is dedicated to run fitacf++ output plotts
    
    Input parameters
    ----------------
    param_file <str>: Parameter file .json
    rad <str>: Radar code
    start <datetime>: Start datetime object
    end <datetime>: End datetime object
    sim_id <str>: Simulation ID
    verbose <bool>: If true print console
    
    Return parameters
    -----------------
    _dict_ {
            folder_location <list(str)>: folder location to that generates 
           }
    """
    scan_info = None
    if param_file is None: scan_info = fetch_print_fit_rec(rad, start, start + dt.timedelta(minutes=5))
    proc = PlotProcessedData(param_file, rad, [start, end], sim_id, scan_info, verbose)
    return proc.out

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_file", default="data/outputs/L100/sas/params.json", 
                        help="Output parameter file from the fitacf++ run")
    parser.add_argument("-r", "--rad", default=None, help="SuperDARN radar code")
    parser.add_argument("-s", "--start", default=None, help="Start date (e.g. 2015-03-17T03)", type=prs.parse)
    parser.add_argument("-e", "--end", default=None, help="End date (e.g. 2015-03-17T03:02)", type=prs.parse)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-sid", "--sim_id", default="L100", help="Simulation ID, need to store data into this folder (default L100)")
    args = parser.parse_args()
    logger.info(f"Simulation run using fitacf_plots.__main__")
    if args.verbose:
        logger.info("Parameter list for plotting simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    _o = run_fitacf_amgeo_plotting(args.param_file, args.rad, args.start, args.end, args.sim_id, args.verbose)
    logger.info(f"Simulation output from fitacf_plots.__main__\n{json.dumps(_o, sort_keys=True, indent=4)}")
    pass