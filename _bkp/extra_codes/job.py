#!/usr/bin/env python

"""job.py: module is dedicated to simulate large chuck of data <6>-hr and store the data."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np

np.random.seed(0)

import os
import sys
sys.path.append("sd/")
import datetime as dt
import argparse
from analysis import Simulate, Process2Movie
import dateutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pro", "--program", type=int, default=0, help="Run program based on nmber (0-simulate, 1-fetch and plot data)")
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,17), help="Start date (default 2015-03-17)", 
            type=dateutil.parser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,18), help="End date (default 2015-03-18)", 
            type=dateutil.parser.isoparse)
    parser.add_argument("-d", "--dur", type=int, default=2, help="Scan duration in mins (default 2)")
    parser.add_argument("-t", "--themis", type=int, default=6, help="Themis beam number (default 6)")
    parser.add_argument("-f", "--dofilter", action="store_false", help="Do filtering (default True)")
    parser.add_argument("-m", "--movie", action="store_true", help="Make movies (default False)")
    parser.add_argument("-sk", "--skills", action="store_true", help="Do filtering (default False)")
    parser.add_argument("-ip", "--inv_plot", action="store_true", help="Investigative plots (defult False)")
    parser.add_argument("-sv", "--save", action="store_false", help="Save data to file (defult True)")
    parser.add_argument("-gft", "--gflg_type", type=int, default=-1, help="Ground scatter flag estimation type (default -1)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-th", "--thresh", type=float, default=.7, help="Threshold value for filtering (default 0.7)")
    parser.add_argument("-pth", "--pth", type=float, default=.25, help="Probability threshold for KDE (default 0.25)")
    parser.add_argument("-lth", "--lth", type=float, default=.2, help="Probability cut-off for IS (default 0.2)")
    parser.add_argument("-uth", "--uth", type=float, default=.8, help="Probability cut-off for GS (default 0.8)")
    parser.add_argument("-sid", "--sim_id", default="L100", help="Simulation ID, need to store data into this folder (default L100)")
    args = parser.parse_args()

    start = args.start
    run = 1 
    while start < args.end:
        end = start + dt.timedelta(hours=6)
        end = end if end < args.end else args.end
        cmd = "python simulate.py -pro {pro} -r {r} -s {s} -e {e} -d {d} -t {t} -gft {gft} -th {th}"\
                " -pth {pth} -lth {lth} -uth {uth} -sid {sid}".format(pro=args.program, r=args.rad, 
                        s=start.strftime("%Y-%m-%dT%H:%M"), e=end.strftime("%Y-%m-%dT%H:%M"),
                        d=args.dur, t=args.themis, gft=args.gflg_type, th=args.thresh, pth=args.pth,
                        lth=args.lth, uth=args.uth, sid=args.sim_id)
        if not args.dofilter: cmd = cmd + " -f" 
        if args.movie: cmd = cmd + " -m" 
        if args.skills: cmd = cmd + " -sk" 
        if args.inv_plot: cmd = cmd + " -ip" 
        if not args.save: cmd = cmd + " -sv" 
        if args.clear and run==1: cmd = cmd + " -c" 
        if args.verbose: cmd = cmd + " -v" 
        print("\n ",cmd)
        os.system(cmd)
        start = end
        run += 1
    print("")
    os.system("rm -rf sd/__pycache__")
