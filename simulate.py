#!/usr/bin/env python

"""simulate.py: module is dedicated to simulate days of data and store the data."""

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

import sys
sys.path.append("sd/")
import datetime as dt
import argparse
import dateutil

import utils
from analysis import Simulate, Process2Movie

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pro", "--program", type=int, default=0, help="Run program based on nmber (0-simulate, 1-fetch and plot data)")
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,17,3), help="Start date (default 2015-03-17T03)", 
            type=dateutil.parser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,17,3,2), help="End date (default 2015-03-17T03:02)", 
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
    parser.add_argument("-pl_xu", "--plt_xulim", type=int, default=18, help="Upper X-lim of plotting (default 18)")
    parser.add_argument("-pl_xl", "--plt_xllim", type=int, default=-2, help="Lower X-lim of plotting (default -2)")
    parser.add_argument("-pl_yu", "--plt_yulim", type=int, default=70, help="Upper Y-lim of plotting (default 70)")
    parser.add_argument("-pl_yl", "--plt_yllim", type=int, default=0, help="Lower Y-lim of plotting (default 0)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    utils.save_cmd(sys.argv, "data/outputs/{rad}/{sim_id}/".format(rad=args.rad, sim_id=args.sim_id))
    stype = "themis" if args.themis>0 else None
    if args.program == 0:
        sim = Simulate(args.rad, [args.start, args.end],
            {"stype":stype, "beam":args.themis, "dur":args.dur}, inv_plot=args.inv_plot, dofilter=args.dofilter, 
            make_movie=args.movie, gflg_type=args.gflg_type, skills=args.skills, save=args.save, clear=args.clear, 
            thresh=args.thresh, pth=args.pth, pbnd=[args.lth, args.uth], verbose=args.verbose, sim_id=args.sim_id, 
            plt_xulim=args.plt_xulim, plt_xllim=args.plt_xllim, plt_yulim=args.plt_yulim, plt_yllim=args.plt_yllim)
    elif args.program == 1:
        proc = Process2Movie(args.rad, [args.start, args.end], args.sim_id, {"stype":stype, "beam":args.themis, "dur":args.dur})
        print("\n TODO\n")
    else: print("\n Invalid option, try 'python simulate.py -h'\n")
    import os
    os.system("rm *.log")
    os.system("rm -rf sd/__pycache__")
