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
from analysis import Simulate


if __name__ == "__main__":
    sim = Simulate("sas", [dt.datetime(2015,3,17,3), dt.datetime(2015,3,17,4)],
            {"stype":"themis", "beam":6, "dur":2.}, inv_plot=False, dofilter=True, make_movie=True,
            gflg_type=-1, skills=True, save=True, clear=True)
    import os
    os.system("rm *.log")
    os.system("rm -rf sd/__pycache__")
