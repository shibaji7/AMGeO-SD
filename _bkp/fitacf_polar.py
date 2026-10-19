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

import matplotlib.pyplot as plt

from get_fit_data import FetchData

if __name__ == "__main__":
    rad = "rkn"
    dates = [
        dt.datetime(2015,3,17), dt.datetime(2015,3,18)
    ]
    fd = FetchData(rad, dates)
    beams, _ = fd.fetch_data()
    df = fd.convert_to_pandas(beams)
    df["srange"] = df.slist * df.rsep + df.frang
    plt.hist(df[df.srange<=600].v, bins=50, color="r", histtype="step", label="<=600 km")
    plt.hist(df[df.srange>600].v, bins=50, color="b", histtype="step", label=">600 km")
    plt.legend(loc=1)
    plt.xlim(-2000, 2000)
    plt.ylabel("Counts")
    plt.xlabel("Velocity")
    plt.savefig("hitogram.png")