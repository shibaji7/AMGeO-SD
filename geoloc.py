#!/usr/bin/env python

"""geoloc.py: module is dedicated to run davitpy (python 2.7) and store glat-glon into files."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import datetime as dt
import argparse
from netCDF4 import Dataset
import time
import os

from davitpy.pydarn.radar import radar
from davitpy.pydarn.radar.radFov import fov


def geolocate_radar_fov(rad):
    """ Geolocate each range cell """
    r = radar(code=rad)
    s = r.sites[0]
    f = fov(site=s)
    blen, glen = len(f.beams), len(f.gates)
    print(glen)
    glat, glon = np.zeros((blen, glen)), np.zeros((blen, glen))
    for i,b in enumerate(f.beams):
        for j,g in enumerate(f.gates):
            glat[i,j], glon[i,j] = f.latCenter[b,g], f.lonCenter[b,g]
    fname = "data/sim/{rad}.geolocate.data.nc".format(rad=rad)
    rootgrp = Dataset(fname, "w", format="NETCDF4")
    rootgrp.description = """ Fitacf++ : Geolocated points for each range cells. """
    rootgrp.history = "Created " + time.ctime(time.time())
    rootgrp.source = "AMGeO - SD data processing"
    rootgrp.createDimension("nbeam", blen)
    rootgrp.createDimension("ngate", glen)
    beam = rootgrp.createVariable("beams","i1",("nbeam",))
    gate = rootgrp.createVariable("gates","i1",("ngate",))
    beam[:], gate[:], = f.beams, f.gates
    _glat = rootgrp.createVariable("lat","f4",("nbeam","ngate"))
    _glon = rootgrp.createVariable("lon","f4",("nbeam","ngate"))
    _glat[:], _glon[:] = glat, glon
    os.system("gzip "+fname)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    args = parser.parse_args()
    geolocate_radar_fov(args.rad)