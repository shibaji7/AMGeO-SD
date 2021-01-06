#!/usr/bin/env python

"""geopos.py: module is dedicated to run glat-glon into files."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import pydarn
import sys
sys.path.append("tools/")
import rad_fov
import geoPack
import aacgmv2
from loguru import logger

def geolocate_radar_fov(rad, time=None):
    """ 
    Geolocate each range cell 
    Input parameters
    ----------------
    rad <str>: Radar code
    time <datetime> (optional): datetime object to calculate magnetic cordinates
    
    Return parameters
    -----------------
    _dict_ {
            beams <list(int)>: Beams
            gates <list(int)>: Gates
            glat [beams, gates] <np.float>: Geogarphic latitude
            glon [beams, gates] <np.float>: Geogarphic longitude
            azm [beams, gates] <np.float>: Geogarphic azimuth of ray-bearing
            mlat [beams, gates] <np.float>: Magnetic latitude
            mlon [beams, gates] <np.float>: Magnetic longitude
            mazm [beams, gates] <np.float>: Magnetic azimuth of ray-bearing
           }
    
    Calling sequence
    ----------------
    from geopos import geolocate_radar_fov
    _dict_ = geolocate_radar_fov("bks")
    """
    logger.info(f"Geolocate radar ({rad}) FoV")
    hdw = pydarn.read_hdw_file(rad)
    fov_obj = rad_fov.CalcFov(hdw=hdw, altitude=300.)
    blen, glen = hdw.beams, hdw.gates
    if glen >= 110: glen = 110
    glat, glon, azm = np.zeros((blen, glen)), np.zeros((blen, glen)), np.zeros((blen, glen))
    mlat, mlon, mazm = np.zeros((blen, glen))*np.nan, np.zeros((blen, glen))*np.nan, np.zeros((blen, glen))*np.nan
    for i,b in enumerate(range(blen)):
        for j,g in enumerate(range(glen)):
            if j < 110: 
                glat[i,j], glon[i,j] = fov_obj.latCenter[b,g], fov_obj.lonCenter[b,g]
                d = geoPack.calcDistPnt(fov_obj.latFull[b,g], fov_obj.lonFull[b,g], 300,
                        distLat=fov_obj.latFull[b, g + 1], distLon=fov_obj.lonFull[b, g + 1], distAlt=300)
                azm[i,j] = d["az"]
                
                if time is not None:
                    mlat[i,j], mlon[i,j], _ = aacgmv2.convert_latlon(fov_obj.latFull[b,g], fov_obj.lonFull[b,g],
                                                                     300., time, method_code="G2A")
                    mlat2, mlon2, _ = aacgmv2.convert_latlon(fov_obj.latFull[b,g+1], fov_obj.lonFull[b,g+1], 
                                                             300., time, method_code="G2A")
                    d = geoPack.calcDistPnt(mlat[i,j], mlon[i,j], 300, distLat=mlat2, distLon=mlon2, distAlt=300)
                    mazm[i,j] = d["az"]
    _dict_ = {"beams": range(blen), "gates": range(glen), "glat": glat, "glon": glon, 
              "azm": azm, "mlat": mlat, "mlon": mlon, "mazm": mazm}
    return _dict_

# Testing code
if __name__ == "__main__":
    import datetime as dt
    _ = geolocate_radar_fov("bks")
    _dict_ = geolocate_radar_fov("bks", dt.datetime(2015,1,1))
    print(_dict_["mlat"], _dict_["mlon"], _dict_["mazm"])