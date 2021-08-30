#!/usr/bin/env python

"""analysis.py: module is dedicated to run all analysis."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
import datetime as dt
import pandas as pd
import time
from netCDF4 import Dataset, date2num
from loguru import logger

import utils
from get_fit_data import FetchData
from boxcar_filter import Filter

import multiprocessing as mp
from functools import partial

np.random.seed(0)

class Simulate(object):
    """Class to simulate all boxcar filters for duration of the time"""

    def __init__(self, rad, date_range, scan_prop, verbose=False, _dict_={}, cores=16):
        """
        initialize all variables
        rad: radar code
        date_range: range of datetime [stime, etime]
        scan_prop: Properties of scan {'rad': 'sas', 's_time': 2, 't_beam': 6, 's_mode': 'themis'}
        verbose: print all sys logs
        _dict_: other key words for processing, plotting and saving the processed data
        """
        self.rad = rad
        self.date_range = date_range
        self.scan_prop = scan_prop
        self.verbose = verbose
        self.gflg_type = -1
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        _, _, self.rad_type = utils.get_radar(self.rad)
        self._create_dates()
        self._fetch()
        self.cores = cores
        self.p0 = mp.Pool(self.cores)
        self.out_dir = utils.get_config("BASE_DIR") + _dict_["sim_id"] + "/" + self.rad + "/"
        self.parse_out_dic()
        if hasattr(self, "clear") and self.clear: self._clear()
        if hasattr(self, "dofilter") and self.dofilter: self._dofilter()
        if hasattr(self, "save") and self.save: self._to_files()
        return
    
    def parse_out_dic(self):
        """
        Parse output directory and folders
        """
        self.out = {}
        self.out["out_dir"] = self.out_dir
        self.out["out_files"] = {}
        self.out["out_files"]["h5_file"] = self.out_dir + "data_{s}_{e}.h5".format(s=self.date_range[0].strftime("%Y%m%d.%H%M"),
                                                                     e=self.date_range[1].strftime("%Y%m%d.%H%M"))
        self.out["out_files"]["nc_file"] = self.out_dir + "data_{s}_{e}.nc".format(s=self.date_range[0].strftime("%Y%m%d.%H%M"),
                                                                     e=self.date_range[1].strftime("%Y%m%d.%H%M"))
        self.out["out_files"]["param_file"] = self.out_dir + "params.json"
        self.out["out_files"]["script_file"] = self.out_dir + "script.cmd"
        return

    def _clear(self):
        """
        Clean the output folder
        """
        if self.verbose: logger.info("Clear output folder")
        os.system("rm -rf {out_dir}*".format(out_dir=self.out_dir))
        return

    def _to_files(self):
        """
        Save filterd value to .nc and .h5 file
        """
        
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang", "time", "intt.sc", "intt.us", "mppul"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]
        fname = self.out["out_files"]["h5_file"]
        _u = {key: [] for key in v_params + s_params}
        _du = {key: [] for key in v_params + s_params}
        blen, glen = 0, 110
        for fscan in self.fscans[1:-1]:
            blen += len(fscan.beams)
            for b in fscan.beams:
                l = len(getattr(b, "slist"))
                for p in v_params:
                    _u[p].extend(getattr(b, p))
                    _du[p].append(getattr(b, p))
                for p in s_params:
                    _u[p].extend([getattr(b, p)]*l)
                    _du[p].append(getattr(b, p))
        
        pd.DataFrame.from_records(_u).to_hdf(fname, key="df")
        os.system("gzip " + fname)

        if self.verbose: logger.info(f"Shape of the filtered beam dataset [beams X gates]- {blen},{glen}")
        fname = self.out_dir + "data_{s}_{e}.nc".format(s=self.date_range[0].strftime("%Y%m%d.%H%M"),
                                                        e=self.date_range[1].strftime("%Y%m%d.%H%M"))
        rootgrp = Dataset(fname, "w", format="NETCDF4")
        rootgrp.description = """
                                 Fitacf++ : Boxcar filtered data.
                                 Filter parameter: weight matrix - default; threshold - {th}; GS Flag Type - {gflg_type}
                                 Parameters (Thresholds) - CONV(q=[{l},{u}]), KDE(p={pth},q=[{l},{u}])
                                 (thresh:{th},pbnd:{l}-{u},pth:{pth},gflg_type:{gflg_type})
                              """.format(th=self.thresh, gflg_type=self.gflg_type, l=self.lth, u=self.uth, pth=self.pth)
        rootgrp.history = "Created " + time.ctime(time.time())
        rootgrp.source = "AMGeO - SD data processing"
        rootgrp.beam_per_scan = len(self.fscans[0].beams)
        rootgrp.createDimension("nbeam", blen)
        rootgrp.createDimension("ngate", glen)
        beam = rootgrp.createVariable("nbeam","i1",("nbeam",))
        gate = rootgrp.createVariable("ngate","i1",("ngate",))
        beam[:], gate[:] = range(blen), range(glen)
        times = rootgrp.createVariable("time", "f8", ("nbeam",))
        times.units = "hours since 1970-01-01 00:00:00.0"
        times.calendar = "julian"
        times[:] = date2num(_du["time"],units=times.units,calendar=times.calendar)
        s_params, type_params, desc_params = ["bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"],\
                ["i1","f4","f4","i1","f4","f4","f4","i1"],\
                ["Beam numbers", "Sky Noise", "Frequency", "Scan Flag", "Max. Range Gate", "Integration sec", "Integration u.sec",
                        "Number of pulses"]
        for _i, k in enumerate(s_params):
            tmp = rootgrp.createVariable(k, type_params[_i],("nbeam",))
            tmp.description = desc_params[_i]
            tmp[:] = np.array(_du[k])
        v_params, desc_params = ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"],\
                ["LoS Velocity (+ve: towards the radar, -ve: away from radar)", "LoS Width",
                        "GS Flag (%d)"%self.gflg_type, "LoS Power", "Gate", "GS Flag (Conv)", "GS Flag (KDE)",
                        "MAD of LoS Velociy after filtering"]
        _g = _du["slist"]
        for _i, k in enumerate(v_params):
            tmp = rootgrp.createVariable(k, "f4", ("nbeam", "ngate"))
            tmp.description = desc_params[_i]
            _m = np.empty((blen,glen))
            _m[:], x = np.nan, _du[k]
            for _j in range(blen):
                _m[_j,_g[_j]] = np.array(x[_j])
            tmp[:] = _m

        v_params, desc_params = ["v", "w_l", "gflg", "p_l", "v_e", "slist"],\
                ["LoS Velocity (Fitacf; +ve: towards the radar, -ve: away from radar)", "LoS Width (Fitacf)", 
                        "GS Flag (%d) (Fitacf)"%self.gflg_type, 
                        "LoS Power (Fitacf)", "LoS Velocity Error", "Gate (Fitacf)"]
        s_params, type_params, desc_params = ["time", "bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"],\
                ["i1","f4","f4","i1","f4","f4","f4","i1"],\
                ["Beam numbers", "Sky Noise", "Frequency", "Scan Flag", "Max. Range Gate", "Integration sec", "Integration u.sec",                                       "Number of pulses"]
        fblen = 0
        _ru = {key: [] for key in v_params+s_params}
        for rscan in self.scans[1:-1]:
            fblen += len(rscan.beams)
            for b in rscan.beams:
                for p in v_params:
                    _ru[p].append(getattr(b, p))
                for p in s_params:
                    _ru[p].append(getattr(b, p))
        if self.verbose: logger.info(f"Shape of the beam dataset [beams X gates]- {fblen},{glen}")
        s_params = ["bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"]
        rootgrp.createDimension("fitacf_nbeam", fblen)
        fitacf_nbeam = rootgrp.createVariable("fitacf_nbeam","i1",("fitacf_nbeam",))
        fitacf_nbeam[:] = range(fblen)
        for _i, k in enumerate(s_params):
            tmp = rootgrp.createVariable("fitacf_" + k, type_params[_i],("fitacf_nbeam",))
            tmp.description = desc_params[_i]
            tmp[:] = np.array(_ru[k])
        times = rootgrp.createVariable("fitacf_time", "f8", ("fitacf_nbeam",))
        times.units = "hours since 1970-01-01 00:00:00.0"
        times.calendar = "julian"
        times[:] = date2num(_ru["time"],units=times.units,calendar=times.calendar)
        _g = _ru["slist"]
        for _i, k in enumerate(v_params):
            tmp = rootgrp.createVariable("fitacf_"+k, "f4", ("fitacf_nbeam", "ngate"))
            tmp.description = desc_params[_i]
            _m = np.empty((fblen,glen))
            _m[:], x = np.nan, _ru[k]
            for _j in range(fblen):
                if len(_m[_j,_g[_j]]) == len(np.array(x[_j])): _m[_j,_g[_j]] = np.array(x[_j])
            tmp[:] = _m
        rootgrp.close()
        os.system("gzip " + fname)
        return

    def _dofilter(self):
        """
        Do filtering for all scan
        """
        self.fscans, fscans = [], []
        if self.verbose: logger.info("Very slow method to boxcar filtering.")
        for i in range(len(self.dates)):
            scans = self.scans[i:i+3]
            fscans.append(scans)
        j = 0
        partial_filter = partial(self.filter.doFilter, gflg_type=self.gflg_type)
        for s in self.p0.map(partial_filter, fscans):
            if np.mod(j,10)==0: logger.info(f"Running median filter for scan at - {self.dates[j]}")
            self.fscans.append(s)
            j += 1
        if self.verbose: logger.info(f"Total number of scans invoked by median filter {j}")
        return

    def _fetch(self):
        """
        Fetch all the data with given date range
        """
        drange = self.date_range
        drange[0], drange[1] = drange[0] - dt.timedelta(minutes=2*self.scan_prop["s_time"]),\
                drange[1] + dt.timedelta(minutes=3*self.scan_prop["s_time"])
        start, end = drange[0].strftime("%Y-%m-%d %H:%M"), drange[1].strftime("%Y-%m-%d %H:%M")
        if self.verbose: logger.info(f"Fetch data from time range - [{start}, {end}]")
        self.io = FetchData(self.rad, drange, ftype=self.ftype, verbose=self.verbose)
        _, scans = self.io.fetch_data(by="scan", scan_prop=self.scan_prop)
        self._org_scans = scans
        self.filter = Filter(thresh=self.thresh, pbnd=[self.lth, self.uth], pth=self.pth, verbose=self.verbose)
        self.scans = []
        if self.verbose: logger.info(f"Discarding repeating beams and reorganize scan for median filter")
        for s in scans:
            self.scans.append(self.filter._discard_repeting_beams(s))
        return

    def _create_dates(self):
        """
        Create all dates to run filter simulation
        """
        if self.verbose: logger.info("Creating time sequence")
        self.dates = []
        dates = (self.date_range[1] + dt.timedelta(minutes=self.scan_prop["s_time"]) - 
                self.date_range[0]).total_seconds() / (self.scan_prop["s_time"] * 60) 
        for d in range(int(dates) + 2):
            self.dates.append(self.date_range[0] - dt.timedelta(minutes=self.scan_prop["s_time"]) 
                    + dt.timedelta(minutes=d*self.scan_prop["s_time"]))
        return


