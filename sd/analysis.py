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

from utils import Skills
from get_sd_data import FetchData
from boxcar_filter import Filter
from plot_lib import InvestigativePlots as IP, MovieMaker as MM

np.random.seed(0)

class Simulate(object):
    """Class to simulate all boxcar filters for duration of the time"""

    def __init__(self, rad, date_range, scan_prop, verbose=False, **keywords):
        """
        initialize all variables
        rad: radar code
        date_range: range of datetime [stime, etime]
        scan_prop: Properties of scan {"stype":"normal", "dur":1.} or {"stype":"themis", "beam":6, "dur":2.}
        verbose: print all sys logs
        **keywords: other key words for processing, plotting and saving the processed data
                - inv_plot: method for investigative plots
                - dofilter: method for running the filter
                - make_movie: create movie
                - skills: estimate skills
                - save: save to hdf and netcdf files
                - clear: clear the storage folder
        """
        self.rad = rad
        self.date_range = date_range
        self.scan_prop = scan_prop
        self.verbose = verbose
        self.gflg_type = -1
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        self._create_dates()
        self._fetch()
        if hasattr(self, "clear") and self.clear: self._clear()
        if hasattr(self, "dofilter") and self.dofilter: self._dofilter()
        if hasattr(self, "inv_plot") and self.inv_plot: self._invst_plots()
        if hasattr(self, "make_movie") and self.make_movie: self._make_movie()
        if hasattr(self, "skills") and self.skills: self._estimate_skills()
        if hasattr(self, "save") and self.save: self._to_files()
        return

    def _clear(self):
        """
        Clean the output folder
        """
        os.system("rm -rf data/outputs/{rad}/{sim_id}/*".format(rad=self.rad, sim_id=self.sim_id))
        return

    def _estimate_skills(self):
        """
        Estimate skills of the overall clusteing
        """
        v_params=["v", "w_l", "p_l", "slist"]
        s_params=["bmnum"]
        self.skills, labels = {}, {"CONV":[], "KDE":[]}
        _u = {key: [] for key in v_params + s_params}
        for fscan in self.fscans:
            for b in fscan.beams:
                labels["CONV"].extend(getattr(b, "gflg_conv"))
                labels["KDE"].extend(getattr(b, "gflg_kde"))
                l = len(getattr(b, "slist"))
                for p in v_params:
                    _u[p].extend(getattr(b, p))
                for p in s_params:
                    _u[p].extend([getattr(b, p)]*l)
        for name in ["CONV","KDE"]:
            self.skills[name] = Skills(pd.DataFrame.from_records(_u).values, np.array(labels[name]), name, verbose=self.verbose)
        return

    def _to_files(self):
        """
        Save filterd value to .nc and .h5 file
        """
        if not os.path.exists("data/outputs/{rad}/{sim_id}/".format(rad=self.rad, sim_id=self.sim_id)): 
            os.system("mkdir -p data/outputs/{rad}/{sim_id}/".format(rad=self.rad, sim_id=self.sim_id))
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang", "time"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde"]
        fname = "data/outputs/{rad}/{sim_id}/data.h5".format(rad=self.rad, sim_id=self.sim_id)
        _u = {key: [] for key in v_params + s_params}
        _du = {key: [] for key in v_params + s_params}
        for fscan in self.fscans:
            for b in fscan.beams:
                l = len(getattr(b, "slist"))
                for p in v_params:
                    _u[p].extend(getattr(b, p))
                    _du[p].append(getattr(b, p))
                for p in s_params:
                    _u[p].extend([getattr(b, p)]*l)
                    _du[p].append(getattr(b, p))
        pd.DataFrame.from_records(_u).to_hdf(fname, key="df")

        fname = "data/outputs/{rad}/{sim_id}/data.nc".format(rad=self.rad, sim_id=self.sim_id)
        from netCDF4 import Dataset
        import time
        rootgrp = Dataset(fname, "w", format="NETCDF4")
        rootgrp.description = """
                                 Fitacf++ : Boxcar filtered data.
                                 Filter parameter: weight matrix - default; threshold - {th}
                                 Parameters (Thresholds) - CONV(q=[{l},{u}]), KDE(p={pth},q=[{l},{u}])
                              """.format(th=self.thresh, l=self.pbnd[0], u=self.pbnd[1], pth=self.pth)
        blen, glen = len(self.fscans)*len(self.fscans[0].beams), 75
        rootgrp.history = "Created " + time.ctime(time.time())
        rootgrp.source = "AMGeO - SD data processing"
        rootgrp.createDimension("beam", blen)
        rootgrp.createDimension("gate", glen)
        beam = rootgrp.createVariable("beam","i1",("beam",))
        gate = rootgrp.createVariable("gate","i1",("gate",))
        beam[:], gate[:], = range(blen), range(glen) 
        s_params, type_params, desc_params = ["bmnum","noise.sky", "tfreq", "scan", "nrang"], ["i1","f4","f4","i1","f4"],\
                ["Beam numbers", "Sky Noise", "Frequency", "Scan Flag", "Max. Range Gate"]
        for _i, k in enumerate(s_params):
            tmp = rootgrp.createVariable(k, type_params[_i],("beam",))
            tmp.description = desc_params[_i]
            tmp[:] = np.array(_du[k])
        v_params, desc_params = ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde"],\
                ["LoS Velocity", "LoS Width", "GS Flag (%d)"%self.gflg_type, "LoS Power", "Gate", "GS Flag (Conv)", "GS Flag (KDE)"]
        _g = _du["slist"]
        for _i, k in enumerate(v_params):
            tmp = rootgrp.createVariable(k, "f4", ("beam", "gate"))
            tmp.description = desc_params[_i]
            _m = np.empty((blen,glen))
            _m[:], x = np.nan, _du[k]
            for _j in range(blen):
                _m[_j,_g[_j]] = np.array(x[_j])
            tmp[:] = _m
        return

    def _dofilter(self):
        """
        Do filtering for all scan
        """
        self.fscans = []
        if self.verbose: print("\n Very slow method to boxcar filtering.")
        for i, e in enumerate(self.dates):
            scans = self.scans[i:i+3]
            print(" Date - ", e)
            self.fscans.append(self.filter.doFilter(scans, gflg_type=self.gflg_type))
        return

    def _make_movie(self):
        """
        Make movies out of the plots
        """
        if self.verbose: print("\n Very slow method to plot and create movies.")
        _m = MM(self.rad, self.dates, self.scans, "raw", sim_id=self.sim_id)
        _m.exe({"gflg_type":self.gflg_type})
        if hasattr(self, "dofilter") and self.dofilter:
            _m.figure_name = "med_filt"
            _m.scans = self.fscans
            _m.exe({"thresh":0.7, "gs":"gflg_conv", "pth":0.25, "pbnd":[0.2, 0.8], "zparam":"p_l"})
            _m.exe({"thresh":0.7, "gs":"gflg_kde", "pth":0.25, "pbnd":[0.2, 0.8], "zparam":"p_l"})
        return

    def _invst_plots(self):
        """
        Method for investigative plots - 
            1. Plot 1 X 1 panel plots
            2. Plot 2 X 2 panel plots
            3. Plot 3 - 2 filter analysis plots
        """
        if self.verbose: print("\n Slow method to plot 3 different types of investigative plots.")
        for i, e in enumerate(self.dates):
            scans = self.scans[i:i+3]
            if hasattr(self, "dofilter") and self.dofilter: scans.append(self.fscans[i])
            ip = IP("1plot", e, self.rad, self.sim_id, scans, {"thresh":0.7, "gs":"gflg_conv", "pth":0.5, "pbnd":[0.2, 0.8], 
                "gflg_type": self.gflg_type})
            ip.draw()
            ip.draw("4plot")
            if hasattr(self, "dofilter") and self.dofilter: ip.draw("5plot")
        return

    def _fetch(self):
        """
        Fetch all the data with given date range
        """
        drange = self.date_range
        drange[0], drange[1] = drange[0] - dt.timedelta(minutes=2*self.scan_prop["dur"]),\
                drange[1] + dt.timedelta(minutes=2*self.scan_prop["dur"])
        io = FetchData(self.rad, drange)
        _, scans = io.fetch_data(by="scan", scan_prop=self.scan_prop)
        self.filter = Filter(thresh=self.thresh, pbnd=self.pbnd, pth=self.pth, verbose=self.verbose)
        self.scans = []
        for s in scans:
            self.scans.append(self.filter._discard_repeting_beams(s))
        return

    def _create_dates(self):
        """
        Create all dates to run filter simulation
        """
        self.dates = []
        dates = (self.date_range[1] + dt.timedelta(minutes=self.scan_prop["dur"]) - self.date_range[0]).seconds / (self.scan_prop["dur"] * 60) 
        for d in range(int(dates)):
            self.dates.append(self.date_range[0] + dt.timedelta(minutes=d*self.scan_prop["dur"]))
        return


if __name__ == "__main__":
    sim = Simulate("sas", [dt.datetime(2015,3,17,3,2), dt.datetime(2015,3,17,3,2)], 
            {"stype":"themis", "beam":6, "dur":2.}, inv_plot=True, dofilter=True, 
            gflg_type=-1, skills=True, save=True)
    import os
    os.system("rm *.log")
    os.system("rm -rf __pycache__")
