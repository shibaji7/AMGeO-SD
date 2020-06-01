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

    def __init__(self, rad, date_range, scan_prop, verbose=True, **keywords):
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
                - save: save to hdf files
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
        return

    def _clear(self):
        """
        Clean the output folder
        """
        os.system("rm -rf data/outputs/{rad}/*".format(rad=self.rad))
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
            self.skills[name] = Skills(pd.DataFrame.from_records(_u).values, np.array(labels[name]), name, verbose=True)
        if hasattr(self, "save") and self.save: self._to_hdf5()
        return

    def _to_hdf5(self):
        """
        Save filterd value to .h5 file
        """
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde"]
        fname = "data/outputs/{rad}/data.h5".format(rad=self.rad)
        _u = {key: [] for key in v_params + s_params}
        for fscan in self.fscans:
            for b in fscan.beams:
                l = len(getattr(b, "slist"))
                for p in v_params:
                    _u[p].extend(getattr(b, p))
                for p in s_params:
                    _u[p].extend([getattr(b, p)]*l)
        pd.DataFrame.from_records(_u).to_hdf(fname, key="df")
        return

    def _dofilter(self):
        """
        Do filtering for all scan
        """
        self.fscans = []
        if self.verbose: print("\n Very slow method to boxcar filtering.")
        for i, e in enumerate(self.dates):
            scans = self.scans[i:i+3]
            self.fscans.append(self.filter.doFilter(scans, gflg_type=self.gflg_type))
        return

    def _make_movie(self):
        """
        Make movies out of the plots
        """
        if self.verbose: print("\n Very slow method to plot and create movies.")
        _m = MM(self.rad, self.dates, self.scans, "raw")
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
            ip = IP("1plot", e, self.rad, scans, {"thresh":0.7, "gs":"gflg_conv", "pth":0.5, "pbnd":[0.2, 0.8], 
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
        self.filter = Filter()
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