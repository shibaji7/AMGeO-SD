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

import datetime as dt

from get_sd_data import FetchData
from boxcar_filter import Filter
from plot_lib import InvestigativePlots as IP

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
                - 
        """
        self.rad = rad
        self.date_range = date_range
        self.scan_prop = scan_prop
        self._create_dates()
        self._fetch()
        self.verbose = verbose
        for p in keywords.keys():
            setattr(self, p, keywords[p])
        if hasattr(self, "inv_plot") and self.inv_plot: self._invst_plots()
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
            ip = IP("1plot", e, self.rad, self.scans[i:i+3])
            ip.draw()
            ip.draw("4plot")
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
            {"stype":"themis", "beam":6, "dur":2.}, inv_plot=True)
    import os
    os.system("rm *.log")
    os.system("rm -rf __pycache__")
