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
from netCDF4 import Dataset, date2num, num2date
import time
import glob
import copy

import utils
from utils import Skills
from get_sd_data import FetchData, Beam, Scan, beam_summary
from boxcar_filter import Filter, MiddleLatFilter
from plot_lib import InvestigativePlots as IP, MovieMaker as MM

np.random.seed(0)


def printFitRec(rad, dates):
    """ Print fit Record list """
    print(" Summary: " , beam_summary(rad, dates[0], dates[0] + dt.timedelta(minutes=10)))
    return

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
        _, _, self.rad_type = utils.get_radar(self.rad)
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
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang", "time", "intt.sc", "intt.us", "mppul"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]
        fname = "data/outputs/{rad}/{sim_id}/data_{s}_{e}.h5".format(rad=self.rad, sim_id=self.sim_id, 
                s=self.date_range[0].strftime("%Y%m%d.%H%M"), e=self.date_range[1].strftime("%Y%m%d.%H%M"))
        _u = {key: [] for key in v_params + s_params}
        _du = {key: [] for key in v_params + s_params}
        blen, glen = 0, 110
        for fscan in self.fscans:
            blen += len(fscan.beams)
            for b in fscan.beams:
                l = len(getattr(b, "slist"))
                for p in v_params:
                    _u[p].extend(getattr(b, p))
                    _du[p].append(getattr(b, p))
                for p in s_params:
                    _u[p].extend([getattr(b, p)]*l)
                    _du[p].append(getattr(b, p))
        if hasattr(self, "hdf") and self.hdf:
            pd.DataFrame.from_records(_u).to_hdf(fname, key="df")
            os.system("gzip " + fname)

        print("\n Shape of the beam dataset -",blen,glen)
        fname = "data/outputs/{rad}/{sim_id}/data_{s}_{e}.nc".format(rad=self.rad, sim_id=self.sim_id,
                s=self.date_range[0].strftime("%Y%m%d.%H%M"), e=self.date_range[1].strftime("%Y%m%d.%H%M"))
        rootgrp = Dataset(fname, "w", format="NETCDF4")
        rootgrp.description = """
                                 Fitacf++ : Boxcar filtered data.
                                 Filter parameter: weight matrix - default; threshold - {th}; GS Flag Type - {gflg_type}
                                 Parameters (Thresholds) - CONV(q=[{l},{u}]), KDE(p={pth},q=[{l},{u}])
                                 (thresh:{th},pbnd:{l}-{u},pth:{pth},gflg_type:{gflg_type})
                              """.format(th=self.thresh, gflg_type=self.gflg_type, l=self.pbnd[0], u=self.pbnd[1], pth=self.pth)
        rootgrp.history = "Created " + time.ctime(time.time())
        rootgrp.source = "AMGeO - SD data processing"
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
        for rscan in self.scans:
            fblen += len(rscan.beams)
            for b in rscan.beams:
                for p in v_params:
                    _ru[p].append(getattr(b, p))
                for p in s_params:
                    _ru[p].append(getattr(b, p))
        print("\n Shape of the beam dataset -",fblen,glen)
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

    def slow_filter(self, fscans):
        """
        Do filter for sub-auroral scatter
        """
        df = pd.DataFrame()
        from sklearn.cluster import DBSCAN
        sid = []
        for i, fsc in enumerate(fscans):
            dx = self.io.convert_to_pandas(fsc.beams, v_params=["v", "w_l", "slist"])
            df = df.append(dx)
            sid.extend([i]*len(dx))
        df["sid"] = sid
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(df[["bmnum","slist"]].values)
        df["labels"] = clustering.labels_
        df["gflg"] = [np.nan] * len(clustering.labels_)
        #for l in set(clustering.labels_):
            #u = df[df.labels==l]
            #v, w = np.array(u.v), np.array(u.w_l)
            #gflg = ( (w-(50-(0.7*(v+5)**2))) < 0 ).astype(int)
            #if np.abs(np.median(v)) < 50: gflg = ( (w-(50-(0.7*(v+5)**2))) < 0 ).astype(int)
            #else: gflg = [0]*len(v)
            #df.loc[df.labels==l, "gflg"] = gflg
            #df.loc[df.labels==l, "gflg"] = max(set(gflg), key=gflg.tolist().count)
        for i, fsc in enumerate(fscans):
            dx = df[df.sid==i]
            for l in set(clustering.labels_):
                u = dx[dx.labels==l]
                if len(u) > 0:
                    v, w = np.array(u.v), np.array(u.w_l)
                    gflg = ( (w-(50-(0.7*(v+5)**2))) < 0 ).astype(int)
                    dx.loc[dx.labels==l, "gflg"] = max(set(gflg), key=gflg.tolist().count)
            for b in fsc.beams:
                b.gsflg[3] = np.array(dx[dx.bmnum==b.bmnum].gflg)
        self.gflg_type = 3
        return

    def _dofilter(self):
        """
        Do filtering for all scan
        """
        self.fscans = []
        if self.verbose: print("\n Very slow method to boxcar filtering.")
        if self.rad_type == "mid":
            #midlatfilt = MiddleLatFilter(self.rad, self._org_scans)
            #midlatfilt.doFilter(self.io)
            pass
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
        _m = MM(self.rad, self.dates, self.scans, "raw", sim_id=self.sim_id, 
                xlim=[self.plt_xllim, self.plt_xulim], ylim=[self.plt_yllim, self.plt_yulim])
        _m.exe({"gflg_type":self.gflg_type})
        if hasattr(self, "dofilter") and self.dofilter:
            _m.figure_name = "med_filt"
            _m.scans = self.fscans
            _m.exe({"thresh":self.thresh, "gs":"gflg_conv", "pth":self.pth, "pbnd":self.pbnd, "zparam":"p_l"})
            _m.exe({"thresh":self.thresh, "gs":"gflg_kde", "pth":self.pth, "pbnd":self.pbnd, "zparam":"p_l"})
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
            ip = IP("1plot", e, self.rad, self.sim_id, scans, {"thresh":self.thresh, "gs":self.gflg_cast, "pth":self.pth, "pbnd":self.pbnd, 
                "gflg_type": self.gflg_type}, xlim=[self.plt_xllim, self.plt_xulim], ylim=[self.plt_yllim, self.plt_yulim])
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
                drange[1] + dt.timedelta(minutes=3*self.scan_prop["dur"])
        self.io = FetchData(self.rad, drange)
        _, scans = self.io.fetch_data(by="scan", scan_prop=self.scan_prop)
        self._org_scans = scans
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
        dates = (self.date_range[1] + dt.timedelta(minutes=self.scan_prop["dur"]) - 
                self.date_range[0]).total_seconds() / (self.scan_prop["dur"] * 60) 
        for d in range(int(dates) + 2):
            self.dates.append(self.date_range[0] - dt.timedelta(minutes=self.scan_prop["dur"]) 
                    + dt.timedelta(minutes=d*self.scan_prop["dur"]))
        return

class Process2Movie(object):
    """Class to fetch processed oata and create movies"""
    
    def __init__(self, rad, date_range, sim_id, scan_prop, verbose=True):
        """
        initialize all variables
        rad: radar code
        date_range: range of datetime [stime, etime]
        verbose: print all sys logs
        sim_id: simulation id
        """
        self.rad = rad
        self.date_range = date_range
        self.verbose = verbose
        self.scan_prop = scan_prop
        self.sim_id = sim_id
        self._fetch_data()
        self._create_dates()
        self._to_scans()
        self._invst_plots()
        self._make_movie()
        return

    def _create_dates(self):
        """
        Create all dates to run plotting method
        """
        self.drange = [self.date_range[0] - dt.timedelta(minutes=self.scan_prop["dur"]),
                self.date_range[1] + dt.timedelta(minutes=self.scan_prop["dur"])]
        self.dates = []
        dates = (self.date_range[1] + dt.timedelta(minutes=self.scan_prop["dur"]) -
                self.date_range[0]).total_seconds() / (self.scan_prop["dur"] * 60)
        for d in range(int(dates)):
            self.dates.append(self.date_range[0] + dt.timedelta(minutes=d*self.scan_prop["dur"]))
        return

    def _ext_params(self, params):
        """
        Extract parameters from the list
        """
        for p in params:
            k, v = p.split(":")[0].replace(" ",""), p.split(":")[1]
            if k == "thresh": setattr(self, k, float(v))
            if k == "pth": setattr(self, k, float(v))
            if k == "gflg_type": setattr(self, k, int(v))
            if k == "pbnd":
                x = v.split("-")
                setattr(self, k, [float(x[0]), float(x[1])])
        return

    def _get_nc_obj(self, s_params, v_params, pref):
        """
        Method that takes a beam sounding and convert to obj
        """
        d = dict(zip(s_params+v_params, ([] for _ in s_params+v_params)))
        d["time"] = []
        s_params = copy.copy(s_params)
        s_params.remove("time")
        for _o in self.datastore:
            params = _o.description.split("\n")[-2].replace("(", "").replace(")", "").split(",")
            self._ext_params(params)
            times = num2date(_o.variables[pref+"time"][:], _o.variables[pref+"time"].units, _o.variables[pref+"time"].calendar,
                    only_use_cftime_datetimes=False)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            d["time"].extend(times)
            for x in s_params:
                d[x].extend(_o.variables[pref+x][:])
            for x in v_params:
                d[x].append(_o.variables[pref+x][:].tolist())
        for x in s_params:
            d[x] = np.array(d[x])
        for x in v_params:
            u = d[x][0]
            for i in range(1,len(d[x])):
                u = np.append(u, d[x][i], axis=0)
            d[x] = u
        return d

    def _to_scans(self):
        """
        Convert to scans and filter scan
        """
        gscans = []
        prefxs = ["fitacf_", ""]
        vparams = [["v", "w_l", "gflg", "p_l", "v_e"], ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]]
        s_params = ["time", "bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"]
        for pref, v_params in zip(prefxs, vparams):
            _b, _s = [], []
            d = self._get_nc_obj(s_params, v_params, pref)
            for i, time in enumerate(d["time"]):
                bm = Beam()
                bm.set_nc(time, d, i, s_params, v_params)
                _b.append(bm)
            scan, sc =  0, Scan(None, None, self.scan_prop["stype"])
            sc.beams.append(_b[0])
            for dx in _b[1:]:
                if dx.scan == 1:
                    sc.update_time()
                    _s.append(sc)
                    sc = Scan(None, None, self.scan_prop["stype"])
                    sc.beams.append(dx)
                else: sc.beams.append(dx)
            sc.update_time()
            _s.append(sc)
            gscans.append(_s)
        self.scans, self.fscans = gscans[0], gscans[1]
        return

    def _fetch_data(self):
        """
        Fetch the data inside the folder
        """
        gzfn = "data/outputs/{rad}/{sim_id}/*.nc.gz".format(rad=self.rad, sim_id=self.sim_id)
        gzfns = glob.glob(gzfn)
        gzfns.sort()
        self.datastore = []
        for gf in gzfns:
            os.system("gzip -d " + gf)
            self.datastore.append(Dataset(gf.replace(".gz","")))
            os.system("gzip " + gf.replace(".gz",""))
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
            scans.append(self.fscans[i])
            ip = IP("1plot", e, self.rad, self.sim_id, scans, {"thresh":self.thresh, "gs":self.gflg_cast, "pth":self.pth, "pbnd":self.pbnd,
                "gflg_type": self.gflg_type}, xlim=[self.plt_xllim, self.plt_xulim], ylim=[self.plt_yllim, self.plt_yulim])
            ip.draw()
            ip.draw("4plot")
            ip.draw("5plot")
        return

    def _make_movie(self):
        """
        Make movies out of the plots
        """
        if self.verbose: print("\n Very slow method to plot and create movies.")
        _m = MM(self.rad, self.dates, self.scans, "raw", sim_id=self.sim_id, 
                xlim=[self.plt_xllim, self.plt_xulim], ylim=[self.plt_yllim, self.plt_yulim])
        _m.exe({"gflg_type":self.gflg_type})
        _m.figure_name = "med_filt"
        _m.scans = self.fscans
        _m.exe({"thresh":self.thresh, "gs":"gflg_conv", "pth":self.pth, "pbnd":self.pbnd, "zparam":"p_l"})
        _m.exe({"thresh":self.thresh, "gs":"gflg_kde", "pth":self.pth, "pbnd":self.pbnd, "zparam":"p_l"})
        return


if __name__ == "__main__":
    sim = Simulate("sas", [dt.datetime(2015,3,17,3,2), dt.datetime(2015,3,17,3,2)], 
            {"stype":"themis", "beam":6, "dur":2.}, inv_plot=True, dofilter=True, 
            gflg_type=-1, skills=True, save=True)
    import os
    os.system("rm *.log")
    os.system("rm -rf __pycache__")
