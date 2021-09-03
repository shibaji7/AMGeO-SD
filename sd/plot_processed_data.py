#!/usr/bin/env python

"""plot_processed_data.py: module is dedicated to utility functions for the plotting processed data."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import glob
from loguru import logger
import datetime as dt
import json
import copy
import numpy as np
from netCDF4 import Dataset, num2date
import pandas as pd

import utils
import plotlib
from get_fit_data import Scan, Beam
from plotlib import InvestigativePlots as IP, MovieMaker as MM

def date_hook(_dict_):
    for (key, value) in _dict_.items():
        try:
            _dict_[key] = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except: pass
    return _dict_

class PlotProcessedData(object):
    """Class to fetch processed oata and create movies"""
    
    def __init__(self, param_file, rad=None, date_range=None, sim_id=None, scan_prop=None, cid=None, verbose=True):
        """
        initialize all variables
        """
        self.verbose = verbose
        self.cid = cid
        if self.check(param_file, rad, date_range, sim_id, scan_prop):
            logger.error("Either 'param_file' is not None otherwise (rad, date_range, sim_id, scan_prop) not None")
            raise Exception("Either 'param_file' is not None otherwise (rad, date_range, sim_id, scan_prop) not None")
        if param_file:
            with open(param_file, "r") as o: loaded_dict = json.loads("".join(o.readlines()), object_hook=date_hook)
            self.rad = loaded_dict["rad"]
            self.date_range = [loaded_dict["start"], loaded_dict["end"]]
            self.sim_id = loaded_dict["sim_id"]
            self.scan_prop = {"rad": loaded_dict["rad"], "s_time": loaded_dict["s_time"], 
                              "t_beam": loaded_dict["t_beam"], "s_mode": loaded_dict["s_mode"]}
            self.loaded_dict = loaded_dict
            for p in loaded_dict.keys():
                setattr(self, p, loaded_dict[p])
        else:
            self.rad = rad
            self.date_range = date_range
            self.scan_prop = scan_prop
            self.sim_id = sim_id
        self.out = {"inv_plots": {"1plot": [], "4plot": [], "5plot": []}, "movies": []}
        if ("clustered" in self.loaded_dict.keys()) and self.loaded_dict["clustered"]:
            self._create_dates()
            self._fetch_h5()
            self.frame2scan()
        else:
            self._fetch_data()
            self._to_scans()
        self._invst_plots()
        self._make_movie()
        return
    
    def check(self, param_file, rad, date_range, sim_id, scan_prop):
        """
        Check all the parameters
        """
        if self.verbose: logger.info("Checking parameters")
        if param_file is not None: return False
        if (rad is not None) and (date_range is not None) and (sim_id is not None) and (scan_prop is not None): return False
        else: return True
        
    def _create_dates(self):
        """
        Create all dates to run plotting method
        """
        if self.verbose: logger.info("Creating time sequence")
        self.drange = [self.date_range[0] - dt.timedelta(minutes=self.scan_prop["s_time"]),
                self.date_range[1] + dt.timedelta(minutes=self.scan_prop["s_time"])]
        self.dates = []
        dates = (self.date_range[1] + dt.timedelta(minutes=self.scan_prop["s_time"]) -
                self.date_range[0]).total_seconds() / (self.scan_prop["s_time"] * 60)
        for d in range(int(dates)):
            self.dates.append(self.date_range[0] + dt.timedelta(minutes=d*self.scan_prop["s_time"]))
        return
    
    def _fetch_data(self):
        """
        Fetch the data inside the folder
        """
        if hasattr(self, "loaded_dict") and hasattr(self.loaded_dict, "out_files") and\
            hasattr(self.loaded_dict["out_files"], "nc_file"): gzfn = self.loaded_dict["out_files"]["nc_file"]
        else: gzfn = utils.build_base_folder_structure(self.rad, self.sim_id) + "*.nc.gz"
        if self.verbose: logger.info(f"Fetching data from NetCDF file - {gzfn}")
        gzfns = glob.glob(gzfn)
        gzfns.sort()
        self.datastore = []
        for gf in gzfns:
            os.system("gzip -d " + gf)
            self.datastore.append(Dataset(gf.replace(".gz","")))
            os.system("gzip " + gf.replace(".gz",""))
        return
    
    def _fetch_h5(self):
        """
        Fetch the data inside the folder
        """
        self.datastore = {"d":{}, "f":{}}
        if hasattr(self, "loaded_dict") and ("out_files" in self.loaded_dict.keys()) and\
            ("h5_file" in self.loaded_dict["out_files"].keys()) and\
            ("files" in self.loaded_dict["out_files"]["h5_file"].keys()) and\
            (len(self.loaded_dict["out_files"]["h5_file"]["files"]) > 0): 
            files = self.loaded_dict["out_files"]["h5_file"]["files"]
            for f in files:
                cluster = int(f.split("_")[-1].replace(".h5", "")[1:])
                if self.verbose: logger.info(f"Fetching data from HDF {cluster} file - {f}")
                
                self.datastore[f.split("_")[-1][0]][cluster] = pd.read_hdf(f, "df", parse_dates=["time"])
        else: logger.error(f"No HDF5 (.h5) files to plot.")
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
        bps = self.datastore[0].beam_per_scan
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
        return d, bps

    def _to_scans(self):
        """
        Convert to scans and filter scan
        """
        if self.verbose: logger.info("Converting datasets to scans")
        gscans = []
        prefxs = ["fitacf_", ""]
        vparams = [["v", "w_l", "gflg", "p_l", "v_e"], ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]]
        s_params = ["time", "bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"]
        for pref, v_params in zip(prefxs, vparams):
            _b, _s = [], []
            d, bps = self._get_nc_obj(s_params, v_params, pref)
            for i, time in enumerate(d["time"]):
                bm = Beam()
                bm.set_nc(time, d, i, s_params, v_params)
                _b.append(bm)
            for j in range(int(len(_b)/bps)):
                sc = Scan(None, None, self.scan_prop["s_mode"])
                sc.beams.extend(_b[j*bps:(j+1)*(bps-1)])
                sc.update_time()
                _s.append(sc)
            gscans.append(_s)
        self.scans, self.fscans = gscans[0], gscans[1]
        return
    
    def frame2scan(self):
        """
        Convert to scans and filter scan from h5 files
        """
        self.dates = []
        if self.cid is None: self.cid = 0 
        fD, D = self.datastore["f"][self.cid], self.datastore["d"][self.cid]
        fg, g = fD.groupby(["scnum"]), D.groupby(["scnum"])
        if self.verbose: logger.info("Converting datasets to scans")
        s_params = ["time", "bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul"]
        self.scans = []
        v_params = ["v", "w_l", "gflg", "p_l", "slist", "v_e"]
        mapper = {v_params[i]: v_params[i].replace("fitacf_", "") for i in range(len(v_params))}
        for _, o in g:
            _b, ox = [], o.groupby(["bmnum"])
            for _, x in ox:
                m = x[s_params+v_params]
                d = m.rename(columns=mapper).to_dict("list")
                bm = Beam()
                bm.set(d["time"][0], d, s_params, v_params, 0)
                _b.append(bm)
            sc = Scan(None, None, self.scan_prop["s_mode"])
            sc.beams = _b
            sc.update_time()
            self.scans.append(sc)
        v_params = ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]
        self.fscans = []
        for n, o in fg:
            _b, ox = [], o.groupby(["bmnum"])
            for _, x in ox:
                m = x[s_params+v_params]
                d = m.to_dict("list")
                bm = Beam()
                bm.set(d["time"][0], d, s_params, v_params, 0)
                _b.append(bm)
            sc = Scan(None, None, self.scan_prop["s_mode"])
            sc.beams = _b
            sc.update_time()
            self.fscans.append(sc)
            self.dates.append(sc.stime)
        logger.info(f"Total lengths - {len(self.fscans)}")
        return

    def _invst_plots(self):
        """
        Method for investigative plots -
        1. Plot 1 X 1 panel plots
        2. Plot 2 X 2 panel plots
        3. Plot 3 - 2 filter analysis plots
        """
        if self.verbose: logger.info("Slow method to plot 3 different types of investigative plots.")
        xlim = [int(utils.get_config("X_Low", "figures.invst_plots")), int(utils.get_config("X_Upp", "figures.invst_plots"))]
        ylim = [int(utils.get_config("Y_Low", "figures.invst_plots")), int(utils.get_config("Y_Upp", "figures.invst_plots"))]
        folder = utils.build_base_folder_structure(self.rad, self.sim_id) + "inv_plots/"
        logger.info(f"Lengths L(dates,scans,fscans)-({len(self.dates)}, {len(self.scans)}, {len(self.fscans)})")
        logger.info(f"Lengths Start,end-({self.dates[0]}, {self.dates[-1]})")
        logger.info(f"Lengths Start,end~(scans)-({self.scans[0].stime}, {self.scans[-1].etime})")
        logger.info(f"Lengths Start,end~(fscans)-({self.fscans[0].stime}, {self.fscans[-1].etime})")
        #for i, e in enumerate(self.dates[:-1]):
        for i in range(1,len(self.scans)-2):
            e = self.fscans[i-1].stime
            scans = self.scans[i-1:i+2]
            scans.append(self.fscans[i-1])
            ip = IP("1plot", e, self.rad, self.sim_id, scans, 
                    {"thresh": self.loaded_dict["thresh"], "gs": self.loaded_dict["gflg_cast"], 
                     "pth": self.loaded_dict["pth"], "pbnd": [self.loaded_dict["lth"], self.loaded_dict["uth"]],
                     "gflg_type": self.gflg_type, "cid":self.cid}, xlim=xlim, ylim=ylim, folder=folder)
            self.out["inv_plots"]["1plot"].append(ip.draw())
            self.out["inv_plots"]["4plot"].append(ip.draw("4plot"))
            self.out["inv_plots"]["5plot"].append(ip.draw("5plot"))
        return

    def _make_movie(self):
        """
        Make movies out of the plots
        """
        if self.verbose: logger.info("Very slow method to plot and create movies.")
        xlim = [int(utils.get_config("X_Low", "figures.invst_plots")), int(utils.get_config("X_Upp", "figures.invst_plots"))]
        ylim = [int(utils.get_config("Y_Low", "figures.invst_plots")), int(utils.get_config("Y_Upp", "figures.invst_plots"))]
        folder = utils.build_base_folder_structure(self.rad, self.sim_id) + "movies/"
        if os.path.exists(folder): os.system("rm -rf {folder}*".format(folder=folder))
        _m = MM(self.rad, self.dates, self.scans, "raw", sim_id=self.sim_id, xlim=xlim, ylim=ylim, folder=folder)
        self.out["movies"].append(_m.exe({"gflg_type": self.loaded_dict["gflg_type"]}))
        _m.figure_name = "med_filt"
        _m.scans = self.fscans
        self.out["movies"].append(_m.exe({"thresh": self.loaded_dict["thresh"], "gs": "gflg_conv", "pth": self.loaded_dict["pth"], 
                "pbnd": [self.loaded_dict["lth"], self.loaded_dict["uth"]], "zparam":"p_l"}))
        self.out["movies"].append(_m.exe({"thresh": self.loaded_dict["thresh"], "gs": "gflg_kde", "pth": self.loaded_dict["pth"],
                "pbnd": [self.loaded_dict["lth"], self.loaded_dict["uth"]], "zparam":"p_l"}))
        return