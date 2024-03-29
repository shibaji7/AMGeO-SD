#!/usr/bin/env python

"""clustering_algo.py: module is dedicated to run all clusteing analysis and classification."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.append("tools/")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import stats, ndimage
from loguru import logger
import traceback
import json
import cv2
from scipy.optimize import curve_fit
import traceback
import copy

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import boxcox

import utils
from fit_records import fetch_print_fit_rec
from get_fit_data import FetchData
from boxcar_filter import Filter

from plotlib import *
from matplotlib.dates import date2num
import multiprocessing as mp

import time
from netCDF4 import Dataset, date2num

from string import ascii_uppercase
from fan_rti import RangeTimeIntervalPlot as RTI, FanPlots
from matplotlib.dates import date2num
        


def get_mean_sza(dn, mbeam, mgate, rad="bks"):
    import rad_fov
    from pysolar.solar import get_altitude
    import pydarn
    hdw = pydarn.read_hdw_file(rad)
    rfov = rad_fov.CalcFov(hdw=hdw, altitude=300.)
    lat, lon = rfov.latFull[mbeam, mgate], rfov.lonFull[mbeam, mgate]
    dn = dn.replace(tzinfo=dt.timezone.utc)
    sza = np.round(90. - get_altitude(lat, lon, dn), 2)
    return sza

class BeamGateTimeFilter(object):
    """ Class to filter middle latitude radars """
    
    def __init__(self, rad, _dict_={}, cores=16):
        self.rad = rad
        self._dict_ = _dict_
        self.eps = self._dict_["eps"]
        self.min_samples = self._dict_["min_samples"]
        self.beam_gate_clusters_file = utils.get_config("BASE_DIR") + self._dict_["sim_id"] + "/" + self.rad +\
                    "/beamgate/beam_gate_clusters.csv"
        self.beam_gate_time_tracker_file = utils.get_config("BASE_DIR") + self._dict_["sim_id"] + "/" + self.rad +\
                    "/beam_gate_time_tracker.json"
        self.movie_start, self.movie_end = self._dict_["start"], self._dict_["end"]
        self.cores = cores
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        
        self.out_dir = utils.get_config("BASE_DIR") + _dict_["sim_id"] + "/" + self.rad + "/"
        self.parse_out_dic()
        return
    
    def parse_out_dic(self):
        """
        Parse output directory and folders
        """
        self.out = {"clustered": True}
        self.out["out_dir"] = self.out_dir
        self.out["out_files"] = {"h5_file": {"template":None, "files":[]}, "nc_file": {"template":None, "files":[]}}
        self.out["out_files"]["h5_file"]["template"] = self.out_dir + "data_{s}_{e}_%s.h5"\
            .format(s=self._dict_["start"].strftime("%Y%m%d.%H%M"),
                    e=self._dict_["end"].strftime("%Y%m%d.%H%M"))
        self.out["out_files"]["nc_file"]["template"] = self.out_dir + "data_{s}_{e}_%s.nc"\
            .format(s=self._dict_["start"].strftime("%Y%m%d.%H%M"),
                    e=self._dict_["end"].strftime("%Y%m%d.%H%M"))
        
        start, end = self._dict_["start"], self._dict_["end"]
        self.out["out_files"]["h5_file_master"] = self.out_dir + "data_{s}_{e}.h5".format(s=start.strftime("%Y%m%d.%H%M"),
                                                                                          e=end.strftime("%Y%m%d.%H%M"))
        self.out["out_files"]["nc_file_master"] = self.out_dir + "data_{s}_{e}.nc".format(s=start.strftime("%Y%m%d.%H%M"),
                                                                                          e=end.strftime("%Y%m%d.%H%M"))
        return
    
    def create_movie(self, fps=1):
        """ Create movie from static image to track """
        files = []
        start, end = self.movie_start, self.movie_end
        dn, dur = start, self._dict_["dur"]
        pathOut = utils.get_config("BASE_DIR") + self._dict_["sim_id"] + "/" + self.rad + "/01_beam_gate_time_tracker.avi"
        while dn < end:
            start, ed = dn, dn + dt.timedelta(minutes=dur)
            fname = utils.get_config("BASE_DIR") + self._dict_["sim_id"] + "/" + self.rad +\
                        "/beamgate/figs/%s_%s/"%(start.strftime("%Y-%m-%d-%H-%M"), ed.strftime("%Y-%m-%d-%H-%M")) +\
                        "04_gate_boundary_track.png"
            if os.path.exists(fname): files.append(fname)
            dn = dn + dt.timedelta(minutes=dur)
        frame_array = [cv2.imread(f) for f in files]
        height, width, layers = frame_array[0].shape
        size = (width,height)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for frame in frame_array:
            out.write(frame)
        out.release()
        return
    
    def run_bgc_algo(self):
        s_params=["bmnum", "noise.sky", "tfreq", "scan", "nrang", "time", "intt.sc", "intt.us", "mppul"]
        fv_params=["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad"]
        v_params=["v", "w_l", "gflg", "p_l", "slist", "v_e"]
        self.filter = Filter(thresh=self.thresh, pbnd=[self.lth, self.uth], pth=self.pth, verbose=self.verbose)
        self.scan_map = {}
        if self._dict_["start"] and not os.path.exists(self.beam_gate_clusters_file):
            rec_list = []
            _dict_ = copy.copy(self._dict_)
            start, end_date = self._dict_["start"], self._dict_["end"]
            dn, dur = start, self._dict_["dur"]
            ti = 0
            p = mp.Pool(self.cores)
            start_scnum = 0
            while dn < end_date:
                self.scan_info = fetch_print_fit_rec(self.rad, dn, dn + dt.timedelta(minutes=5), file_type=self._dict_["ftype"])
                start = dn - dt.timedelta(minutes=self.scan_info["s_time"])
                end = min(dn + dt.timedelta(minutes=dur) + dt.timedelta(minutes=2*self.scan_info["s_time"]),
                          end_date + dt.timedelta(minutes=2*self.scan_info["s_time"]))
                io = FetchData( self.rad, [start, end], ftype=self._dict_["ftype"], verbose=self._dict_["verbose"])
                _, scans = io.fetch_data(by="scan", scan_prop=self.scan_info)
                nscans, fscans = [], []
                logger.info(f" Into median filtering (cores): {self.cores}")
                for i in range(1,len(scans)-2):
                    nscans.append(scans[i-1:i+2])
                for sn in p.map(self.filter.doFilter, nscans):
                    fscans.append(sn)
                fD = io.scans_to_pandas(fscans, s_params, fv_params, start_scnum)
                D = io.scans_to_pandas(scans, s_params, v_params, start_scnum)
                self.scan_map[dn] = (fD, D)
                _dict_["start"], _dict_["end"] = dn, dn + dt.timedelta(minutes=dur)
                bgc = BeamGateFilter(self.rad, scans, ti, eps=self.eps, min_samples=self.min_samples, _dict_=_dict_)
                bgc.doFilter(io, themis=self.scan_info["t_beam"])
                dn = dn + dt.timedelta(minutes=dur)
                ti += 1
                rec_list.extend(bgc.recs)
                plt.close()
                start_scnum += len(fscans)
            self.recdf = pd.DataFrame.from_records(rec_list)
            self.recdf.to_csv(self.beam_gate_clusters_file, header=True, index=False)
        else: 
            start, end = self._dict_["start"], self._dict_["end"]
            dn, dur = start, self._dict_["dur"]
            self.scan_info = fetch_print_fit_rec(self.rad, dn, dn + dt.timedelta(minutes=5), file_type=self._dict_["ftype"])
            self.recdf = pd.read_csv(self.beam_gate_clusters_file)
        #self.create_movie()
        return self
    
    def compare_box(self, box_range, box_point):
        """ Check two boxes are linked or not """
        chk = False
        if box_range[0] < box_point[0] < box_range[1] and box_range[2] < box_point[1] < box_range[3]: chk = True
        return chk
    
    def run_time_track_algo(self):
        """ Tracking the temporal variations of the clusters """
        ti_max = np.max(self.recdf.time_intv) + 1
        bucket = self.recdf.to_dict("records")
        clusters, i = {}, 0
        while len(bucket) > 0:
            clusters[i] = []
            clusters[i].append(bucket[0])
            ti_ini = bucket[0]["time_intv"] + 1
            box_range = bucket[0]["beam_low"], bucket[0]["beam_high"], bucket[0]["gate_low"], bucket[0]["gate_high"]
            box_point = bucket[0]["mean_beam"], bucket[0]["mean_gate"]
            bucket.remove(bucket[0])
            for ti in range(ti_ini, ti_max):
                udf = self.recdf[self.recdf.time_intv==ti]
                is_break = True
                for _, ux in udf.iterrows():
                    now_box_range = ux["beam_low"], ux["beam_high"], ux["gate_low"], ux["gate_high"]
                    now_box_point = ux["mean_beam"], ux["mean_gate"]
                    if self.compare_box(now_box_range, box_point) and\
                            self.compare_box(box_range, now_box_point):
                        d = ux.to_dict()
                        if d in bucket:
                            clusters[i].append(d)
                            box_range, box_point = now_box_range, now_box_point
                            bucket.remove(d)
                            is_break = False
                            break
                if is_break: break
            i += 1
        with open(self.beam_gate_time_tracker_file, "w") as f: f.write(json.dumps(clusters, sort_keys=True, indent=4))
        return self.scan_info
    
    def save_data(self):
        v_params=["v", "w_l", "gflg", "p_l", "slist", "v_e"]
        if os.path.exists(self.beam_gate_time_tracker_file) and len(self.scan_map.keys()) > 0:
            logger.info(" Save all cluster data.")
            with open(self.beam_gate_time_tracker_file, "r") as f: clusters_obj = json.loads("\n".join(f.readlines()))
            clusters = clusters_obj.keys()
            h5fname = self.out["out_files"]["h5_file"]["template"]
            for j, c in enumerate(clusters):
                objs = clusters_obj[c]
                start, end = dt.datetime.strptime(objs[0]["start"], "%Y-%m-%dT%H:%M:%S"),\
                    dt.datetime.strptime(objs[-1]["end"], "%Y-%m-%dT%H:%M:%S")
                fDb, Db = pd.DataFrame(), pd.DataFrame()
                for i, o in enumerate(objs):
                    start, end = dt.datetime.strptime(o["start"], "%Y-%m-%dT%H:%M:%S"),\
                    dt.datetime.strptime(o["end"], "%Y-%m-%dT%H:%M:%S")
                    gate_high, gate_low = o["gate_high"], o["gate_low"]
                    beam_high, beam_low = o["beam_high"], o["beam_low"]
                    (fD, D) = self.scan_map[start]
                    fD = fD[(fD.slist>=gate_low) & (fD.slist<=gate_high) & (fD.bmnum>=beam_low) & (fD.bmnum<=beam_high)]
                    D = D[(D.slist>=gate_low) & (D.slist<=gate_high) & (D.bmnum>=beam_low) & (D.bmnum<=beam_high)]
                    for p in v_params:
                        fD["fitacf_"+p] = D[p]
                    fDb, Db = pd.concat([fDb, fD]), pd.concat([Db, D])
                fDb["cluster_tag"], Db["cluster_tag"] = int(c), int(c)
                fDb.to_hdf(h5fname%("f%02d"%int(c)), key="df")
                Db.to_hdf(h5fname%("d%02d"%int(c)), key="df")
                self.out["out_files"]["h5_file"]["files"].append(h5fname%("f%02d"%int(c)))
                self.out["out_files"]["h5_file"]["files"].append(h5fname%("d%02d"%int(c)))
        else: logger.error(" File not found - ", self.beam_gate_time_tracker_file)
        return
    
    def repackage_to_netcdf(self):
        s_params=["bmnum", "intt.sc", "intt.us", "mppul", "noise.sky", "nrang", "scan", "scnum", "tfreq", "time"]
        v_params=["p_l", "v", "v_mad", "w_l", "fitacf_v", "fitacf_w_l", "fitacf_gflg", "fitacf_p_l", "slist", 
                    "fitacf_slist", "fitacf_v_e", "cluster_tag", "ribiero_gflg", "gflg", "gflg_conv", "gflg_kde"]
        if os.path.exists(self.beam_gate_time_tracker_file):
            h5fname = self.out["out_files"]["h5_file"]["template"]
            with open(self.beam_gate_time_tracker_file, "r") as f: clusters_obj = json.loads("\n".join(f.readlines()))
            clusters = clusters_obj.keys()
            o = pd.DataFrame()
            for j, c in enumerate(clusters):
                fname = h5fname%("f%02d"%int(c))
                o = pd.concat([o, pd.read_hdf(fname, key="df", parse_dates=["time"])])
            o = o.dropna()
            o.cluster_tag = np.array(o.cluster_tag).astype(int)
            o = utils._run_riberio_threshold_on_rad(o)
            o.gflg_conv, o.gflg_kde = o.ribiero_gflg, o.ribiero_gflg
            io = FetchData( self.rad, [self._dict_["start"], self._dict_["end"]], ftype=self._dict_["ftype"],
                           verbose=self._dict_["verbose"])
            fscans = io.pandas_to_scans(o, self.scan_info["s_mode"], s_params, v_params)
            
            ##########################
            # To HDF5 files
            ##########################
            fname = self.out["out_files"]["h5_file_master"]
            o.to_hdf(fname, key="df")
            os.system("gzip " + fname)
            
            
            ##########################
            # To NC files
            ##########################
            s_params=["bmnum", "intt.sc", "intt.us", "mppul", "noise.sky", "nrang", "scan", "scnum", "tfreq", "time"]
            v_params=["p_l", "v", "v_mad", "w_l", "slist", "cluster_tag", "ribiero_gflg", "gflg", "gflg_conv", "gflg_kde"]
            _du = {key: [] for key in v_params + s_params}
            blen, glen = 0, 110
            for fscan in fscans[1:-1]:
                blen += len(fscan.beams)
                for b in fscan.beams:
                    for p in v_params:
                        _du[p].append(getattr(b, p))
                    for p in s_params:
                        _du[p].append(getattr(b, p))

            if self.verbose: logger.info(f"Shape of the filtered beam dataset [beams X gates]- {blen},{glen}")
            fname = self.out["out_files"]["nc_file_master"]
            rootgrp = Dataset(fname, "w", format="NETCDF4")
            rootgrp.description = """
                                     Fitacf++ : Boxcar filtered data.
                                     Filter parameter: weight matrix - default; threshold - {th}; GS Flag Type - {gflg_type}
                                     Parameters (Thresholds) - (thresh:{th}, gflg_type:{gflg_type})
                                  """.format(th=self.thresh, gflg_type="Ribiero")
            rootgrp.history = "Created " + time.ctime(time.time())
            rootgrp.source = "AMGeO - SD data processing"
            rootgrp.beam_per_scan = len(fscans[0].beams)
            rootgrp.createDimension("nbeam", blen)
            rootgrp.createDimension("ngate", glen)
            beam = rootgrp.createVariable("nbeam","i1",("nbeam",))
            gate = rootgrp.createVariable("ngate","i1",("ngate",))
            beam[:], gate[:] = range(blen), range(glen)
            times = rootgrp.createVariable("time", "f8", ("nbeam",))
            times.units = "hours since 1970-01-01 00:00:00.0"
            times.calendar = "julian"
            times[:] = date2num(_du["time"],units=times.units,calendar=times.calendar)
            s_params, type_params, desc_params = ["bmnum","noise.sky", "tfreq", "scan", "nrang", "intt.sc", "intt.us", "mppul", "scnum"],\
                    ["i1","f4","f4","i1","f4","f4","f4","i1","i1"],\
                    ["Beam numbers", "Sky Noise", "Frequency", "Scan Flag", "Max. Range Gate", "Integration sec", "Integration u.sec",
                            "Number of pulses", "Scan number"]
            for _i, k in enumerate(s_params):
                tmp = rootgrp.createVariable(k, type_params[_i],("nbeam",))
                tmp.description = desc_params[_i]
                tmp[:] = np.array(_du[k])
            v_params, desc_params = ["v", "w_l", "gflg", "p_l", "slist", "gflg_conv", "gflg_kde", "v_mad", "cluster_tag", "ribiero_gflg"],\
                    ["LoS Velocity (+ve: towards the radar, -ve: away from radar)", "LoS Width",
                            "GS Flag (%d)"%self.gflg_type, "LoS Power", "Gate", "GS Flag (Conv)", "GS Flag (KDE)",
                            "MAD of LoS Velociy after filtering", "Cluster ID", "Ribiero Flag"]
            _g = _du["slist"]
            for _i, k in enumerate(v_params):
                tmp = rootgrp.createVariable(k, "f4", ("nbeam", "ngate"))
                tmp.description = desc_params[_i]
                _m = np.empty((blen,glen))
                _m[:], x = np.nan, _du[k]
                for _j in range(blen):
                    _m[_j,np.array(_g[_j]).astype(int)] = np.array(x[_j])
                tmp[:] = _m

            rootgrp.close()
            os.system("gzip " + fname)
            ##########################################
        else: logger.error(" File not found - ", self.beam_gate_time_tracker_file)
        return
    
    
class BeamGateFilter(object):
    """ Class to filter middle latitude radars """

    def __init__(self, rad, scans, time_intv, eps=2, min_samples=10, _dict_={}):
        """
        initialize variables
        
        rad: Radar code
        scans: List of scans
        eps: Radius of DBSCAN
        min_samples: min samplese of DBSCAN
        """
        logger.info("BeamGate Cluster")
        self.rad = rad
        self.time_intv = time_intv
        filt = Filter()
        self.scans = [filt._discard_repeting_beams(s) for s in scans]
        self.eps = eps
        self.min_samples = min_samples
        self.boundaries = {}
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self.fig_folder = utils.get_config("BASE_DIR") + self.sim_id + "/" + self.rad +\
                    "/beamgate/figs/%s_%s/"%(self.start.strftime("%Y-%m-%d-%H-%M"), self.end.strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.fig_folder): os.system("mkdir -p " + self.fig_folder)
        return

    def extract_gatelims(self, df):
        """
        Extract gate limits for individual clusters
        """
        glims = {}
        for l in set(df.labels):
            if l >= 0:
                u = df[df.labels==l]
                if len(u) > 0: glims[l] = [np.min(u.slist) + 1, np.max(u.slist) - 1]
        return glims

    def filter_by_dbscan(self, df, bm):
        """
        Do filter by dbscan name
        """
        du, sb = df[df.bmnum==bm], [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l), 
                "sound":sb[0], "echo":len(du), "v_mad": stats.median_absolute_deviation(du.v), 
                "p_mad": stats.median_absolute_deviation(du.p_l), "w_mad": stats.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all":
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else:
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        l = du.groupby("time")
        logger.info(f" Beam Analysis:  {bm}, {len(l)}")
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        tfreq = np.round(np.mean(du.tfreq/1e3), 2)
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        eco, Eco = utils.smooth(eco, self.window), utils.smooth(Eco, self.window)
        glims, labels = {}, []
        ds = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(du[["slist"]].values)
        du["labels"] = ds.labels_
        names = {}
        for j, r in enumerate(set(ds.labels_)):
            x = du[du.labels==r]
            glims[r] = [np.min(x.slist), np.max(x.slist)]
            names[r] = "C"+str(j)
            if r >= 0: self.boundaries[bm].append({"peak": Rng[np.min(x.slist) + Eco[np.min(x.slist):np.max(x.slist)+1].argmax()],
                "ub": np.max(x.slist), "lb": np.min(x.slist), "value": np.max(eco)*xpt, "bmnum": bm, "echo": len(x), "sound": sb[0],
                "tfreq": tfreq})
        logger.info(f" Individual clster detected: {set(du.labels)}")
        loc_folder = self.fig_folder + "gate_summary/"
        if not os.path.exists(loc_folder): os.system("mkdir -p " + loc_folder)
        fig_file = loc_folder + "gate_{bm}_summary.png".format(bm="%02d"%bm)
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco), fig_file, sb)
        return

    def filter_by_SMF(self, df, bm, method=None):
        """
        Filter by Simple Minded Filter
        method: np/ndimage
        """
        def local_minima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True): 
            """Find all local maxima of the array, separated by at least min_distance."""
            array = np.asarray(array)
            cval = 0 
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest" 
            else: mode = "constant" 
            cval = array.max()+1 
            min_points = array == ndimage.minimum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
            troughs = [indices[min_points] for indices in np.indices(array.shape)][0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] <= np.max(du.slist) - min_distance: troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        def local_maxima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True):
            array = np.asarray(array)
            cval = 0
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest"
            else: mode = "constant"
            cval = array.max()+1
            max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval)
            peaks = [indices[max_points] for indices in np.indices(array.shape)][0]
            return peaks

        def local_minima_np(array):
            """ Local minima by numpy stats """
            troughs = signal.argrelmin(eco, order=self.order)[0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] != np.max(du.slist): troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        if bm == "all": du = df.copy()
        else: du = df[df.bmnum==bm]
        l = len(du.groupby("time"))
        logger.info(f" Beam Analysis:  {bm}, {l}")
        sb = [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l),
                "sound":sb[0], "echo":len(du), "v_mad": stats.median_absolute_deviation(du.v),
                "p_mad": stats.median_absolute_deviation(du.p_l), "w_mad": stats.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all": 
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else: 
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        tfreq = np.round(np.mean(du.tfreq/1e3), 2)
        glims, labels = {}, []
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        eco, Eco = utils.smooth(eco, self.window), utils.smooth(Eco, self.window)
        du["labels"] = [np.nan] * len(du)
        names = {}
        if method == "np": troughs = local_minima_np(np.array(eco))
        else: troughs = local_minima_ndimage(np.array(eco), min_distance=5)
        peaks = local_maxima_ndimage(np.array(eco), min_distance=5)
        logger.info(f" Gate bounds (T,P): {troughs}, {peaks}")
        peaks = np.append(peaks, np.median(troughs[-2:]))
        #if len(peaks) + 1 < len(troughs): troughs = troughs[:len(peaks) + 1]
        logger.info(f" Gate bounds adjustment (T,P): {troughs}, {peaks}")
        for r in range(len(troughs)-1):
            glims[r] = [troughs[r], troughs[r+1]]
            du["labels"] = np.where((du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r]), r, du["labels"])
            names[r] = "C" + str(r)
            if r >= 0: 
                self.boundaries[bm].append({"peak": peaks[r], "ub": troughs[r+1], "lb": troughs[r], 
                                            "value": np.max(eco)*xpt, "bmnum": bm, 
                                            "echo": len(du[(du["slist"]<=troughs[r+1]) 
                                                           & (du["slist"]>=troughs[r])]), "sound": sb[0],
                                           "tfreq":tfreq})

        du["labels"] =  np.where(np.isnan(du["labels"]), -1, du["labels"])
        logger.info(f" Individual clster detected:  {set(du.labels)}")
        loc_folder = self.fig_folder + "gate_summary/"
        if not os.path.exists(loc_folder): os.system("mkdir -p " + loc_folder)
        fig_file = loc_folder + "gate_{bm}_summary.png".format(bm="%02d"%bm)
        to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco), fig_file, sb)
        return

    def doFilter(self, io, window=11, order=1, beams=range(4,24), themis=15):
        """
        Do filter for sub-auroral scatter
        """
        self.gen_summary = {}
        self.clust_idf = {}
        self.order = order
        self.window = window
        df = pd.DataFrame()
        for i, fsc in enumerate(self.scans):
            dx = io.convert_to_pandas(fsc.beams, v_params=["p_l", "v", "w_l", "slist"])
            df = df.append(dx)
        logger.info(f" Beam range -  {np.min(df.bmnum)}, {np.max(df.bmnum)}")
        if beams == None or len(beams) == 0: 
            bm = "all"
            self.boundaries[bm] = []
            self.filter_by_SMF(df, bm)
        else:
            for bm in beams:
                self.boundaries[bm] = []
                self.gen_summary[bm] = {}
                if bm==themis: 
                    try:
                        #self.filter_by_SMF(df, bm)
                        self.filter_by_dbscan(df, bm)
                    except:
                        logger.error(f"SMF failed for this themis beam {themis}")
                        traceback.print_exc()
                        self.filter_by_dbscan(df, bm)
                else: self.filter_by_dbscan(df, bm)
        title = "Date: %s [%s-%s] UT | %s"%(df.time.tolist()[0].strftime("%Y-%m-%d"),
                df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M"), self.rad.upper())
        self.gc = []
        for x in self.boundaries.keys():
            for j, m in enumerate(self.boundaries[x]):
                self.gc.append(m)
        self.ubeam, self.lbeam = np.max(df.bmnum), np.min(df.bmnum)
        self.sma_bgspace()
        self.probabilistic_cluster_identification(df)
        fname = self.fig_folder + "01_gate_boundary.png"
        beam_gate_boundary_plots(self.boundaries, self.clusters, None, 
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname)
        fname = self.fig_folder + "03_gate_boundary_id.png"
        beam_gate_boundary_plots(self.boundaries, self.clusters, self.clust_idf,
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname, gflg_type=self.gflg_type)
        fname = self.fig_folder + "05_general_stats.png"
        general_stats(self.gen_summary, fname=fname)
        self.curves = {}
        for c in self.clusters.keys():
            cluster = self.clusters[c]
            try:
                if len(self.clusters[c]) > 3: self.curves[c] = self.fit_curves(cluster)
            except:
                try:
                    self.curves[c] = self.fit_curves(cluster, "parabola")
                except: logger.error(f"Error in fit_curves(line), cluster_stats, Cluster #{c}")
                logger.error(f"Error in fit_curves(parabola), cluster_stats, Cluster #{c}")
            try:
                loc_folder = self.fig_folder + "clust_stats/"
                if not os.path.exists(loc_folder): os.system("mkdir " + loc_folder)
                fname =  loc_folder + "clust_%02d_stats.png"%c
                cluster_stats(df, cluster, fname, title+" | Cluster# %02d"%c)
            except:
                logger.error(f"Error in cluster stats, cluster_stats, Cluster #{c}")
            try:
                loc_folder = self.fig_folder + "ind_clust_stats/"
                if not os.path.exists(loc_folder): os.system("mkdir " + loc_folder)
                fname =  loc_folder + "ind_clust_%02d_stats.png"%c
                individal_cluster_stats(self.clusters[c], df, fname, 
                    title+" | Cluster# %02d"%c+"\n"+"Cluster ID: _%s_"%self.clust_idf[c].upper())
            except:
                logger.error(f"Error in cluster stats, individal_cluster_stats, Cluster #{c}")
        self.save_cluster_info(df.copy())
        return
    
    def fit_curves(self, cluster, curve="line"):
        """ Fit a parabola through beam versus gate edges """
        
        dats = {}
        for cx in cluster:
            if cx["bmnum"] not in dats.keys(): dats[cx["bmnum"]] = {"beams": cx["bmnum"], "ub": cx["ub"], "lb": cx["lb"]}
            else:
                dats[cx["bmnum"]]["ub"] = cx["ub"] if dats[cx["bmnum"]]["ub"] < cx["ub"] else dats[cx["bmnum"]]["ub"]
                dats[cx["bmnum"]]["lb"] = cx["lb"] if dats[cx["bmnum"]]["lb"] > cx["lb"] else dats[cx["bmnum"]]["lb"]
        beams, ubs, lbs = np.sort([k for k in dats.keys()]), np.sort([dats[k]["ub"] for k in dats.keys()]),\
                            np.sort([dats[k]["lb"] for k in dats.keys()])
        if curve=="parabola": func = lambda x, ac, bc: ac*np.sqrt(x)-bc
        elif curve=="line": func = lambda x, ac, bc: ac + bc*x
        popt_ubs, _ = curve_fit(func, beams, ubs)
        popt_lbs, _ = curve_fit(func, beams, lbs)
        popt_ubs, popt_lbs = np.round(popt_ubs, 2), np.round(popt_lbs, 2)
        return {"beams": beams, "p_ubs": popt_ubs, "p_lbs": popt_lbs, "curve": curve}

    def save_cluster_info(self, df):
        fname = self.fig_folder + "02_cluster_info.csv"
        recs = []
        for c in self.clusters.keys():
            if len(self.clusters[c]) > 3:
                Cm = {"start":self.start.strftime("%Y-%m-%dT%H:%M:%S"), "end": self.end.strftime("%Y-%m-%dT%H:%M:%S"),
                      "time_intv":self.time_intv, "cluster":c, "beam_low":np.nan, "beam_high":np.nan,
                      "gate_low":np.nan, "gate_high":np.nan, "mean_beam":np.nan, "mean_gate":np.nan, 
                      "mean_tfreq":np.nan, "mean_sza": np.nan}
                mbeam, mgate = 0, 0
                beamdiv = 0
                for i, u in enumerate(self.clusters[c]):
                    mbeam += u["bmnum"]*(u["ub"]-u["lb"])
                    beamdiv += int(u["ub"]-u["lb"])
                    if i == 0: 
                        Cm["beam_low"], Cm["beam_high"] = u["bmnum"], u["bmnum"]
                        Cm["gate_low"], Cm["gate_high"] = u["lb"], u["ub"]
                        Cm["mean_tfreq"] = u["tfreq"]
                    else:
                        Cm["beam_low"] = u["bmnum"] if u["bmnum"] < Cm["beam_low"] else Cm["beam_low"]
                        Cm["beam_high"] = u["bmnum"] if u["bmnum"] > Cm["beam_high"] else Cm["beam_high"]
                        Cm["gate_low"] = u["lb"] if u["lb"] < Cm["gate_low"] else Cm["gate_low"]
                        Cm["gate_high"] = u["ub"] if u["ub"] > Cm["gate_high"] else Cm["gate_high"]
                mgate = int((Cm["gate_low"]+Cm["gate_high"])/2)
                Cm["mean_gate"], Cm["mean_beam"] = mgate, int(mbeam/beamdiv)
                Cm["mean_sza"] = get_mean_sza(df.time.tolist()[int(len(df)/2)], Cm["mean_beam"], Cm["mean_gate"], self.rad)
                recs.append(Cm)
        with open(fname, "w") as f:
            start, end = self.start.strftime("%Y-%m-%d %H:%M"), self.end.strftime("%Y-%m-%d %H:%M")
            f.write(f"# Header line: Timeline - {start},{end}; # Cluster - {len(self.clusters.keys())}\n")
        pd.DataFrame.from_records(recs).to_csv(fname, mode="a", header=True, index=False)
        self.recs = recs
        title = "Date: %s [%s-%s] UT | %s"%(df.time.tolist()[0].strftime("%Y-%m-%d"),
                df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M"), self.rad.upper())
        fname = self.fig_folder + "04_gate_boundary_track.png"
        beam_gate_boundary_tracker(recs, None, glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title, fname=fname)
        fname = self.fig_folder + "06_gate_boundary_track.png"
        #beam_gate_boundary_tracker(recs, self.curves, glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
        #        fname=fname)
        return
    
    def cluster_identification(self, df, qntl=[0.05,0.95]):
        """ Idenitify the cluster based on Idenitification """
        for c in self.clusters.keys():
            V = []
            for cls in self.clusters[c]:
                v = np.array(df[(df.slist>=cls["lb"]) & (df.slist<=cls["ub"]) & (df.bmnum==cls["bmnum"])].v)
                V.extend(v.tolist())
            V = np.array(V)
            l, u = np.quantile(V,qntl[0]), np.quantile(V,qntl[1])
            Vmin, Vmax = np.min(V[(V>l) & (V<u)]), np.max(V[(V>l) & (V<u)])
            Vmean, Vmed = np.mean(V[(V>l) & (V<u)]), np.median(V[(V>l) & (V<u)])
            Vsig = np.std(V[(V>l) & (V<u)])
            self.clust_idf[c] = "us"
            if Vmin > -20 and Vmax < 20: self.clust_idf[c] = "gs"
            elif (Vmin > -50 and Vmax < 50) and (Vmed-Vsig < -20 or Vmed+Vsig > 20): self.clust_idf[c] = "sais"
        return
    
    def probabilistic_cluster_identification(self, df):
        """ Idenitify the cluster based on Idenitification """
        df["cluster_id"] = [np.nan] * len(df)
        prob_dctor = ScatterTypeDetection()
        for c in self.clusters.keys():
            for cls in self.clusters[c]:
                df["cluster_id"] = np.where((df.slist<=cls["lb"]) & (df.slist<=cls["ub"]) & (df.bmnum==cls["bmnum"]),
                                            c, df.cluster_id)
            du = df[df["cluster_id"]==c]
            prob_dctor.copy(du)
            du, prob_clusters = prob_dctor.run(thresh=[self.lth, self.uth], pth=self.pth)
            auc = prob_clusters[self.gflg_type][c]["auc"]
            if prob_clusters[self.gflg_type][c]["type"] == "IS": auc = 1-auc
            if prob_clusters[self.gflg_type][c]["type"] == "US": self.clust_idf[c] = "us"
            else: self.clust_idf[c] = "%.1f"%(auc*100) + "%" + prob_clusters[self.gflg_type][c]["type"].lower()
        return

    def sma_bgspace(self):
        """
        Simple minded algorithm in B.G space
        """
        def range_comp(x, y, pcnt=0.7):
            _cx = False
            insc = set(x).intersection(y)
            if len(x) < len(y) and len(insc) >= len(x)*pcnt: _cx = True
            if len(x) > len(y) and len(insc) >= len(y)*pcnt: _cx = True
            return _cx
        def find_adjucent(lst, mxx):
            mxl = []
            for l in lst:
                if l["peak"] >= mxx["lb"] and l["peak"] <= mxx["ub"]: mxl.append(l)
                elif mxx["peak"] >= l["lb"] and mxx["peak"] <= l["ub"]: mxl.append(l)
                elif range_comp(range(l["lb"], l["ub"]+1), range(mxx["lb"], mxx["ub"]+1)): mxl.append(l)
            return mxl
        def nested_cluster_find(bm, mx, j, case=-1):
            if bm < self.lbeam and bm > self.ubeam: return
            else:
                if (case == -1 and bm >= self.lbeam) or (case == 1 and bm <= self.ubeam):
                    mxl = find_adjucent(self.boundaries[bm], mx)
                    for m in mxl:
                        if m in self.gc:
                            del self.gc[self.gc.index(m)]
                            self.clusters[j].append(m)
                            nested_cluster_find(m["bmnum"] + case, m, j, case)
                            nested_cluster_find(m["bmnum"] + (-1*case), m, j, (-1*case))
                return
        self.clusters = {}
        j = 0
        while len(self.gc) > 0:
            self.clusters[j] = []
            mx = max(self.gc, key=lambda x:x["value"])
            self.clusters[j].append(mx)
            if mx in self.gc: del self.gc[self.gc.index(mx)]
            nested_cluster_find(mx["bmnum"] - 1, mx, j, case=-1)
            nested_cluster_find(mx["bmnum"] + 1, mx, j, case=1)
            j += 1
        return
    
class ScatterTypeDetection(object):
    """ Detecting scatter type """

    def __init__(self):
        """ kind: 0- individual, 2- KDE by grouping """
        return
    
    def copy(self, df):
        self.df = df.fillna(0)
        return

    def run(self, cases=[0,1,2], thresh=[1./3.,2./3.], pth=0.5):
        self.pth = pth
        self.thresh = thresh
        self.clusters = {}
        for cs in cases:
            self.case = cs
            self.clusters[self.case] = {}
            for kind in range(2):
                if kind == 0: 
                    self.indp()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                if kind == 1: 
                    self.kde()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                    self.df["proba_%d"%(cs)] = self.proba
        return self.df.copy(), self.clusters

    def kde_beam(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings("ignore", "The iteration is not making good progress")
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        self.proba = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            self.clusters[self.case][bm] = {}
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams) 
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                    if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                    if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                    ## KDE part
                    f[f==0.] = np.random.uniform(0.001, 0.01, size=len(f[f==0.]))
                    f[f==1.] = np.random.uniform(0.95, 0.99, size=len(f[f==1.]))
                    try:
                        a, b, loc, scale = beta.fit(f, floc=0., fscale=1.)
                        auc = 1 - beta.cdf(self.pth, a, b, loc=0., scale=1.)
                        if np.isnan(auc): auc = np.mean(f)
                        #print(" KDE Estimating AUC: %.2f"%auc, c)
                    except:
                        auc = np.mean(f)
                        #print(" Estimating AUC: %.2f"%auc, c)
                    if auc < self.thresh[0]: gflg, ty = 0., "IS"
                    elif auc >= self.thresh[1]: gflg, ty = 1., "GS"
                    else: gflg, ty = -1, "US"
                    self.gs_flg[clust_mask] = gflg
                    self.proba[clust_mask] = f
                    self.clusters[self.case][bm][c] = {"auc": auc, "type": ty}
        return
    
    def kde(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings("ignore", "The iteration is not making good progress")
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        self.proba = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1: self.gs_flg[clust_mask] = -1
            else:
                v, w = vel[clust_mask], wid[clust_mask]
                if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                ## KDE part
                f[f==0.] = np.random.uniform(0.001, 0.01, size=len(f[f==0.]))
                f[f==1.] = np.random.uniform(0.95, 0.99, size=len(f[f==1.]))
                try:
                    a, b, loc, scale = beta.fit(f, floc=0., fscale=1.)
                    auc = 1 - beta.cdf(self.pth, a, b, loc=0., scale=1.)
                    if np.isnan(auc): auc = np.mean(f)
                    #print(" KDE Estimating AUC: %.2f"%auc, c)
                except:
                    auc = np.mean(f)
                    #print(" Estimating AUC: %.2f"%auc, c)
                if auc < self.thresh[0]: gflg, ty = 0., "IS"
                elif auc >= self.thresh[1]: gflg, ty = 1., "GS"
                else: gflg, ty = -1, "US"
                self.gs_flg[clust_mask] = gflg
                self.proba[clust_mask] = f
                self.clusters[self.case][c] = {"auc": auc, "type": ty}
        return

    def indp(self):
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["cluster_id"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams)
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    gflg = np.zeros(len(v))
                    if self.case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
                    if self.case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
                    if self.case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
                    self.gs_flg[clust_mask] = gflg
        return

class DBScan(object):
    """ Class to cluster 2D or 3D data """
    
    def __init__(self, args):
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        self.s_params = ["bmnum", "intt.sc", "intt.us", "mppul", "noise.sky", "nrang", "scan", "tfreq", "time", "channel"]
        self.v_params = ["p_l", "v", "w_l", "slist", "gflg"]
        self.out_dir = utils.get_config("BASE_DIR") + self.sim_id + "/" + self.rad + "/"    
        self.atrs, self.scan_info = self.update_vals()
        self.io = FetchData( self.rad, [self.start, self.end], ftype=self.ftype, verbose=self.verbose)
        self.beams, self.scans = self.io.fetch_data(by="scan", scan_prop=self.scan_info)
        self.frame = self.io.scans_to_pandas(self.scans, self.s_params, self.v_params)
        if self.docluster: self.run_2D_cluster_bybeam()
        return
    
    def update_vals(self):
        calgo = "GMM (%s) - DBSCAN"%self.run_gmm
        atrs = {"type":"sub-auroral", "gflg_type":"us_c", 
                "gflg_type_desc":"Generated by clustering algo (%s)."%calgo}
        atrs.update(self.__dict__)
        dtime = (self.end - self.start).total_seconds()/2 
        dn = self.start + dt.timedelta(seconds=dtime)
        scan_info = fetch_print_fit_rec(self.rad, dn, dn + dt.timedelta(minutes=5))
        atrs.update(scan_info)
        for k in atrs.keys():
            if type(atrs[k]) == bool: atrs[k] = int(atrs[k])
            if type(atrs[k]) == list: atrs[k] = "-".join(atrs[k])
            if type(atrs[k]) == dt.datetime: atrs[k] = atrs[k].strftime("%Y-%m-%dT%H:%M:%S")
            if atrs[k] is None: atrs[k] = ""
        return atrs, scan_info
    
    def plot_rti_images(self, _o, b):
        _o["mdates"] = _o.time.apply(lambda x: date2num(x))
        rti = RTI(100, unique_times=np.unique(_o.mdates), fig_title="")
        rti.addParamPlot(_o, b, "", xlabel="")
        rti.addCluster(_o, b, "")
        rti.addGSIS(_o, b, "", xlabel="Time, [UT]")
        rti.save(self.out_dir + "%02d.png"%b)
        rti.close()
        return
    
    def gmm_on_existing_cluster(self, data, features=["v"]):
        if len(data) >= self.n_data_th:
            estimator = GaussianMixture(n_components=self.n_clusters,
                                        covariance_type=self.cov, max_iter=100, reg_covar=1e-3,
                                        random_state=0, n_init=5, init_params="kmeans")
            estimator.fit(self.get_gmm_box_cox(data, features)[features])
            clust_flg = estimator.predict(data[features])
        else: clust_flg = np.copy(data.cluster_tag)
        return clust_flg
    
    def get_gmm_box_cox(self, data, features):
        for feature in features:
            vals = np.copy(data[feature])
            vals[np.abs(vals)<=1e-5] = 1e-5
            vals = boxcox(np.abs(vals))[0]
            vals[np.isnan(vals)] = 0.
            vals[np.isinf(vals)] = 0.
            data[feature] = vals
        return data
    
    def run_dbscan(self, _o, b):
        _o.slist, _o.scnum, eps, min_samples = _o.slist/self.eps_g, _o.scnum/self.eps_s, self.eps, self.min_samples
        if self.scan_info["s_mode"] != "normal" and self.scan_info["t_beam"] == b: min_samples = self.min_samples*self.tb_mult
        ds = DBSCAN(eps=self.eps, min_samples=min_samples).fit(_o[["slist", "scnum"]].values)
        labels = np.copy(ds.labels_)
        return labels
    
    def run_gmm_dbscan(self, _o, b):
        logger.info(" Running DBSCAN - GMM for beams: %02d"%(b))
        
#         _o = self.run_dbscan(_o, b)
#         labels = np.copy(_o.cluster_tag)
#         for ct in np.unique(labels):
#             gmm_labels = self.gmm_on_existing_cluster((_o[_o.cluster_tag==ct]).copy())
#             gmm_labels += (np.max(labels) + 1)
#             labels[labels==ct] = gmm_labels
#         _o["cluster_tag"] = labels
        
        _o["cluster_tag"] = self.gmm_on_existing_cluster(_o.copy())
        labels = np.copy(_o.cluster_tag)
        db_labels = np.zeros_like(labels) * np.nan
        dclust = np.max(labels) + 1
        for ct in np.unique(labels):
            ls = self.run_dbscan(_o[_o.cluster_tag==ct], b)
            ls[ls>=0] += dclust
            db_labels[labels==ct] = ls
        return db_labels
    
    def run_2D_cluster_bybeam(self):
        o = pd.DataFrame()
        self.v_params = self.v_params + ["cluster_tag", "gflg_ribiero"]
        self.s_params = self.s_params + ["scnum"]
        gmm_beams = []
        if self.run_gmm:
            if "A" in self.run_gmm.upper(): gmm_beams = np.unique(self.frame.bmnum)
            elif "T" in self.run_gmm.upper(): gmm_beams = [self.scan_info["t_beam"]]
            else: gmm_beams = [int(x) for x in self.run_gmm.split(",")]
        if self.scan_info["s_mode"] != "normal": 
            logger.info(f"Radar mode - {self.scan_info['s_mode']} on beam - {self.scan_info['t_beam']}")
        for ch in np.unique(self.frame.channel):
            frame = self.frame[self.frame.channel==ch]
            for b in np.unique(frame.bmnum):
                _o = frame[frame.bmnum==b]
                if b in gmm_beams: _o["cluster_tag"] = self.run_gmm_dbscan(_o, b)
                else: _o["cluster_tag"] = self.run_dbscan(_o, b)
                _o = utils._run_riberio_threshold_on_rad(_o, vth=self.velocity_threhold)
                self.plot_rti_images(_o, b)
                o = pd.concat([o, _o])
        o = o.sort_values("time")
        self.plot_fov_images(o, [dt.datetime(2015,3,17,4,34), dt.datetime(2015,3,17,4,36), dt.datetime(2015,3,17,4,38)])
        o["gflg_conv"], o["gflg_kde"] = o.gflg_ribiero, o.gflg_ribiero
        if self.save:
            scans, bmax = self.io.pandas_to_scans(o, self.scan_info["s_mode"], self.s_params, self.v_params)
            fov = utils.geolocate_radar_fov(self.rad, [s.stime for s in scans], azm=True)
            ds = utils.to_xarray(scans, fov, bmax, self.atrs)
            ds.to_netcdf(self.out_dir + "%s_%s_%s.fitacf++.nc"%(self.rad, self.start.strftime("%Y%m%d%H%M"),
                                                                self.end.strftime("%Y%m%d%H%M")))
        return
    
    def plot_fov_images(self, o, dns):
        fov = FanPlots(figsize=(15,10), nrows=2, ncols=3)
        for i, dn in enumerate(dns):
            ax = fov.plot_fov(o[(o.time>=dn) & (o.time < dn + dt.timedelta(minutes=self.scan_info["s_time"]))], 
                              dn, self.rad, gflg_mask="gflg", add_colorbar=False)
            if i==0: ax.grid_on(draw_labels=[False, False, True, True])
            ax.enum(dtype=r"$fitacf$")
        for i, dn in enumerate(dns):
            if i==2: ax = fov.plot_fov(o[(o.time>=dn) & (o.time < dn + dt.timedelta(minutes=self.scan_info["s_time"]))], 
                                       dn, self.rad, gflg_mask="gflg_ribiero")
            else: ax = fov.plot_fov(o[(o.time>=dn) & (o.time < dn + dt.timedelta(minutes=self.scan_info["s_time"]))], 
                                    dn, self.rad, gflg_mask="gflg_ribiero", add_colorbar=False)
            ax.grid_on(draw_labels=[False, False, False, False])
            ax.enum(dtype=r"$fitacf^{++}$", add_coord=False, add_date=False)
        fov.fig.subplots_adjust(hspace=0.2)
        fov.save(self.out_dir + "%s_fov.png"%dn.strftime("%Y-%m-%d %H:%M"))
        fov.close()
        return
    
    def run_med_filt(self):
        self.v_params = self.v_params + ["v_mad", "gflg_conv", "gflg_kde"]
        return