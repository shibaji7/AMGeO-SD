#!/usr/bin/env python

"""fit_records.py: module is dedicated to fetch fitacf data from files."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Kunduri, B.; Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

# standard libs
import datetime
import sys
sys.path.append("sd/")
# SD libs
import pydarn
import rad_fov
import geoPack
import aacgmv2
# Data Analysis libs
import pandas as pd
from loguru import logger

from get_fit_data import FetchData


class PrintFitRec(object):
    """ 
        Class to write Fit records and produce beam summary.
        
        Read fit level files and print the records into csv
        Parameters
        ----------
        start_date : datetime
            the start time as a datetime
        end_date : datetime
            the end time as a datetime
        rad : radar code string
            the 3 letter radar code, eg "bks"
        out_file_dir : str
            the txt file dir we are outputting to
        fileType : Optional[str]
            the filetype to read, "fitex","fitacf","lmfit", "fitacf3";

        Adapted from DaViTPy
    """
    
    def __init__(self, rad, start_date, end_date, file_type="fitacf3", out_file_dir="/tmp/"):
        """ Initialize all parameters """
        self.rad = rad
        self.start_date = start_date
        self.end_date = end_date
        self.file_type = file_type
        self.io = FetchData(rad, [start_date, end_date], ftype=file_type, verbose=True)
        self.out_file_dir = out_file_dir
        s_params = ["bmnum", "noise.sky", "tfreq", "scan", "nrang", "noise.search",
                    "xcf", "scan", "rsep", "frang", "channel", "cp", "intt.sc", "intt.us",
                    "mppul", "smsep", "lagfr"]
        v_params = ["v", "w_l", "gflg", "p_l", "slist", "pwr0", "v_e"]
        self.beams, _ = self.io.fetch_data(s_params=s_params, v_params=v_params)
        return
    
    def print_fit_record(self):
        """
        Print fit level records
        """
        fname = None
        if self.beams is not None and len(self.beams) > 0:
            fname = self.out_file_dir + "{rad}.{dn}.{start}.{end}.txt".format(rad=self.rad, dn=self.start_date.strftime("%Y%m%d"),
                                                                     start=self.start_date.strftime("%H%M"), 
                                                                              end=self.end_date.strftime("%H%M"))
            f = open(fname, "w")
            print("\n Working through: ", self.rad)
            hdw = pydarn.read_hdw_file(self.rad)
            fov_obj = rad_fov.CalcFov(hdw=hdw, rsep=self.beams[0].rsep,\
                                      ngates=self.beams[0].nrang, altitude=300.)
            for b in self.beams:
                f.write(b.time.strftime("%Y-%m-%d  "))
                f.write(b.time.strftime("%H:%M:%S  "))
                f.write(self.rad + " ")
                f.write(self.file_type + "\n")
                f.write("bmnum = " + str(b.bmnum))
                f.write("  tfreq = " + str(b.tfreq))
                f.write("  sky_noise_lev = " +
                        str(round(getattr(b, "noise.sky"))))
                f.write("  search_noise_lev = " +
                        str(round(getattr(b, "noise.search"))))
                f.write("  xcf = " + str(getattr(b, "xcf")))
                f.write("  scan = " + str(getattr(b, "scan")) + "\n")
                f.write("npnts = " + str(len(getattr(b, "slist"))))
                f.write("  nrang = " + str(getattr(b, "nrang")))
                f.write("  channel = " + str(getattr(b, "channel")))
                f.write("  cpid = " + str(getattr(b, "cp")) + "\n")
                
                # Write the table column header
                f.write("{0:>4s} {13:>5s} {1:>5s} / {2:<5s} {3:>8s} {4:>3s} "
                        "{5:>8s} {6:>8s} {7:>8s} {8:>8s} {9:>8s} {10:>8s} "
                        "{11:>8s} {12:>8s}\n".
                        format("gate", "pwr_0", "pwr_l", "vel", "gsf", "vel_err",
                               "width_l", "geo_lat", "geo_lon", "geo_azm",
                               "mag_lat", "mag_lon", "mag_azm", "range"))
                
                # Cycle through each range gate identified as having scatter in
                # the slist
                for i,s in enumerate(b.slist):
                    lat_full = fov_obj.latFull[b.bmnum]
                    lon_full = fov_obj.lonFull[b.bmnum]
                    
                    d = geoPack.calcDistPnt(lat_full[s], lon_full[s], 300,
                                            distLat=lat_full[s + 1],
                                            distLon=lon_full[s + 1],
                                            distAlt=300)
                    gazm = d["az"]
                    # get aacgm_coords
                    mlat, mlon, alt = aacgmv2.convert_latlon(lat_full[s],lon_full[s],300.,b.time,method_code="G2A")
                    mlat2, mlon2, alt2 = aacgmv2.convert_latlon(lat_full[s+1],lon_full[s+1],300.,b.time,method_code="G2A")
                    d = geoPack.calcDistPnt(mlat, mlon, 300, distLat=mlat2,
                                            distLon=mlon2, distAlt=300)
                    mazm = d["az"]
                    
                    f.write("{0:4d} {13:5d} {1:>5.1f} / {2:<5.1f} {3:>8.1f} "
                            "{4:>3d} {5:>8.1f} {6:>8.1f} {7:>8.2f} {8:>8.2f} "
                            "{9:>8.2f} {10:>8.2f} {11:>8.2f} {12:>8.2f}\n".
                            format(s, getattr(b, "pwr0")[i],
                                   getattr(b, "p_l")[i], getattr(b, "v")[i],
                                   getattr(b, "gflg")[i], getattr(b, "v_e")[i],
                                   getattr(b, "w_l")[i], lat_full[s], lon_full[s],
                                   gazm, mlat, mlon, mazm,  getattr(b, "frang") +
                                   s * getattr(b, "rsep")))
                f.write("\n")
        f.close()
        return {"fname": fname}

    def beam_summary(self, check_abnormal_scan=False, show_rec=10):
        """ Produce beam summary for beam sounding and scann mode """

        mode = "normal"
        def to_normal_scan_id(dx, key="scan"):
            """ Convert to normal scan code """
            sid = np.array(dx[key])
            sid[sid!=0] = 1
            dx[key] = sid
            return dx

        def _estimate_scan_duration(dx):
            """ Calculate scan durations in minutes """
            dx = dx[dx.scan==1]
            #dx = dx[dx.bmnum==dx.bmnum.tolist()[1]]
            sdur = (dx.time.tolist()[-1].to_pydatetime() - dx.time.tolist()[-2].to_pydatetime()).total_seconds()
            return int( (sdur+10)/60. )

        def _estimate_themis_beam(sdur, dx):
            """ Estimate themis mode and beam number if any """
            th = -1 #No themis mode
            if sdur > 1:
                dx = dx[dx.time < dx.time.tolist()[0].to_pydatetime() + datetime.timedelta(minutes=sdur)]
                lst = dx.bmnum.tolist()
                th = max(set(lst), key=lst.count)
            return th

        cols = ["time","channel","bmnum","scan","tfreq","frang","smsep","rsep","cp","nrang","mppul",
                "lagfr","intt.sc","intt.us","noise.sky"]
        _dict_ = {k: [] for k in cols}

        for b in self.beams:
            # sometimes there are some records without data!
            # discard those!
            if b.slist is None or len(b.slist) == 0 : continue
            _dict_["time"].append(b.time)
            for c in cols: 
                if c != "time": _dict_[c].append(getattr(b, c))
        d = pd.DataFrame.from_records(_dict_)
        logger.info(" Data source:")
        print(d[cols].head(show_rec))
        if check_abnormal_scan: d = to_normal_scan_id(d, key="scan")
        dur = _estimate_scan_duration(d)
        logger.info(f" Duration of scan: {dur} min(s)")
        themis = _estimate_themis_beam(dur, d)
        fname = self.out_file_dir + "{rad}.{dn}.{start}.{end}.csv".format(rad=self.rad, dn=self.start_date.strftime("%Y%m%d"),
                                                                     start=self.start_date.strftime("%H%M"), 
                                                                          end=self.end_date.strftime("%H%M"))
        d[cols].to_csv(fname, header=True, index=False)
        if themis >= 0: mode = "themis"
        return {"rad": self.rad, "s_time": dur, "t_beam": themis, "fname": fname, "s_mode": mode}
    
    
def fetch_print_fit_rec(rad, start_date, end_date, beam_summ=True, file_type="fitacf", out_file_dir="/tmp/", 
                        check_abnormal_scan=False, show_rec=10):
    """
        Parameters
        ----------
        start_date : datetime
            the start time as a datetime
        end_date : datetime
            the end time as a datetime
        rad : radar code string
            the 3 letter radar code, eg "bks"
        out_file_dir : str
            the txt file dir we are outputting to
        fileType : Optional[str]
            the filetype to read, "fitex","fitacf","lmfit", "fitacf3"
        beam_summ: Optional[str]
            Run fit rec or beam summary
    """
    _pfr_ = PrintFitRec(rad, start_date, end_date, file_type=file_type, out_file_dir=out_file_dir)
    if beam_summ: _dic_ = _pfr_.beam_summary(check_abnormal_scan, show_rec)
    else: _dic_ = _pfr_.print_fit_record()
    return _dic_

# Test scripts
if __name__ == "__main__":
    import datetime as dt
    fetch_print_fit_rec( "sas", dt.datetime(2015,3,17,3), dt.datetime(2015,3,17,3,5) )