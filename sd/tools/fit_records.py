# standard libs
import bz2
import glob
import datetime
import sys
import csv
# SD libs
import pydarnio
import pydarn
import rad_fov
import geoPack
import aacgmv2
# Data Analysis libs
import pandas
import numpy
# plotting libs
import datetime as dt

def _create_files(reg_ex, start, end):
    """
    Create file names from date and radar code
    """
    files = []
    days = (end - start).days + 1
    ent = -1
    for d in range(-1,days):
        e = start + dt.timedelta(days=d)
        fnames = glob.glob(reg_ex.format(year=e.year) + e.strftime("%Y%m%d") + "*")
        fnames.sort()
        for fname in fnames:
            tm = fname.split(".")[1]
            sc = fname.split(".")[2]
            dus = dt.datetime.strptime(fname.split(".")[0].split("/")[-1] + tm + sc, "%Y%m%d%H%M%S")
            due = dus + dt.timedelta(hours=2)
            if (ent == -1) and (dus <= start <= due): ent = 0
            if ent == 0: files.append(fname)
            if (ent == 0) and (dus <= end <= due): ent = -1
    return files

def print_fit_record(\
         start_date, end_date,\
         radar_list, base_dir="/sd-data/", out_file="/tmp/sd_list.txt", file_type="fitacf"\
         ):
    """
    Read fit level files and print the records into csv
    Parameters
    ----------
    start_date : datetime
        the start time as a datetime
    base_dir : Base directory where SDARN data is stored.
               Expect data in the format it is stored in
               the VT SuperDARN server! eg "/sd-data/"
               
    end_date : datetime
        the end time as a datetime
    radar_list : list of radar code strings
        the 3 letter radar code, eg "bks"
    outfile : str
        the txt file we are outputting to
    fileType : Optional[str]
        the filetype to read, "fitex","fitacf","lmfit", "fitacf3";
    
    Adapted from DaViTPy
    """
    f = open(out_file, "w")
    loop_date = start_date
    while loop_date <= end_date:
        _yr_dir = base_dir + str(loop_date.year) + "/" + file_type + "/"
        for _rad in radar_list:
            print("Working through: ",loop_date, _rad)
            hdw = pydarn.read_hdw_file(_rad)
            _curr_dir = _yr_dir + _rad + "/"
            # get the files
            #_flist = glob.glob(_curr_dir + loop_date.strftime("%Y%m%d") + "*")
            reg_ex = base_dir +  "{year}/" + file_type + "/" + _rad + "/"
            _flist = _create_files(reg_ex, start_date, end_date)
            if len(_flist) == 0:
                print("No files found! try another file type or date")
                continue
            for _f in _flist:
                with bz2.open(_f) as fp: 
                    fitacf_stream = fp.read()
                reader = pydarnio.SDarnRead(fitacf_stream, True)
                records = reader.read_fitacf()
                print("reading records from: ", _f)
                # ok note here that I"m using the first record
                # to calculate the fov! Need to check if this is ok?
                # Do we do it for every record? But that slows down
                # things considerably.
                fov_obj = rad_fov.CalcFov(\
                        hdw=hdw, rsep=records[0]["rsep"],\
                        ngates=records[0]["nrang"], altitude=300.\
                           )
                for _rec in records:
                    # sometimes there are some records without data!
                    # discard those!
                    if "slist" not in _rec:
                        continue
                    # print all of the range gate values.
                    t = datetime.datetime(
                            _rec["time.yr"], _rec["time.mo"],
                            _rec["time.dy"], _rec["time.hr"],
                            _rec["time.mt"], _rec["time.sc"]
                        )
                    if t >= start_date and t <= end_date:
                        f.write(t.strftime("%Y-%m-%d  "))
                        f.write(t.strftime("%H:%M:%S  "))
                        f.write(_rad + " ")
                        f.write(file_type + "\n")
                        f.write("bmnum = " + str(_rec["bmnum"]))
                        f.write("  tfreq = " + str(_rec["tfreq"]))
                        f.write("  sky_noise_lev = " +
                                str(int(round(float(_rec["noise.sky"])))))
                        f.write("  search_noise_lev = " +
                                str(int(round(float(_rec["noise.search"])))))
                        f.write("  xcf = " + str(_rec["xcf"]))
                        f.write("  scan = " + str(+_rec["scan"]) + "\n")
                        f.write("npnts = " + str(len(_rec["slist"])))
                        f.write("  nrang = " + str(_rec["nrang"]))
                        f.write("  channel = " + str(_rec["channel"]))
                        f.write("  cpid = " + str(_rec["cp"]) + "\n")
                        
                        # Write the table column header
                        f.write("{0:>4s} {13:>5s} {1:>5s} / {2:<5s} {3:>8s} {4:>3s} "
                                "{5:>8s} {6:>8s} {7:>8s} {8:>8s} {9:>8s} {10:>8s} "
                                "{11:>8s} {12:>8s}\n".
                                format("gate", "pwr_0", "pwr_l", "vel", "gsf", "vel_err",
                                       "width_l", "geo_lat", "geo_lon", "geo_azm",
                                       "mag_lat", "mag_lon", "mag_azm", "range"))
                        
                        # Cycle through each range gate identified as having scatter in
                        # the slist
                        for i,s in enumerate(_rec["slist"]):
                            lat_full = fov_obj.latFull[_rec["bmnum"]]
                            lon_full = fov_obj.lonFull[_rec["bmnum"]]
                            
                            d = geoPack.calcDistPnt(lat_full[s], lon_full[s], 300,
                                          distLat=lat_full[s + 1],
                                          distLon=lon_full[s + 1],
                                          distAlt=300)
                            gazm = d["az"]
                            # get aacgm_coords
                            mlat, mlon, alt = aacgmv2.convert_latlon(lat_full[s],lon_full[s],300.,t,method_code="G2A")
                            mlat2, mlon2, alt2 = aacgmv2.convert_latlon(lat_full[s+1],lon_full[s+1],300.,t,method_code="G2A")
                            d = geoPack.calcDistPnt(mlat, mlon, 300, distLat=mlat2,
                                                  distLon=mlon2, distAlt=300)
                            mazm = d["az"]
                            
                            f.write("{0:4d} {13:5d} {1:>5.1f} / {2:<5.1f} {3:>8.1f} "
                                    "{4:>3d} {5:>8.1f} {6:>8.1f} {7:>8.2f} {8:>8.2f} "
                                    "{9:>8.2f} {10:>8.2f} {11:>8.2f} {12:>8.2f}\n".
                                    format(s, _rec["pwr0"][i],
                                           _rec["p_l"][i], _rec["v"][i],
                                           _rec["gflg"][i], _rec["v_e"][i],
                                           _rec["w_l"][i], lat_full[s], lon_full[s],
                                           gazm, mlat, mlon, mazm,  _rec["frang"] +
                                           s * _rec["rsep"]))
                        f.write("\n")
        loop_date += datetime.timedelta(days=1)
    f.close()
    
#example
if __name__ == "__main__":
    start_date = datetime.datetime(2018,1,1)
    end_date = datetime.datetime(2018,1,1,1)
    base_dir = "/sd-data/"
    radar_list = ["bks"]
    out_file = "/tmp/bks_data.txt"
    
    print_fit_record(\
         start_date, end_date,\
         radar_list, base_dir,\
         out_file=out_file, file_type="fitacf"\
         )
    
    import os
    os.system("rm -rf *.log")
    os.system("rm -rf __pycache__/")
