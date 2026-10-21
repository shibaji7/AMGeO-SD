"""Reading SuperDARN data products."""

from amgeo_sd.io.fitacf import (
    LEGACY_FITACFPP_SUFFIXES,
    FitacfReader,
    beams_to_dataframe,
    beams_to_scans,
    find_fitacf_files,
    read_fitacf_file,
    records_to_beams,
    scans_to_dataframe,
)
from amgeo_sd.io.load import expand_paths, load_frame, load_scans, read_fitacfpp_nc
from amgeo_sd.io.summary import estimate_scan_info
from amgeo_sd.io.writer import FITACFPP_SUFFIX, beam_to_record, fitacfpp_name, scans_to_records, write_fitacfpp

__all__ = [
    "FitacfReader",
    "find_fitacf_files",
    "read_fitacf_file",
    "records_to_beams",
    "beams_to_scans",
    "estimate_scan_info",
    "beams_to_dataframe",
    "scans_to_dataframe",
    "expand_paths",
    "load_scans",
    "load_frame",
    "read_fitacfpp_nc",
    "beam_to_record",
    "scans_to_records",
    "fitacfpp_name",
    "write_fitacfpp",
    "FITACFPP_SUFFIX",
    "LEGACY_FITACFPP_SUFFIXES",
]
