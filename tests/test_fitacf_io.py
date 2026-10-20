"""Tests for amgeo_sd.io.fitacf: in-memory records first, real files from tests/data after."""

import datetime as dt

import numpy as np
import pytest

from amgeo_sd.io import (
    FitacfReader,
    beams_to_dataframe,
    beams_to_scans,
    find_fitacf_files,
    read_fitacf_file,
    records_to_beams,
    scans_to_dataframe,
)


def make_record(t: dt.datetime, bmnum: int, scan: int = 0, slist=(3, 4, 7), **over) -> dict:
    n = len(slist)
    rec = {
        "time.yr": t.year,
        "time.mo": t.month,
        "time.dy": t.day,
        "time.hr": t.hour,
        "time.mt": t.minute,
        "time.sc": t.second,
        "time.us": t.microsecond,
        "bmnum": bmnum,
        "scan": scan,
        "tfreq": 10500,
        "noise.sky": 2.5,
        "nrang": 75,
        "channel": 1,
        "cp": 3,
        "rsep": 45,
        "frang": 180,
        "smsep": 300,
        "lagfr": 1200,
        "mppul": 8,
        "intt.sc": 3,
        "intt.us": 0,
    }
    if n:
        rec.update(
            slist=np.array(slist, dtype=np.int16),
            v=np.linspace(-50, 50, n, dtype=np.float32),
            v_e=np.ones(n, np.float32),
            w_l=np.full(n, 20, np.float32),
            p_l=np.full(n, 10, np.float32),
            gflg=np.zeros(n, np.int8),
        )
    rec.update(over)
    return rec


T0 = dt.datetime(2015, 3, 17, 2, 0, 0)


def three_scans(beams_per_scan=3):
    recs = []
    for s in range(3):
        for b in range(beams_per_scan):
            recs.append(make_record(T0 + dt.timedelta(minutes=s, seconds=b * 3), b, scan=int(b == 0)))
    return recs


def test_record_to_beam_fields():
    beam = records_to_beams([make_record(T0, 5, scan=-1)])[0]
    assert beam.time == T0 and beam.bmnum == 5
    assert beam.scan == 1  # abnormal start (-1) folded to 1
    assert beam.noise_sky == 2.5 and beam.intt_sc == 3 and beam.tfreq == 10500
    assert beam.slist.tolist() == [3, 4, 7] and beam.v.shape == beam.gflg.shape == (3,)


def test_beam_without_scatter_is_kept_with_empty_arrays():
    beam = records_to_beams([make_record(T0, 2, slist=())])[0]
    assert beam.n_echoes == 0 and beam.v.size == 0


def test_missing_optional_fields_are_filled():
    rec = make_record(T0, 0)
    del rec["v_e"], rec["gflg"]
    beam = records_to_beams([rec])[0]
    assert np.isnan(beam.v_e).all() and (beam.gflg == -1).all()


def test_length_mismatch_raises():
    rec = make_record(T0, 0)
    rec["v"] = np.zeros(2, np.float32)
    with pytest.raises(ValueError, match="'v'"):
        records_to_beams([rec])


def test_time_window_is_inclusive():
    recs = [make_record(T0 + dt.timedelta(seconds=3 * i), 0) for i in range(5)]
    beams = records_to_beams(recs, T0 + dt.timedelta(seconds=3), T0 + dt.timedelta(seconds=9))
    assert [b.time for b in beams] == [T0 + dt.timedelta(seconds=s) for s in (3, 6, 9)]


def test_scan_splitting():
    scans = beams_to_scans(records_to_beams(three_scans(4)))
    assert [len(s.beams) for s in scans] == [4, 4, 4]
    assert scans[1].stime == T0 + dt.timedelta(minutes=1)
    assert scans[1].etime == T0 + dt.timedelta(minutes=1, seconds=9)
    assert scans[0].freq == 10500


def test_scan_flag_repeated_at_same_time_does_not_split():
    recs = [
        make_record(T0, 0, scan=1, channel=1),
        make_record(T0, 0, scan=1, channel=2),
        make_record(T0 + dt.timedelta(seconds=3), 1),
    ]
    assert len(beams_to_scans(records_to_beams(recs))) == 1


def test_beams_to_scans_empty():
    assert beams_to_scans([]) == []


def test_dataframes():
    scans = beams_to_scans(records_to_beams(three_scans(2)))
    df = scans_to_dataframe(scans)
    assert len(df) == 3 * 2 * 3  # scans * beams * echoes
    assert sorted(df.scnum.unique()) == [0, 1, 2] and sorted(df.sbnum.unique()) == [0, 1]
    assert df.time.dtype.kind == "M"
    assert {"slist", "v", "w_l", "p_l", "gflg", "bmnum", "noise_sky"} <= set(df.columns)


def test_dataframe_extra_columns_and_empty_beams():
    beams = records_to_beams([make_record(T0, 0), make_record(T0, 1, slist=())])
    beams[0].extra["v_mad"] = np.arange(3.0)
    df = beams_to_dataframe(beams, extra=["v_mad"])
    assert df.v_mad.tolist() == [0.0, 1.0, 2.0]
    assert beams_to_dataframe([]).empty


def test_dataframe_dtypes_survive_beams_without_echoes():
    """An echo-less beam used to give an untyped empty frame that turned all columns to object on concat."""
    recs = [
        make_record(T0, 0, scan=1),
        make_record(T0 + dt.timedelta(seconds=3), 1, slist=()),
        make_record(T0 + dt.timedelta(seconds=6), 2),
    ]
    df = scans_to_dataframe(beams_to_scans(records_to_beams(recs)))
    assert df["v"].dtype == np.float64 and df["slist"].dtype == np.int64 and df["time"].dtype.kind == "M"
    assert df["gflg"].dtype == np.int8 and len(df) == 6
    empty = scans_to_dataframe(beams_to_scans(records_to_beams([make_record(T0, 0, slist=())])))
    assert empty.empty and empty["v"].dtype == np.float64 and empty["scnum"].dtype == np.int64


def test_find_files_by_name_and_overlap(tmp_path):
    d = tmp_path / "2015" / "fitacf" / "bks"
    d.mkdir(parents=True)
    names = [
        "20150316.2300.00.bks.a.fitacf.bz2",  # runs 23:00-01:00, overlaps day start
        "20150317.0100.00.bks.a.fitacf.bz2",  # 01:00-03:00
        "20150317.0300.00.bks.a.fitacf.bz2",  # 03:00-05:00
        "20150317.0500.00.bks.a.fitacf.bz2",
    ]  # after the window
    for n in names:
        (d / n).touch()
    got = find_fitacf_files("bks", dt.datetime(2015, 3, 17, 2), dt.datetime(2015, 3, 17, 4), root=tmp_path)
    assert [p.name for p in got] == names[1:3]


@pytest.mark.parametrize("suffix", ["", ".bz2", ".gz"])
def test_read_file_decompresses_before_decoding(tmp_path, monkeypatch, suffix):
    import bz2
    import gzip

    import pydarnio

    seen = {}
    monkeypatch.setattr(pydarnio, "read_fitacf", lambda b, mode: (seen.setdefault("b", b), ([{"x": 1}], None))[1])
    path = tmp_path / ("20150317.0200.00.bks.a.fitacf" + suffix)
    opener = {"": open, ".bz2": bz2.open, ".gz": gzip.open}[suffix]
    with opener(path, "wb") as fp:
        fp.write(b"payload")
    assert read_fitacf_file(path) == [{"x": 1}]
    assert seen["b"] == b"payload"


def test_archive_lookup_skips_fitacfpp_files(tmp_path):
    d = tmp_path / "2015" / "fitacf" / "bks"
    d.mkdir(parents=True)
    (d / "20150317.0200.00.bks.a.fitacf.bz2").touch()
    (d / "20150317.0200.00.bks.fitacf.pp.bz2").touch()
    (d / "20150317.0200.00.bks.fitacf.pl.bz2").touch()  # earlier name of the same files
    got = find_fitacf_files("bks", dt.datetime(2015, 3, 17, 2), dt.datetime(2015, 3, 17, 3), root=tmp_path)
    assert [p.name for p in got] == ["20150317.0200.00.bks.a.fitacf.bz2"]


def test_find_files_needs_root(monkeypatch):
    monkeypatch.delenv("AMGEO_SD_DATA", raising=False)
    with pytest.raises(ValueError, match="AMGEO_SD_DATA"):
        find_fitacf_files("bks", T0, T0)


# ---- real data -------------------------------------------------------------------


@pytest.mark.realdata
def test_real_file_decodes(fitacf_files):
    records = read_fitacf_file(fitacf_files[0])
    assert records, "no records decoded"
    beams = records_to_beams(records)
    assert len(beams) == len(records)
    times = [b.time for b in beams]
    assert times == sorted(times)
    for b in beams:
        assert 0 <= b.bmnum < 32 and b.nrang > 0
        assert b.slist.size == b.v.size == b.w_l.size == b.p_l.size == b.gflg.size
        if b.n_echoes:
            assert b.slist.min() >= 0 and b.slist.max() < b.nrang


@pytest.mark.realdata
def test_real_file_scans_and_dataframe(fitacf_files):
    beams = records_to_beams(read_fitacf_file(fitacf_files[0]))
    scans = beams_to_scans(beams)
    assert sum(len(s.beams) for s in scans) == len(beams)
    assert len(scans) >= 2
    assert all(s.stime <= s.etime for s in scans)
    df = scans_to_dataframe(scans)
    assert len(df) == sum(b.n_echoes for b in beams)
    assert df.scnum.max() == len(scans) - 1


@pytest.mark.realdata
def test_real_reader_window(fitacf_files):
    beams = records_to_beams(read_fitacf_file(fitacf_files[0]))
    t0 = beams[0].time
    reader = FitacfReader("any", t0, t0 + dt.timedelta(minutes=30), files=[fitacf_files[0]])
    got = reader.beams()
    assert got and all(t0 <= b.time <= t0 + dt.timedelta(minutes=30) for b in got)


@pytest.mark.realdata
def test_real_full_day_scans_are_complete(fitacf_files):
    """Every radar-day in tests/data: scans have a constant beam count and unique beam numbers."""
    from collections import defaultdict

    by_dir = defaultdict(list)
    for f in fitacf_files:
        by_dir[f.parent].append(f)
    for files in by_dir.values():
        reader = FitacfReader("any", dt.datetime(2000, 1, 1), dt.datetime(2100, 1, 1), files=files)
        scans = reader.scans()
        sizes = {len(s.beams) for s in scans}
        assert len(sizes) == 1, f"{files[0].parent.name}: uneven scan sizes {sizes}"
        assert all(len({b.bmnum for b in s.beams}) == len(s.beams) for s in scans)
        assert all((s.etime - s.stime).total_seconds() < 180 for s in scans)


@pytest.mark.realdata
def test_cli_info(fitacf_files, capsys):
    from amgeo_sd.cli import main

    assert main(["info", str(fitacf_files[0].parent)]) == 0
    out = capsys.readouterr().out
    assert "scans" in out and "echoes" in out
