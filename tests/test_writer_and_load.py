"""fitacf++ DMAP writer, multi-format loader and fitacf version handling."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from amgeo_sd.io import (
    FitacfReader,
    beams_to_scans,
    fitacfpp_name,
    load_frame,
    load_scans,
    read_fitacf_file,
    records_to_beams,
    scans_to_records,
    write_fitacfpp,
)
from amgeo_sd.pipelines import run_median_filter

DATA = Path(__file__).parent / "data"
T0 = dt.datetime(2025, 2, 8, 2, 10)


def synthetic_record(t, bmnum, slist=(3, 4), scan=0, rev=(3, 0)):
    n = len(slist)
    rec = {
        "time.yr": np.int16(t.year),
        "time.mo": np.int16(t.month),
        "time.dy": np.int16(t.day),
        "time.hr": np.int16(t.hour),
        "time.mt": np.int16(t.minute),
        "time.sc": np.int16(t.second),
        "time.us": np.int32(t.microsecond),
        "bmnum": np.int16(bmnum),
        "scan": np.int16(scan),
        "channel": np.int16(0),
        "cp": np.int16(151),
        "nrang": np.int16(75),
        "tfreq": np.int16(11000),
        "noise.sky": np.float32(2.0),
        "rsep": np.int16(45),
        "frang": np.int16(180),
        "fitacf.revision.major": np.int32(rev[0]),
        "fitacf.revision.minor": np.int32(rev[1]),
    }
    if n:
        rec.update(
            slist=np.array(slist, np.int16),
            v=np.linspace(-20, 20, n, dtype=np.float32),
            v_e=np.ones(n, np.float32),
            w_l=np.full(n, 30, np.float32),
            p_l=np.full(n, 9, np.float32),
            gflg=np.zeros(n, np.int8),
            qflg=np.ones(n, np.int8),
            pwr0=np.zeros(75, np.float32),
        )
    return rec


@pytest.mark.parametrize("rev,expect", [((2, 0), "2.0"), ((2, 5), "2.5"), ((3, 0), "3.0")])
def test_all_fitacf_versions_read_the_same_way(rev, expect):
    """fitacf 2.0, 2.5 and 3.0 share one record layout; the revision is kept in Beam.fit_version."""
    recs = [synthetic_record(T0 + dt.timedelta(seconds=3 * i), i, scan=int(i == 0), rev=rev) for i in range(4)]
    beams = records_to_beams(recs)
    assert {b.fit_version for b in beams} == {expect}
    assert [b.slist.tolist() for b in beams] == [[3, 4]] * 4


def test_missing_revision_and_optional_fields_are_tolerated():
    rec = synthetic_record(T0, 0)
    for k in ("fitacf.revision.major", "fitacf.revision.minor", "v_e", "qflg"):
        rec.pop(k)
    b = records_to_beams([rec])[0]
    assert b.fit_version == "" and np.isnan(b.v_e).all()


def test_scan_flag_is_reset_when_the_first_beam_has_no_echo(tmp_path):
    """Dropping scan-start beams must not merge scans in the written file."""
    recs = []
    for m in range(3):
        for b in range(4):
            recs.append(
                synthetic_record(T0 + dt.timedelta(minutes=m, seconds=3 * b), b, scan=int(b == 0), slist=(3, 4, 5))
            )
    reader = FitacfReader("x", T0 - dt.timedelta(hours=1), T0 + dt.timedelta(hours=1), files=[])
    reader._records = recs
    scans = beams_to_scans(reader.beams())
    for s in scans:
        s.beams = s.beams[1:]  # lose the scan-start beam of every scan
        for b in s.beams:
            b.extra = {
                "v_mad": np.zeros(b.n_echoes),
                "gflg_conv": np.zeros(b.n_echoes),
                "gflg_kde": np.full(b.n_echoes, np.nan),
                "clabel": np.zeros(b.n_echoes),
            }
    path = write_fitacfpp(tmp_path / fitacfpp_name("tst", T0), scans_to_records(scans, recs, "cmd"))
    back = beams_to_scans(records_to_beams(read_fitacf_file(path)))
    assert [len(s.beams) for s in back] == [3, 3, 3]
    assert (
        back[0].beams[0].extra["gflg_kde"].tolist() == [np.nan, np.nan, np.nan]
        or np.isnan(back[0].beams[0].extra["gflg_kde"]).all()
    )


def test_writer_replaces_instead_of_appending(tmp_path):
    recs = [synthetic_record(T0 + dt.timedelta(seconds=3 * i), i) for i in range(3)]
    beams = records_to_beams(recs)
    from amgeo_sd.models import Scan

    for b in beams:
        b.extra = {"v_mad": np.zeros(2)}
    scan = Scan(beams=beams).update()
    out = scans_to_records([scan], recs)
    path = tmp_path / "x.fitacf.pl.bz2"
    for _ in range(2):
        write_fitacfpp(path, out)
    assert len(read_fitacf_file(path)) == 3
    assert not list(tmp_path.glob("*.part"))


def test_scans_to_records_needs_the_source_record():
    from amgeo_sd.models import Scan

    b = records_to_beams([synthetic_record(T0, 0)])[0]
    with pytest.raises(KeyError, match="no source record"):
        scans_to_records([Scan(beams=[b]).update()], [])


def test_load_scans_rejects_mixed_inputs_and_missing_files(tmp_path):
    (tmp_path / "a.nc").touch()
    (tmp_path / "b.fitacf").touch()
    with pytest.raises(ValueError, match="mix"):
        load_scans([tmp_path])
    with pytest.raises(FileNotFoundError):
        load_scans([tmp_path / "empty_dir_that_does_not_exist"])


# ------------------------------------------------------------------------ real data
@pytest.fixture(scope="module")
def bks_run(tmp_path_factory, fitacf_files):
    folder = DATA / "bks_20250208"
    if not folder.is_dir():
        pytest.skip("bks_20250208 not unpacked")
    out = tmp_path_factory.mktemp("bks")
    start, end = dt.datetime(2025, 2, 8, 2, 10), dt.datetime(2025, 2, 8, 2, 40)
    res = run_median_filter(
        "bks", start, end, files=sorted(folder.glob("*.bz2")), out_base=out, sim_id="W", kde=False, argv=["t"]
    )
    return start, end, res


@pytest.mark.realdata
def test_nc_and_dmap_outputs_describe_the_same_echoes(bks_run):
    start, end, res = bks_run
    df_nc, m_nc = load_frame([res["nc_file"]])
    df_pl, m_pl = load_frame([res["fitacf_file"]])
    assert (m_nc["kind"], m_pl["kind"]) == ("fitacf++", "fitacf++")
    assert m_pl["fit_version"] == "2.5" and m_nc["rad"] == m_pl["rad"] == "bks"
    assert len(df_nc) == len(df_pl) == res["n_echoes"]
    key = ["time", "bmnum", "slist"]
    a, b = df_nc.sort_values(key).reset_index(drop=True), df_pl.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(a["v"], b["v"], rtol=1e-6)  # DMAP stores float32
    np.testing.assert_array_equal(a["gflg"], b["gflg"])
    np.testing.assert_allclose(a["v_mad"], b["v_mad"], rtol=1e-6)
    np.testing.assert_array_equal(a["clabel"], b["clabel"])
    assert a["gflg_kde"].isna().all() and b["gflg_kde"].isna().all()  # kde=False -> NaN survives -1 encoding


@pytest.mark.realdata
def test_dmap_output_keeps_standard_fitacf_fields_and_scan_structure(bks_run):
    start, end, res = bks_run
    recs = read_fitacf_file(res["fitacf_file"])
    assert {"pwr0", "ltab", "qflg", "elv", "origin.command", "amgeo.model", "v_mad"} <= set(recs[0])
    assert recs[0]["amgeo.model"] == "median"
    scans, meta = load_scans([res["fitacf_file"]])
    assert len(scans) == res["n_scans"]
    assert all(len({b.bmnum for b in s.beams}) == len(s.beams) for s in scans)
    # original per-echo fields survive for gates that had an echo, NaN for cells the filter added
    r = next(r for r in recs if len(r["slist"]) > 3)
    assert np.isfinite(r["p_s"]).any() and r["pwr0"].shape == (75,) and r["slist"].dtype == np.int16


@pytest.mark.realdata
def test_rerunning_a_pipeline_does_not_duplicate_records(bks_run):
    start, end, res = bks_run
    n = len(read_fitacf_file(res["fitacf_file"]))
    again = run_median_filter(
        "bks",
        start,
        end,
        files=sorted((DATA / "bks_20250208").glob("*.bz2")),
        out_base=Path(res["out_dir"]).parents[1],
        sim_id="W",
        kde=False,
        argv=["t"],
    )
    assert len(read_fitacf_file(again["fitacf_file"])) == n


@pytest.mark.realdata
def test_raw_fitacf_frame_reports_version(fitacf_files):
    df, meta = load_frame([DATA / "gbr_20250208"], dt.datetime(2025, 2, 8, 3), dt.datetime(2025, 2, 8, 3, 10))
    assert meta["kind"] == "fitacf" and meta["fit_version"] == "2.5" and meta["rad"] == "gbr" and meta["extra"] == []
    assert df["time"].min() >= np.datetime64("2025-02-08T03:00") and len(df) > 100
