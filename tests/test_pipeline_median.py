"""scan-mode estimation, NetCDF export and the end-to-end median pipeline (bks mid-lat, gbr high-lat)."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from amgeo_sd import export
from amgeo_sd.cli import main
from amgeo_sd.config import load_settings
from amgeo_sd.geo import radar_fov
from amgeo_sd.io.summary import estimate_scan_info
from amgeo_sd.models import Beam, Scan
from amgeo_sd.pipelines import run_median_filter

DATA = Path(__file__).parent / "data"
GOLDEN = np.load(Path(__file__).parent / "golden" / "fov_legacy_20250208.npz")
T0 = dt.datetime(2025, 2, 8, 2, 0)


def beam(t, bm, scan=0, gates=(5, 6), channel=0):
    n = len(gates)
    return Beam(
        time=t,
        bmnum=bm,
        scan=scan,
        channel=channel,
        nrang=75,
        tfreq=11000.0,
        noise_sky=2.0,
        slist=np.array(gates, dtype=int),
        v=np.arange(n, dtype=float),
        v_e=np.ones(n),
        w_l=np.full(n, 30.0),
        p_l=np.full(n, 8.0),
        gflg=np.zeros(n, np.int8),
    )


# ----------------------------------------------------------------------- scan info
def test_scan_info_one_minute_scans():
    beams = [beam(T0 + dt.timedelta(minutes=m, seconds=3 * b), b, scan=int(b == 0)) for m in range(4) for b in range(5)]
    assert estimate_scan_info(beams) == {"s_time": 1, "t_beam": -1, "s_mode": "normal"}


def test_scan_info_empty_scan_start_beam_does_not_double_the_period():
    beams = [
        beam(
            T0 + dt.timedelta(minutes=m, seconds=3 * b), b, scan=int(b == 0), gates=() if (b == 0 and m == 1) else (5,)
        )
        for m in range(5)
        for b in range(5)
    ]
    assert estimate_scan_info(beams)["s_time"] == 1


def test_scan_info_themis_two_minute_scans():
    beams = []
    for m in range(0, 8, 2):  # a 2-min scan: beam 7 is sounded between every beam
        for i, b in enumerate(range(4)):
            beams.append(beam(T0 + dt.timedelta(minutes=m, seconds=12 * i), b, scan=int(i == 0)))
            beams.append(beam(T0 + dt.timedelta(minutes=m, seconds=12 * i + 6), 7))
    info = estimate_scan_info(beams)
    assert info == {"s_time": 2, "t_beam": 7, "s_mode": "themis"}


def test_scan_info_defaults_without_scan_starts():
    assert estimate_scan_info([]) == {"s_time": 1, "t_beam": -1, "s_mode": "normal"}
    assert estimate_scan_info([beam(T0, 0, scan=1)])["s_time"] == 1


# ----------------------------------------------------------------------- export
def small_scans():
    scans = []
    for i in range(2):
        bs = [
            beam(T0 + dt.timedelta(minutes=i, seconds=b), b, scan=int(b == 0), gates=(5 + b, 6 + b)) for b in range(3)
        ]
        for b in bs:
            b.extra["v_mad"] = np.array([0.5, 0.25])
        scans.append(Scan(beams=bs).update())
    return scans


def test_dataset_layout_and_values(tmp_path):
    fov = radar_fov("bks", T0)
    ds = export.scans_to_dataset(
        small_scans(), fov, {"rad": "bks", "flag": True, "when": T0, "list": ["a", "b"], "none": None}
    )
    assert ds.sizes == {"beams": 25, "gates": 111, "scan_num": 2, "sound": 3}
    assert ds["v"].dims == ("scan_num", "sound", "gates")
    assert ds["v"].values[1, 2, 7] == 0.0 and ds["v"].values[1, 2, 8] == 1.0
    assert np.isnan(ds["v"].values[1, 2, 9])
    assert ds["v_mad"].values[0, 1, 6] == 0.5 and "clabel" not in ds  # absent extras are not written
    assert ds["bmnum"].values.tolist() == [[0, 1, 2], [0, 1, 2]] and ds["bmnum"].dtype == np.int32
    np.testing.assert_array_equal(ds["gdlat"].values, fov["glat"])
    assert (
        ds.attrs["flag"] == 1
        and ds.attrs["when"] == "2025-02-08T02:00:00"
        and ds.attrs["list"] == "a-b"
        and ds.attrs["none"] == ""
    )
    path = export.write_netcdf(ds, tmp_path / "sub" / "x.nc")
    back = xr.open_dataset(path)
    np.testing.assert_array_equal(back["v"].values, ds["v"].values)
    assert back["noise.sky"].shape == (2, 3) and back["time"].dtype.kind == "M"


def test_dataset_rejects_gate_outside_radar_and_empty_input():
    fov = radar_fov("bks", T0, azm=False)
    scans = small_scans()
    scans[0].beams[0].slist = np.array([5, 200])
    with pytest.raises(ValueError, match="outside"):
        export.scans_to_dataset(scans, fov)
    with pytest.raises(ValueError, match="no scans"):
        export.scans_to_dataset([], fov)


def test_save_run_metadata(tmp_path):
    export.save_run_metadata(["-m", "amgeo_sd", "median"], '[run]\nrad = "bks"\n', tmp_path)
    assert (tmp_path / "script.cmd").read_text().startswith("python -m amgeo_sd median")
    assert (tmp_path / "params.toml").read_text().startswith("[run]")


# ----------------------------------------------------------------------- end to end, real data
CASES = [("bks", "bks_20250208", 24), ("gbr", "gbr_20250208", 16)]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def real_run(request, tmp_path_factory, fitacf_files):
    rad, folder, nbeams = request.param
    if not (DATA / folder).is_dir():
        pytest.skip(f"{folder} not unpacked")
    out = tmp_path_factory.mktemp(rad)
    start, end = dt.datetime(2025, 2, 8, 3, 0), dt.datetime(2025, 2, 8, 3, 30)
    res = run_median_filter(
        rad,
        start,
        end,
        files=sorted((DATA / folder).glob("*.bz2")),
        out_base=out,
        sim_id="T1",
        n_jobs=1,
        argv=["pytest"],
    )
    return rad, nbeams, start, end, res


@pytest.mark.realdata
def test_pipeline_outputs(real_run):
    rad, nbeams, start, end, res = real_run
    assert res["scan_info"] == {"s_time": 1, "t_beam": -1, "s_mode": "normal"}
    assert 25 <= res["n_scans"] <= 31 and res["n_echoes"] > 500
    out_dir = Path(res["out_dir"])
    assert out_dir.name == rad and out_dir.parent.name == "T1"
    assert {p.name for p in out_dir.iterdir()} == {
        Path(res["nc_file"]).name,
        Path(res["fitacf_file"]).name,
        "params.toml",
        "script.cmd",
    }
    assert res["fitacf_file"].endswith(f"{rad}.fitacf.pl.bz2")
    # the written parameters are a valid config for the same run
    again = load_settings(out_dir / "params.toml")
    assert (
        again.run.rad == rad
        and again.run.start == dt.datetime(2025, 2, 8, 3)
        and again.median.thresh == 0.6
        and again.median.edge_gates == "legacy"
    )
    text = (out_dir / "params.toml").read_text()
    assert "# " in text and "[provenance]" in text and "[result]" in text and "n_echoes" in text


@pytest.mark.realdata
def test_pipeline_netcdf_matches_in_memory_result(real_run):
    rad, nbeams, start, end, res = real_run
    ds = xr.open_dataset(res["nc_file"])
    assert ds.sizes["beams"] == nbeams + 1 and ds.sizes["scan_num"] == res["n_scans"]
    assert int(np.isfinite(ds["v"].values).sum()) == res["n_echoes"]
    assert ds["scan_stime"].values.min() >= np.datetime64(start) and ds["scan_stime"].values.max() <= np.datetime64(end)
    # spot check one scan, beam, gate against the filtered Scan objects
    scan = res["scans"][5]
    for j, b in enumerate(scan.beams[:4]):
        assert ds["bmnum"].values[5, j] == b.bmnum
        np.testing.assert_allclose(ds["v"].values[5, j, b.slist], b.v)
        np.testing.assert_allclose(ds["gflg_conv"].values[5, j, b.slist], b.extra["gflg_conv"])
        np.testing.assert_array_equal(ds["clabel"].values[5, j, b.slist], b.extra["clabel"])
    # geometry equals the legacy golden arrays
    np.testing.assert_allclose(ds["gdlat"].values, GOLDEN[f"{rad}_glat"], atol=1e-9)
    np.testing.assert_allclose(ds["gdazm"].values, GOLDEN[f"{rad}_gazm"], atol=1e-9, equal_nan=True)


@pytest.mark.realdata
def test_pipeline_flags_are_physical(real_run):
    rad, nbeams, start, end, res = real_run
    ds = xr.open_dataset(res["nc_file"])
    for name, allowed in (("gflg", {0, 1}), ("gflg_conv", {0, 1, 2}), ("gflg_kde", {0, 1, 2})):
        vals = ds[name].values[np.isfinite(ds[name].values)]
        assert set(np.unique(vals)) <= allowed, name
    v = ds["v"].values[np.isfinite(ds["v"].values)]
    assert np.abs(v).max() < 3000 and ds["v_mad"].values[np.isfinite(ds["v_mad"].values)].min() >= 0
    gs = ds["gflg"].values[np.isfinite(ds["gflg"].values)].mean()
    assert 0.0 < gs < 1.0  # both populations present


@pytest.mark.realdata
def test_filter_keeps_a_sensible_fraction_of_raw_echoes(real_run):
    rad, nbeams, start, end, res = real_run
    from amgeo_sd.io import FitacfReader

    raw = FitacfReader(rad, start, end, files=sorted((DATA / f"{rad}_20250208").glob("*.bz2"))).scans()
    raw_n = sum(b.n_echoes for s in raw for b in s.beams)
    assert 0.3 * raw_n < res["n_echoes"] < raw_n


@pytest.mark.realdata
def test_cli_median(tmp_path, capsys, fitacf_files):
    folder = DATA / "gbr_20250208"
    if not folder.is_dir():
        pytest.skip("gbr_20250208 not unpacked")
    rc = main(
        [
            "median",
            "-r",
            "gbr",
            "-s",
            "2025-02-08T05:00",
            "-e",
            "2025-02-08T05:20",
            "-i",
            str(folder),
            "--no-kde",
            "-o",
            str(tmp_path),
            "-sid",
            "C1",
        ]
    )
    assert rc == 0
    assert "filtered scans" in capsys.readouterr().out
    assert list((tmp_path / "C1" / "gbr").glob("*.fitacf++.nc"))


def test_pipeline_errors(tmp_path):
    with pytest.raises(RuntimeError, match="no fitacf data"):
        run_median_filter("bks", T0, T0 + dt.timedelta(minutes=10), files=[], out_base=tmp_path)


@pytest.mark.realdata
def test_geometry_backend_switch_reaches_netcdf_params_and_plots(tmp_path, fitacf_files):
    folder = DATA / "bks_20250208"
    if not folder.is_dir():
        pytest.skip("bks_20250208 not unpacked")
    start, end = dt.datetime(2025, 2, 8, 3, 0), dt.datetime(2025, 2, 8, 3, 8)
    files = sorted(folder.glob("*.bz2"))
    out = {
        g: run_median_filter(
            "bks",
            start,
            end,
            files=files,
            out_base=tmp_path,
            sim_id=g,
            kde=False,
            write_fitacf=False,
            geometry=g,
            altitude=300.0,
            argv=["t"],
        )
        for g in ("legacy", "pydarn")
    }
    ds = {g: xr.open_dataset(r["nc_file"]) for g, r in out.items()}
    assert ds["legacy"].attrs["geometry_backend"] == "legacy" and ds["pydarn"].attrs["geometry_backend"] == "pydarn"
    np.testing.assert_allclose(ds["legacy"]["gdlat"].values, GOLDEN["bks_glat"], atol=1e-9)
    assert np.abs(ds["pydarn"]["gdlat"].values - ds["legacy"]["gdlat"].values).max() > 0.05
    np.testing.assert_array_equal(ds["legacy"]["v"].values, ds["pydarn"]["v"].values)  # geometry never changes the data
    assert load_settings(Path(out["pydarn"]["out_dir"]) / "params.toml").geometry.backend == "pydarn"
    rc = main(
        [
            "plot",
            "fov",
            "-i",
            out["pydarn"]["nc_file"],
            "-t",
            "2025-02-08T03:04",
            "-p",
            "v",
            "--geometry",
            "pydarn",
            "-o",
            str(tmp_path),
            "--name",
            "g",
        ]
    )
    assert rc == 0 and (tmp_path / "g.png").exists()


@pytest.mark.realdata
def test_edge_gates_setting_reaches_params_and_netcdf(tmp_path, fitacf_files):
    folder = DATA / "bks_20250208"
    if not folder.is_dir():
        pytest.skip("bks_20250208 not unpacked")
    res = run_median_filter(
        "bks",
        dt.datetime(2025, 2, 8, 3, 0),
        dt.datetime(2025, 2, 8, 3, 8),
        files=sorted(folder.glob("*.bz2")),
        out_base=tmp_path,
        sim_id="E",
        kde=False,
        write_fitacf=False,
        edge_gates="missing",
        argv=["t"],
    )
    ds = xr.open_dataset(res["nc_file"])
    assert ds.attrs["edge_gates"] == "missing" and ds.attrs["recommended_flags"] == "gflg_kde, gflg_conv"
    assert "legacy v-w rule" in ds.attrs["description"] or "legacy v-w rule" in ds.attrs["description"].replace(
        "\n", " "
    )
    assert load_settings(Path(res["out_dir"]) / "params.toml").median.edge_gates == "missing"
    rc = main(
        [
            "median",
            "--print-config",
            "-r",
            "bks",
            "-s",
            "2025-02-08T03:00",
            "-e",
            "2025-02-08T03:08",
            "--edge-gates",
            "missing",
        ]
    )
    assert rc == 0
