"""DBSCAN / GMM model: Ribiero rule, clustering, pipeline outputs, checked against the legacy logic on real bks / gbr data."""

import datetime as dt
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from reference_dbscan import gmm_components as ref_gmm_components  # noqa: E402
from reference_dbscan import legacy_gmm_dbscan, reference_cluster, ribiero_gs_flg  # noqa: E402

from amgeo_sd.cli import main  # noqa: E402
from amgeo_sd.clustering import cluster_beam, cluster_scans, gmm_components, parse_gmm_beams  # noqa: E402
from amgeo_sd.clustering.dbscan import box_cox_abs  # noqa: E402
from amgeo_sd.config import Dbscan, Settings, load_settings  # noqa: E402
from amgeo_sd.io import FitacfReader, load_frame, read_fitacf_file, records_to_beams, scans_to_dataframe  # noqa: E402
from amgeo_sd.models import Beam, Scan  # noqa: E402
from amgeo_sd.pipelines import run_dbscan_model  # noqa: E402
from amgeo_sd.plotting import plot_fan, plot_rti  # noqa: E402
from amgeo_sd.scatter import ribiero_flags, ribiero_gs_flag  # noqa: E402

DATA = Path(__file__).parent / "data"
T0 = dt.datetime(2025, 2, 8, 3, 0)
NORMAL = {"s_time": 1, "t_beam": -1, "s_mode": "normal"}
# the original code's DBSCAN values; tests that check the lattice geometry of DBSCAN pin them explicitly
LEGACY = Dbscan(eps=1.0, min_samples=5)


# ------------------------------------------------------------------------------------------ Ribiero rule
@pytest.mark.parametrize("hours,limit", [(0.5, 0.5), (1.5, 0.475), (2.5, 0.33), (5.0, 0.2), (20.0, 0.15)])
def test_ribiero_ratio_limits_by_duration(hours, limit):
    """Ionospheric (False) exactly when high/low velocity ratio exceeds the limit of the cluster's duration."""
    n_slow = 100
    for n_fast, ground in ((int(limit * n_slow) - 1, True), (int(limit * n_slow) + 2, False)):
        v = np.r_[np.full(n_slow, 5.0), np.full(n_fast, 80.0)]
        assert ribiero_gs_flag(v, hours, 25.0) is ground
    assert ribiero_gs_flag(np.full(10, 80.0), hours, 25.0) is False  # no slow echo: R = 1 > every limit


def test_ribiero_matches_legacy_function_on_random_clusters():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(3, 80))
        t_days = np.sort(rng.uniform(0, rng.choice([0.02, 0.05, 0.1, 0.2, 0.8]), n))
        v = rng.normal(0, rng.choice([5, 30, 100]), n)
        assert ribiero_gs_flag(np.abs(v), (t_days[-1] - t_days[0]) * 24, 25.0) == bool(ribiero_gs_flg(v, t_days, 25.0))


def test_ribiero_flags_noise_low_gates_and_per_cluster():
    t = np.datetime64("2025-02-08T03:00") + np.arange(12) * np.timedelta64(60, "s")
    labels = np.array([-1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    v = np.array([0, 0, 5, 5, 5, 5, 90, 90, 90, 90, 5, 5.0])
    gates = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3, 3])  # cluster 2 sits below gate 7
    flags = ribiero_flags(labels, v, t, gates, vth=25.0, min_gate=7)
    assert flags.tolist() == [-1, -1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1]
    assert flags.dtype == np.int8


# ------------------------------------------------------------------------------------------ clustering pieces
def blob_frame(gates, scans, beam=3, v=5.0):
    rows = [{"slist": g, "scnum": s, "bmnum": beam, "channel": 0, "v": v + 0.1 * g} for s in scans for g in gates]
    return pd.DataFrame(rows)


def test_parse_gmm_beams():
    beams = [0, 1, 2, 3]
    assert parse_gmm_beams("", beams) == set()
    assert parse_gmm_beams("a", beams) == {0, 1, 2, 3}
    assert parse_gmm_beams("T", beams, t_beam=2) == {2} and parse_gmm_beams("T", beams) == set()
    assert parse_gmm_beams("1, 3", beams) == {1, 3}


def test_dbscan_blob_and_isolated_points():
    blob = blob_frame(range(10, 16), range(8))  # 6 gates x 8 scans
    df = pd.concat([blob, blob_frame([50], [0, 4])], ignore_index=True)
    tag = cluster_beam(df, LEGACY)
    interior = ((blob["slist"] > 10) & (blob["slist"] < 15) & (blob["scnum"] > 0) & (blob["scnum"] < 7)).to_numpy()
    assert (tag[: len(blob)][interior] == 0).all()  # core points all in cluster 0
    assert tag[len(blob) :].tolist() == [-1, -1]  # two lone echoes are noise
    assert (tag[: len(blob)] == -1).sum() == 4  # only the four corners have no core neighbour


def test_eps_scaling_and_themis_multiplier():
    sparse = blob_frame(range(10, 20, 2), range(0, 12, 2))  # points 2 apart in gate and scan
    assert (cluster_beam(sparse, LEGACY) == -1).all()  # eps = 1 cannot bridge a gap of 2
    scaled = cluster_beam(
        sparse, Dbscan(eps=1.0, min_samples=5, eps_g=2.0, eps_s=2.0)
    )  # scaled space: lattice neighbours at distance 1
    assert (scaled >= 0).sum() >= 20 and set(scaled) <= {-1, 0}
    blob = blob_frame(range(10, 14), range(6))  # 24 echoes
    normal = cluster_beam(blob, LEGACY, themis=False)
    assert (normal == 0).sum() == 20 and (normal == -1).sum() == 4  # all but the four corners
    assert (cluster_beam(blob, LEGACY, themis=True) == -1).all()  # min_samples 5 * 10 = 50 > 24


def test_gmm_components_small_beam_single_component_and_determinism():
    v = np.random.default_rng(0).normal(0, 20, 99)
    assert (gmm_components(v, Dbscan()) == 0).all()  # fewer than n_data_th = 100
    big = np.r_[np.random.default_rng(1).normal(-20, 3, 150), np.random.default_rng(2).normal(90, 3, 150)]
    p = Dbscan(n_clusters=2)
    a, b = gmm_components(big, p), gmm_components(big, p)
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 2 and len(np.unique(a[:150])) == 1 and a[0] != a[-1]  # |v| = 20 and 90: separable


def test_box_cox_handles_constant_and_zero_velocities():
    assert (box_cox_abs(np.full(10, 7.0)) == 0).all()
    out = box_cox_abs(np.array([0.0, 1.0, -3.0, 50.0]))
    assert np.isfinite(out).all()


def test_gmm_labels_are_unique_across_components():
    """Two velocity populations on the same gates/scans: each gets its own DBSCAN cluster; no label is shared."""
    n = 120
    rng = np.random.default_rng(3)
    df = pd.concat(
        [
            blob_frame(range(10, 14), range(30)).assign(v=rng.normal(20, 2, 120)),
            blob_frame(range(10, 14), range(30)).assign(v=rng.normal(-90, 2, 120)),
        ],
        ignore_index=True,
    )
    assert len(df) == 2 * n
    p = Dbscan(n_clusters=2, n_data_th=50)
    tag = cluster_beam(df, p, use_gmm=True)
    first, second = set(tag[:n]) - {-1}, set(tag[n:]) - {-1}
    assert first and second and first.isdisjoint(second)
    legacy = legacy_gmm_dbscan(
        df, asdict(p), themis=False
    )  # same partition, but legacy labels collide across components
    assert set(legacy[:n]) & set(legacy[n:]) - {-1}


def mk_scan(i, beams):
    return Scan(beams=beams, stime=T0 + dt.timedelta(minutes=i)).update()


def mk_beam(bm, i, gates, v=5.0, channel=0):
    g = np.asarray(gates, dtype=int)
    return Beam(
        time=T0 + dt.timedelta(minutes=i, seconds=bm),
        bmnum=bm,
        channel=channel,
        nrang=75,
        scan=int(bm == 0),
        slist=g,
        v=np.full(len(g), v),
        v_e=np.ones(len(g)),
        w_l=np.full(len(g), 20.0),
        p_l=np.full(len(g), 9.0),
        gflg=np.zeros(len(g), np.int8),
    )


def test_cluster_scans_writes_aligned_extras_and_handles_empty_beams():
    scans = [
        mk_scan(i, [mk_beam(0, i, range(10, 15)), mk_beam(1, i, []), mk_beam(2, i, range(30, 33), v=90.0)])
        for i in range(8)
    ]
    cluster_scans(scans, NORMAL, Dbscan())
    for s in scans:
        for b in s.beams:
            assert set(b.extra) == {"cluster_tag", "gflg_ribiero"}
            assert len(b.extra["cluster_tag"]) == len(b.extra["gflg_ribiero"]) == b.n_echoes
    b0, b2 = scans[3].beams[0], scans[3].beams[2]
    assert (b0.extra["cluster_tag"] == 0).all() and (b0.extra["gflg_ribiero"] == 1).all()  # slow, 7 minutes: ground
    assert (b2.extra["cluster_tag"] == 0).all() and (b2.extra["gflg_ribiero"] == 0).all()  # fast: ionospheric


def test_channels_are_clustered_separately():
    def make():
        scans = [
            mk_scan(i, [mk_beam(0, i, range(10, 15), channel=1), mk_beam(0, i, range(10, 15), channel=2)])
            for i in range(4)
        ]
        for s in scans:
            s.beams[1].time += dt.timedelta(seconds=1)
        return scans

    def tags(p):
        scans = make()
        cluster_scans(scans, NORMAL, p)
        return np.concatenate([b.extra["cluster_tag"] for s in scans for b in s.beams])

    # one channel alone: a core point has at most 5 neighbours (itself + 4); merged channels would double that
    assert (tags(Dbscan(eps=1.0, min_samples=7)) == -1).all()
    assert (tags(Dbscan(eps=1.0, min_samples=4)) >= 0).any()


def bridge_scans(seed=0, occupancy=0.75):
    """Fast dense patches A (scans 0-39) and B (60-99) joined by a sparse, slow 'bridge' (scans 40-59) of fragments."""
    rng = np.random.default_rng(seed)
    scans = []
    for i in range(100):
        if i < 40 or i >= 60:
            gates, v = range(10, 21), 60.0
        else:
            gates, v = [g for g in range(12, 22) if rng.random() < occupancy], 8.0
        scans.append(mk_scan(i, [mk_beam(3, i, gates, v=v)]))
    return scans


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_radius_links_sparse_fragments_to_their_parent_cluster(seed):
    """The reason for the eps = 2 / min_samples = 8 defaults: with the legacy values the sparse bridge breaks into small
    fragments that are classified one by one (slow -> ground scatter); with the defaults it joins the fast patches around
    it and is classified with them (ionospheric). The bridge is 75% occupied; much sparser bridges still fragment."""

    def bridge_stats(params):
        scans = cluster_scans(bridge_scans(seed), NORMAL, params)
        tag = np.concatenate([b.extra["cluster_tag"] for s in scans[40:60] for b in s.beams])
        flag = np.concatenate([b.extra["gflg_ribiero"] for s in scans[40:60] for b in s.beams])
        main = np.concatenate([b.extra["cluster_tag"] for s in scans[:40] for b in s.beams])
        main = np.bincount(main[main >= 0].astype(int)).argmax()
        return (tag == main).mean(), (flag == 0).mean(), (flag == 1).mean()

    in_main, ionospheric, ground = bridge_stats(Dbscan())
    assert in_main > 0.9 and ionospheric > 0.9 and ground < 0.05
    in_main_old, ionospheric_old, ground_old = bridge_stats(LEGACY)
    assert in_main_old < 0.3 and ionospheric_old < 0.3 and ground_old > 0.2  # fragments classified on their own


def test_dbscan_defaults_are_the_documented_ones():
    p = Dbscan()
    assert (p.eps, p.min_samples, p.eps_g, p.eps_s, p.tb_mult, p.velocity_threshold) == (2.0, 8, 1.0, 1.0, 10, 25.0)
    assert (LEGACY.eps, LEGACY.min_samples) == (1.0, 5)  # what the legacy code used


def test_dbscan_settings_validation(tmp_path):
    base = '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[dbscan]\n'
    for bad, msg in (
        ("eps = 0\n", "positive"),
        ("min_samples = 0\n", "at least 1"),
        ('cov = "x"\n', "cov"),
        ('gmm = "q"\n', "gmm"),
    ):
        f = tmp_path / "c.toml"
        f.write_text(base + bad)
        with pytest.raises(ValueError, match=msg):
            load_settings(f, require="dbscan")
    f.write_text(base + 'gmm = "3, 7"\nn_clusters = 4\n')
    s = load_settings(f, require="dbscan")
    assert (s.dbscan.gmm, s.dbscan.n_clusters, s.dbscan.eps) == ("3, 7", 4, 2.0)
    assert Settings().dbscan == Dbscan()


# ------------------------------------------------------------------------------------------ real data
@pytest.fixture(scope="module", params=[("bks", 12), ("gbr", 7)], ids=["bks", "gbr"])
def run(request, tmp_path_factory, fitacf_files):
    rad, beam = request.param
    folder = DATA / f"{rad}_20250208"
    if not folder.is_dir():
        pytest.skip(f"{folder.name} not unpacked")
    start, end = T0, T0 + dt.timedelta(minutes=30)
    out = tmp_path_factory.mktemp(rad)
    res = run_dbscan_model(
        rad,
        start,
        end,
        files=sorted(folder.glob("*.bz2")),
        out_base=out,
        sim_id="D",
        argv=["t"],
        input_paths=[str(folder)],
    )
    reader = FitacfReader(rad, start, end, files=sorted(folder.glob("*.bz2")))
    return rad, beam, folder, start, end, res, reader, out


@pytest.mark.realdata
@pytest.mark.parametrize("gmm", ["", "A"])
def test_clustering_equals_legacy_logic_on_real_data(run, gmm):
    rad, beam, folder, start, end, res, reader, _ = run
    scans = reader.scans()
    frame = scans_to_dataframe(scans)
    p = Dbscan(gmm=gmm)
    if gmm:  # legacy crashes on beams with fewer echoes than n_data_th; compare on the beams where it runs
        busy = frame.groupby("bmnum").size()
        p = Dbscan(gmm=",".join(str(b) for b in busy[busy >= p.n_data_th].index))
    cluster_scans(scans, NORMAL, p)
    tag = np.concatenate([b.extra["cluster_tag"] for s in scans for b in s.beams if b.n_echoes])
    flag = np.concatenate([b.extra["gflg_ribiero"] for s in scans for b in s.beams if b.n_echoes])
    gmm_beams = {int(x) for x in p.gmm.split(",") if x}
    want_tag, want_flag = reference_cluster(frame, asdict(p), NORMAL, gmm_beams)
    if not gmm:
        np.testing.assert_array_equal(tag, want_tag)
        np.testing.assert_array_equal(flag, want_flag)
        return
    # with GMM the legacy labels collide between components; ours must refine them and agree on all non-GMM beams
    for b, rows in frame.groupby("bmnum"):
        ours, theirs = tag[rows.index], want_tag[rows.index]
        if b not in gmm_beams:
            np.testing.assert_array_equal(ours, theirs)
            continue
        assert ((ours < 0) == (theirs < 0)).all()  # same noise
        pairs = pd.DataFrame({"o": ours, "t": theirs}).drop_duplicates()
        assert pairs.groupby("o")["t"].nunique().max() == 1  # each of our clusters lies inside one legacy label
        comp = ref_gmm_components(frame.loc[rows.index], asdict(p))
        assert len(np.unique(comp)) > 1


@pytest.mark.realdata
def test_pipeline_outputs_and_provenance(run):
    rad, beam, folder, start, end, res, reader, out = run
    out_dir = Path(res["out_dir"])
    assert {f.name for f in out_dir.iterdir()} == {
        Path(res["nc_file"]).name,
        Path(res["fitacf_file"]).name,
        "params.toml",
        "script.cmd",
    }
    assert (
        res["n_scans"] >= 25
        and res["n_clusters"] > 5
        and 0 < res["fraction_noise"] < 0.9
        and 0 < res["fraction_ground"] < 1
    )
    again = load_settings(out_dir / "params.toml", require="dbscan")
    assert again.run.rad == rad and again.dbscan == Dbscan() and again.input.paths == [str(folder)]
    text = (out_dir / "params.toml").read_text()
    assert "[dbscan]" in text and "[result]" in text and "n_clusters" in text and "[median]" not in text


@pytest.mark.realdata
def test_netcdf_has_raw_echoes_with_labels(run):
    rad, beam, folder, start, end, res, reader, _ = run
    ds = xr.open_dataset(res["nc_file"])
    raw_n = sum(b.n_echoes for b in reader.beams())
    assert int(np.isfinite(ds["v"].values).sum()) == res["n_echoes"] == raw_n  # nothing removed
    for name in ("cluster_tag", "gflg_ribiero"):
        assert int(np.isfinite(ds[name].values).sum()) == raw_n
    assert set(np.unique(ds["gflg_ribiero"].values[np.isfinite(ds["gflg_ribiero"].values)])) <= {-1.0, 0.0, 1.0}
    assert (
        ds.attrs["gflg_type"] == "us_c" and ds.attrs["geometry_backend"] == "legacy" and ds.attrs["dbscan_eps"] == 2.0
    )


@pytest.mark.realdata
def test_dmap_output_keeps_every_original_record(run):
    rad, beam, folder, start, end, res, reader, _ = run
    raw, new = reader.records(), read_fitacf_file(res["fitacf_file"])
    assert len(new) == len(raw)  # empty beams kept too
    assert [r["scan"] for r in new] == [r["scan"] for r in raw]  # original scan flags
    assert new[0]["amgeo.model"] == "dbscan" and new[0]["origin.command"].startswith("python t")
    for a, b in zip(raw, new, strict=True):
        if "slist" in a:
            np.testing.assert_array_equal(a["slist"], b["slist"])
            np.testing.assert_array_equal(a["v"], b["v"])
            np.testing.assert_array_equal(a["qflg"], b["qflg"])  # original quality flags untouched
            np.testing.assert_array_equal(a["v_e"], b["v_e"])
            assert len(b["cluster_tag"]) == len(b["gflg_ribiero"]) == len(a["slist"])
    back = records_to_beams(new)
    want = {(b.time, b.bmnum): b for s in res["scans"] for b in s.beams}
    for b in back[:200]:
        np.testing.assert_array_equal(
            b.extra.get("cluster_tag", []), want[(b.time, b.bmnum)].extra.get("cluster_tag", [])
        )


@pytest.mark.realdata
def test_rerun_is_deterministic_and_does_not_duplicate(run):
    rad, beam, folder, start, end, res, reader, out = run
    again = run_dbscan_model(rad, start, end, files=sorted(folder.glob("*.bz2")), out_base=out, sim_id="D", argv=["t"])
    assert len(read_fitacf_file(again["fitacf_file"])) == len(reader.records())
    a, b = xr.open_dataset(res["nc_file"]), xr.open_dataset(again["nc_file"])
    np.testing.assert_array_equal(a["cluster_tag"].values, b["cluster_tag"].values)


@pytest.mark.realdata
def test_plots_of_cluster_output(run, tmp_path):
    rad, beam, folder, start, end, res, reader, _ = run
    df, meta = load_frame([res["nc_file"]])
    assert {"cluster_tag", "gflg_ribiero"} <= set(meta["extra"])
    plot_rti(df, beam, ("v", "cluster_tag", "gflg_ribiero"), tmp_path / "r.png", rad=rad, kind="fitacf++ (DBSCAN)")
    plot_fan(df, rad, T0 + dt.timedelta(minutes=10), ("v", "gflg_ribiero"), tmp_path / "f.png", geometry="pydarn")
    assert (tmp_path / "r.png").stat().st_size > 10000 and (tmp_path / "f.png").stat().st_size > 10000
    dmap_df, m2 = load_frame([res["fitacf_file"]])
    assert m2["fit_version"] == "2.5" and len(dmap_df) == len(df)


@pytest.mark.realdata
def test_dbscan_cli_with_toml_and_overrides(run, tmp_path, capsys):
    rad, beam, folder, start, end, res, reader, _ = run
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        f'[run]\nrad = "{rad}"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T03:12:00\nsim_id = "C"\nout = "{tmp_path}"\n'
        f'[input]\npaths = ["{folder}"]\n[dbscan]\nmin_samples = 4\n'
    )
    assert main(["dbscan", "-c", str(cfg), "-eps", "1.5", "--no-fitacf", "--geometry", "pydarn"]) == 0
    out = capsys.readouterr().out
    assert "clusters" in out and "fitacf_file" not in out
    assert main(["dbscan", "--print-config", "-c", str(cfg), "-gmm", "A"]) == 0
    printed = capsys.readouterr().out
    assert '"gmm": "A"' in printed and '"min_samples": 4' in printed
    nc = next((tmp_path / "C" / rad).glob("*.nc"))
    attrs = xr.open_dataset(nc).attrs
    assert attrs["dbscan_eps"] == 1.5 and attrs["dbscan_min_samples"] == 4 and attrs["geometry_backend"] == "pydarn"
