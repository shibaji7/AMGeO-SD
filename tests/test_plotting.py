"""RTI and field-of-view plots for fitacf and fitacf++ data (rendered headless, checked structurally)."""

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from amgeo_sd.cli import main  # noqa: E402
from amgeo_sd.config import Settings, load_settings  # noqa: E402
from amgeo_sd.io import load_frame  # noqa: E402
from amgeo_sd.pipelines import run_median_filter, run_plot  # noqa: E402
from amgeo_sd.plotting import FanPlots, RangeTimeIntervalPlot, plot_fan, plot_rti, plot_rti_compare, select_scan  # noqa: E402
from amgeo_sd.plotting.grid import grid_time_gate  # noqa: E402
from amgeo_sd.plotting.style import get_cluster_cmap  # noqa: E402

DATA = Path(__file__).parent / "data"
START, END = dt.datetime(2025, 2, 8, 3, 0), dt.datetime(2025, 2, 8, 3, 30)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def synthetic_frame(n_times=6, gates=(5, 6, 7), beam=2):
    t0 = np.datetime64("2025-02-08T03:00:00")
    rows = []
    for i in range(n_times):
        for g in gates:
            rows.append(
                {
                    "time": t0 + np.timedelta64(60 * i, "s"),
                    "bmnum": beam,
                    "slist": g,
                    "v": 10.0 * i - 20,
                    "w_l": 30.0,
                    "p_l": 12.0,
                    "gflg": i % 2,
                    "nrang": 75,
                    "scnum": i,
                    "frang": 180.0,
                    "rsep": 45.0,
                    "cluster_tag": 0 if i < 3 else 1,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- grid
def test_grid_means_duplicates_and_inserts_gap_columns():
    df = synthetic_frame(4)
    df = pd.concat([df, df.iloc[:1].assign(v=100.0)])  # a second echo in the same cell
    x, y, z = grid_time_gate(df, "v", 75)
    assert z.shape == (4, 75) and y.tolist() == list(range(75))
    assert z[0, 5] == pytest.approx((-20 + 100) / 2) and np.isnan(z[0, 20])
    late = df.assign(time=df["time"] + np.timedelta64(3600, "s"))
    x2, _, z2 = grid_time_gate(pd.concat([df, late]), "v", 75)
    assert len(x2) == 4 + 1 + 4 and np.isnan(z2[4]).all()  # NaN column marks the gap


def test_cluster_cmap_noise_is_black_and_deterministic():
    a, b = get_cluster_cmap(5, True), get_cluster_cmap(5, True)
    assert a(0) == (0, 0, 0, 1.0) and [a(i) for i in range(5)] == [b(i) for i in range(5)]


# ------------------------------------------------------------------------- RTI
def test_rti_panels_for_numeric_flag_and_cluster_columns(tmp_path):
    rti = plot_rti(synthetic_frame(), 2, ("v", "gflg", "cluster_tag"), tmp_path / "a.png", rad="tst", kind="fitacf 2.5")
    assert (tmp_path / "a.png").stat().st_size > 5000
    titles = [ax.get_title(loc="left") for ax in rti.fig.axes if ax.get_title(loc="left")]
    assert titles == ["(a) v", "(b) gflg", "(c) cluster_tag"]
    assert rti.fig.texts[0].get_text().startswith("TST beam 2, 2025-02-08 (fitacf 2.5)")


def test_rti_errors_are_explicit():
    df = synthetic_frame()
    rti = RangeTimeIntervalPlot(75, num_subplots=1)
    with pytest.raises(ValueError, match="no echoes for beam 9"):
        rti.add_param(df, 9)
    with pytest.raises(ValueError, match="unknown parameter"):
        rti.add_param(df, 2, param="nonsense")
    with pytest.raises(ValueError, match="cluster plots need"):
        rti.add_clusters(df.drop(columns="cluster_tag"), 2)
    rti.add_param(df, 2)
    with pytest.raises(ValueError, match="created with 1 panels"):
        rti.add_param(df, 2)


def test_rti_compare_stacks_datasets(tmp_path):
    rti = plot_rti_compare(
        {"raw": synthetic_frame(), "filtered": synthetic_frame(4)}, 2, "v", tmp_path / "c.png", rad="tst", flag="gflg"
    )
    assert len([ax for ax in rti.fig.axes if ax.get_title(loc="left")]) == 4
    assert (tmp_path / "c.png").exists()


# ------------------------------------------------------------------------- FoV
def test_select_scan_picks_nearest_scan_start():
    df = synthetic_frame(6)
    sc = select_scan(df, dt.datetime(2025, 2, 8, 3, 2, 20))
    assert sc["scnum"].unique().tolist() == [2]
    with pytest.raises(ValueError, match="no data"):
        select_scan(df.iloc[:0], START)
    with pytest.raises(ValueError, match="scnum"):
        select_scan(df.drop(columns="scnum"), START)


def test_fan_options_are_validated():
    with pytest.raises(ValueError, match="coords"):
        FanPlots(coords="mlt")
    fan = FanPlots(projection="sphere")
    with pytest.raises(ValueError, match="projection"):
        fan.plot_fov(synthetic_frame().assign(bmnum=3), START, "bks")


def test_fan_geo_and_aacgm_draw_cells(tmp_path):
    df = synthetic_frame().assign(bmnum=3, gflg=lambda d: d["gflg"].where(d["scnum"] != 0, 1))
    for coords in ("geo", "aacgmv2"):
        fan = plot_fan(df, "bks", START, ("v", "gflg"), tmp_path / f"{coords}.png", coords=coords)
        polys = [c for ax in fan.fig.axes for c in ax.collections if c.__class__.__name__ == "PolyCollection"]
        assert polys and sum(len(c.get_paths()) for c in polys) >= 3
        assert (tmp_path / f"{coords}.png").stat().st_size > 5000


def test_layfovs(tmp_path):
    fan = FanPlots(figsize=(5, 5))
    fan.plot_layfovs(["bks", "fhe"], dt.datetime(2025, 2, 8, 3))
    assert fan.save(tmp_path / "lay.png") and (tmp_path / "lay.png").exists()


# ------------------------------------------------------------------------- real data, both radars
@pytest.fixture(scope="module", params=[("bks", 12), ("gbr", 7)], ids=["bks", "gbr"])
def run(request, tmp_path_factory, fitacf_files):
    rad, beam = request.param
    folder = DATA / f"{rad}_20250208"
    if not folder.is_dir():
        pytest.skip(f"{folder.name} not unpacked")
    out = tmp_path_factory.mktemp(rad)
    res = run_median_filter(
        rad, START, END, files=sorted(folder.glob("*.bz2")), out_base=out, sim_id="P", kde=False, argv=["t"]
    )
    return rad, beam, folder, res, out


@pytest.mark.realdata
def test_all_input_formats_give_the_same_picture_data(run):
    rad, beam, folder, res, _ = run
    frames = {
        k: load_frame([p], START, END)[0]
        for k, p in {"raw": folder, "nc": res["nc_file"], "dmap": res["fitacf_file"]}.items()
    }
    assert len(frames["nc"]) == len(frames["dmap"]) < len(frames["raw"])
    for df in frames.values():
        sel = df[df["bmnum"] == beam]
        x, y, z = grid_time_gate(sel, "v", 75)
        assert np.isfinite(z).any()


@pytest.mark.realdata
def test_rti_of_raw_and_fitacfpp_on_real_data(run, tmp_path):
    rad, beam, folder, res, _ = run
    raw, mr = load_frame([folder], START, END)
    pp, mp = load_frame([res["nc_file"]], START, END)
    assert (mr["kind"], mr["fit_version"], mr["rad"]) == ("fitacf", "2.5", rad) and (mp["kind"], mp["rad"]) == (
        "fitacf++",
        rad,
    )
    rti = plot_rti(
        pp, beam, ("v", "w_l", "p_l", "gflg", "gflg_conv", "v_mad"), tmp_path / "pp.png", rad=rad, kind="fitacf++"
    )
    assert len([a for a in rti.fig.axes if a.get_title(loc="left")]) == 6
    plot_rti_compare({"fitacf 2.5": raw, "fitacf++": pp}, beam, "v", tmp_path / "cmp.png", rad=rad, flag="gflg")
    assert (tmp_path / "cmp.png").stat().st_size > 20000


@pytest.mark.realdata
def test_fan_of_raw_and_fitacfpp_on_real_data(run, tmp_path):
    rad, beam, folder, res, _ = run
    raw, _ = load_frame([folder], START, START + dt.timedelta(minutes=10))
    pp, _ = load_frame([res["fitacf_file"]], START, END)
    when = dt.datetime(2025, 2, 8, 3, 5)
    nc, _ = load_frame([res["nc_file"]], START, END)
    for name, df, flag in (
        ("raw", raw, "gflg"),
        ("pp", pp, "gflg"),
        ("nc", nc, "gflg_conv"),
    ):  # the DMAP has no gflg_conv
        fan = plot_fan(df, rad, when, ("v", flag), tmp_path / f"{name}.png")
        n = sum(
            len(c.get_paths())
            for ax in fan.fig.axes
            for c in ax.collections
            if c.__class__.__name__ == "PolyCollection"
        )
        assert n >= 2 * len(select_scan(df, when)) * 0.9, f"{name}: expected roughly one polygon per echo and panel"


@pytest.mark.realdata
def test_plot_cli_and_toml_config(run, tmp_path, capsys):
    rad, beam, folder, res, out = run
    cfg = tmp_path / "plot.toml"
    cfg.write_text(
        f'[plot]\ninputs = ["{folder}"]\ncompare = ["{res["nc_file"]}"]\nbeam = {beam}\nparams = ["v"]\nflag = "gflg"\n'
        f'start = 2025-02-08T03:00:00\nend = 2025-02-08T03:30:00\nout = "{tmp_path}"\nname = "x"\n'
    )
    assert main(["plot", "rti", "-c", str(cfg)]) == 0
    assert capsys.readouterr().out.strip().endswith("x.png") and (tmp_path / "x.png").exists()
    assert (
        main(
            [
                "plot",
                "fov",
                "-i",
                res["nc_file"],
                "-t",
                "2025-02-08T03:10",
                "-p",
                "v",
                "gflg_conv",
                "--coords",
                "aacgmv2",
                "-o",
                str(tmp_path),
                "--name",
                "y",
                "--format",
                "pdf",
            ]
        )
        == 0
    )
    assert (tmp_path / "y.pdf").exists()
    assert main(["plot", "rti", "--print-config", "-i", "a", "--beam", "4"]) == 0
    assert '"beam": 4' in capsys.readouterr().out


def test_plot_settings_validation():
    s = Settings()
    with pytest.raises(ValueError, match="inputs is empty"):
        s.validate("plot")
    s.plot.inputs, s.plot.coords = ["x"], "mlt"
    with pytest.raises(ValueError, match="coords"):
        s.validate("plot")


def test_run_plot_rejects_unknown_kind_and_missing_files(tmp_path):
    s = load_settings(None, {"plot": {"inputs": [str(tmp_path / "nothing_here")]}}, require="plot")
    with pytest.raises(ValueError, match="unknown plot kind"):
        run_plot("histogram", s)
    with pytest.raises(FileNotFoundError):
        run_plot("rti", s)
