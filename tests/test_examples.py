"""Smoke tests of examples/median_day_analysis on a 30-minute window of real bks data, and of `make analyse-day`."""

import datetime as dt
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "median_day_analysis"
DATA = Path(__file__).parent / "data" / "bks_20250208"
START, END = dt.datetime(2025, 2, 8, 3, 0), dt.datetime(2025, 2, 8, 3, 30)

sys.path.insert(0, str(EXAMPLES))
import analysis_day  # noqa: E402
import analysis_flags  # noqa: E402
import analysis_more  # noqa: E402
import plot_day  # noqa: E402

from amgeo_sd.pipelines import run_dbscan_model, run_median_filter  # noqa: E402


@pytest.fixture(scope="module")
def work(tmp_path_factory, fitacf_files):
    if not DATA.is_dir():
        pytest.skip("bks_20250208 not unpacked")
    work = tmp_path_factory.mktemp("analysis") / "bks"
    files = sorted(DATA.glob("*.bz2"))
    run_median_filter(
        "bks", START, END, files=files, out_base=work, sim_id="day", kde=False, write_fitacf=False, argv=["t"]
    )
    run_dbscan_model("bks", START, END, files=files, out_base=work, sim_id="day-dbscan", write_fitacf=False, argv=["t"])
    return work


def common(work, *extra):
    return [
        "--rad",
        "bks",
        "--work",
        str(work),
        "--data",
        str(DATA),
        "--start",
        START.isoformat(),
        "--end",
        END.isoformat(),
        *extra,
    ]


@pytest.mark.realdata
def test_analysis_day_prints_tables_and_writes_sweep(work, capsys):
    analysis_day.main(common(work, "--thresholds", "0.5,0.7", "--jobs", "1"))
    out = capsys.readouterr().out
    for section in ("retention by DBSCAN cluster class", "retention by fitacf ground-scatter flag", "threshold sweep"):
        assert section in out
    sweep = pd.read_csv(work / "sweep_bks.csv")
    assert sweep["thresh"].tolist() == [0.5, 0.7]
    assert sweep["kept%"].iloc[0] > sweep["kept%"].iloc[1]  # a stricter threshold keeps fewer echoes
    assert sweep["noise_removed%"].iloc[0] < sweep["noise_removed%"].iloc[1]


@pytest.mark.realdata
def test_other_analyses_run(work, capsys):
    analysis_more.main(common(work))
    out = capsys.readouterr().out
    assert "range edges" in out and "flag agreement" in out and "velocity change" in out
    analysis_flags.main(common(work))
    out = capsys.readouterr().out
    assert "Ribiero cluster flag" in out and "median rule says GS but Ribiero says IS" in out


@pytest.mark.realdata
def test_plot_day_makes_the_three_figures(work, capsys):
    plot_day.main(common(work, "--beam", "12", "--dpi", "60"))
    made = sorted(p.name for p in (work / "figures").glob("*.png"))
    assert made == ["rti_bks_b12_dbscan.png", "rti_bks_b12_raw_vs_median.png", "threshold_tradeoff_bks.png"]
    assert all((work / "figures" / n).stat().st_size > 5000 for n in made)


def test_missing_outputs_give_a_helpful_message(tmp_path, fitacf_files):  # reads the raw data first
    with pytest.raises(SystemExit, match="make median"):
        analysis_day.main(["--rad", "bks", "--work", str(tmp_path), "--data", str(DATA)])


def make(*args):
    return subprocess.run(["make", *args], cwd=ROOT, capture_output=True, text=True)


def test_make_analyse_day_dry_run_and_usage():
    ok = make("-n", "analyse-day", "RAD=gbr", "DAY=2025-02-08")
    assert ok.returncode == 0
    for part in (
        "median RAD=gbr",
        "SID=day ",
        "dbscan RAD=gbr",
        "SID=day-dbscan",
        "analysis_day.py",
        "analysis_more.py",
        "analysis_flags.py",
        "plot_day.py",
        "--work data/analysis/gbr",
        "tests/data/gbr_20250208",
    ):
        assert part in ok.stdout, part
    bad = make("analyse-day")
    assert bad.returncode == 2 and "usage: make analyse-day" in bad.stdout + bad.stderr
