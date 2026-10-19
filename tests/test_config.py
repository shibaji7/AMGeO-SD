import datetime as dt
from pathlib import Path

import pytest

from amgeo_sd.config import Input, Median, Output, Run, Settings, dump_settings_toml, load_settings

CONFIGS = Path(__file__).parent.parent / "configs"


def test_defaults_file_matches_dataclass_defaults():
    s = load_settings(CONFIGS / "default.toml")
    d = Settings()
    assert s.median == d.median and s.output == d.output
    assert (s.run.sim_id, s.run.out, s.run.jobs) == (d.run.sim_id, d.run.out, d.run.jobs)
    assert s.input.paths == [] and s.input.ftype == "fitacf"


def test_demo_configs_load():
    for name, rad in (("median_bks_demo", "bks"), ("median_gbr_demo", "gbr")):
        s = load_settings(CONFIGS / f"{name}.toml")
        assert s.run.rad == rad and s.run.start == dt.datetime(2025, 2, 8, 3) and s.input.paths


def test_precedence_defaults_file_overrides(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text(
        '[run]\nrad = "gbr"\nstart = 2025-02-08T00:00:00\nend = 2025-02-08T01:00:00\njobs = 2\n[median]\nthresh = 0.5\n'
    )
    s = load_settings(f, {"run": {"jobs": 8, "rad": None}, "median": {"kde": False}})
    assert (s.run.rad, s.run.jobs, s.median.thresh, s.median.kde) == ("gbr", 8, 0.5, False)


@pytest.mark.parametrize(
    "text,msg",
    [
        ('[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\nthreshold = 1\n', "unknown key"),
        ('[runs]\nrad = "bks"\n', "unknown table"),
        ('[run]\nrad = "bks"\nstart = 2025-02-08T05:00:00\nend = 2025-02-08T04:00:00\n', "before start"),
        (
            '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[median]\nkde = "yes"\n',
            "true or false",
        ),
        (
            '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[median]\nlth = 0.9\nuth = 0.1\n',
            "lth < uth",
        ),
        (
            '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[output]\nnetcdf = false\nfitacf = false\n',
            "at least one",
        ),
        ('[run]\nrad = "bks"\n', "missing required"),
        ('[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\njobs = 2.5\n', "whole number"),
    ],
)
def test_invalid_configs_are_rejected_with_a_clear_message(tmp_path, text, msg):
    f = tmp_path / "bad.toml"
    f.write_text(text)
    with pytest.raises(ValueError, match=msg):
        load_settings(f)


def test_dump_is_annotated_and_round_trips(tmp_path):
    s = Settings(
        run=Run("bks", dt.datetime(2025, 2, 8, 3, 0, 0, 500), dt.datetime(2025, 2, 8, 4), "S1", "out dir", 3),
        input=Input(["a b.bz2", "dir"], None, "fitacf3"),
        median=Median(0.6, 0.3, 0.1, 0.9, False, True, 2.5, 7),
        output=Output(False, True),
    )
    text = dump_settings_toml(
        s,
        provenance={"command": 'python -m amgeo_sd "x"', "software_version": "0.1.0"},
        result={"n_scans": 3, "nc_file": "/tmp/a.nc"},
    )
    assert text.count("   # ") >= 19 and "[provenance]" in text and "[result]" in text  # a comment per parameter
    assert "rad    = " in text and "kde" in text and "false" in text
    f = tmp_path / "p.toml"
    f.write_text(text)
    back = load_settings(f)
    assert back.to_dict() == s.to_dict()


def test_dump_marks_unset_values_as_comments():
    text = dump_settings_toml(Settings())
    assert "# rad " in text and "(unset)" in text
