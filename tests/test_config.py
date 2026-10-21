import datetime as dt
import re
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


def test_geometry_table(tmp_path):
    f = tmp_path / "g.toml"
    f.write_text(
        '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[geometry]\nbackend = "pydarn"\naltitude = 250\n'
    )
    s = load_settings(f)
    assert (s.geometry.backend, s.geometry.altitude) == ("pydarn", 250.0)
    assert load_settings(f, {"geometry": {"backend": "legacy"}}).geometry.backend == "legacy"
    assert "[geometry]" in dump_settings_toml(s) and "pydarn" in dump_settings_toml(s)
    f.write_text(f.read_text().replace("pydarn", "davitpy"))
    with pytest.raises(ValueError, match="backend"):
        load_settings(f)


def test_median_defaults_and_edge_gates_validation(tmp_path):
    d = Settings().median
    assert (d.thresh, d.edge_gates) == (0.6, "legacy")
    f = tmp_path / "e.toml"
    f.write_text(
        '[run]\nrad = "bks"\nstart = 2025-02-08T03:00:00\nend = 2025-02-08T04:00:00\n[median]\nedge_gates = "missing"\nthresh = 0.7\n'
    )
    s = load_settings(f)
    assert (s.median.edge_gates, s.median.thresh) == ("missing", 0.7)
    assert re.search(r'edge_gates\s*=\s*"missing"', dump_settings_toml(s))
    f.write_text(f.read_text().replace("missing", "clip"))
    with pytest.raises(ValueError, match="edge_gates"):
        load_settings(f)


def test_extra_fields_setting_defaults_to_off_and_is_set_by_toml_and_flag(tmp_path, capsys):
    from amgeo_sd.cli import main

    assert Settings().output.extra_fields is False
    toml = tmp_path / "p.toml"
    toml.write_text("[output]\nextra_fields = true\n")
    assert load_settings(
        toml, {"run": {"rad": "bks", "start": "2025-02-08T03:00", "end": "2025-02-08T04:00"}}
    ).output.extra_fields
    for cmd in ("median", "dbscan"):
        base = [cmd, "--print-config", "-r", "bks", "-s", "2025-02-08T03:00", "-e", "2025-02-08T04:00"]
        assert main(base) == 0 and '"extra_fields": false' in capsys.readouterr().out
        assert main([*base, "--extra-fields"]) == 0 and '"extra_fields": true' in capsys.readouterr().out
