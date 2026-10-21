"""The package builds into a wheel that holds everything a `pip install` user needs."""

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import amgeo_sd
from amgeo_sd.cli import main
from amgeo_sd.config import load_settings

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheel")
    cmd = [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--no-build-isolation", "-w", str(out), "-q"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=out)
    if res.returncode:
        pytest.skip(f"cannot build a wheel here (needs setuptools>=77): {res.stderr[-300:]}")
    (path,) = out.glob("amgeo_sd-*.whl")
    return path


def test_wheel_contains_the_package_and_the_typing_marker(wheel):
    names = set(zipfile.ZipFile(wheel).namelist())
    for expected in (
        "amgeo_sd/__init__.py",
        "amgeo_sd/py.typed",
        "amgeo_sd/cli.py",
        "amgeo_sd/api_doc.py",
        "amgeo_sd/io/fitacf.py",
        "amgeo_sd/filters/median.py",
        "amgeo_sd/clustering/dbscan.py",
        "amgeo_sd/plotting/rti.py",
        "amgeo_sd/geo/fov.py",
    ):
        assert expected in names, expected
    assert not [n for n in names if n.startswith(("tests", "_bkp", "examples", "data"))]


def test_wheel_metadata(wheel):
    z = zipfile.ZipFile(wheel)
    meta = z.read(next(n for n in z.namelist() if n.endswith("METADATA"))).decode()
    assert f"Version: {amgeo_sd.__version__}" in meta
    assert "License-Expression: MIT" in meta
    assert "Requires-Python: >=3.12" in meta
    for dep in ("pydarn", "pydarnio", "aacgmv2", "cartopy", "xarray", "scikit-learn"):
        assert f"Requires-Dist: {dep}" in meta, dep
    entry = z.read(next(n for n in z.namelist() if n.endswith("entry_points.txt"))).decode()
    assert "amgeo-sd = amgeo_sd.cli:main" in entry


def test_declared_dependencies_are_the_imported_ones():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].lower() for d in deps}
    assert not names & {"h5py", "tables", "pyyaml", "pysolar"}, "unused dependency"
    assert {"cartopy", "aacgmv2", "pydarn", "pydarnio"} <= names


def test_init_config_writes_a_file_the_loader_accepts(tmp_path, capsys):
    path = tmp_path / "mine.toml"
    assert main(["init-config", "-o", str(path)]) == 0
    assert main(["init-config", "-o", str(path)]) == 1  # refuses to overwrite
    assert main(["init-config", "-o", str(path), "--force"]) == 0
    text = path.read_text()
    for key in ("thresh", "edge_gates", "min_samples", "velocity_threshold", "backend", "netcdf", "inputs"):
        assert key in text
    s = load_settings(path, {"run": {"rad": "bks", "start": "2025-02-08T03:00", "end": "2025-02-08T04:00"}})
    assert s.median.thresh == 0.6 and s.dbscan.min_samples == 8
