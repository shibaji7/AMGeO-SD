"""Geometry port checked against arrays produced by the legacy code (tests/golden)."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from amgeo_sd.geo import hardware, radar_fov

GOLDEN = np.load(Path(__file__).parent / "golden" / "fov_legacy_20250208.npz")
DATE = dt.datetime(2025, 2, 8)


@pytest.mark.parametrize("rad", ["bks", "gbr"])
def test_fov_matches_legacy(rad):
    fov = radar_fov(rad, DATE)
    np.testing.assert_allclose(fov["glat"], GOLDEN[f"{rad}_glat"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fov["glon"], GOLDEN[f"{rad}_glon"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fov["gazm"], GOLDEN[f"{rad}_gazm"], rtol=0, atol=1e-9, equal_nan=True)


def test_hardware_and_shapes():
    hdw = hardware("bks", DATE)
    assert (hdw.beams, hdw.gates) == (24, 110)
    assert hardware("gbr", DATE).gates == 100
    fov = radar_fov("gbr", DATE, azm=False)
    assert fov["glat"].shape == (17, 101) and "gazm" not in fov
    assert fov["beams"][-1] == 16 and fov["gates"][-1] == 100


def test_fov_is_finite_and_range_increases_along_beams():
    for rad in ("bks", "gbr"):
        fov = radar_fov(rad, DATE)
        assert np.isfinite(fov["glat"]).all() and np.isfinite(fov["glon"]).all()
        site = hardware(rad, DATE).geographic
        la, lo = np.radians(fov["glat"]), np.radians(fov["glon"])
        s_la, s_lo = np.radians(site.lat), np.radians(site.lon)
        a = np.sin((la - s_la) / 2) ** 2 + np.cos(la) * np.cos(s_la) * np.sin((lo - s_lo) / 2) ** 2
        dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
        # the standard virtual-height model switches branch near gate 14, which pulls the ground
        # range back by <10 km once per beam (legacy behaviour, also present in the golden arrays)
        steps = np.diff(dist_km, axis=1)
        assert ((steps <= 0).sum(axis=1) <= 1).all() and steps.min() > -10
        assert 100 < dist_km[:, 0].min() and dist_km[:, -1].max() < 6000
