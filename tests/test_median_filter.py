"""Median-filter tests: vectorised implementation vs. the legacy-loop oracle, on real data."""

import datetime as dt

import numpy as np
import pytest
from reference_median import reference_filter

from amgeo_sd.filters import MedianFilter, cluster_scan, discard_repeating_beams, kde_flag
from amgeo_sd.io import FitacfReader
from amgeo_sd.models import Beam, Scan

T0 = dt.datetime(2025, 1, 1)


def mk_beam(bm, t, gates, v=10.0, w=20.0, g=0, nrang=30):
    gates = np.asarray(gates, dtype=int)
    n = len(gates)
    return Beam(
        time=t,
        bmnum=bm,
        nrang=nrang,
        tfreq=10000.0,
        noise_sky=1.0,
        scan=int(bm == 0),
        slist=gates,
        v=np.full(n, v),
        v_e=np.ones(n),
        w_l=np.full(n, w),
        p_l=np.full(n, 5.0),
        gflg=np.full(n, g, np.int8),
    )


def mk_scan(i, beams):
    return Scan(beams=beams, stime=T0 + dt.timedelta(minutes=i)).update()


def block_scans(gates=range(10, 15), beams=range(3, 6), v=10.0, w=20.0, g=0):
    return [
        mk_scan(i, [mk_beam(b, T0 + dt.timedelta(minutes=i, seconds=b), gates, v, w, g) for b in beams])
        for i in range(3)
    ]


# ------------------------------------------------------------- synthetic, exact expectations
def test_uniform_block_survives_and_keeps_values():
    out = MedianFilter(cluster=False).filter_window(block_scans(v=10, w=20))
    assert [b.bmnum for b in out.beams] == [3, 4, 5]
    centre = out.beams[1]
    assert centre.slist.tolist() == list(range(10, 15))
    np.testing.assert_allclose(centre.v, 10)
    np.testing.assert_allclose(centre.extra["v_mad"], 0)
    assert (centre.gflg == 1).all()  # w=20 is not > -3*10+90=60 -> ground scatter
    assert (centre.extra["gflg_conv"][1:-1] == 0).all()  # neighbours all carry the fitacf flag 0


def test_fast_wide_spectra_is_ionospheric():
    out = MedianFilter(cluster=False).filter_window(block_scans(v=10, w=100))
    assert (out.beams[1].gflg == 0).all()  # w=100 > 60


def test_isolated_echo_is_removed():
    scans = block_scans()
    scans[1].beams.append(mk_beam(9, T0, [25]))
    scans[1].update()
    out = MedianFilter(cluster=False).filter_window(scans)
    assert 9 not in [b.bmnum for b in out.beams]


def test_conv_flag_follows_neighbour_ground_fraction():
    flt = MedianFilter(cluster=False, kde=False)
    # edge gates see empty neighbours (counted in the denominator), so only the interior is unambiguous
    assert (flt.filter_window(block_scans(g=1)).beams[1].extra["gflg_conv"][1:-1] == 1).all()
    assert (flt.filter_window(block_scans(g=0)).beams[1].extra["gflg_conv"][1:-1] == 0).all()
    mixed = block_scans(g=0)
    for b in mixed[0].beams + mixed[2].beams:  # previous and next scan: ground -> 30/55 of the weight = 0.55
        b.gflg[:] = 1
    mid = flt.filter_window(mixed).beams[1]
    assert (mid.extra["gflg_conv"][1:-1] == 2).all() and np.isnan(mid.extra["gflg_kde"]).all()


def edge_block_scans(gates):
    return [
        mk_scan(i, [mk_beam(b, T0 + dt.timedelta(minutes=i, seconds=b), gates, v=10.0) for b in (3, 4, 5)])
        for i in range(3)
    ]


def test_edge_gates_option_first_and_last_gate():
    """A block touching gate 0 and the last gate: the legacy rule can give those cells at most 40/55 = 73% of the weight."""
    nrang = 30
    for gates, edge_cells in ((range(0, 5), [0]), (range(nrang - 5, nrang), [nrang - 1])):
        scans = edge_block_scans(gates)
        for thresh, legacy_keeps in ((0.7, True), (0.8, False)):
            got = {}
            for mode in ("legacy", "missing"):
                out = MedianFilter(thresh=thresh, edge_gates=mode, cluster=False, kde=False).filter_window(scans)
                got[mode] = set(out.beams[1].slist.tolist())  # the middle beam has all neighbours
            assert set(edge_cells) <= got["missing"], "missing mode keeps the cell at the radar's range edge"
            assert (set(edge_cells) <= got["legacy"]) is legacy_keeps
            assert got["legacy"] <= got["missing"]


def test_edge_gates_changes_conv_denominator_at_the_edge_only():
    scans = edge_block_scans(range(0, 6))
    for s in scans:
        for b in s.beams:
            b.gflg[:] = 1  # all neighbours ground scatter
    a = MedianFilter(thresh=0.6, edge_gates="legacy", cluster=False, kde=False).filter_window(scans).beams[1]
    b = MedianFilter(thresh=0.6, edge_gates="missing", cluster=False, kde=False).filter_window(scans).beams[1]
    assert a.slist.tolist() == b.slist.tolist()
    assert a.extra["gflg_conv"][0] == 2.0 and b.extra["gflg_conv"][0] == 1.0  # 73% (undetermined) vs 100% (ground)
    np.testing.assert_array_equal(a.extra["gflg_conv"][1:-1], b.extra["gflg_conv"][1:-1])


def test_median_filter_defaults_and_edge_gate_validation():
    flt = MedianFilter()
    assert (flt.thresh, flt.edge_gates) == (0.6, "legacy")
    with pytest.raises(ValueError, match="edge_gates"):
        MedianFilter(edge_gates="clip")


def test_median_filter_input_validation():
    with pytest.raises(ValueError):
        MedianFilter().filter_window(block_scans()[:2])
    with pytest.raises(ValueError):
        MedianFilter(weights=np.ones((3, 3)))


def test_discard_repeating_beams():
    t = T0
    scan = mk_scan(0, [mk_beam(5, t, [1]), mk_beam(2, t, [4]), mk_beam(5, t, [9]), mk_beam(7, t, [])])
    out = discard_repeating_beams(scan)
    assert [b.bmnum for b in out.beams] == [2, 5]
    assert out.beams[1].slist.tolist() == [1]  # first of the repeated beams wins


def test_cluster_scan_labels_blob_and_noise():
    blob = [mk_beam(b, T0, range(10, 14)) for b in range(3, 7)]
    stray = mk_beam(15, T0, [60])
    scan = cluster_scan(mk_scan(0, blob + [stray]), eps=2, min_samples=5)
    assert all((b.extra["clabel"] == 0).all() for b in scan.beams[:4])
    assert scan.beams[-1].extra["clabel"].tolist() == [-1]


def test_kde_flag_is_deterministic_and_bounded():
    gfx = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=float)
    a = kde_flag(gfx, (0.2, 0.8), 0.25, np.random.default_rng(1))
    b = kde_flag(gfx, (0.2, 0.8), 0.25, np.random.default_rng(1))
    assert a == b and a in (0.0, 1.0, 2.0)
    assert kde_flag(np.ones(12), (0.2, 0.8), 0.25, np.random.default_rng(0)) == 1.0
    assert kde_flag(np.zeros(12), (0.2, 0.8), 0.25, np.random.default_rng(0)) == 0.0


# ------------------------------------------------------------- real data vs. legacy-loop oracle
@pytest.fixture(scope="session")
def real_scans(fitacf_files):
    """~12 minutes of consecutive scans per radar, from the first file of each radar-day folder."""
    out = {}
    for f in fitacf_files:
        if f.parent.name in out or not f.parent.name.endswith("20250208"):
            continue
        rad = f.parent.name.split("_")[0]
        t0 = dt.datetime.strptime(f.name[:13], "%Y%m%d.%H%M")
        if f.name[:13].endswith("0200") or f.name[:13].endswith("0201"):
            reader = FitacfReader(rad, t0, t0 + dt.timedelta(minutes=13), files=[f])
            out[f.parent.name] = [discard_repeating_beams(s) for s in reader.scans()]
    if not out:
        pytest.skip("no 20250208 radar-day folders with the 02:00 file")
    return out


def _as_dict(scan):
    return {
        b.bmnum: {
            "slist": b.slist,
            "v": b.v,
            "w_l": b.w_l,
            "p_l": b.p_l,
            "v_mad": b.extra["v_mad"],
            "gflg": b.gflg,
            "gflg_conv": b.extra["gflg_conv"],
            "gflg_kde": b.extra["gflg_kde"],
        }
        for b in scan.beams
    }


# (threshold, edge_gates): the legacy 0.7 and the new default 0.6, and the strict thresholds where the gate edge matters most
SETTINGS = [(0.7, "legacy"), (0.6, "legacy"), (0.6, "missing"), (0.8, "missing")]


@pytest.mark.realdata
@pytest.mark.parametrize("thresh,edge", SETTINGS)
@pytest.mark.parametrize("idx", [3, 7])
def test_vectorised_equals_legacy_loops(real_scans, idx, thresh, edge):
    for name, scans in real_scans.items():
        window = scans[idx - 1 : idx + 2]
        got = _as_dict(MedianFilter(thresh=thresh, edge_gates=edge, cluster=False).filter_window(window))
        want = reference_filter(window, thresh=thresh, edge_gates=edge)
        assert got.keys() == want.keys(), name
        assert sum(len(d["slist"]) for d in want.values()) > 40, f"{name}: window too empty to be a test"
        for bm in want:
            for key, ref in want[bm].items():
                np.testing.assert_allclose(got[bm][key], ref, rtol=0, atol=1e-9, err_msg=f"{name} beam {bm} {key}")


@pytest.mark.realdata
@pytest.mark.parametrize("thresh,must_gain", [(0.6, True), (0.8, False)])
def test_missing_edge_gates_only_change_the_first_and_last_gate(real_scans, thresh, must_gain):
    """Neighbour weights differ only for cells whose gate +-1 leaves the radar's range, so interior gates are identical
    and the edge gates can only gain echoes."""
    gained = 0
    for scans in real_scans.values():
        legacy = _as_dict(MedianFilter(thresh=thresh, edge_gates="legacy", cluster=False).filter_window(scans[2:5]))
        missing = _as_dict(MedianFilter(thresh=thresh, edge_gates="missing", cluster=False).filter_window(scans[2:5]))
        nrang = scans[3].beams[0].nrang
        for bm, m in missing.items():
            lg = legacy.get(bm, {"slist": np.empty(0, int)})
            interior_m, interior_l = (
                np.isin(x["slist"], range(1, nrang - 1)) if len(x["slist"]) else x["slist"] > 0 for x in (m, lg)
            )
            np.testing.assert_array_equal(m["slist"][interior_m], lg["slist"][interior_l])
            for key in ("v", "w_l", "p_l", "gflg"):
                np.testing.assert_array_equal(m[key][interior_m], lg[key][interior_l])
            edge_m = set(m["slist"][~interior_m].tolist())
            edge_l = set(lg["slist"][~interior_l].tolist()) if len(lg["slist"]) else set()
            assert edge_l <= edge_m
            gained += len(edge_m - edge_l)
    assert gained > 0 or not must_gain, "expected some first/last-gate echoes to be recovered"


@pytest.mark.realdata
def test_filter_scans_parallel_equals_serial_and_is_deterministic(real_scans):
    scans = next(iter(real_scans.values()))[:6]
    flt = MedianFilter()
    serial = flt.filter_scans(scans)
    parallel = flt.filter_scans(scans, n_jobs=2)
    again = flt.filter_scans(scans)
    assert len(serial) == len(scans) - 2
    for a, b, c in zip(serial, parallel, again, strict=True):
        assert [x.bmnum for x in a.beams] == [x.bmnum for x in b.beams] == [x.bmnum for x in c.beams]
        for x, y, z in zip(a.beams, b.beams, c.beams, strict=True):
            for key in ("v", "w_l", "slist", "gflg"):
                np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
                np.testing.assert_array_equal(getattr(x, key), getattr(z, key))
            np.testing.assert_array_equal(x.extra["gflg_kde"], y.extra["gflg_kde"])
            np.testing.assert_array_equal(x.extra["clabel"], y.extra["clabel"])


@pytest.mark.realdata
def test_filter_reduces_noise_and_keeps_most_structure(real_scans):
    for name, scans in real_scans.items():
        raw = sum(b.n_echoes for s in scans[2:5] for b in s.beams)
        out = MedianFilter(cluster=False, kde=False).filter_scans(scans[1:6])
        kept = sum(b.n_echoes for s in out for b in s.beams)
        assert 0.3 * raw < kept < raw, f"{name}: kept {kept} of {raw}"
