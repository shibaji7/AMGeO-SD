"""Field-of-view ("fan") maps of one scan, for fitacf and fitacf++ tables, in geographic or AACGM coordinates.

Port of the legacy ``FanPlots`` (``sd/fan_rti.py``, ``sd/plotlib.py``) and the ``SDCarto`` GeoAxes subclass
(``tools/sd_carto.py``). The custom registered axes class is gone: a plain cartopy axes plus helper
functions does the same, and the data layer no longer depends on ``descartes``. Each echo is drawn as the
quadrilateral of its range cell, using the legacy field-of-view geometry (:func:`amgeo_sd.geo.radar_fov`).

``coords="aacgmv2"`` converts cell corners, radar position and coastlines to AACGM (300 km for the data, 0 km for
the coastlines) and draws magnetic latitude/longitude on the same map projection.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence

import aacgmv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from amgeo_sd.geo import hardware, radar_fov
from amgeo_sd.plotting.style import FLAG_CLASSES, PARAMS, apply_style

COORDS = ("geo", "aacgmv2")
#: Fan colour scales (the legacy fan plots used +-30 m/s in 5 m/s steps for velocity).
FAN_RANGES = {"v": (-30, 30, 5), "w_l": (0, 100, 10), "p_l": (0, 30, 5), "v_mad": (0, 30, 5)}
GATE_HEIGHT_KM = 300.0


def select_scan(df: pd.DataFrame, date: dt.datetime) -> pd.DataFrame:
    """Rows of the scan whose first sounding is nearest to ``date`` (uses ``scnum`` when present)."""
    if df.empty:
        raise ValueError("no data to plot")
    if "scnum" not in df:
        raise ValueError("table has no 'scnum' column; build it with amgeo_sd.io.load_frame / scans_to_dataframe")
    starts = df.groupby("scnum")["time"].min()
    nearest = (starts - np.datetime64(date)).abs().idxmin()
    return df[df["scnum"] == nearest]


def _to_frame(lat, lon, coords: str, date: dt.datetime, height: float = GATE_HEIGHT_KM):
    """Geographic (lat, lon) -> the frame the map is drawn in."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if coords == "geo":
        return lat, lon
    mlat, mlon, _ = aacgmv2.convert_latlon_arr(lat.ravel(), lon.ravel(), height, date, method_code="G2A")
    return np.reshape(mlat, lat.shape), np.reshape(mlon, lon.shape)


def _projection(lat: np.ndarray, lon: np.ndarray, kind: str):
    ok = np.isfinite(lat) & np.isfinite(lon)
    clat, clon = float(np.mean(lat[ok])), float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(lon[ok]))))))
    if kind == "polar" or (kind == "auto" and clat > 70):
        return ccrs.NorthPolarStereo(central_longitude=clon)
    if kind in ("auto", "ortho"):
        return ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    raise ValueError(f"projection must be auto, polar or ortho, not {kind!r}")


class FanPlots:
    """A figure of ``nrows x ncols`` field-of-view maps.

    Parameters
    ----------
    coords : ``"geo"`` or ``"aacgmv2"``
        Coordinate frame of the map.
    projection : ``"auto"``, ``"polar"`` or ``"ortho"``
        ``auto`` is orthographic around the field of view, or polar stereographic above 70 deg latitude.
    """

    def __init__(
        self,
        figsize=(6, 6),
        nrows: int = 1,
        ncols: int = 1,
        dpi: int = 120,
        coords: str = "geo",
        projection: str = "auto",
    ):
        if coords not in COORDS:
            raise ValueError(f"coords must be one of {COORDS}")
        apply_style()
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.nrows, self.ncols, self.coords, self.projection_kind = nrows, ncols, coords, projection
        self._created = 0
        self._latlon = ccrs.PlateCarree()

    # ------------------------------------------------------------------ axes
    def _add_axis(
        self, date: dt.datetime, lat: np.ndarray, lon: np.ndarray, terminator: bool = False, coast: str = "110m"
    ):
        self._created += 1
        proj = _projection(lat, lon, self.projection_kind)
        ax = self.fig.add_subplot(self.nrows, self.ncols, self._created, projection=proj)
        ax.proj = proj  # data coordinates (x, y) of artists are in this projection
        ax.plot_date = date
        xyz = proj.transform_points(self._latlon, lon[np.isfinite(lon)], lat[np.isfinite(lat)])
        pad_x, pad_y = 0.12 * np.ptp(xyz[:, 0]), 0.12 * np.ptp(xyz[:, 1])
        ax.set_extent(
            [xyz[:, 0].min() - pad_x, xyz[:, 0].max() + pad_x, xyz[:, 1].min() - pad_y, xyz[:, 1].max() + pad_y],
            crs=proj,
        )
        self._coastlines(ax, date, coast)
        self._grid(ax)
        if terminator and self.coords == "geo":
            from cartopy.feature.nightshade import Nightshade

            ax.add_feature(Nightshade(date, alpha=0.2))
        return ax

    def _coastlines(self, ax, date: dt.datetime, resolution: str):
        if self.coords == "geo":
            ax.coastlines(resolution=resolution, linewidth=0.4)
            return
        feat = cfeature.NaturalEarthFeature("physical", "coastline", resolution)
        for geom in feat.geometries():
            for line in getattr(geom, "geoms", [geom]):
                lon, lat = (np.asarray(c) for c in line.xy)
                mlat, mlon = _to_frame(lat, lon, "aacgmv2", date, 0.0)
                bad = ~(np.isfinite(mlat) & np.isfinite(mlon))
                bad[1:] |= np.abs(np.diff(mlon)) > 180  # the line crosses the +-180 meridian of the new frame
                breaks = np.flatnonzero(bad)
                for seg_lat, seg_lon in zip(np.split(mlat, breaks), np.split(mlon, breaks), strict=True):
                    ok = np.isfinite(seg_lat) & np.isfinite(seg_lon)
                    if ok.sum() > 1:
                        ax.plot(seg_lon[ok], seg_lat[ok], transform=self._latlon, color="k", lw=0.4)

    def _grid(self, ax):
        try:
            gl = ax.gridlines(
                crs=self._latlon, draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
            )
            gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 15))
            gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 10))
            gl.top_labels = gl.right_labels = False
        except Exception as exc:  # label placement is not supported by every projection
            logger.debug(f"gridline labels unavailable: {exc}")
            ax.gridlines(crs=self._latlon, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    # ------------------------------------------------------------------ overlays
    def overlay_radar(self, ax, rad: str, date: dt.datetime, color: str = "darkblue"):
        """Mark the radar site and label it with the radar code."""
        site = hardware(rad, date).geographic
        lat, lon = _to_frame([site.lat], [site.lon], self.coords, date, 0.0)
        ax.scatter(lon, lat, s=25, marker="D", color=color, zorder=5, transform=self._latlon)
        ax.text(
            lon[0] + 1.5,
            lat[0] - 1.5,
            rad.upper(),
            color=color,
            transform=self._latlon,
            ha="left",
            va="top",
            fontsize=9,
        )

    def overlay_fov(self, ax, fov_lat: np.ndarray, fov_lon: np.ndarray, color: str = "k", lw: float = 1.0):
        """Outline of the field of view (all beams, gates 0..ngates)."""
        nb, ng = fov_lat.shape[0] - 1, fov_lat.shape[1] - 1
        idx = (
            [(0, g) for g in range(ng + 1)]
            + [(b, ng) for b in range(nb + 1)]
            + [(nb, g) for g in range(ng, -1, -1)]
            + [(b, 0) for b in range(nb, -1, -1)]
        )
        b, g = np.array(idx).T
        ax.plot(fov_lon[b, g], fov_lat[b, g], color=color, lw=lw, transform=self._latlon, zorder=4)

    # ------------------------------------------------------------------ data
    def plot_fov(
        self,
        df: pd.DataFrame,
        date: dt.datetime,
        rad: str,
        param: str = "v",
        gflg_mask: str | None = "gflg",
        add_colorbar: bool = True,
        title: str = "",
        vrange: tuple[float, float, float] | None = None,
        max_gate: int | None = None,
        terminator: bool = False,
    ):
        """Draw the scan nearest to ``date``: echoes coloured by ``param``; ground scatter (``gflg_mask`` == 1) gray.

        ``param`` may also be a flag column (``gflg``, ``gflg_conv``, ``gflg_kde``, ``gflg_ribiero``), drawn with the
        IS / GS / undetermined classes. Returns the axes.
        """
        scan = select_scan(df, date)
        t0 = pd.Timestamp(scan["time"].min()).to_pydatetime()
        nrang = int(scan["nrang"].max())
        first = scan.iloc[0]
        frang, rsep = float(first["frang"]), float(first["rsep"])
        if not (np.isfinite(frang) and np.isfinite(rsep)):  # e.g. NetCDF files without these variables
            logger.debug("frang/rsep missing in the data; assuming 180 / 45 km")
            frang, rsep = 180.0, 45.0
        fov = radar_fov(rad, t0, frang=frang, rsep=rsep, azm=False)
        ng = min(fov["glat"].shape[1] - 1, max_gate or nrang)
        fov_lat, fov_lon = _to_frame(fov["glat"][:, : ng + 1], fov["glon"][:, : ng + 1], self.coords, t0)
        ax = self._add_axis(t0, fov_lat, fov_lon, terminator)
        self.overlay_fov(ax, fov_lat, fov_lon)
        self.overlay_radar(ax, rad, t0)

        d = scan[scan["slist"] < ng]
        b, g = d["bmnum"].values.astype(int), d["slist"].values.astype(int)
        corners = [(b, g), (b + 1, g), (b + 1, g + 1), (b, g + 1)]
        quads = np.stack(
            [np.column_stack([fov_lon[bb, gg], fov_lat[bb, gg]]).reshape(len(d), 2) for bb, gg in corners], axis=1
        )
        xy = ax.proj.transform_points(self._latlon, quads[..., 0], quads[..., 1])[..., :2]
        values = d[param].values.astype(float)

        if param.startswith("gflg"):
            present = [c for c in FLAG_CLASSES if c in set(np.unique(values[np.isfinite(values)]).astype(int))]
            for cls in present:
                m = values == cls
                ax.add_collection(
                    mcoll.PolyCollection(xy[m], facecolor=FLAG_CLASSES[cls][0], edgecolor="none", zorder=3)
                )
            ax.legend(
                handles=[mpatches.Patch(color=FLAG_CLASSES[c][0], label=FLAG_CLASSES[c][1]) for c in present],
                loc="lower left",
                fontsize=7,
            )
        else:
            lo, hi, step = (
                vrange or FAN_RANGES.get(param) or (PARAMS[param].vmin, PARAMS[param].vmax, PARAMS[param].step)
            )
            bounds = np.arange(lo, hi + step / 2, step)
            cmap = plt.get_cmap(PARAMS[param].cmap if param in PARAMS else "viridis")
            norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="both")
            ground = (d[gflg_mask].values == 1) if gflg_mask and gflg_mask in d else np.zeros(len(d), bool)
            if ground.any():
                ax.add_collection(mcoll.PolyCollection(xy[ground], facecolor="0.5", edgecolor="none", zorder=3))
            pc = mcoll.PolyCollection(
                xy[~ground], array=values[~ground], cmap=cmap, norm=norm, edgecolor="none", zorder=3
            )
            ax.add_collection(pc)
            if add_colorbar:  # carved out of this panel, so neighbouring panels are not overlapped
                cb = self.fig.colorbar(pc, ax=ax, ticks=bounds[::2], shrink=0.6, pad=0.04, fraction=0.04)
                cb.set_label(PARAMS[param].label if param in PARAMS else param)
                if ground.any():
                    ax.legend(handles=[mpatches.Patch(color="0.5", label="GS")], loc="lower left", fontsize=7)
        ax.text(
            0.99, 1.02, f"{t0:%Y-%m-%d %H:%M} UT", transform=ax.transAxes, ha="right", fontweight="bold", fontsize=9
        )
        ax.text(0.01, 1.02, title or rad.upper(), transform=ax.transAxes, ha="left", color="r", fontsize=9)
        ax.text(0.5, 1.02, f"coords: {self.coords}", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=8)
        return ax

    def plot_layfovs(self, rads: Sequence[str], date: dt.datetime):
        """Outlines of several radars' fields of view on one map."""
        fovs = {r: radar_fov(r, date, azm=False) for r in rads}
        lat = np.concatenate([_to_frame(f["glat"], f["glon"], self.coords, date)[0].ravel() for f in fovs.values()])
        lon = np.concatenate([_to_frame(f["glat"], f["glon"], self.coords, date)[1].ravel() for f in fovs.values()])
        ax = self._add_axis(date, lat, lon)
        for r, f in fovs.items():
            flat, flon = _to_frame(f["glat"], f["glon"], self.coords, date)
            self.overlay_fov(ax, flat, flon)
            self.overlay_radar(ax, r, date)
        return ax

    def add_instrument(
        self, ax, lat: float, lon: float, color: str = "darkblue", marker: str = "D", ms: float = 40, name: str = ""
    ):
        """Mark another instrument at geographic (``lat``, ``lon``) and label it ``name`` (map must be in geo coordinates)."""
        ax.scatter([lon], [lat], s=ms, color=color, marker=marker, transform=self._latlon, zorder=5)
        ax.text(lon + 1, lat - 1, name, ha="center", va="center", transform=self._latlon, color="darkgreen")

    def save(self, path: str | os.PathLike, dpi: int | None = None) -> str:
        """Save the figure (format from the file extension) and return the path."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"saved FoV figure {path}")
        return str(path)

    def close(self) -> None:
        """Release the figure."""
        self.fig.clf()
        plt.close(self.fig)


def plot_fan(
    df: pd.DataFrame,
    rad: str,
    date: dt.datetime,
    params: Sequence[str] = ("v",),
    path: str | os.PathLike | None = None,
    coords: str = "geo",
    projection: str = "auto",
    kind: str = "",
    gflg_mask: str | None = "gflg",
    dpi: int = 120,
    **kwargs,
) -> FanPlots:
    """One map per entry of ``params`` for the scan nearest to ``date`` (side by side)."""
    fan = FanPlots(
        figsize=(5.5 * len(params) + 1, 5.5), nrows=1, ncols=len(params), dpi=dpi, coords=coords, projection=projection
    )
    for p in params:
        fan.plot_fov(df, date, rad, param=p, gflg_mask=gflg_mask, title=f"{rad.upper()} {kind}".strip(), **kwargs)
    if path:
        fan.save(path)
    return fan
