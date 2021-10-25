#!/usr/bin/env python

"""sd_carto.py: utility module for Costom Carto py geoaxes to plot data on aacgmv2 coordinates."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Kunduri, B.; Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import numpy as np

import cartopy
from cartopy.mpl.geoaxes import GeoAxes
import aacgmv2
import numpy
from shapely.geometry import  MultiLineString, mapping, LineString, Polygon
from descartes.patch import PolygonPatch
from matplotlib.projections import register_projection
import copy
import datetime as dt
import pydarn

import sys
sys.path.append("tools/")
import rad_fov
import utils

class SDCarto(GeoAxes):
    name = "sdcarto"

    def __init__(self, *args, **kwargs):
        self.supported_coords = [ "geo", "aacgmv2", "aacgmv2_mlt" ]
        if "coords" in kwargs and kwargs["coords"] is not None:
            self.coords = kwargs.pop("coords")
            if self.coords not in self.supported_coords:
                err_str = "Coordinates not supported, choose from : "
                for _n,_sc in enumerate(self.supported_coords):
                    if _n + 1 != len(self.supported_coords): err_str += _sc + ", "
                    else: err_str += _sc
                raise TypeError(err_str)
        else: self.coords = "geo"
        if "map_projection" in kwargs and kwargs["map_projection"] is not None: self.map_projection = kwargs.pop("map_projection")
        else: self.map_projection = cartopy.crs.PlateCarree()
        if "rad" in kwargs and kwargs["rad"] is not None: self.rad = kwargs.pop("rad")
        if "plot_date" in kwargs and kwargs["plot_date"] is not None: self.plot_date = kwargs.pop("plot_date")
        else:
            if self.coords == "aacgmv2" or self.coords == "aacgmv2_mlt":
                raise TypeError("Need to provide a date using 'plot_date' keyword for aacgmv2 plotting")
        super().__init__(map_projection=self.map_projection,*args, **kwargs)
        return
    
    def overaly_coast_lakes(self, resolution="50m", color="black", **kwargs):
        """  Overlay AACGM coastlines and lakes  """
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        # overaly coastlines
        feature = cartopy.feature.NaturalEarthFeature("physical", "coastline",
                resolution, **kwargs)
        self.add_feature( cartopy.feature.COASTLINE, **kwargs )
        self.add_feature( cartopy.feature.LAKES, **kwargs )
        return

    def coastlines(self,resolution="50m", color="black", **kwargs):
        # details!
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        feature = cartopy.feature.NaturalEarthFeature("physical", "coastline",
                resolution, **kwargs)
        return self.add_feature(feature, **kwargs)
    
    def add_feature(self, feature, **kwargs):
        if "edgecolor" not in kwargs: kwargs["edgecolor"] = "black"
        if self.coords == "geo":
            super().add_feature(feature, **kwargs)
        else:
            aacgm_geom = self.get_aacgm_geom(feature)
            aacgm_feature = cartopy.feature.ShapelyFeature(aacgm_geom, cartopy.crs.Geodetic(), **kwargs)
            kwargs["facecolor"] = "none"
            super().add_feature(aacgm_feature, **kwargs)
        return

    def grid_on(self, tx=cartopy.crs.PlateCarree(), draw_labels=[False, False, True, True], 
                linewidth=0.5, color="gray", alpha=0.5, linestyle="--"):
        """ Adding grids to map """
        import matplotlib.ticker as mticker
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl = self.gridlines(crs=tx, draw_labels=True,
                linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)
        gl.top_labels = draw_labels[0]
        gl.bottom_labels = draw_labels[2]
        gl.right_labels = draw_labels[1]
        gl.left_labels = draw_labels[3]
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 15))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 15))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        return
    
    def overlay_radar(self, rad, tx=cartopy.crs.PlateCarree(), zorder=2, markerColor="darkblue", markerSize=15, fontSize=10, font_color="k",
            xOffset=None, yOffset=-0.1, annotate=True):
        """ Adding the radar location """
        if hasattr(self, rad) and rad is None: rad = self.rad 
        self.hdw = pydarn.read_hdw_file(rad)
        print(" Radar:", self.hdw.geographic.lon, self.hdw.geographic.lat)
        self.scatter([self.hdw.geographic.lon], [self.hdw.geographic.lat], s=markerSize, marker="D",
                color=markerColor, zorder=zorder, transform=tx)
        nearby_rad = [["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
                            ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"]]
        if annotate:
            if rad in nearby_rad[0]: xOff, ha = 0.1 if not xOffset else xOffset, 0
            elif rad in nearby_rad[1]: xOff, ha = -0.1 if not xOffset else xOffset, 1
            else: xOff, ha = 0.0, 0.5
            lon, lat = self.hdw.geographic.lon, self.hdw.geographic.lat
            x, y = self.projection.transform_point(lon, lat, src_crs=tx)
        return
    
    def overlay_fov(self, rad=None, tx=cartopy.crs.PlateCarree(), maxGate=75, rangeLimits=None, beamLimits=None,
            model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
            fovObj=None, zorder=2, lineColor="k", lineWidth=1, ls="-"):
        """ Overlay radar FoV """
        if hasattr(self, rad) and rad is None: rad = self.rad 
        self.maxGate = maxGate
        lcolor = lineColor
        from numpy import transpose, ones, concatenate, vstack, shape
        self.hdw = pydarn.read_hdw_file(rad)
        sgate = 0
        egate = self.hdw.gates if not maxGate else maxGate
        ebeam = self.hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        self.rad_fov = rad_fov.CalcFov(hdw=self.hdw, ngates=egate)
        xyz = self.projection.transform_points(tx, self.rad_fov.lonFull, self.rad_fov.latFull)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
            x[ebeam, egate:sgate:-1],
            x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
            y[ebeam, egate:sgate:-1],
            y[ebeam:sbeam:-1, sgate]))
        self.plot(contour_x, contour_y, color=lcolor, zorder=zorder, linewidth=lineWidth, ls=ls)
        if fovColor:
            contour = transpose(vstack((contour_x, contour_y)))
            polygon = Polygon(contour)
            patch = PolygonPatch(polygon, facecolor=fovColor, edgecolor=fovColor, alpha=fovAlpha, zorder=zorder)
            self.add_patch(patch)
        return
    
    def enum(self, add_date=True, add_coord=True, dtype=None):
        if add_coord: self.text(0.01, 1.05, "Coords: %s"%self.coords, horizontalalignment="left",
                                verticalalignment="center", transform=self.transAxes)
        if dtype is not None: self.text(1.03, 0.1, dtype, ha="center", va="center", transform=self.transAxes, rotation=90)
        if add_date: self.text(0.99, 1.05, self.plot_date.strftime("%Y-%m-%d %H:%M") + " UT", horizontalalignment="right",
                verticalalignment="center", transform=self.transAxes)
        return

    def add_dn_terminator(self, **kwargs):
        """ Adding day night terminator """
        from cartopy.feature.nightshade import Nightshade
        if self.plot_date:
            ns_feature = Nightshade(self.plot_date, alpha=0.2)
            super().add_feature(feature, **kwargs)
        return
    
    def data_lay(self, df, rf, p_name, to, fm, idx=None, gflg_mask=None):
        """
        Data scatter-plots
        """
        o = df[df[gflg_mask]==idx] if (idx is not None) and (gflg_mask is not None) else df.copy()
        lons, lats = rf.lonFull, rf.latFull
        Xb, Yg, Px = utils.get_gridded_parameters(o, xparam="bmnum", yparam="slist", zparam=p_name, rounding=True)
        Xb, Yg = Xb.astype(int), Yg.astype(int)
        lons, lats = lons[Xb.ravel(), Yg.ravel()].reshape(Xb.shape), lats[Xb.ravel(), Yg.ravel()].reshape(Xb.shape)
        XYZ = to.transform_points(fm, lons, lats)
        Px = np.ma.masked_invalid(Px)
        return XYZ[:,:,0], XYZ[:,:,1], Px

    def overlay_radar_data(self, rad, df, to, fm, p_name = "v", 
                           p_max=100, p_min=-100, p_step=10, p_ub=9999, p_lb=-9999,
                           cmap="Spectral", add_colorbar=True, colorbar_label="Velocity [m/s]", 
                           gflg_mask=None, **kwargs):
        """ 
            Adding radar data
            df: dataframe object
        """
        cmap = matplotlib.pyplot.get_cmap("jet") if cmap is None else matplotlib.pyplot.get_cmap(cmap)
        gflg_map={
            1: {"col": matplotlib.colors.ListedColormap(["0.5"]), "key":"gs"},
            -1: {"col": matplotlib.colors.ListedColormap(["0.2"]), "key":"us"},
            0: {"col": cmap, "key": None},
        }
        nbeam = np.max(np.max(df.bmnum)) + 1
        if self.maxGate: nrange = self.maxGate
        else: nrange = np.max(np.max(dat["slist"])) + 1
        hdw = pydarn.read_hdw_file(rad)
        rf = rad_fov.CalcFov(hdw=hdw, ngates=nrange, nbeams=nbeam)
        
        p_ranges = list(range(p_min, p_max + 1, p_step))
        p_ranges.insert(0, p_lb)
        p_ranges.append(p_ub)
        
        if gflg_mask:
            for key in gflg_map.keys():
                cmap = gflg_map[key]["col"]
                X, Y, Px = self.data_lay(df, rf, p_name, to, fm, idx=key, gflg_mask=gflg_mask)
                self.pcolormesh(X, Y, Px.T, transform=to, cmap=cmap, vmax=p_max, vmin=p_min, **kwargs)
                if add_colorbar and gflg_map[key]["key"] is None: self._add_colorbar(p_ranges, cmap, label=colorbar_label)
                else: self._add_key_specific_colorbar(cmap, label=gflg_map[key]["key"], idh=key)
        else:
            X, Y, Px = self.data_lay(df, rf, p_name, to, fm)
            self.pcolormesh(X, Y, Px.T, transform=to, cmap=cmap, vmax=p_max, vmin=p_min, **kwargs)
            if add_colorbar: self._add_colorbar(p_ranges, cmap, label=colorbar_label)
            
#         if gflg_mask:
#             _, _, Is = utils.get_gridded_parameters(df[df[gflg_mask]==0], xparam="bmnum", 
#                                                     yparam="slist", zparam=gflg_mask, rounding=True)
#             Is = Is.astype(bool)
#             Pi = np.ma.masked_invalid(np.ma.masked_where(Is, Px))
#             self.pcolormesh(XYZ[:,:,0], XYZ[:,:,1], Pi.T, transform=to, cmap=cmap,
#                             vmax=p_max, vmin=p_min, **kwargs)
            
#             unique_ids = np.unique(df[gflg_mask])
#             for idx in np.unique(df[gflg_mask]):
#                 if idx in gflg_map.keys():
#                     _, _, Ss = utils.get_gridded_parameters(df[df[gflg_mask]==idx], xparam="bmnum", 
#                                                             yparam="slist", zparam=gflg_mask, rounding=True)
#                     Ss = Ss.astype(bool)
#                     Ps = np.ma.masked_invalid(np.ma.masked_where(np.ma.masked_where(np.logical_not(Ss==idx), Px)))
#                     gcmap = matplotlib.colors.ListedColormap([gflg_map[idx]["col"]])
#                     self.pcolormesh(XYZ[:,:,0], XYZ[:,:,1], Ps.T, transform=to, cmap=gcmap,
#                                 vmax=p_max, vmin=p_min, **kwargs)
#         else:
#             Px = np.ma.masked_invalid(Px)
#             self.pcolormesh(XYZ[:,:,0], XYZ[:,:,1], Px.T, cmap=cmap, transform=to, alpha=0.8,
#                           vmax=p_max, vmin=p_min, **kwargs)
#        if add_colorbar: 
#            self._add_colorbar(p_ranges, cmap, label=colorbar_label)
#             if gflg_mask: 
#                 unique_ids = np.unique(df[gflg_mask])
#                 for idx in np.unique(df[gflg_mask]):
#                     if idx in gflg_map.keys():
#                         gcmap = matplotlib.colors.ListedColormap([gflg_map[idx]["col"]])
#                         self._add_key_specific_colorbar(gcmap, label=gflg_map[idx]["key"], idh=idx)
        return
    
    def _add_colorbar(self, bounds, colormap, label=""):
        """ Add a colorbar to the right of an axis. """
        pos = self.get_position()
        cpos = [pos.x1 + 0.035, pos.y0 + 0.25*pos.height,
                0.03, pos.height * 0.5]            # this list defines (left, bottom, width, height)
        cax = matplotlib.pyplot.gcf().add_axes(cpos)
        norm = matplotlib.colors.BoundaryNorm(bounds[::2], colormap.N)
        cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=colormap,
                norm=norm,
                ticks=bounds[::2],
                spacing="uniform",
                orientation="vertical")
        cb2.set_label(label)
        # Remove the outer bounds in tick labels
        ticks = [str(i) for i in bounds[::2]]
        ticks[0], ticks[-1] = "", ""
        cb2.ax.set_yticklabels(ticks)
        return
    
    def overlay_fitacfp_radar_data(self, df, p_name = "v", tx=cartopy.crs.PlateCarree(), 
                           p_max=100, p_min=-100, p_step=10, p_ub=9999, p_lb=-9999,
                           cmap=matplotlib.pyplot.get_cmap("Spectral"), 
                           add_colorbar=True, colorbar_label="Velocity [m/s]", gflg_key="gflg", 
                           gflg_map={1:{"key":"gs", "col":"0.8"}, 2:{"key":"us", "col":"0.5"}}, **kwargs):
        """ 
            Adding radar data
            dat: dict()-> with "bmnum", "slist" and "v" or other parameter in list of list format
        """
        if len(dat["bmnum"]) == 0: return
        nbeam = np.max(np.max(dat["bmnum"])) + 1
        if self.maxGate: nrange = self.maxGate
        else: nrange = np.max(np.max(dat["slist"])) + 1
        hdw = pydarn.read_hdw_file(self.rad)
        rf = rad_fov.CalcFov(hdw=hdw, ngates=nrange, nbeams=nbeam)
        lons, lats = rf.lonFull, rf.latFull
        
        Xb, Yg, Px = utils.get_gridded_parameters(df, xparam="bmnum", yparam="slist", zparam=zparam)
        lons, lats = lons[Xb.ravel(), Yg.ravel()], lats[Xb.ravel(), Yg.ravel()]
        p_ranges = list(range(p_min, p_max + 1, p_step))
        p_ranges.insert(0, p_lb)
        p_ranges.append(p_ub)       
                
        #Px, Gs = np.zeros((nbeam, nrange))*np.nan, np.zeros((nbeam, nrange))
        #idbs, idgs = dat["bmnum"], dat["slist"]
        #params, gflgs = dat[p_name], dat[gflg_key]
        #for idb, idg, par, gflg in zip(idbs, idgs, params, gflgs):
        #    idb = np.array(idb)[np.array(idg) < nrange]
        #    par = np.array(par)[np.array(idg) < nrange]
        #    gflg = np.array(gflg)[np.array(idg) < nrange]
        #    idg = np.array(idg)[np.array(idg) < nrange]
        #    if len(par) > 0: 
        #        Px[idb, np.round(idg).astype(int)] = par
        #        Gs[idb, np.round(idg).astype(int)] = gflg
        unique = max(gflg_map.keys())
        for uq in range(int(unique) + 1):
            Puq = np.ma.masked_invalid(np.ma.masked_where(np.logical_not(Gs==uq), Px))
            if uq > 0: cmap = matplotlib.colors.ListedColormap([gflg_map[uq]["col"]])
            self.pcolormesh(lons, lats, Puq, transform=tx, cmap=cmap,
                        vmax=p_max, vmin=p_min, **kwargs)
            if add_colorbar:
                if uq==0: self._add_colorbar(p_ranges, cmap, label=colorbar_label)
                else: self._add_key_specific_colorbar(cmap, label=gflg_map[uq]["key"], idh=uq)
        return
    
    def _add_key_specific_colorbar(self, colormap, label="gs", idh=1):
        """ Add a colorbar to the right of an axis. """
        pos = self.get_position()
        cpos = [pos.x1 + 0.035, pos.y0 + ((idh-1)*.1+0.05)*pos.height,
                0.03, 0.02]            # this list defines (left, bottom, width, height)
        cax = matplotlib.pyplot.gcf().add_axes(cpos)
        cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=colormap,
                spacing="uniform",
                orientation="vertical")
        cb2.ax.tick_params(size=0)
        cb2.ax.text(0.5,-.3,label,fontdict={"size":10}, ha="center", va="center")
        # Remove the outer bounds in tick labels
        cb2.ax.set_yticklabels([])
        return
    
register_projection(SDCarto)


