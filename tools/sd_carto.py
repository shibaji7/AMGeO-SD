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

class SDCarto(GeoAxes):
    name = "sdcarto"

    def __init__(self, *args, **kwargs):
        self.supported_coords = [ "geo", "aacgmv2", "aacgmv2_mlt" ]
        if "coords" in kwargs and kwargs.pop("coords") is not None:
            self.coords = kwargs.pop("coords")
            if self.coords not in self.supported_coords:
                err_str = "Coordinates not supported, choose from : "
                for _n,_sc in enumerate(self.supported_coords):
                    if _n + 1 != len(self.supported_coords): err_str += _sc + ", "
                    else: err_str += _sc
                raise TypeError(err_str)
        else: self.coords = "geo"
        if "map_projection" in kwargs and kwargs.pop("map_projection") is not None: self.map_projection = kwargs.pop("map_projection")
        else: self.map_projection = cartopy.crs.PlateCarree()
        if "rad" in kwargs and kwargs.pop("rad") is not None: self.rad = kwargs.pop("rad")
        else: self.rad = "bks"
        if "plot_date" in kwargs and kwargs.pop("plot_date") is not None: self.plot_date = kwargs.pop("plot_date")
        else:
            if self.coords == "aacgmv2" or self.coords == "aacgmv2_mlt":
                raise TypeError("Need to provide a date using "plot_date" keyword for aacgmv2 plotting")
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

    def grid_on(self, tx=cartopy.crs.PlateCarree(), draw_labels=[True, True, True, True], 
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
    
    def overlay_radar(self, tx=cartopy.crs.PlateCarree(), zorder=2, markerColor="darkblue", markerSize=15, fontSize=10, font_color="k",
            xOffset=None, yOffset=-0.1, annotate=True):
        """ Adding the radar location """
        self.hdw = pydarn.read_hdw_file(self.rad)
        print(" Radar:", self.hdw.geographic.lon, self.hdw.geographic.lat)
        self.scatter([self.hdw.geographic.lon], [self.hdw.geographic.lat], s=markerSize, marker="D",
                color=markerColor, zorder=zorder, transform=tx)
        nearby_rad = [["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
                            ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"]]
        if annotate:
            if self.rad in nearby_rad[0]: xOff, ha = 0.1 if not xOffset else xOffset, 0
            elif self.rad in nearby_rad[1]: xOff, ha = -0.1 if not xOffset else xOffset, 1
            else: xOff, ha = 0.0, 0.5
            lon, lat = self.hdw.geographic.lon, self.hdw.geographic.lat
            x, y = self.projection.transform_point(lon, lat, src_crs=tx)
        return
    
    def overlay_fov(self, tx=cartopy.crs.PlateCarree(), maxGate=75, rangeLimits=None, beamLimits=None,
            model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
            fovObj=None, zorder=2, lineColor="k", lineWidth=1, ls="-"):
        """ Overlay radar FoV """
        self.maxGate = maxGate
        lcolor = lineColor
        from numpy import transpose, ones, concatenate, vstack, shape
        self.hdw = pydarn.read_hdw_file(self.rad)
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
    
    def enum(self, bounds=[-120, -70, 25, 70], text_coord=False, dtype=None):
        if bounds is not None: self.set_extent(bounds)
        if text_coord: 
            if self.coords == "geo": self.text(1.05, 0.7, "Geographic Coordinates", horizontalalignment="center",
                                               verticalalignment="center", transform=self.transAxes, rotation=90)
        self.text(0.1, 1.02, "Rad: "+self.rad, horizontalalignment="center",
                verticalalignment="center", transform=self.transAxes)
        if dtype is not None: self.text(1.03, 0.1, dtype, ha="center", va="center", transform=self.transAxes, rotation=90)
        if self.plot_date: self.text(0.75, 1.02, self.plot_date.strftime("%Y-%m-%d %H:%M") + " UT", horizontalalignment="center",
                verticalalignment="center", transform=self.transAxes)
        return

    def add_dn_terminator(self, **kwargs):
        """ Adding day night terminator """
        from cartopy.feature.nightshade import Nightshade
        if self.plot_date:
            ns_feature = Nightshade(self.plot_date, alpha=0.2)
            super().add_feature(feature, **kwargs)
        return

    def overlay_radar_data(self, df, p_name = "v", tx=cartopy.crs.PlateCarree(), 
                           p_max=100, p_min=-100, p_step=10, p_ub=9999, p_lb=-9999,
                           cmap=matplotlib.pyplot.get_cmap("Spectral"), 
                           add_colorbar=True, colorbar_label="Velocity [m/s]", 
                           gflg_mask=True, **kwargs):
        """ 
            Adding radar data
            df: dataframe object
        """
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
        #params, gflgs = dat[p_name], dat["gflg"]
        #for idb, idg, par, gflg in zip(idbs, idgs, params, gflgs):
        #    idb = np.array(idb)[np.array(idg) < nrange]
        #    par = np.array(par)[np.array(idg) < nrange]
        #    gflg = np.array(gflg)[np.array(idg) < nrange]
        #    idg = np.array(idg)[np.array(idg) < nrange]
        #    if len(par) > 0: 
        #        Px[idb, np.round(idg).astype(int)] = par
        #        Gs[idb, np.round(idg).astype(int)] = gflg
        if not gflg_mask: 
            Px = np.ma.masked_invalid(Px)
            self.pcolormesh(lons, lats, Px, transform=tx, cmap=cmap,
                          vmax=p_max, vmin=p_min, **kwargs)    
        else:
            Gs = Gs.astype(bool)
            Pi, Pg = np.ma.masked_invalid(np.ma.masked_where(Gs, Px)),\
                        np.ma.masked_invalid(np.ma.masked_where(np.logical_not(Gs), Px))
            self.pcolormesh(lons, lats, Pi, transform=tx, cmap=cmap,
                            vmax=p_max, vmin=p_min, **kwargs)
            gcmap = matplotlib.colors.ListedColormap(["0.6"])
            self.pcolormesh(lons, lats, Pg, transform=tx, cmap=gcmap,
                            vmax=p_max, vmin=p_min, **kwargs)
        if add_colorbar: 
            self._add_colorbar(p_ranges, cmap, label=colorbar_label)
            if gflg_mask: self._add_key_specific_colorbar(gcmap)
        return
    
    def _add_colorbar(self, bounds, colormap, label=""):
        """ Add a colorbar to the right of an axis. """
        pos = self.get_position()
        cpos = [pos.x1 + 0.035, pos.y0 + 0.25*pos.height,
                0.01, pos.height * 0.5]            # this list defines (left, bottom, width, height)
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
    
    def overlay_fitacfp_radar_data(self, dat, p_name = "v", tx=cartopy.crs.PlateCarree(), 
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
                0.01, 0.02]            # this list defines (left, bottom, width, height)
        cax = matplotlib.pyplot.gcf().add_axes(cpos)
        cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=colormap,
                spacing="uniform",
                orientation="vertical")
        cb2.ax.tick_params(size=0)
        cb2.ax.text(0.5,-.3,label,fontdict={"size":6}, ha="center", va="center")
        # Remove the outer bounds in tick labels
        cb2.ax.set_yticklabels([])
        return
    
    
    def get_aacgm_geom(self, feature, out_height=300. ):
        new_i = []
        # cartopy.feature.COASTLINE
        for _n,i in enumerate(feature.geometries()):
            aa = mapping(i)
            mag_list = []
            geo_coords = aa["coordinates"]
            for _ni, _list in enumerate(geo_coords):
                mlon_check_jump_list = []
                split_mag_list = None
                if len(_list) == 1:
                    _loop_list = _list[0]
                else:
                    _loop_list = _list
                for _ngc, _gc in enumerate(_loop_list):
                    _mc = aacgmv2.get_aacgm_coord(_gc[1], _gc[0], out_height, self.plot_date)
                    if numpy.isnan(_mc[0]):
                        continue 
                    mlon_check_jump_list.append( _mc[1] )
                    if self.coords == "aacgmv2":
                        mag_list.append( (_mc[1], _mc[0]) )
                    else:
                        if _mc[2]*15. > 180.:
                            mag_list.append( (_mc[2]*15.-360., _mc[0]) )
                        else:
                            mag_list.append( (_mc[2]*15., _mc[0]) )
                # check for unwanted jumps
                mlon_check_jump_list = numpy.array( mlon_check_jump_list )

                jump_arr = numpy.diff( mlon_check_jump_list )
                bad_inds = numpy.where( numpy.abs(jump_arr) > 10.)[0]
                # delete the range of bad values
                # This is further complicated because
                # in some locations mlon jumps from -177 to +178
                # and this causes jumps in the maps! To deal with 
                # this we'll split arrays of such jumps 
                # (these jumps typically have just one bad ind )
                # and make them into two seperate entities (LineStrings)
                # so that shapely will treat them as two seperate boundaries!
                if len(bad_inds) > 0:
                    if len(bad_inds) > 1:
                        mag_list = [i for j, i in enumerate(mag_list) if j-1 not in numpy.arange(bad_inds[0], bad_inds[1])]
                    else:
                        split_mag_list = mag_list[bad_inds[0]+1:]
                        mag_list = mag_list[:bad_inds[0]+1]
                mag_coords = tuple(mag_list)
                if len(mag_list) > 1:
                    new_i.append( mag_coords )
                if split_mag_list is not None:
        #             print(split_mag_list)
                    if len(split_mag_list) > 1:
                        new_i.append( tuple(split_mag_list) )

        aacgm_coast = MultiLineString( new_i )
        return aacgm_coast
    
    def mark_latitudes(self, lat_arr, lon_location=45, **kwargs):
        """
        mark the latitudes
        Write down the latitudes on the map for labeling!
        we are using this because cartopy doesn't have a 
        label by default for non-rectangular projections!
        """
        if isinstance(lat_arr, list):
            lat_arr = numpy.array(lat_arr)
        else:
            if not isinstance(lat_arr, numpy.ndarray):
                raise TypeError('lat_arr must either be a list or numpy array')
        # make an array of lon_location
        lon_location_arr = numpy.full( lat_arr.shape, lon_location )
        proj_xyz = self.projection.transform_points(\
                            cartopy.crs.Geodetic(),\
                            lon_location_arr,\
                            lat_arr
                            )
        # plot the lats now!
        out_extent_lats = False
        for _np,_pro in enumerate(proj_xyz[..., :2].tolist()):
            # check if lats are out of extent! if so ignore them
            lat_lim = self.get_extent(crs=cartopy.crs.Geodetic())[2::]
            if (lat_arr[_np] >= min(lat_lim)) and (lat_arr[_np] <= max(lat_lim)):
                self.text( _pro[0], _pro[1], str(lat_arr[_np]), **kwargs)
            else:
                out_extent_lats = True
        if out_extent_lats:
            print( "some lats were out of extent ignored them" )

    def mark_longitudes(self, lon_arr=numpy.arange(-180,180,60), **kwargs):
        """
        mark the longitudes
        Write down the longitudes on the map for labeling!
        we are using this because cartopy doesn't have a 
        label by default for non-rectangular projections!
        This is also trickier compared to latitudes!
        """
        if isinstance(lon_arr, list):
            lon_arr = numpy.array(lon_arr)
        else:
            if not isinstance(lon_arr, numpy.ndarray):
                raise TypeError('lat_arr must either be a list or numpy array')
        # get the boundaries
        [x1, y1], [x2, y2] = self.viewLim.get_points()
        bound_lim_arr = []
        right_bound = LineString(([-x1, y1], [x2, y2]))
        top_bound = LineString(([x1, -y1], [x2, y2]))
        bottom_bound = LineString(([x1, y1], [x2, -y2]))
        left_bound = LineString(([x1, y1], [-x2, y2]))
        plot_outline = MultiLineString( [\
                                        right_bound,\
                                        top_bound,\
                                        bottom_bound,\
                                        left_bound\
                                        ] )
        # get the plot extent, we'll get an intersection
        # to locate the ticks!
        plot_extent = self.get_extent(cartopy.crs.Geodetic())
        line_constructor = lambda t, n, b: numpy.vstack(\
                        (numpy.zeros(n) + t, numpy.linspace(b[2], b[3], n))\
                        ).T
        for t in lon_arr:
            xy = line_constructor(t, 30, plot_extent)
            # print(xy)
            proj_xyz = self.projection.transform_points(\
                            cartopy.crs.PlateCarree(), xy[:, 0], xy[:, 1]\
                            )
            xyt = proj_xyz[..., :2]
            ls = LineString(xyt.tolist())
            locs = plot_outline.intersection(ls)
            if not locs:
                continue
            # we need to get the alignment right
            # so get the boundary closest to the label
            # and plot it!
            closest_bound =min( [\
                            right_bound.distance(locs),\
                            top_bound.distance(locs),\
                            bottom_bound.distance(locs),\
                            left_bound.distance(locs)\
                            ] )
            if closest_bound == right_bound.distance(locs):
                ha = 'left'
                va = 'top'
            elif closest_bound == top_bound.distance(locs):
                ha = 'left'
                va = 'bottom'
            elif closest_bound == bottom_bound.distance(locs):
                ha = 'left'
                va = 'top'
            else:
                ha = 'right'
                va = 'top'
            if self.coords == "aacgmv2_mlt":
                marker_text = str(int(t/15.))
            else:
                marker_text = str(t)
            self.text( locs.bounds[0],locs.bounds[1], marker_text, ha=ha, va=va)

register_projection(SDCarto)


