"""Functions and wrapper functions for making and formatting map plots in
north polar stereographic map projections.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import tools

import posterproxy as psp


# Define here default appearance/style parameters to be used dynamically in
# functions below. Then, they can be overridden by, e.g., .style_ms.py:

default_ice_edge_colors         = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
default_ice_edge_linestyles     = ["-"]

default_track_colors            = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
default_track_linewidths        = [mpl.rcParams["lines.linewidth"]]
default_track_linestyles        = ["-"]

default_track_markers           = ["o"]
default_track_markers_s         = [9.]
default_track_markers_fc        = ["line"]
default_track_markers_ec        = ["none"]

default_cbar_kw                 = {"orientation": "horizontal", "extend": "both"}

default_grid_lats               = np.arange(60., 90., 5.)
default_grid_lons               = np.arange(0., 360., 30.)
default_grid_lats_lon_lims      = [np.array([0.0, 360.0]) for k in range(len(default_grid_lats))]
default_grid_lons_lat_lims      = [50., 89.5]

default_grid_lat_labels         = False
default_grid_lat_labels_lats    = np.array([65., 70., 75., 80.])
default_grid_lat_labels_lons    = np.array([315.]*len(default_grid_lat_labels_lats))
default_grid_lat_labels_theta_0 = 9.
default_grid_lat_labels_fmt     = r"$%.0f\degree$N"

default_grid_line_kw            = {}
default_grid_lat_labels_text_kw = {"fontsize": .5*mpl.rcParams["font.size"]}

default_land_patch_kw = {}
default_coast_line_kw = {}


# Z-order of plot elements (list from top to bottom):
#
# Note special element 'rasterization': if ax.set_rasterization_zorder() is set
# everything below this zorder is rasterized in SVG/PDF output; everything above
# remains as vector.
#
zorder_elements = [
    "legend",
    "vectors_atm",
    "track",
    "legend_qv",
    "sector_boundary_line",
    "grid",
    "contour_atm",
    "text_seas_label",
    "coastlines",
    "land",
    "vectors_ice",
    "contour_ice",
    "contour_ocean",
    "rasterization",
    "pcm_ice",
    "pcm_ocean"
]


def get_zorder(element):
    if element in zorder_elements:
        # Factor of 10 provides some 'wiggle room' between each defined element
        # so that ad-hoc adjustments can be made (e.g., zorder+1)
        return 10*(len(zorder_elements) - zorder_elements.index(element))
    else:
        print(f"plotting.maps: Warning: \'{element}\' not in list")
        return 1


def pcm_kw(levels, cmap, extend="both", zorder=get_zorder("pcm_ice")):
    """Returns dictionary of keyword arguments to be passed to pcolormesh()."""
    return {"shading": "flat", "zorder": zorder,
            **tools.discrete_cmap_for_pcolormesh(cmap, levels,
               set_over=extend in ["both", "max"],
               set_under=extend in ["both", "min"])}


def add_ice_edge_contours(axs, x, y, aice_data, levels=[.15], t_indices=[0,-1],
                          colors=None, linestyles=None, zorder=None, **kwargs):
    """Plot sea ice edge (or other quantity) contours at start and end dates
    on one or more axes.


    Parameters
    ----------
    axs : matplotlib.Axes or array of such
        The axes(s) on which to plot.

    x, y : arrays (nj, ni)
        Coordinate variables.

    aice : array (nt, nj, ni)
        Sea ice concentration (or other variable) data as a function of time
        and coordinates.


    Optional parameters
    -------------------
    levels : list of float (default: [.15])
        The levels of aice to draw contours.

    t_indices : length-nk list of int (default: [0, -1])
        Time indices (axis 0 of aice) to draw contours at.

    colors, linestyles : length-nv list of matplotlib color and linestyle
                         identifiers, or None (default).
        Colors and linestyles to draw start (element 0) and end (element 1)
        contours. Set to None to use defaults.

    zorder : float or None (default)
        The zorder to draw contours at. If None, use default.

    Additional keyword arguments are passed to plt.contour().
    """

    if type(axs) not in [np.ndarray]:
        axs_use = np.array(axs)
    else:
        axs_use = axs

    colors     = tools._set_kw_cycler(colors    , default_ice_edge_colors    )
    linestyles = tools._set_kw_cycler(linestyles, default_ice_edge_linestyles)

    if zorder is None:
        zorder = get_zorder("contour_ice")

    for ax in axs_use.flatten():
        for k in range(len(t_indices)):
            ax.contour(x, y, aice_data[t_indices[k],:,:], levels=levels,
                       colors=colors[k%len(colors)], zorder=zorder,
                       linestyles=linestyles[k%len(linestyles)], **kwargs)


def add_cyclone_tracks(axs, x_tracks, y_tracks, jt_markers=None, colors=None,
                       linewidths=None, linestyles=None, markers=None,
                       markers_s=None, markers_fc=None, markers_ec=None,
                       zorder=None, **kwargs):
    """Plot one or more cyclone tracks on one or more axes.


    Parameters
    ----------
    axs : matplotlib.Axes or array of such
        The axes(s) on which to plot.

    x_tracks, y_tracks : length-nk list of array
        Coordinates as a function of time for each track, where nk is the
        number of tracks to plot (each of which may have a different number
        of coordinates).


    Optional parameters
    -------------------
    jt_markers : length-nk list of int, or None
        A time index for each track at which to draw a marker. If None,
        do not draw markers at all.

    colors, linewidths, linestyles : length-nk list of matplotlib color,
                                     linewidth, and linestyle identifiers,
                                     or None (default).
        Colors and linestyles to draw each track. Set to None to use defaults.

    markers, markers_s, markers_fc, markers_ec
        As above but for the markers (if drawn): marker symbols, sizes,
        facecolors and edgecolors respectively. The last two can be set
        to 'line' to match the corresponding track line color. All can be
        set to None to use defaults.

    zorder : float or None (default)
        The zorder to draw contours at. If None, use default.

    Additional keyword arguments are passed to plt.plot().
    """

    if type(axs) not in [np.ndarray]:
        axs_use = np.array(axs)
    else:
        axs_use = axs

    colors     = tools._set_kw_cycler(colors    , default_track_colors    )
    linestyles = tools._set_kw_cycler(linestyles, default_track_linestyles)
    linewidths = tools._set_kw_cycler(linewidths, default_track_linewidths)

    markers    = tools._set_kw_cycler(markers   , default_track_markers   )
    m_s        = tools._set_kw_cycler(markers_s , default_track_markers_s )
    m_fc       = tools._set_kw_cycler(markers_fc, default_track_markers_fc)
    m_ec       = tools._set_kw_cycler(markers_fc, default_track_markers_ec)

    if zorder is None:
        zorder = get_zorder("track")

    for ax in axs.flatten():
        for k in range(len(x_tracks)):
            ax.plot(x_tracks[k], y_tracks[k], zorder=zorder,
                    color=colors[k%len(colors)],
                    linewidth=linewidths[k%len(linewidths)],
                    linestyle=linestyles[k%len(linestyles)], **kwargs)

            if jt_markers is not None and jt_markers[k] is not None:

                if m_fc[k%len(m_fc)] == "line":
                    mfc_k = colors[k%len(colors)]
                else:
                    mfc_k = m_fc[k%len(m_fc)]

                if m_ec[k%len(m_ec)] == "line":
                    mec_k = colors[k%len(colors)]
                else:
                    mec_k = m_ec[k%len(m_ec)]

                ax.scatter(x_tracks[k][jt_markers[k]],
                           y_tracks[k][jt_markers[k]],
                           zorder=zorder+1, edgecolor=mec_k, facecolor=mfc_k,
                           marker=markers[k%len(markers)], s=m_s[k%len(m_s)])


def add_quiver_keys(axs, qvs, values, labels, rect_xy1=(.985, .985),
                    rect_width=.25, rect_height=.105, arrow_frac=.25,
                    **kwargs):
    """Add a quiver key like a legend on one or more axes.


    Parameters
    ----------
    axs : matplotlib.Axes or length-nq list of such
        The axes(s) on which to add the key(s).

    qvs : mpl.quiver.Quiver or length-nq of such
        The Quiver object(s) as returned by a call to plt.quiver().

    values : float or length-nq list of float
        The value(s) passed to the 'U' parameter of plt.quiverkey(),
        corresponding to the size(s) of the vector(s).

    labels : string or length-nq list of string
        The label(s) to be placed under the arrow(s) in the key(s).


    Optional parameters
    -------------------
    rect_xy1 : tuple of float, default = (0.985, 0.985)
        The coordinates of the upper-right corner of the legend background box
        in units of axes fraction.

    rect_width, rect_height : float, default 0.25 and 0.105 respectively
        The width and height of the background box in units of axes fraction.

    arrow_frac : float, default = 0.25
        Distance between the top of the legend background box and the y-
        coordinate of the arrow drawn in the key, as a fraction of the total
        box height.

    Additional keyword arguments are passed to mpl.Patches.Rectangle() to set
    the style of the legend background box. Default is a white box with black
    border and default zorder.

    """

    qk_rect_kw = tools._add_default_kw({**kwargs},
                                       {"facecolor": "w", "edgecolor": "k",
                                        "zorder": get_zorder("legend_qv")})

    # Note quiver key zorder is inherited from quiver itself;
    # doesn't seem possible to override

    # Common X and Y parameters (location of key) for ax.quiverkey():
    qk_X = rect_xy1[0] - .5*rect_width
    qk_Y = rect_xy1[1] - arrow_frac*rect_height

    # Common keyword arguments for ax.quiverkey():
    qk_kw = {"coordinates": "axes", "labelpos": "S", "labelsep": .05,
             "fontproperties": {"size": mpl.rcParams["font.size"] - 2.}}

    # Coordinates of lower-left corner of background box:
    rect_xy0 = (rect_xy1[0] - rect_width, rect_xy1[1] - rect_height)

    for ax, qv, value, label in zip(axs, qvs, values, labels):
        ax.quiverkey(qv, qk_X, qk_Y, value, label, **qk_kw)
        ax.add_patch(mpl.patches.Rectangle(rect_xy0, rect_width, rect_height,
                                           transform=ax.transAxes, **qk_rect_kw))


def psp_xy_gridlines(axs, psp_xy_kw, lons=None, lats=None, lons_lat_lims=None,
                     lats_lon_lims=None, line_kw={},  lat_labels=None,
                     lat_labels_lats=None, lat_labels_lons=None,
                     lat_labels_theta_0=None, lat_labels_fmt=None,
                     lat_labels_text_kw={}):
    """Add longitude/latitude gridlines to a map plot on one or more axes
    plotted using the posterproxy (psp) library.


    Parameters
    ----------
    axs : matplotlib.Axes or list of such
        The axes(s) on which to add the gridlines.

    psp_xy_kw : dictionary
        Dictionary containing the keyword arguments used to define the map
        projection in the posterproxy (psp) library.


    Optional parameters
    -------------------
    The following parameters are set to None by default, in which case the default
    values set in the module (if not updated by a style module) are used:

    lons, lats : list or array or None
        Longitudes and latitiudes at which to draw gridlines

    lons_lat_lims, lats_lon_lims : array (2,) or (n,2) or None
        Latitude limits for longitude gridlines and longitude limits for
        latitude gridlines, respectively, where n is the number of gridlines
        (i.e., length of lons or lats, respectively). If one set of limits
        passed, these are used for all of the relevant gridlines.

    line_kw : dictionary or None
        Matplotlib Line2D properties for the gridline appearance.

    lat_labels : bool, default = True
        Whether to add labels for latitude grid lines (controlled by following
        parameters).

    lat_labels_lats : list or array or None
        Latitudes to label (must be a subset of lats).

    lat_labels_lons : float or list or array or None
        Longitudes at which to write latitude labels. If one value, use this
        for all latitude labels. Otherwise, a different longitude per latitude
        label can be set.

    lat_labels_theta_0 : float or None
        For the first labelled latitude grid line (lat_labels_lats[0]), this is
        arc angle of the latitude circle to remove, centred on lat_labels_lons
        (or the first element of that) to make space for the latitude text label.
        The other latitude labels are scaled automatically so that the same arc
        *length* is removed in all labelled latitude grid lines.

    lat_labels_fmt : str or None
        String with % formatter for latitude labels.

    lat_labels_text_kw : dict or None
        Text properties used for latitude labels.
    """

    lons          = tools._set_default(lons         , default_grid_lons)
    lats          = tools._set_default(lats         , default_grid_lats)
    lons_lat_lims = tools._set_default(lons_lat_lims, default_grid_lons_lat_lims)
    lats_lon_lims = tools._set_default(lats_lon_lims, default_grid_lats_lon_lims)

    lat_labels    = tools._set_default(lat_labels, default_grid_lat_labels)

    if lat_labels:
        lat_labels_lats     = tools._set_default(lat_labels_lats    , default_grid_lat_labels_lats)
        lat_labels_lons     = tools._set_default(lat_labels_lons    , default_grid_lat_labels_lons)
        lat_labels_theta_0  = tools._set_default(lat_labels_theta_0 , default_grid_lat_labels_theta_0)
        lat_labels_fmt      = tools._set_default(lat_labels_fmt     , default_grid_lat_labels_fmt)

    if line_kw is None:
        line_kw = default_grid_line_kw
    else:
        line_kw = tools._add_default_kw(line_kw, default_grid_line_kw)

    if lat_labels_text_kw is None:
        lat_labels_text_kw = default_grid_lat_labels_text_kw
    else:
        lat_labels_text_kw = tools._add_default_kw(lat_labels_text_kw,
                                                   default_grid_lat_labels_text_kw)

    # Override z-orders and alignment parameters:
    line_kw["zorder"]            = get_zorder("grid")
    lat_labels_text_kw["zorder"] = line_kw["zorder"] + 1
    lat_labels_text_kw["ha"]     = "center"
    lat_labels_text_kw["va"]     = "center"

    if type(axs) not in [np.ndarray]:
        axs = np.array([axs])
        unlist = True
    else:
        unlist = False

    if lat_labels:

        nlats = len(lat_labels_lats)

        # ** Note this code should be in psp package but is a bit experimental
        #    at the time of writing **
        #
        # Need the radii in (x,y):
        lats_x, lats_y = psp.lonlat_to_xy_npsp(
            np.array([90. for j in range(nlats)]), lat_labels_lats, **psp_xy_kw)

        lats_radii = np.sqrt(lats_x**2 + lats_y**2)

        # Want the same arc length L 'free' (no grid line drawn) for each
        # latitude. Set for the first latitude, which defines the arc angle 
        # free (theta) that can be scaled for other latitudes (r_j*theta_j):
        theta = np.array([lat_labels_theta_0 * lats_radii[0]/lats_radii[j]
                          for j in range(nlats)])

        for j in range(nlats):
            xj, yj = psp.lonlat_to_xy_npsp(np.array([lat_labels_lons[j]]),
                                           lat_labels_lats[j], **psp_xy_kw)

            for ax in axs.flatten():
                ax.annotate(lat_labels_fmt % lat_labels_lats[j], (xj, yj),
                            rotation=lat_labels_lons[j], **lat_labels_text_kw)

            # Work out the longitude limits (if required -- that is, if this
            # label corresponds to a gridline):
            if int(lat_labels_lats[j]) in lats.astype(int):
                lats_lon_lims[list(lats).index(lat_labels_lats[j])] = \
                    np.array([lat_labels_lons[j] + .5 * theta[j],
                              lat_labels_lons[j] - .5 * theta[j]])

        psp.xy_longitude_gridlines(axs, lons=lons, lat_lims=lons_lat_lims,
                                   line_kw=line_kw, psp_xy_kw=psp_xy_kw)

        psp.xy_latitude_gridlines(axs, lats=lats, lon_lims=lats_lon_lims,
                                  line_kw=line_kw, psp_xy_kw=psp_xy_kw)

    else:  # no latitude labels => just plot usual circular grid lines:

        psp.xy_gridlines(axs, lons=lons, lat_lims=lons_lat_lims,
                         lats=lats, line_kw=line_kw, psp_xy_kw=psp_xy_kw)

    if unlist:
        axs = axs[0]


def psp_xy_land_overlay(axs, psp_xy_kw, scale="110m", land_patch_kw=None,
                        coast_line_kw=None):
    """Add land overlay to a map plot on one or more axes plotted using the
    posterproxy (psp) library. This is a wrapper to psp.xy_land_overlay().

    Parameters
    ----------
    axs : matplotlib.Axes or list of such
        The axes(s) on which to add land/coastlines.

    psp_xy_kw : dictionary
        Dictionary containing the keyword arguments used to define the map
        projection in the posterproxy (psp) library.

    Optional parameters
    -------------------
    scale : str {'110m', '50m'}, default = '110m'
        The Natural Earth data scale to use.

    land_patch_kw : dict or None
        Matplotlib Patch properties for the land overlay. If None, uses the
        defaults set in the module or as overridden in a style module.

    coast_line_kw : dict or None
        Matplotlib Line2D properties for the coastline overlay. If None, uses
        the defaults set in the module or as overriden in a style module.

    Note that the zorder of land and coastlines is always taken from the module
    definition of zorders for all elements (i.e., 'zorder' is ignored in the
    two kw dictionaries above).
    """

    if land_patch_kw is None:
        land_patch_kw = default_land_patch_kw
    else:
        land_patch_kw = tools._add_default_kw(land_patch_kw, default_land_patch_kw)

    if coast_line_kw is None:
        coast_line_kw = default_coast_line_kw
    else:
        coast_line_kw = tools._add_default_kw(coast_line_kw, default_coast_line_kw)

    # Override z-orders:
    land_patch_kw["zorder"] = get_zorder("land")
    coast_line_kw["zorder"] = get_zorder("coastlines")

    psp.xy_land_overlay(axs, scale=scale, psp_xy_kw=psp_xy_kw,
                        land_patch_kw=land_patch_kw, coast_line_kw=coast_line_kw)


def set_colorbars(sm, cbar_axs, cbar_tick_labels,
        signed_cbar_tick_labels=True, cbar_tick_labels_fmt="{:.1f}",
        rasterized_cbar_solids=False, cbar_kw={}):
    """Create colorbars from specified mappables on specified axes.
    

    Parameters
    ----------
    sm : length-N list of matplotlib.cm.ScalarMappable
        E.g., as returned by ax.pcolormesh().

    cbar_axs : length-N list or array of matplotlib axes.
        The axes from which to create colorbars corresponding to each mappable.

    cbar_tick_labels : bla


    Optional parameters
    -------------------
    signed_cbar_tick_labels : bool or length-N list of bool, default = True
        Whether to include the sign (- or +, none for 0) on tick labels (True)
        or not (False). One value passed implies the behaviour for all
        colorbars, otherwise a list of values gives per-colorbar behaviour.

    cbar_tick_labels_fmt : str or length-N list of str, default = '{:.1f}'
        String with format placeholder for tick labels. Either one value passed
        to be used for all, otherwise a list for per-colorbar formatter.

    rasterized_cbar_solids : bool or length-N list of bool, default = False
        Whether to rasterize the color parts of the colorbar. Either one value
        passed to set the behaviour for all colorbars, otherwise a list of
        values to set per-colorbar behaviour.

    cbar_kw : dict or length-N list of dict, default = {}
        Other keyword arguments passed to plt.colorbar().


    Returns
    -------
    cbars : length-N list of matplotlib.figure.Figure.colorbar instances
        The created colorbars.

    """

    n_cbars = len(cbar_axs)

    if type(signed_cbar_tick_labels) in [bool]:
        signed_cbar_tick_labels = [signed_cbar_tick_labels
                                   for j in range(n_cbars)]

    if type(cbar_tick_labels_fmt) in [str]:
        cbar_tick_labels_fmt = [cbar_tick_labels_fmt]*n_cbars

    if type(rasterized_cbar_solids) in [bool]:
        rasterized_cbar_solids = [rasterized_cbar_solids]*n_cbars


    if type(cbar_kw) in [dict]:
        cbar_kw = [tools._add_default_kw(cbar_kw, default_cbar_kw)
                   for j in range(n_cbars)]
    elif type(cbar_kw) in [list]:
        cbar_kw = [tools._add_default_kw(cbar_kw[j%len(cbar_kw)], default_cbar_kw)
                   for j in range(n_cbars)]

    cbars = np.array([plt.colorbar(sm[j], cax=cbar_axs[j], **cbar_kw[j])
                      for j in range(n_cbars)])

    for j in range(n_cbars):

        tools.set_cbar_axis_ticks(cbars[j], cbar_tick_labels[j],
            labels_fmt=cbar_tick_labels_fmt[j],
            signed_labels=signed_cbar_tick_labels[j])

        cbars[j].outline.set_linewidth(mpl.rcParams["axes.linewidth"])
        cbars[j].outline.set_zorder(1.E10)
        cbars[j].ax.tick_params(which="both", size=0)
        cbars[j].solids.set_rasterized(rasterized_cbar_solids[j])

    return cbars
