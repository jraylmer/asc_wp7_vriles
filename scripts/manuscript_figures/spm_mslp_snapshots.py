"""Create a multi-panel figure showing snapshots of mean sea level pressure
(MSLP) anomalies during the passage of a specified cyclone track. This is
used to generated supplemental Figure S2.
"""

from datetime import datetime as dt, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.data import atmo, cice, tracks
from src.plotting import maps, style_ms, tools

import posterproxy as psp


# --------------------------------------------------------------------------- #
# Some hard-coded plot parameters:
# =========================================================================== #
#
# MSLP anomaly contours and labels (latter must be a subset of the former):
psl_clev        = np.arange(-20., 20.01, 2.)
psl_clev_labels = psl_clev[1::2]

# Contour colours (MSLP and ice edge respectively):
psl_color       = "k"
iel_color       = "b"
# --------------------------------------------------------------------------- #


def parse_cmd_args():
    """Define and parse command line arguments using the argparse module.
    Returns the parsed arguments.
    """

    prsr = script_tools.argument_parser(usage="Cyclone track/MSLP snapshots")

    prsr.add_argument("-k", "--track-id", type=str, default=6716,
                      help="Track ID(s) to add to plot")

    prsr.add_argument("-y", "--track-year", type=int, default=2004,
                      help="Year of track (-k/--track-ids)")

    prsr.add_argument("-t", "--time-indices", type=int, nargs=4,
                      default=[6, 18, 30, 37],
                      help="Track time indices to show snapshots")

    # Options for the map projection (posterproxy package):
    prsr.add_argument("--extent-latitude", type=float, default=70.,
                      help="Outer latitude of plot")
    prsr.add_argument("--central_longitude", type=float, default=0.,
                      help="Central longitude of plot")
    prsr.add_argument("--xy-offset", type=float, nargs=2, default=(-.25, .25),
                      help="Translation of viewport in (x,y) space for plot")

    # Generic plotting command-line arguments (contains --savefig, etc.):
    script_tools.add_plot_cmd_args(prsr, fig_names="figS2")

    return prsr.parse_args()


def main():

    cmd = parse_cmd_args()
    cfg.set_config(*cmd.config)

    # Load and prepare track data:
    dates, lons, lats, vors= tracks.get_track_data(cmd.track_id, cmd.track_year)

    dt_start = dates[0]
    dt_end   = dates[-1]

    print(f"\nCyclone track ID        : {cmd.track_id}")
    print(f"Start date/time of track: {dt_start.strftime('%HZ %d %b %Y')}")
    print(f"Final date/time of track: {dt_end.strftime('%HZ %d %b %Y')}")
    print(f"Maximum vorticity       : {np.max(vors)*1.e5:.2f} x 10-5 s-1")
    print(f"        (which occurs at: {dates[np.argmax(vors)].strftime('%HZ %d %b %Y')})")

    jt_plot = [min(i, len(dates)) for i in cmd.time_indices]

    print("\nLoading CICE coordinate data")
    ulon, ulat, tlon, tlat = cice.get_grid_data(["ULON", "ULAT", "TLON", "TLAT"],
                                                slice_to_atm_grid=False)

    print("Loading CICE history data")
    date_cice, aice_cice = cice.get_history_data(["aice_d"], dt_min=dt_start,
        dt_max=dt_end, frequency="daily", set_miss_to_nan=True,
        slice_to_atm_grid=False)

    aice_cice = np.where(aice_cice[0] > 1., np.nan, aice_cice[0])
    aice_cice = np.where(aice_cice < 0., np.nan, aice_cice)

    # Time indices to plot sea ice/ice edge (it is daily resolution while
    # the tracks are 6 hourly):
    jt_plot_ice = [np.argmin([abs((j - dates[i]).total_seconds())
                              for j in date_cice])
                   for i in jt_plot]

    # Load raw atmospheric forcing data for sea level pressure (function
    # loads and subtracts the climatology data for psl automatically. Also
    # note raw JRA-55-do atmospheric reanalysis data is 3 hourly, although
    # the tracks are resolved to 6 hours, so we 'seltime' every 2 steps):
    #
    print("Loading atmospheric pressure data")
    dddd, plon, plat, psl = atmo.get_atmospheric_reanalysis_raw_data("psl",
        dt_min=dt_start, dt_max=dt_end, seltime=2, clip_lat=55.)

    # Transform lon/lat to north polar stereographic (x,y)
    # Arguments passed to posterproxy lonlat_to_xy_npsp() function:
    psp_xy_kw = {"extent_latitude"  : cmd.extent_latitude,
                 "central_longitude": cmd.central_longitude,
                 "xy_offset"        : cmd.xy_offset}

    xk, yk = psp.lonlat_to_xy_npsp(lons, lats, **psp_xy_kw)
    xu, yu = psp.lonlat_to_xy_npsp(ulon, ulat, **psp_xy_kw)
    xt, yt = psp.lonlat_to_xy_npsp(tlon, tlat, **psp_xy_kw)
    xp, yp = psp.lonlat_to_xy_npsp(plon, plat, **psp_xy_kw)


    fig, axs = plt.subplots(ncols=4, nrows=1,
                            figsize=(    style_ms.fig_width_double_column,
                                     .28*style_ms.fig_width_double_column ))

    psp.prepare_axes(axs)  # prepare for map plotting in psp (must come first)

    # Align the subplots [must come after prepare_axes which repositions axes]:
    tools.distribute_subplots(axs, l=.015, r=.015, t=.05, b=.0,
                                   s_hor=.015, s_ver=0.)

    # Add titles to subplots with the time step datetimes:
    tools.add_subplot_panel_titles(axs,
                                   titles=[dates[i].strftime("%HZ %d %b %Y")
                                           for i in jt_plot],
                                   title_kw={"pad": 4, "zorder": 10000})

    # Note: zorders are set in the wrapper functions so the order of plotting
    # different elements should not generally matter (see maps.py)
    #
    # Make the land/coastlines slightly paler than default to make contours
    # clearer. Also remove latitude labels (plot is crowded enough as it is):
    #
    maps.psp_xy_land_overlay(axs, psp_xy_kw,
                             land_patch_kw={"facecolor": [.90]*3},
                             coast_line_kw={"edgecolor": [.85]*3})
    maps.psp_xy_gridlines(axs, psp_xy_kw, lat_labels=False)

    for i in range(4):
        # MSLP anomaly contours:
        cs = axs[i].contour(xp, yp, psl[jt_plot[i],:,:], levels=psl_clev,
                            colors=psl_color, negative_linestyles="--",
                            linewidths=.35,
                            zorder=maps.get_zorder("contour_atm"))

        axs[i].clabel(cs, levels=psl_clev_labels, fontsize=3,
                      inline_spacing=3, fmt=lambda x: f"{abs(x):.0f}")

        # Ice edge contours on first frame (so only index 0 if i==0) and
        # current frame (indices [0,-1] for i>0):
        maps.add_ice_edge_contours(axs[:,np.newaxis][[i],:],
                                   xt, yt, aice_cice[:jt_plot_ice[i]+1,:,:],
                                   colors=[iel_color],
                                   t_indices=[0] if i==0 else [0,-1])

        maps.add_cyclone_tracks(axs[:,np.newaxis][[i],:],
                                [xk[:jt_plot[i]+1]], [yk[:jt_plot[i]+1]],
                                jt_markers=[jt_plot[i]])

    # Set figure metadata, window title for interactive use, both containing
    # some information about what is plotted, then show/save:
    #
    if cmd.savefig_title is None:  # construct and use default metadata title
        fig_title = "Cyclone track and MSLP snapshots: "
        if dt_start.month == dt_end.month:
            fig_title += dt_start.strftime("%d") + u"\u2013"
        else:
            fig_title += dt_start.strftime("%d %b") + u" \u2013 "
        fig_title += dt_end.strftime("%d %b %Y")

    else:  # using custom title
        fig_title= cmd.savefig_title

    # For interactive use: create short descriptive string for figure
    # window title (this isn't saved with the figure):
    descr_str = "Track/MSLP snapshots: "
    if dt_start.month == dt_end.month:
        descr_str += dt_start.strftime("%d") + u"\u2013"
        descr_str += dt_end.strftime("%d %b %Y")
    else:
        descr_str += dt_start.strftime("%d %b") + u" \u2013 "
        descr_str += dt_end.strftime("%d %b %Y")

    fig.canvas.manager.set_window_title(descr_str + " ")

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     set_raster_level=True, fig_metadata={"Title": fig_title})


if __name__ == "__main__":
    main()
