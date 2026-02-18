"""Create figure showing the change in sea ice due to dynamics broken down into
advection and divergence terms. The script is based on _case_study.py, and is
used for supplemental figure S1.
"""

from datetime import datetime as dt, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.data import cice
from src.plotting import maps, style_ms, symbols, tools

import posterproxy as psp


def parse_cmd_args():
    """Define and parse command line arguments using the argparse module.
    Returns the parsed arguments.
    """

    prsr = script_tools.argument_parser(
        usage="Advection, divergence, and dynamics changes")

    # These options are used to determine which VRILE to load:
    prsr.add_argument("-r", "--region", type=int, default=5,
                      choices=[r for r in range(8)], help="Region index")
    prsr.add_argument("-k", "--rank", type=int, default=5,
                      help="Rank of VRILE to select for specified -r/--region")

    # Alternative: specify start and end datetimes and load data for that
    # time period (this allows 'cyclone, but no VRILE' case studies):
    prsr.add_argument("--dt-start", type=int, nargs=3, default=(1950, 1, 1),
                      help="Manual datetime start Y M D (-r and -k ignored if "
                           + "this is later than default 1950-01-01)")

    prsr.add_argument("--dt-end", type=int, nargs=3, default=(1950, 1, 1),
                      help="Manual datetime end Y M D (-r and -k ignored if "
                           + "this is later than default 1950-01-01)")

    # Options for the map projection (posterproxy package):
    prsr.add_argument("--extent-latitude", type=float, default=70.,
                      help="Outer latitude of plot")
    prsr.add_argument("--central_longitude", type=float, default=0.,
                      help="Central longitude of plot")
    prsr.add_argument("--xy-offset", type=float, nargs=2, default=(0., 0.),
                      help="Translation of viewport in (x,y) space for plot")

    # Generic plotting command-line arguments (contains --savefig, etc.):
    script_tools.add_plot_cmd_args(prsr, fig_names="figS1")

    return prsr.parse_args()


def main():

    cmd = parse_cmd_args()
    cfg.set_config(*cmd.config)

    # Identify what to load: time period for raw data
    # This depends on whether the options --dt-{start,end} are set:
    #
    if dt(*cmd.dt_start) > dt(1950, 1, 1) or dt(*cmd.dt_end) > dt(1950, 1, 1):

        # No specific VRILE to load => set region, rank, VRILE indices to NaN:
        cmd.region = np.nan
        cmd.rank   = np.nan
        vrile_id   = np.nan

        # Set the start/end dates; add 1 day to dt_end because cmd argument 
        # specifies a day, but by default dt hour == minute == 0, so this
        # ensures we include all of cmd.dt_end:
        dt_start = dt(*cmd.manual_dt_start)
        dt_end   = dt(*cmd.manual_dt_end) + timedelta(days=1)

    else:

        # Load the VRILE data for specified region only:
        vriles_cice = cache.load("vriles_cice_1979-2023.pkl")[cmd.region]

        # Identify the VRILE index based on the specified rank:
        vrile_id = np.argmin(abs(vriles_cice["vriles_joined_rates_rank"] - cmd.rank))

        # Total number of VRILEs for this sector (used in figure name):
        n_vriles_r = vriles_cice["n_joined_vriles"]

        # Set the start/end dates to be that of the specified VRILE:
        dt_start = vriles_cice["date_bnds_vriles_joined"][vrile_id][0]
        dt_end = vriles_cice["date_bnds_vriles_joined"][vrile_id][1]

    # Print info:
    print(f"VRILE ID                  : ", end="")
    if np.isnan(vrile_id):
        print("-")
    else:
        print(cfg.reg_labels_long[cmd.region] + f", rank = {cmd.rank} of "
              + f"{n_vriles_r} (ID = {vrile_id})")
    print(f"Date start                : {dt_start.strftime('%d %b %Y')}")
    print(f"Date end                  : {dt_end.strftime('%d %b %Y')}")

    data_2d = dict()

    print("\nLoading CICE coordinate data")
    ulon, ulat, tlon, tlat = cice.get_grid_data(["ULON", "ULAT", "TLON", "TLAT"],
                                                slice_to_atm_grid=False)

    print("Loading CICE history data")
    data_cice_hist = cice.get_history_data(["hi_d", "dvidtd_d", "aice_d"],
        dt_min=dt_start, dt_max=dt_end, frequency="daily",
        slice_to_atm_grid=False)[1]

    print("Loading CICE processed data")
    data_cice_proc = cice.get_processed_data("div_curl", ["div_u_d"],
        dt_min=dt_start, dt_max=dt_end, frequency="daily",
        slice_to_atm_grid=False)[1]

    print("Calculating advection and divergence terms")
    data_2d["dvidtd"]     = np.nansum(data_cice_hist[1], axis=0)
    data_2d["dvidtd_div"] = np.nansum(100.*86400.*data_cice_hist[0]*data_cice_proc[0], axis=0)
    data_2d["dvidtd_adv"] = data_2d["dvidtd"] - data_2d["dvidtd_div"]


    # Set plotting parameters -- mostly hardcoded for simplicity:
    #
    # Unlike the general _case_study.py script, here we just use the same for
    # all three panels as they show a decomposition, so do not need to define
    # anything per variable.
    #
    levels               = np.arange(-36., 36.001, 8.)
    cbar_tick_labels     = [-36., -20., 20., 36.]
    cbar_tick_labels_fmt = "{:.0f}"
    cbar_title           = "$" + symbols.delta_vice + "$ (cm)"

    # Keyword arguments to plt.pcolormesh()
    pcm_kw = maps.pcm_kw(levels, style_ms.cmap["dvidtd"])

    # Transform lon/lat to north polar stereographic (x,y)
    # Arguments passed to posterproxy lonlat_to_xy_npsp() function:
    psp_xy_kw = {"extent_latitude"  : cmd.extent_latitude,    # default 70.
                 "central_longitude": cmd.central_longitude,  # default  0.
                 "xy_offset"        : cmd.xy_offset}          # default (0,0)

    xt, yt = psp.lonlat_to_xy_npsp(tlon, tlat, **psp_xy_kw)  # for ice edge
    xu, yu = psp.lonlat_to_xy_npsp(ulon, ulat, **psp_xy_kw)  # for dvidtd_*

    fig, axs = plt.subplots(ncols=3,
                            figsize=(style_ms.fig_width_double_column,
                                     .425*style_ms.fig_width_double_column))

    psp.prepare_axes(axs)  # prepare for map plotting in psp
    #                      # (must before distribute_subplots)

    tools.distribute_subplots(axs, l=.025, r=.025, b=.17, t=.03, s_hor=.025)

    # Create axes for a single horizontal colorbar:
    cbar_ax = fig.add_axes([0,0,1,1])
    tools.distribute_subplots(np.array([cbar_ax]), l=.25, r=.25, b=.08, t=.875)

    # Add titles/labels to subplots and colorbars:
    tools.add_subplot_panel_titles(axs,
        titles=["Total dynamic change", "Divergence", "Advection"],
        title_kw={"pad": 4, "zorder": 10000})

    tools.add_subplot_panel_titles(np.array([cbar_ax]), titles=[cbar_title],
                                   add_panel_labels=False,
                                   title_kw={"loc": "center", "pad": 4})

    # Note: zorders are set in the wrapper functions so the order of plotting
    # different elements should not generally matter (see maps.py):
    maps.psp_xy_land_overlay(axs, psp_xy_kw)
    maps.psp_xy_gridlines(axs, psp_xy_kw)

    # Main plots (pcolormesh):
    for ax, v in zip(axs, ["dvidtd", "dvidtd_div", "dvidtd_adv"]):
        ax.set_rasterization_zorder(maps.get_zorder("rasterization"))
        pcm = ax.pcolormesh(xu, yu, data_2d[v][1:,1:], **pcm_kw)

    maps.set_colorbars([pcm], np.array([cbar_ax]),
        [cbar_tick_labels], signed_cbar_tick_labels=[True],
        cbar_tick_labels_fmt=[cbar_tick_labels_fmt])

    # Add ice edge change to each panel:
    maps.add_ice_edge_contours(axs, xt, yt, data_cice_hist[-1])

    # Final part is to set figure metadata, window title for interactive use,
    # both containing some information about what is plotted, then show/save
    #
    if cmd.savefig_title is None:  # construct and use default metadata title
        if np.isnan(vrile_id):
            fig_title = "Sea ice change due to dynamics"
        else:
            fig_title = ("Dynamical sea ice change during case study: "
                         + cfg.reg_labels_long[cmd.region]
                         + f" sector, VRILE ID {vrile_id}, rank {cmd.rank} of "
                         + f"{n_vriles_r}, ")

        if dt_start.month == dt_end.month:
            fig_title += dt_start.strftime("%d") + u"\u2013"
        else:
            fig_title += dt_start.strftime("%d %b") + u" \u2013 "
        fig_title += dt_end.strftime("%d %b %Y")

    else:
        fig_title= cmd.savefig_title

    # For interactive use: create short descriptive string for figure
    # window title (this isn't saved with the figure):
    if np.isnan(vrile_id):
        descr_str = ""
    else:
        descr_str = f"{cfg.reg_labels_long[cmd.region]} VRILE "
        descr_str += f"{vrile_id}, rank {cmd.rank}/{n_vriles_r}, "

    if dt_start.month == dt_end.month:
        descr_str += dt_start.strftime("%d") + u"\u2013"
        descr_str += dt_end.strftime("%d %b %Y")

    else:
        descr_str += dt_start.strftime("%d %b") + u" \u2013 "
        descr_str += dt_end.strftime("%d %b %Y")

    descr_str += " "
    fig.canvas.manager.set_window_title(descr_str)

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     set_raster_level=True, fig_metadata={"Title": fig_title})


if __name__ == "__main__":
    main()
