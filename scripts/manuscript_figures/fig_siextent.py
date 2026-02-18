"""Create figure showing sea ice extent annual time series of monthly means and
the climatological seasonal cycle in the model and SSM/I observations.

In the manuscript this is Fig. 1 for the pan Arctic (a) September time series
and (b) 2003-2023 climatology, although command-line options -r/--region <R>,
-m/--month <M>, --years-clim <Y1 <Y2> can change this (where the defaults are
R = 0, and M = 9, and Y1, Y2 = 2003, 2023).

"""

import calendar
from datetime import datetime as dt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src import script_tools
from src.io import config as cfg
from src.data import cice, ssmi
from src.plotting import style_ms, tools


cice_line_kw = {"color": "k", "label": "Simulation", "zorder": 10.}

cice_mark_kw = {"facecolor": "k", "edgecolor": "none", "s": 6., "zorder": 11.}

ssmi_fill_kw = {"facecolor": "tab:grey", "label": "SSM/I", "edgecolor": "none",
                "alpha": .3, "zorder": 6.}


def main():

    prsr = script_tools.argument_parser(
        usage="Plot sea ice extent time series and seasonal cycle")

    script_tools.add_plot_cmd_args(prsr, fig_names="fig1")

    prsr.add_argument("-r", "--region", type=int, default=0,
                      help="Region to plot (default = 0, Pan Arctic)")
    prsr.add_argument("-m", "--month", type=int, default=9,
                      help="Month to plot time series (1=Jan)")
    prsr.add_argument("--years-clim", type=int, nargs=2, default=(2003, 2023),
                      help="Year range for climatological seasonal cycle")

    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Load sea ice extent from CICE:
    date_cice, sie_cice = cice.get_processed_data_regional("sea_ice_extent",
        "sie", dt_min=dt(1979, 1, 1, 12, 0), dt_max=dt(2023, 12, 31, 12, 0),
        frequency="monthly", region_nc_names=[cfg.reg_nc_names[cmd.region]])

    date_nt, _, sie_nt = ssmi.load_data("sea_ice_extent", frequency="monthly",
        which_dataset="nt", dt_range=(dt(1979, 1, 1, 12), dt(2023, 12, 31, 12)))

    date_bt, _, sie_bt = ssmi.load_data("sea_ice_extent", frequency="monthly",
        which_dataset="bt", dt_range=(dt(1979, 1, 1, 12), dt(2023, 12, 31, 12)))

    # Get years for time axis (should be same for all):
    years    = np.arange(date_cice[0].year, date_cice[-1].year+1, 1)
    years_nt = np.arange(date_nt[0].year  , date_nt[-1].year+1  , 1)
    years_bt = np.arange(date_bt[0].year  , date_bt[-1].year+1  , 1)

    n_years    = len(years)
    n_years_nt = len(years_nt)
    n_years_bt = len(years_bt)

    # Reshape into (n_years, 12) so month time series and seasonal cycle can be
    # easily extracted. Also remove outer list (corresponds to region, already
    # selected for CICE, need to select index for SSM/I):
    sie_cice = np.reshape(sie_cice[0]       , (n_years   , 12))
    sie_nt   = np.reshape(sie_nt[cmd.region], (n_years_nt, 12))
    sie_bt   = np.reshape(sie_bt[cmd.region], (n_years_bt, 12))

    # Construct time axis for monthly climatology (in plot, axis labels are
    # replaced with first letter of month so this is arbitrary):
    t_months = np.arange(.5, 12., 1.)

    # Indices of year axis of sie_* where to average for climatology and month:
    jclim     = (years >= cmd.years_clim[0]) & (years <= cmd.years_clim[1])
    jclim_nt  = (years_nt >= cmd.years_clim[0]) & (years_nt <= cmd.years_clim[1])
    jclim_bt  = (years_bt >= cmd.years_clim[0]) & (years_bt <= cmd.years_clim[1])
    jmon      = cmd.month - 1

    print("Correlation between CICE and SSM/I time series ("
          + calendar.month_name[jmon+1] + ", "
          + cfg.reg_labels_long[cmd.region] + "):")

    for x, data in zip(["NASA Team", "Bootstrap"], [sie_nt, sie_bt]):
        print(f"  {x}: {np.corrcoef(sie_cice[:,jmon], data[:,jmon])[0,1]:.3f}")

    # Create figure:
    fig, axs = plt.subplots(figsize=(style_ms.fig_width_double_column,
                                     .5*style_ms.fig_height_double_column),
                            ncols=2, nrows=1)

    # Manually set subplot layouts:
    tools.distribute_subplots(axs, l=.06, r=.015, t=.145, b=.105, s_hor=.075)

    for ax in axs:
        ax.set_ylabel(r"$10^6$ km$^2$", labelpad=4)
        for spine in ax.spines:
            ax.spines[spine].set_zorder(1.E10)

    # Fix x-limits/ticks for time series panel:
    # (y-axis ticks set below after plotting data)
    tools.set_axis_ticks(axs[0], 1970., 2030., 10.,
        minor_tick_divisions=5, lims_actual=(1978., 2024.), which="x")

    # For climatology panel, fix x-limits/ticks and change labels to months:
    tools.set_axis_ticks(axs[1], .5, 11.5, 1., minor_tick_divisions=0,
                              lims_actual=(0., 12.), which="x")

    axs[1].set_xticklabels("JFMAMJJASOND")

    # Plot data:
    axs[0].plot(years, sie_cice[:,jmon], **cice_line_kw)
    axs[0].fill_between(years_nt, sie_nt[:,jmon], sie_bt[:,jmon], **ssmi_fill_kw)

    axs[1].plot(   t_months, np.mean(sie_cice[jclim,:], axis=0), **cice_line_kw)
    axs[1].scatter(t_months, np.mean(sie_cice[jclim,:], axis=0), **cice_mark_kw)

    axs[1].fill_between(t_months, np.nanmean(sie_nt[jclim_nt,:], axis=0),
                                  np.nanmean(sie_bt[jclim_bt,:], axis=0),
                        **ssmi_fill_kw)

    # Set y-axis limits/ticks and add legends (customised for Fig. 1, i.e.,
    # if plotting default pan Arctic, September):
    if cmd.region == 0 and cmd.month == 9:
        tools.set_axis_ticks(axs[0], 2., 8., 1., minor_tick_divisions=0,
                                  lims_actual=(2.8, 7.8), which="y")

        tools.set_axis_ticks(axs[1], 4., 12., 2., minor_tick_divisions=4,
                                  lims_actual=(4, 12), which="y")

        axs[0].annotate(cice_line_kw["label"], (1980., 5.5),
                        color=cice_line_kw["color"],
                        fontsize=mpl.rcParams["axes.labelsize"]-1,
                        fontweight="bold", ha="left", va="top")

        axs[0].annotate(ssmi_fill_kw["label"], (1989., 7.75),
                        color=ssmi_fill_kw["facecolor"],
                        fontsize=mpl.rcParams["axes.labelsize"]-1,
                        fontweight="bold", ha="left", va="top")
    else:
        # Automatic y-axis limits and standard legend for other combinations
        # (note: fix y-axis limits so they don't automatically adjust further
        # after adding the climatology lines below):
        axs[0].set_ylim(axs[0].get_ylim())
        axs[1].set_ylim(axs[1].get_ylim())
        axs[0].legend()

    # Add lines on time series panel (a) to indicate climatology period
    # (must come after fixing y-axis limits):
    for j in range(2):
        axs[0].plot([cmd.years_clim[j]]*2, axs[0].get_ylim(),
                    color="darkgray", linestyle=(0, (3, 3)), zorder=7)

    # Construct subplot titles and add them via plotting:
    title_0 = (cfg.reg_labels_long[cmd.region] + " "
               + calendar.month_name[cmd.month]  + " sea ice extent")

    title_1  = (cfg.reg_labels_long[cmd.region] + " "
                + f"{cmd.years_clim[0]}" + u"\u2013" + f"{cmd.years_clim[1]}"
                + " monthly climatology")

    tools.add_subplot_panel_titles(axs, titles=[title_0, title_1])

    # Default figure metadata title:
    if cmd.savefig_title is None:
        cmd.savefig_title = (f"{cfg.reg_labels_long[cmd.region]} sea ice "
                             + "extent in CICE and SSM/I observations")

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     fig_metadata={"Title": cmd.savefig_title})


if __name__ == "__main__":
    main()
