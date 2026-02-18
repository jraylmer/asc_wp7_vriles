"""Plot box plots of the distributions of CICE and SSM/I VRILE magnitudes
separated by classification for the former and whether associated with cyclone
presence or not (both).
"""

from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.plotting import style_ms, tools


def main():

    prsr = script_tools.argument_parser(usage="Plot box plots")
    prsr.add_argument("--ssmi-dataset", type=str, default="bt",
                      choices=["bt", "nt"])
    script_tools.add_plot_cmd_args(prsr, fig_names="fig7")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Identify case studies on box plots;
    # they are the following lists of lists [sector_id, rank]:
    case_studies_cice     = [[4, 23], [5, 5] , [4, 38]]

    if cmd.ssmi_dataset == "nt":
        case_studies_ssmi = [         [5, 33], [4, 49]]
    else:
        case_studies_ssmi = [[4, 35], [5, 38], [4, 21]]

    # Later append: [VRILE ID, VRILE magnitude, C vs NC] for ssmi
    # also D vs T for cice

    vrds     = "_joined"

    # For units, multiply all data by this y_factor, and set plot
    # y_label to match:
    y_factor = 1000.
    y_label  = r"VRILE magnitude ($10^3$ km$^2$ day$^{-1}$)"

    # Load cached VRILE and track data:
    vriles_cice = cache.load("vriles_cice_1979-2023.pkl")
    vriles_ssmi = cache.load(f"vriles_ssmi-{cmd.ssmi_dataset}_1979-2023.pkl")
    tracks      = cache.load("tracks_filtered_1979-2023.pkl")

    vriles_cice_track_ids = [x["track_indices"] for x in vriles_cice]
    vriles_ssmi_track_ids = [x["track_indices"] for x in vriles_ssmi]
    track_vrile_ids       = tracks[1+tracks[0].index("vriles_cice_indices")]

    nr = len(vriles_cice)  # number of regions (here includes pan Arctic)

    # Determine the VRILE ID and magnitude, cyclone associated or not, and
    # classification (CICE only) for each selected case study above. This
    # information is used to determine where to plot the case studies on the
    # final figures:
    #
    for cs in range(len(case_studies_cice)):
        r = case_studies_cice[cs][0]  # its region
        k = case_studies_cice[cs][1]  # its rank

        # Append the index of the VRILE in the results array:
        i = np.argmin(abs(vriles_cice[r][f"vriles{vrds}_rates_rank"]-k))
        case_studies_cice[cs].append(i)

        # Append the VRILE magnitude:
        case_studies_cice[cs].append(vriles_cice[r][f"vriles{vrds}_rates"][i])

        # Append whether it matches a track(s) or not:
        case_studies_cice[cs].append(len(vriles_cice_track_ids[r][i]) > 0)

        # Append the classification (-1 to +1):
        case_studies_cice[cs].append(vriles_cice[r][f"vriles{vrds}_class"][i])

    # Repeat for the SSM/I case studies:
    for cs in range(len(case_studies_ssmi)):
        r = case_studies_ssmi[cs][0]
        k = case_studies_ssmi[cs][1]
        i = np.argmin(abs(vriles_ssmi[r][f"vriles{vrds}_rates_rank"]-k))
        case_studies_ssmi[cs].append(i)
        case_studies_ssmi[cs].append(vriles_ssmi[r][f"vriles{vrds}_rates"][i])
        case_studies_ssmi[cs].append(len(vriles_ssmi_track_ids[r][i]) > 0)

    # Separate VRILEs into cyclone-associated (*_track_data) or not
    # (*_notrack_data). For CICE VRILEs, further separate each into dynamically
    # or thermodynamically dominated. All are done by region. First first region
    # of the VRILE data as loaded corresponds to pan Arctic, which we skip, but
    # replace with the sum over all sectors at the end so there are the same
    # number of 'regions' in total.
    #
    therm_track_data = [[] for j in range(nr)]
    dynam_track_data = [[] for j in range(nr)]
    ssmi_track_data  = [[] for j in range(nr)]

    therm_notrack_data = [[] for j in range(nr)]
    dynam_notrack_data = [[] for j in range(nr)]
    ssmi_notrack_data  = [[] for j in range(nr)]

    for r in range(1, nr):  # skip r = 0 (pan Arctic)
        for v in range(vriles_cice[r][f"n{vrds}_vriles"]):

            y_rv = y_factor*abs(vriles_cice[r][f"vriles{vrds}_rates"][v])

            dtv = vriles_cice[r][f"date_bnds_vriles{vrds}"][v]  # alias

            # This check on the datetime bounds is only needed if the bounds
            # are changed to check the sensitivity to a different month range
            # (discussed in the manuscript):
            if dtv[0] > dt(dtv[0].year, 4, 30) and dtv[1] < dt(dtv[1].year, 10, 1):

                # Add this VRILE rate of change of SIE (y_rv) to the
                # appropriate list in region r AND r=0 (all sectors):
                if len(vriles_cice_track_ids[r][v]) > 0:
                    if vriles_cice[r][f"vriles{vrds}_class"][v] > 0:
                        therm_track_data[0].append(y_rv)    # Cyclone + Thermo.
                        therm_track_data[r].append(y_rv)    # -------------------
                    else:
                        dynam_track_data[0].append(y_rv)    # Cyclone + Dynam.
                        dynam_track_data[r].append(y_rv)    # -------------------
                else:
                    if vriles_cice[r][f"vriles{vrds}_class"][v] > 0:
                        therm_notrack_data[0].append(y_rv)  # No Cyclone + Thermo.
                        therm_notrack_data[r].append(y_rv)  # --------------------
                    else:
                        dynam_notrack_data[0].append(y_rv)  # No Cyclone + Dynam.
                        dynam_notrack_data[r].append(y_rv)  # --------------------

        # Repeat for SSM/I, except there is no classification sub-setting:
        for v in range(vriles_ssmi[r][f"n{vrds}_vriles"]):

            y_rv = y_factor * abs(vriles_ssmi[r][f"vriles{vrds}_rates"][v])

            dtv = vriles_ssmi[r][f"date_bnds_vriles{vrds}"][v]

            if (dtv[0] > dt(dtv[0].year, 4, 30)) and (dtv[1] < dt(dtv[1].year, 10, 1)):
                if len(vriles_ssmi_track_ids[r][v]) > 0:
                    ssmi_track_data[0].append(y_rv)
                    ssmi_track_data[r].append(y_rv)
                else:
                    ssmi_notrack_data[0].append(y_rv)
                    ssmi_notrack_data[r].append(y_rv)


    # Some paramters for the plot style and layout:
    color_therm = "tab:red"   # color for CICE thermo. box plot
    color_dynam = "tab:blue"  # color for CICE dynam. box plot
    color_cice  = "k"         # color for CICE text (labels under box plots)
    color_ssmi  = "tab:grey"  # color for SSM/I box plot and labels

    # The box plots are drawn at arbitrary x coordinates. We set the range
    # 0-1 for the first set of box plots (cyclone associated) and 1-2 for the
    # second set (no cyclone associated):
    xlims = [0, 2]

    # The following parameters control the spacing etc. between the different
    # box plots in the same arbitrary x units:
    _margin   = 0.1
    _gap_cice = 0.075
    _gap_ssmi = 0.1
    _xw_ratio = 1.5

    # Calculate the widths of the CICE and SSM/I box plots:
    bp_xw_cice = (1. - 2.*_margin - _gap_ssmi - _gap_cice) / (2. + _xw_ratio)
    bp_xw_ssmi = _xw_ratio * bp_xw_cice

    # Calculate the position of the CICE left (dynamic) and right
    # (thermodynamic) box plots, and the position of the SSM/I box plot:
    bp_x0_cice = _margin + .5 * bp_xw_cice
    bp_x1_cice = bp_x0_cice + bp_xw_cice + _gap_cice
    bp_x0_ssmi = bp_x1_cice + .5 * (bp_xw_cice + bp_xw_ssmi) + _gap_ssmi

    # General keyword arguments passed to ax.boxplot():
    bp_kw = {"showfliers": False}

    # Separate linewidth (lw) and linestyles (ls) for the cyclone (_c) and
    # no cyclone (_nc) box plots, both CICE and SSM/I VRILEs:
    lw_c  = 2. * mpl.rcParams["lines.linewidth"]
    lw_nc = 1. * mpl.rcParams["lines.linewidth"]

    ls_c  = "-"
    ls_nc = "-"  # (well okay I set the same, but in case we change later)

    # Construct keyword arguments passed to ax.boxplot() in each category:
    bp_thm_c_kw  = {"boxprops": {"color": color_therm, "linewidth": lw_c,
                                 "linestyle": ls_c},
                    "widths": bp_xw_cice, **bp_kw}

    bp_dyn_c_kw  = {"boxprops": {"color": color_dynam, "linewidth": lw_c,
                                 "linestyle": ls_c},
                    "widths": bp_xw_cice, **bp_kw}

    bp_obs_c_kw  = {"boxprops": {"color": color_ssmi, "linewidth": lw_c,
                                 "linestyle": ls_c},
                    "widths": bp_xw_ssmi, **bp_kw}

    bp_thm_nc_kw = {"boxprops": {"color": color_therm, "linewidth": lw_nc,
                                 "linestyle": ls_nc},
                    "widths": bp_xw_cice, **bp_kw}

    bp_dyn_nc_kw = {"boxprops": {"color": color_dynam, "linewidth": lw_nc,
                                 "linestyle": ls_nc},
                    "widths": bp_xw_cice, **bp_kw}

    bp_obs_nc_kw = {"boxprops": {"color": color_ssmi, "linewidth": lw_nc,
                                 "linestyle": ls_nc},
                    "widths": bp_xw_ssmi, **bp_kw}

    # Add in some common keyword arguments:
    for dct in [bp_thm_c_kw , bp_dyn_c_kw , bp_obs_c_kw,
                bp_thm_nc_kw, bp_dyn_nc_kw, bp_obs_nc_kw]:
        for key in ["medianprops", "whiskerprops", "capprops"]:
            dct[key] = dct["boxprops"]

    # Keyword arguments for ax.scatter() for the case studies:
    scatter_kw = {"marker": "o", "s": 6, "zorder": 100}

    # Below each box plot, we write the number of data points it contains.
    # For CICE, we write the sum of the thermodynamic and dynamic categories
    # as well. So we have text labels at two vertical positions in each case,
    # given by the following as fraction of y-axes heights:
    ytxt0 = .100
    ytxt1 = .015

    # Keyword arguments for the text labels:
    txt_kw = {"fontsize": mpl.rcParams["axes.labelsize"] - 2,
              "ha": "center", "va": "center", "clip_on": False}

    # Create the figure layout from the common template, without the legends:
    fig, axs = style_ms.fig_layout_2x4(ax_r=.01, ax_b=.06, ax_s_hor=.04,
                                       ax_s_ver=.14, ylabel=y_label,
                                       vrile_class_cbar=False,
                                       lines_legend=False)

    # Add subplot panel titles (we use the long labels for all except
    # East-Siberian--Chukchi Sea, which is too long so replace with short):
    axs_titles = ["All sectors"] + cfg.reg_labels_long[1:]
    axs_titles[6] = cfg.reg_labels_short[6]
    tools.add_subplot_panel_titles(axs, axs_titles)

    # Hard-code the y-axis limits for simplicity. These must be set first as
    # the text labels are plotted on the axes and use the y-limits to deterine
    # their y-coordinates:
    axs[0,0].set_ylim(0, 20)
    axs[0,1].set_ylim(0, 15)
    axs[0,2].set_ylim(0, 15)
    axs[0,3].set_ylim(0, 15)
    axs[1,0].set_ylim(0, 15)
    axs[1,1].set_ylim(0, 20)
    axs[1,2].set_ylim(0, 25)
    axs[1,3].set_ylim(0, 15)

    for r in range(nr):

        row = r // 4
        col = r % 4

        axs[row,col].boxplot([dynam_track_data[r]], positions=[bp_x0_cice], **bp_dyn_c_kw)
        axs[row,col].boxplot([therm_track_data[r]], positions=[bp_x1_cice], **bp_thm_c_kw)
        axs[row,col].boxplot([ssmi_track_data[r]] , positions=[bp_x0_ssmi], **bp_obs_c_kw)

        axs[row,col].boxplot([dynam_notrack_data[r]], positions=[1. + bp_x0_cice], **bp_dyn_nc_kw)
        axs[row,col].boxplot([therm_notrack_data[r]], positions=[1. + bp_x1_cice], **bp_thm_nc_kw)
        axs[row,col].boxplot([ssmi_notrack_data[r]] , positions=[1. + bp_x0_ssmi], **bp_obs_nc_kw)

        # y-coordinates for the text labels (fraction specified earlier * y-axis height,
        # noting that all y lower limits are set to zero):
        ytxt0_j = ytxt0 * axs[row,col].get_ylim()[1]
        ytxt1_j = ytxt1 * axs[row,col].get_ylim()[1]

        # Add the text labels for the number of data points in each box plot:
        axs[row,col].annotate(f"{len(dynam_track_data[r])}", (bp_x0_cice, ytxt0_j),
                              color=color_dynam, fontweight="bold", **txt_kw)
        axs[row,col].annotate(f"{len(therm_track_data[r])}", (bp_x1_cice, ytxt0_j),
                              color=color_therm, fontweight="bold", **txt_kw)
        axs[row,col].annotate(f"{len(ssmi_track_data[r])}" , (bp_x0_ssmi, ytxt1_j),
                              color=color_ssmi, fontweight="bold", **txt_kw)

        # For CICE, add the sum of thermo. + dynamic underneath:
        axs[row,col].annotate(f"{len(dynam_track_data[r]) + len(therm_track_data[r])}",
                              (.5*(bp_x0_cice + bp_x1_cice), ytxt1_j),
                              color=color_cice, fontweight="bold", **txt_kw)

        # Repeat for the no-cyclone associated half of the plot:
        axs[row,col].annotate(f"{len(dynam_notrack_data[r])}", (1. + bp_x0_cice, ytxt0_j),
                              color=color_dynam, **txt_kw)
        axs[row,col].annotate(f"{len(therm_notrack_data[r])}", (1. + bp_x1_cice, ytxt0_j),
                              color=color_therm, **txt_kw)
        axs[row,col].annotate(f"{len(ssmi_notrack_data[r])}", (1. + bp_x0_ssmi, ytxt1_j),
                              color=color_ssmi, **txt_kw)
        axs[row,col].annotate(f"{len(dynam_notrack_data[r]) + len(therm_notrack_data[r])}",
                              (1. + .5*(bp_x0_cice + bp_x1_cice), ytxt1_j),
                              color=color_cice, **txt_kw)

    # Add the case studies as markers on the appropriate panel:
    for cs in range(len(case_studies_cice)):
        row = case_studies_cice[cs][0] // 4
        col = case_studies_cice[cs][0] % 4

        if case_studies_cice[cs][5] > 0:  # thermodynamic
            if case_studies_cice[cs][4]:  # has cyclone(s)
                x = bp_x1_cice  # horizontal position
                mew = lw_c      # marker edge width
            else:  # no cyclone(s)
                x = 1. + bp_x1_cice
                mew = lw_nc
            mec = color_therm  # marker edge color
        else:  # dynamic
            if case_studies_cice[cs][4]:  # has cyclone(s)
                x = bp_x0_cice
                mew = lw_c
            else:  # no cyclone(s)
                x = 1. + bp_x0_cice
                mew = lw_nc
            mec = color_dynam

        y = y_factor*abs(case_studies_cice[cs][3])  # vertical position

        axs[row,col].scatter(x, y, facecolor="none", edgecolor=mec,
                             linewidth=mew, **scatter_kw)

    # Repeat for SSM/I VRILEs:
    for cs in range(len(case_studies_ssmi)):
        row = case_studies_ssmi[cs][0] // 4
        col = case_studies_ssmi[cs][0] % 4

        if case_studies_ssmi[cs][4]:  # has cyclone(s)
            x = bp_x0_ssmi
            mew = lw_c
        else:
            x = 1. + bp_x0_ssmi
            mew = lw_nc

        y = y_factor*abs(case_studies_ssmi[cs][3])

        axs[row,col].scatter(x, y, facecolor="none", edgecolor=color_ssmi,
                             linewidth=mew, **scatter_kw)

    # Set the horizontal axes limits, remove ticks and add custom labels:
    for ax in axs.flatten():
        ax.set_xlim(xlims)
        ax.set_xticks([.5, 1.5])
        ax.tick_params(which="both", axis="x", bottom=False)
        ax.set_xticklabels(["Cyclone", "No cyclone"],
                           fontsize=mpl.rcParams["axes.labelsize"]-1)
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.spines["bottom"].set_visible(False)

    # Set figure metadata Title:
    if cmd.savefig_title is None:
        savefig_title = "Box plots of VRILE and Arctic cyclone matches in CICE"
    else:
        savefig_title = cmd.savefig_title

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     fig_metadata={"Title": savefig_title})


if __name__ == "__main__":
    main()
