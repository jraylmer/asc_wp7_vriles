"""Plot annual counts of VRILEs in the model and observations and cyclone
tracks as time series. In the manuscript this is Fig. 5. This script also
calculates correlations, trends, and standard deviations, saved to a text file.
"""

from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from tabulate import tabulate

from src import script_tools
from src.io import cache, config as cfg
from src.plotting import style_ms, tools


def main():

    prsr = script_tools.argument_parser(usage="Plot time series")
    prsr.add_argument("--ssmi-dataset", type=str, default="bt", choices=["bt", "nt"])
    script_tools.add_plot_cmd_args(prsr, fig_names="fig5")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    vriles_cice = cache.load("vriles_cice_1979-2023.pkl")
    vriles_ssmi = cache.load(f"vriles_ssmi-{cmd.ssmi_dataset}_1979-2023.pkl")
    track_data  = cache.load("tracks_filtered_1979-2023.pkl")

    # See scripts/tracks/filter_tracks.py:
    #
    # track_data[0] -> list of headers
    # track_data[1] -> 'TRACKS_PER_YEAR'
    # track_data[2] -> 'TRACK IDS'
    # track_data[3] -> 'DATETIMES'
    # track_data[8] -> 'SECTOR_FLAG'

    years = np.arange(1979, 2024, 1).astype(np.int32)

    n_years   = len(years)
    n_regions = len(vriles_cice)

    # Initialise counts per region per year for VRILEs and tracks
    #
    # We will also want the totals across all sectors. For simplicity we append
    # the totals to the end of the following arrays, so set the first axis to
    # have length 1 more than the number of regions. In practice this means:
    #
    # n_vriles_cice = [Pan Arctic, Sector 1, ..., Sector N, Sum of sectors 1-N]
    #
    # etc. For VRILEs the last element is simply the sum but for tracks it must
    # be calculated separately as a single track can be associated with
    # multiple sectors -- that is already calculated and is given by
    # track_data[1] which we assign here straight away:
    #
    n_vriles_cice  = np.zeros((n_regions+1, n_years))
    n_vriles_ssmi  = np.zeros((n_regions+1, n_years))
    n_tracks       = np.zeros((n_regions+1, n_years))
    n_tracks[-1,:] = track_data[1]

    # For simulated VRILEs, also get the mean classification per year/region:
    vrile_class = np.zeros((n_regions+1, n_years))

    # Compute number of tracks per region per year. This info is already in
    # the filtered track data ('SECTOR_FLAG'):
    for trk_id in range(len(track_data[2])):

        # Find index of years array which this track belongs to:
        jy = np.argmin(abs(track_data[3][trk_id][0].year - years))

        # Add to n_tracks_regional array in the right year and region:
        for r in range(n_regions):
            if any(track_data[8][trk_id][r,:]):
                n_tracks[r,jy] += 1

    # Store correlations (r), standard deviations (s), and trends (m) between/
    # for each dataset, by region, during the loop below:
    r_vrile_cice_ssmi = np.zeros(n_regions+1)
    r_vrile_cice_trk  = np.zeros(n_regions+1)
    r_vrile_ssmi_trk  = np.zeros(n_regions+1)

    s_vrile_cice = np.zeros(n_regions+1)
    s_vrile_ssmi = np.zeros(n_regions+1)
    s_trk        = np.zeros(n_regions+1)

    m_vrile_cice = np.zeros(n_regions+1)
    m_vrile_ssmi = np.zeros(n_regions+1)
    m_trk        = np.zeros(n_regions+1)

    # Standard errors of the slopes:
    me_vrile_cice = np.zeros(n_regions+1)
    me_vrile_ssmi = np.zeros(n_regions+1)
    me_trk        = np.zeros(n_regions+1)

    for r in range(n_regions):

        # Year of each VRILE in region r:
        vrile_r_y = [x.year for x in vriles_cice[r]["date_vriles"]]

        # For each VRILE of this region, add 1 to the count, and add its
        # classification metric to vrile_class array declared above (at the end
        # we normalise by number of VRILEs in the region to get the mean):
        for j in range(len(vrile_r_y)):
            n_vriles_cice[r,vrile_r_y[j]-1979] += 1
            # Get the classification; it is missing, set to zero
            # (means no particular classification):
            vrile_r_j_class = vriles_cice[r]["vriles_class"][j]
            if np.isnan(vrile_r_j_class):
                vrile_r_j_class = 0.
            vrile_class[r,vrile_r_y[j]-1979] += vrile_r_j_class

        # Now normalise by number of VRILEs in this region to get the
        # mean classification (per year):
        vrile_class[r,:] /= n_vriles_cice[r,:]

        # Repeat the above for SSM/I VRILEs, except there is no classification:
        vrile_r_y = [x.year for x in vriles_ssmi[r]["date_vriles"]]

        for j in range(len(vrile_r_y)):
            n_vriles_ssmi[r, vrile_r_y[j]-1979] += 1

    # Now get totals across all regions (excluding pan Arctic at index=0):
    n_vriles_cice[-1,:] = np.sum(n_vriles_cice[1:-1,:], axis=0)
    n_vriles_ssmi[-1,:] = np.sum(n_vriles_ssmi[1:-1,:], axis=0)
    vrile_class  [-1,:] = np.mean( vrile_class[1:-1,:], axis=0)

    # Compute the correlations etc. for each region (now including total):
    for r in range(n_regions+1):
        r_vrile_cice_ssmi[r] = np.corrcoef(n_vriles_cice[r,:], n_vriles_ssmi[r,:])[0,1]
        r_vrile_cice_trk[r]  = np.corrcoef(n_vriles_cice[r,:], n_tracks[r,:])[0,1]
        r_vrile_ssmi_trk[r]  = np.corrcoef(n_vriles_ssmi[r,:], n_tracks[r,:])[0,1]

        s_vrile_cice[r] = np.std(n_vriles_cice[r,:])
        s_vrile_ssmi[r] = np.std(n_vriles_ssmi[r,:])
        s_trk[r]        = np.std(n_tracks[r,:])

        for data, m_out, me_out in zip(
                [n_vriles_cice, n_vriles_ssmi, n_tracks],
                [m_vrile_cice , m_vrile_ssmi , m_trk   ],
                [me_vrile_cice, me_vrile_ssmi, me_trk  ]):
            lreg = scipy.stats.linregress(years, data[r,:])
            m_out[r]  = lreg[0]
            me_out[r] = lreg[4]

    # Prepare a table of correlations for the output:
    headers = ["Region", "r(CICE VRILEs vs. SSMI VRILEs)", "r(CICE VRILEs, tracks)",
                         "r(SSMI VRILEs, tracks)"]

    region_labels = cfg.reg_labels_short + ["ALL"]

    rows = [[region_labels[r], r_vrile_cice_ssmi[r], r_vrile_cice_trk[r],
             r_vrile_ssmi_trk[r]] for r in range(n_regions+1)]

    # Start text output for file (and print) with a header and then add table:
    txt_out = dt.today().strftime("%H:%M %a %d %b %Y") + "\n\n"
    txt_out += tabulate(rows, headers=headers, floatfmt=".3f")

    # Add table for standard deviations:
    headers = ["Region", "Std. Dev. (VRILEs CICE)", "Std. Dev. (VRILEs Obs.)",
                         "Std. Dev. (tracks)"]
    rows = [[region_labels[r], s_vrile_cice[r], s_vrile_ssmi[r], s_trk[r]]
            for r in range(n_regions+1)]

    txt_out += "\n\n" + tabulate(rows, headers=headers, floatfmt=".1f")

    # Add table for trends and their standard errors:
    headers = ["Region"   , "CICE VRILE trend", "std. err.", "Obs. VRILE trend",
               "std. err.", "Track trend"     , "std. err."]

    rows = [[region_labels[r], m_vrile_cice[r], me_vrile_cice[r], m_vrile_ssmi[r],
             me_vrile_ssmi[r], m_trk[r], me_trk[r]] for r in range(n_regions+1)]

    txt_out += "\n\n" + tabulate(rows, headers=headers, floatfmt=".3f")

    # Add note of which SSM/I dataset is used:
    txt_out += "\n\nNote: SSM/I dataset is "

    if cmd.ssmi_dataset == "bt":
        txt_out += "Bootstrap\n"
    else:
        txt_out += "NASA Team\n"

    print(txt_out)
    cache.write_txt(txt_out,
        f"n_vriles_ssmi-{cmd.ssmi_dataset}_tracks_correlations_trends.txt",
        directory=cfg.data_path["tables"])

    # ======================================================================= #

    # Keyword arguments for line styles (and scatter for CICE classifications):
    track_line_kw = {"zorder": 18, "color": "limegreen", "linestyle": "-"}
    ssmi_line_kw  = {"zorder": 19, "color": "tab:gray", "linestyle": "--"}
    cice_line_kw  = {"zorder": 20, "color": "k"}

    cice_scatter_kw = {"zorder": 21, "edgecolors": "k", "s": 12,
                       "linewidths": mpl.rcParams["lines.linewidth"],
                       "clip_on": False}

    fig, axs = style_ms.fig_layout_2x4(invert=True, fig_height_frac=.95,
                                       ax_l=.06, ax_r=.01, ax_t=.04, ax_b=.13,
                                       ax_s_ver=.075, ax_s_hor=.05,
                                       ylabel="Annual counts",
                                       cbar_y0=.04, cbar_height=.018,
                                       lines_legend=True, lines_legend_dy=.005,
                                       lines_ssmi_kw=ssmi_line_kw,
                                       lines_tracks_kw=track_line_kw)

    # Want panel (a) to be sum of all sectors (last element of data arrays)
    # Remaining panels (b-h) are regions (pan Arctic, r=0, is excluded):
    for r in [-1] + [j for j in range(1, n_regions)]:
        row = r // 2 if r > 0 else 0
        col = r % 2 if r > 0 else 0

        axs[row,col].plot(years, n_tracks[r,:]     , **track_line_kw)
        axs[row,col].plot(years, n_vriles_ssmi[r,:], **ssmi_line_kw)
        axs[row,col].plot(years, n_vriles_cice[r,:], **cice_line_kw)

        axs[row,col].scatter(years, n_vriles_cice[r,:], c=vrile_class[r,:],
                             cmap=style_ms.vrile_class_cmap,
                             norm=style_ms.vrile_class_norm, **cice_scatter_kw)

    axs[0,0].set_ylim(0, 100)
    axs[0,1].set_ylim(0, 25)
    axs[1,0].set_ylim(0, 25)
    axs[1,1].set_ylim(0, 25)
    axs[2,0].set_ylim(0, 25)
    axs[2,1].set_ylim(0, 25)
    axs[3,0].set_ylim(0, 35)
    axs[3,1].set_ylim(0, 25)

    for ax in axs.flatten():
        ax.grid(axis="y")
        tools.set_axis_ticks(ax, 1970, 2030, 10,
                             lims_actual=(1978, 2024), which="x")

    tools.add_subplot_panel_titles(axs,
        ["All sectors"] + [cfg.reg_labels_long[r] for r in range(1,n_regions)])

    # Set title for figure metadata:
    if cmd.savefig_title is None:
        savefig_title = "Annual counts of VRILEs and Arctic cyclones per region"
    else:
        savefig_title = cmd.savefig_title

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     fig_metadata={"Title": savefig_title})


if __name__ == "__main__":
    main()
