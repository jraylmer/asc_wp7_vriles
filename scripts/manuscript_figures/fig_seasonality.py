"""Plot histograms of the seasonal distribution of VRILEs in the model and
observations, and JRA-55 cyclone tracks. The model distributions are color-
coded by the mean VRILE classification.
"""

import calendar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.data import tracks
from src.plotting import style_ms, tools


def bin_vriles(vriles_cice, vriles_ssmi, bin_edges, joined=True, density=True):
    """Bin VRILEs into time of year bins for all sectors combined (NOT the
    pan-Arctic -- the first vrile results index 0 is not used) and sectors
    individually, for CICE and SSM/I.


    Parameters
    ----------
    vriles_cice, vriles_ssmi : list of dict
        The saved data for CICE and SSM/I VRILEs 'results dictionaries'
        containing the various data per region. The first index of each is
        assumed to be the pan Arctic and ignored.

    bin_edges : array of int
        The bin edges corresponding to a horizontal 'day of year' axis.
        The number of bins is nb = len(bin_edges) - 1.


    Optional parameters
    -------------------
    joined : bool, default = True
        If True, bin the 'joined' VRILEs.

    density : bool, default = True
        If True, determine frequency density rather than frequency in each bin.


    Returns
    -------
    b_cice : array of shape (nr, nb) of float
        The frequency (density) of CICE VRILEs by region as a function of bin,
        where index 0 corresponds to the sum of all sectors and 1--nr
        correspond to the remaining regions/sectors as in the input data at
        the same indices.

    b_cice_class : array of shape (nr, nb) of float
        The mean classification of VRILEs in each element of binned_cice.

    b_ssmi : array of shape (nr, nb) of float
        As in binned_cice but for SSM/I VRILEs.
    """

    nb = len(bin_edges) - 1
    nr = len(vriles_cice)
    bw = bin_edges[1:] - bin_edges[:-1]  # bin widths

    b_cice = np.zeros((nr, nb)).astype(float)
    b_ssmi = np.zeros((nr, nb)).astype(float)

    # For CICE, average VRILE classification per bin:
    b_cice_class = np.zeros((nr, nb)).astype(float)

    # VRILE results dictionary string for joined or not:
    vrds = f"vriles{'_joined' if joined else ''}"

    # Again assume index r = 0 is pan Arctic and skip, but we use that index
    # of the outputs (b_cice, b_cice_class, b_ssmi) for the sum of all sectors:
    for r in range(1, nr):

        # Convert VRILE centre dates to day of year, accounting for leap year
        # (excluded from CICE simulation and SSM/I already):
        doy_cice_r = np.array([v.timetuple().tm_yday - 1*(calendar.isleap(v.year))
                               for v in vriles_cice[r][f"date_{vrds}"]])

        for v in range(len(doy_cice_r)):
            
            # Find which bin this VRILE goes into:
            in_bin_v = np.array([bin_edges[b] <= doy_cice_r[v] < bin_edges[b+1]
                                for b in range(nb)])

            # in_bin_j is 0 for each bin except one which is 1, so add to the
            # final counts array:
            b_cice[r,:] += in_bin_v

            # Also add its class to the average classified array at the
            # corresponding bin (to get the average class we divide by the bin
            # count after the loop):
            #
            # Need to check for unclassifiable state (note this NaN case only
            # happens 1 in ~500 times):
            if not np.isnan(vriles_cice[r][f"{vrds}_class"][v]):
                b_cice_class[r,:] += in_bin_v*vriles_cice[r][f"{vrds}_class"][v]

        # Repeat for SSM/I, except the classification:
        doy_obs_r = np.array([v.timetuple().tm_yday - 1*(calendar.isleap(v.year))
                              for v in vriles_ssmi[r][f"date_{vrds}"]])

        for v in range(len(doy_obs_r)):
            b_ssmi[r,:] += np.array([bin_edges[b] <= doy_obs_r[v] < bin_edges[b+1]
                                     for b in range(nb)])

    # Divide by the bin counts to get the mean classification:
    b_cice_class /= b_cice

    # All-sector mean classification per bin (divide by average of the
    # individual-sector averages for each bin):
    b_cice_class[0,:] = np.nanmean(b_cice_class[1:,:], axis=0)

    # All-sector bin counts (add across sectors for each bin):
    b_cice[0,:] = np.sum(b_cice[1:,:], axis=0)
    b_ssmi[0,:] = np.sum(b_ssmi[1:,:], axis=0)

    if density:
        # Normalise so the sum of products of bin widths and bin counts is 1:
        for r in range(nr):
            b_cice[r,:] /= np.sum(b_cice[r,:]*bw)
            b_ssmi[r,:] /= np.sum(b_ssmi[r,:]*bw)

    return b_cice, b_cice_class, b_ssmi


def bin_track_data(t_dts, t_sec, bin_edges, n_sector_nei=1, density=True):
    """Bin cyclone tracks into time of year bins for all sectors combined
    (NOT the pan-Arctic) and sectors individually.


    Parameters
    ----------
    The following two arrays are as saved in the track filtering script
    get_filtered_tracks.py.

    t_dts : length nk list of 1D arrays of datetime.datetime
        The datetime coordinates of the nk cyclone tracks.

    t_sec : length nk list of arrays of shape (nr,) of bool
        Flags indicating which region a track goes through.

    bin_edges : array of int
        The bin edges corresponding to a horizontal 'day of year' axis.
        The number of bins is nb = len(bin_edges) - 1.


    Optional parameters
    -------------------
    n_sector_nei : int, default = 1
        The number of nearest neighbouring sectors of a given sector r such
        that a track which passes through any of the sectors r +/- n_sector_nei
        is considered to be associated with sector r and thus binned
        accordingly.

    density : bool, default = True
        If True, determine frequency density rather than frequency in each bin.


    Returns
    -------
    b_track : array of shape (nr, nb) of float
        The frequency (density) of cyclone tracks by region as a function of
        bin, where index 0 corresponds to the sum of all sectors and 1--nr
        correspond to the individual regions/sectors.

    """

    nk = len(t_dts)
    nr = np.shape(t_sec[0])[0]
    nb = len(bin_edges) - 1
    bw = bin_edges[1:] - bin_edges[:-1]  # bin widths

    b_track = np.zeros((nr, nb)).astype(float)

    allowed_sectors = tracks.allowed_sectors(n_nei=n_sector_nei)

    for k in range(nk):
        # Day of year coordinates for track k:
        doy_k = np.array([x.timetuple().tm_yday - 1*(calendar.isleap(x.year))
                          for x in t_dts[k]])

        for b in range(nb):
            # Boolean array for each coordinate of track k:
            # True if in time range for bin b:
            dt_check = (bin_edges[b] <= doy_k) & (doy_k < bin_edges[b+1])

            # For each sector (except pan Arctic), if any coordinate is in the
            # time range and any of the allowed sector bounds, then track k
            # goes into bin b, so increase the corresponding counter by 1
            # (note that a single track can thus appear in multiple bins):
            for r in range(1, nr):
                sec_check = np.any(t_sec[k][allowed_sectors[r],:], axis=0)
                b_track[r,b] += any(dt_check & sec_check)

    # All sectors combined:
    b_track[0,:] = np.sum(b_track[1:,:], axis=0)

    if density:
        # Normalise so the sum of products of bin widths and bin counts is 1:
        for r in range(nr):
            b_track[r,:] /= np.sum(b_track[r,:]*bw)

    return b_track


def main():
    
    prsr = script_tools.argument_parser(usage="Histograms")
    prsr.add_argument("--ssmi-dataset", type=str, default="bt",
                      choices=["bt", "nt"])
    prsr.add_argument("--bins", type=str, default="10d", choices=["10d", "m"],
                      help="Bin widths (10 or 11 days, or monthly)")
    prsr.add_argument("--unjoined-vriles", action="store_true",
                      help="Bin unjoined VRILEs instead of joined (default)")
    script_tools.add_plot_cmd_args(prsr, fig_names="fig6")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Determine bin edges for a horizontal axis equal to the day of year
    # Define the monthly bin edges regardless as they are used to set the
    # plot axis labels later:
    bin_edges_mon = np.array([121, 152, 182, 213, 244, 274])

    if cmd.bins == "10d":
        # Split each month into 10 or 11 day bins so that there
        # are three roughly-equally-sized bins per month:
        bin_edges = np.array([121, 131, 141, 152, 162, 172, 182, 192,
                              202, 213, 223, 233, 244, 254, 264, 274])
    else:
        bin_edges = bin_edges_mon

    n_bins = len(bin_edges) - 1
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Load the cached VRILE and filtered track data:
    vriles_cice = cache.load("vriles_cice_1979-2023.pkl")
    vriles_ssmi = cache.load(f"vriles_ssmi-{cmd.ssmi_dataset}_1979-2023.pkl") 
    track_data  = cache.load("tracks_filtered_1979-2023.pkl")

    # n_regions includes pan Arctic (index 0), but in the functions below this
    # data is ignored and index 0 is set to the distribution across all sectors
    n_regions = len(vriles_cice)

    b_cice, b_cice_class, b_ssmi = bin_vriles(vriles_cice, vriles_ssmi,
                                              bin_edges=bin_edges,
                                              joined=not cmd.unjoined_vriles)

    b_tracks = bin_track_data(track_data[3], track_data[8],
                              bin_edges=bin_edges, n_sector_nei=1)

    frq = 1.E3  # factor to multiply frequency density by in plots

    # Keyword arguments for rectangle patches (CICE histogram):
    cice_rect_kw = {"edgecolor": "k", "zorder": -10, "clip_on": False}

    # Keyword arguments for SSM/I VRILEs and tracks, the histograms of which
    # are plotted as line plots:
    ssmi_line_kw  = {"color": "k", "linestyle": (0, (3, 3)), "zorder": -5,
                     "linewidth": 2*mpl.rcParams["lines.linewidth"]}

    track_line_kw = {"color": "limegreen", "zorder": -6,
                     "linewidth": ssmi_line_kw["linewidth"]}


    # Create figure and axes layout from template:
    fig, axs = style_ms.fig_layout_2x4(ylabel=r"Frequency density ($\times{}10^{-3}$)",
                                       lines_legend=True, lines_ssmi_kw=ssmi_line_kw,
                                       lines_tracks_kw=track_line_kw)

    # Add subplot panel titles (we use the long labels for all except
    # East-Siberian--Chukchi Sea, which is too long so replace with short):
    axs_titles = ["All sectors"] + cfg.reg_labels_long[1:]
    axs_titles[6] = cfg.reg_labels_short[6]
    tools.add_subplot_panel_titles(axs, axs_titles)

    # Create scalar mappable matching color bar (added in fig_layout_2x4) to
    # create colors for bars, plotted below:
    cm_sm = plt.cm.ScalarMappable(cmap=style_ms.vrile_class_cmap,
                                  norm=style_ms.vrile_class_norm)

    for r in range(n_regions):
        row = r // 4  # indices of axs[:,:] for this region
        col = r % 4

        # Histogram as bars for CICE:
        for b in range(n_bins):
            axs[row,col].add_patch(mpl.patches.Rectangle(
                (bin_edges[b], 0), width=bin_widths[b], height=frq*b_cice[r,b],
                facecolor=cm_sm.to_rgba(b_cice_class[r][b]), **cice_rect_kw))

        # Histograms as line plots for SSM/I VRILEs and cyclones:
        for data, kw in zip([frq*b_ssmi[r,:], frq*b_tracks[r,:]],
                            [ssmi_line_kw   , track_line_kw]):
            axs[row,col].plot(bin_edges[:-1] + .5*bin_widths, data, **kw)

    # For horizontal axis labels, set the major ticks as monthly-bin centers
    # These will be used to position the labels, but the ticks set invisible
    # Then the minor ticks (one halfway between each major tick) will be shown:
    #
    for ax in axs.flatten():
        ax.set_ylim(ymin=0.)
        ax.set_xlim(bin_edges_mon[0], bin_edges_mon[-1])
        ax.set_xticks(bin_edges_mon[:-1] + .5*(bin_edges_mon[1:] - bin_edges_mon[:-1]))
        ax.set_xticks(bin_edges_mon, minor=True)
        ax.set_xticklabels("MJJAS")
        ax.tick_params(axis="both", which="both", direction="out")
        ax.tick_params(axis="x", which="minor", bottom=True,
                       size=mpl.rcParams["xtick.major.size"])
        ax.tick_params(axis="x", which="major", bottom=False, pad=0.)
        ax.tick_params(axis="y", which="major", pad=1)
        ax.set_facecolor("none")

    # Default figure metadata title:
    if cmd.savefig_title is None:
        savefig_title = ("Seasonal and regional distribution of VRILEs and "
                         + "Arctic cyclones")
    else:
        savefig_title = cmd.savefig_title

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     fig_metadata={"Title": savefig_title})


if __name__ == "__main__":
    main()
