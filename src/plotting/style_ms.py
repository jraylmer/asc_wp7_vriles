"""Defines colors, colormaps, and linestyles for different plot elements in the
manuscript figure style. Also provides a function for a common figure layout
with a customised external legend.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import maps, symbols, tools


_lw_0 = 0.5            # linewidth scale
_fs_0 = 7.5            # fontsize scale
_dfs  = 1.0            # fontsize adjustment scale
_ms_0 = 3.5            # default marker size
_font = "Nimbus Sans"  # default font

cmap = {
    "daidt"      : "RdBu",
    "dvidtd"     : "PiYG",
    "dvidtt"     : "PiYG",
    "div_strair" : "PuOr",
    "curl_strair": "PuOr",
    "qlw"        : "Spectral_r",
    "qsw"        : "Spectral_r",
    "qnet"       : "Spectral_r",
    "seb_ai"     : "Spectral_r",
    "ssmi"       : "RdBu",
    "t2"         : "RdBu_r",
    "meltb"      : "PiYG",
    "meltl"      : "PiYG",
    "meltt"      : "PiYG"
}

# Colorbar used to illustrate VRILE classification
# See e.g., function fig_layout_2x4():
#
vrile_class_cmap  ="RdBu_r"
vrile_class_norm  = mpl.colors.BoundaryNorm(np.linspace(-1, 1, 11), ncolors=256)
vrile_class_ticks = np.array([-1., -.6, -.2, .2, .6, 1.])

# Figure width options (in inches). These match AMS style:
#
# https://www.ametsoc.org/index.cfm/ams/publications/author-information/
#     figure-information-for-authors/#Size
#
fig_width_single_column  = 3.2
fig_width_2third_pwidth  = 4.5
fig_width_double_column  = 5.5
fig_width_extra_large    = 6.5

# Default figure heights:
fig_height_single_column = fig_width_single_column * 9./16.
fig_height_2third_pwidth = fig_width_2third_pwidth * 9./16.
fig_height_double_column = fig_width_double_column * 9./16.
fig_height_extra_large   = fig_width_extra_large   * 9./16.


def _set_mpl_rcParams(font=_font, fs_0=_fs_0, lw_0=_lw_0, ms_0=_ms_0, dfs=_dfs):
    """Set customised matplotlib rcParams. See

        matplotlib.org/stable/users/explain/customizing.html

    for explanation of these rcParams.
    """

    # Sans-serif fonts is the default family for text and math text already
    # (which is what we want), so just need to set the font for the sans-serif
    # family. Parameter 'font.sans-serif' is a list of fonts of which the first
    # available is selected and used. So prepend the desired font to that list:
    mpl.rcParams["font.sans-serif"].insert(0, font)

    # Update default font size:
    mpl.rcParams["font.size"] = fs_0

    # Set math text to use custom settings, which means that each style (bold,
    # italic, etc.) is specified individually. The default values for those are
    # fine (point to the appropriate sans-serif fonts variants) so just need to
    # set that mathtext fontset to custom:
    mpl.rcParams["mathtext.fontset"] = "custom"

    # No cursive variant is available for the default font choice:
    mpl.rcParams["mathtext.cal"] = "sans:italic"

    # Set various plot defaults. Line plots [ax.plot()]:
    mpl.rcParams["lines.linewidth"]       = lw_0
    mpl.rcParams["lines.markeredgewidth"] = lw_0
    mpl.rcParams["lines.markersize"]      = ms_0
    mpl.rcParams["lines.dash_joinstyle"]  = "miter"
    mpl.rcParams["lines.dash_capstyle"]   = "projecting"
    mpl.rcParams["lines.solid_joinstyle"] = "miter"
    mpl.rcParams["lines.solid_capstyle"]  = "projecting"

    # Other types of plotting:
    mpl.rcParams["scatter.edgecolors"] = "none"

    # Axes parameters:
    mpl.rcParams["axes.titlesize"]      = fs_0
    mpl.rcParams["axes.titlepad"]       = 8.
    mpl.rcParams["axes.titleweight"]    = "normal"
    mpl.rcParams["axes.titlelocation"]  = "left"
    mpl.rcParams["xaxis.labellocation"] = "right"
    mpl.rcParams["yaxis.labellocation"] = "top"
    mpl.rcParams["axes.labelsize"]      = fs_0
    mpl.rcParams["axes.labelpad"]       = 8.
    mpl.rcParams["axes.spines.right"]   = False
    mpl.rcParams["axes.spines.top"]     = False
    mpl.rcParams["axes.axisbelow"]      = True
    mpl.rcParams["axes.linewidth"]      = lw_0

    # Axes tick parameters:
    for x, side_ticks, side_noticks in zip(["x"     , "y"    ],
                                           ["bottom", "left" ],
                                           ["top"   , "right"]):
        mpl.rcParams[f"{x}tick.direction"]            = "in"
        mpl.rcParams[f"{x}tick.labelsize"]            = fs_0
        mpl.rcParams[f"{x}tick.major.width"]          = lw_0
        mpl.rcParams[f"{x}tick.major.size"]           = 2.
        mpl.rcParams[f"{x}tick.major.pad"]            = 3.
        mpl.rcParams[f"{x}tick.minor.visible"]        = False
        mpl.rcParams[f"{x}tick.minor.width"]          = lw_0
        mpl.rcParams[f"{x}tick.minor.size"]           = 1.

        mpl.rcParams[f"{x}tick.{side_ticks}"]         = True
        mpl.rcParams[f"{x}tick.major.{side_ticks}"]   = True
        mpl.rcParams[f"{x}tick.minor.{side_ticks}"]   = False
        mpl.rcParams[f"{x}tick.{side_noticks}"]       = False
        mpl.rcParams[f"{x}tick.major.{side_noticks}"] = False
        mpl.rcParams[f"{x}tick.minor.{side_noticks}"] = False

    # Grid off by default, but set style anyway:
    mpl.rcParams["axes.grid"]      = False
    mpl.rcParams["grid.color"]     = [.95]*3
    mpl.rcParams["grid.linewidth"] = lw_0
    mpl.rcParams["grid.linestyle"] = "-"

    # Legends:
    mpl.rcParams["legend.fancybox"]  = False
    mpl.rcParams["legend.fontsize"]  = fs_0 - 2*dfs
    mpl.rcParams["legend.facecolor"] = "none"
    mpl.rcParams["legend.edgecolor"] = "none"

    # Figure setup (figsize must be given in inches):
    mpl.rcParams["figure.figsize"] = (fig_width_double_column,
                                      fig_height_double_column)

    # Miscelleneous:
    mpl.rcParams["axes.autolimit_mode"] = "round_numbers"
    mpl.rcParams["patch.linewidth"]     = lw_0
    mpl.rcParams["svg.fonttype"]        = "none"  # keep text and embed fonts


def _set():
    """Set manuscript figure style by updating default parameters in matplotlib
    and local modules.
    """

    print("Setting manuscript figure style")

    _set_mpl_rcParams()

    # Update style parameters in maps module:
    maps.default_ice_edge_colors     = ["k"]
    maps.default_ice_edge_linestyles = ["--", "-"]
    maps.default_track_colors        = ["limegreen"]
    maps.default_track_linewidths    = [2.*mpl.rcParams["lines.linewidth"]]
    maps.default_track_linestyles    = ["-"]

    maps.default_sector_bound_color = "tab:red"
    maps.default_grid_lats          = np.concatenate((np.arange(60., 90., 5.), [89.5]))
    maps.default_grid_lons          = np.arange(0., 360., 30.)
    maps.default_grid_lats_lon_lims = [np.array([0.0, 360.0])
                                       for k in range(len(maps.default_grid_lats))]
    maps.default_grid_lons_lat_lims = [50., 89.5]

    maps.default_grid_lat_labels      = True
    maps.default_grid_lat_labels_lats = np.array([65., 70., 75., 80.])
    maps.default_grid_lat_labels_lons = np.array([315.]*4)

    maps.default_grid_line_kw            = {}
    maps.default_grid_lat_labels_text_kw = {"color": [.6]*3,
                                           "fontsize": .5*mpl.rcParams["font.size"]}

    maps.default_grid_line_kw  = {"alpha": .2, "color": [.2]*3,
                                  "linewidth": mpl.rcParams["grid.linewidth"]}

    maps.default_land_patch_kw = {"facecolor": "lightgrey"}
    maps.default_coast_line_kw = {"edgecolor": "k",
                                  "linewidth": mpl.rcParams["grid.linewidth"]}


def fig_layout_2x4(invert=False, fig_width=fig_width_double_column,
                   fig_height_frac=.55, ax_l=.07, ax_r=.015, ax_t=.07, ax_b=.21,
                   ax_s_ver=.12, ax_s_hor=.05, xlabel=None, xlabel_ax_b_frac=.7,
                   ylabel=None, ylabel_ax_l_frac=.3,
                   vrile_class_cbar=True, cbar_y0=.07, cbar_height=.03,
                   vrile_class_ticks=vrile_class_ticks,
                   vrile_class_cmap=vrile_class_cmap,
                   vrile_class_norm=vrile_class_norm,
                   lines_legend=False, lines_legend_x0=.725, lines_legend_dy=.01,
                   lines_legend_width=.04, lines_ssmi_kw={},
                   lines_tracks_kw={}):
    """Create a common layout for figures that have 2 rows by 4 columns of
    panels, with a custom legend at the bottom containing a colorbar for VRILE
    classification and a line legend drawn on the figure.

    Some parameters below change the precise layout but there are some tuning
    factors hard-coded in some places.


    Optional parameters
    -------------------
    invert : bool, default = False
        If True, arrange as 4 rows by 2 columns (in which case, adjustment of
        the other parameters below may be necessary).

    fig_width : float
        The figure width in inches (default: fig_width_double_column, set in
        style_ms module).

    fig_height_frac : float, default = 0.55
        The figure height expressed as a fraction of fig_width.

    ax_l, ax_r, ax_t, ax_b, ax_s_ver, ax_s_hor
        Parameters passed to plotting.tools.distribute_subplots() function,
        describing the axes left, right, top, and bottom margins, and vertical
        and horizontal spacing between rows/columns, respectively, all in
        figure units.

    xlabel : str or None (default)
        If a string, uses this as a common x-axis label plotted under and
        centered horiztonally against the bottom-most row of subplots.

    xlabel_ax_b_frac : float, default = 0.7
        Vertical position of the centre of the xlabel, if drawn, expressed as a
        fraction of ax_b.

    ylabel : str or None (default)
        If a string, uses this as a common y-axis label plotted on the left
        of and centered vertically against the left-most column of subplots.

    ylabel_ax_l_frac : float, default = 0.3
        Horizontal position of the centre of the ylabel, if drawn, expressed
        as a fraction of ax_l.

    vrile_class_cbar : bool, default = True
        Whether to add the colorbar for VRILE classification metric on the
        bottom-left half of the figure.

    cbar_y0, cbar_height : float, defaults 0.07 and 0.03 respectively
        The y-coordinate of the bottom and height of the colorbar expressed
        in figure units.

    vrile_class_ticks : list or array of float
        Which values to draw ticks and labels for the colorbar.

    vrile_class_cmap, vrile_class_norm
        Matplotlib colormap identifier and Normalize instance, to create
        a ScalarMappable instance hence defining the colorbar.

    lines_legend : bool, default = False
        If True, add a legend for 'SSM/I' and 'JRA-55 cyclones' lines
        in the space to the right of the colorbar.

    lines_legend_x0, lines_legend_dy, lines_legend_width : float
        Respectively, the horizontal coordinate of the left edge of the
        lines, additional vertical spacing with respect to the top/bottom edge
        of the colorbar for the vertical position of the two lines, and
        the line widths, all in figure units (the lines are drawn centered
        with respect to the position of the colorbar).

    lines_ssmi_kw, lines_tracks_kw : dict
        Keyword arguments passed to mpl.lines.Line2D defining the line
        appearances (these should correspond to the plots).


    Returns
    -------
    fig, axs
        Matplotlib Figure instance and array of Axes.

    """

    fig, axs = plt.subplots(nrows=4 if invert else 2, ncols=2 if invert else 4,
                            figsize=(fig_width, fig_height_frac*fig_width))

    tools.distribute_subplots(axs, l=ax_l, r=ax_r, b=ax_b, t=ax_t,
                              s_ver=ax_s_ver, s_hor=ax_s_hor)

    for ax in axs.flatten():
        for spine in ax.spines:
            ax.spines[spine].set_zorder(1E10)

    if xlabel is not None:
        fig.text(ax_l + (1. - ax_l - ax_r)/2., xlabel_ax_b_frac*ax_b, xlabel,
                 ha="center", va="center",
                 fontsize=mpl.rcParams["axes.labelsize"])

    if ylabel is not None:
        fig.text(ylabel_ax_l_frac*ax_l, ax_b + (1. - ax_b - ax_t)/2., ylabel,
                 ha="center", va="center", rotation=90.,
                 fontsize=mpl.rcParams["axes.labelsize"])

    if vrile_class_cbar:
        # The cbar position is with respect to the axes positions but there
        # are some tuning factors (I can't remember exactly where the 2-0.25
        # etc. came from... something to do with making it about half of the
        # space spanned by the axes and the 0.25 is related to making space
        # for the {thermo}dynamic text labels either end):
        cbar_ax = fig.add_axes([ax_l + 0.3*0.25*(1 - ax_l - ax_r - ax_s_hor),
                                cbar_y0,
                                (2-0.25)*0.25*(1 - ax_l - ax_r),
                                cbar_height])

        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=vrile_class_cmap,
                                                  norm=vrile_class_norm),
                            cax=cbar_ax, orientation="horizontal")

        cbar_ax.set_xlabel("Mean simulated VRILE classification, $"
                           + symbols.vrile_class + "$",
                           loc="center", labelpad=3)

        cbar_ax.tick_params(which="major", axis="x", direction="out",
                            left=False, right=False, top=False, bottom=True)

        cbar.minorticks_off()
        cbar_ax.xaxis.set_label_position("top")
        cbar.set_ticks(vrile_class_ticks)
        cbar.ax.set_xticklabels(
            ["%s%.1f" % ("+" if i > 0 else ("" if i == 0. else u"\u2212"), abs(i))
             for i in cbar.get_ticks()])

        # Text labels at each end of the colorbar. The factor 0.005 gives some
        # additional horizontal padding between the text and colorbar edge.
        # The 0.45 factor in the text vertical position accounts for the text
        # bounding box being used to centre; as such the text itself does not
        # appear exactly centered 'visually'. So the 0.45 offsets it a bit to
        # compensate:
        for txt, x0, ha in zip(
                ["dynamic", "thermodynamic"],
                [cbar_ax.get_position().x0 - .005, cbar_ax.get_position().x1 + .005],
                ["right", "left"]):
            fig.text(x0, cbar_y0 + .45*cbar_height, txt, ha=ha, va="center",
                     fontstyle="italic", fontsize=mpl.rcParams["axes.labelsize"]-1)

    # Legend entries for observations and cyclones:
    if lines_legend:

        tools._add_default_kw(lines_tracks_kw, {"color": "limegreen"})
        tools._add_default_kw(lines_ssmi_kw  , {"color": "tab:gray",
                                                "linestyle": "--"})

        for y_offset, label, line_kw in zip(
                [cbar_height + lines_legend_dy, -lines_legend_dy ],
                ["SSM/I VRILEs"               , "JRA-55 cyclones"],
                [lines_ssmi_kw                , lines_tracks_kw  ]):

            fig.add_artist(mpl.lines.Line2D(
                [lines_legend_x0, lines_legend_x0 + lines_legend_width],
                [cbar_y0 + y_offset]*2, **line_kw))

            # The 1.2 gives some padding between right end of line and label:
            fig.text(lines_legend_x0 + lines_legend_width*1.2,
                     cbar_y0 + y_offset, label, ha="left", va="center",
                     fontsize=mpl.rcParams["axes.labelsize"])

    return fig, axs


# Invoke the set function on import:
_set()
