"""Common plotting functions."""

from datetime import datetime as dt, timezone as tz
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..io import config as cfg


def _set_default(x_in, x_default):
    """Set default value (x_default) of some parameter if it (x_in) is passed
    in as None.
    """
    if x_in is None:
        return x_default
    else:
        return x_in


def _set_kw_cycler(cyc_in, cyc_default):
    """Set property 'cycler' used to iterate over mpl appearance keyword
    arguments in various functions in this module. Input 'cyc_in' is the
    input which can either be None, a list/tuple/array, or something else.
    If None, returns cyc_default, which is passed in as the default cycle.
    If a list/tuple/array, returns cyc_in unchanged. Otherwise, returns
    [cyc_in], i.e., assuming that cyc_in is a single property (typically a
    string, e.g., 'red' for a color) and placing it into a list.
    """
    if cyc_in is None:
        return cyc_default
    elif cyc_in not in [list, tuple, np.ndarray]:
        return [cyc_in]
    else:
        return cyc_in



def _add_default_kw(kw_in, kw_default):
    """Add default keyword arguments (kw_default, dict) to input keyword
    arguments (kw_in, dict) if not present. This is to allow defaults to be
    overridden in various routines below. Returns a new (copy, possibly
    modified) version of kw_in.
    """

    kw_out = kw_in.copy()
    
    for k in kw_default.keys():
        if k not in kw_in.keys():
            kw_out[k] = kw_default[k]
    
    return kw_out


def distribute_subplots(axs, l=0.15, r=0.05, b=0.15, t=0.10, s_hor=0.05,
                        s_ver=0.05, vertically=False):
    """Uniformly distribute the subplots (array of matplotlib Axes instances)
    on a figure within specified margins and subplot spacing.
    
    
    Parameters
    ----------
    axs : array of matplotlib.axes.Axes instances [e.g., as returned by
          plt.subplots()]
        The subplots/axes to position, <= 2 dimensions.
    
    
    Optional parameters
    -------------------
    l, r, b, t: float, between 0 and 1
        Left, right, bottom, and top margins respectively, in figure
        coordinates.
    
    s_hor : float, default = 0.05
        Horizontal spacing between columns of subplots, in figure coordinates.
    
    s_ver : float, default = 0.05
        Vertical spacing between rows of subplots, in figure coordinates.
    
    vertically : bool, default = False
        This parameter is only used in cases where a 1D array of Axes is passed
        as axs. In this case the distribution direction is ambiguous (it could
        be a row or a column of subplots) so must be specified here: by
        default, the subplots are distributed horizontally.
    """
    
    # Input needs to be a 2D array. If a single Axes instance is input, or a 1D
    # array of Axes, expand the dimensions to make it a 2D array, and set a
    # flag to remove the expanded dimensions at the end:
    if np.ndim(axs) == 0:
        axs = np.array([[axs]])
        squeeze = True
    elif np.ndim(axs) == 1:
        if vertically:
            axs = axs[:, np.newaxis]
        else:
            axs = axs[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    
    ny, nx = np.shape(axs)
    
    # Compute the equal widths and heights of each Axes:
    axw = (1 - l - r - s_hor*(nx-1)) / nx
    axh = (1 - t - b - s_ver*(ny-1)) / ny
    
    for j in range(ny):
        for i in range(nx):
            axs[j,i].set_position([l + i*(s_hor + axw),
                                   b + (ny-j-1)*(s_ver + axh),
                                   axw, axh], which="both")
    
    if squeeze:
        axs = axs.squeeze()


def set_axis_ticks(ax, tick_start, tick_end, tick_step, minor_tick_divisions=0,
                   lims_actual=None, which="x"):
    """Set an Axes (ax) major ticks manually on a specified axis (which = 'x'
    or 'y') from tick_start to tick_end (inclusive) in steps of tick_step.


    Optional parameters
    -------------------
    lims_actual: tuple of float (xmin, xmax) or None
        The actual axes limits, if different from (tick_start, tick_end).

    which : str 'x' or 'y', default = 'x'
        Which axis to apply ticks to.
    """

    ticks = np.arange(tick_start, tick_end+tick_step/2, tick_step)

    if which == "x":
        ax.set_xticks(ticks)
        if lims_actual is None:
            ax.set_xlim(ticks[0], ticks[-1])
        else:
            ax.set_xlim(lims_actual)
    else:
        ax.set_yticks(ticks)
        if lims_actual is None:
            ax.set_ylim(ticks[0], ticks[-1])
        else:
            ax.set_ylim(lims_actual)


def set_cbar_axis_ticks(cbar, ticks, labels_fmt="{:.1f}", signed_labels=True):
    """Set ticks on a colorbar."""

    cbar.set_ticks(ticks)

    if signed_labels:
        cbar.set_ticklabels([(("+" if i > 0 else (u"\u2212" if i < 0 else ""))
                              + labels_fmt.format(abs(i)))
                             for i in cbar.get_ticks()])


def add_subplot_panel_titles(axs, titles=None, panel_label_fmt="({})",
                             panel_labels="abcdefghij", add_panel_labels=True,
                             title_kw={}):
    """Add subplot panel titles (descriptive).
    """

    if titles is None:
        titles = [""]

    if len(titles) < len(axs):
        titles += [""]*(len(axs)-len(titles))

    for j in range(min(len(titles), len(axs.flatten()))):
        axs.flatten()[j].set_title(
            panel_label_fmt.format(panel_labels[j])*add_panel_labels
            + " " + titles[j], **title_kw)


def discrete_cmap_for_pcolormesh(cmap_name, levels, set_under=False,
                                 set_over=False):
    """Determine the parameters required to make a colormap with discrete
    boundaries for use with plt.pcolormesh(). The reason for this is that
    that plotting function does not accept a 'levels' parameter.


    Parameters
    ----------
    cmap_name : str
        Name of a registered colormap.

    levels : array-like
        Data values corresponding to boundaries of each color interval. This
        should start at the smallest value (corresponding to the lower limit of
        the first interval) and and at the largest value (corresponding to the
        upper limit of the last interval). For example,

            levels = np.linspace(0., 1., 11)

        for 10 intervals [0, .1], [.1, .2], ..., [.9, 1.].


    Optional parameters
    -------------------
    set_under, set_over : bool, default False (both)
        Whether to set the first (set_under) and/or last (set_over) interval

    Returns
    -------
    Dictionary with keys 'cmap' and 'norm' set to the required values to be
    passed to plt.pcolormesh(). This can be expanded as required, e.g.:

        >>> cmap_kw = discrete_cmap_for_pcolormesh('RdBu', levels)
        >>> plt.pcolormesh(x, y, z, **cmap_kw)

    """

    if set_under or set_over:

        # Need to sub-sample the colours first:
        cmap_original = plt.cm.get_cmap(cmap_name, len(levels) + (set_under and set_over))

        colors_original = list(cmap_original(np.arange(len(levels)+1)))

        if set_over:
            custom_cmap = mpl.colors.ListedColormap(
                colors_original[set_under:-1], cmap_name + "_cutoff")

        else:
            custom_cmap = mpl.colors.ListedColormap(colors_original[set_under:],
                                                    cmap_name + "_cutoff")

        if set_over:
            custom_cmap.set_over(colors_original[-1])

        if set_under:
            custom_cmap.set_under(colors_original[0])

    else:
        custom_cmap = plt.cm.get_cmap(cmap_name, len(levels) - 1)

    custom_norm = mpl.colors.BoundaryNorm(levels, ncolors=len(levels) - 1)

    return {"cmap": custom_cmap, "norm": custom_norm}


def save_figure(fig, file_name="", set_raster_level=False, raster_dpi=300.,
                file_fmts=[".svg"], fig_metadata={}, save_dir=None):
    """Save a figure (fig, matplotlib Figure instance) to a specified file(s)
    in one or more formats.


    Optional parameters
    -------------------
    file_name : str, default = ''
        File name excluding extension. If '' or None, the figure canvas window
        title is used.

    set_raster_level : bool, default = False
        Set to True if figure contains rasterized elements and saving to a
        vector format. In this case the savefig dpi is set so that those
        elements are rasterized.

    raster_dpi : float, default = 300.
        DPI for rasterized elements if set_raster_level = True.

    file_fmts : list of str, default = ['.png', '.svg']
        File extensions for image formats to save.

    fig_metadata : dict, default = {}
        Metadata added to SVG output only. Only specific keys are allowed; see

        https://matplotlib.org/stable/api/backend_svg_api.html#matplotlib.backends.backend_svg.FigureCanvasSVG.print_svg

        for details. Generally, only 'Title' (case sensitive) should be
        provided here.

    save_dir : str or pathlib.Path or None
        Directory to save figures. If None, get from src.io.config.
    """

    # Known raster and vector forms to check during loop below:
    raster_fmts = [".gif", ".jpg", ".png"]
    vector_fmts = [".eps", ".pdf", ".ps", ".svg"]

    if save_dir is None:
        save_dir = cfg.fig_save_dir

    # Make save directory if it doesn"t exist:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set fig file names to the window title if not provided:
    if file_name is None or len(file_name) == 0:
        file_name = fig.canvas.manager.get_window_title()

    file_name = file_name.replace(" ", "_")

    for fmt in file_fmts:

        kw = {}

        # Set DPI if saving as a raster format or with raster elements embedded
        # in a vector format:
        if (fmt in raster_fmts or (fmt in vector_fmts and set_raster_level)):
            kw["dpi"] = raster_dpi

        # Metadata is added to SVG output only. Only certain keys are allowed
        # (by the backend SVG writer) and an error is raised if something else
        # is passed. They are case sensitive. In scripts, generally only the
        # 'Title' need be passed in to this function. Other common values are
        # added here. It is only possible to add one 'Contributor' (there seems
        # to be a backend bug as only one from the list is saved).
        if "svg" in fmt:
            kw["metadata"] = _add_default_kw(fig_metadata,
                {"Date": dt.now(tz.utc).strftime("%H:%M UTC %d %b %Y"),
                 "Language": "English", "Contributor": [cfg.author]})

        fig.savefig(Path(save_dir, file_name + fmt), **kw)
        print(f"Saved: {str(Path(save_dir, file_name + fmt))}")


def finish_fig(fig, savefig=False, no_show=False, **kwargs):
    """Final steps for use in scripts that generate figures; either saves the
    figure or displays it interactively. Takes the Figure instance (fig), an
    optional parameter savefig (bool: whether to save), an optional parameter
    no_show (bool: supress plt.show if True, default False so that figures are
    generally shown interactively, automatically), and additional keyword
    arguments are passed to plot_tools.save_figure().
    """
    if savefig:
        save_figure(fig, **kwargs)
    elif not no_show:
        plt.show()

