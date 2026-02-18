"""Create two sets of scatter plots of changes in sea ice area/extent/volume
plotted against VRILE diagnostics. Various options for the x/y axis quantities
are coded to be selected by command-line flags including detrended/anomalies or
absolute quantities. The primary reason for the two sets (i.e., two figures of
8 subplots) is for looking for a quantity which 'characterises' or gives some
insight into the dynamically-dominated VRILEs and one for thermodynamically-
dominated VRILEs.

By default, this script generates Figs. 8 and 9 of the manuscript.

By running the script with --thm-x t2, the thermodynamic plot (second figure)
given as supplementary figure S4 can be generated.
"""

from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate

from src import script_tools
from src.io import cache, config as cfg
from src.plotting import style_ms, symbols, tools


# Define all the x or y axes labels. The keys of this dictionary will also
# become the choices for command line arguments to choose what to plot:
#
axes_labels = {
    "div_strair" : r"Wind-stress divergence ($10^{-7}$ N m$^{-3}$)",
    "curl_strair": r"Wind-stress curl ($10^{-7}$ N m$^{-3}$)",
    "seb_ai"     : r"Surface energy balance (W m$^{-2}$)",
    "qlw"        : r"Surface downwelling longwave radiation (W m$^{-2}$)",
    "qsw"        : r"Surface downwelling shortwave radiation (W m$^{-2}$)",
    "qnet"       : r"Surface net downwelling radiation (W m$^{-2}$)",
    "t2"         : r"Surface air temperature ($\degree$C)",
    "meltb"      : r"Sea ice basal melt rate (cm day$^{-1}$)",
    "meltl"      : r"Sea ice lateral melt rate (cm day$^{-1}$)",
    "meltt"      : r"Sea ice top melt rate (cm day$^{-1}$)",
    "dsie"       : r"|$\Delta$SIE| ($10^3$ km$^2$ day$^{-1}$)",
    "daidt"      : r"|$\Delta$SIA| ($10^3$ km$^2$ day$^{-1}$)",
    "dvidt"      : r"|$\Delta$SIV| (km$^3$ day$^{-1}$)",
    "daidtt"     : r"|$\Delta$SIA| (thermodynamics; $10^3$ km$^2$ day$^{-1}$)",
    "daidtd"     : r"|$\Delta$SIA| (dynamics; $10^3$ km$^2$ day$^{-1}$)",
    "dvidtt"     : r"|$\Delta$SIV| (thermodynamics; km$^3$ day$^{-1}$)",
    "dvidtd"     : r"|$\Delta$SIV| (dynamics; km$^3$ day$^{-1}$)"
}

# Use the string '_det' to distinguish detrended/anomaly versions
axes_labels["div_strair_det"]  = r"Wind-stress divergence anomaly ($10^{-7}$ N m$^{-3}$)"
axes_labels["curl_strair_det"] = r"Wind-stress curl anomaly ($10^{-7}$ N m$^{-3}$)"
axes_labels["seb_ai_det"]      = r"Surface energy balance anomaly (W m$^{-2}$)"
axes_labels["qlw_det"]         = r"Surface downwelling longwave radiation anomaly (W m$^{-2}$)"
axes_labels["qsw_det"]         = r"Surface downwelling shortwave radiation anomaly (W m$^{-2}$)"
axes_labels["qnet_det"]        = r"Surface net downwelling radiation anomaly (W m$^{-2}$)"
axes_labels["t2_det"]          = r"Surface air temperature anomaly ($\degree{}$C)"
axes_labels["meltb_det"]       = r"Sea ice basal melt rate anomaly (cm day$^{-1}$)"
axes_labels["meltl_det"]       = r"Sea ice lateral melt rate anomaly (cm day$^{-1}$)"
axes_labels["meltt_det"]       = r"Sea ice top melt rate anomaly (cm day$^{-1}$)"
axes_labels["dsie_det"]        = r"VRILE magnitude ($10^3$ km$^2$ day$^{-1}$)"
axes_labels["daidt_det"]       = r"|$\Delta$SIA| (detrended; $10^3$ km$^2$ day$^{-1}$)"
axes_labels["dvidt_det"]       = r"|$\Delta$SIV| (detrended; km$^3$ day$^{-1}$)"
axes_labels["daidtd_det"]      = r"|$\Delta$SIA| (dynamics; detrended; $10^3$ km$^2$ day$^{-1}$)"
axes_labels["daidtt_det"]      = r"|$\Delta$SIA| (thermo.; detrended; $10^3$ km$^2$ day$^{-1}$)"
axes_labels["dvidtd_det"]      = r"|$\Delta$SIV| (dynamics; detrended; km$^3$ day$^{-1}$)"
axes_labels["dvidtt_det"]      = r"|$\Delta$SIV| (thermo.; detrended; km$^3$ day$^{-1}$)"

# List of options for command line flags:
_allowed_metrics = list(axes_labels.keys())

# For each possible quantity, define a scale and offset such that the actual
# data plotted is scale*data + offset, e.g., for units change. Set these to
# 1 and 0 respectively initially, then adjust for some cases as required below:
#
scale  = dict.fromkeys(list(axes_labels.keys()), 1.)
offset = dict.fromkeys(list(axes_labels.keys()), 0.)

# Change the sign of all area/volume tendency terms (=> magnitudes):
for x in ["d{}idt{}", "d{}idtd{}", "d{}idtt{}"]:
    for y in "av":
        for z in ["", "_det"]:
            scale[x.format(y,z)] = -1.

offset["t2"] = -273.15  # convert undetrended air temp. to degC


def get_data(cache_diags_filename, cmd_diags, data_out):
    """Generic function to load cached diagnostics."""

    # Load this dataset:
    data = cache.load(cache_diags_filename)

    # cmd_diags does not include the diagnostic names in the same format
    # as what is in the cached files (needs '_d' at the end and, for detrended,
    # not '_det' at the end but 'detrended_' at the beginning). Create a
    # corrected list for use in this function:
    _cmd_diags = []
    for it in cmd_diags:
        if it.endswith("_det"):  # detrended diagnostic
            _cmd_diags.append(f"detrended_{it[:-4]}_d")
        else:  # non-detrended diagnostic
            _cmd_diags.append(f"{it}_d")

    # Loop over actual diagnostics saved (header of data):
    for d in range(len(data[0])):
        # Proceed if this diagnostic is required:
        if data[0][d] in _cmd_diags:
            # Find indices of _cmd_diags that need to be assigned this
            # diagnostic (could be more than one):
            j_out = [j for j in range(len(_cmd_diags))
                    if _cmd_diags[j] == data[0][d]]
            # Access, scale/offset, and assign the data to each required index
            # (note each data element apart from the header is a list of array):
            for j in j_out:
                data_out[j] = [offset[cmd_diags[j]] + scale[cmd_diags[j]]*arr
                               for arr in data[1+d]]

    # An exception: if cached data is non-detrended history/atmo fields
    # and qnet is required, must be got from qsw + qlw:
    if "qnet_d" in _cmd_diags and "qlw_d" in data[0] and "qsw_d" in data[0]:
        j_out = [j for j in range(len(_cmd_diags))
                 if _cmd_diags[j] == "qnet_d"]
        i_lw = data[0].index("qlw_d")
        i_sw = data[0].index("qsw_d")
        for j in j_out:
            data_out[j] = [scale["qlw"]*data[i_lw][k] + scale["qsw"]*data[i_sw][k]
                           for k in range(len(data[i_lw]))]


def is_case_study(region_index, vrile_index, case_studies=[]):
    """Return boolean indicating if specified region and vrile indices
    correspond to a case study.
    """
    is_case = False
    for cs in range(len(case_studies)):
        if (region_index == case_studies[cs][0]
                and vrile_index == case_studies[cs][2]):
            is_case = True
            break
    return is_case


def main():

    prsr = script_tools.argument_parser(usage="Scatter plots")

    prsr.add_argument("--thm-only", action="store_true",
                      help="Plot thermodynamic VRILEs only")
    prsr.add_argument("--dyn-only", action="store_true",
                      help="Plot dynamic VRILEs only")

    prsr.add_argument("--dyn-x", type=str, default="div_strair_det",
                      choices=_allowed_metrics)
    prsr.add_argument("--dyn-y", type=str, default="daidtd_det",
                      choices=_allowed_metrics)

    prsr.add_argument("--thm-x", type=str, default="qlw_det",
                      choices=_allowed_metrics)
    prsr.add_argument("--thm-y", type=str, default="daidtt_det",
                      choices=_allowed_metrics)

    script_tools.add_plot_cmd_args(prsr, n_figures=2,
                                   fig_names=["fig8", "fig9"],
                                   fig_titles=[None, None])
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Case studies [sector_id, rank] to highlight on the plot:
    case_studies = [[0, 1], [4, 23], [5, 5], [4, 38]]
    # Later append: [VRILE ID, VRILE magnitude, classification, cyclone or not]

    xy_diag = [cmd.dyn_x, cmd.dyn_y, cmd.thm_x, cmd.thm_y]

    xy_data   = [np.zeros(0) for j in range(4)]    # data to plot
    xy_except = [False for j in range(4)]          # to check for exceptions
    xy_labels = [axes_labels[x] for x in xy_diag]  # axes labels

    # Load data and assign to xy_data list as required:
    file_prefix = "vriles_cice_1979-2023"

    get_data(f"{file_prefix}_diagnostics_div_curl.pkl" , xy_diag, xy_data)
    get_data(f"{file_prefix}_diagnostics_hist_atmo.pkl", xy_diag, xy_data)
    get_data(f"{file_prefix}_diagnostics_hist_atmo_detrended.pkl", xy_diag, xy_data)

    xy_except = [len(xy_data[j]) == 0 for j in range(len(xy_data))]
    if any(xy_except):
        raise Exception("Not all of the required data was loaded:\n"
                        + ("  Dynamics plot: x not loaded\n"*xy_except[0])
                        + ("  Dynamics plot: y not loaded\n"*xy_except[1])
                        + ("  Thermo.  plot: x not loaded\n"*xy_except[2])
                        + ("  Thermo.  plot: y not loaded\n"*xy_except[3]))

    # This next (long) part of the script determines and prints various
    # correlations/slopes for each sector/panel, separating by dynamic/
    # thermodynamic and cyclone/non-cyclone associated, in case those values
    # are of interest. The results are printed and saved to a text file.
    #
    # We force use of joined VRILEs for now (as that is how the diagnostics are
    # computed) but keep the text variable 'vrds' in case this changes in the
    # future:
    #
    vrds = "_joined"

    # The text output will be saved to a file under cfg.data_path['tables'].
    # The filename needs to contain some info. about the diagnostics being used:
    txt_fname =  f"scatter_stats_{xy_diag[1]}_vs_{xy_diag[0]}_and_"
    txt_fname += f"{xy_diag[3]}_vs_{xy_diag[2]}.txt"

    # Prepare the text to be written with some header information:
    txt  = "Scatter plot statistics\n"
    txt += "-----------------------\n\n"
    txt += dt.today().strftime("Generated: %H:%M %a %d %b %Y") + "\n\n"
    
    txt += f"Using {'un' if vrds=='' else ''}joined VRILES\n\n"

    txt += f"Dynamics       plot, X -quantity: {xy_diag[0]}\n"
    txt += f"Dynamics       plot,  Y-quantity: {xy_diag[1]}\n"
    txt += f"Thermodynamics plot, X -quantity: {xy_diag[2]}\n"
    txt += f"Thermodynamics plot,  Y-quantity: {xy_diag[3]}\n\n"

    # Load VRILE data (needed for classifcations and track indices):
    vriles = cache.load("vriles_cice_1979-2023.pkl")

    n_regions = len(vriles)
    vrile_track_ids = [x["track_ids"] for x in vriles]

    # Add information about the case studies to the text output (this also
    # serves as a sanity check that the data is being loaded correctly):
    if len(case_studies) > 0:
        txt += "Case studies identified on figures:\n"

    for cs in range(len(case_studies)):
        r = case_studies[cs][0]
        k = case_studies[cs][1]

        # Append the index of the VRILE in the results array:
        i = np.argmin(abs(vriles[r][f"vriles{vrds}_rates_rank"] - k))
        case_studies[cs].append(i)

        # Append the VRILE magnitude:
        case_studies[cs].append(vriles[r][f"vriles{vrds}_rates"][i])

        # Append the classification (-1 to +1):
        case_studies[cs].append(vriles[r][f"vriles{vrds}_class"][i])

        # Append whether it matches a track(s) or not:
        case_studies[cs].append(len(vrile_track_ids[r][i]) > 0)

        txt += f"  {cs+1}. "
        txt += vriles[r][f"date_bnds_vriles{vrds}"][i,0].strftime("%Y%m%d")
        txt += "-"
        txt += vriles[r][f"date_bnds_vriles{vrds}"][i,1].strftime("%Y%m%d")
        txt += f", {cfg.reg_labels_short[r]} sector, rank: "
        txt += f"{vriles[r][f'vriles{vrds}_rates_rank'][i]:02}/"
        txt += f"{vriles[r][f'n{vrds}_vriles']:02}, class: C = "
        txt += ("+" if vriles[r][f"vriles{vrds}_class"][i] >= 0. else "-")
        txt += f"{abs(vriles[r][f'vriles{vrds}_class'][i]):.3f}, "
        txt += "associated track IDs: "

        if len(vrile_track_ids[r][i]) > 0:
            txt += ", ".join([str(x) for x in vrile_track_ids[r][i]]) + "\n"
        else:
            txt += "none\n"

    txt += "\n"

    # Create header rows for three separate tables:
    #
    #     Table 1: considering all VRILEs on each panel together
    #     Table 2: separate by classification (C) in each panel
    #     Table 3: separate by cyclone vs non-cyclone in each panel
    #
    # Tabulate the correlation (r) and slope (s) of the data in each subset for
    # each set of scatter plots (the 'dynamic' and 'thermodynamic' quantities):
    #
    headers_all = ["SEC", "N", "r(DYN)", "s(DYN)", "r(THM)", "s(THM)"]
    headers_cls = ["SEC", "C<=0", "r(DYN)", "s(DYN)", "r(THM)", "s(THM)",
                          "C>0" , "r(DYN)", "s(DYN)", "r(THM)", "s(THM)"]
    headers_cyc = ["SEC", "T>0" , "r(DYN)", "s(DYN)", "r(THM)", "s(THM)",
                          "T=0" , "r(DYN)", "s(DYN)", "r(THM)", "s(THM)"]

    # Helper functions for formatting table entries:
    def fmt_corrcoef(x, y):
        return f"{np.corrcoef(x, y)[0,1]:.3f}"

    def fmt_regression(x, y):
        return f"{np.polyfit(x, y, 1)[0]:.2e}"

    rows_all = []
    rows_cls = []
    rows_cyc = []

    n_tot = []  # append total number of VRILEs in each sector
    n_thm = []  # number of thermodynamic (C > 0) VRILEs in each sector
    n_cyc = []  # number with non-zero associated tracks (T > 0) in each sector

    # Combine arrays in each loop so that we can compute "all sectors combined"
    # and append to the end.
    #
    # To avoid confusion* with the 'dynamic'/'thermodynamic' quantity being
    # plotted and the 'dynamic'/'thermodynamic' classification being separated
    # here, use (xd_, yd_) and (xt_, yt_) to refer to the former and then 'dyn'
    # and 'thm' to refer to the classification in the following variable names:
    xd_all = [] ; yd_all = []
    xt_all = [] ; yt_all = []

    # Separate thermodynamic (C > 0) and dynamic (C <= 0):
    xd_dyn = [] ; yd_dyn = []  # dynamic x,y quantities, DYNamic VRILEs only
    xd_thm = [] ; yd_thm = []  # dynamic x,y quantities, THerModynamic VRILEs

    xt_dyn = [] ; yt_dyn = []  # thermodynamic x,y quantities, DYNamic VRILEs
    xt_thm = [] ; yt_thm = []  # thermodynamic x,y quantities, THerModynamic VRILEs

    # Separate cyclone (T > 0) vs non-cyclone (T = 0). Use '_cyc' and '_noc'
    # for these in analogy to '_dyn' and '_thm' above:
    xd_cyc = [] ; yd_cyc = []  # dynamic x,y quantities, CYClone associated VRILEs
    xd_noc = [] ; yd_noc = []  # dynamic x,y quantities, NOt Cyclone associated VRILEs

    xt_cyc = [] ; yt_cyc = []  # (similar for thermodynamic x,y quantities)
    xt_noc = [] ; yt_noc = []

    # *okay it's still confusing but I don't know how else to do it right now
    #
    # Also in the text output to keep the headers of the tables short, 'DYN'
    # and 'THM' are used to distinguish the 'dynamic'/'thermodynamic' x,y
    # quantities, not the classification, as done here for variable names...

    for r in range(n_regions):
        n_tot.append(vriles[r][f"n{vrds}_vriles"])

        # X and Y data: all VRILEs:
        xd_all.append(xy_data[0][r])
        yd_all.append(xy_data[1][r])
        xt_all.append(xy_data[2][r])
        yt_all.append(xy_data[3][r])

        # Separate thermodynamic (C > 0) and dynamic (C <= 0):
        xd_dyn.append(xd_all[-1][vriles[r][f"vriles{vrds}_class"] <= 0])
        yd_dyn.append(yd_all[-1][vriles[r][f"vriles{vrds}_class"] <= 0])
        xd_thm.append(xd_all[-1][vriles[r][f"vriles{vrds}_class"]  > 0])
        yd_thm.append(yd_all[-1][vriles[r][f"vriles{vrds}_class"]  > 0])

        xt_dyn.append(xt_all[-1][vriles[r][f"vriles{vrds}_class"] <= 0])
        yt_dyn.append(yt_all[-1][vriles[r][f"vriles{vrds}_class"] <= 0])
        xt_thm.append(xt_all[-1][vriles[r][f"vriles{vrds}_class"]  > 0])
        yt_thm.append(yt_all[-1][vriles[r][f"vriles{vrds}_class"]  > 0])

        # Separate cyclone (T > 0) and non-cyclone (T = 0):
        v_has_cyclone = np.array([len(vrile_track_ids[r][v]) > 0
                                  for v in range(n_tot[-1])])

        xd_cyc.append(xd_all[-1][v_has_cyclone])
        yd_cyc.append(yd_all[-1][v_has_cyclone])
        xd_noc.append(xd_all[-1][~v_has_cyclone])
        yd_noc.append(yd_all[-1][~v_has_cyclone])

        xt_cyc.append(xt_all[-1][v_has_cyclone])
        yt_cyc.append(yt_all[-1][v_has_cyclone])
        xt_noc.append(xt_all[-1][~v_has_cyclone])
        yt_noc.append(yt_all[-1][~v_has_cyclone])

        n_thm.append(np.sum(vriles[r][f"vriles{vrds}_class"] > 0))
        n_cyc.append(np.sum(v_has_cyclone))

        # Add statistics to the relevant rows:
        rows_all.append([cfg.reg_labels_short[r], n_tot[-1],
            fmt_corrcoef(  xd_all[-1], yd_all[-1]),
            fmt_regression(xd_all[-1], yd_all[-1]),
            fmt_corrcoef(  xt_all[-1], yt_all[-1]),
            fmt_regression(xt_all[-1], yt_all[-1])])

        rows_cls.append([cfg.reg_labels_short[r], n_tot[-1] - n_thm[-1],
            fmt_corrcoef(  xd_dyn[-1], yd_dyn[-1]),
            fmt_regression(xd_dyn[-1], yd_dyn[-1]),
            fmt_corrcoef(  xt_dyn[-1], yt_dyn[-1]),
            fmt_regression(xt_dyn[-1], yt_dyn[-1]),
            n_thm[-1],
            fmt_corrcoef(  xd_thm[-1], yd_thm[-1]),
            fmt_regression(xd_thm[-1], yd_thm[-1]),
            fmt_corrcoef(  xt_thm[-1], yt_thm[-1]),
            fmt_regression(xt_thm[-1], yt_thm[-1])])

        rows_cyc.append([cfg.reg_labels_short[r], n_cyc[-1],
            fmt_corrcoef(  xd_cyc[-1], yd_cyc[-1]),
            fmt_regression(xd_cyc[-1], yd_cyc[-1]),
            fmt_corrcoef(  xt_cyc[-1], yt_cyc[-1]),
            fmt_regression(xt_cyc[-1], yt_cyc[-1]),
            n_tot[-1] - n_cyc[-1],
            fmt_corrcoef(  xd_noc[-1], yd_noc[-1]),
            fmt_regression(xd_noc[-1], yd_noc[-1]),
            fmt_corrcoef(  xt_noc[-1], yt_noc[-1]),
            fmt_regression(xt_noc[-1], yt_noc[-1])])

    # Now do the combined row (excluding index 0 which is pan Arctic VRILEs):
    rows_all.append(["ALL", np.sum(n_tot[1:]),
        fmt_corrcoef(  np.concatenate(xd_all[1:]), np.concatenate(yd_all[1:])),
        fmt_regression(np.concatenate(xd_all[1:]), np.concatenate(yd_all[1:])),
        fmt_corrcoef(  np.concatenate(xt_all[1:]), np.concatenate(yt_all[1:])),
        fmt_regression(np.concatenate(xt_all[1:]), np.concatenate(yt_all[1:]))])

    rows_cls.append(["ALL", np.sum(n_tot[1:]) - np.sum(n_thm[1:]),
        fmt_corrcoef(  np.concatenate(xd_dyn[1:]), np.concatenate(yd_dyn[1:])),
        fmt_regression(np.concatenate(xd_dyn[1:]), np.concatenate(yd_dyn[1:])),
        fmt_corrcoef(  np.concatenate(xt_dyn[1:]), np.concatenate(yt_dyn[1:])),
        fmt_regression(np.concatenate(xt_dyn[1:]), np.concatenate(yt_dyn[1:])),
        np.sum(n_thm[1:]),
        fmt_corrcoef(  np.concatenate(xd_thm[1:]), np.concatenate(yd_thm[1:])),
        fmt_regression(np.concatenate(xd_thm[1:]), np.concatenate(yd_thm[1:])),
        fmt_corrcoef(  np.concatenate(xt_thm[1:]), np.concatenate(yt_thm[1:])),
        fmt_regression(np.concatenate(xt_thm[1:]), np.concatenate(yt_thm[1:]))])

    rows_cyc.append(["ALL", np.sum(n_cyc[1:]),
        fmt_corrcoef(  np.concatenate(xd_cyc[1:]), np.concatenate(yd_cyc[1:])),
        fmt_regression(np.concatenate(xd_cyc[1:]), np.concatenate(yd_cyc[1:])),
        fmt_corrcoef(  np.concatenate(xt_cyc[1:]), np.concatenate(yt_cyc[1:])),
        fmt_regression(np.concatenate(xt_cyc[1:]), np.concatenate(yt_cyc[1:])),
        np.sum(n_tot[1:]) - np.sum(n_cyc[1:]),
        fmt_corrcoef(  np.concatenate(xd_noc[1:]), np.concatenate(yd_noc[1:])),
        fmt_regression(np.concatenate(xd_noc[1:]), np.concatenate(yd_noc[1:])),
        fmt_corrcoef(  np.concatenate(xt_noc[1:]), np.concatenate(yt_noc[1:])),
        fmt_regression(np.concatenate(xt_noc[1:]), np.concatenate(yt_noc[1:]))])

    txt += "Below, 'DYN'/'THM' refer to the dynamic/thermodynamic x,y "
    txt += "quantities above,\n"
    txt += f"e.g., r(DYN) is the correlation between '{xy_diag[0]}' and "
    txt += f"'{xy_diag[1]}'\n\n"
    txt += "All VRILEs (number in each sector/region is N):\n\n"
    txt += tabulate(rows_all, headers=headers_all)
    txt += "\n\n\nSeparated by classification (C <= 0 are dynamic):\n\n"
    txt += tabulate(rows_cls, headers=headers_cls)
    txt += "\n\n\nSeparated by cyclone vs. non-cyclone (T = number of tracks):\n\n"
    txt += tabulate(rows_cyc, headers=headers_cyc)
    txt += "\n\n"

    print(txt)
    cache.write_txt(txt, txt_fname, directory=cfg.data_path["tables"])


    # Ad-hoc rcParams changes for these plots:
    mpl.rcParams["axes.autolimit_mode"] = "data"

    # Scatter keyword (kw) arguments per category:
    # --- not a case study, no cyclone:
    s_kw    = {"s": 12, "marker": "o", "clip_on": False}
    # --- not a case study, cyclone:
    s_kw_cy = {"edgecolors": "k", "linewidths": 1.5*mpl.rcParams["lines.linewidth"], **s_kw}
    # --- case study, no cyclone:
    s_kw_cs = {"s": 50, "marker": "*", "zorder": 100000, "clip_on": False}
    # --- case study, cyclone:
    s_kw_cs_cy = {"edgecolors": s_kw_cy["edgecolors"],
                  "linewidths": s_kw_cy["linewidths"], **s_kw_cs}

    # Properties for the x = 0 and y = 0 gridlines:
    zero_line_kw = {"linewidth":mpl.rcParams["grid.linewidth"],
                    "linestyle":"-", "color": "lightgrey", "zorder": -10}

    # Commont arguments to pass to fig layout template function:
    fig_2x4_kw = {"ax_l": .08, "ax_r": .02, "ax_b": .25, "ax_s_hor": .065}

    fig1, axs1 = style_ms.fig_layout_2x4(xlabel=xy_labels[0], ylabel=xy_labels[1], **fig_2x4_kw)
    fig2, axs2 = style_ms.fig_layout_2x4(xlabel=xy_labels[2], ylabel=xy_labels[3], **fig_2x4_kw)

    # Create scalar mappable matching color bar (added in fig_layout_2x4) to
    # create colors for scatter points plotted below:
    cm_sm = plt.cm.ScalarMappable(cmap=style_ms.vrile_class_cmap,
                                  norm=style_ms.vrile_class_norm)

    # Plot data:
    for r in range(n_regions):
        for v in range(vriles[r][f"n{vrds}_vriles"]):

            class_rv = vriles[r][f"vriles{vrds}_class"][v]
            color_rv = cm_sm.to_rgba(class_rv)

            # Determine which style => which keyword arguments for scatter:
            if is_case_study(r, v, case_studies):
                if len(vrile_track_ids[r][v]) > 0:  # cyclone associated
                    skw = s_kw_cs_cy
                else:
                    skw = s_kw_cs
            else:
                if len(vrile_track_ids[r][v]) > 0:
                    skw = s_kw_cy
                else:
                    skw = s_kw

            # We might not be plotting this point if command-line options
            # specify to plot thermo. or dynamic only, so check first:
            if (   (not cmd.thm_only and not cmd.dyn_only)
                or (cmd.thm_only and class_rv > 0.)
                or (cmd.dyn_only and class_rv < 0.)):

                for dj, axs in zip([0,2], [axs1, axs2]):
                    axs[r//4,r%4].scatter(xy_data[dj][r][v], xy_data[1+dj][r][v],
                                          facecolors=color_rv, **skw)

    # Format the axes titles and grid lines. For the titles, use the region
    # long labels except for East Siberian--Chukchi Sea (use the short label):
    axs_titles    = cfg.reg_labels_long
    axs_titles[6] = cfg.reg_labels_short[6]

    for axs in [axs1, axs2]:

        tools.add_subplot_panel_titles(axs, axs_titles)

        for ax in axs.flatten():

            ax.set_facecolor("none")

            # Fix the axes limits so we can determine whether to draw the
            # zero gridlines (if one just plots them anyway, the axes limits
            # will automatically change to compensate):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if min(xlim) < 0. and max(xlim) > 0.:
                ax.axvline(0, **zero_line_kw)
            if min(ylim) < 0. and max(ylim) > 0.:
                ax.axhline(0, **zero_line_kw)

    for j, fig, txt in zip([0,1], [fig1, fig2], ["Dynamic", "Thermodynamic"]):
        
        # Add a legend for the different types of scatter points, to both figures:
        # Use some axes (which will be set invisible later) to help position it,
        # at the following (ad-hoc) bbox coordinates:
        ax_leg_l = axs[1,2].get_position().x0
        ax_leg_b = .02
        ax_leg_w = axs[1,3].get_position().x1 - ax_leg_l
        ax_leg_h = axs[1,2].get_position().y0 - .14

        ax_leg = fig.add_axes([ax_leg_l, ax_leg_b, ax_leg_w, ax_leg_h])

        ax_leg.set_facecolor("none")
        ax_leg.tick_params(which="both", axis="both", left=False, right=False,
                           top=False, bottom=False, labelleft=False,
                           labelright=False, labeltop=False, labelbottom=False)

        ax_leg.set_xlim(0,1)
        ax_leg.set_ylim(0,1)

        for spine in ax_leg.spines:
            ax_leg.spines[spine].set_visible(False)

        txt_kw = {"ha": "left", "va": "center",
                  "fontsize": mpl.rcParams["axes.labelsize"]-1}

        # Scatter points are arranged in 2x2; set the positions on ax_leg:
        xleg_l = .32  # left
        xleg_r = .67  # right
        yleg_t = .87  # top
        yleg_b = .29  # bottom
        xpad   = .04  # padding between scatter point and its text label

        ax_leg.annotate("Has cyclone", (xleg_l + xpad, yleg_t), **txt_kw)
        ax_leg.annotate("No cyclone" , (xleg_r + xpad, yleg_t), **txt_kw)
        ax_leg.annotate("Case study\n(has cyclone)", (xleg_l + xpad, yleg_b), **txt_kw)
        ax_leg.annotate("Case study\n(no cyclone)" , (xleg_r + xpad, yleg_b), **txt_kw)

        ax_leg.scatter([xleg_l], [yleg_t], facecolors="tab:grey", **s_kw_cy   )
        ax_leg.scatter([xleg_r], [yleg_t], facecolors="tab:grey", **s_kw      )
        ax_leg.scatter([xleg_l], [yleg_b], facecolors="tab:grey", **s_kw_cs_cy)
        ax_leg.scatter([xleg_r], [yleg_b], facecolors="tab:grey", **s_kw_cs   )

        # Set figure metadata Title and save (if flag --savefigs present)
        # Supress automatic plt.show() (no_show=True) because we are currently
        # modifying both figures in a loop, and we need both figures completed:
        if cmd.savefig_titles[j] is None:
            savefig_titles = f"{txt} VRILE drivers"
        else:
            savefig_titles = cmd.savefig_titles[j]

        tools.finish_fig(fig, savefig=cmd.savefigs, no_show=True,
                         file_name=cmd.savefig_names[j],
                         fig_metadata={"Title": savefig_titles})

    if not cmd.savefigs:
        plt.show()


if __name__ == "__main__":
    main()

