"""Run VRILE identification and classification code on CICE sea ice extent
and cache the 'results dictionaries' (python dictionary for each analysis
region containing the dates, magnitudes and other metadata for the VRILEs).

This script also calculates various 'diagnostics over VRILEs' -- various
history and 'processed' diagnostics averaged over the area of sea ice extent
loss associated with each VRILE -- and caches the results.
"""

from pathlib import Path

import numpy as np

from src import script_tools
from src.io import cache, config as cfg, sumtabtxt
from src.data import cice
from src.diagnostics import vriles, vrile_diagnostics


def main():

    prsr = script_tools.argument_parser(usage="Calculate VRILEs in CICE")
    script_tools.add_vrile_cmd_args(prsr, ssmi=False)
    script_tools.add_vrile_classification_cmd_args(prsr)
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    id_vriles_kw, join_vriles_kw, dt_min, dt_max = \
        script_tools.get_id_vriles_options(cmd, footer=True)

    print("Loading data")
    date, sie = cice.get_processed_data_regional(cmd.metric, "sie", dt_min=dt_min,
        dt_max=dt_max, region_nc_names=cfg.reg_nc_names)

    # For metadata in VRILE results dictionaries and summary tables:
    titles = [f"{k} VRILEs (CICE)" for k in cfg.reg_labels_long]

    n_regions = len(sie)

    print(f"Identifying VRILEs in {n_regions} regions")
    vrile_results = [vriles.identify(date, sie[k], n_ma=cmd.n_moving_average,
                                     id_vriles_kw=id_vriles_kw,
                                     join_vriles_kw=join_vriles_kw,
                                     data_title=titles[k])
                     for k in range(n_regions)]

    region_masks = cice.get_region_masks(slice_to_atm_grid=True)

    # Common keyword arguments to be passed to module vrile_diagnostics
    # function calculate_averages_over_vriles():
    caov_kw = {"year_range"    : [dt_min.year, dt_max.year],
               "months_allowed": cmd.months_allowed,
               "verbose"       : True}

    # String to insert into history/processed data variable names:
    bud = "v" if cmd.class_metric == "volume" else "a"

    # For classification, need to integrate budget (volume or area) tendencies
    # due to thermodynamics and due to dynamics over VRILEs.
    #
    # If using the detrended method (default), these are 'processed'
    # diagnostics to be calculated in advance (script 'detrended_2d.py')
    #
    # Otherwise, just use the direct model history outputs
    #
    if cmd.class_no_detrend:
        hist_diags  = [f"d{bud}idtt", f"d{bud}idtd"]
        proc_metric = None
        proc_diags  = []
    else:  # default
        hist_diags  = []
        proc_metric = "hist_detrended"
        proc_diags = [f"detrended_d{bud}idtt_d", f"detrended_d{bud}idtd_d"]

    # Classify both 'unjoined' and 'joined' VRILEs:
    for x, k in zip(["", "joined "], ["vriles_class", "vriles_joined_class"]):

        print(f"Classifying {x}VRILEs")
        avg_hist, avg_proc, _ = vrile_diagnostics.compute_averages_over_vriles(
            vrile_results, region_masks, hist_diags=hist_diags,
            proc_diags=proc_diags, proc_metric=proc_metric,
            joined="joined" in x, **caov_kw)

        if cmd.class_no_detrend:
            dsi_thm = avg_hist[0]
            dsi_dyn = avg_hist[1]
        else:
            dsi_thm = avg_proc[0]
            dsi_dyn = avg_proc[1]

        # Classification index C = -1 if dynamic, +1 if thermodynamic
        for r in range(len(vrile_results)):
            Cr = vrile_diagnostics.classify(dsi_thm[r], dsi_dyn[r])
            vrile_results[r][k] = Cr.copy()  # add to results dictionary

    # Save the results dictionaries (now with classifications):
    cache.save(vrile_results, f"vriles_cice_{dt_min.year}-{dt_max.year}.pkl")

    # Save the summary tables:
    additional_metadata = {"Time range" : f"{dt_min.strftime('%d %b %Y')} to "
                                          + f"{dt_max.strftime('%d %b %Y')}",
                           "Description": cfg.title}

    sumtabtxt.save_tables(vrile_results, id_vriles_kw,
            which=[True]*4, vresults_labels=cfg.reg_labels_short,
            additional_metadata=additional_metadata, verbose=True,
            save_dir=Path(cfg.data_path["tables"],
                          f"vriles_cice_{dt_min.year}-{dt_max.year}"))


    # Calculate other 'diagnostics over VRILEs' from history data
    # 
    # List of history diagnostics. Here, it is implicitly modified in the
    # compute_averages_over_vriles() function so that what it returns does not
    # correspond to this list. Specifically, we want surface energy balance
    # (SEB), which is not a history output but can be computed from 'fsurf_ai'
    # and 'fcondtop_ai' (SEB = former minus the latter). So load 'aice' and
    # those diagnostics (don't really need 'aice', but need to be careful
    # because that function requires 'aice' to do the calculations so loads it
    # by prepending it to the list. So, when we use the 'hist_function' option
    # (below) to implicitly calculate SEB 'on-the-fly', it is less confusing to
    # just add it here manually:
    #
    hist_diags = ["aice_d", "fsurf_ai_d", "fcondtop_ai_d"]

    def calc_seb_ai(x):
        """Calculate surface energy balance, seb_ai = fsurf_ai - fcondtop_ai

        This implicitly converts the list of data [aice, fsurf_ai, fcondtop_ai]
        into [aice, seb_ai, fsurf_ai] in function compute_averages_over_vriles(). 
        """
        return [x[0], x[1] - x[2], x[1]]

    avg_hist, _, _ = vrile_diagnostics.compute_averages_over_vriles(
        vrile_results, region_masks, hist_function=calc_seb_ai,
        hist_diags=hist_diags, norm_time_hist=True, norm_area_hist=True,
        norm_unit_hist=1., **caov_kw)

    # Common first part of cached file names:
    save_file_start = f"vriles_cice_{dt_min.year}-{dt_max.year}_diagnostics"

    # Save the list of diagnostics (avg_hist) prepended with the list of
    # diagnostic names, as a 'header':
    cache.save([["aice_d", "seb_ai_d", "fsurf_ai_d"]] + avg_hist,
               f"{save_file_start}_hist.pkl")


    # Calculate other 'diagnostics over VRILEs' from 'processed' data
    #
    # List of 'metrics' (poor earlier choice of terminology -- essentially,
    # refers to the 'processed' data sub-directory -- the corresponding
    # diagnostics, normalisation options:
    #
    proc_metrics = ["hist_detrended", "div_curl"]

    proc_diags = [[f"detrended_{x}_d" for x in ["daidtt", "daidtd", "dvidtt", "dvidtd", "seb_ai"]],
                  ["div_strair_d", "curl_strair_d"]]

    # Options: norm_time_proc, norm_area_proc, norm_unit_proc:
    proc_norm = [[True, [False]*4 + [True], [1.0E11]*4 + [1.]],
                 [True, False             , 1.0E5            ]]

    for k in range(len(proc_metrics)):
        _, avg_proc, _ = vrile_diagnostics.compute_averages_over_vriles(
            vrile_results, region_masks, proc_metric=proc_metrics[k],
            proc_diags=proc_diags[k], norm_time_proc=proc_norm[k][0],
            norm_area_proc=proc_norm[k][1], norm_unit_proc=proc_norm[k][2],
            **caov_kw)

        # Save the list of diagnostics (avg_proc) prepended with the list of
        # diagnostic names as a 'header':
        cache.save([proc_diags[k]] + avg_proc,
                   f"{save_file_start}_{proc_metrics[k]}.pkl")


if __name__ == "__main__":
    main()

