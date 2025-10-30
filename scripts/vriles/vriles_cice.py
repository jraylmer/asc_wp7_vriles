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
            norm_time_hist=False, norm_time_proc=False,
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
    # correspond to this list. Specifically, we want to add the total ice
    # concentration and volume tendencies and calculate the surface energy
    # balance (SEB), which are not history outputs but can be computed from
    # the dynamic/thermodynamic components d{a,v}idt{t,d}_d, and for SEB the
    # 'fsurf_ai' and 'fcondtop_ai' (SEB = former minus the latter).
    #
    # So load 'aice' and the required diagnostics. We don't really need 'aice',
    # but need to be careful because that function requires 'aice' to do the
    # calculations so loads it by prepending it to the list. So, when we use
    # the 'hist_function' option (below) to implicitly calculate SEB 'on-the-
    # fly', it is less confusing to just add it here manually. Also, the list
    # of diagnostics must not be changed in length, hence the duplication of
    # daidtt_d and dvidtt_d below acting as placeholders for daidt_d and
    # dvidt_d (total tendencies):
    #
    hist_diags = ["aice_d", "daidtd_d", "daidtt_d", "daidtt_d", "dvidtd_d",
                  "dvidtt_d", "dvidtt_d", "meltb_d", "meltl_d", "meltt_d",
                  "fsurf_ai_d", "fcondtop_ai_d"]

    # We pass 'hist_diags' to the diagnostic function for processing, which
    # then ends up updating the actual diagnostics after using the
    # hist_diag_converter() function defined below. We will want the names of
    # the latter to cache the data:

    hist_diags_updated = ["aice_d"  , "daidtd_d", "daidtt_d"  , "daidt_d",
                                      "dvidtd_d", "dvidtt_d"  , "dvidt_d",
                                      "meltb_d" , "meltl_d"   , "meltt_d",
                                      "seb_ai_d", "fsurf_ai_d"]

    def hist_diag_converter(x):
        """This will be used to convert the list of data corresponding to
        hist_diags into the list corresponding to hist_diags_updated inside
        the diagnostics function compute_averages_over_vriles().
        """
        return [x[0], x[1], x[2], x[1] + x[2],  # aice, daidtd, daidtt, daidt
                      x[4], x[5], x[4] + x[5],  #       dvidtd, dvidtt, dvidt
                      x[7], x[8], x[9]       ,  #       meltb , meltl , meltt
                      x[10] - x[11], x[11]]     #       seb_ai, fsurf_ai


    # We also want some atmospheric forcing variables, which are handled by the
    # same diagnostics function but with separate inputs:
    atmo_diags = ["t2_d", "qlw_d", "qsw_d"]

    avg_hist, _, avg_atmo = vrile_diagnostics.compute_averages_over_vriles(
        vrile_results, region_masks, hist_function=hist_diag_converter,
        hist_diags=hist_diags, norm_time_hist=True,
        norm_area_hist=[True] + [False]*6 + [True]*3 + [True]*2,
        norm_unit_hist=[1.  ] + [1.e11]*6 + [1.  ]*3 + [1.  ]*2,
        atmo_fields=atmo_diags, norm_time_atmo=True,
        norm_area_atmo=True, norm_unit_atmo=1., **caov_kw)

    # Clarification of units in the above:
    #
    # daidt are in %/day = 100*fraction/day
    #     Divide by 100 to get values in fraction/day
    #     Then integrating (norm_area_hist = False) => m^2/day
    #     Then put in 10^3 km^2/day => divide by 10^9
    #     Overall, divide by 10^11 => final units 10^3 km^2/day
    #                                             =============
    #
    # dvidt are in cm/day
    #     Divide by 100 to get values in m/day
    #     Then integrating => m^3/day
    #     Then put in km^3/day => divide by 10^9
    #     Overall, divide by 10^11 => final units km^3/day
    #                                             ========
    #
    # melt rates (melt*_d) are in cm/day
    #     We just calculate mean melt rates, both time/area normalised
    #     => no change in units
    #
    # For seb_ai and fsurf_ai, and atmospheric fields, both time/area
    # normalised => no change in units

    # Common first part of cached file names:
    save_file_start = f"vriles_cice_{dt_min.year}-{dt_max.year}_diagnostics"

    # Save the list of diagnostics (avg_hist + avg_atmo) prepended with the
    # list of diagnostic names, as a 'header':

    cache.save([hist_diags_updated + atmo_diags] + avg_hist + avg_atmo,
               f"{save_file_start}_hist_atmo.pkl")


    # Calculate other 'diagnostics over VRILEs' from 'processed' data
    #
    # (1) The div_curl diagnostics [subdirectory of cfg.data_path['proc_d'] is
    # referred to as 'metric' in the call to the compute_averages_over_vriles()
    # function... poor earlier choice of terminology]
    #
    _, avg_proc, _ = vrile_diagnostics.compute_averages_over_vriles(
        vrile_results, region_masks, proc_metric="div_curl",
        proc_diags=["div_strair_d", "curl_strair_d"],
        norm_time_proc=True, norm_area_proc=True, norm_unit_proc=1.e-7,
        **caov_kw)  # units become 10^-7 N m-3

    # Save the list of diagnostics (avg_proc) prepended with the list of
    # diagnostic names as a 'header':
    cache.save([["div_strair_d", "curl_strair_d"]] + avg_proc,
                f"{save_file_start}_div_curl.pkl")

    # (2) Detrended diagnostics from CICE. Here, also include detrended
    # atmospheric forcing fields (they are loaded/processed via the 'atmo'
    # parameters, because those fields are on the extended atmosphere grid):
    #
    proc_diags  = [f"detrended_{x}_d"
                   for x in ["daidtd", "daidtt", "daidt",
                             "dvidtd", "dvidtt", "dvidt",
                             "meltb" , "meltl" , "meltt",
                             "seb_ai", "div_strair", "curl_strair"]]

    atmo_fields = [f"detrended_{x}_d" for x in ["t2", "qlw", "qsw", "qnet"]]

    _, avg_proc, avg_atmo = vrile_diagnostics.compute_averages_over_vriles(
        vrile_results, region_masks, proc_metric="hist_detrended",
        proc_diags=proc_diags, norm_time_proc=True,
        norm_area_proc=[False]*6 + [True]*6,
        norm_unit_proc=[1.e11]*6 + [1., 1., 1., 1., 1.e-7, 1.e-7],
        atmo_fields=atmo_fields, atmo_file_prefix="atmo_detrended",
        norm_time_atmo=True, norm_area_atmo=True, norm_unit_atmo=1.)

    # Note units: same as non-detrended versions above:

    cache.save([proc_diags + atmo_fields] + avg_proc + avg_atmo,
               f"{save_file_start}_hist_atmo_detrended.pkl")


if __name__ == "__main__":
    main()

