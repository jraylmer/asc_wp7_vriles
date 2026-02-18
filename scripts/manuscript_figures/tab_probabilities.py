"""Calculate the marginal and conditional probabilities of cyclone track
presence in each sector and tabulate the results, including the implied
ratio of conditional to marginal probability of a VRILE occuring in each
sector using Bayes' theorem. This is used for Table 1 in the manuscript.
"""

from datetime import datetime as dt, timedelta
from pathlib import Path

import numpy as np
from tabulate import tabulate

from src import script_tools
from src.io import cache, config as cfg, sumtabtxt
from src.data import tracks
from src.diagnostics import track_diagnostics


def main():

    prsr = script_tools.argument_parser(
        usage="Calculate track/VRILE occurrence probabilities")

    script_tools.add_vrile_cmd_args(prsr, ssmi=True)
    script_tools.add_track_filter_cmd_args(prsr)
    script_tools.add_track_vrile_matching_cmd_args(prsr)

    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    id_vriles_kw, join_vriles_kw, dt_min, dt_max = \
        script_tools.get_id_vriles_options(cmd, footer=False)

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=False)

    script_tools.get_track_vrile_matching_options(cmd, header=False,
                                                  footer=True)

    allowed_sectors = tracks.allowed_sectors(n_nei=cmd.track_n_sector_neighbours)

    yrng = f"{dt_min.year:04}-{dt_max.year:04}"

    # Load cached VRILE (simulation and SSM/I) and filtered track data
    # (need to run relevant scripts first):
    vriles_cice_fname = f"vriles_cice_{yrng}.pkl"
    vriles_ssmi_fname = f"vriles_ssmi-{cmd.ssmi_dataset}_{yrng}.pkl"
    tracks_fname      = f"tracks_filtered_{yrng}.pkl"

    vriles_cice_data = cache.load(vriles_cice_fname)
    vriles_ssmi_data = cache.load(vriles_ssmi_fname)
    track_data       = cache.load(tracks_fname)

    n_sectors  = len(vriles_cice_data)

    # Extract required track data (IDs, datetimes and sector for each coordinate)
    # Assumes headers set as in the script generating the filtered track data:
    track_ids = track_data[1+track_data[0].index("TRACK_IDS")]
    track_dts = track_data[1+track_data[0].index("DATETIMES")]
    track_sec = track_data[1+track_data[0].index("SECTOR_FLAG")]

    # First, calculate the marginal probability of a cyclone being present in
    # each sector, P(C), as the fraction of the whole study period where there
    # is a cyclone track present in the sector or its nearest neighbours (via
    # the 'allowed_sectors' indices).
    #
    # Construct a daily datetime array for the months of May-Sep for all years:
    n_days_per_year = 31 + 30 + 31 + 31 + 30
    dt_day1         = dt(dt_min.year, 5, 1)
    dts_one_year    = np.array([dt_day1 + timedelta(days=j)
                                for j in range(n_days_per_year)])

    date_check = np.concatenate([[dt(y, x.month, x.day) for x in dts_one_year]
                                 for y in range(dt_min.year, dt_max.year+1, 1)])

    prob_c, _ = track_diagnostics.get_track_frequency(
        date_check, track_dts, track_sec, allowed_sectors=allowed_sectors,
        verbose=True)

    # We want all sectors except CAN, and all sectors at the end.
    # Initially, index 0 is pan-Arctic but that is fine for cyclones in any
    # sector, and CAN is the last index.
    #
    prob_c = np.concatenate((prob_c[1:-1], prob_c[[0]]))

    # We also want probabilities for all sectors except Labrador (at the
    # very end). For P(C), calculate this next (note: in the default case of
    # assuming nearest neighbouring sectors are allowed for each sector,
    # this will be the same as P(C) for the whole Arctic):
    allowed_sectors_nolab = [
             [1,2,3,4,5,6,7,8] if cmd.track_n_sector_neighbours==1
        else [2,3,4,5,6,7]] + [[] for j in range(8)]

    prob_c_nolab = track_diagnostics.get_track_frequency(
        date_check, track_dts, track_sec,
        allowed_sectors=allowed_sectors_nolab, verbose=True)[0][0]

    prob_c = np.concatenate((prob_c, [prob_c_nolab]))


    # Now calculate the conditional probability of a cyclone being present
    # given that a VRILE is occurring, in each sector, P(C|V). Use the same
    # approach as above, but instead of using the full date_check array,
    # create a subsetted array of datetimes corresponding to when VRILEs are
    # occurring in each sector. This is done for both CICE and SSMI VRILEs.
    #
    prob_cv_cice = np.zeros(len(prob_c))
    prob_cv_ssmi = np.zeros(len(prob_c))

    for prob_cv, vrile_data in zip([prob_cv_cice    , prob_cv_ssmi    ],
                                   [vriles_cice_data, vriles_ssmi_data]):

        # Prepare datetime arrays for all sectors combined
        # and that without Labrador:
        dates_all       = []
        dates_all_nolab = []

        for r in range(1,len(vriles_cice_data)):  # skip pan-Arctic (index 0)

            # Determine the set of dates on which VRILEs occur for this
            # region/sector. Brute-force approach -- append dates to a
            # list 'dates_r' while looping over VRILE start/end date bounds:
            dates_r = []
            for v in range(vrile_data[r]["n_joined_vriles"]):
                dt_vj = vrile_data[r]["date_bnds_vriles_joined"][v][0]
                dt_v1 = vrile_data[r]["date_bnds_vriles_joined"][v][1]

                # Append all days between start and end bounds:
                while dt_vj <= dt_v1:
                    dates_r.append(dt_vj)
                    dt_vj = dt_vj + timedelta(days=1)

            dates_r = np.array(sorted(dates_r))

            dates_all.append(dates_r)

            if r != 1:
                dates_all_nolab.append(dates_r)

            # Run track frequency function with dates_r as the reference period:
            tfreq_sec, _ = track_diagnostics.get_track_frequency(
                dates_r, track_dts, track_sec, allowed_sectors=allowed_sectors,
                verbose=True)

            # That function is currently checking all sectors.
            # Here, we only need the result for sector r (index r-1 in prob_cv,
            # because we skip pan Arctic in current loop but individual sectors
            # are first :
            prob_cv[r-1] = tfreq_sec[r]

        # Calculate and append P(C|V) for all sectors combined and for that
        # without Labrador sector:
        for dates_array, index in zip([dates_all, dates_all_nolab], [-2, -1]):
            dates_arr = np.array(sorted(list(set(list(np.concatenate(dates_array))))))

            tfreq_sec, _ = track_diagnostics.get_track_frequency(
                dates_arr, track_dts, track_sec, allowed_sectors=allowed_sectors)

            prob_cv[index] = tfreq_sec[0]


    # Invert the conditional probability using Bayes' theorem; we just want
    # the increase in probability of VRILE associated with cyclones:
    #
    #     P(V|C) / P(V) = P(C|V) / P(C) = prov_cv / prob_c
    #
    rel_prob_cice = prob_cv_cice / prob_c
    rel_prob_ssmi = prob_cv_ssmi / prob_c


    # Tabulate the results:
    headers = ["Sector\n", "P(C)\n", "\nN_V", "CICE\nP(C|V)", "\nP(V|C)/P(V)",
                                     "\nN_V", "SSMI\nP(C|V)", "\nP(V|C)/P(V)"]

    floatfmt = ("", "", "", "", ".3f", "", "", ".3f")

    sector_labels =  cfg.reg_labels_long[1:8]
    sector_labels += ["All sectors", "All sectors except Labrador"]

    n_vriles_cice  = [vriles_cice_data[r]["n_joined_vriles"] for r in range(1,8)]
    n_vriles_cice += [sum(n_vriles_cice), sum(n_vriles_cice[1:8])]
    n_vriles_ssmi  = [vriles_ssmi_data[r]["n_joined_vriles"] for r in range(1,8)]
    n_vriles_ssmi += [sum(n_vriles_ssmi), sum(n_vriles_ssmi[1:8])]

    rows = []

    for j in range(len(sector_labels)):
        rows.append([sector_labels[j], f"{prob_c[j]:.3%}",
                     n_vriles_cice[j], f"{prob_cv_cice[j]:.3%}",
                     rel_prob_cice[j],
                     n_vriles_ssmi[j], f"{prob_cv_ssmi[j]:.3%}",
                     rel_prob_ssmi[j]])

    table = "\n" + tabulate(rows, headers=headers, floatfmt=floatfmt) + "\n"
    print(table)

    # Extra text description for file output:
    table_info = (  "VRILE/Cyclone probabilities\n"
                  + "---------------------------\n"
                  + "N_V         = number of joined VRILEs\n"
                  + "P(C)        = fraction of time cyclone(s) in vicinity to influence "
                  + "sea ice in sector\n"
                  + "P(C|V)      = as above but as fraction of time periods that VRILEs "
                  + "are occuring in sector\n"
                  + "P(V|C)/P(V) = probability of VRILE occuring given cyclone relative "
                  + "to marginal probability of VRILE\n"
                  + "\n"
                  + "Note N_V is irrelevant to these probabilities; they are just included "
                  + "as a sanity check that the\ncode is selecting the right sectors.\n\n"
                  + f"SSM/I dataset: {'NASA Team' if cmd.ssmi_dataset=='nt' else 'Bootstrap'}\n\n")

    cache.write_txt(table_info + table,
                    f"vriles_cice_ssmi-{cmd.ssmi_dataset}_cyclone_probabilities.txt",
                    directory=cfg.data_path["tables"])


if __name__ == "__main__":
    main()

