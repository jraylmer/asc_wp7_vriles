"""This script determines the number of VRILEs that have been classified
'normally', in the sense that the classification metric R is well defined,
and the number that are edge cases of purely thermodynamically driven
because there are no dynamic changes (or vice-versa), and the number that
have undefined classification. Specifically this refers to the classification
metric C defined as:

         (  R     if T <  0 and D  < 0    'normal' classification
    C = <  +1     if T <  0 and D >= 0    purely thermodynamic
         ( -1     if T >= 0 and D  < 0    purely dynamic
         ( NaN    if T >= 0 and D >= 0    undefined

where R = (|T| - |D|) / (|T| + |D|), T is the change in sea ice volume due
to thermodynamic processes integrated across the VRILE time period and spatial
extent, and D is similarly defined for sea ice volume changes due to dynamics.
See module src/diagnostics/vrile_diagnostics.py, function classify_aylmer_2().
"""

import numpy as np

from src import script_tools
from src.io import cache, config as cfg


def main():

    prsr = script_tools.argument_parser(usage="Count VRILEs per classification")
    cmd  = prsr.parse_args()
    cfg.set_config(*cmd.config)

    vriles_cice = cache.load("vriles_cice_1979-2023.pkl")

    n_regions = len(vriles_cice)

    n_total = 0  # total number of VRILEs, any classification
    n_class = 0  # classification index works by default
    n_therm = 0  # exactly +1 (fully thermodynamic)
    n_dynam = 0  # exactly -1 (fully dynamic)
    n_undef = 0  # undefined classfication (= NaN)

    # Maintain a list of all unclassified VRILEs so they can be identified:
    v_unclass = []

    for r in range(1, n_regions):

        n_total += vriles_cice[r]["n_joined_vriles"]

        for v in range(vriles_cice[r]["n_joined_vriles"]):

            class_rv = vriles_cice[r][f"vriles_joined_class"][v]

            if np.isnan(class_rv):
                n_undef += 1
                v_unclass.append([r,v])
            elif class_rv == -1.:
                n_dynam += 1
            elif class_rv == 1.:
                n_therm += 1
            else:
                n_class += 1

    print(f"\nTotal number of VRILEs = {n_total:>3}")
    print(f"    of which 'normal'  = {n_class:>3} ({n_class/n_total:.1%})")
    print(f"    of which -1        = {n_dynam:>3} ({n_dynam/n_total:.1%})")
    print(f"    of which +1        = {n_therm:>3} ({n_therm/n_total:.1%})")
    print(f"    of which undefined = {n_undef:>3} ({n_undef/n_total:.1%})")

    if len(v_unclass) > 0:
        print("\nUnclassified VRILEs:\n--------------------")
        for k in range(len(v_unclass)):
            r = v_unclass[k][0]
            i = v_unclass[k][1]
            n = vriles_cice[r]["n_joined_vriles"]
            vrank = vriles_cice[r]["vriles_joined_rates_rank"][i]
            dt0, dt1 = vriles_cice[r]["date_bnds_vriles_joined"][i,:]

            print(f"{k+1}. Region: {r} ({cfg.reg_labels_short[r]}); "
                  + f"VRILE ID: {v}; rank: {vrank} of {n}; dates: "
                  + dt0.strftime("%d-%m-%Y to ") + dt1.strftime("%d-%m-%Y"))

        print("")

if __name__ == "__main__":
    main()
