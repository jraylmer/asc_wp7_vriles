"""Run VRILE identification and classification code on SSM/I sea ice extent
data that has been calculated on the CICE grid, and cache the 'results
dictionaries' (python dictionary for each analysis region containing the dates,
magnitudes and other metadata for the VRILEs). Unlike for CICE, there is an
additional moving average filter applied to the data before the VRILE
identification code is run, set with option --ssmi-filter-n-days.

"""

from datetime import datetime as dt
from pathlib import Path

import numpy as np

from src import script_tools
from src.io import cache, config as cfg, sumtabtxt
from src.data import ssmi
from src.diagnostics import vriles, vrile_diagnostics


def moving_average(x, n=5):
    """Calculate n-point moving average of x."""
    x_ma = np.zeros(len(x) - n + 1)
    for i in range(len(x_ma)):
        x_ma[i] = np.nanmean(x[i:i+n])
    return x_ma


def main():

    prsr = script_tools.argument_parser(usage="Calculate VRILEs in SSM/I")
    script_tools.add_vrile_cmd_args(prsr, ssmi=True)
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    id_vriles_kw, join_vriles_kw, dt_min, dt_max = \
        script_tools.get_id_vriles_options(cmd, footer=True)

    print("Loading data")

    # Careful: data prior to 21 August 1987 is every two days, and 1 Jan 1979
    # is a missing day. Therefore, load 31 Dec 1978 for which the interpolation
    # function will use it and 2 Jan 1979 to estimate 1 Jan 1979.
    #
    # There is a similar issue for year 1982, 1984, 1985, and 1987 if those are
    # specified as the start of the the year range. Below, load one day before
    # dt_min (and then restrict range after pre-processing) should ensure this
    # always fixed automatically.

    date, _, sie = ssmi.load_data("sea_ice_extent", frequency="daily",
        which_dataset=cmd.ssmi_dataset, regions=cfg.reg_nc_names,
        dt_range=(dt(dt_min.year-1, 12, 31), dt_max))

    # Interpolate the missing days:
    sie = ssmi.fill_missing_data(sie)

    # Now select the actual required datetime range:
    jt = np.array([x >= dt_min for x in date])
    date = date[jt]
    sie = [sie_array[jt] for sie_array in sie]

    # Compute cmd.obs_filter_n_days day moving average filter:
    if cmd.ssmi_filter_n_days > 1:
        print(f"Applying {cmd.ssmi_filter_n_days}-day moving average filter")
        for k in range(len(sie)):
            sie[k] = moving_average(sie[k], n=cmd.ssmi_filter_n_days)
        # Slice date array to the same coordinates:
        date = date[(cmd.ssmi_filter_n_days//2):-((cmd.ssmi_filter_n_days-1)//2)]
 
    # For metadata in VRILE results dictionaries and summary tables:
    ssmi_title = "NASA Team" if cmd.ssmi_dataset == "nt" else "Bootstrap"

    titles = [f"{k} VRILEs (SSM/I {ssmi_title})" for k in cfg.reg_labels_long]

    print(f"Identifying VRILEs in {len(sie)} regions")
    vriles_ssmi = [vriles.identify(date, sie[k], n_ma=cmd.n_moving_average,
                                   id_vriles_kw=id_vriles_kw,
                                   join_vriles_kw=join_vriles_kw,
                                   data_title=titles[k])
                   for k in range(len(sie))]

    # Save the summary tables (save the actual data at the end, as below
    # we first add a key for the matches to corresponding CICE VRILEs):
    additional_metadata = {"Time range" : f"{dt_min.strftime('%d %b %Y')} to "
                                          + f"{dt_max.strftime('%d %b %Y')}",
                           "Description": cfg.title}

    sumtabtxt.save_tables(vriles_ssmi, id_vriles_kw,
        which=[True]*4, vresults_labels=cfg.reg_labels_short,
        additional_metadata=additional_metadata, verbose=True,
        save_dir=Path(cfg.data_path["tables"],
            f"vriles_ssmi-{cmd.ssmi_dataset}_{dt_min.year}-{dt_max.year}"))

    # Save the 'intersection' of CICE and these obs. VRILEs.
    # First load CICE VRILEs from cache:
    vriles_cice = cache.load(f"vriles_cice_{dt_min.year}-{dt_max.year}.pkl")

    # Save text files to this directory:
    dir_matches = Path(cfg.data_path["tables"],
        f"vrile_matches_cice_to_ssmi-{cmd.ssmi_dataset}_{dt_min.year}-{dt_max.year}")

    dir_matches.mkdir(exist_ok=True, parents=True)

    for k in range(len(sie)):

        # Function below updates metadata of vrile results dictionaries
        # with indices of joined VRILEs matching one in the other set, and
        # saves a text table summarising this information.
        #
        filename_k = Path(dir_matches, f"{cfg.reg_labels_short[k]}.txt")

        vriles_cice[k], vriles_ssmi[k] = sumtabtxt.save_vrile_set_intersection(
            vriles_cice[k], vriles_ssmi[k], filename_k, id_vriles_kw,
            v1_label="cice", v2_label=f"ssmi-{cmd.ssmi_dataset}",
            v1_title="CICE", v2_title=f"SSM/I {ssmi_title}",
            region_name=cfg.reg_labels_long[k],
            additional_metadata=additional_metadata, sort_by_rank=True)

    # Save the results dictionaries:
    cache.save(vriles_ssmi,
        f"vriles_ssmi-{cmd.ssmi_dataset}_{dt_min.year}-{dt_max.year}.pkl")

    # Overwrite cached data for CICE VRILEs:
    cache.save(vriles_cice, f"vriles_cice_{dt_min.year}-{dt_max.year}.pkl")


if __name__ == "__main__":
    main()

