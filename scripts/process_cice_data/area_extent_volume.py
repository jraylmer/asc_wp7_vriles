"""Calculate sea ice area, extent, and volume in different regions
of the CICE domain.
"""

from datetime import datetime as dt
from pathlib import Path

import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import cice, nc_tools
from src.diagnostics import diagnostics as diag


def main():

    prsr = script_tools.argument_parser(usage="Calculate SIA/SIE/SIV in CICE")
    prsr.add_argument("-y", "--year-range", type=int,nargs=2, default=[1979]*2)
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Use keys "sia", "sie", and "siv" for sea ice area, extent, and volume
    # respectively, to store data/metadata in dictionaries so that nc saving
    # function can be used in a loop (see end of script). Each diagnostic is
    # saved in a separate file, 1 per year, with all regional calculations
    # which includes pan Arctic
 
    # Attributes for time (variable name, and attributes including "units" and
    # "calendar" which are used to determine datetime values:
    nc_t_name = "time"
    nc_t_attr = {"bounds"       : f"{nc_t_name}_bnds",
                 "calendar"     : "365_day",
                 "long_name"    : "time",
                 "standard_name": "time",
                 "units"        : "days since 1979-01-01"}   

    # Attributes for all other variables (i.e., the diagnostics):
    nc_var_attr = {}
    nc_var_attr["sia"] = {"standard_name": "sea_ice_area",
                          "long_name"    : "Sea ice area",
                          "units"        : "1e6 km2"}
    nc_var_attr["sie"] = {"standard_name": "sea_ice_extent",
                          "long_name"    : "Sea ice extent",
                          "units"        : "1e6 km2"}
    nc_var_attr["siv"] = {"standard_name": "sea_ice_volume",
                          "long_name"    : "Sea ice volume",
                           "units"       : "1e3 km3"}

    # Shared attributes:
    for k in nc_var_attr.keys():
        nc_var_attr[k]["cell_methods"] = f"{nc_t_name}: mean"

    # Shared global attributes in all files;
    # "title" to be formatted with long_names above:
    nc_global_attr = {"title": "CICE diagnostics: {}"}
    for x in ["author", "contact", "institution"]:
        if getattr(cfg, x) != "":
            nc_global_attr[x] = getattr(cfg, x)

    if cfg.title != "":
        nc_global_attr["comment"] = cfg.title

    if cfg.cice_title != "":
        nc_global_attr["source"] = cfg.cice_title

    # Load grid data (grid cell areas "tarea" and region masks):
    tarea     = cice.get_grid_data(["tarea"])[0]  # array
    regions   = cice.get_region_masks()           # tuple of array
    n_regions = len(regions)

    # Sub-directories of cfg.data_path["proc_?"] for outputs
    # (nc_tools.save_netcdf() creates directories if they do not exist):
    sdir_name = {"sia": "sea_ice_area",
                 "sie": "sea_ice_extent",
                 "siv": "sea_ice_volume"}

    # File names for outputs, with a format for "d" (daily) or "m" (monthly)
    # and the year of the file:
    file_name = {"sia": "sia_{}_{:04}.nc",
                 "sie": "sie_{}_{:04}.nc",
                 "siv": "siv_{}_{:04}.nc"}

    # Two netCDF dimensions (time and bnd), same for all so prepare first:
    nc_dims = {}
    nc_dims[nc_t_name] = {"size": None}
    nc_dims["bnd"]  = {"size": 2}

    for y in range(cmd.year_range[0], cmd.year_range[1]+1, 1):

        # Create new datetimes and bounds (rather than using those in the CICE
        # history, for consistency); save time and time_bnds into dictionary
        # for daily and monthly:
        time = {}
        time_bnds = {}

        date_d, date_bnds_d, time["d"], time_bnds["d"] = \
            nc_tools.dt_daily(y, nc_units=nc_t_attr["units"],
                              nc_calendar=nc_t_attr["calendar"])

        date_m, date_bnds_m, time["m"], time_bnds["m"] = \
            nc_tools.dt_monthly(y, nc_units=nc_t_attr["units"],
                                nc_calendar=nc_t_attr["calendar"])

        # Load CICE daily sea ice concentration and volume data from history:
        # Note "hi_d" is sea ice volume per unit grid cell area:
        _, data_cice = cice.get_history_data(["aice_d", "hi_d"],
            dt_min=dt(y, 1, 1), dt_max=dt(y, 12, 31, 23, 0), frequency="daily",
            set_miss_to_nan=True, slice_to_atm_grid=False)

        # Save diagnostics for each region into this dictionary,
        # again for loop calling nc_tools.save_netcdf() below:
        data_out = {}
        for j in range(n_regions):
            r = cfg.reg_nc_names[j]
            data_out[f"sia_d_{r}"] = diag.sea_ice_area(data_cice[0], tarea,
                                                       mask=regions[j])
            data_out[f"sie_d_{r}"] = diag.sea_ice_extent(data_cice[0], tarea,
                                                         mask=regions[j])
            data_out[f"siv_d_{r}"] = diag.sea_ice_volume(data_cice[1],
                                                         data_cice[0], tarea,
                                                         mask=regions[j])

            # Calculate monthly means as well, from daily means:
            for x in nc_var_attr.keys():
                _, data_out[f"{x}_m_{r}"] = \
                    diag.monthly_mean(date_d, data_out[f"{x}_d_{r}"],
                                      date_bnds_m)

        for t in "dm":

            # Variable for time and time bounds is the same for all:
            nc_vars_t = {}
            nc_vars_t[nc_t_name] = {"data": time[t], "dims": (nc_t_name,),
                                    "attr": nc_t_attr}

            nc_vars_t[f"{nc_t_name}_bnds"] = {"data": time_bnds[t],
                                              "dims": (nc_t_name,"bnd")}

            for x in nc_var_attr.keys():

                # Update global attribute "title" for this diagnostic:
                nc_global_attr_x = {**nc_global_attr}
                nc_global_attr_x["title"] = nc_global_attr_x["title"].format(
                    nc_var_attr[x]["long_name"].lower())

                nc_vars_x = {}  # all netCDF variables (time + all regions)

                for r in cfg.reg_nc_names:
                    nc_vars_x[f"{x}_{t}_{r}"] = {"data": data_out[f"{x}_{t}_{r}"],
                                                 "dims": (nc_t_name,),
                                                 "attr": nc_var_attr[x]}

                # Finally, save the file:
                nc_tools.save_netcdf(file_name[x].format(t, y), nc_dims,
                                     {**nc_vars_t, **nc_vars_x},
                                     nc_global_attr=nc_global_attr_x,
                                     sort_attr=True,
                                     dir_save=Path(cfg.data_path[f"proc_{t}"],
                                                   sdir_name[x]))


if __name__ == "__main__":
    main()

