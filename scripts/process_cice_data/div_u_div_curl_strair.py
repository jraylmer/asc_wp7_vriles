"""Calculate 2D fields of sea ice velocity divergence and the wind-stress
divergence and curl from daily model outputs for sea ice drift (uvel_d, vvel_d)
and winds stress components (strairx_d and strairy_d). Also saves monthly
averages from the daily fields.
"""

from datetime import datetime as dt
from pathlib import Path

import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import cice, nc_tools
from src.diagnostics import diagnostics as diag


def main():

    prsr = script_tools.argument_parser(usage="Calculate div/curl of strair")
    prsr.add_argument("-y", "--year-range", type=int,nargs=2, default=[1979]*2)
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Use keys as the output variable names to store data/metadata in
    # dictionaries so that nc saving function can be used in a loop (see end
    # of script). All diagnostics are saved in the same file, 1 per year.

    # Attributes for time (variable name, and attributes including "units" and
    # "calendar" which are used to determine datetime values:
    nc_t_name = "time"
    nc_t_attr = {"bounds"       : f"{nc_t_name}_bnds",
                 "calendar"     : "365_day",
                 "long_name"    : "time",
                 "standard_name": "time",
                 "units"        : "days since 1979-01-01"}   

    # Attributes for all other variables (i.e., the diagnostics):
    # (note: currently no CF standard name for wind-stress div/curl)
    nc_var_attr = {}
    nc_var_attr["curl_strair"] = {"long_name": "Wind stress curl"      , "units": "N m-3"}
    nc_var_attr["div_strair"]  = {"long_name": "Wind stress divergence", "units": "N m-3"}
    nc_var_attr["div_u"]       = {"long_name": "Sea ice drift divergence", "units": "s-1",
                                  "standard_name": "divergence_of_sea_ice_velocity"}

    # Shared attributes:
    for k in nc_var_attr.keys():
        nc_var_attr[k]["cell_measures"] = "area: tarea"
        nc_var_attr[k]["cell_methods"]  = f"{nc_t_name}: mean area: mean"
        nc_var_attr[k]["coordinates"]   = "TLON TLAT"

    # Shared global attributes in all files:
    nc_global_attr = {}
    nc_global_attr["title"] = "CICE diagnostics: divergence and curl"

    for x in ["author", "contact", "institution"]:
        if getattr(cfg, x) != "":
            nc_global_attr[x] = getattr(cfg, x)

    if cfg.title != "":
        nc_global_attr["comment"] = cfg.title

    if cfg.cice_title != "":
        nc_global_attr["source"] = cfg.cice_title

    nc_global_attr["external_variables"] = "tarea"

    # Sub-directories of cfg.data_path["proc_?"] for outputs
    # (nc_tools.save_netcdf() creates directories if they do not exist):
    sdir_name = "div_curl"

    # File names for outputs, with a format for "d" (daily) or "m" (monthly)
    # and the year of the file:
    file_name = "div_curl_{}_{:04}.nc"

    # Load grid cell areas "tarea", cell heights "HTN" and "HTE", and cell
    # centre coordinates "TLON" and "TLAT:
    tarea, htn, hte, tlon, tlat = cice.get_grid_data(
        ["tarea", "HTN", "HTE", "TLON", "TLAT"])

    ny, nx = np.shape(tarea)

    # NetCDF dimensions are the same for all so prepare first:
    nc_dims = {}
    nc_dims[nc_t_name] = {"size": None}
    nc_dims["bnd"]     = {"size": 2}
    nc_dims["y"]       = {"size": ny}
    nc_dims["x"]       = {"size": nx}

    # Also need to save the coordinates:
    nc_cvars = {}
    nc_cvars["TLON"] = {"data": tlon, "dims": ("y","x"),
                        "attr": {"long_name"    : "T-cell centre longitude",
                                 "standard_name": "longitude",
                                 "units"        : "degrees_east"}}
    nc_cvars["TLAT"] = {"data": tlat, "dims": ("y","x"),
                        "attr": {"long_name"    : "T-cell centre latitude",
                                 "standard_name": "latitude",
                                 "units"        : "degrees_north"}}

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

        # Load CICE daily velocity and atmosphere-ice stress data from history:
        _, data_cice = cice.get_history_data(
            ["uvel_d", "vvel_d", "strairx_d", "strairy_d"],
            dt_min=dt(y, 1, 1), dt_max=dt(y, 12, 31, 23, 0), frequency="daily",
            set_miss_to_nan=False, slice_to_atm_grid=False)

        uvel    = data_cice[0]
        vvel    = data_cice[1]
        strairx = data_cice[2]
        strairy = data_cice[3]

        # Create land mask from wherever velocity is undefined:
        land_mask = np.where(abs(uvel[0,:,:]) > 1.E10, np.nan, 1)

        # Missing to zero:
        uvel    = np.where(abs(uvel)    > 1.E10, 0., uvel   )
        vvel    = np.where(abs(vvel)    > 1.E10, 0., vvel   )
        strairx = np.where(abs(strairx) > 1.E10, 0., strairx)
        strairy = np.where(abs(strairy) > 1.E10, 0., strairy)
    
        data_out = {}
        data_out["curl_strair_d"] = diag.curl(htn, hte, tarea, strairx, strairy)
        data_out["div_strair_d"]  = diag.divergence(htn, hte, tarea, strairx, strairy)
        data_out["div_u_d"]       = diag.divergence(htn, hte, tarea, uvel, vvel)

        for x in list(data_out.keys()):
            data_out[x.replace("_d", "_m")] = \
                diag.monthly_mean(date_d, data_out[x], date_bnds_m)[1]

        for x in list(data_out.keys()):
            data_out[x] *= land_mask[np.newaxis,:,:]

        for t in "dm":

            # Variable for time and time bounds is the same for all:
            nc_vars_t = {}
            nc_vars_t[nc_t_name] = {"data": time[t], "dims": (nc_t_name,),
                                    "attr": nc_t_attr}

            nc_vars_t[f"{nc_t_name}_bnds"] = {"data": time_bnds[t],
                                              "dims": (nc_t_name,"bnd")}

            nc_vars_x = {}
            for x in nc_var_attr.keys():
                nc_vars_x[f"{x}_{t}"] = {"data": data_out[f"{x}_{t}"],
                                         "dims": (nc_t_name,"y","x"),
                                         "attr": nc_var_attr[x]}

            # Finally, save the file:
            nc_tools.save_netcdf(file_name.format(t, y), nc_dims,
                                 {**nc_vars_t, **nc_cvars, **nc_vars_x},
                                 nc_global_attr=nc_global_attr,
                                 sort_attr=True,
                                 dir_save=Path(cfg.data_path[f"proc_{t}"],
                                               sdir_name))


if __name__ == "__main__":
    main()

