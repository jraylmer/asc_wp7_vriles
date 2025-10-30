"""Calculate detrended 2D fields from daily model outputs. Here, 'detrended'
means that a moving average filter of width 31 days is applied to the daily
data, from which a mean seasonal cycle and trend for each day of the year
is computed, and these two components are subtracted from the original data.
This method is consistent with the default detrending method used to identify
VRILEs from sea ice extent.
"""

from datetime import datetime as dt
from pathlib import Path

import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import cice, nc_tools
from src.diagnostics import vriles, diagnostics as diag


def main():

    prsr = script_tools.argument_parser(
        usage="Calculate detrended history data (2D fields)")

    prsr.add_argument("-y", "--year-range", type=int,nargs=2,
                      default=(1979, 2023))
    prsr.add_argument("-n", "--n-moving-average", type=int, default=31,
                      help="Moving-average filter width (days; odd integer)")

    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    y1 = cmd.year_range[0]
    y2 = cmd.year_range[1]
    nyears = y2 - y1 + 1

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
    nc_var_attr = {}
    # --- variables already in history output (add derived diagnostics later):
    nc_var_attr["daidtd"] = {"long_name": "area tendency dynamics"  , "units": "%/day"}
    nc_var_attr["daidtt"] = {"long_name": "area tendency thermo"    , "units": "%/day"}
    nc_var_attr["dvidtd"] = {"long_name": "volume tendency dynamics", "units": "cm/day"}
    nc_var_attr["dvidtt"] = {"long_name": "volume tendency thermo"  , "units": "cm/day"}
    nc_var_attr["meltb"]  = {"long_name": "basal ice melt"          , "units": "cm/day"}
    nc_var_attr["meltl"]  = {"long_name": "top ice melt"            , "units": "cm/day"}
    nc_var_attr["meltt"]  = {"long_name": "lateral ice melt"        , "units": "cm/day"}

    # Shared global attributes in all files:
    nc_global_attr = {}
    nc_global_attr["title"] = "CICE diagnostics: detrended history data"

    for x in ["author", "contact", "institution"]:
        if getattr(cfg, x) != "":
            nc_global_attr[x] = getattr(cfg, x)

    if cfg.title != "":
        nc_global_attr["comment"] = cfg.title
    else:
        nc_global_attr["comment"] = ""

    nc_global_attr["comment"] += ("; computed by subtracting the mean seasonal "
                                  + "cycle and linear trend on each day of "
                                  + f"year over {y1:04}-{y2:04}, each defined "
                                  + f"after applying a {cmd.n_moving_average}-"
                                  + "day moving average filter, from the "
                                  + "original data, at each grid point.")

    if cfg.cice_title != "":
        nc_global_attr["source"] = cfg.cice_title

    nc_global_attr["external_variables"] = "tarea"

    # Sub-directories of cfg.data_path["proc_?"] for outputs
    # (nc_tools.save_netcdf() creates directories if they do not exist):
    sdir_name = "hist_detrended"

    # File names for outputs, with a format for "d" (daily) or "m" (monthly)
    # and the year of the file:
    file_name = "hist_detrended_{}_{:04}.nc"

    # Load grid T-cell centre coordinates "TLON" and "TLAT:
    tlon, tlat = cice.get_grid_data(["TLON", "TLAT"])

    ny, nx = np.shape(tlon)

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

    fields = [f"{x}_d" for x in nc_var_attr.keys()]

    # Additional fields needed for derived outputs:
    fields_der = ["fsurf_ai_d", "fcondtop_ai_d"]  # for surface energy balance

    # Load CICE daily data (ALL years) from history:
    _, data_cice = cice.get_history_data(fields + fields_der,
        dt_min=dt(y1, 1, 1), dt_max=dt(y2, 12, 31, 23, 0), frequency="daily",
        set_miss_to_nan=True, slice_to_atm_grid=False)

    data_cice_extra = data_cice[len(fields):]
    data_cice       = data_cice[:len(fields)]

    # Add total tendencies and surface energy balance
    # (seb_ai = fsurf_ai - fcondtop):
    fields += ["daidt_d", "dvidt_d", "seb_ai_d"]

    nc_var_attr["daidt"]  = {"long_name": "area tendency total"   , "units": "%/day"}
    nc_var_attr["dvidt"]  = {"long_name": "volume tendency total" , "units": "cm/day"}
    nc_var_attr["seb_ai"] = {"long_name": "surface energy balance", "units": "W m-2"}

    data_cice += [  data_cice[fields.index("daidtd_d")]
                  + data_cice[fields.index("daidtt_d")],    # = daidt
                    data_cice[fields.index("dvidtd_d")]
                  + data_cice[fields.index("dvidtt_d")],    # = dvidt
                  data_cice_extra[0] - data_cice_extra[1]]  # = seb_ai

    # Add processed diagnostics (wind stress divergence/curl):
    _, data_proc = cice.get_processed_data("div_curl",
        ["div_strair_d", "curl_strair_d"], dt_min=dt(y1, 1, 1),
        dt_max=dt(y2, 12, 31, 23, 0), frequency="daily",
        slice_to_atm_grid=False)

    fields += ["div_strair_d", "curl_strair_d"]

    nc_var_attr["div_strair"]  = {"long_name": "Wind stress divergence", "units": "N m-3"}
    nc_var_attr["curl_strair"] = {"long_name": "Wind stress curl"      , "units": "N m-3"}

    data_cice += data_proc

    # Create land mask and then set missing to 0 for purposes of calculating
    # trends (then later apply land mask to put it back):
    land_mask = np.where(np.isnan(data_cice[0][0,:,:]), np.nan, 1.0)

    # Add shared variable attributes:
    for k in nc_var_attr.keys():
        nc_var_attr[k]["cell_measures"] = "area: tarea"
        nc_var_attr[k]["cell_methods"]  = f"{nc_t_name}: mean area: mean"
        nc_var_attr[k]["coordinates"]   = "TLON TLAT"

    # Store residual components, reshaped into (nyears,365,ny,nx), here:
    data_out = {}

    for j, f in zip(range(len(data_cice)), fields):

        # Set missing to 0:
        data_cice[j] = np.where(np.isnan(data_cice[j]), 0., data_cice[j])

        _, residual_j, _ = vriles.seasonal_trend_decomposition_periodic_2D(
                data_cice[j], n_ma=cmd.n_moving_average)

        residual_j *= land_mask[np.newaxis,:,:]
        residual_j = np.reshape(residual_j, (nyears, 365, ny, nx))

        data_out[f] = residual_j

    # Loop over each individual year and save each file
    # and also calculate monthly means:
    for jy in range(nyears):

        y = y1 + jy

        # Output arrays for this year:
        data_out_y = {}
        for f in list(data_out.keys()):
            data_out_y[f] = data_out[f][jy,:,:,:]

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

        for f in list(data_out_y.keys()):
            data_out_y[f.replace("_d", "_m")] = \
                diag.monthly_mean(date_d, data_out_y[f], date_bnds_m)[1]

        for t in "dm":

            # Variable for time and time bounds is the same for all:
            nc_vars_t = {}
            nc_vars_t[nc_t_name] = {"data": time[t], "dims": (nc_t_name,),
                                    "attr": nc_t_attr}

            nc_vars_t[f"{nc_t_name}_bnds"] = {"data": time_bnds[t],
                                              "dims": (nc_t_name,"bnd")}

            nc_vars_x = {}
            for x in nc_var_attr.keys():
                nc_vars_x[f"detrended_{x}_{t}"] = {"data": data_out_y[f"{x}_{t}"],
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

