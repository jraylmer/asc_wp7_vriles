"""Calculate detrended 2D fields from atmospheric forcing data. Here, 'detrended'
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
from src.data import atmo, nc_tools
from src.diagnostics import vriles, diagnostics as diag


def main():

    prsr = script_tools.argument_parser(
        usage="Calculate detrended atmospheric data (2D fields)")

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
    # ---- Variables already available as daily means (add derived ones later).
    #      For simplicity of code, define the basic attributes here manually
    #      rather than inherifrom the input daily means. The variable names
    #      (keys) match those of the input files as well.
    nc_var_attr["t2"]   = {"long_name"    : "Near-surface air temperature",
                           "standard_name": "air_temperature",
                           "units"        : "K"}
    nc_var_attr["qlw"]  = {"long_name"    : "Surface downwelling longwave radiation",
                           "standard_name": "downwelling_longwave_flux_in_air",
                           "units"        : "W m-2"}
    nc_var_attr["qsw"]  = {"long_name"    : "Surface downwelling shortwave radiation",
                           "standard_name": "downwelling_shortwave_flux_in_air",
                           "units"        : "W m-2"}

    fields_to_load = list(nc_var_attr.keys())

    # ---- Derived variables from the above:
    nc_var_attr["qnet"] = {"long_name"    : "Surface net downwelling longwave + shortwave radiation",
                           "units"        : "W m-2"}
    # Note: closest CF standard_name seems to be
    # 'surface_downwelling_radiative_flux_in_sea_water', but strictly speaking
    # this is not what qnet will be as both fluxes are defined over land and
    # ocean, and an 'in_air' version does not seem to exist => no standard_name

    # Shared attributes:
    for k in nc_var_attr.keys():
        nc_var_attr[k]["cell_measures"] = "area: tarea"
        nc_var_attr[k]["cell_methods"]  = f"{nc_t_name}: mean area: mean"
        nc_var_attr[k]["coordinates"]   = "nav_lon nav_lat"

    # Shared global attributes in all files:
    nc_global_attr = {}
    nc_global_attr["title"] = "CICE diagnostics: detrended atmospheric forcing data"

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
        nc_global_attr["source"] = cfg.reanalysis_source

    nc_global_attr["external_variables"] = "tarea"

    # File names for outputs, with a format for 'd' (daily) or 'm' (monthly)
    # and the year of the file:
    file_name = "atmo_detrended_{}_y{:04}.nc"

    # Pre-load coordinates (different from CICE grid as 131 grid here, which
    # we keep (and same names 'nav_lon', 'nav_lat') for consistency with
    # atmospheric data. Load any field arbitrarily just to save coordinates:
    _, lon, lat, _ = atmo.get_atmospheric_data_time_averages(["t2"],
        dt_min=dt(y1, 1, 1), dt_max=dt(y1, 1, 2), slice_to_cice_grid=False)

    ny, nx = np.shape(lon)

    # NetCDF dimensions are the same for all so prepare first:
    nc_dims = {}
    nc_dims[nc_t_name] = {"size": None}
    nc_dims["bnd"]     = {"size": 2}
    nc_dims["y"]       = {"size": ny}
    nc_dims["x"]       = {"size": nx}

    # Also need to save the coordinates:
    nc_cvars = {}
    nc_cvars["nav_lon"] = {"data": lon, "dims": ("y", "x"),
                           "attr": {"standard_name": "longitude",
                                    "units"        : "degrees_east"}}
    nc_cvars["nav_lat"] = {"data": lat, "dims": ("y", "x"),
                           "attr": {"standard_name": "latitude",
                                    "units"        : "degrees_north"}}


    # Store residual components, reshaped into (nyears,365,ny,nx), here:
    data_out = {}

    # Load atmospheric forcing daily data (ALL years)
    data = list(atmo.get_atmospheric_data_time_averages(fields_to_load,
        freq="daily", dt_min=dt(y1, 1, 1), dt_max=dt(y2, 12, 31),
        slice_to_cice_grid=False, auto_units=False))[3:]

    # Make the list variables in 'data' match the total list of fields
    # for the loop of the detrending step below. Specifically, create
    # qnet = qsw + qlw and update the list of fields:
    #
    fields = fields_to_load + ["qnet"]
    data.append(  data[fields_to_load.index("qsw")]
                + data[fields_to_load.index("qlw")])

    # Atmospheric forcing data does not contain missing data so the part of the
    # CICE-diagnostics version of this script that temporarily sets land values
    # to zero is not required here.

    for j in range(len(fields)):
        _, residual_j, _ = vriles.seasonal_trend_decomposition_periodic_2D(
            data[j], n_ma=cmd.n_moving_average)

        data_out[f"{fields[j]}_d"] = np.reshape(residual_j, (nyears, 365, ny, nx))

    # Loop over each individual year and save each file
    # and also calculate monthly means:
    for jy in range(nyears):

        y = y1 + jy

        # Output arrays for this year:
        data_out_y = {}
        for f in list(data_out.keys()):
            data_out_y[f] = data_out[f][jy,:,:,:]

        # Create new datetimes and bounds; save time and time_bnds into
        # dictionary for daily and monthly:
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

            nc_tools.save_netcdf(file_name.format(t, y), nc_dims,
                                 {**nc_vars_t, **nc_cvars, **nc_vars_x},
                                 nc_global_attr=nc_global_attr,
                                 sort_attr=True,
                                 dir_save=Path(cfg.data_path[f"atmo_{t}"]))


if __name__ == "__main__":
    main()

