"""Calculate daily and monthly averages of the atmospheric forcing data after
it has been interpolated onto the CICE grid. All variables are saved in one
netCDF file for each year, for each of daily/monthly averages, and inherit
the global and variable attributes from the forcing files (except for time).
"""

from pathlib import Path

import netCDF4 as nc
import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import nc_tools
from src.diagnostics import diagnostics as diag


# New time attributes for daily/monthly average outputs:
nc_t_attr = {"bounds"       : "time_bnds",
             "calendar"     : "365_day",
             "long_name"    : "time",
             "standard_name": "time",
             "units"        : "days since 1979-01-01"}


def _nc_attr_dict(ncdat_in, varname=None):
    """Helper function: get a dictionary of netCDF global or variable (if
    string parameter 'varname' is given) attributes from a nc.Dataset.
    """
    if varname is None:
        return {k: ncdat_in.getncattr(k) for k in ncdat_in.ncattrs()}
    else:
        return {k: ncdat_in.variables[varname].getncattr(k)
                for k in ncdat_in.variables[varname].ncattrs()}


def main():

    prsr = script_tools.argument_parser(
        usage="Calculate atmospheric forcing daily/monthly averages")
    prsr.add_argument("-f", "--fields", type=str, nargs="*",
                      default=["t2", "q2", "psl", "u10", "v10", "qlw",
                               "qsw", "precip", "snow"])
    prsr.add_argument("-y", "--year-range", type=int, nargs=2,
                      default=(1979, 2023))
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Output file name with format placeholder for 'd' or 'm' and then year:
    file_name = "atmo_fields_{}_y{}.nc"

    # NetCDF dimensions to be passed to nc_tools.save_netcdf(). Start with
    # the time-related dimensions as they do not change, and add coordinate-
    # releated dimensions during the main loop below:
    nc_dims = {"time": {"size": None}, "bnd": {"size": 2}}

    # Add the netCDF global/variable attributes and coordinate-variable
    # attributes to these dictionaries during main loop below:
    nc_global_attr = {}
    nc_var_attr = {}
    nc_cvars = {}

    # Define a boolean to indicate which fields, referenced by the file name
    # field, are instantaneous (as opposed to an accumulation/average) as this
    # affects the computation of daily mean:
    #
    is_inst = {"t2": True, "q2": True, "u10": True, "v10": True, "psl": True,
               "qlw": False, "qsw": False, "precip": False, "snow": False}

    # Begin main loop over years requested:
    for y in range(cmd.year_range[0], cmd.year_range[1]+1, 1):

        # Daily and monthly datetime bounds for this year:
        date_d, date_bnds_d, time_d, time_bnds_d = \
            nc_tools.dt_daily(y, nc_units=nc_t_attr["units"],
                              nc_calendar=nc_t_attr["calendar"])

        date_m, date_bnds_m, time_m, time_bnds_m = \
            nc_tools.dt_monthly(y, nc_units=nc_t_attr["units"],
                                nc_calendar=nc_t_attr["calendar"])

        # Define variable for time and time bounds for outputs, for use in
        # function nc_tools.save_netcdf() at the end:
        nc_vars_t = {"d": {"data": time_d, "dims": ("time",), "attr": nc_t_attr},
                     "m": {"data": time_m, "dims": ("time",), "attr": nc_t_attr}}

        nc_vars_t_bnds = {"d": {"data": time_bnds_d, "dims": ("time", "bnd")},
                          "m": {"data": time_bnds_m, "dims": ("time", "bnd")}}

        # Begin loop over requested fields, saving the daily/monthly-averaged
        # fields into this dictionary with keys 'f_d' or 'f_m':
        data_out = {}

        for f in cmd.fields:

            # Variable names are the same as the field name used in the file
            # names in all cases except for precipitation:
            v = "rain" if f == "precip" else f

            # Load the forcing netCDF files explicitly rather than using the
            # src.data.atmo interface because we need to get various netCDF
            # attributes too. So, construct the path to the input file:
            #
            nc_file = str(Path(cfg.data_path["atmo_forc"], f"{f}_y{y}.nc"))

            # Open the netCDF file and load the data as required:
            with nc.Dataset(nc_file, "r") as ncdat:
                data = np.array(ncdat.variables[v])

                # Fill out the netCDF attributes (don't need to do this
                # every loop iteration, so just do it for the first year):
                if y == cmd.year_range[0]:
                    # Get variable attributes and append new ones:
                    nc_var_attr[v] = _nc_attr_dict(ncdat, v)
                    nc_var_attr[v]["cell_methods"] = "time: mean"

                    # Get the coordinates and global metadata (don't need to
                    # do this for every field, so just do it for the first):
                    if f == cmd.fields[0]:

                        # Global attributes (separate for daily and monthly):
                        for t, x in zip("dm", ["Daily", "Monthly"]):
                            nc_global_attr[t] = _nc_attr_dict(ncdat)
                            nc_global_attr[t]["title"] = (f"{x} means of "
                                + nc_global_attr[t]["title"])

                        lon = np.array(ncdat.variables["nav_lon"])
                        lat = np.array(ncdat.variables["nav_lat"])

                        # Update the nc_dims for later:
                        ny, nx = np.shape(lon)
                        nc_dims["y"] = {"size": ny}
                        nc_dims["x"] = {"size": nx}

                        # Add the coordinate variables to be combined later
                        # with the time and data variables:
                        nc_cvars["nav_lon"] = {"data": lon, "dims": ("y", "x"),
                            "attr": _nc_attr_dict(ncdat, "nav_lon")}

                        nc_cvars["nav_lat"] = {"data": lat, "dims": ("y", "x"),
                            "attr": _nc_attr_dict(ncdat, "nav_lat")}

            # If f is an instantaneous field, append the first time step of
            # the next year's file so that complete daily mean is computed on
            # December 31 (unless it doesn't exist, in which case neglect this):
            if is_inst[f] and y < cmd.year_range[1]:
                with nc.Dataset(nc_file.replace(f"_y{y}.nc", f"_y{y+1}.nc")) as ncdat:
                    data = np.concatenate((data, ncdat.variables[v][[0],:,:]), axis=0)

            # Compute daily mean (note we have already removed leap days from
            # forcing data and assume that all outputs will cover a whole year):
            #
            if is_inst[f]:
                data_out[f"{v}_d"] = np.zeros((365, ny, nx))

                # Forcing data is 3 hourly, starting at 00:00. So, for the
                # first daily mean, we want the mean calculated assuming a
                # linearly-interpolated profile between the first nine data
                # points, i.e., hours 00, 03, 06, 09, 12, 15, 18, 21, and 00
                # of the next day. This corresponds to a weighted average with
                # the first and last points (the 00 hours) with half the weight
                # of those of the 03-21 hour points (intuitively, this is each
                # 00 value 'contributing' half to each daily mean either side):
                #
                weights = [.5] + [1.]*7 + [.5]

                # Calculation of last time step is different if we do not have
                # the first time step (00:00 on Jan 01) of the next year:
                #
                for k in range(365 if y < cmd.year_range[1] else 364):
                    data_out[f"{v}_d"][k,:,:] = \
                        np.average(data[8*k:8*k+9,:,:], weights=weights, axis=0)

                # Last time step if needed: for simplicity, calculate the
                # unweighted mean of the last 8 points (hours 00-21 of 31 Dec):
                if y == cmd.year_range[1]:
                    data_out[f"{v}_d"][-1,:,:] = np.mean(data[-8:,:,:], axis=0)

            else:
                # Accumulations are already means (i.e., 3 hourly means)
                # so just take mean of each consecutive set of 8 data values:
                data_out[f"{v}_d"] = \
                    np.mean(np.reshape(data, (365, 8, ny, nx)), axis=1)

            # Compute monthly mean from daily means:
            data_out[f"{v}_m"] = diag.monthly_mean(date_d, data_out[f"{v}_d"],
                                                   date_bnds_m)[1]

        # Save all to netCDF for this year, one file each for daily/monthly:
        for t in "dm":

            # Variable information to pass to nc_tools.save_netcdf(); start
            # with the time and coordinate variables, then append the fields:
            nc_vars = {"time": nc_vars_t[t], "time_bnds": nc_vars_t_bnds[t],
                       **nc_cvars}

            for v in nc_var_attr.keys():
                nc_vars[f"{v}_{t}"] = {"data": data_out[f"{v}_{t}"],
                                       "dims": ("time", "y", "x"),
                                       "attr": nc_var_attr[v]}

            nc_tools.save_netcdf(file_name.format(t, y), nc_dims, nc_vars,
                nc_global_attr=nc_global_attr[t], sort_attr=True,
                dir_save=Path(cfg.data_path[f"atmo_{t}"]))


if __name__ == "__main__":
    main()

