"""Cleanup script for CDO-interpolated atmospheric data, which should not be
run directly; instead, it is run automatically by the bash script

    scripts/process_atmo_data/interp_to_cice_grid.sh

to finalise the interpolated data in the format required by CICE.

The input file (flag -i), input (JRA-55-do) variable name (-v), and year (-y)
are required. The output directory can also be specified (flags -o) but if not
they it is read from the config.

Wind is a special case, as the (u,v) components need to be rotated onto the
local direction of the CICE grid using the (sine/cosine of) grid angle files
(the transformation requires both components). In this case, two values should
be passed to each of the flags -i and -v, corresponding to the interpolated
(u,v) components respectively, and this script will then rotate and save both
components.
"""

from calendar import monthrange
from datetime import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import nc_tools


# Various metadata for each variable including filename prefixes and
# netCDF variable attributes. Keys are the input (JRA-55-do) names:
metadata = {
    "tas" : {"varname"      : "t2",
             "filename"     : "t2",
             "units"        : "K",
             "standard_name": "air_temperature",
             "long_name"    : "Near-surface air temperature"},
    "huss": {"varname"      : "q2",
             "filename"     : "q2",
             "units"        : "1",
             "standard_name": "specific_humidity",
             "long_name"    : "Near-surface specific humidity"},
    "rlds": {"varname"      : "qlw",
             "filename"     : "qlw",
             "units"        : "W m-2",
             "standard_name": "downwelling_longwave_flux_in_air",
             "long_name"    : "Surface downwelling longwave radiation"},
    "rsds": {"varname"      : "qsw",
             "filename"     : "qsw",
             "units"        : "W m-2",
             "standard_name": "downwelling_shortwave_flux_in_air",
             "long_name"    : "Surface downwelling shortwave radiation"},
    "prra": {"varname"      : "rain",
             "filename"     : "precip",
             "units"        : "kg m-2 s-1",
             "standard_name": "rainfall_flux",
             "long_name"    : "Surface rainfall flux"},
    "prsn": {"varname"      : "snow",
             "filename"     : "snow",
             "units"        : "kg m-2 s-1",
             "standard_name": "snowfall_flux",
             "long_name"    : "Surface snowfall flux"},
    "psl" : {"varname"      : "psl",
             "filename"     : "psl",
             "units"        : "Pa",
             "standard_name": "air_pressure_at_mean_sea_level",
             "long_name"    : "Sea level pressure"},
    "uas" : {"varname"      : "u10",
             "filename"     : "u10",
             "units"        : "m s-1",
             "standard_name": "x_wind",
             "long_name"    : "Near-surface wind x-component"},
    "vas" : {"varname"      : "v10",
             "filename"     : "v10",
             "units"        : "m s-1",
             "standard_name": "y_wind",
             "long_name"    : "Near-surface wind y-component"}
}


# Attributes for coordinate variables:
nc_time_attr = {"calendar": "noleap", "units": "days since 1900-01-01"}
nc_lon_attr  = {"standard_name": "longitude", "units": "degrees_east"}
nc_lat_attr  = {"standard_name": "latitude" , "units": "degrees_north"}


def main():

    prsr = script_tools.argument_parser(
        usage="Clean interpolated NetCDF forcing files")
    prsr.add_argument("-y", "--year"    , type=int,            required=True)
    prsr.add_argument("-v", "--variable", type=str, nargs="*", required=True)
    prsr.add_argument("-i", "--in-file" , type=str, nargs="*", required=True)
    prsr.add_argument("-o", "--out-dir" , type=str, default="")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Determine the output directory for the forcing data. If the input -o
    # is set to '' (empty string), i.e., not passed to this script, then get
    # it from the config.
    #
    if cmd.out_dir == "":
        out_dir = cfg.data_path["atmo_forc"]
    else:
        out_dir = cmd.out_dir

    # Global attributes common to all variables:
    nc_global_attr = {
        "title": f"{cfg.reanalysis_name} {cfg.reanalysis_frequency} "
                 + "atmospheric forcing prepared for CPOM CICE",
        "source": cfg.reanalysis_source, "author": cfg.author}

    # Open interpolated data (input). There is at least one input file
    # (two if cleaning the wind components):
    with nc.Dataset(Path(cmd.in_file[0]), mode="r") as ncdat:
        lon      = np.array(ncdat.variables["TLON"])
        lat      = np.array(ncdat.variables["TLAT"])
        var_data = [np.array(ncdat.variables[cmd.variable[0]])]

    # If there is a second, load it, then proceed with the rotation
    # transformation under the assumption that these are uas and vas:
    if len(cmd.in_file) == 2:
        with nc.Dataset(Path(cmd.in_file[1]), mode="r") as ncdat:
            var_data.append(np.array(ncdat.variables[cmd.variable[1]]))

        # Check we are dealing with wind components (they could be
        # either way around):
        if   cmd.variable[0] == "uas" and cmd.variable[1] == "vas":
            ju = 0 ; jv = 1  # indices of u/v in var_data list
        elif cmd.variable[0] == "vas" and cmd.variable[1] == "uas":
            ju = 1 ; jv = 0
        else:
            raise ValueError("For two inputs, expected uas and vas (got "
                             + f"{cmd.variable[0]} and {cmd.variable[1]})")

        # Load the CICE grid angle data [do not use the rotate_vectors()
        # function in src.data because that transforms from CICE grid
        # back to regular lon/lat. Here we want the opposite/'original'
        # transformation from lon/lat to CICE grid, which uses the same
        # angle data but the inverse transform is not coded there]:
        with nc.Dataset(cfg.data_path["sina131"]) as ncdat:
            sina = np.array(ncdat.variables["sina"])

        with nc.Dataset(cfg.data_path["cosa131"]) as ncdat:
            cosa = np.array(ncdat.variables["cosa"])

        # Rotation transformation (note sina and cosa have a length-1
        # dimension on axis 0 corresponding to time):
        uout =  var_data[ju]*cosa + var_data[jv]*sina
        vout = -var_data[ju]*sina + var_data[jv]*cosa

        # Update variable list (and set missing to zero; note NaNs come from
        # the angle data over land, so this step is not needed for the other
        # variables which are defined on land and over ocean. CICE cannot
        # handle missing values when reading netCDF forcing files):
        var_data[ju] = np.where(np.isnan(uout), 0., uout)
        var_data[jv] = np.where(np.isnan(vout), 0., vout)

    nt, ny, nx  = np.shape(var_data[0])

    # Create 3-hourly time data for one year, first as calendar dates,
    # then convert using the nc.date2num() function:
    cal_dates = []
    for mn in range(1, 13):
        # Use monthrange to get number of days in month mn. However, we use a
        # no-leap calendar. Therefore, just choose some arbitrary non-leap year
        # (here, 1999) so that for Feb (mn == 2) this gives 28, not 29:
        for day in range(1, monthrange(1999, mn)[1] + 1):
            for hr in range(0, 24, 3):
                cal_dates.append(dt(cmd.year, mn, day, hr, 0, 0, 0))

    time = nc.date2num(cal_dates, units=nc_time_attr["units"],
                       calendar=nc_time_attr["calendar"])

    # Set the arguments needed for the nc_tools.save_netcdf() function.
    # Dimensions (we don't bother with bounds for forcing files):
    #
    nc_dims = {"time": {"size": None}, "x": {"size": nx}, "y": {"size": ny}}

    # Variables: set the coordinate variables here, then loop over variables
    # below [to cover case of 2 inputs (wind) or 1 input (all others)]:
    nc_vars = {"time"    : {"data": time, "dims": ("time", ), "attr": nc_time_attr},
               "nav_lon" : {"data": lon , "dims": ("y", "x"), "attr": nc_lon_attr},
               "nav_lat" : {"data": lat , "dims": ("y", "x"), "attr": nc_lat_attr}}

    for i in range(len(cmd.in_file)):
        # Copy coordinate variables from above then add the data variable:
        nc_vars_i = {**nc_vars}
        nc_vars_i[metadata[cmd.variable[i]]["varname"]] = {
            "data": var_data[i],
            "dims": ("time", "y", "x"),
            "attr": {"coordinates"  : "nav_lon nav_lat",
                     "long_name"    : metadata[cmd.variable[i]]["long_name"],
                     "standard_name": metadata[cmd.variable[i]]["standard_name"],
                     "units"        : metadata[cmd.variable[i]]["units"]}
        }

        # For input to CICE v5.*, format must be NETCDF3_CLASSIC uncompressed:
        nc_tools.save_netcdf(
            f"{metadata[cmd.variable[i]]['filename']}_y{cmd.year}.nc",
            nc_dims, nc_vars_i, nc_global_attr=nc_global_attr,
            dir_save=Path(out_dir), compress=False,
            nc_Dataset_kw={"format": "NETCDF3_CLASSIC"})


if __name__ == "__main__":
    main()

