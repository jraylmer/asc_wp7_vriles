"""Script to merge raw SSM/I daily sea ice concentration files into one file
for a given year and prepare for interpolation.

Version 2 of NASA Team (NSIDC-0051) and version 4 of Bootstrap (NSIDC-0079)
provide individual netCDF files for each day of the year. For earlier years
where the data is actually once every two days, as well as the missing period
from mid December 1987 to January 1988, files are still provided on the missing
days but they contain no data. Also, the variable names are labelled with the
sensor ('N07_ICECON', 'N08_ICECON', etc.) which can differ per file. As such,
combining with e.g., CDO mergetime fails.

This script does it manually by checking each file's contents and always saving
a merged file consistently with 365 time steps (leap days are also ignored
here), with missing days set to NaN. The script will also work if the files
for such missing days are simply not downloaded (i.e., unavailable at the raw
data directory).

This is a preparatory step before interpolating the data to the CICE grid. Sea
ice concentration values less than 0 and larger than 1 (missing/land flags) are
set to 0 here for the purposes of interpolation; afterwards, the CICE land
mask is applied (outside of this script). The (possibly time-varying) pole-hole
mask is also determined from the integer flags in the raw SSM/I data and save
separately to the concatenated data. Externally to this script, both are
interpolated onto the CICE grid, and then the interpolated pole-hole mask is
applied to the interpolated sea ice concentration data.

This script also returns to the console two values needed for interpolation:
the location of the land mask file and output directory, both of which are
determined from the config. In the bash script:

    PYOUT=$(python ./scripts/process_ssmi_data/mergetime.py [options])

    LMSKFILE=${PYOUT[-2]}
    OUTDIR=${PYOUT[-1]}

"""

from datetime import datetime as dt, timedelta
from pathlib import Path

import numpy as np
import netCDF4 as nc

from src import script_tools
from src.io import config as cfg
from src.data import nc_tools


# NetCDF time calendar and units to save data with:
nc_t_calendar = "365_day"
nc_t_units    = "days since 1979-01-01"

# NetCDF global attributes to save data with [rest set in main()]:
nc_global_attr = {"title": "SSM/I sea ice concentrations on CICE grid"}


def get_aice_var(ncdat_in):
    """Identify the netCDF variable name corresponding to sea ice concentration
    in the input netCDF file (it is not the same for all input files as they
    are named with the specific instrument/sensor).
    """

    # These variables do *not* correspond to aice:
    other_vars = ["t", "time", "x", "y", "crs"]

    # Get a list of all other variable names in ncdat_in:
    possible_names = [str(vname) for vname     in ncdat_in.variables
                                  if vname not in other_vars        ]

    if len(possible_names) == 0:
        return ""
    elif len(possible_names) == 1:
        return possible_names[0]
    else:
        print(f"Warning: {len(possible_names)} possible variables for aice")
        return possible_names[0]


def main():

    prsr = script_tools.argument_parser(usage="Prepare raw daily SSM/I data")
    prsr.add_argument("-y", "--year", type=int, default=1979)
    prsr.add_argument("-o", "--output-file-aice", type=str)
    prsr.add_argument("-p", "--output-file-pmask", type=str)
    prsr.add_argument("-d", "--dataset", type=str, default="nt", choices=["nt", "bt"])
    prsr.add_argument("-a", "--attributes", type=str, nargs="*", default=[],
                      help="Additional key:val pairs for netcdf global attributes")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Update netCDF global attributes from config:
    for x in ["author", "contact", "institution"]:
        if getattr(cfg, x) != "":
            nc_global_attr[x] = getattr(cfg, x)

    if cfg.title != "":
        nc_global_attr["comment"] = cfg.title

    # File names are assumed to be formatted as they come from the NSIDC
    # Also set further global attributes here and data scale factor and
    # pole-hole flag values, which depend on dataset:
    if cmd.dataset == "nt":

        file_fmt = "NSIDC0051_SEAICE_PS_N25km_{:04}{:02}{:02}_v2.0.nc"

        scale_factor   = 0.004
        pole_hole_flag = 251

        nc_global_attr["source"] = ("DiGirolamo, N., C. L. Parkinson, D. J. "
            + "Cavalieri, P. Gloersen, and H. J. Zwally. (2022). Sea Ice "
            + "Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS "
            + "Passive Microwave Data, Version 2 [Data Set]. Boulder, "
            + "Colorado USA. NASA National Snow and Ice Data Center "
            + "Distributed Active Archive Center. "
            + "https://doi.org/10.5067/MPYG15WAA4WX. Date Accessed "
            + "04-15-2024.")
    else:

        file_fmt = "NSIDC0079_SEAICE_PS_N25km_{:04}{:02}{:02}_v4.0.nc"

        scale_factor   = 0.001
        pole_hole_flag = 1100

        # Note: Bootstrap only provides a 'missing value' (1100) and 'land'
        # (1200) flag, unlike NASA Team which provides a separate flag for
        # the pole hole. The former flag does identify the pole hole, but
        # unlike NASA Team the actual area fluctuates day-to-day (even before
        # interpolation). It is not clear to me why this happens.

        nc_global_attr["source"] = ("Comiso, J. C. (2023). Bootstrap Sea Ice "
            + "Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS, "
            + "Version 4 [Data Set]. Boulder, Colorado USA. NASA National "
            + "Snow and Ice Data Center Distributed Active Archive Center. "
            + "https://doi.org/10.5067/X5LG68MH013O. Date Accessed "
            + "04-15-2024.")

    # Add additional global attributes from command line:
    for j in range(len(cmd.attributes)//2):
        att = cmd.attributes[2*j]
        val = cmd.attributes[2*j+1]
        if att in nc_global_attr.keys():  # append to existing attribute
            nc_global_attr[att] += f"; {val}"
        else:  # create new attribute
            nc_global_attr[att] = val

    # Create daily datetime/bounds. Note the SSM/I data has leap days
    # (obviously), but since we don't need it for comparing with CICE (as
    # that does not include leap days), might as well also skip them here:
    date, _, time, time_bnds = nc_tools.dt_daily(cmd.year, no_leap=True,
                                                 nc_units=nc_t_units,
                                                 nc_calendar=nc_t_calendar)

    # Need longitudes/latitudes in the output files for interpolation, but
    # these are not included in the raw data. Instead, get them from the
    # ancilliary dataset (NSIDC-0771):
    #
    with nc.Dataset(cfg.data_path["ssmi_raw_lonlat"], "r") as ncdat:
        # We transpose arrays (and aice when read in during loop below)
        # because the indexing in the raw data is reversed for some reason:
        lon = np.array(ncdat.variables["longitude"]).T
        lat = np.array(ncdat.variables["latitude"]).T

    ny, nx = np.shape(lon)

    # Prefill the aice array for all time/space with a missing value flag,
    # initially -1 arbitrarily (later it will change to NaN, but need to first
    # load raw data as integer -- see comments below). Below, skip days that do
    # not have data (empty files) and update those that do with the actual data:
    aice = -np.ones((len(date), ny, nx), dtype=int)

    for j in range(len(date)):

        file_j = Path(cfg.data_path[f"ssmi_{cmd.dataset}_raw"],
                      f"{date[j].year:04}",
                      file_fmt.format(date[j].year, date[j].month, date[j].day))

        try:
            with nc.Dataset(file_j, "r") as ncdat:
                aice_var = get_aice_var(ncdat)
                if aice_var != "":
                    # Get the data. It is convenient to initially load as the
                    # unscaled integer values (in range 0-255) it is stored as
                    # then convert at the end, so that the flag for pole hole
                    # can be used. So, extract variable first and deactivate
                    # autoscaling. Also, remove any singleton dimensions and
                    # then transpose to correct orientation:
                    nc_var = ncdat.variables[aice_var]
                    nc_var.set_auto_scale(False)
                    aice[j,:,:] = np.squeeze(nc_var[:]).astype(int).T

        except FileNotFoundError:
            print(f"Warning: no file found for {date[j].strftime('%Y%m%d')}")

    # Create pole hole mask as a function of time (it changes in size when the
    # instrument changes, although this only happens in the middle of one year,
    # 1987, where it changes from SMMR to SSM/I).
    #
    pmask = np.where(aice == pole_hole_flag, np.nan, 1.)

    # Create sea ice concentration array for output. For interpolation purposes
    # (after this script), set missing values and land flags to 0, except
    # complete missing days (initially set as -1 above) which can now be set
    # to NaN. Note the CICE land mask and pole hole masks are applied after
    # interpolation:
    aice = np.float64(aice) * scale_factor    # apply scale factor
    aice = np.where(aice < 0., np.nan, aice)  # complete missing days
    aice = np.where(aice > 1., 0., aice)      # all other flags to 0

    # Save concatenated data using nc_tools.save_netcdf() function.
    # Prepare the arguments to this function (see docs)...
    #
    # NetCDF dimensions:
    nc_dims = {"time": {"size": None}, "bnd": {"size": 2},
               "ny"  : {"size": ny}  , "nx" : {"size": nx}}

    # Attributes for each netCDF variable (apart from lon/lat, these are
    # carried over to the final, interpolated outputs):
    nc_t_attr     = {"bounds": "time_bnds", "calendar": nc_t_calendar, "units": nc_t_units}
    nc_lon_attr   = {"units": "degrees_east"}
    nc_lat_attr   = {"units": "degrees_north"}
    nc_pmask_attr = {"coordinates": "lon lat", "long_name": "Pole hole mask"}
    nc_aice_attr  = {"coordinates": "lon lat", "long_name": "sea ice concentration",
                     "standard_name": "sea_ice_area_fraction", "units": "1"}

    # NetCDF variables:
    nc_vars = {}
    nc_vars["time"]      = {"data": time     , "dims": ("time",)      , "attr": nc_t_attr  }
    nc_vars["time_bnds"] = {"data": time_bnds, "dims": ("time", "bnd"), "attr": {}         }
    nc_vars["lon"]       = {"data": lon      , "dims": ("ny", "nx")   , "attr": nc_lon_attr}
    nc_vars["lat"]       = {"data": lat      , "dims": ("ny", "nx")   , "attr": nc_lat_attr}
   
    nc_var_pmask = {"data": pmask, "dims": ("time", "ny", "nx"), "attr": nc_pmask_attr}
    nc_var_aice  = {"data": aice , "dims": ("time", "ny", "nx"), "attr": nc_aice_attr }

    out_file_pmask = Path(cmd.output_file_pmask).resolve()
    out_file_aice  = Path(cmd.output_file_aice).resolve()

    nc_tools.save_netcdf(str(out_file_aice.name), nc_dims, {"aice": nc_var_aice, **nc_vars},
                         nc_global_attr=nc_global_attr,
                         dir_save=out_file_aice.parent, compress=False)

    nc_tools.save_netcdf(str(out_file_pmask.name), nc_dims, {"pmask": nc_var_pmask, **nc_vars},
                         nc_global_attr=nc_global_attr,
                         dir_save=out_file_pmask.parent, compress=False)

    # Get the required info for the bash script: the path to the land mask file
    # and the output directory for sea ice concentration on the CICE grid:
    path_lmsk = cfg.data_path["lmsk129"]
    path_out  = Path(cfg.data_path[f"ssmi_{cmd.dataset}_cice"],
                     "aice", "daily").resolve()

    # Leading space and supression of newline character at the end ensures bash
    # script can convert output to array with last two elements being the two
    # paths (this probably won't work if the paths contain spaces...):
    print(f" {str(path_lmsk)} {str(path_out)}", end="")


if __name__ == "__main__":
    main()

