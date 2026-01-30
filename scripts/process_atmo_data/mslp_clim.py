"""Calculate sea level pressure climatology from raw atmospheric forcing data.
"""

from pathlib import Path

import netCDF4 as nc
import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import nc_tools as nct


def main():

    prsr = script_tools.argument_parser(
        usage="Calculate MSLP climatology for anomaly calculations")
    prsr.add_argument("-y", "--year-range", type=int, nargs=2,
                      default=(1981, 2010))
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Open the raw sea level pressure data. It's very large so calculate
    # annual means with each year loaded, then average all the years at the end:
    psl_ymeans = []
    for y in range(cmd.year_range[0], cmd.year_range[1]+1, 1):

        print(f"Calculating {y} mean")

        nc_files = sorted([str(x) for x in Path(cfg.data_path["atmo_rean"],
                                                "psl").glob(f"*{y}*-{y}*.nc")])

        if y == cmd.year_range[0]:  # also get lon/lat and attributes
            lon, lon_bnds, lat, lat_bnds, psl = nct.get_arrays(nc_files,
                ["lon", "lon_bnds", "lat", "lat_bnds"], ["psl"])

            with nc.Dataset(nc_files[0], "r") as ncdat:
                lon_attr = ncdat.variables["lon"].__dict__
                lat_attr = ncdat.variables["lat"].__dict__
                psl_attr = ncdat.variables["psl"].__dict__
                glo_attr = ncdat.__dict__

            del psl_attr["_FillValue"]  # causes issues when saving

        else:
            psl, = nct.get_arrays(nc_files, [], ["psl"])

        # Reshape with singleton time dimension for later climatological mean:
        psl_ymeans.append(np.reshape(np.mean(psl, axis=0),
                                     (1, *np.shape(psl[0,:,:]))))

    print(f"Calculating {cmd.year_range[0]}-{cmd.year_range[1]} climatology")
    psl_clim = np.mean(np.concatenate(psl_ymeans), axis=0)

    # Save to NetCDF in the 'miscellaneous' data directory
    #
    # Set the dimensions, variables and attributes
    # as required by nct.save_netcdf():
    #
    nc_dims = {"lat": {"size": len(lat)}, "lon": {"size": len(lon)}, "bnds": {"size": 2}}

    nc_vars = {"lon"     : {"data": lon     , "dims": ("lon")        , "attr": lon_attr},
               "lon_bnds": {"data": lon_bnds, "dims": ("lon", "bnds"), "attr": {}},
               "lat"     : {"data": lat     , "dims": ("lat")        , "attr": lat_attr},
               "lat_bnds": {"data": lat_bnds, "dims": ("lat", "bnds"), "attr": {}},
               "psl_clim": {"data": psl_clim, "dims": ("lat", "lon") , "attr": psl_attr}}

    # Save the climatology:
    nct.save_netcdf(f"psl_clim_{cmd.year_range[0]}-{cmd.year_range[1]}.nc", nc_dims, nc_vars,
                    nc_global_attr=glo_attr, dir_save=Path(cfg.data_path[f"misc"]))


if __name__ == "__main__":
    main()
