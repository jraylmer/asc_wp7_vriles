"""Generate region masks files. The region boundaries are defined in the script.
"""

from pathlib import Path
from datetime import datetime as dt

import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import atmo, cice, nc_tools


# Define 'regions_name', prefix for output file name, and the regions themselves
regions_name = "regions_seas"

# Region definitions: each key is the name of a region, values of which are
# dictionaries containing 'min_lon', 'max_lon', 'min_lat', and 'max_lat',
# which are primarily used for the netCDF metadata and denote bounds of the
# regions. The actual regions are generated from data under the key 'bboxes',
# which is a list of lists of length 4. Each length 4 list is a bounding box
# of the form [lon_0, lon_1, lat_0, lat_1], i.e., spanning the range from
# longitudes lon_0 to lon_1 and latitudes lat_0 to lat_1. In most cases
# there is just one list (but it still needs to be wrapped within an outer
# list), but multiple bboxes allow slightly more complex regions to be
# constructed. Here it is used for the 'gin_seas' region, to handle the
# longitude wrapping as this region passes the date line.
#
regions = {
    "pan_Arctic": {
        "min_lon":  0., "max_lon": 360.,
        "min_lat": 45., "max_lat":  90.,
        "bboxes" : [[0., 360., 45., 90.]]},
    "gin_seas": {
        "min_lon": 315., "max_lon": 20.,
        "min_lat":  60., "max_lat": 90.,
        "bboxes" : [[315., 360., 60., 90.], [0., 20., 60., 90.]]},
    "barents_sea": {
        "min_lon": 20., "max_lon": 60.,
        "min_lat": 65., "max_lat": 90.,
        "bboxes" : [[20., 60., 65., 90.]]},
    "kara_sea": {
        "min_lon": 60., "max_lon": 100.,
        "min_lat": 65., "max_lat":  90.,
        "bboxes" : [[60., 100., 65., 90.]]},
    "laptev_sea": {
        "min_lon": 100., "max_lon": 150.,
        "min_lat":  65., "max_lat":  90.,
        "bboxes" : [[100., 150., 65., 90.]]},
    "east_siberian_chukchi_seas": {
        "min_lon": 150., "max_lon": 200,
        "min_lat":  65., "max_lat":  90.,
        "bboxes" : [[150., 200., 65., 90.]]},
    "beaufort_sea": {
        "min_lon" : 200., "max_lon" : 245.,
        "min_lat" :  65., "max_lat" :  90.,
        "bboxes"  : [[200., 245., 65., 90.]]},
    "canadian_archipelago": {
        "min_lon" : 245., "max_lon" : 275.,
        "min_lat" :  60., "max_lat" :  90.,
        "bboxes"  : [[245., 275., 60., 90.]]},
    "labrador_sea": {
        "min_lon" : 275., "max_lon" : 315.,
        "min_lat" :  60., "max_lat" :  90.,
        "bboxes"  : [[275., 315., 60., 90.]]}
}


def main():

    prsr = script_tools.argument_parser(usage="Create region masks on CICE grid")
    prsr.add_argument("-g", "--grid", type=int, default=129, choices=[129,131],
                      help="Generate on the inner (129) or full (131) domain")
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Missing and valid values for outputs:
    missing_value = 0
    valid_value   = 1

    # We can generate region masks for either the 'inner' grid (CICE history
    # outputs, with ny = 129 grid points) or on the 'full' domain that atmospheric
    # forcing is provided on (ny = 131 grid points). We need to infer a land mask,
    # so we load some arbitrary history or atmospheric forcing data to do this
    # and get the coordinates (lon/lat):
    #
    if cmd.grid == 129:

        # Load CICE coordinate data; we just want the lon/lats
        # on the inner domain (ny = 129):
        lon, lat = cice.get_grid_data(["TLON", "TLAT"])

        # Load some arbitary CICE data; we need a field to infer the land mask:
        field = cice.get_history_data(["aice_d"], dt_min=dt(1980, 1, 1),
            dt_max=dt(1980, 1, 2), frequency="daily", set_miss_to_nan=False)[1][0][0,:,:]

        lmask = np.where(field > 1E10, missing_value, valid_value).astype(np.int32)

        # For netCDF output coordinate variable names/attributes:
        lon_name      = "TLON"
        lat_name      = "TLAT"
        lon_long_name = "T-grid center longitude"
        lat_long_name = "T-grid center latitude"

    else:

        # Load some arbitrary atmospheric forcing data (which is on the
        # full ny = 131 domain), as we need the coordinates and to infer
        # the land mask. Specifically we load wind, as that is set to 0
        # over land. This method is a bit dodgy, but it does work:
        _, lon, lat, field = atmo.get_atmospheric_forcing_data("u10",
            dt(1980,1,1), dt(1980,1,2), slice_to_cice_grid=False)
 
        lmask = np.where(abs(field[0,:,:]) <= 1.E-5,
                         missing_value, valid_value).astype(np.int32)

        # For netCDF output coordinate variable names/attributes:
        lon_name      = "nav_lon"
        lat_name      = "nav_lat"
        lon_long_name = "longitude"
        lat_long_name = "latitude"

    ny, nx = np.shape(lon)
    lon = lon % 360.  # ensure range for bounds checking below

    region_names = list(regions.keys())
    region_prefix = "mask_"  # + name of region

    n_regions = len(region_names)

    # Set ocean points to missing if they are not in the specified region:
    region_masks = np.zeros((n_regions, ny, nx))

    for k in range(n_regions):
        # Sub-regions: a list of lists, each sub-list contains four integers
        # corresponding to lon. min., max., lat. min., max., respectively:
        sub_reg = regions[region_names[k]]["bboxes"]

        # Start with basic land mask:
        reg_k = lmask.copy()

        # Go over each sub-region and check each location is valid for that
        # sub-region and land mask:
        sub_check = valid_value * np.ones((len(sub_reg),ny,nx))

        for j in range(len(sub_reg)):
            sub_check[j,:,:] = np.where(  (lon >= sub_reg[j][0])
                                        & (lon <= sub_reg[j][1])
                                        & (lat >= sub_reg[j][2])
                                        & (lat <= sub_reg[j][3])
                                        & (lmask == valid_value),
                                        valid_value, missing_value)

        region_masks[k,:,:] = np.logical_and(np.any(sub_check, axis=0), lmask)


    # Save to netCDF: prepare data in format required by nc_tools function:
    nc_global_attr = {"title": f"Region masks for CICE 104x{cmd.grid} NHemi. grid"}

    nc_dims = {"y": {"size": ny}, "x": {"size": nx}}

    nc_vars = {}
    nc_vars[lon_name] = {"data": lon, "dims": ("y", "x"),
                         "attr": {"long_name"    : lon_long_name,
                                  "standard_name": "longitude",
                                  "units"        : "degrees_east"}}
    nc_vars[lat_name] = {"data": lat, "dims": ("y", "x"),
                         "attr": {"long_name"    : lat_long_name,
                                  "standard_name": "latitude",
                                  "units"        : "degrees_north"}}

    for k in range(n_regions):
        r_name = region_names[k]
        nc_vars[f"{region_prefix}{r_name}"] = {
            "data": region_masks[k,:,:], "dims": ("y", "x"),
            "attr": {"coordinates"  : f"{lon_name} {lat_name}",
                     "comment"      :   f"{missing_value} = land or missing, "
                                      + f"{valid_value} = ocean and in region",
                     "missing_value": missing_value,
                     "min_longitude": regions[r_name]["min_lon"],
                     "max_longitude": regions[r_name]["max_lon"],
                     "min_latitude" : regions[r_name]["min_lat"],
                     "max_latitude" : regions[r_name]["max_lat"]}
        }

    nc_tools.save_netcdf(f"{regions_name}_masks_104x{cmd.grid}.nc", nc_dims,
                         nc_vars, nc_global_attr=nc_global_attr,
                         sort_attr=False, compress=False,
                         dir_save=cfg.data_path[f"regs{cmd.grid}"].parent)


if __name__ == "__main__":
    main()
