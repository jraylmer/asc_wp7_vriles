"""Calculate SSM/I sea ice extent in different regions of the CICE domain. The
bash script to interpolate SSM/I raw data onto the CICE grid (same directory)
generates the required input data to this script.
"""

from datetime import datetime as dt
from pathlib import Path

import numpy as np

from src import script_tools
from src.io import config as cfg
from src.data import cice, ssmi, nc_tools
from src.diagnostics import diagnostics as diag


def main():

    prsr = script_tools.argument_parser(usage="Calculate SIE in SSM/I on CICE grid")
    prsr.add_argument("-y", "--year-range", type=int,nargs=2, default=[1979]*2)
    prsr.add_argument("-d", "--dataset", type=str, default="nt", choices=["nt", "bt"])
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Sea ice extent is saved in a separate file, 1 per year, with all regional
    # calculations which includes pan Arctic
 
    # Attributes for time (variable name, and attributes including "units" and
    # "calendar" which are used to determine datetime values:
    nc_t_name = "time"
    nc_t_attr = {"bounds"       : f"{nc_t_name}_bnds",
                 "calendar"     : "365_day",
                 "long_name"    : "time",
                 "standard_name": "time",
                 "units"        : "days since 1979-01-01"}   

    # Attributes for sea ice extent (same for all regions):
    nc_sie_attr = {"cell_methods": f"{nc_t_name}: mean",
                   "long_name": "Sea ice extent",
                   "standard_name": "sea_ice_extent", "units": "1e6 km2"}

    # Global attributes in all files:
    nc_global_attr = {"title": "SSM/I sea ice extent calculated on CICE grid"}
    for x in ["author", "contact", "institution"]:
        if getattr(cfg, x) != "":
            nc_global_attr[x] = getattr(cfg, x)

    if cfg.title != "":
        nc_global_attr["comment"] = cfg.title

    if cmd.dataset == "nt":
        nc_global_attr["source"] = ("DiGirolamo, N., C. L. Parkinson, D. J. "
            + "Cavalieri, P. Gloersen, and H. J. Zwally. (2022). Sea Ice "
            + "Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS "
            + "Passive Microwave Data, Version 2 [Data Set]. Boulder, "
            + "Colorado USA. NASA National Snow and Ice Data Center "
            + "Distributed Active Archive Center. "
            + "https://doi.org/10.5067/MPYG15WAA4WX. Date Accessed "
            + "04-15-2024.")
    else:
        nc_global_attr["source"] = ("Comiso, J. C. (2023). Bootstrap Sea Ice "
            + "Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS, "
            + "Version 4 [Data Set]. Boulder, Colorado USA. NASA National "
            + "Snow and Ice Data Center Distributed Active Archive Center. "
            + "https://doi.org/10.5067/X5LG68MH013O. Date Accessed "
            + "04-15-2024.")

    # Load CICE grid data (grid cell latitudes 'TLAT' and areas 'tarea'
    # and region masks):
    tarea, lat = cice.get_grid_data(["tarea", "TLAT"])
    regions    = cice.get_region_masks()  # tuple of array
    n_regions  = len(regions)

    # Two netCDF dimensions (time and bnd), same for all so prepare first:
    nc_dims = {}
    nc_dims[nc_t_name] = {"size": None}
    nc_dims["bnd"]  = {"size": 2}

    for y in range(cmd.year_range[0], cmd.year_range[1]+1, 1):

        # Create new datetimes and bounds save time and time_bnds into
        # dictionary for daily and monthly:
        time = {}
        time_bnds = {}

        date_d, date_bnds_d, time["d"], time_bnds["d"] = \
            nc_tools.dt_daily(y, nc_units=nc_t_attr["units"],
                              nc_calendar=nc_t_attr["calendar"])

        date_m, date_bnds_m, time["m"], time_bnds["m"] = \
            nc_tools.dt_monthly(y, nc_units=nc_t_attr["units"],
                                nc_calendar=nc_t_attr["calendar"])

        # Load SSM/I sea ice concentration on CICE grid data:
        _, _, data_ssmi = ssmi.load_data("aice", frequency="daily",
            which_dataset=cmd.dataset, dt_range=(dt(y, 1, 1), dt(y, 12, 31, 23)))

        aice = data_ssmi[0]

        # Some days sea ice concentration is completely missing, but this is
        # lost when calculating extent as it uses np.nanmean(), which leads to
        # sie being set to 0. Identify the indices of such days so that missing
        # values can be added back. It does not need to be done by region:
        jd_miss = np.all(np.isnan(aice), axis=(1,2))

        # Sea ice extent needs to include the pole hole which is currently
        # masked out (set to NaN) in the data. Set such locations to 1 so that
        # they count as 'ice covered' and hence contribute to extent. We assume
        # that such regions are always occupied by ice of concentration >= 15%.
        # Note that this is mainly only relevant up to around 2008 where the pole
        # hole is relatively large, but even then it is the region poleward of
        # 85N. After that it gets smaller (around 89N):
        aice = np.where(np.isnan(aice) & (lat > 85.), 1., aice)

        # Save diagnostics for each region into this dictionary,
        # again for loop calling nc_tools.save_netcdf() below:
        data_out = {}
        for j in range(n_regions):

            r = cfg.reg_nc_names[j]

            data_out[f"sie_d_{r}"] = diag.sea_ice_extent(aice, tarea,
                                                         mask=regions[j])

            # Add complete missing days back:
            data_out[f"sie_d_{r}"][jd_miss] = np.nan

            # Calculate monthly means from daily means:
            _, data_out[f"sie_m_{r}"] = diag.monthly_mean(date_d,
                                                          data_out[f"sie_d_{r}"],
                                                          date_bnds_m)

        for t in "dm":

            nc_vars = {}
            nc_vars[nc_t_name] = {"data": time[t], "dims": (nc_t_name,),
                                  "attr": nc_t_attr}

            nc_vars[f"{nc_t_name}_bnds"] = {"data": time_bnds[t],
                                            "dims": (nc_t_name,"bnd")}

            for r in cfg.reg_nc_names:
                nc_vars[f"sie_{t}_{r}"] = {"data": data_out[f"sie_{t}_{r}"],
                                           "dims": (nc_t_name,),
                                           "attr": nc_sie_attr}

            nc_tools.save_netcdf(f"sie_{t}_{y}.nc", nc_dims, nc_vars,
                nc_global_attr=nc_global_attr, sort_attr=True,
                dir_save=Path(cfg.data_path[f"ssmi_{cmd.dataset}_cice"],
                              "sea_ice_extent",
                              "daily" if t=="d" else "monthly"))


if __name__ == "__main__":
    main()

