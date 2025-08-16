"""Loading AMSR-E/AMSR2 sea ice extent/area/concentration datasets
which have been calculated on the native or CICE grid.
"""

from datetime import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np

from ..io import config as cfg
from . import nc_tools as nct


def _get_data_path(diagnostic="sie_cice_grid_pole_filled", frequency="daily"):
    """Get full path to directory containing netCDF files (include AMSR-E as
    links if not in same directory).
    """
    if "cice_grid" in diagnostic:
        return Path(cfg.data_path["amsr_cice_grid"], diagnostic, frequency)
    else:
        return Path(cfg.data_path["amsr_raw"], diagnostic, frequency)


def _get_file_fmt(diagnostic, frequency="daily"):
    """Get the file name of the netCDF files without the year."""

    fq = frequency[0].lower()  # short alias for frequency

    # Each diagnostic has a file formatted with the year
    # (same for each diagnostic, so append at the end):
    file_fmts = {
        "siconc"                       : f"siconc_{fq}",
        "siconc_cice_grid"             : f"siconc_cice_grid_{fq}",
        "siconc_cice_grid_pole_filled" : f"siconc_cice_grid_pole_filled_{fq}",
        "sie_cice_grid"                : f"sie_cice_grid_{fq}",
        "sie_cice_grid_pole_filled"    : f"sie_cice_grid_pole_filled_{fq}",
        "sie_reg_cice_grid"            : f"sie_reg_cice_grid_pole_filled_{fq}",
        "sie_reg_cice_grid_pole_filled": f"sie_reg_cice_grid_pole_filled_{fq}"
    }

    if diagnostic not in file_fmts.keys():
        raise KeyError("src.data.amsr: netCDF file format not defined for "
                       + f"diagnostic '{diagnostic}'")

    return file_fmts[diagnostic] + "_{}.nc"


def _get_nc_var_names(diagnostic, frequency="daily", regions=None):
    """Get the list of netCDF variable names."""

    fq = frequency[0].lower()  # short alias for frequency

    nc_var_names = {
        "siconc"                       : [f"siconc"],
        "siconc_cice_grid"             : [f"siconc"],
        "sie_cice_grid_pole_filled"    : [f"sie"],
        "sie_reg_cice_grid_pole_filled": [f"sie_{x}" for x in regions]
    }

    if diagnostic not in nc_var_names.keys():
        raise KeyError("src.data.amsr: netCDF variable names not defined for "
                       + f"diagnostic '{diagnostic}'")

    return nc_var_names[diagnostic]


def _get_nc_coord_var_names(diagnostic):
    """Get list of netCDF coordinate variable names (spatial coordinates,
    usually empty unless siconc).
    """

    nc_coord_names = {"siconc"                      : ["lon", "lat"],
                      "siconc_cice_grid"            : ["lon", "lat"],
                      "siconc_cice_grid_pole_filled": ["lon", "lat"]}

    if diagnostic not in nc_coord_names.keys():
        return []
    else:
        return nc_coord_names[diagnostic]


def load_data(diagnostic, frequency="daily", remove_leap=True, regions=None,
              dt_range=(dt(2002,1,1,12,0), dt(2023,12,31,12,0))):
    """Load AMSR-E/AMSR2 data.


    Parameters
    ----------
    diagnostic : str
        The name of the diagnostic to load.


    Optional parameters
    -------------------
    frequency : str, default = 'daily'
        The frequency of data to load (either 'daily' or 'monthly').

    remove_leap : bool, default = True
        Whether to remove all instances of February 29 from data, if
        frequency == 'daily'. This reduces the size of the returned arrays
        (as opposed to setting such values to NaN).

    regions : list of str or None (default)
        Regions to load if applicable (e.g., sea ice extent). If None,
        gets from config.

    dt_range : length-2 tuple of datetime.datetime
        The start and end dates of the data range to load. Default is all of
        2002 to 2023.


    Returns
    ------
    date : array (nt,) of datetime.datetime
        The datetime coordinates of the data.

    data_coord_vars : possibly-empty list of array
        If applicable, coordinate variables of the data.

    data : list of array (nt,*)
        The data.

    """

    if regions is None:
        regions = cfg.reg_nc_names

    data_path = _get_data_path(diagnostic, frequency)
    file_fmt = _get_file_fmt(diagnostic, frequency)

    if diagnostic in ["siconc"] and frequency == "daily":
        
        # For raw sea ice conc. (i.e., not on CICE grid) data is saved monthly
        
        # First year: start and end months:
        ms = dt_range[0].month
        me = 12 if dt_range[1].year > dt_range[0].year else (dt_range[1].month)

        data_files = [str(Path(data_path,
                               file_fmt.format(f"{dt_range[0].year}-{m:02}")))
                      for m in range(ms, me+1, 1)]

        if me == 12:
            if dt_range[1].year > dt_range[0].year + 1:
                # Middle years (all months):
                for y in range(dt_range[0].year+1, dt_range[1].year):
                    data_files += [str(Path(data_path,
                                            file_fmt.format(f"{y}-{m:02}")))
                                   for m in range(1, 13)]

            # Last year (will always include January):
            ms = 1
            me = dt_range[1].month

            data_files += [str(Path(data_path,
                                    file_fmt.format(f"{dt_range[1].year}-{m:02}")))
                           for m in range(ms, me+1, 1)]

        if len(data_files) > 1:
            
            if dt_range[0].year == dt_range[1].year:
                print(f"Loading: {str(data_path)}/"
                      + file_fmt.format(f"{dt_range[0].year}-" + "{"
                                        + f"{dt_range[0].month:02}.."
                                        + f"{dt_range[1].month:02}" +"}"))
            else:
                print(f"Loading: {str(data_path)}/"
                      + file_fmt.format("{" + f"{dt_range[0].year}-"
                                        + f"{dt_range[0].month:02}.."
                                        + f"{dt_range[1].year}-"
                                        + f"{dt_range[1].month:02}" +"}"))
        else:
            print(f"Loading: {data_files[0]}")

    else:
        # For all other data, it is saved yearly

        data_files = [str(Path(data_path, file_fmt.format(y)))
                      for y in range(dt_range[0].year, dt_range[1].year+1)]

        if len(data_files) > 1:
            print(f"Loading: {str(data_path)}/"
                + file_fmt.format("{"+f"{dt_range[0].year}.."
                                  + f"{dt_range[1].year}"+"}"))
        else:
            print(f"Loading: {data_files[0]}")

    t_name, t_units, t_calendar = nct.get_nc_time_props(data_files[0])

    nc_vars = _get_nc_var_names(diagnostic, frequency, regions=regions)
    nc_vars = [t_name] + nc_vars
    nc_coord_vars = _get_nc_coord_var_names(diagnostic)

    data = nct.get_arrays(data_files, nc_coord_vars, nc_vars)

    if len(nc_coord_vars) > 0:
        data_coord_vars = [x.copy() for x in data[:len(nc_coord_vars)]]
        data = data[len(nc_coord_vars):]
    else:
        data_coord_vars = []

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                              calendar=t_calendar))

    data = data[1:]

    jt = [(x >= dt_range[0]) & (x <= dt_range[1]) for x in date]
    date = date[jt]
    data = [data[j][jt] for j in range(len(data))]

    if remove_leap and frequency != "monthly":
        jt = [not (x.day==29 and x.month==2) for x in date]
        date = date[jt]
        data = [data[j][jt] for j in range(len(data))]

    return date, data_coord_vars, data

