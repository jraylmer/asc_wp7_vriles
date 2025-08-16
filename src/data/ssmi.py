"""Provide functions for loading SSM/I sea ice extent/area/concentration datasets
which have been calculated on the sea ice model grid.
"""

from datetime import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np

from ..io import config as cfg
from . import nc_tools as nct


_dataset_id = {"nt": "NSIDC-0051_nasateam_v2", "bt": "NSIDC-0079_bootstrap_v4"}


def _get_data_path(diagnostic="sie_regional", frequency="daily",
                   which_dataset="nt"):
    """Get full path to directory containing netCDF files.
    """
    return Path(cfg.data_path[f"ssmi_{which_dataset}"], diagnostic, frequency)


def _get_file_fmt(diagnostic, frequency="daily", which_dataset="nt"):
    """Get the file name of the netCDF files without the year.
    """

    # Each diagnostic has a file formatted with the year
    # (same for each diagnostic, so append at the end):
    file_fmts = {"siconc"      : f"{_dataset_id[which_dataset]}_{frequency}",
                 "sie"         : f"sie_{frequency}",
                 "sie_regional": f"sie_regional_{frequency}"}

    if diagnostic not in file_fmts.keys():
        raise KeyError("src.data.ssmi: netCDF file format not defined for "
                       + f"diagnostic '{diagnostic}'")

    return file_fmts[diagnostic] + "_{}.nc"


def _get_nc_var_names(diagnostic, frequency="daily", regions=None):
    """Get the list of netCDF variable names -- regional versions, if applicable.
    """

    if regions is None:
        regions = cfg.reg_nc_names

    nc_var_names = {"siconc"      : ["siconc"],
                    "sie"         : ["sie"],
                    "sie_regional": [f"sie_{x}" for x in regions]}

    if diagnostic not in nc_var_names.keys():
        raise KeyError("src.data.ssmi: netCDF variable name not defined for "
                       + f"diagnostic '{diagnostic}'")

    return nc_var_names[diagnostic]


def _get_nc_coord_var_names(diagnostic):
    """Get list of netCDF coordinate variable names (spatial coordinates,
    usually empty unless diagnostic == 'siconc').
    """

    nc_coord_names = {"siconc": ["TLON", "TLAT"]}

    if diagnostic not in nc_coord_names.keys():
        return []
    else:
        return nc_coord_names[diagnostic]


def load_data(diagnostic, frequency="daily", which_dataset="nt",
              remove_leap=True, regions=None,
              dt_range=(dt(1979,1,1,12,0), dt(2023,12,31,12,0))):
    """Load SSM/I data.


    Parameters
    ----------
    diagnostic : str
        The name of the diagnostic to load.


    Optional parameters
    -------------------
    frequency : str, default = 'daily'
        The frequency of data to load (either 'daily' or 'monthly').

    which_dataset : str, default = 'nt'
        Either 'nt' or 'bt' for the NASA Team and Bootstrap datasets,
        respectively.

    remove_leap : bool, default = True
        Whether to remove all instances of February 29 from data, if
        frequency == 'daily'. This reduces the size of the returned arrays
        (as opposed to setting such values to NaN).

    regions : list of str or None (default)
        Regions to load if applicable (e.g., sea ice extent). If None,
        gets from config.

    dt_range : length-2 tuple of datetime.datetime
        The start and end dates of the data range to load. Default is all of
        1979 to 2023.


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

    data_path = _get_data_path(diagnostic, frequency, which_dataset)
    file_fmt = _get_file_fmt(diagnostic, frequency, which_dataset)

    # Data is saved yearly:
    data_files = [str(Path(data_path, file_fmt.format(y)))
                  for y in range(dt_range[0].year, dt_range[1].year+1)]

    t_name, t_units, t_calendar = nct.get_nc_time_props(data_files[0])
    
    nc_vars = _get_nc_var_names(diagnostic, frequency, regions=regions)
    nc_vars = [t_name] + nc_vars
    nc_coord_vars = _get_nc_coord_var_names(diagnostic)

    print(f"Loading: {str(data_path)}/"
          + file_fmt.format("{"+f"{dt_range[0].year}..{dt_range[1].year}"+"}"))

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


def fill_missing_data(date, data):
    """Linear interpolation of missing data, only for the earlier years that
    are every two days, hence converting to daily time series. Note: does not
    touch the large gap between December 1987 and January 1988; only
    interpolates on a missing day when both surrounding days have data. Note
    also that start/end values are never interpolated.


    Parameters
    ----------
    data : array (nt,) or length n_datasets list of array (nt,)
        The dataset(s) to interpolate. These are assumed to be daily data with
        missing values set to NaN, but no specific checks are done in this
        function.


    Returns
    -------
    data_out : array (nt,) or length n_datasets list of array (nt,)
        The interpolated datasets(s).

    """

    if type(data) in [np.ndarray]:
        data_out = [data]
        unlist = True
    elif type(data) in [list]:
        data_out = data
        unlist = False
    else:
        raise TypeError(f"Cannot handle input data type: {repr(type(data))}")

    n_datasets = len(data)

    def _interp(xm, xp):
        """Linear interpolation for xm < x < xp."""
        return xm + .5*(xp-xm)

    # Loop over all data except start and end:
    for t in range(1, len(data) - 1):
        # Only interpolate missing data using surrounding
        # points (if multiple missing in a row, interpolation
        # doesn't work):
        for d in range(n_datasets):
            if np.isnan(data_out[d][t]):
                data_out[d][t] = _interp(data_out[d][t-1], data_out[d][t+1])

    if unlist:
        data_out = data_out[0]

    return data_out

