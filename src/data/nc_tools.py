"""General functions for dealing with netCDF data.
"""

import calendar
from datetime import datetime as dt, timedelta
from pathlib import Path
import warnings

import netCDF4 as nc
import numpy as np


def cftime_to_datetime(dt_cf):
    """Convert list or array of cf datetimes to regular python datetimes.
    """
    return np.array([dt(x.year, x.month, x.day, x.hour, x.minute,
                        x.second, x.microsecond) for x in dt_cf])


def dt_daily(year, no_leap=True, hour_offset=12, dt_range=None,
             nc_units="days since 1979-01-01", nc_calendar="365_day"):
    """Calculate regular python datetime values for each day of a specified
    year and the corresponding datetime bounds, and the corresponding time
    and time bounds values for use in netCDF outputs.


    Parameters
    ----------
    year : int
        The year to calculate datetimes for.


    Optional parameters
    -------------------
    no_leap : bool, optional, default = True
        If True, excludes February 29 if year is a leap year.

    hour_offset : int, optional, default = 12
        Hour of day for datetimes (default is to have the value at the center
        of each day). This parameter must be in the range [0, 23]. Note also
        that the bounds for each are always 00:00 of the current day and
        00:00 of the next day.

    dt_range : length-2 iterable of datetime.datetime or None (default)
        The overall range of datetime to include. The year is replaced with
        year; this option is used to return a subset of the entire year's
        worth of coordinates cut off at the beginning and/or end. If None
        (default), this does has no effect.

    nc_units : str, default = 'days since 1979-01-01'
        The CF units string that converts dates into times.

    nc_calendar : str, default = '365_day'
        The CF calendar string that converts dates into times.


    Returns
    -------
    date : array (nt,) of datetime.datetime
        The datetime values.

    date_bnds : array (nt,2) of datetime.datetime
        The datetime bounds for each value in date.

    time : array (nt,) of float
        The time values corresponding to date with specified units and
        calendar.

    time_bnds : array (nt,2) of float
        The time values corresponding to date_bnds with specified units
        and calendar.

    """

    # Ensure hours to offset datetime by is valid:
    hour_offset = min(24, hour_offset)
    hour_offset = max(hour_offset, 0)

    date = []
    date_bnds = []

    for m in range(1, 13):
        # Loop over days of current month; the second argument returned by the
        # calendar.monthrange() function gives the number of days for a
        # specifed month of the specified year (including leap days):
        for d in range(1, calendar.monthrange(year, m)[1]+1):

            # Lower bound is the current year, month, and day, at 00:00:
            dt_1 = dt(year, m, d, 0, 0, 0, 0)

            # Add exactly one day to get the upper bound:
            dt_2 = dt_1 + timedelta(days=1)

            date_bnds.append([dt_1, dt_2])

            # The datetime for the data point is found by adding some hours
            # (by default 12 for the centre of the day) to the lower bound:
            date.append(dt_1 + timedelta(hours=hour_offset))

    # Convert to arrays:
    date = np.array(date)
    date_bnds = np.array(date_bnds)

    if no_leap and calendar.isleap(year):
        is_feb29 = np.array([(x.day == 29) and (x.month == 2)
                             for x in date_bnds[:,0]])

        date = date[~is_feb29]
        date_bnds = date_bnds[~is_feb29,:]

        # Also update the upper bound of Feb 28 to March 1 00:00
        # (will have initially been set to Feb 29 00:00):
        is_feb28 = np.array([(x.day == 28) and (x.month == 2)
                             for x in date_bnds[:,0]])

        date_bnds[is_feb28,1] += timedelta(days=1)

    # Subset to required dt range, if needed:
    if dt_range is not None:

        jt  = (date >= dt_range[0].replace(year=year)) \
            & (date <= dt_range[1].replace(year=year))

        date = date[jt]
        date_bnds = date_bnds[jt,:]

    # Calculate the corresponding time coordinates:
    time      = nc.date2num(date     , units=nc_units, calendar=nc_calendar)
    time_bnds = nc.date2num(date_bnds, units=nc_units, calendar=nc_calendar)

    return date, date_bnds, time, time_bnds


def dt_monthly(year_start, n_years=1, dt_range=None,
               nc_units="days since 1979-01-01", nc_calendar="365_day",
               offset="center"):
    """Calculate regular python datetime values for each month of a specified
    year, range of years, and overall datetime range, and the corresponding
    datetime bounds, and corresponding time and time bounds for use in netCDF
    outputs.


    Parameters
    ----------
    year_start : int
        First (and possibly only) year to compute monthly datetime values for.


    Optional parameters
    -------------------
    n_years : int >= 1, default = 1
        Number of years, including year_start, to compute for.

    dt_range : length-2 iterable of datetime.datetime or None (default)
        The overall range of datetime to include. This option is used to
        effectively alter the start date to not be January of year_start
        and the end date to not be December of year_start + (n_years - 1).
        If None (default), this has no effect.

    nc_units : str, default = 'days since 1979-01-01'
        The CF units string that converts dates into times.

    nc_calendar : str, default = '365_day'
        The CF calendar string that converts dates into times.


    Returns
    -------
    date : array (nt,) of datetime.datetime
        The datetime values at the centre of bounds.

    date_bnds : array (nt,2) of datetime.datetime
        The datetime bounds for each value in date.

    time : array (nt,) of float
        The time values corresponding to date with specified units and
        calendar.

    time_bnds : array (nt,2) of float
        The time values corresponding to date_bnds with specified units
        and calendar.

    """

    date = []
    date_bnds = []

    for y in range(year_start, year_start + n_years, 1):
        for m in range(1, 13):
            # Lower bound is the current year, month, day 01 at 00:00:
            dt_1 = dt(y, m, 1, 0, 0, 0, 0)

            # Upper bound is 00:00 on the first day of the next month, which,
            # if we are in December, is January 01 of the next year:
            dt_2 = dt(y + (m == 12), 1 if m == 12 else (m+1), 1, 0, 0, 0, 0)

            date_bnds.append([dt_1, dt_2])

            # The datetime for the data point is at the centre of the bounds:
            date.append(dt_1 + .5*(dt_2 - dt_1))

    # Convert to arrays:
    date = np.array(date)
    date_bnds = np.array(date_bnds)

    # Subset to required dt range, if needed:
    if dt_range is not None:
        jt  = (date >= dt_range[0]) & (date <= dt_range[1])
        date = date[jt]
        date_bnds = date_bnds[jt,:]

    # Calculate the corresponding time coordinates:
    time      = nc.date2num(date     , units=nc_units, calendar=nc_calendar)
    time_bnds = nc.date2num(date_bnds, units=nc_units, calendar=nc_calendar)

    return date, date_bnds, time, time_bnds


def get_nc_time_props(ncfile, possible_names=["t", "time"]):
    """Determine the name, units and calendar of the time variable in a
    specified NetCDF file (input as string or pathlib.Path instance).
    Returns the name, units and calendar attribute (all str) of the time
    variable. If the time variable cannot be identified, or if the
    attribute(s) does not exist, returns None in place of each (warnings are
    raised).
    """

    with nc.Dataset(ncfile, mode="r") as ncdat:

        # Determine the variable name of the time coordinate. Raise warning if it
        # does not exist and return None for each attribute:
        for name in possible_names:
            if name in ncdat.variables:
                nc_t_name = name
                break
        else:
            nc_t_name = None
            warnings.warn(f"Unable to identify time variable in file {ncfile}")

        if nc_t_name is None:
            t_units = None
            t_calendar = None
        else:
            if hasattr(ncdat.variables[nc_t_name], "units"):
                t_units = ncdat.variables[nc_t_name].units
            else:
                t_units = None
                warnings.warn(f"Variable '{nc_t_name}' has no units in {ncfile}")

            if hasattr(ncdat.variables[nc_t_name], "calendar"):
                t_calendar = ncdat.variables[nc_t_name].calendar
            else:
                t_calendar = None
                warnings.warn("Variable '{nc_t_name}' has no calendar in {ncfile}")

    return nc_t_name, t_units, t_calendar


def get_arrays(ncfiles, coordinate_vars=[], diagnostic_vars=[],
               np_concatenate_kwargs={}):
    """
    Load an arbitrary set of coordinate variables and variables from a set of
    NetCDF data files into NumPy arrays.

    Variables specified in coordinate_vars are loaded from one file only and
    not concatenated, while those specified in diagnostic_vars are loaded from
    all files and are concatenated (along axis=0 by default).


    Parameters
    ----------
    ncfiles : list of str or pathlib.Path
        List of paths to each NetCDF file. Note that files are loaded and
        diagnostic variables (see below) are concatenated in the order of this
        list. There is no checking of time coordinates.

    coordinate_vars : list of str
        Names (matching variable name in the NetCDF files) of the coordinates
        to load from the ncfiles[0]. This list is for non-concatenated variables
        (latitude, pressure level, etc., although note that the time, if
        required, should be put as a diagnostic variable).

    diagnostic_vars : list of str
        Names (matching variable name in the NetCDF files) of the diagnostic
        variables to be loaded from all ncfiles and concatenated. These are
        concatenated in the order of ncfiles.


    Optional parameters
    -------------------
    np_concatenate_kwargs : dict, default = {}

        Keyword arguments passed to NumPy.concatenate(). Note that the default
        concatenation axis, 0, usually corresponds to time and thus, usually,
        does not need to be changed.


    Returns
    -------
    [coordinate_var_1, ...,] [diagnostic_var_1, ...]

    NumPy arrays of the requested coordinate variables, in the order specified
    by coordinate_vars, followed by the requested diagnostic variables, in the order
    specified by diagnostic_vars.


    Example
    -------
    Suppose the variable 't2m', as a function of time 't', latitude 'lat', and
    longitude 'lon', on a global 1 degree fixed grid, is stored across 5 NetCDF
    files containing monthly averages for one year each. This can be loaded using:

        >>> import numpy as np
        >>> import nc_tools as nct
        >>>
        >>> ncfiles = [f'./file{j+1}.nc' for j in range(5)]  # list of files
        >>>
        >>> lat, lon, t, t2m = nct.get_arrays(ncfiles, ['lat','lon'], ['t','t2m'])
        >>>
        >>> np.shape(lon)
        (360,)
        >>> np.shape(lat)
        (180,)
        >>> np.shape(time)
        (60,)
        >>> np.shape(t2m)
        (60, 180, 360)

    """

    n_cvars = len(coordinate_vars)
    n_dvars = len(diagnostic_vars)

    # Each array will be stored in a list (separately for
    # coordinates (cvar_arrays) and diagnostics (dvar_arrays):
    cvar_arrays = [None for j in range(n_cvars)]
    dvar_arrays = [None for j in range(n_dvars)]

    # Load the first dataset to get dimensions, coordinates, and initialise:
    with nc.Dataset(ncfiles[0], mode="r") as ncds_0:

        for k in range(n_cvars):
            cvar_arrays[k] = np.array(ncds_0.variables[coordinate_vars[k]])

        for k in range(n_dvars):
            dvar_arrays[k] = np.array(ncds_0.variables[diagnostic_vars[k]])

    # Loop over remaining datasets and concatenate the diagnostics:
    for j in range(1, len(ncfiles)):
        with nc.Dataset(ncfiles[j], mode="r") as ncds_j:
            for k in range(len(diagnostic_vars)):
                dvar_arrays[k] = np.concatenate((dvar_arrays[k],
                                                 np.array(ncds_j.variables[
                                                     diagnostic_vars[k]])),
                                                **np_concatenate_kwargs)

    return tuple(cvar_arrays + dvar_arrays)


def save_netcdf(filename, nc_dims, nc_vars, nc_global_attrs={},
                dir_save=Path.home(), nc_mode="w"):
    """General function for saving data to a netCDF file.


    Parameters
    ----------
    filename : str
        File name to save (e.g., 'my_file.nc').

    nc_dims : dict {'dim_1': {'size': <int> or None}, ['dim_2': ...]}
        NetCDF dimensions. Each key is the name of a dimension to create,
        and the values are dictionaries containing the key 'size', which
        must be set to either an integer representing the size of the
        dimension, or None (for an unlimited dimension; typically, time).
        At least one dimension must be defined.

    nc_vars : dict {'var_1': {'data': <array>,
                              'dims': tuple of <str>,
                              'attr': dict {<attribute>: <str>}},
                    ['var_2': {...}]}
        NetCDF variables. Each key is the name of a variable to create,
        and the values are dictionaries containing two mandatory keys:

            'data': array
                The data to save to this variable.

            'dims': tuple of str
                The names of the netCDF dimensions corresponding to the
                shape of data.

        and one optional key:
            'attr': dict
                The netCDF attributes ('units', 'standard_name', etc.,)
                for this variable. Can be empty or not present.


    Optional parameters
    -------------------
    nc_global_attrs : dict
        NetCDF global attributes for the file. Can be empty (default).

    dir_save : str or pathlib.Path
        Path to the directory to save data.

    nc_mode : str, default = 'w'
        Read/write Mode for opening the netCDF data. Default is 'w'
        (i.e., write and overwrite if file already exists).

    """

    if not filename.endswith(".nc"):
        filename += ".nc"

    Path(dir_save).mkdir(parents=True, exist_ok=True)

    with nc.Dataset(Path(dir_save, filename), nc_mode) as ncdat:

        for attr in nc_global_attrs.keys():
            ncdat.setncattr(attr, nc_global_attrs[attr])

        for dim in nc_dims.keys():
            ncdat.createDimension(dim, nc_dims[dim]["size"])

        for var in nc_vars.keys():
            ncdat.createVariable(var, nc_vars[var]["data"].dtype,
                                 nc_vars[var]["dims"])

            if "attr" in nc_vars[var].keys():
                for attr in nc_vars[var]["attr"].keys():
                    ncdat.variables[var].setncattr(attr, nc_vars[var]["attr"][attr])

            ncdat.variables[var][:] = nc_vars[var]["data"][:]

    print(f"Saved: {str(Path(dir_save, filename))}")

