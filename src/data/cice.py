"""Provide functions for reading in and pre-processing the ice model output
data ('history' diagnostics) and the post-processed diagnostics ('processed'
diagnostics).
"""

from datetime import datetime as dt, timedelta
from pathlib import Path
import warnings

import netCDF4 as nc
import numpy as np

from ..io import config as cfg
from . import nc_tools as nct


def slice_cice_data_to_atm_grid(lon, lat, cice_data):
    """Removes array slices from CICE data that do not correspond to
    coordinates in the atmospheric data. This is related to the 131x104 versus
    129x104 grid definitions that are used for atmospheric forcing data and by
    CICE internally, respectively. See function slice_atm_data_to_cice_grid()
    of module atmo.py for details.

    For pcolormesh plots where cell corners are required, data should be sliced
    by this function. It should then be plotted with ULON and ULAT directly from
    CICE. This is necessary due to a bug in which T cell bounds are not output
    correctly by CICE )at least, the version we have used), and the slicing to do
    this happens to be the same as that required for matching with atmospheric
    data. If matching to atmospheric data as well, a further slice of [1:,1:]
    should be taken of the CICE data (i.e., after a call to this function) and
    the corresponding cell corners for pcolormesh are ULON[1:,1:] and
    ULAT[1:,1:].

    Confused? See documentation of function slice_atm_data_to_cice_grid() in
    module atmo.py for more information.


    Parameters
    ----------
    lon, lat : array (129, 104)
        Longitude and latitude coordinates, respectively, loaded from CICE
        history or 'processed' data files.

    cice_data : ndarray of shape (nt, 129, 104)
        CICE data array, where the first axis corresponds to time.


    Returns
    -------
    lon, lat : array (128, 103)
        Sliced longitude and latitude arrays, respectively.

    cice_data : array (nt, 128, 103)
        Sliced CICE data array.

    """
    return lon[1:,1:], lat[1:,1:], cice_data[:,1:,1:]


def get_grid_data(fields=["ULON", "ULAT", "TLON", "TLAT", "tarea"],
                  slice_to_atm_grid=False):
    """Load grid data (coordinates and cell measures) into arrays. Takes one
    parameter 'fields#, which is an iterable of str containing the netCDF
    variable names. Default is:

        ['ULON', 'ULAT', 'TLON', 'TLAT', 'tarea']

    Returns the corresponding data arrays in the same order.

    Optional parameter: slice_to_atm_grid: bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid (see documentation of function
        slice_cice_data_to_atm_grid in this module).

    """

    grid_data = nct.get_arrays([cfg.data_path["grid"]], fields, [])

    if slice_to_atm_grid:

        # Tuple does not support item assignment; convert to list:
        grid_data = list(grid_data)

        for k in range(len(grid_data)):
            # Function expects 2D, 2D, 3D array for lon, lat, and data
            # respectively. Adapt it here for all 2D by just sending in the
            # first argument and the other two as dummy arrays of zeros:
            grid_data[k], _, _ = slice_cice_data_to_atm_grid(
                grid_data[k], np.zeros((2,2)), np.zeros((1,2,2)))

        # Convert back to tuple for returning:
        grid_data = tuple(grid_data)

    return grid_data


def get_region_masks(reg_nc_names=None, append_mask=True,
                     slice_to_atm_grid=False, ny=129):
    """Load specified CICE region masks from a specified mask netCDF file.


    Optional parameters
    -------------------
    reg_nc_names : list of str or None
        Names of the NetCDF variables in the masks file corresponding to the
        masks required. If None, gets this list from the config.

    append_mask : bool, default = True
        Whether to append names in region_nc_names to 'mask_', for the netCDF
        variable names (default).

    slice_to_atm_grid: bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid (see documentation of function
        slice_cice_data_to_atm_grid routine in this module). If ny != 129,
        this is ignored.

    ny : int, default = 129
        Load region masks for the ny x 104 grid where ny = 129 or 131.


    Returns
    -------
    mask_1, [mask_2, ...] : 2D arrays
        The mask arrays in the order corresponding to region_nc_names.

    """

    if reg_nc_names is None:
        reg_nc_names = cfg.reg_nc_names

    with nc.Dataset(cfg.data_path[f"regs{ny}"], mode="r") as ncdat:
        region_masks = [np.array(ncdat.variables[append_mask*"mask_" + k])
                        for k in reg_nc_names]

    if slice_to_atm_grid and ny == 129:
        
        # Tuple does not support item assignment; convert to list:
        region_masks = list(region_masks)

        for k in range(len(region_masks)):
            # Routine expects 2D, 2D, 3D array for lon, lat, and data
            # respectively. Adapt it here for all 2D by just sending in the
            # first argument and the other two as dummy arrays of zeros:
            region_masks[k], _, _ = slice_cice_data_to_atm_grid(
                region_masks[k], np.zeros((2,2)), np.zeros((1,2,2)))

        # Convert back to tuple for returning:
        region_masks = tuple(region_masks)

    if slice_to_atm_grid and ny != 129:
        warnings.warn("src.data.cice.region_masks: not slicing to "
                      + f"atmosphere grid since ny = {ny}")

    return region_masks


def get_history_data(nc_var_names, dt_min=dt(1980,1,1,12,0),
                     dt_max=dt(1980,12,31,12,0), frequency="daily",
                     offset_timedelta="none", set_miss_to_nan=True,
                     slice_to_atm_grid=False):
    """Load specified CICE ice history data (for daily or monthly data [FN1]).


    Parameters
    ----------
    nc_var_names : length-nv iterable of str
        The netCDF variable names to load.


    Optional parameters
    -------------------
    dt_min, dt_max : datetime.datetime
                     default = 1980/01/01 12:00
                               1980/12/31 12:00
        Start and end datetimes for which to load data. If frequency (below) is
        'daily', hour and minute are overwritten as 12 and 0 respectively, so
        that daily averages written at 12:00 are compared correctly with these
        limits.

    frequency : {'daily', 'monthly'}
        Frequency subset to load (default is 'daily').

    offset_timedelta : datetime.timedelta or 'auto'
        If 'auto', defaults to -12 hours (daily) or -15 days (monthly) which
        sets daily average time stamps to mid points of the averaging interval
        (CICE writes averages at the end of the time averaging interval), and
        monthly average time stamps to roughly the middle. Otherwise, a
        datetime.timedelta may be specified explicitly.

    set_miss_to_nan : bool, default = True
        Set missing values (land, in most cases) to numpy.nan.

    slice_to_atm_grid : bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid. [FN2]


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Time coordinates as datetimes.

    data : length nv list of array (nt,*)
        The specified processed diagnostic data.

    [FN3]


    Notes
    -----
    [FN1] For sub-daily output use get_subdaily_history_data() in this module.
    [FN2] See documentation of slice_cice_data_to_atm_grid() in this module.
    [FN3] Does not return spatial coordinates as those can be loaded from
          the get_grid_data() function in this module.

    """

    if frequency == "daily":
        dt_min = dt_min.replace(hour=12, minute=0)
        dt_max = dt_max.replace(hour=12, minute=0)

    nc_vars = ["time"] + nc_var_names

    # Determine list of netCDF files (as full paths):
    file_list = [str(Path(cfg.data_path[f"hist_{frequency[0]}"],
                          f"iceh.{frequency}" + f".{y:04}.nc"))
                 for y in range(dt_min.year, dt_max.year+1, 1)]

    if type(offset_timedelta) == str:
        if frequency == "daily":
            offset_timedelta = timedelta(hours=-12)
        elif frequency == "monthly":
            offset_timedelta = timedelta(days=-15)

    data = list(nct.get_arrays(file_list, [], nc_vars))

    # Get calendar and time units:
    _, t_units, t_cal = nct.get_nc_time_props(file_list[0])

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                  calendar=t_cal)) + offset_timedelta

    # Get time slice:
    jt = [(x >= dt_min) and (x <= dt_max) for x in date]
    date = date[jt]
    for k in range(1, len(data)):
        data[k] = data[k][jt]

    if slice_to_atm_grid:
        for k in range(1, len(data)):
            # We don't load lon/lat here so just use dummy
            # 2D coordinate arrays of zeros:
            _, _, data[k] = slice_cice_data_to_atm_grid(
                np.zeros((2,2)), np.zeros((2,2)), data[k])

    if set_miss_to_nan:
        for k in range(1, len(data)):
            data[k] = np.where(abs(data[k]) > 1E20, np.nan, data[k])

    data = data[1:]  # remove added time coordinate from data list

    return date, data


def get_subdaily_history_data(nc_var_names, dt_min, dt_max, seltime=1,
                              frequency="3hourly", hist_avg=False,
                              offset_timedelta=timedelta(hours=0),
                              set_miss_to_nan=True, slice_to_atm_grid=False):
    """Load specified CICE ice history data (high frequency output [FN1]). Raw
    output data from CICE is assumed to be combined into monthly files.


    Parameters
    ----------
    nc_var_names : length-nv iterable of str
        The netCDF variable names to load.

    dt_min, dt_max : datetime.datetime
        Start and end datetimes for which to load data. To avoid saving very
        large arrays into memory these parameters are required.


    Optional parameters
    -------------------
    frequency : str, default = '3hourly'
        Name of subdirectory of history containing data (which should indicate
        the frequency subset).

    hist_avg : bool, default = False
        Whether to look for and load <frequency> averages or instantaneous
        output (default). 

    offset_timedelta : datetime.timedelta
        Offset added to datetime values. Default is zero (i.e., no offset,
        which is not needed for instantaneous diagnostics which is also the
        default behaviour). If averages are loaded, this should be set to
        minus half the averaging frequency.

    set_miss_to_nan : bool, default = True
        Sets missing values (land, in most cases) to numpy.nan.

    slice_to_atm_grid : bool, default = False
        If True, slices the data arrays so that the domain lines up with
        the atmospheric grid. [FN2]


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Time coordinates as datetimes.

    data : length nv list of array (nt,*)
        The specified processed diagnostic data.

    [FN3]


    Notes
    -----
    [FN1] For daily or monthly output use get_history_data() in this module.
    [FN2] See documentation of slice_cice_data_to_atm_grid() in this module.
    [FN3] Does not return spatial coordinates as those can be loaded from
          the get_grid_data() function in this module.

    """

    nc_vars = ["time"] + nc_var_names

    # NetCDF base filename format:
    file_fmt = "iceh" + ("_inst"*(not hist_avg))
    file_fmt += "." + frequency + ".{:04}-{:02}.nc"

    # Get path to where this data is stored:
    dat_path = cfg.data_path[f"hist_{frequency[:2]}"]

    # Determine netCDF files (as full paths). Years required to iterate over
    # determined by start and end dates:
    years_reqd = np.arange(dt_min.year, dt_max.year+1, 1)

    if len(years_reqd) == 1:

        # Months required:
        months_reqd = np.arange(dt_min.month, dt_max.month+1, 1)

        # Loop over months and append file paths:
        file_list = [str(Path(dat_path, file_fmt.format(years_reqd[0], m)))
                     for m in months_reqd]
    else:
        raise Exception(f"Loading {frequency} data for more than one year is "
                        + "not currently supported")

    data = list(nct.get_arrays(file_list, [], nc_vars))

    # Get calendar and time units:
    _, t_units, t_cal = nct.get_nc_time_props(file_list[0])

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                  calendar=t_cal)) + offset_timedelta

    # Get time slice (this is necessary since data is saved monthly and we may
    # wish to start or end somewhere in the middle of the month):
    jt = [(x >= dt_min) and (x <= dt_max) for x in date]
    date = date[jt]
    for k in range(1, len(data)):
        data[k] = data[k][jt]

    if seltime > 1:
        # Sub-sample time axis (select every seltime step):
        date = date[::seltime]
        data[0] = data[0][::seltime]
        for k in range(1, len(data)):
            data[k] = data[k][::seltime,:,:]

    if slice_to_atm_grid:
        for k in range(1, len(data)):
            # We don't load lon/lat here so just use dummy
            # 2D coordinate arrays of zeros:
            _, _, data[k] = slice_cice_data_to_atm_grid(
                np.zeros((2,2)), np.zeros((2,2)), data[k])

    if set_miss_to_nan:
        for k in range(1, len(data)):
            data[k] = np.where(abs(data[k]) > 1E20, np.nan, data[k])

    # Remove the added time coordinate from data list:
    data = data[1:]

    return date, data


def get_processed_data(metric, nc_var_names, dt_min=dt(1980, 1, 1),
                       dt_max=dt(2022, 12, 31), frequency="daily",
                       slice_to_atm_grid=False):
    """Load specified CICE 'processed' diagnostic data. [FN1]


    Parameters
    ----------
    metric : str
        Which processed diagnostic to load (e.g., 'sea_ice_extent').
        Corresponds to the subdirectory containing the relevant netCDF files.

    nc_var_names : length-nv iterable of str
        The netCDF variable names to load.


    Optional parameters
    -------------------
    dt_min, dt_max : datetime.datetime
                     default = 1980/01/01 12:00
                               2022/12/31 12:00
        Start and end datetimes for which to load data. If frequency (below) is
        'daily', hour and minute are overwritten as 12 and 0 respectively, so
        that daily averages written at 12:00 are compared correctly within
        these limits.

    frequency : str, default = 'daily'
        Frequency subset to load ('daily', 'monthly', 'hourly', etc., depending
        on what is available).

    slice_to_atm_grid : bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid. [FN2,3]


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Time coordinates as datetimes.

    data : length-nv list of ndarray (nt,*)
        The specified processed diagnostic data.

    [FN4]


    Notes
    -----
    [FN1] Reliant on consistent file naming and locations.
    [FN2] See documentation of slice_cice_data_to_atm_grid() in this module.
    [FN3] Following from [FN2], this is set to False by default so that the
          function does not try to slice 1D data such as sea_ice_extent.
    [FN4] Does not return spatial coordinates as some processed metrics are
          area-integrated (e.g., extent) and some are 2D (e.g., div_u). The
          coordinates are the same as those loaded from the function
          get_grid_data() in this module, i.e., those for the native CICE grid.

    """

    if frequency == "daily":
        dt_min = dt_min.replace(hour=12, minute=0)
        dt_max = dt_max.replace(hour=12, minute=0)

    nc_vars = ["time"] + nc_var_names

    # Get full path to where this data is stored:
    dat_path = Path(cfg.data_path[f"proc_{frequency[0]}"], metric)

    # Load relevant netCDF files at this location for year range
    # (processed metrics saved yearly assumed):
    file_list = []
    for y in range(dt_min.year, dt_max.year+1, 1):
        file_list += sorted([str(x) for x in dat_path.glob(f"*{y}*.nc")])

    data = list(nct.get_arrays(file_list, [], nc_vars))

    # Get calendar and time units:
    _, t_units, t_cal = nct.get_nc_time_props(file_list[0])

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                              calendar=t_cal))

    jt = [(x >= dt_min) and (x <= dt_max) for x in date]
    date = date[jt]
    for k in range(1, len(data)):
        data[k] = data[k][jt]

    if slice_to_atm_grid:
        for k in range(1, len(data)):
            # We don't load lon/lat here so just use dummy
            # 2D coordinate arrays of zeros:
            _, _, data[k] = slice_cice_data_to_atm_grid(
                np.zeros((2,2)), np.zeros((2,2)), data[k])

    data = data[1:]  # remove time coordinate from data list

    return date, data


def get_subdaily_processed_data(metric, nc_var_names, dt_min=dt(2018, 9, 1, 0),
                                dt_max=dt(2018, 9, 30, 21),
                                frequency="3hourly", hist_avg=False,
                                slice_to_atm_grid=False):
    """Load specified CICE processed diagnostic data. [FN1]


    Parameters
    ----------
    metric : str
        Which processed diagnostic to load (e.g., 'sea_ice_extent').
        Corresponds to the directory containing the relevant NetCDF files.

    nc_var_names : length-nv iterable of str
        The netCDF variable names to load.


    Optional parameters
    -------------------
    dt_min, dt_max : datetime.datetime
                     default = 1980/01/01 12:00
                               2022/12/31 12:00
        Start and end datetimes for which to load data. If frequency (below) is
        'daily', hour and minute are overwritten as 12 and 0 respectively, so
        that daily averages written at 12:00 are compared correctly within
        these limits.

    frequency : str, default = '3hourly'
        Frequency subset to load ('3hourly', etc., depending on what is
        available).

    slice_to_atm_grid : bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid. [FN2,3]


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Time coordinates as datetimes.

    data : length-nv list of array (nt,*)
        The specified processed diagnostic data.

    [FN4]


    Notes
    -----
    [FN1] Reliant on consistent file naming and locations.
    [FN2] See documentation of slice_cice_data_to_atm_grid() in this module.
    [FN3] Following from [FN2], this is set to False by default so that the
          function does not try to slice 1D data such as sea_ice_extent.
    [FN4] Does not return spatial coordinates as some processed metrics are
          area-integrated (e.g., extent) and some are 2D (e.g., div_u). The
          coordinates are the same as those loaded from the function
          get_grid_data() in this module, i.e., those for the native CICE grid.
    """
    
    nc_vars = ["time"] + nc_var_names
    
    # Get full path to where this data is stored:
    dat_path = Path(cfg.data_path[f"proc_{frequency[:2]}"], metric)

    # Load relevant netCDF files at this location for year range
    # (processed metrics saved yearly and monthly assumed):
    # 
    # Does not support multiple years
    # 
    file_list = []
    for m in range(dt_min.month, dt_max.month+1, 1):
        file_list += sorted([str(x) for x in
                             dat_path.glob("*" + ("" if hist_avg else "inst")
                                               + f"*{dt_min.year}-{m:02}.nc")])

    data = list(nct.get_arrays(file_list, [], nc_vars))

    # Get calendar and time units:
    _, t_units, t_cal = nct.get_nc_time_props(file_list[0])

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                              calendar=t_cal))

    jt = [(x >= dt_min) and (x <= dt_max) for x in date]
    date = date[jt]
    for k in range(1, len(data)):
        data[k] = data[k][jt]

    if slice_to_atm_grid:
        for k in range(1, len(data)):
            # We don't load lon/lat here so just use dummy
            # 2D coordinate arrays of zeros:
            _, _, data[k] = slice_cice_data_to_atm_grid(
                np.zeros((2,2)), np.zeros((2,2)), data[k])

    data = data[1:]  # remove time coordinate from data list

    return date, data


def get_processed_data_regional(metric, nc_var, dt_min=dt(1980, 1, 1, 12, 0),
                                dt_max=dt(2022, 12, 31, 12, 0),
                                frequency="daily", region_nc_names=None,
                                slice_to_atm_grid=False):
    """Load specified CICE regional-'processed' diagnostic data.


    Parameters
    ----------
    metric : str
        Which processed diagnostic to load (e.g., 'sea_ice_extent').

    nc_var : str
        The name of the netCDF variable without the appended frequency or
        region_name [FN1].


    Optional parameters
    -------------------
    dt_min, dt_max : datetime.datetime
                     default = 1980/01/01 12:00 UTC
                               2022/12/31 12:00 UTC
        Start and end datetimes for which to load data. Hour and minute are
        overwritten as 12 and 0 respectively, so that daily averages written at
        12:00 are compared correctly with these limits.

    frequency : str, default = 'daily'
        Frequency subset to load ('daily', 'monthly', 'hourly', etc., depending
        on what is available).

    region_nc_names: length nr list of str or None
        The nr names of the regions appended to the diagnostic variable in each
        netCDF file (if None, defaults to those specified in config).

    slice_to_atm_grid : bool, default = False
        If True, slices the data arrays so that the domain lines up with the
        atmospheric grid. [FN2,FN3]


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Time coordinates as datetimes.

    data : length nr list of array arrays (nt,*)
        The processed diagnostics loaded as specified for each region.

    [FN4]


    Notes
    -----
    [FN1] Assumes that netCDF variable names for each region have
          '_x_region_name' appended, where x is d or m (see src.io.config for
          netCDF region names).
    [FN2] See documentation of slice_cice_data_to_atm_grid() in this module.
    [FN3] Following from [FN2], this is set to False by default so that the
          function does not try to slice 1D data such as sea_ice_extent.
    [FN4] Does not return spatial coordinates as some processed metrics are
          area-integrated (e.g., extent) and some are 2D (e.g., div_u). The
          coordinates are the same as those loaded from the function
          get_grid_data() in this module, i.e., those for the native CICE grid.

    """

    dt_min = dt_min.replace(hour=12, minute=0)
    dt_max = dt_max.replace(hour=12, minute=0)

    if region_nc_names is None:
        region_nc_names = cfg.region_names

    nc_vars = ["time"] + [f"{nc_var}_{frequency[0]}_{region}"
                          for region in region_nc_names]

    # Get full path to where this data is stored:
    dat_path = Path(cfg.data_path[f"proc_{frequency[0]}"], metric)

    # Load relevant netCDF files at this location for year range
    # (processed metrics saved yearly assumed):
    file_list = []
    for y in range(dt_min.year, dt_max.year+1, 1):
        file_list += sorted([str(x) for x in dat_path.glob(f"*{y}.nc")])

    data = list(nct.get_arrays(file_list, [], nc_vars))

    # Get calendar and time units:
    _, t_units, t_calendar = nct.get_nc_time_props(file_list[0])

    date = nct.cftime_to_datetime(nc.num2date(data[0], units=t_units,
                                              calendar=t_calendar))

    jt = [(x >= dt_min) and (x <= dt_max) for x in date]
    date = date[jt]
    for k in range(1, len(data)):
        data[k] = data[k][jt]

    if slice_to_atm_grid:
        for k in range(1, len(data)):
            # We don't load lon/lat here so just use dummy
            # 2D coordinate arrays of zeros:
            _, _, data[k] = slice_cice_data_to_atm_grid(
                np.zeros((2,2)), np.zeros((2,2)), data[k])

    data = data[1:]  # remove time coordinate from data list

    return date, data

