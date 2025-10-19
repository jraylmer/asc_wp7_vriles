"""Provide functions for reading in and pre-processing atmospheric forcing data
used to drive the ice model, the raw/global reanalysis data, and the daily/
monthly averages of the forcing data.
"""

from datetime import datetime as dt, timedelta
from pathlib import Path
import warnings

import netCDF4 as nc
import numpy as np

from ..io import config as cfg
from . import nc_tools as nct


def slice_atm_data_to_cice_grid(lon, lat, atm_data):
    """Removes array slices from atmospheric data that do not correspond to
    coordinates in CICE history output. This is related to the 131x104 versus
    129x104 grid definitions that are used for atmospheric forcing data and
    by CICE internally, respectively.

    Because of the way these grids line up, the first two and last j slices
    (i.e., spatial axis = 0), and the first i slice (i.e., spatial axis = 1)
    do not match up to the CICE 129 by 104 grid. These are removed by this
    function so that the resulting arrays are of shape (128, 103).

    This means that there are also slices in CICE history output that do not
    correspond to any slices in the atmospheric data. Thus if atmospheric and
    CICE diagnostics are required on the precisely the same grid/coordinates,
    the latter must also be sliced (see function slice_cice_data_to_atm_grid()
    in module cice.py).

    For contour plots where cell point locations are required, data sliced by
    this function can be plotted with its own lon/lat coordinates, or
    equivalently the native CICE T-cell TLON/TLAT coordinates indexed [1:,1:].

    For pcolormesh plots where cell corners are required, data should be sliced
    by this function and a further slice of [1:,1:] is needed. It should then be
    plotted with ULON and ULAT directly from CICE with slices [1:,1] which act as
    the T-cell corners. This further slicing of the atmospheric data and used of
    U-cell coordinates is necessary due to a bug in which T-cell bounds are not
    output correctly by CICE (at least, the version we have used).

    Confused? See documentation of function slice_cice_data_to_atm_grid() for
    more information.


    Parameters
    ----------
    lon, lat : array (131, 104)
        Longitude and latitude coordinates, respectively, loaded from atmospheric
        forcing data files.

    atm_data : array (nt, 131, 104)
        Atmospheric data array, where the first axis corresponds to time.


    Returns
    -------
    lon, lat : array (128, 103)
        Sliced longitude and latitude arrays, respectively.

    atm_data : array (nt, 128, 103)
        Sliced atmospheric data array.

    """
    return lon[2:-1,1:], lat[2:-1,1:], atm_data[:,2:-1,1:]


def auto_set_units(atm_data, field_name):
    """Carry out common unit conversions on a dataset atm_data (array)
    corresponding to field_name (str).
    """
    if field_name in ["psl", "mslp"]:
        atm_data /= 100.  # convert from Pa to hPa
    elif field_name in ["t2", "tas","tos", "t2m"]:
        # convert from K to degrees_celsius
        # Note: SST processed data ("sst'") is already in degrees C:
        atm_data -= 273.15


def get_atmospheric_data_time_averages(nc_var_names, freq="daily",
                                       dt_min=dt(1980,  1,  1, 12, 0),
                                       dt_max=dt(1980, 12, 31, 12, 0),
                                       slice_to_cice_grid=True,
                                       auto_units=True):
    """Load specified daily or monthly average atmospheric data over a
    specified time range [for the actual high-frequency forcing use
    function get_atmospheric_forcing_data() in this module].


    Parameters
    ----------
    nc_var_names : length-nv list of str
        The netCDF variable names to load.


    Optional parameters
    -------------------
    freq : str {'daily', 'monthly'}
        Which averages to load (default: 'daily'). This should also match the
        subdirectory from which data are loaded.

    dt_min, dt_max : datetime.datetime
                     default = 1980/01/01 12:00
                               1980/12/31 12:00
        Start and end datetimes for which to load data. For freq = 'daily',
        hour and minute are overwritten as 12 and 0 respectively, so that daily
        averages written at 12:00 are compared correctly with these limits. [FN1]

    slice_to_cice_grid : bool, default = True
        If True (default), slices the data arrays so that the domain lines up
        with the CICE grid. [FN2]

    auto_units : bool, default = True
        Automatically convert units from K to degrees celsius and from Pa to
        hPa where appropriate.


    Returns
    -------
    date : 1D array of datetime.datetime
        Datetime coordinates corresponding to the mid point of averaging
        intervals.

    lon, lat : 2D array of float
        Longitude and latitude coordinates (cell centers) respectively

    data : 3D array of float
        Specified data (field).


    Notes
    -----
    [FN1] Saved averages are datetime stamped at the mid point of the averaging
          interval, e.g., for the daily average on for 1980-01-01, the time
          coordinate corresponds to 12Z on 1980-01-01, with time bounds 0Z on
          1980-01-01 and 0Z on 1980-01-02. This function compares time
          coordinates (rather than time bounds) to date_min and date_max.

    [FN2] See documentation of slice_atm_data_to_cice_grid() in this module.

    """

    if freq == "daily":
        dt_min = dt_min.replace(hour=12, minute=0)
        dt_max = dt_max.replace(hour=12, minute=0)

    # Atmospheric data is saved by year; get list of years
    # that need to be loaded based on date range specified:
    years_reqd = np.arange(dt_min.year, dt_max.year+1, 1).astype(int)

    # List of full paths to netCDF files:
    nc_files = [str(Path(cfg.data_path[f"atmo_{freq[0]}"],
                    f"atmo_fields_{freq[0]}_y{y}.nc")) for y in years_reqd]

    # Load data arrays [3:] including time [2] and spatial coordinates [0,1]:
    data = list(nct.get_arrays(nc_files, ["nav_lon", "nav_lat"],
                               ["time"] + [f"{x}_{freq[0]}"
                                           for x in nc_var_names]))

    # Need also the time units and calendar to work out datetime stamps:
    _, t_units, t_cal = nct.get_nc_time_props(nc_files[0])

    date = nct.cftime_to_datetime(nc.num2date(data[2], units=t_units,
                                              calendar=t_cal))

    # Get time slice: indices matching specified date range:
    j_t = [(x >= dt_min) and (x <= dt_max) for x in date]

    date = date[j_t]

    for k in range(3, len(data)):
        data[k] = data[k][j_t]

        if slice_to_cice_grid:  # spatial slice
            lon, lat, data[k] = slice_atm_data_to_cice_grid(data[0], data[1],
                                                            data[k])

        if auto_units:
            auto_set_units(data[k], nc_var_names[k-3])

    return date, data[0], data[1], *data[3:]


def get_atmospheric_forcing_data(field, dt_min, dt_max, seltime=1,
                                 slice_to_cice_grid=True,
                                 auto_units=True):
    """Load specified atmospheric forcing (i.e., high frequency) data over a
    specified time range [for daily or other averages use function
    get_atmospheric_data_time_averages() in this module].


    Parameters
    ----------
    field : str
        Field name, which should match that in the netCDF file name and
        variable name.

    dt_min, dt_max : datetime.datetime
        Start and end datetimes for which to load data. To avoid saving very
        large arrays into memory these parameters are required.
    
    
    Optional parameters
    -------------------
    seltime : int, default = 1
        If different from 1, selects every seltime time step (e.g., to pick
        every 6 hours from 3 hourly data set seltime = 2). [FN1]

    slice_to_cice_grid : bool, default = True
        If True (default), slices the data arrays so that the domain lines up
        with the CICE grid. [FN2]

    auto_units : bool, default = True
        Automatically convert units if appropriate/defined for field. [FN3]


    Returns
    -------
    date : 1D array of datetime.datetime
        Datetime coordinates corresponding to the mid point of averaging
        intervals.

    lon, lat : 2D array of float
        Longitude and latitude coordinates (cell centers or point locations as
        appropriate), respectively

    data : 3D array of float
        Specified data.


    Notes
    -----
    [FN1] Time slice based on date range is taken first, then based on seltime.
    [FN2] See documentation of slice_atm_data_to_cice_grid() in this module.
    [FN3] See documentation of auto_set_units() in this module.

    """

    # Atmospheric forcing data is saved by year; get list of years that need to
    # be loaded based on date range specified:
    years_reqd = np.arange(dt_min.year, dt_max.year+1, 1).astype(int)

    # List of full paths to netCDF files:
    nc_files = [str(Path(cfg.data_path["atmo_forc"], f"{field}_y{y}.nc"))
                for y in years_reqd]

    # Load data arrays including time and spatial coordinates:
    lon, lat, time, data = nct.get_arrays(nc_files, ["nav_lon", "nav_lat"],
                                          ["time", field])

    # Need also the time units and calendar to work out datetime stamps:
    _, t_units, t_cal = nct.get_nc_time_props(nc_files[0])

    date = nct.cftime_to_datetime(nc.num2date(time, units=t_units,
                                              calendar=t_cal))

    # Get time slice: indices matching specified date range:
    j_t = [(x >= dt_min) and (x <= dt_max) for x in date]

    date = date[j_t]
    data = data[j_t]

    if seltime > 1:
        # Sub-sample time axis (select every seltime step):
        date = date[::seltime]
        data = data[::seltime,:,:]

    if slice_to_cice_grid:  # spatial slice
        lon, lat, data = slice_atm_data_to_cice_grid(lon, lat, data)

    if auto_units:
        auto_set_units(data, field)

    return date, lon, lat, data


def get_atmospheric_reanalysis_raw_data(field, dt_min, dt_max, seltime=1,
                                        reanalysis="JRA-55-do",
                                        auto_units=True, clip_lat=45.,
                                        wrap_lon=True, time_nc_name="time",
                                        lonlat_nc_names=["lon", "lat"]):
    """Load raw reanalysis data (i.e., full time resolution, full spatial
    resolution and range available with no interpolation to CICE grid).

    Assumes reanalysis data is available on the native grid with 1D coordinate
    arrays (i.e., fixed grid; although, these are returned as meshgrid
    coordinates for consistency with CICE/input forcing datasets).


    Parameters
    ----------
    field : str
        Name of field to load (this should match the subdirectory name
        containing this data and the netCDF variable name).

    dt_min, dt_max : datetime.datetime
        Start and end datetimes for which to load data. To avoid saving very
        large arrays into memory these parameters are required.


    Optional parameters
    -------------------
    seltime : int, default = 1
        If different from 1, selects every seltime time step (e.g., to pick
        every 6 hours from 3 hourly data set seltime = 2). [FN1]

    reanalysis : str, default = 'JRA-55-do'
        Name of reanalysis to load (should match subdirectory name).

    auto_units : bool, default = True
        Automatically convert units if appropriate. [FN2]

    clip_lat : float or None, default = 45.
        Latitude in degrees north at which to clip data, i.e., data northward
        of the nearest latitude to clip_lat are returned. If None, no such
        clipping is done and the whole, possibly global, data is returned.

    wrap_lon : bool, default = True
        Whether to duplicate the last row of data in the longitude direction
        (useful for contour plots to avoid the line of apparent missing data).

    time_nc_name : str, default = 'time'
        Name of the netCDF variable corresponding to time.

    lonlat_nc_name : length-2 list of str, default = ['lon', 'lat']
        Names of netCDF longitude and latitude coordinate variables,
        respectively.


    Returns
    -------
    date : array (nt,) of datetime.datetime
        Datetime coordinates.

    lon, lat : array (nlat, nlon) of float
        Longitude and latitude coordinates as 2D meshgrid.

    data : ndarray of shape (nt, nlat, nlon)
        The specified reanalysis data/field.


    Notes
    -----
    [FN1] Time slice based on date range is taken first, then based on seltime.
    [FN2] See documentation of auto_set_units() in this module.

    """

    nc_files = []
    for y in range(dt_min.year, dt_max.year+1, 1):
        nc_files += sorted([str(x) for x in Path(
                cfg.data_path["atmo_rean"], field).glob(f"*{y}*-{y}*.nc")])

    lon, lat, time, data = nct.get_arrays(
        nc_files, lonlat_nc_names, [time_nc_name, field])

    _, t_units, t_cal = nct.get_nc_time_props(nc_files[0])

    date = nct.cftime_to_datetime(nc.num2date(time, units=t_units,
                                              calendar=t_cal))

    if seltime > 1:
        date = date[::seltime]
        data = data[::seltime,:,:]

    j_t = [(x >= dt_min) and (x <= dt_max) for x in date]

    date = date[j_t]
    data = data[j_t]

    if auto_units:
        auto_set_units(data, field)

    if clip_lat is not None:
        j_lat = np.argmin(abs(lat - clip_lat))  # better to use where(lat > clip_lat)?
        lat = lat[j_lat:]
        data = data[:,j_lat:,:]

    if wrap_lon:
        lon = np.concatenate((lon, lon[[0]]))
        data = np.concatenate((data, data[:,:,[0]]), axis=2)

    lon, lat = np.meshgrid(lon, lat)

    return date, lon, lat, data

