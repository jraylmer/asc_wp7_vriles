"""Provides functions to identify VRILEs from sea ice extent time series."""

import calendar
import warnings

import numpy as np


def _check_str_inputs(x, x_valid, opt_name=""):
    """Check string input x is one of the valid options (list x_valid) for
    option name opt_name.
    """
    if x not in x_valid:
        raise ValueError(f"Invalid option {opt_name}: '{x}'; choose from "
                         + "{'" + "', '".join(x_valid) + "}")


def moving_average_filter(x, n_ma=31, fill_val=np.nan):
    """Simple moving average low-pass filter.


    Parameters
    ----------
    x : array (nt,)


    Optional parameters
    -------------------
    n_ma : int, default = 31
        Filter width, i.e., number of adjacent time steps to average over. In
        this implementation, n_ma must be odd: if the input is even, the value
        is increased by one and a warning is raised.

    fill_val : float or (default) NaN
        So that the filtered data has the same shape as the input, the start
        and end (n_ma - 1)/2 values that cannot be averaged are filled with
        this value.


    Returns
    -------
    x_ma : array (nt,)
        Filtered values.

    """

    if n_ma < 1:
        n_ma = 1
    elif n_ma % 2 == 0:
        warnings.warn("Moving average filter width needs to be odd; "
                      + f"setting n_ma = {n_ma} + 1 = {n_ma+1}",
                      category=RuntimeWarning)
        n_ma += 1

    x_ma = fill_val*np.ones(np.shape(x))

    for k in range(len(x_ma)-n_ma+1):
        x_ma[k+(n_ma-1)//2] = np.nanmean(x[k:k+n_ma])

    return x_ma


def detrend_linear(x):
    """Simple detrending: remove linear trend from a time series x(t). The
    linear trend is determined using ordinary least squares regression via
    the NumPy function polyfit().


    Parameters
    ----------
    x : array (nt,)
        Data values at each time, t.


    Returns
    -------
    x_detrended : array (nt,)
        The detrended data.

    trend : array (nt,)
        The linear trend (such that x_detrended + trend = x).

    """

    t = np.arange(len(x)).astype(np.float32)  # dummy time axis

    m, b = np.polyfit(t[np.isfinite(x)], x[np.isfinite(x)], 1)

    trend = m*t + b
    x_detrended = x - trend

    return x_detrended, trend


def detrend_linear_periodic(x, n_cycle=365):
    """Remove a linear trend from a time series, x, accounting for a different
    trend at different points in a periodic cycle. The trends are determined at
    points [k, k+n_cycle, k+2n_cycle, ...] for k = 0 to (n_cycle-1) using
    ordinary least squares regression via the NumPy function polyfit().


    Parameters
    ----------
    x : array (nt,)
        Data values at each time, t.


    Optional parameters
    -------------------
    n_cycle : int, default = 365
        Number of time steps making up the periodicity.


    Returns
    -------
    x_detrended : array (nt,)
        The detrended data.

    trend : array (nt,)
        The linear trend (such that x_detrended + trend = x).

    """

    nt = len(x)

    t = np.arange(nt).astype(np.float32)  # dummy time axis

    # Have to calculate each trend in an explicit loop
    # in case there is a non-integer number of cycles:
    trend = np.zeros(nt)
    x_detrended = np.zeros(nt)

    for k in range(n_cycle):

        # Select time steps k, k+n_cycle, k+2*n_cycle, ...
        xk = x[k::n_cycle]
        tk = np.arange(len(xk))  # dummy time axis for k

        m, b = np.polyfit(tk[np.isfinite(xk)], xk[np.isfinite(xk)], 1)

        trend[k::n_cycle] = m*tk + b
        x_detrended[k::n_cycle] = xk - (m*tk + b)

    return x_detrended, trend


def remove_cycle(x, n_cycle=365):
    """Simple removal of periodicity by subtracting the average cycle from all
    data.

    Parameters
    ----------
    x : array of shape (nt,)
        Data values at each time, t.


    Optional parameters
    -------------------
    n_cycle : int, default = 365
        Periodicity of the data (number of indices of x array; e.g., 365 for
        removing the seasonal cycle from daily-average data (beware of leap
        days! The CICE runs do not include leap days be default anyway).

    Returns
    -------
    x_no_cycle : array (nt,)
        The data with mean seasonal cycle removed.

    cycle : array (nt,)
        The mean seasonal cycle such that x_no_cycle + cycle = x.

    """

    nt = len(x)
    n_cycles = nt // n_cycle  # number of whole cycles
    
    # Reshape array so that mean cycle can be computed. Here only include whole
    # cycles (i.e., excluding any partial cycles near the end, if a non-integer
    # number of cycles is present in data):
    x_r = np.reshape(x[:n_cycles*n_cycle], (n_cycles, n_cycle))

    if n_cycles*n_cycle != nt:

        # We have a partial cycle at the end and need to include this in the
        # averaging. Extend the reshaped array by one in the axis=0 dimension,
        # filling the first part of axis=1 with the actual data (i.e., the
        # partial cycle) and the rest with NaN:
        x_r_ext = np.concatenate(
            (x[n_cycles*n_cycle:], np.nan*np.ones(n_cycle - nt%n_cycle))
        ).reshape((1, n_cycle))

        x_r = np.concatenate((x_r, x_r_ext), axis=0)

        # Now compute the mean cycle (using nanmean to avoid points in the
        # partial cycle that don't exist):
        mean_cycle = np.nanmean(x_r, axis=0)

        # The cycle time series is the mean cycle repeating n_cycles + 1 times
        # (the +1 for the partial); here also remove the extended data:
        cycle = np.tile(mean_cycle, n_cycles+1)[:nt]

    else:

        # We have an integer number of cycles, so just take mean of all cycles,
        # then the cycle time series is just that mean repeated n_cycle times:
        mean_cycle = np.nanmean(x_r, axis=0)
        cycle = np.tile(mean_cycle, n_cycles)

    x_no_cycle = x - cycle

    return x_no_cycle, cycle


def seasonal_trend_decomposition_simple(x, n_ma=1, n_cycle=365):
    """Simple seasonal-trend decomposition of a time series, x(t), into a mean
    offset, linear trend, mean seasonal cycle/periodicity, and residual term.
    The residual term has zero overall linear trend.


    Parameters
    ----------
    x : array (nt,)
        Data values at each time, t.


    Optional parameters
    -------------------
    n_cycle : int, default = 365
        Periodicity of the data (number of indices of x array; e.g., 365 for
        removing the seasonal cycle from daily-average data (beware of leap
        days! The CICE runs do not include leap days be default anyway).

        If nt % n_cycle != 0, the remainder data points at the end are ignored.

    n_ma : int, default = 1
        Filter width for low pass (moving average) filter that is applied to
        the data before detrending. This filtering is used to define the mean
        seasonal cycle and trend only. The trend and seasonal cycle computed
        in this way are subtracted from the original data to give the residual.
        If this parameter is 1 (default) or less, no such filtering is applied.


    Returns
    -------
    x_trend : array (nt,)
        Linear trend of data with seasonal cycle removed and offset subtracted.

    x_cycle : array (nt,)
        Mean seasonal cycle of linearly-detrended data, with offset subtracted.

    x_residual : array (nt,)
        The residual time series, i.e., that of x with the above trend and
        cycle and offset removed.

    x_offset : float
        The offset value (equal to the mean of input data x).

    """

    x_offset = np.nanmean(x)  # to center on zero

    if n_ma <= 1:

        # Determine the mean seasonal cycle: first remove the linear trend,
        # then obtain the mean seasonal cycle:
        _, x_trend = detrend_linear(x - x_offset)
        _, x_cycle = remove_cycle(x - x_offset - x_trend, n_cycle)

        # Linearly detrend the input data without seasonal cycle
        # (this 'roundabout' method ensures the residuals have no trend):
        _, x_trend = detrend_linear(x - x_offset - x_cycle)

    else:

        # Apply low-pass/moving average filter on raw data for the purpose of
        # determining trend and seasonal cycle only:
        x_ma = moving_average_filter(x - x_offset, n_ma)

        # Now determine the mean seasonal cycle: first remove the linear trend,
        # then obtain the mean seasonal cycle:
        _, x_trend = detrend_linear(x_ma)
        _, x_cycle = remove_cycle(x_ma - x_trend, n_cycle)

        # Linearly detrend the input data without seasonal cycle
        # (this 'roundabout' method ensures the residuals have no trend):
        _, x_trend = detrend_linear(x_ma - x_cycle)

    # Residual is constructed by definition to recover the original dataset, x,
    # when adding all the other components back:
    x_residual = x - x_trend - x_cycle - x_offset

    return x_trend, x_cycle, x_residual, x_offset


def seasonal_trend_decomposition_periodic(x, n_ma=1, n_cycle=365):
    """Simple seasonal-trend decomposition of a time series, x(t), into a mean
    offset, periodic linear trend, and residual term.


    Parameters
    ----------
    x : array (nt,)
        Data values at each time, t.


    Optional parameters
    -------------------
    n_ma : int, default = 1
        Filter width for low pass (moving average) filter that is applied to
        the data before detrending. This filtering is used to define the mean
        seasonal cycle and trend only. The trend and seasonal cycle computed
        in this way are subtracted from the original data to give the residual.
        If this parameter is 1 (default) or less, no such filtering is applied.

    n_cycle : int, default = 365
        Periodicity of the data (number of indices of x array; e.g., 365 for
        removing the seasonal cycle from daily-average data (beware of leap
        days! The CICE runs do not include leap days be default anyway).

        If nt % n_cycle != 0, the remainder data points at the end are ignored.


    Returns
    -------
    x_trend : array (nt,)
        Linear trend of data for each point in the perioidic cycle, with offset
        subtracted.

    x_residual : array (nt,)
        The residual time series, i.e., that of x with the above trend and cycle
        and offset removed.

    x_offset : float
        The offset value (equal to the mean of input data x).

    """

    x_offset = np.nanmean(x)  # to center on zero

    if n_ma <= 1:

        # Detrend raw data as a function of day of year:
        _, x_trend = detrend_linear_periodic(x - x_offset, n_cycle)

    else:

        # Apply low-pass/moving average filter on raw data for the purpose of
        # determining trend only:
        x_ma = moving_average_filter(x - x_offset, n_ma)
        _, x_trend = detrend_linear_periodic(x_ma, n_cycle)

    # Residual is constructed by definition to recover the original dataset, x,
    # when adding all the other components back:
    x_residual = x - x_trend - x_offset

    return x_trend, x_residual, x_offset


def moving_average_filter_2D(z, n_ma=31, fill_val=np.nan):
    """Simple moving average low-pass filter for 2D data.
    
    
    Parameters
    ----------
    z : array (nt, ny, nx)
        Input data as a function of time t and position (y, x).


    Optional parameters
    -------------------
    n_ma : int, default = 31
        Filter width, i.e., number of adjacent time steps to average over. In
        this implementation, n_ma must be odd: if the input is even, the value
        is increased by one and a warning is raised.

    fill_val : float or (default) NaN
        So that the filtered data has the same shape as the input, the start
        and end (n_ma - 1)/2 values that cannot be averaged are filled with
        this value.


    Returns
    -------
    z_ma : array (nt, ny, nx)
        Filtered values at time t and locations (y, x).

    """

    if n_ma < 1:
        n_ma = 1
    elif n_ma % 2 == 0:
        warnings.warn("Moving average filter width needs to be odd: "
                      + f"setting n_ma = {n_ma} + 1 = {n_ma+1}",
                      category=RuntimeWarning)
        n_ma += 1

    z_ma = fill_val*np.ones(np.shape(z))

    for k in range(len(z_ma)-n_ma+1):
        z_ma[k+(n_ma-1)//2,:,:] = np.nanmean(z[k:k+n_ma,:,:], axis=0)

    return z_ma


def detrend_linear_2D(z):
    """Simple detrending: remove linear trend from a time series z(t,y,x). The
    linear trend is determined using ordinary least squares regression via
    NumPy function polyfit().


    Parameters
    ----------
    z : array (nt, ny, nx)
        Data values at each time, t, and position (y, x).


    Returns
    -------
    z_detrended : array (nt, ny, nx)
        The detrended data as a function of time and position.

    trend : array (nt, ny, nx)
        The linear trends (such that z_detrended + trend = z).

    """

    nt, ny, nx = np.shape(z)

    t = np.arange(nt).astype(np.float32)  # dummy time axis

    m, b = np.polyfit(t, np.reshape(z, (nt, ny*nx)), 1)

    m = np.reshape(m, (nt, ny, nx))
    b = np.reshape(b, (nt, ny, nx))

    trend = m*t[:, np.newaxis, np.newaxis] + b
    z_detrended = z - trend

    return z_detrended, trend


def detrend_linear_periodic_2D(z, n_cycle=365):
    """Remove a linear trend from a time series, z(t,y,x), accounting for a
    different trend at different points in a periodic cycle. The trends are
    determined at points [k, k+n_cycle, k+2n_cycle, ...] for k = 0 to
    (n_cycle - 1) using ordinary least squares regression via NumPy function
    polyfit().


    Parameters
    ----------
    z : array (nt, ny, nx)
        Input data as a function of time t and position (y, x). Systematic
        missing values (e.g., due to land) must not be set to NaN: if NaN is
        present the regression will fail. However, entire time slices can be
        filled with NaN, in which case this routine ignores such slices
        for the detrending (useful for smoothing filters). Any time slice
        partially filled with NaN will still fail.

    n_cycle : int, optional (default = 365)
        The number of time steps making up the periodicity.


    Returns
    -------
    z_detrended : array (nt, ny, nx)
        The detrended data.

    trend : array (nt, ny, nx)
        The linear trend (such that z_detrended + trend = z).

    """

    nt, ny, nx = np.shape(z)

    # Have to do calculate each trend in an explicit loop
    # in case there is a non-integer number of cycles:
    trend = np.zeros((nt, ny, nx))
    z_detrended = np.zeros((nt, ny, nx))

    for k in range(n_cycle):

        # Select time steps k, k+n_cycle, k+2*ncycle, ...
        zk = z[k::n_cycle,:,:]

        ntk = np.shape(zk)[0]
        tk = np.arange(ntk)  # dummy time axis for k

        zk_fit = np.reshape(zk, (ntk, ny*nx))

        # Find time slices (axis = 0) where all values are NaN:
        where_nan = np.all(np.isnan(zk_fit), axis=1)
        tk_fit = tk[~where_nan]
        zk_fit = zk_fit[~where_nan,:]

        mk, bk = np.polyfit(tk_fit, zk_fit, 1)

        # Reshape OLS coefficients to 2D array:
        mk = np.reshape(mk, (ny, nx))
        bk = np.reshape(bk, (ny, nx))

        # Assign trend and detrended components for this point k in the cycle:
        trend[k::n_cycle,:,:] = (
            mk[np.newaxis,:,:]*tk[:,np.newaxis,np.newaxis] + bk[np.newaxis,:,:])

        z_detrended[k::n_cycle,:,:] = zk - trend[k::n_cycle,:,:]

    return z_detrended, trend


def seasonal_trend_decomposition_periodic_2D(z, n_ma=31, n_cycle=365):
    """Simple seasonal-trend decomposition of a time series, z(t,y,x), into a
    mean offset, periodic linear trend, mean seasonal cycle/periodicity, and
    residual term. The residual term has zero overall linear trend.


    Parameters
    ----------
    z : array (nt, ny, nx)
        The 2D data to detrend as a function of time t and location (y,x).


    Optional parameters
    -------------------
    n_ma : int, default = 1
        Filter width for temporal low pass (moving average) filter that is
        applied to the data before detrending. This filtering is used to define
        the seasonal trend only. The trend computed in this way is subtracted
        from the original data to give the residual. If this is 1 (default) or
        less, no such filtering is applied.

    n_cycle : int, default = 365
        Periodicity of the data (number of indices of x array; e.g., 365 for
        removing the seasonal cycle from daily-average data (beware of leap
        days! The CICE runs do not include leap days be default anyway).


    Returns
    -------
    z_trend : array (nt, ny, nx)
        Linear trend of data for each point in the perioidic cycle, with offset
        subtracted, for each location (y,x).

    z_residual : array (nt, ny, nx)
        The residual time series, i.e., that of z with the above trend and cycle
        and offset removed.

    z_offset : array (ny, nx)
        The offset values [time mean at (y,x) of input data z].

    """

    z_offset = np.nanmean(z, axis=0)  # to center on zero

    if n_ma <= 1:

        # Detrend raw data as a function of day of year:
        _, z_trend = detrend_linear_periodic_2D(z - z_offset, n_cycle)

    else:

        # Apply low-pass/moving average filter on raw data for the purpose
        # of determining trend only:
        z_ma = moving_average_filter_2D(z - z_offset, n_ma)
        _, z_trend = detrend_linear_periodic_2D(z_ma, n_cycle)

    # Residual is constructed by definition to recover the original dataset,
    # z, when adding all the other components back:
    z_residual = z - z_trend - z_offset

    return z_trend, z_residual, z_offset


def _get_delta(t, x, nt_delta=5, nt_delta_units="days"):
    """Get changes in a time series x(t) over specified time intervals.


    Parameters
    ----------
    t : array (nt,) of either float or datetime.datetime or cftime.datetime
        Time or datetime values.

    x : array (nt,) of float
        Data for each time t (e.g., sea ice extent).


    Optional parameters
    -------------------
    nt_delta = int, default = 5
        Specifies how to compute time differences, depending on the value of
        nt_delta_units (see below).

    nt_delta_units = str, {'indices', 'days' (default)}
        If 'indices', computes time differences as nt_delta array indices
        (i.e., x[j*nt_delta] - x[(j-1)*nt_delta]). If all time values are
        equally spaced, this option is appropriate and faster.

        If 'days', t must contain datetime values, and time differences are
        determined by finding time values that differ by exactly nt_delta days.
        If time values are not equally spaced, use this option.
        
        ** WARNING ** this ('days' option) currently does not work correctly
        if leap days are removed from the data.

    Returns
    -------
    t_dx, t_bnds_dx, dx, dx_bnds : arrays (nv,), (nv, 2), (nv,), and (nv, 2)

        The time coordinates, time bounds, corresponding changes in x, and
        corresponding start/end values of x, respectively.

        Times are at the center of the events (i.e., halfway between each time
        bound). If the input times are datetimes, then the time data are also
        datetimes.

        nv < nt is the number of delta time periods (if daily data is input,
        then nv = nt - nt_delta).

    """

    _check_str_inputs(nt_delta_units, ["indices", "days"], opt_name="nt_delta_units")

    # Compute changes in x as specified:
    if nt_delta_units == "days":

        # Only accept exact time differences (i.e., do not interpolate). So,
        # in general we do not know in advance how many values there will be:
        dx        = []  # changes in x
        dx_bnds   = []  # start and end values of x
        t_bnds_dx = []  # corresponding time bounds
        t_dx      = []  # corresponding (centre) time/date

        # Loop over each input time, find nt_delta days into the future, and if
        # it exists, calculate difference in x:
        for j1 in range(len(t)):
            j2 = np.argwhere(np.array([x.days for x in (t - t[j1])]) == nt_delta)

            if len(j2) > 0:
                dx.append(x[j2[0,0]] - x[j1])
                dx_bnds.append([x[j1], x[j2[0,0]]])
                t_dx.append(t[j1] + (t[j2[0,0]] - t[j1])/2)
                t_bnds_dx.append([t[j1], t[j2[0,0]]])

        dx = np.array(dx)
        dx_bnds = np.array(dx_bnds)
        t_dx = np.array(t_dx)
        t_bnds_dx = np.array(t_bnds_dx)

    else:  # nt_delta_units == "indices"

        dx = x[nt_delta:] - x[:-nt_delta]
        dx_bnds = np.stack((x[:-nt_delta], x[nt_delta:]), axis=1)
        t_bnds_dx = np.stack((t[:-nt_delta], t[nt_delta:]), axis=1)
        t_dx = t[:-nt_delta] + (t[nt_delta:] - t[:-nt_delta])/2

    return t_dx, t_bnds_dx, dx, dx_bnds


def _get_threshold_from_percentile(dx, percentiles=[0., 5.]):
    """Returns a 1D array, of length equal to the input percentiles,
    corresponding to the threshold values of the specified percentiles
    (default is 0 and 5th percentile).
    """
    return np.nanpercentile(dx, percentiles)


def _filter_month(dates, dates_bnds, dx, dx_bnds, months_allowed=[5,6,7,8,9]):
    """Returns dates, dates_bnds, dx, and dx_bnds filtered according to whether
    all of the months spanned by dates_bnds are one of those in months_allowed.
    """
    j = np.array([(x[0].month in months_allowed and x[1].month in months_allowed)
                  for x in dates_bnds]).astype(bool)

    return dates[j], dates_bnds[j,:], dx[j], dx_bnds[j,:]


def _filter_data_range(dates, dates_bnds, dx, dx_bnds,
                       data_min=None, data_max=0.):
    """Returns dates, dates_bnds, dx, and dx_bnds filtered according to whether
    dx is greater than or equal to a specified minimum and less than or equal to a
    specified maximum (if either limit is None, that limit is ignored).
    """

    j = np.ones(np.shape(dx)).astype(bool)

    if data_min is not None:
        j *= np.where(dx >= data_min, True, False)

    if data_max is not None:
        j *= np.where(dx <= data_max, True, False)

    return dates[j], dates_bnds[j,:], dx[j], dx_bnds[j,:]


def _filter_threshold(dates, dates_bnds, dx, dx_bnds, threshold,
                      threshold_type="percent"):
    """Return dates, dates_bnds, dx, and dx_bnds filtered according to whether
    dx exceeds specified threshold values (len-2 array of minimum and maximum
    dx data values). If threshold_type == 'percent', then the two threshold
    values are interpreted as percentiles of dx; otherwise, they are
    interpreted as actual/absolute data values.
    """

    if threshold_type == "percent":
        threshold_values = _get_threshold_from_percentile(dx, percentiles=threshold)
    else:
        threshold_values = threshold

    # Note: isn't this just doing _filter_data_range(*) ?
    j = np.where((dx >= min(threshold_values)) & (dx <= max(threshold_values)), 1, 0).astype(bool)

    return dates[j], dates_bnds[j,:], dx[j], dx_bnds[j,:], threshold_values


def identify_vriles(t, x, nt_delta=5, nt_delta_units="days",
                    threshold=[0., 5.], threshold_type="percent",
                    months_allowed=[5,6,7,8,9], data_min=None, data_max=0.,
                    criteria_order=["data_range", "months", "threshold"]):
    """Identify extreme changes in x (VRILEs) from specified thresholding and
    filtering criteria.

    Parameters
    ----------
    t : array (nt,) of either float or datetime.datetime or cftime.datetime
        Time or datetime values.

    x : array (nt,) of float
        Data for each time t (e.g., sea ice extent).


    Optional parameters
    -------------------
    nt_delta = int, default = 5
        Specifies how to compute time differences, depending on the value of
        nt_delta_units (below).

    nt_delta_units = str, {'indices', 'days' (default)}
        If 'indices', computes time differences as nt_delta array indices
        (i.e., x[j*nt_delta] - x[(j-1)*nt_delta]). If all time values are
        equally spaced, this option is appropriate and faster.

        If 'days', t must contain datetime values, and time differences are
        determined by finding time values that differ by exactly nt_delta days.
        If time values are not equally spaced, use this option.

        ** WARNING ** this ('days' option) currently does not work correctly
        if leap days are removed from the data.
    
    threshold : length-2 iterable [min, max] of float
        The minimum and maximum thresholds applied to changes in x to determine
        if they are an 'extreme event', interpreted according to the value of
        threshold_units.

    threshold_units : str, {'percent' (default), 'value'}
        If 'percent', the top threshold[0] to threshold[1] percent of all changes
        in x are defined to be 'extreme events'.

        If 'value', changes within the range threshold[0] to threshold[1] are
        defined to be 'extreme events'.

        Default is to return the top 5.% of changes.

    data_minimum : float or None (default)
        Minimum data value considered before applying threshold criteria.

    data_maximum : float or None, default = 0.
        Maximum data value considered before applying threshold criteria
        (i.e., by default with data_maximum = 0., only the top 5.% of decreases
        in x are considered, not 5.% of the whole dataset).

    months_allowed : iterable of int, default = [5, 6, 7, 8, 9]
        Further criteria on event identification that only these months
        (where 1 = Jan, etc.) may count. Only applied if t contains datetime,
        and the default is the summer period May to September inclusive.

    criteria_order : length 3 list of str
        Each element should be one of 'months', 'data_range', and 'threshold'.
        This determines the order that criteria are applied; for instance, in
        the default order (as written here), data are first subsetted to the
        months specified in months_allowed followed by the overall data minimum
        and maximum and then the threshold is applied.

        For example, the default options, with

            criteria_order = ['months', 'data_range', 'threshold']

        gives the top 5% of MJJAS 5-day declines, whereas, for example,
        
            criteria_order = ['data_range', 'threshold', 'months']
        
        would give the MJJAS subset of the top 5% of all 5-day declines.


    Returns
    -------
    t_vriles: array (nv,) where nv < nt is the number of events identified
        The time or dates (depending upon type of input t) at which 'extreme
        events' are identified (the centre of time interval). Obviously, nv
        is not known in advance.

    vriles: array, (nv,)
        The 'extreme event' magnitudes (change in x over time interval
        nt_delta * nt_delta_units).

    t_dx: array, (ntd,) where ntd <= nt is the number of all possible changes in x.
        The times or dates of all changes in input data. Again, ntd depends
        on the input data/options.

    dx: array, (ntd,)
        The magnitudes of all changes in input data.

    threshold_value: length 2 list
        The threshold values of dx used to identify vriles (identical to input
        if threshold_type == 'value'; if threshold_type == 'percentile', then
        these are the specified percentiles of dx).

    """

    # Check input options:
    _check_str_inputs(nt_delta_units, ["indices", "days"], opt_name="nt_delta_units")
    _check_str_inputs(threshold_type, ["percent", "value"], opt_name="threshold_type")

    for i in criteria_order:
        _check_str_inputs(i, ["months", "data_range", "threshold"], opt_name="criteria_order")

    if len(criteria_order) != 3:
        raise ValueError(f"src.diagnostics.vriles: expected 3 criteria; got {len(criteria_order)}")
    elif len(set(criteria_order)) != 3:
        raise ValueError(f"src.diagnostics.vriles: criteria_order contains duplicates")

    # Get all, overlapping changes as specified. Save these (_0) as the 'unfiltered' data:
    t_dx_0, t_bnds_dx_0, dx_0, dx_bnds_0 = _get_delta(t, x, nt_delta=nt_delta,
                                                      nt_delta_units=nt_delta_units)

    # Create copies for filtering to identify events:
    t_dx = t_dx_0.copy()
    t_bnds_dx = t_bnds_dx_0.copy()
    dx = dx_0.copy()
    dx_bnds = dx_bnds_0.copy()

    # Process criteria in specified order:
    if criteria_order[0] == "months":

        t_dx, t_bnds_dx, dx, dx_bnds = _filter_month(t_dx, t_bnds_dx, dx,
                                                     dx_bnds, months_allowed)

        if criteria_order[1] == "data_range":

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_data_range(t_dx, t_bnds_dx,
                                                              dx, dx_bnds,
                                                              data_min=data_min,
                                                              data_max=data_max)

            t_dx, t_bnds_dx, dx, dx_bnds, threshold_values = \
                _filter_threshold(t_dx, t_bnds_dx, dx, dx_bnds, threshold,
                                  threshold_type)

        else:  # criteria_order[1] == "threshold"
            
            t_dx, t_bnds_dx, dx, dx_bnds, threshold_values = \
                _filter_threshold(t_dx, t_bnds_dx, dx, dx_bnds, threshold,
                                  threshold_type)

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_data_range(t_dx, t_bnds_dx,
                                                              dx, dx_bnds,
                                                              data_min=data_min,
                                                              data_max=data_max)

    elif criteria_order[0] == "data_range":

        t_dx, t_bnds_dx, dx, dx_bnds = _filter_data_range(t_dx, t_bnds_dx, dx,
                                                          dx_bnds,
                                                          data_min=data_min,
                                                          data_max=data_max)

        if criteria_order[1] == "months":

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_month(t_dx, t_bnds_dx, dx,
                                                         dx_bnds, months_allowed)

            t_dx, t_bnds_dx, dx, dx_bnds, threshold_values = \
                _filter_threshold(t_dx, t_bnds_dx, dx, dx_bnds, threshold,
                                  threshold_type)

        else:  # criteria_order[1] == "threshold"

            t_dx, t_bnds_dx, dx, dx_bnds, threshold_values = \
                _filter_threshold(t_dx, t_bnds_dx, dx, dx_bnds, threshold,
                                  threshold_type)

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_month(t_dx, t_bnds_dx, dx,
                                                         dx_bnds, months_allowed)

    else:  # criteria_order[0] == "threshold"

        t_dx, t_bnds_dx, dx, dx_bnds, threshold_values = \
            _filter_threshold(t_dx, t_bnds_dx, dx, dx_bnds, threshold,
                              threshold_type)
        
        if criteria_order[1] == "months":

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_month(t_dx, t_bnds_dx, dx,
                                                         dx_bnds, months_allowed)

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_data_range(t_dx, t_bnds_dx, dx,
                                                              dx_bnds,
                                                              data_min=data_min,
                                                              data_max=data_max)

        else:  # criteria_order[1] == "data_range"

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_data_range(t_dx, t_bnds_dx,
                                                              dx, dx_bnds,
                                                              data_min=data_min,
                                                              data_max=data_max)

            t_dx, t_bnds_dx, dx, dx_bnds = _filter_month(t_dx, t_bnds_dx, dx,
                                                         dx_bnds, months_allowed)

    return t_dx, t_bnds_dx, dx, dx_bnds, t_dx_0, t_bnds_dx_0, dx_0, dx_bnds_0, \
           threshold_values


def join_vriles(t_bnds, dx_bnds, max_gap=1):
    """Combine events that are overlapping, adjacent, or close (up to a
    specified maximum) in time.

    This requires the start and end values of each event, not the changes, so
    that the overall changes of joined events can be determined.


    Parameters
    ----------
    t_bnds : array (nt, 2), of datetime.datetime
        The datetime bounds of identified events, so that, e.g.,
        t_bnds[0] = [datetime_start, datetime_end] for the first event.

    dx_bnds : array (nt, 2) of float
        The bounds of events, so that, e.g., dx_bnds[0] = [x_start, x_end]
        for the first event.


    Optional paramters
    ------------------
    max_gap : int, default = 1
        
        Controls how close non-overlapping events need to be to be combined
        into one. Two events, A and B with A preceding B, are combined into
        one if the end value of x for A is within this number of days of the
        start value of x for B.

        With max_gap = 1 (default), this combines contiguous events: for
        example, if A ends on 1st Jun and B starts on 2nd June, they are
        combined into one event starting at the original start date of A
        and ending at the original end date of B.

        Use max_gap = 0 to only combine events when they actually overlap, or
        the start and end days are the same.


    Returns
    -------
    t_c : array (ntc,) of datetime.datetime
        Datetime coordinates of the ntc combined events (center of new bounds).
        The number of combined events, ntc <= nt, is not known in advance.

    t_bnds_c : array (ntc, 2) of datetime.datetime
        Datetime bounds of combined events.

    dx_c : array (ntc,) of float
        Changes in x over combined events.

    dx_bnds_c : array (ntc, 2) of float
        Start and end values of combined events.

    """

    t_c = []
    t_bnds_c = []
    dx_c = []
    dx_bnds_c = []

    n_events = np.shape(t_bnds)[0]

    # Start at the beginning (j = 0), and iterate over next events (k) until a
    # non-overlapping one is identified:
    j = 0
    while j < n_events:

        k = j + 1  # consider the next event

        # Increase k until a non-overlapping (up to max. gap) event is found:
        while (k < n_events and (t_bnds[k,0] - t_bnds[k-1,1]).days <= max_gap):
            k += 1

        # The last value of k checked returned False, so we include j to k - 1:
        t_bnds_c.append([t_bnds[j,0], t_bnds[k-1,1]])
        dx_bnds_c.append([dx_bnds[j,0], dx_bnds[k-1,1]])

        # Time coordinate at center of new bounds:
        t_c.append(t_bnds[j,0] + (t_bnds[k-1,1] - t_bnds[j,0])/2)

        # Event is the change in x over the new bounds:
        dx_c.append(dx_bnds[k-1,1] - dx_bnds[j,0])

        j = k  # k is already at the next event from while loop above

    t_c = np.array(t_c)
    t_bnds_c = np.array(t_bnds_c)
    dx_c = np.array(dx_c)
    dx_bnds_c = np.array(dx_bnds_c)

    return t_c, t_bnds_c, dx_c, dx_bnds_c


def _get_vrile_date_bounds_indices(date, date_bnds_vriles):
    """Get the array indices of the original datetime data corresponding to
    identified VRILEs (i.e., a 'reverse engineering' approach, since the
    original data is subsetted several times for different criteria making
    it difficult to keep track at the time of identifying.

    Well, not difficult: more that I did not think of this when originally
    implementing that code.


    Parameters
    ----------
    date : array (nt,) of datetime.datetime
        The datetimes of the original, unfiltered/processed input data.

    date_bnds_vriles: array (nt, 2) of datetime.datetime
        The datetime bounds of identified VRILEs; for example,
        date_bnds_vriles[0] = [datetime_start, datetime_end] for the 1st VRILE.


    Returns
    -------
    j_bounds : array (nt, 2) of int
        The indices of date corresponding to date_bnds_vriles.

    """

    j_bounds = np.zeros(np.shape(date_bnds_vriles)).astype(int)

    yr = np.array([x.year for x in date]).astype(int)
    mn = np.array([x.month for x in date]).astype(int)
    dy = np.array([x.day for x in date]).astype(int)

    for k in range(np.shape(date_bnds_vriles)[0]):
        j_bounds[k,:] = [np.argwhere(  (yr == date_bnds_vriles[k,i].year)
                                     & (mn == date_bnds_vriles[k,i].month)
                                     & (dy == date_bnds_vriles[k,i].day))
                         for i in range(2)]

    return j_bounds


def _get_description_str(detrend_type="seasonal_periodic", nt_delta=5,
                        nt_delta_units="days", threshold=[0., 5.],
                        threshold_type="percent",
                        months_allowed=[5, 6, 7, 8, 9],
                        data_min=None, data_max=None, n_ma=1,
                        criteria_order=["data_range", "months", "threshold"]):
    """Create a short description string intended to be used as a metadata
    tag in VRILEs results dictionary (see function identify() in this module).
    """

    # Description string depends on criteria order, so first construct each
    # component (elements of str_ins) then format afterwards into str_fmt:
    str_fmt = "{}_of_{}_of_{}"
    str_ins = {"months": "mn=", "data_range": "", "threshold": ""}

    # Ensure sorted unique list of months:
    mon_srt = sorted(list(set(months_allowed)))

    if len(mon_sort) < 12:
        str_ins["months"] += "".join([calendar.month_name[x][0].upper()
                                      for x in mon_srt])
    else:
        str_ins["months"] += "all"

    if threshold_type == "percent":
        str_ins["threshold"] += f"{threshold[0]:.0f}-{threshold[1]:.0f}_percentiles"
    else:
        str_ins["threshold"] += f"dSIE={threshold[0]:.2f}_to_{threshold[1]:.2f}"

    if data_min is None:
        if data_max is None:
            str_ins["data_range"] += "whole_dSIE_range"
        else:
            str_ins["data_range"] += f"up_to_dSIE={data_max:.2f}"
    else:
        if data_max is None:
            str_ins["data_range"] += f"dSIE_above_{data_min:.2f}"
        else:
            str_ins["data_range"] += f"dSIE_between_{data_min:.2f}_and_{data_max:.2f}"

    # Now assemble components in order of criteria applied
    # And append filter options:
    return (str_fmt.format(*[str_ins[k] for k in criteria_order[::-1]])
            + f"_opts_dSIE_{nt_delta}d_movavg_{n_ma}d")


def identify(date, sie, detrend_type="seasonal_periodic", n_cycle=365, n_ma=31,
             id_vriles_kw={}, join_vriles_kw={}, data_title="VRILEs in CICE"):
    """Wrapper function which takes input raw sea ice data and processes it to
    identify VRILEs. Returns a dictionary with various arrays and metadata.


    Parameters
    ----------
    date : array (nt,) of datetime.datetime
        Datetime coordinates of sie.
    
    sie : array (nt,)
        The corresponding sea ice extent (or area, volume, etc.) data.


    Optional parameters
    -------------------
    detrend_type : str {'none', 'seasonal', 'seasonal_periodic'}
        Method for detrending data before identifying VRILEs.
        Default is 'seasonal_periodic'.

    n_cycle : int, default = 365
        Only used when detrend_type == 'seasonal' or 'seasonal_periodic';
        the periodicity of sie to use when removing the seasonal cycle in
        days or number of indices.

    n_ma : int, default = 31
        Only used when detrend_type == 'seasonal' or 'seasonal_periodic';
        the size of the moving average filter width to apply to data before
        removing the seasonal cycle.

    id_vriles_kw : dict
        Keyword arguments passed to function identify_vriles(), and also
        used to construct metadata descriptions.

    join_vriles_kw : dict
        Keyword arguments passed to function join_vriles().

    data_title : str, default = "VRILEs in CICE"
        Description used for metadata.


    Returns
    -------
    results : dictionary
    
    This contains various results and metadata. From the detrending step:

        'sie_component_trend'   : array (nt,), trend component
        'sie_component_cycle'   : array (nt,), periodic/seasonal cycle component
        'sie_component_mean'    : float      , mean/offset
        'sie_component_residual': array (nt,), residual component
        
    These sum to the input sie. They are not included if detrend_type == 'none'.
    From the VRILE identification step:

        'indices_vriles_bnds' : array (nv, 2) of int
            Indices of input data corresponding to datetime bounds of VRILEs,
            where nv is the number of VRILEs identified.

        'date_vriles' : array (nv,) of datetime
            Datetimes (centre of bounds) of VRILEs

        'date_bnds_vriles' : array (nv,2) of datetime
            Datetime bounds (start and end date) of VRILEs

        'date_delta_sie' : array (nt,) of datetime
            Datetimes (centre of bounds) of all possible changes in sie (as
            defined by option nt_delta in id_vriles_kw if present, otherwise
            the default value used in the subsequent functions).

        'date_bnds_delta_sie' : array (nt,2) of datetime
            Datetime bounds (start and end date) of all possible changes in sie
            (as defined by option nt_delta in id_vriles_kw if present, otherwise
            the default value used in the subsequent functions).

        'vriles' : array (nv,) of float
            Magnitudes of VRILEs.

        'vriles_bnds' : array (nv,2) of float
            Start/end values of sie for each VRILE in array vriles.

        'delta_sie' : array (nt,) of float
            All possible changes in sie (as defined by option nt_delta in
            id_vriles_kw if present, otherwise the default value used in
            the subsequent functions).

        'sie_bnds' : array (nt,2) of float
            The sie values defining the start/end values of the corresponding
            array delta_sie.

        'threshold' : array (2,) of float
            The minimum and maximum values of sie defining a VRILE.

        'vriles_n_days' : array (nt,) of int
            The number of days defining each VRILE (this will be the same for
            all VRILEs, but is provided for the analogous quantity for joined
            VRILEs; see below).

    Note that, in the above returned arrays, 'sie' and VRILEs refer to the
    detrended data, specifically, the residual component if detrend_type is
    either 'seasonal' or 'seasonal_periodic'.
    
    Analogous keys 'indices_vriles_joined_bnds', 'date_vriles_joined',
                   'data_bnds_vriles_joined'   , 'vriles_joined',
                   'vriles_joined_bnds'        , 'vriles_joined_n_days'

    are defined for the arrays corresponding to the set of nvj joined/combined
    VRILEs, which also have the following additional outputs:

        'vriles_joined_rates' : array (nvj,) of float
            The change in sie over each joined VRILE divided by its time
            bounds (this is so joined VRILEs spanning different lengths of time
            can be compared).

        'vriles_joined_rates_rank' : array (nvj,) of int
            The rank of each joined VRILE based on their rates, so that the
            joined VRILE with rank = 1 has the largest (most negative) mean
            change in sie per unit time.

    Metadata keys:

        'title'            : str, description (currently just set to data_title)
        'description_plain': str, longer description
        'detrend_type'     : str, detrending method (just set to detrend_type)
        'n_vriles'         : int, number of VRILEs
        'n_joined_vriles'  : int, number of joined/combined VRILEs

    """

    # Validate string input options:
    _check_str_inputs(detrend_type, ["none", "seasonal", "seasonal_periodic"],
                      opt_name="detrend_type")

    # Initialise results dictionary (that returned by function):
    results = {"title": data_title, "detrend_type": detrend_type}

    # Detrend data:
    if detrend_type == "seasonal_periodic":
        trend, resid, offset \
            = seasonal_trend_decomposition_periodic(sie, n_ma=n_ma, n_cycle=n_cycle)

    elif detrend_type == "seasonal":

        trend, cycle, resid, offset \
            = seasonal_trend_decomposition_simple(sie, n_ma=n_ma, n_cycle=n_cycle)

    if detrend_type == "none":
        resid = sie
    
    else:
        results["sie_component_trend"]          = trend
        if detrend_type == "seasonal":
            results["sie_component_cycle"]      = cycle
        results["sie_component_mean"]           = offset
        results["sie_component_residual"]       = resid
        results["moving_average_filter_n_days"] = n_ma

    # Identify events from resid:
    t_vriles, t_bnds_vriles, vriles, vriles_bnds, t_dsie, t_bnds_dsie, dsie, \
        sie_bnds, threshold_derived = identify_vriles(date, resid, **id_vriles_kw)

    results["indices_vriles_bnds"] \
        = _get_vrile_date_bounds_indices(date, t_bnds_vriles)

    results["date_vriles"]         = t_vriles
    results["date_bnds_vriles"]    = t_bnds_vriles
    results["date_delta_sie"]      = t_dsie
    results["date_bnds_delta_sie"] = t_bnds_dsie
    results["vriles"]              = vriles
    results["vriles_bnds"]         = vriles_bnds
    results["delta_sie"]           = dsie
    results["sie_bnds"]            = sie_bnds
    results["threshold"]           = threshold_derived

    # This will be the same for each VRILE, but needed to have the analogous
    # property defined for joined VRILEs:
    results["vriles_n_days"] = np.array([(x[1] - x[0]).days for x in t_bnds_vriles])

    # Determine unique/joined events (combine them):
    t_vriles_joined, t_bnds_vriles_joined, vriles_joined, vriles_joined_bnds \
        = join_vriles(t_bnds_vriles, vriles_bnds, **join_vriles_kw)

    results["indices_vriles_joined_bnds"] \
        = _get_vrile_date_bounds_indices(date, t_bnds_vriles_joined)

    results["date_vriles_joined"]      = t_vriles_joined
    results["date_bnds_vriles_joined"] = t_bnds_vriles_joined
    results["vriles_joined"]           = vriles_joined
    results["vriles_joined_bnds"]      = vriles_joined_bnds

    results["vriles_joined_n_days"] \
        = np.array([(x[1] - x[0]).days for x in t_bnds_vriles_joined])

    results["vriles_joined_rates"] \
        = np.array([  (vriles_joined_bnds[j,1]-vriles_joined_bnds[j,0])
                    / (results["vriles_joined_n_days"][j])
                    for j in range(len(vriles_joined))])

    order = results["vriles_joined_rates"].argsort()
    vriles_joined_ranks = order.argsort() + 1

    results["vriles_joined_rates_rank"] = vriles_joined_ranks

    # Other metadata for convenience:
    results["n_vriles"]        = len(vriles)
    results["n_joined_vriles"] = len(vriles_joined)

    results["description"] = _get_description_str(detrend_type=detrend_type,
                                                  n_ma=n_ma, **id_vriles_kw)

    return results

