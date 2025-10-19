"""Provide statistical and physical diagnostics of VRILEs and their
classifications.
"""

import calendar
from datetime import datetime as dt

import numpy as np

from ..data import atmo, cice


def get_vrile_frequency(date_check, event_bnds):
    """Returns fraction of specified time period that specified VRILEs are
    occurring. Can be used for events in general if only specified by a start
    and end date (i.e., this won't work for tracks which have a number of
    coordinates and are tallied based on the sector(s) they pass through).

    Each day is defined as binary "has a VRILE" or "does not have any VRILE".
    The frequency can thus also be interpreted as the ratio of VRILE days to
    non-VRILE days.


    Parameters
    ----------
    date_check : list or array of datetime.date or datetime.datetime
        The datetime coordinates to check, defining the time period that the
        VRILE frequency is calculated with respect to.

    event_bnds :    list of length n_events of length-2 arrays
                 or tuple or list of datetime.datetime
        The datetime bounds defining the start and end of each VRILE.

    Returns
    -------
    vfreq : float in the range [0., 1.]
        The frequency (fraction of time period) that VRILEs are occurring.

    """

    n_events     = len(event_bnds)
    n_date_check = len(date_check)

    # Save a boolean for each date to be checked:
    check = np.zeros(n_date_check).astype(bool)

    for k in range(n_events):
        # Consider each event in turn: for each date in date_check, flip the
        # boolean to True if this event overlaps with it (or leave it as True if
        # it has been flipped by a previous event, hence the use of logical_or):
        check = np.logical_or( check,   (event_bnds[k][0] <= date_check)
                                      & (date_check <= event_bnds[k][1]) )

    # Frequency is the number of days at which one or more VRILEs occur relative
    # to the number of days in the reference date-check period:
    vfreq = float(np.sum(check)) / float(n_date_check)

    return vfreq


def classify_lukovich(T, D):
    """Classification index used by Lukovich et al. (2021), their equation for
    the 'cumulative impacts':

        Q_thermo = |T| / (|T| + |D|)

    Here, it is re-scaled to a -1 (dynamic) to +1 (thermodynamic) range for
    consistency with other classification methods:

        C_lukovich = 2 * Q_thermo - 1

    This classification method has the caveat (not noted by the authors) that it
    is only meaningful if the thermodynamic and dynamics terms, i.e., the T and
    D inputs, have the same sign (regardless of whether T and D are detrended).


    Parameters
    ----------
    T : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        thermodynamic processes.

    D : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        dynamic processes.


    Returns
    -------
    C_lukovich : float or array (nv,)
        Classification index/indices as defined above.


    Reference
    ---------
    Lukovich, J. V. and coauthors, 2021: Summer extreme cyclone impacts on
    Arctic sea ice, J. Climate, 34(12), 4817-4834, doi:10.1175/JCLI-D-19-0925.1

    """
    return 2. * ( abs(T) / (abs(T) + abs(D)) ) - 1.


def classify_aylmer_1(T, D):
    """Classification index that was tried first, a slightly modified functional
    form of the Lukovich metric:

        C_aylmer_1 = (|T| - |D|) / (|T| + |D|)

    This classification method has the caveat that it is only meaningful if the
    thermodynamic and dynamics terms, i.e., the T and D inputs, have the same sign
    (regardless of whether T and D are detrended).


    Parameters
    ----------
    T : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        thermodynamic processes.

    D : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        dynamic processes.


    Returns
    -------
    C_aylmer_1 : float or array (nv,)
        Classification index/indices as defined above.

    """
    return ( abs(T) - abs(D) ) / ( abs(T) + abs(D) )


def classify_aylmer_2(T, D):
    """Classification index that was tried after realising that the Lukovich
    and Aylmer 1 indices do not work for VRILEs unless both T < 0 and D < 0.

    If T > 0 and D < 0 for some VRILE, that indicates ice growth due to
    thermodynamics. If the diagnostics are detrended, then T > 0 indicates that
    ice melts less quickly than usual (assuming we are only ever looking at the
    melt season). Therefore, in order for this event to represent a VRILE it
    must be fully driven by dynamics, and we thus give it a classification index
    of -1.

    Similar logic is applied to the case T < 0 and D > 0:

    For the case T > 0 and D > 0, this should not happen for a VRILE which is by
    definition an extreme loss loss event, but perhaps there is a (hopefully
    rare) case where both T and D are unremarkable and a VRILE occurs because the
    ice was very vulnerable to begin with (e.g., very thin or low concentration).
    In this case, the classification is meaningless and so is set to NaN.

    In summary, this classification method is defined:

                      ( nan         if T >= 0 and D >= 0
        C_aylmer_2 = <  -1          if T >= 0 and D < 0
                      ( +1          if T < 0  and D >= 0
                      ( C_aylmer_1  otherwise (T < 0 and D < 0)

    where

        C_aylmer_1 = (|T| - |D|) / (|T| + |D|),

    which is given by the function classify_aylmer_1.


    Parameters
    ----------
    T : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        thermodynamic processes.

    D : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        dynamic processes.


    Returns
    -------
    C_aylmer_2 : float or array (nv,)
        Classification index/indices as defined above.

    """
    nv = len(T)
    C_aylmer_2 = np.ones(nv, dtype=np.float64)

    for j in range(nv):
        if T[j] < 0. and D[j] < 0.:
            C_aylmer_2[j] = classify_aylmer_1(T[j], D[j])
        elif T[j] >= 0. and D[j] >= 0.:
            C_aylmer_2[j] = np.nan
        elif T[j] >= 0. and D[j] < 0.:
            C_aylmer_2[j] = -1.
        # else T[j] < 0 and D[j] >= 0 => thermodynamic, set to one (already is)

    return C_aylmer_2


def classify(thermo, dynam):
    """Wrapper function for classifying VRILEs into thermodynamic or dynamically
    dominated, using the default classification index.

    All methods [in this module: classify_*(T, D)] are coded to return a value C
    in the range [-1, 1] such that:

             ( -1  fully dynamic
        C = <   0  equally dynamic/thermodynamic
             ( +1  fully thermodynamic


    Parameters
    ----------
    thermo : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        thermodynamic processes.

    dynam : float or array (nv,) of float
        Cumulative change(s) in sea ice volume over a VRILE or event due to
        dynamic processes.


    Returns
    -------
    C : float or array (nv,) of float
        Classification index or indices.

    """
    return classify_aylmer_2(thermo, dynam)


def compute_averages_over_vriles(vrile_results, reg_masks, hist_diags=[],
        proc_diags=[], atmo_fields=[], hist_function=None,
        norm_time_hist=True, norm_area_hist=True, norm_unit_hist=None,
        norm_time_proc=True, norm_area_proc=True, norm_unit_proc=None,
        norm_time_atmo=True, norm_area_atmo=True, norm_unit_atmo=None,
        year_range=[1979, 2023], months_allowed=[5,6,7,8,9], joined=True,
        proc_metric="div_u", aice_thr=.15, verbose=False):
    """Calculate time and spatial integrals or averages of arbitrary quantities
    over the region of newly exposed sea ice during VRILEs.


    Parameters
    ----------
    vrile_results : dict or length n_reg list of dict
        VRILE results or list of such returned by VRILE diagnostics functions.

    reg_masks : list of length n_reg of 2D arrays, shape (ny, nx)
        Region masks corresponding to each VRILE_results dict.


    Optional parameters
    -------------------
    hist_diags : length n_hist of str (default = [])
        Names of diagnostics (i.e., the NetCDF variable names) stored in
        history data to be averaged.

    norm_time_hist : bool or list of bool (default = True)
        Whether to normalise with respect to time (i.e., compute time average if
        True and time integral if False). If a single bool is provided then this
        is used for all history diagnostics.

    norm_area_hist : bool or list of bool (default = True)
        As above but for the area.
    
    norm_unit_hist : float or list of float or None (default)
        Constant that all history diagnostics are divided by (e.g., for changing
        units). If a single float is provided then this is used for all history
        diagnostics. If None, no final normalisation is carried out.

    The above four parameters are also defined for 'processed' diagnostics:

        proc_diags     : default = []
        norm_time_proc : default = True
        norm_area_proc : default = True
        norm_unit_proc : default = None

    and similarly for atmospheric field data (daily averages):

        atmo_fields    : default = []
        norm_time_atmo : default = True
        norm_area_atmo : default = True
        norm_unit_atmo : default = None

    year_range : length 2 list of int (default = [1979, 2023])
        History, 'processed', and atmospheric data are loaded year by year;
        this is the start and end years of the range.

    months_allowed : list of int, default = [5, 6, 7, 8, 9]
        Months (1 = Jan) over which VRILEs are defined (this is only required
        here as an optimisation to minimise data loading).

    joined : bool, default = True
        Whether to consider joined VRILES or the fixed n day changes.

    proc_metric : str, default = 'div_u'
        Corresponds to the subdirectory name containing the NetCDF files
        containing processed 'diagnostics'.

        N.B. limitation: all processed diagnostics must exist within the same
        set of files.

    aice_thr : float, default = .15
        Threshold sea ice concentration to define 'ice' versus 'no ice'.

    verbose : bool, default = False
        Print extra and progress information to the console.


    Returns
    -------
    metrics_hist : length n_hist list of length n_reg list of 1D arrays
        metrics_hist[d][r][v] is the time and area average/integral as
        specified for history diagnostic d, VRILE v of region r.

    metrics_proc : similar to above but for the 'processed' diagnostics
    metrics_atmo : similar to above but for the atmospheric fields

    """

    if type(vrile_results) == dict:
        # one region/set of results; put into list anyway and use vr as
        # alias/shortening for vrile_results:
        vr = [vrile_results]
        # Also set flag to undo this action at the end:
        rm_list = True
    else:
        vr = vrile_results
        rm_list = False

    n_reg = len(vr)

    if "aice_d" in hist_diags:
        rm_aice = False  # see else block
    else:
        # Ice concentration needed to compute mask:
        hist_diags = ["aice_d"] + hist_diags
        # Set flag to remove this at the end (since it wasn't asked for):
        rm_aice = True

    # String to insert into VRILE results dictionary keys:
    vrds = "_joined" if joined else ""

    # List for each diagnostic of list for each region of array of size = number
    # of VRILEs. These are the return values of the function:
    def _metrics_list(n_diag):
        return [[np.zeros(vr[r][f"n{vrds}_vriles"])
                for r in range(len(vr))] for d in range(n_diag)]

    n_hist = len(hist_diags)
    n_proc = len(proc_diags)
    n_atmo = len(atmo_fields)

    metrics_hist = _metrics_list(n_hist)
    metrics_proc = _metrics_list(n_proc)
    metrics_atmo = _metrics_list(n_atmo)

    dt_min = dt(year_range[0], min(months_allowed), 1, 12)
    dt_max = dt(year_range[0], max(months_allowed),
                calendar.monthrange(1999, max(months_allowed))[1], 12)

    if isinstance(norm_time_hist, bool):
        norm_time_hist = [norm_time_hist for j in range(n_hist)]
    if isinstance(norm_time_proc, bool):
        norm_time_proc = [norm_time_proc for j in range(n_proc)]
    if isinstance(norm_time_atmo, bool):
        norm_time_atmo = [norm_time_atmo for j in range(n_atmo)]

    if isinstance(norm_area_hist, bool):
        norm_area_hist = [norm_area_hist for j in range(n_hist)]
    if isinstance(norm_area_proc, bool):
        norm_area_proc = [norm_area_proc for j in range(n_proc)]
    if isinstance(norm_area_atmo, bool):
        norm_area_atmo = [norm_area_atmo for j in range(n_atmo)]

    if isinstance(norm_unit_hist, float):
        norm_unit_hist = [norm_unit_hist for j in range(n_hist)]
    if isinstance(norm_unit_proc, float):
        norm_unit_proc = [norm_unit_proc for j in range(n_proc)]
    if isinstance(norm_unit_atmo, float):
        norm_unit_atmo = [norm_unit_atmo for j in range(n_atmo)]

    # Need an index/counter so we know which metrics_* element to set results to:
    vi = np.zeros(len(vr)).astype(int)

    # Load T-grid cell areas:
    tarea, = cice.get_grid_data(["tarea"], slice_to_atm_grid=True)

    # Region masks only appear multiplied by the grid cell area
    # Do this here to avoid repeat calculation:
    reg_mask_area = [reg_masks[reg]*tarea for reg in range(len(reg_masks))]

    def _compute(jt, jr, jv, mra, mwi, dat, res, norm_time, norm_area):
        """Compute averages over a VRILE area (called in loop
        after events and their time indices identified)."""
        for jd in range(len(dat)):
            # Time integral/average (norm_time is a bool):
            if norm_time[jd]:  # do time integral
                x = np.nansum(dat[jd][jt,:,:]*mwi, axis=0)
            else: # do time average:
                x = np.nanmean(dat[jd][jt,:,:]*mwi, axis=0)
            # Spatial integral/average and assign
            # (norm_area is a float, 1. if for integral):
            res[jd][jr][jv] = np.nansum(x*mra)/norm_area[jd]

    for year in range(year_range[0], year_range[1]+1, 1):

        if verbose:
            print(f"Calculating diagnostics over VRILEs, year {year:04}", end="\r")

        dt_min_y = dt_min.replace(year=year)
        dt_max_y = dt_max.replace(year=year)

        date, hist_data = cice.get_history_data(hist_diags, dt_min=dt_min_y,
            dt_max=dt_max_y, frequency="daily", slice_to_atm_grid=True)

        # Apply any simple/intermediate processing on history data as specified
        # by hist_function:
        if hist_function is not None:
            hist_data = hist_function(hist_data)

        # Do not save time data (assume all datetime malarkey is handled
        # correctly by all data functions...!):
        if n_proc > 0:
            proc_data = cice.get_processed_data(proc_metric, proc_diags,
                dt_min=dt_min_y, dt_max=dt_max_y, frequency="daily",
                slice_to_atm_grid=True)[1]
        else:
            proc_data = []

        # Atmospheric data function also returns datetimes and lon/lat
        # coordinates, so we need the 4th onwards return values, not the
        # second as above for history and processed data which do not return
        # coordinates:
        atmo_data = list(atmo.get_atmospheric_data_time_averages(
                         atmo_fields, "daily", dt_min_y, dt_max_y,
                         slice_to_cice_grid=True))[3:]

        # Ice concentration needed to compute mask:
        aice = hist_data[hist_diags.index("aice_d")]
        aice = np.where(aice > 1., np.nan, aice)
        aice = np.where(aice < 0., np.nan, aice)

        # Loop over each region:
        for reg in range(n_reg):

            # Loop over each VRILE:
            for v in range(vr[reg][f"n{vrds}_vriles"]):

                if vr[reg][f"date_vriles{vrds}"][v].year == year:

                    # Get the datetime bounds of this VRILE:
                    dt_bnds_v = vr[reg][f"date_bnds_vriles{vrds}"][v,:]

                    # Determine the corresponding indices in the history
                    # data datetime array:
                    jt_v = (date >= dt_bnds_v[0]) & (date <= dt_bnds_v[1])
                    # ^ gives array of True/False; just need the start and end
                    # of the block of True):
                    jt1_v = np.argmax(jt_v)
                    jt2_v = len(jt_v) - np.argmax(jt_v[::-1]) - 1

                    # Determine where to average metrics:
                    mask_v = np.where(   (aice[jt1_v,:,:] >= aice_thr)
                                       & (aice[jt2_v,:,:] <= aice_thr), 1, np.nan)

                    # Mask for when ice is present (greater than extent threshold)
                    # during VRILE period:
                    mask_when_ice = np.where((aice[jt_v,:,:] >= .15), 1., np.nan)

                    # Combine common factors (VRILE mask, regionmask, cell area):
                    mask_reg_area_v = mask_v * reg_mask_area[reg]

                    # Normalisation factors (time of this VRILE and total area
                    # of averaging region). Note that vrile_results number of
                    # days is that of the daily changes, not the number of days
                    # describing the event itself:
                    #nf = 1 + vr[reg][f"vriles{vrds}_n_days"][v]
                    na = np.nansum(mask_reg_area_v)
                    
                    _compute(jt_v, reg, vi[reg], mask_reg_area_v, mask_when_ice,
                             hist_data, metrics_hist, norm_time_hist,
                             [na if x else 1. for x in norm_area_hist])

                    _compute(jt_v, reg, vi[reg], mask_reg_area_v, mask_when_ice,
                             proc_data, metrics_proc, norm_time_proc,
                             [na if x else 1. for x in norm_area_proc])

                    _compute(jt_v, reg, vi[reg], mask_reg_area_v, mask_when_ice,
                             atmo_data, metrics_atmo, norm_time_atmo,
                             [na if x else 1. for x in norm_area_atmo])

                    vi[reg] += 1

    if verbose:
        print("")

    # Normalise units:
    if not norm_unit_hist is None:
        for j in range(n_hist):
            for reg in range(n_reg):
                metrics_hist[j][reg] /= norm_unit_hist[j]

    if not norm_unit_proc is None:
        for j in range(n_proc):
            for reg in range(n_reg):
                metrics_proc[j][reg] /= norm_unit_proc[j]

    if not norm_unit_atmo is None:
        for j in range(n_atmo):
            for reg in range(n_reg):
                metrics_atmo[j][reg] /= norm_unit_atmo[j]

    if rm_aice:
        # Remove aice if it was not originally requested via hist_diags,
        # in which case it was temporarily added to the beginning:
        metrics_hist = metrics_hist[1:]

    if rm_list:
        # Only one VRILE results dictionary was passed in
        metrics_hist = [x[0] for x in metrics_hist]
        metrics_proc = [x[0] for x in metrics_proc]
        metrics_atmo = [x[0] for x in metrics_atmo]
        vr = vr[0]

    return metrics_hist, metrics_proc, metrics_atmo


def compute_averages_over_vriles_detrend(vrile_results, reg_masks,
        hist_diags=[], proc_diags=[], atmo_fields=[], hist_function=None,
        norm_time_hist=True, norm_area_hist=True, norm_unit_hist=None,
        norm_time_proc=True, norm_area_proc=True, norm_unit_proc=None,
        norm_time_atmo=True, norm_area_atmo=True, norm_unit_atmo=None,
        year_range=[1979, 2023], joined=True, proc_metric="div_u",
        aice_thr=.15, verbose=False):
    """Calculate time and spatial integrals or averages of arbitrary quantities
    over the region of newly exposed sea ice during VRILEs, similarly to function
    compute_averages_over_vriles(), but here also detrending these quantities
    after spatial integration.

    Specifically, for each VRILE, the spatial integral (or average) and time
    integral (or average) is computed for each specified quantity over the area
    of exposed sea ice over the time bounds of the VRILE. This is then repeated
    for the same day-of-year range for every year of a specified data range
    (1979-2023 by default). This gives a time series of n_year spatial and time
    integrals over the same area and day-of-year range, one of which corresponds
    to the actual VRILE. This time series is then linearly detrended: since each
    time step is the exact same time of year, this effectively accounts for the
    seasonal cycle too. The result is then given from this detrending at the
    time of the actual VRILE, quantifying the anomalous component with respect
    to typical seasonal variations at this location as well as any long-
    term trend.

    Note: this function has to load in a long time period (43 years by default)
    of daily data in one go in order to do the detrending and thus requires
    history data to be combined into yearly files (not default daily output
    from CICE). For this reason (and since it does not make any difference to
    the results), the method we ended up using is to detrend all the data first
    (for each grid point), then using those as inputs to the other function
    compute_averages_over_vriles().


    Parameters
    ----------
    vrile_results : dict or length n_reg list of dict
        VRILE results or list of such returned by VRILE diagnostic functions.

    reg_masks : list of length n_reg of 2D arrays, shape (ny, nx)
        Region masks corresponding to each VRILE_results dict.


    Optional parameters
    -------------------
    hist_diags : length n_hist of str (default = [])
        Names of diagnostics (i.e., the NetCDF variable names) stored in
        history data to be averaged.

    norm_time_hist : bool or list of bool (default = True)
        Whether to normalise with respect to time (i.e., compute time average if
        True and time integral if False). If a single bool is provided then this
        is used for all history diagnostics.

    norm_area_hist : bool or list of bool (default = True)
        As above but for the area.

    norm_unit_hist : float or list of float or None (default)
        Constant that all history diagnostics are divided by (e.g., for changing
        units). If a single float is provided then this is used for all history
        diagnostics. If None, no final normalisation is carried out.

    The above four parameters are also defined for 'processed' diagnostics:

        proc_diags     : default = []
        norm_time_proc : default = True
        norm_area_proc : default = True
        norm_unit_proc : default = None

    and similarly for atmospheric field data (daily averages):

        atmo_fields    : default = []
        norm_time_atmo : default = True
        norm_area_atmo : default = True
        norm_unit_atmo : default = None

    year_range : length 2 list of int (default = [1979, 2023])
        History, processed, and atmospheric data are loaded year by year;
        this is the start and end years.

    joined : bool, default = True
        Whether to consider joined VRILES or the fixed n day changes.
    
    proc_metric : str, default = 'div_u'
        Corresponds to the subdirectory name containing the NetCDF files
        containing 'processed' diagnostics.

        N.B. limitation: all processed diagnostics must exist within the same
        set of files.

    aice_thr : float, default = .15
        Threshold sea ice concentration to define 'ice' versus 'no ice'.

    verbose : bool, default = False
        Print extra and progress information to the console.


    Returns
    -------
    metrics_hist : length n_hist list of length n_reg list of 1D arrays
        metrics_hist[d][r][v] is the time and area average/integral as
        specified for history diagnostic d, VRILE v of region r.
    
    metrics_proc : similar to above but for the 'processed' diagnostics
    metrics_atmo : similar to above but for the atmospheric fields

    """

    if type(vrile_results) == dict:
        # one region/set of results; put into list anyway and use vr as
        # alias/shortening for vrile_results:
        vr = [vrile_results]
        # Also set flag to undo this action at the end:
        rm_list = True
    else:
        vr = vrile_results
        rm_list = False

    n_reg = len(vr)

    if "aice_d" in hist_diags:
        rm_aice = False  # see else block
    else:
        # Ice concentration needed to compute mask:
        hist_diags = ["aice_d"] + hist_diags
        # Set flag to remove this at the end (since it wasn't asked for):
        rm_aice = True

    # String to insert into VRILE results dictionary keys:
    vrds = "_joined" if joined else ""

    # List for each diagnostic of list for each region of array of size = number
    # of VRILEs. These are the return values of the function:
    def _metrics_list(n_diag):
        return [[np.zeros(vr[r][f"n{vrds}_vriles"])
                for r in range(len(vr))] for d in range(n_diag)]

    n_hist = len(hist_diags)
    n_proc = len(proc_diags)
    n_atmo = len(atmo_fields)

    metrics_hist = _metrics_list(n_hist)
    metrics_proc = _metrics_list(n_proc)
    metrics_atmo = _metrics_list(n_atmo)

    dt_min = dt(year_range[0], 1, 1, 12)
    dt_max = dt(year_range[1], 12, 31, 12)

    years = np.arange(year_range[0], year_range[1]+1, 1).astype(int)
    nyears = len(years)

    if isinstance(norm_time_hist, bool):
        norm_time_hist = [norm_time_hist for j in range(n_hist)]
    if isinstance(norm_time_proc, bool):
        norm_time_proc = [norm_time_proc for j in range(n_proc)]
    if isinstance(norm_time_atmo, bool):
        norm_time_atmo = [norm_time_atmo for j in range(n_atmo)]
 
    if isinstance(norm_area_hist, bool):
        norm_area_hist = [norm_area_hist for j in range(n_hist)]
    if isinstance(norm_area_proc, bool):
        norm_area_proc = [norm_area_proc for j in range(n_proc)]
    if isinstance(norm_area_atmo, bool):
        norm_area_atmo = [norm_area_atmo for j in range(n_atmo)]
    
    if isinstance(norm_unit_hist, float):
        norm_unit_hist = [norm_unit_hist for j in range(n_hist)]
    if isinstance(norm_unit_proc, float):
        norm_unit_proc = [norm_unit_proc for j in range(n_proc)]
    if isinstance(norm_unit_atmo, float):
        norm_unit_atmo = [norm_unit_atmo for j in range(n_atmo)]

    # Load T-grid cell areas:
    tarea, = cice.get_grid_data(["tarea"], slice_to_atm_grid=True)

    # Region masks only appear multiplied by the grid cell area
    # Do this here to avoid repeat calculation:
    reg_mask_area = [reg_masks[reg]*tarea for reg in range(len(reg_masks))]

    def _compute(jt1, jt2, jr, jv, yv, mra, dat, res, norm_time, norm_area,
                 plot=False):
        """Compute detrended averages over a VRILE area (called in loop
        after events and their time indices identified)."""

        for jd in range(len(dat)):

            vals = np.zeros(nyears)

            for k in range(nyears):
                # Time integral/average:
                x = np.nansum(dat[jd][jt1+k*365:jt2+k*365+1,:,:], axis=0)
                x /= norm_time[jd]
                # Spatial integral/average and assign:
                vals[k] = np.nansum(x*mra) / norm_area[jd]

            # Detrend:
            m_jd, b_jd = np.polyfit(years, vals, 1)
            vals -= b_jd + m_jd*years

            # Assign result
            res[jd][jr][jv] = vals[yv]

    if verbose:
        print(f"Loading history data, years {dt_min.year}-{dt_max.year}")
    
    date, hist_data = cice.get_history_data(hist_diags, dt_min=dt_min,
        dt_max=dt_max, frequency="daily", slice_to_atm_grid=True)

    # Apply any simple/intermediate processing on history data as specified
    # by hist_function:
    if not hist_function is None:
        hist_data = hist_function(hist_data)

    if verbose and n_proc > 0:
        print(f"Loading processed data, years {dt_min.year}-{dt_max.year}")

    # Do not save time data (assume all datetime malarkey is
    # handled correctly by all data functions...!):
    proc_data = cice.get_processed_data(proc_metric, proc_diags,
        dt_min=dt_min, dt_max=dt_max, frequency="daily",
        slice_to_atm_grid=True)[1]
 
    if verbose and n_atmo > 0:
        print(f"Loading atmospheric data, years {dt_min.year}-{dt_max.year}")

    # Atmospheric data function also returns datetimes and lon/lat
    # coordinates, so we need the 4th onwards return values, not the
    # second as above for history and processed data which do not return
    # coordinates:
    atmo_data = list(atmo.get_atmospheric_data_time_averages(
                     atmo_fields, "daily", dt_min_y, dt_max_y,
                     slice_to_cice_grid=True))[3:]

    # Ice concentration needed to compute mask:
    aice = hist_data[hist_diags.index("aice_d")]
    aice = np.where(aice > 1., np.nan, aice)
    aice = np.where(aice < 0., np.nan, aice)

    # Loop over each region:
    for reg in range(n_reg):

        if verbose:
            print(f"Region #{reg}")

        # Loop over each VRILE:
        for v in range(vr[reg][f"n{vrds}_vriles"]):

            # Get the datetime bounds of this VRILE:
            dt_bnds_v = vr[reg][f"date_bnds_vriles{vrds}"][v,:]
            # But we just want the day of year (doy):
            v_doy_1 = dt_bnds_v[0].timetuple().tm_yday
            v_doy_2 = dt_bnds_v[1].timetuple().tm_yday

            # Deal with leap years (datetime.datetime accounts for Feb 29):
            if calendar.isleap(dt_bnds_v[0].year):
                if dt_bnds_v[0].month > 2:
                    v_doy_1 -= 1
                if dt_bnds_v[1].month > 2:
                    v_doy_2 -= 1

            # The time indices are one less than these (e.g., January 1st of
            # year_min is index 0 but doy = 1:
            jt1_v = v_doy_1 - 1
            jt2_v = v_doy_2 - 1

            # Determine where to average metrics. This is based on the sea ice
            # state at the above doy range *but also for the year of this
            # VRILE*! Leap years don't exist in this data so just add 365 to
            # however many years there are between starting date and VRILE
            # bound dates:
            delta_yr_v = dt_bnds_v[0].year - dt_min.year
            mask_v = np.where(
                (aice[jt1_v + 365*delta_yr_v,:,:] >= aice_thr) &
                (aice[jt2_v + 365*delta_yr_v,:,:] <= aice_thr),
                1, np.nan)
            
            # Combine common factors (VRILE mask, regionmask and cell area):
            mask_reg_area_v = mask_v*reg_mask_area[reg]
            
            # Normalisation factors (time of this VRILE and total area of
            # averaging region). Note that vrile_results number of days is
            # that of the daily changes, not the number of days describing the
            # event itself:
            nf = 1 + vr[reg][f"vriles{vrds}_n_days"][v]

            if np.isnan(mask_v).all():
                na = 1.
            else:
                na = np.nansum(mask_reg_area_v)

            if na == 0.:
                na = 1.

            _compute(jt1_v, jt2_v, reg, v, delta_yr_v,
                mask_reg_area_v, hist_data, metrics_hist,
                [nf if x else 1. for x in norm_time_hist],
                [na if x else 1. for x in norm_area_hist],
                False)
            
            _compute(jt1_v, jt2_v, reg, v, delta_yr_v,
                mask_reg_area_v, proc_data, metrics_proc,
                [nf if x else 1. for x in norm_time_proc],
                [na if x else 1. for x in norm_area_proc],
                False)
            
            _compute(jt1_v, jt2_v, reg, v, delta_yr_v,
                mask_reg_area_v, atmo_data, metrics_atmo,
                [nf if x else 1. for x in norm_time_atmo],
                [na if x else 1. for x in norm_area_atmo],
                False)

    if verbose:
        print("")

    # Normalise units:
    if not norm_unit_hist is None:
        for j in range(n_hist):
            for reg in range(n_reg):
                metrics_hist[j][reg] /= norm_unit_hist[j]

    if not norm_unit_proc is None:
        for j in range(n_proc):
            for reg in range(n_reg):
                metrics_proc[j][reg] /= norm_unit_proc[j]

    if not norm_unit_atmo is None:
        for j in range(n_atmo):
            for reg in range(n_reg):
                metrics_atmo[j][reg] /= norm_unit_atmo[j]

    if rm_aice:
        # Remove aice if it was not originally requested via hist_diags,
        # in which case it was temporarily added to the beginning:
        metrics_hist = metrics_hist[1:]

    if rm_list:
        # Only one VRILE results dictionary was passed in
        metrics_hist = [x[0] for x in metrics_hist]
        metrics_proc = [x[0] for x in metrics_proc]
        metrics_atmo = [x[0] for x in metrics_atmo]
        vr = vr[0]

    return metrics_hist, metrics_proc, metrics_atmo

