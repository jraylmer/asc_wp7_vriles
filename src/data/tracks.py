"""Provide functions for loading and filtering vorticity track data."""

from datetime import datetime as dt, timedelta
from pathlib import Path

import numpy as np

from ..io import config as cfg


# Default filter_vor_min below is the 95th percentile of all 1979-2023 tracks
# entering 70N and poleward from the JRA-55 T63 dataset

filter_vor_min = 8.889056

_default_track = {
    "dt_0"            : dt(1979, 3, 31, 18),
    "step_hours"      : 6,
    "vor_unit_factor" : 1.E-5,
    "track_file_fmt"  : "tr_trs_pos_VOR850_{}",
    "filter_vor_min"  : filter_vor_min,
    "filter_lat_min"  : 70.,
    "filter_n_check"  : 1,
    "filter_dt_range" : (dt(1979, 5, 1, 0), dt(1979, 9, 30, 18))
}

_default_allowed_sectors_n_neighbours = 1


def _count_consecutive_true(x):
    """Returns an array of int corresponding to the number of
    consecutive True values of input boolean array x.


    Example
    -------
    >>> x = [True, False, True, True, True, False]
    >>> _count_consecutive_true(x)
    array([1, 3])

    """
    return np.diff(
        np.where(np.concatenate(([x[0]], x[:-1] != x[1:], [True])))[0]
    )[::2].astype(int)


def get_track_data(track_id, year,
                   dt_0            = _default_track["dt_0"],
                   step_hours      = _default_track["step_hours"],
                   vor_unit_factor = _default_track["vor_unit_factor"],
                   track_file_fmt  = _default_track["track_file_fmt"],
                   track_data_dir  = None):
    """Get datetimes, coordinates, and relative vorticity values from an Arctic
    cyclone case identified using Kevin Hodges feature-tracking algorithm, for
    a specified track ID and year. Makes some assumptions about the way this
    data is saved, including that:

        * Each track is always identified by a line starting with
          "TRACK_ID  {track_id}" (with two spaces), followed by a line (here
          skipped) indicating the number of time stamps for this track.

        * Data lines are of the form "{int} {float} {float} {float} \n"
          corresponding to time index, longitude, latitude, and relative
          vorticity, respectively.


    Parameters
    ----------
    track_id : int or str
        Track ID number.

    year : int
        Year in which track track_id exists.


    Optional parameters
    -------------------
    Note: defaults of the following parameters are set in module global
    variables.

    dt_0 : datetime.datetime instance
        Reference datetime that the file time code of 0 corresponds to. The
        year is arbitrary here, and is replaced with input year.

    step_hours : int
        Number of hours that a time code step of one corresponds to; e.g., time
        code of 1 means 1 step_hours hours from dt_0. Default settings are such
        that a time code of 1 corresponds to 00Z on 1st May (of input year).

    vor_unit_factor : float
        Vorticity values are multiplied by this to set correct units.

    track_file_fmt : str with one format placeholder {}
        Filesnames, to be formatted with the year.

    track_data_dir : str or Path-like instance or None
        Directory under which data files are stored. If None, get from config.


    Returns
    -------
    date : array (nt,) of datetime.datetime
        The datetime stamps, where nt is the number of timestamps for this
        track.

    lon : array (nt,) of float
        Longitude coordinates as a function of time (date) in degrees east.

    lat : array (nt,) of float
        Latitude coordinates as a function of time (date) in degrees north.

    vor : array (nt,) of float
        Relative vorticity values as a function of time (date).

    """

    dt_0 = dt_0.replace(year=year)

    if track_data_dir is None:
        track_data_dir = cfg.data_path["tracks"] 

    # Identify where the track coordinates are by searching for this string:
    line_search = f"TRACK_ID  {track_id}"

    track_file = Path(track_data_dir, track_file_fmt.format(year))

    with open(track_file, "r") as data:

        lines = data.readlines()
        nlines = len(lines)

        # Iterate over line numbers (lnum), breaking when track ID is found:
        for lnum in range(nlines):
            if line_search in lines[lnum]:
                # Found the track; start line number skips the current line
                # ("TRACK_ID  XXXXXX") and the next one ("POINT_NUM  XX"):
                lnum_start = lnum + 2
                break
        else:
            raise Exception(f"TRACK_ID {track_id} not found in {track_file}")

        # Iterate again from starting line number until the next track data
        # (line starting "TRACK_ID") or the end of file is reached. The value
        # of lnum at this point is one more than the final line of data for the
        # specified track ID:
        for lnum in range(lnum_start, nlines):
            if lines[lnum].startswith("TRACK_ID"):
                break
            # else end of file, i.e., lnum = nlines - 1

        # Save track data before leaving with block and closing the file:
        track_dat = lines[lnum_start:lnum]

    nt = len(track_dat)

    # Prepare arrays for datetime.datetime values, longitude and latitude
    # coordinates, and relative vorticity, respectively:
    date = np.array([dt_0 for k in range(nt)])
    lon = np.zeros(nt).astype(np.float32)
    lat = np.zeros(nt).astype(np.float32)
    vor = np.zeros(nt).astype(np.float32)

    for k in range(nt):

        # Split line. The first component is the integer time code; the second
        # and third are the longitude and latitude as decimals; the fourth is
        # the vorticity as a standard-form value (e.g., 1.234567e+01).
        strs_k = track_dat[k].split()

        date[k] += timedelta(hours=step_hours*int(strs_k[0]))        
        lon[k] = np.float32(strs_k[1])
        lat[k] = np.float32(strs_k[2])

        # In case there is no space between the vorticity value and newline:
        if strs_k[3].endswith("\n"):
            strs_k[3] = strs_k[3][:-2]

        vor[k] = np.float32(strs_k[3])*vor_unit_factor

    return date, lon, lat, vor


def allowed_sectors(n_nei=_default_allowed_sectors_n_neighbours, n_sec=9):
    """Get the list of sector indices j for each sector i that a track is
    considered to 'pass over' sector i if it passes over any sector j.
    Currently, this depends only on the number of nearest-neigbouring sectors
    allowed, n_nei, which is the first input parameter (integer), and assumes
    that the sectors are all contiguous and cyclical.

    For example, with n_nei = 0, a track is only considered to pass over sector
    i if its coordinates pass through those of sector i only. So this function
    will return the following list of lists:

        [ [0], [1], [2], ..., [n_sec-1] ]

    where n_sec is the total number of sectors. But with n_nei = 1,
    a track is considered to pass through sector i if its coordinates pass
    through sector i-1, i, or i+1, and this function will return:

        [ [n_sec-1, 0, 1], [0, 1, 2], ..., [n_sec-2, n_sec-1, 0] ]

    hence the assumption of cyclical and contiguous sectors.


    Optional parameters
    -------------------
    n_nei : int
        Number of nearest-neighbouring sectors a track is allowed to pass
        through in order to count as passing through a given sector.

        Currently, only n_nei = 0 or 1 are coded. This is because of the way
        the regions are currently defined and used throughout the rest of the
        code. The 'regions' definitions include the pan-Arctic, which is not a
        'sector'. Without wanting to change the way the region indices are
        handled in general (yet), leaving this as is for now.

    n_sec : int
        Total number of sectors.


    Returns
    -------
    j_sectors_allowed : length n_sec list of [length 1 + 2*n_nei list of int]
        Sector indices allowed for each track to be considered to 'pass over'
        each sector.

    """

    if n_nei == 0:
        # Only allowed to be in corresponding sector (no neighbours allowed)
        j_sec_allowed = [[i] for i in range(n_sec)]

    elif n_nei == 1:
        # Can be in sector j +/- 1, wrapping around
        # and accounting for the first being pan-Arctic
        j_sec_allowed = [
            [0],                  # PAN (0)
            [n_sectors-1, 1, 2],  # LAB (1)
            [1, 2, 3],            # GIN (2)
            [2, 3, 4],            # BAR (3)
            [3, 4, 5],            # KAR (4)
            [4, 5, 6],            # LAP (5)
            [5, 6, 7],            # ESC (6)
            [6, 7, 8],            # BEA (7)
            [7, 8, 1],            # CAN (8)
        ]

    else:
        raise Exception(f"src.data.tracks: n_nei = {n_nei} not supported")

    return j_sec_allowed


def get_filtered_tracks(year,
                        vor_min         = _default_track["filter_vor_min"],
                        lat_min         = _default_track["filter_lat_min"],
                        n_check         = _default_track["filter_n_check"],
                        dt_range        = _default_track["filter_dt_range"],
                        sector_lon_bnds = None,
                        dt_0            = _default_track["dt_0"],
                        step_hours      = _default_track["step_hours"],
                        vor_unit_factor = _default_track["vor_unit_factor"],
                        track_file_fmt  = _default_track["track_file_fmt"],
                        track_data_dir  = None,
                        sort_by_date    = False):
    """Get datetimes, coordinates, and relative vorticity values for a given
    year of tracks from Hodges feature-tracking algorithm, filtering with
    specified latitude and vorticity conditions. Makes some assumptions about
    the way this data is saved, including that:

        * Each track is always identified by a line starting with
          "TRACK_ID  {track_id}" (with two spaces), followed by a line (here
          skipped) indicating the number of time stamps for this track.

        * Data lines are of the form '{int} {float} {float} {float} \\n'
          corresponding to time index, longitude, latitude, and relative
          vorticity, respectively.


    Parameters
    ----------
    year : int
        Year to load (track data files are saved per year).


    Optional parameters
    -------------------
    Note: defaults of the following parameters are set in module global
    variables.

    vor_min : float
        Minimum relative vorticity, in units of vor_unit_factor, that a track
        must have for at least n_check consecutive coordinates.

    lat_min : float
        Minimum latitude that a track must traverse on at least n_check
        consecutive coordinates.

    n_check : int
        Number of consecutive coordinates that the vorticity and latitude
        conditions must be satisfied on.

    dt_range : length-2 tuple of datetime.datetime
        The datetime range a track is allowed to be within (all coordinates).
        The year is arbirtary here, and is replaced with the input year.

    sector_lon_bnds : list of list of tuple or None
        For each region, a list of longitude ranges (0-360 degrees East range) 
        defining each sector boundary. Most will be one tuple, but multiple
        ranges are allowed (e.g., to have a sector containing 0). If None
        (default), get from the config.

    dt_0 : datetime.datetime instance
        Reference datetime that the file time code of 0 corresponds to. The
        year is arbitrary here, and is replaced with input year.

    step_hours : int
        Number of hours that a time code step of one corresponds to; e.g., time
        code of 1 means 1 step_hours hours from dt_0. Default settings are such
        that a time code of 1 corresponds to 00Z on 1st May (of input year).

    vor_unit_factor : float
        Vorticity values are multiplied by this to set correct units.

    track_file_fmt : str with one format placeholder {}
        Filenames, to be formatted with the year.

    track_data_dir : str or pathlib.Path or None
        Directory under which data files are stored. If None, get from config.

    sort_by_date : bool, default = False
        Option to sort returned track data by the first datetime stamp (default
        switched off to avoid any issues with earlier saved data).


    Returns
    -------
    Returns seven lists of length n_filtered_tracks each containing either float
    or arrays of length nt_k for each track k. The number of timesteps for each
    track k (i.e., nt_k) is different and, as with n_filtered_tracks, not known
    in advance:

    tr_tid : list of int
        Track IDs.

    tr_dts : list of array of datetime.datetime
        Datetime coordinates of points for each track.

    tr_lon, tr_lat : list of array of float
        Longitude and latitude coordinates respectively.

    tr_vor : list of array of float
        Vorticity values along each track (note that values are multiplied by
        vor_unit_factor).

    tr_vor_max : list of float
        Vorticity maxima along each track where the latitude criteria is
        satisfied.

    tr_sec : list of array (n_sectors,) of bool
        Boolean array for each track which is True if it passes through sector
        j and meets latitude and vorticity criteria for j in range(n_sectors).
        Sector definitions are currently hardcoded.
    """

    if track_data_dir is None:
        track_data_dir = cfg.data_path["tracks"]

    if sector_lon_bnds is None:
        sector_lon_bnds = cfg.reg_sector_lon_bnds

    n_sectors = len(sector_lon_bnds)

    sector_lat_min = [lat_min]*n_sectors

    dt_0 = dt_0.replace(year=year)
    dt_range = tuple([x.replace(year=year) for x in dt_range])

    track_file = Path(track_data_dir, track_file_fmt.format(year))

    # Letting npt = number of time steps for each track:
    tr_tid = []  # will contain integers
    tr_dts = []  # will contain arrays (npt,) of dt
    tr_lon = []  # will contain arrays (npt,) of float
    tr_lat = []  # will contain arrays (npt,) of float
    tr_vor = []  # will contain arrays (npt,) of float
    tr_vor_max = []  # will contain float
    tr_sec = []  # will contain arrays (n_sectors, npt) of bool

    with open(track_file, "r") as data:

        lines = data.readlines()
        n_lines = len(lines)

        lnum = 0

        while lnum < n_lines:

            if lines[lnum].startswith("TRACK_ID"):

                # New track; next line is number of points:
                npt_k = int(lines[lnum+1].split()[-1])

                # Start and end line numbers for data:
                lnum_k_start = lnum + 2
                lnum_k_end = lnum_k_start + npt_k - 1

                date_k = np.array([dt_0 for j in range(npt_k)])
                lons_k = np.zeros(npt_k).astype(float)
                lats_k = np.zeros(npt_k).astype(float)
                vort_k = np.zeros(npt_k).astype(float)

                for j in range(npt_k):

                    strs_k_j = lines[lnum_k_start+j].split()
                    date_k[j] += timedelta(hours=step_hours*int(strs_k_j[0]))
                    lons_k[j] = np.float32(strs_k_j[1])
                    lats_k[j] = np.float32(strs_k_j[2])

                    # In case there is no space between the
                    # vorticity value and the newline:
                    if strs_k_j[3].endswith("\n"):
                        strs_k_j[3] = strs_k_j[3][:-2]
                    vort_k[j] = np.float32(strs_k_j[3])

                # Latitude must be greater than lat_min AND vorticity must be
                # greater than vor_min for at least n_check consecutive points:
                lat_and_vor_check = any(_count_consecutive_true(
                    (lats_k >= lat_min) & (vort_k >= vor_min)) >= n_check)

                time_check = all([dt_range[0] <= x <= dt_range[1]
                                  for x in date_k])

                conditions = [lat_and_vor_check, time_check]

                if all(conditions):

                    tr_tid.append(lines[lnum].split()[-1])
                    tr_dts.append(date_k)
                    tr_lon.append(lons_k)
                    tr_lat.append(lats_k)
                    tr_vor.append(vort_k*vor_unit_factor)

                    tr_vor_max.append(np.max(vort_k[lats_k >= lat_min])
                                      * vor_unit_factor)

                    # Find if it traverses each sector:
                    ts_r_k = np.zeros((n_sectors, npt_k)).astype(bool)

                    # Vorticity check is the same for each sector:
                    check_vor = vort_k >= vor_min

                    for r in range(n_sectors):
                        check_lat_r = lats_k >= sector_lat_min[r]

                        check_lon_r = np.any(np.array([
                            (lons_k >= sector_lon_bnds[r][rr][0]) &
                            (lons_k <= sector_lon_bnds[r][rr][1])
                            for rr in range(len(sector_lon_bnds[r]))
                        ]), axis=0)

                        ts_r_k[r,:] = (check_lat_r & check_lon_r & check_vor)

                    tr_sec.append(ts_r_k)

                # Jump to next track:
                lnum = lnum_k_end + 1

            else:
                lnum += 1

    if sort_by_date:
        index_order = np.argsort([x[0] for x in tr_dts])
        tr_tid = [tr_tid[j] for j in index_order]
        tr_dts = [tr_dts[j] for j in index_order]
        tr_lon = [tr_lon[j] for j in index_order]
        tr_lat = [tr_lat[j] for j in index_order]
        tr_vor = [tr_vor[j] for j in index_order]
        tr_vor_max = [tr_vor_max[j] for j in index_order]
        tr_sec = [tr_sec[j] for j in index_order]

    return tr_tid, tr_dts, tr_lon, tr_lat, tr_vor, tr_vor_max, tr_sec


def get_filtered_tracks_multiyear(years=np.arange(1979, 2024, 1, dtype=int),
                                  verbose=True, **kw):
    """Wrapper function for get_filtered_tracks(), running it for multiple
    years. Returns an array of the total filtered track counts per year and
    then the multi-year concatenated arrays of track data.


    Optional arguments
    ------------------
    years : list or array of int, default: np.arange(1979, 2024, 1)
        The years to load and filter tracks.

    verbose : bool, default=True
        Filtering takes some time; if True, print progress to the console.

    Additional keyword arguments are passed to function get_filtered_tracks()
    in this module.


    Returns
    -------
    tracks_per_year : array of int
        Total number of filtered tracks in each year.

    Then the concatenated lists of various track data as returned by function
    get_filtered_tracks(), of which there are seven, so that with the above,
    there are eight returned values in total.

    """

    tracks_per_year = np.zeros(len(years)).astype(int)
    track_ids = []
    track_dts = []
    track_lon = []
    track_lat = []
    track_vor = []
    track_vor_max = []
    track_sec = []

    for y in range(len(years)):

        if verbose:
            print(f"Filtering tracks: {years[y]}", end="\r")

        res = get_filtered_tracks(years[y], **kw)

        track_ids += res[0]
        track_dts += res[1]
        track_lon += res[2]
        track_lat += res[3]
        track_vor += res[4]
        track_vor_max += res[5]
        track_sec += res[6]

        tracks_per_year[y] = len(res[0])

    if verbose:
        print("")

    return tracks_per_year, track_ids, track_dts, track_lon, \
           track_lat, track_vor, track_vor_max, track_sec

