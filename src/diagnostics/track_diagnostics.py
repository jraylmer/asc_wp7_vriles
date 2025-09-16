"""Provides statistical diagnostics for tracks (counts per region and
matching to VRILEs).
"""

import datetime
from datetime import datetime as dt

import numpy as np

from src.data import tracks  # for allowed_sectors() function


def get_track_region_counts(track_sec,
                            allowed_sectors=tracks.allowed_sectors(),
                            verbose=True):
    """Count the number of tracks passing through each region including the
    allowed neighbouring regions if specified.


    Parameters
    ----------
    track_sec : list of length n_track of arrays (n_regions, npt) of bool
        Whether each track passes through each defined sector at each of its
        coordinates, where n_regions is the number of regions (sectors)
        checked. This is as returned by the src.data.tracks.filter_tracks()
        function.


    Optional parameters
    -------------------
    allowed_sectors : list of length n_regions of list of int
        The indices of regions that a track is allowed to be associated with in
        order for it to 'pass through' a certain sector. For example:

            allowed_sectors[1] = [8, 1, 2]
        
        means that a track is considered to 'pass through' the region with
        index 1 if its coordinates satisfy the bounds of regions with indices 8
        OR 1 OR 2.

        Default is for the default regions and allows a track to pass through a
        given region or its nearest neighbours (this default is set in the
        module tracks of the data sub-package).

    verbose : bool, default = True
        Whether to print progress (one line, as percentage) to the console.


    Returns
    -------
    t_counts : array (n_regions,) of int
        The number of tracks that 'pass through' each region (see above).

    """

    n_tracks  = len(track_sec)
    n_regions = np.shape(track_sec[0])[0]

    t_counts  = np.zeros(n_regions).astype(int)

    for k in range(n_tracks):

        if verbose:
            print(f"Calculating track counts per region: {100.*k/n_tracks:.0f}%",
                  end="\r")

        for r in range(n_regions):
            # Add one to region r if any coordinate of track k passes through the
            # region r (or other allowed regions for region r):
            t_counts[r] += np.any(track_sec[k][allowed_sectors[r],:])

    if verbose:
        print("Calculating track counts per region: 100%")

    return t_counts


def get_track_frequency(date_check, track_dts, track_sec,
                        allowed_sectors=tracks.allowed_sectors(),
                        verbose=True):
    """Returns fraction of specified time period that specified tracks are
    occurring, both for regions and on the whole. Each day is defined as
    binary 'has a track' or 'does not have a track'.


    Parameters
    ----------
    date_check : list or array of datetime.date or datetime.datetime
        The datetime coordinates to check, defining the time period that the
        track frequency is calculated with respect to.

    track_dts : list of length n_track of arrays (npt,) of datetime.datetime
        Datetime coordinates for each track, where npt is the number of
        coordinates (different per track).

    track_sec : list of length n_track of arrays (n_regions, npt) of bool
        Whether each track passes through each defined sector at each of its
        coordinates, where n_regions is the number of regions (sectors) checked.
        This is as returned by the src.data.tracks.filter_tracks() function.


    Optional parameters
    -------------------
    allowed_sectors : list of length n_regions of list of int
        The indices of regions that a track is allowed to be associated with in
        order for it to 'pass through' a certain sector. For example:

            allowed_sectors[1] = [8, 1, 2]
        
        means that a track is considered to 'pass through' the region with
        index 1 if its coordinates satisfy the bounds of regions with indices 8
        OR 1 OR 2.

        Default is for the default regions and allows a track to pass through a
        given region or its nearest neighbours (this default is set in the
        module tracks of the data sub-package).

    verbose : bool, default = True
        Whether to print progress (one line, percentage) to the console (this
        function is not very efficient).


    Returns
    -------
    tfreq_sec : array (n_regions,) of float
        Number of days for which there is at least one track per number of days
        in the reference date-check period, as a function of region.

    tfreq_all : float
        Number of days for which there is at least one track in any region per
        number of days in the reference date-check period.

    """

    # Ensure date_check is at hour == minute == 0 for comparisons:
    date_check = np.array([x.replace(hour=0, minute=0) for x in date_check])

    n_tracks     = len(track_dts)
    n_date_check = len(date_check)
    n_regions    = np.shape(track_sec[0])[0]

    # Don't really need need to count occurrences of tracks on each date but in
    # terms of the code it is easier to just add to the array elements rather
    # than wrapping another logical_or around the results in the loop below:
    count_date_all = np.zeros(n_date_check).astype(int)
    count_date_sec = np.zeros((n_regions, n_date_check)).astype(int)

    for k in range(n_tracks):

        if verbose:
            print(f"Calculating track frequency: {100.0*k/n_tracks:.0f}%",
                  end="\r")

        # Consider each track in turn, and assign to the count_* arrays +1 on
        # each date if the track passes through the relevant sector(s) on the
        # relevant date.
        #
        # Account for the latitude threshold implicitly: because the lat_min is
        # built into the sector check, choose any sector (track_sec[k], any
        # along axis 0) and this will sample all track coordinates >= lat_min.
        #
        # In the below call to np.any(), the first array, with
        # dt(...) == date_check, has shape (npt, n_date_check).
        #
        # The second needs a new 1-axis as the check is just on the npt
        # coordinate sectors, not the date_checking.
        #
        # Then we want to apply any along the 0-axis corresonding to the track
        # coordinates, leaving a bool result for each date_check that is assigned
        # to count_date_all:
        count_date_all += np.any(
            np.logical_and([dt(track_dts[k][kk].year, track_dts[k][kk].month, track_dts[k][kk].day) == date_check
                            for kk in range(len(track_dts[k]))],
                           np.any(track_sec[k], axis=0)[:,np.newaxis]),
            axis=0)

        # Similarly for the sectors/regions, picking and checking the allowed_sectors
        # from the track_sec arrays:
        for r in range(n_regions):
            count_date_sec[r,:] += np.any(
                np.logical_and([dt(track_dts[k][kk].year, track_dts[k][kk].month, track_dts[k][kk].day) == date_check
                                for kk in range(len(track_dts[k]))],
                               np.any(track_sec[k][allowed_sectors[r],:], axis=0)[:,np.newaxis]),
                axis=0)

    if verbose:
        print("Calculating track frequency: 100%")

    # Frequency is the number of days for which >= 1 tracks occur (in a given
    # sector) relative to the number of days in the reference date-check period:
    tfreq_sec = np.sum(count_date_sec > 0, axis=1).astype(float) / float(n_date_check)

    tfreq_all = float(np.sum(count_date_all > 0)) / float(n_date_check)

    return tfreq_sec, tfreq_all


def match_tracks_to_vriles(vrile_date_bnds, track_dts, track_sec,
                           allowed_sectors=tracks.allowed_sectors(1),
                           track_max_lead_days=0, track_max_lag_days=0,
                           verbose=True):
    """Match vorticity tracks to VRILEs by checking whether each track passes
    through the right region(s)/sector(s) at the right times, possibly also
    accounting for time lead/lag.


    Parameters
    ----------
    vrile_date_bnds : length n_regions list of array (nv, 2)
        The datetime bounds (start and end dates) of each VRILE per region (nv
        being different, in general, for each region).

    track_dts : list of length n_track of arrays (npt,) of datetime.datetime
        Datetime coordinates for each track, where npt is the number of
        coordinates (different per track).

    track_sec : list of length n_track of arrays (n_regions, npt) of bool
        Whether each track passes through each defined sector at each of its
        coordinates, where n_regions is the number of regions (sectors) checked.
        This is as returned by the src.data.tracks.filter_tracks() function.


    Optional parameters
    -------------------
    allowed_sectors : list of length n_regions of list of int
        The indices of regions that a track is allowed to be associated with in
        order for it to 'pass through' a certain sector. For example:

            allowed_sectors[1] = [8, 1, 2]
        
        means that a track is considered to 'pass through' the region with
        index 1 if its coordinates satisfy the bounds of regions with indices 8
        OR 1 OR 2.

        Default is for the default regions and allows a track to pass through a
        given region or its nearest neighbours (this default is set in the
        module tracks of the data sub-package).

    track_max_lead_days : int, default = 0
        Allow a track to match a VRILE if it occurs in the right sectors but up
        to this number of days before the VRILE starts.

    track_max_lag_days : int, default = 0
        Allow a track to match a VRILE if it occurs in the right sectors but up
        to this number of days after the VRILE ends.

    verbose : bool, default = True
        Print progress to console.


    Returns
    -------
    v_tr_indices : len n_regions list of len nv list of array (n_tr_match,) of int
        Indices of the array of all tracks for which a given VRILE in a given
        region matches to. Specifically,

            v_tr_indices[r][v] = array([k1, k2, ...])

        gives the track array indices [k1, k2, ...] which match up to VRILE v
        in region r. Note that the track IDs are NOT unique across all years,
        so it does make sense here to save the indices of the track data itself
        rather than the track IDs. In the above, n_tr_match is different per
        r and v, in general.

    tr_v_indices : length n_track list of list of length-2 list of int
        Region and VRILE indices associated with each track. Specifically,

            tr_v_indices[k] = [ [r1, v1], [r2, v2], ...]

        gives the region and VRILE array indices [r, v] of each VRILE matching
        up to track index k.

    """

    n_regions          = len(vrile_date_bnds)
    n_vriles_by_region = [len(x) for x in vrile_date_bnds]
    n_tracks_anywhere  = len(track_dts)

    # Save list of list of arrays for the matching track-list indices, so that
    #
    #        v_tr_indices[r][v] = array([k1, k2, ... ])
    #
    # gives the track array indices [k1, k2, ... ] for VRILE v in region r:
    v_tr_indices = [ [np.zeros(0).astype(np.int64) for v in range(n_vriles_by_region[r])]
                     for r in range(n_regions)]

    # Save the inverse: the VRILEs matching each track (can't do this by region
    # as tracks are not independently defined per region like the VRILEs are).
    #
    #        tr_v_indices[k] = [ [r1, v1], [r2, v2], ... ]
    # 
    # gives the region and VRILE array indices [r1, v1] for track k:
    tr_v_indices = [ [] for k in range(n_tracks_anywhere)]

    for r in range(n_regions):
    
        if verbose:
            print(f"Finding track+VRILE matches, region: {r}", end="\r")

        # Track conditions:
        # (1) must be in one of the 'allowed sectors' for this region
        # (2) must have at least one time coordinate between dtv1 and dtv2
        #     (subject to any lead/lag allowance)

        # Condition (1) is same for all VRILEs in this region:
        sector_match = [ np.any(trk[allowed_sectors[r],:], axis=0)
                         for trk in track_sec]

        for v in range(n_vriles_by_region[r]):

            # VRILE datetime bounds (start and end dates):
            dt_vi_1 = vrile_date_bnds[r][v,0]
            dt_vi_2 = vrile_date_bnds[r][v,1]

            # Allow for track to lead VRILE:
            dt_vi_1 += datetime.timedelta(days=-abs(track_max_lead_days))

            # Allow for track to lag VRILE:
            dt_vi_2 += datetime.timedelta(days=abs(track_max_lag_days))

            # Condition (2) above:
            time_match = [(trk_dt >= dt_vi_1) & (trk_dt <= dt_vi_2)
                          for trk_dt in track_dts]

            matches_v = [any(sector_match[k] & time_match[k])
                         for k in range(n_tracks_anywhere)]

            # Save the track-array indices for this region/VRILE:
            v_tr_indices[r][v] = np.argwhere(matches_v)[:,0]

            # Save the indices of this region/VRILE to the list of matches for
            # for each matching track:
            for trk in v_tr_indices[r][v]:
                tr_v_indices[trk].append([r, v])

    if verbose:
        print("")

    return v_tr_indices, tr_v_indices

