"""General diagnostics module."""

import numpy as np


def sea_ice_area(siconc, areacello, mask=None, threshold=[0., 1.],
                 weight_by_siconc=True, units=1.E-12):
    """Compute total sea ice area over a specified mask and concentration range.


    Parameters
    ----------
    siconc : 2D (nj, ni) or 3D (nt, nj, ni) array of float
        Sea ice concentration field, as a function of space [2D; (nj, ni)] or
        of time and space [3D; (nt, nj, ni)].

    areacello : 2D (nj, ni) array of float
        Grid cell areas in m^2.


    Optional parameters
    -------------------
    mask : 2D (nj, ni) or 3D (nt, nj, ni) array of float in the range [0., 1.]
        Weighting factor applied for the purposes of masking locations.
        Default = None (i.e., no spatial masking applied).

    threshold: array-like of float [low, high]
        Sea ice area is calculated wherever siconc is greater than or equal to
        low AND less than or equal to high. Default = [0., 1.]. Must be in the
        same units as siconc.

    weight_by_siconc : bool, default = True
        Whether to weight grid cell areas by siconc (i.e., if set to False,
        compute sea ice extent).

    units : float, default = 1.E-12
        Scaling factor which is multiplied at the end of the calculation.
        Default puts the value in 10^6 km^2, assuming that siconc is a fraction
        (rather than a percentage).


    Returns
    -------
    siarea : array of float, of shape (nt,)

    """

    weight = siconc if weight_by_siconc else 1.

    siarea = np.nansum( weight * areacello[np.newaxis,:,:]
                               * (siconc >= threshold[0])
                               * (siconc <= threshold[1]),
                        axis=(1,2)) * units

    return siarea


def sea_ice_extent(siconc, areacello, **sea_ice_area_kwargs):
    """Compute total sea ice extent over a specified mask and concentration
    range.


    Parameters
    ----------
    siconc : 2D (nj, ni) or 3D (nt, nj, ni) array of float
        Sea ice concentration field, as a function of space [2D; (nj, ni)] or
        of time and space [3D; (nt, nj, ni)].

    areacello : 2D (nj, ni) array of float
        Grid cell areas in m^2.

    Additional keyword arguments are passed to function sea_ice_area(). Note
    that argument 'weight_by_siconc' is here overridden to be False. If
    'threshold' is not provided, it is set to [.15, 1.].

    Returns
    -------
    siextent : array of float, of shape (nt,)

    """

    sea_ice_area_kwargs["weight_by_siconc"] = False

    if "threshold" not in sea_ice_area_kwargs.keys():
        sea_ice_area_kwargs["threshold"] = [.15, 1.]

    return sea_ice_area(siconc, areacello, **sea_ice_area_kwargs)


def sea_ice_volume(sithick, siconc, areacello, siconc_threshold=[0., 1.],
                   units=1.E-12):
    """Compute total sea ice volume.


    Parameters
    ----------
    siconc : 2D (nj, ni) or 3D (nt, nj, ni) array of float
        Sea ice concentration field, as a function of space [2D; (nj, ni)] or
        of time and space [3D; (nt, nj, ni)].

    sithick : 2D (nj, ni) or 3D (nt, nj, ni) array of float
        Grid cell mean ice thickness field (total sea ice volume per unit grid
        cell area), as a function of space [2D; (nj, ni)] or of time and space
        [3D; (nt, nj, ni)], in m.

    areacello : 2D (nj, ni) array of float
        Grid cell areas in m^2.


    Optional parameters
    -------------------
    siconc_threshold : array-like of float [low, high]
        Sea ice volume is calculated wherever siconc is greater than or equal
        to low AND less than or equal to high. Default = [0., 1.]. Must be in
        the same units as siconc.

    units : float, default = 1.E-12
        Scaling factor multiplied at the end of the calculation. Default puts
        the value in 10^3 km^3, assuming that siconc is a fraction (rather than
        percentage), areacello is in m^2, and sithick is in m.


    Returns
    -------
    sivolu : array of float, of shape (nt,)

    """

    sivolu = np.nansum( areacello[np.newaxis,:,:] * sithick
                        * (siconc >= siconc_threshold[0])
                        * (siconc <= siconc_threshold[1]),
                        axis=(1,2)) * units

    return sivolu


def divergence(dx, dy, areacello, u, v):
    """Compute the 2D divergence of a vector field defined on the U-grid of
    CICE (points located at northeast corners of cells, i.e., the Arakawa B
    grid) as a function of time and space, consistent with the internal
    dynamics and transport scheme of CICE.


    Parameters
    ----------
    dx : array, shape (ny, nx)
        T-cell widths on the north side (CICE grid variable HTN)

    dy : array, shape (ny, nx)
        T-cell heights on the east side (CICE grid variable HTE).
        These must be in the same units as dx.

    areacello : array, shape (ny, nx)
        T-cell areas (CICE grid variable tarea).
        These must be in units of [dx]*[dy] == [dx]**2 == [dy]**2.

    u : array, shape (nt, ny, nx)
        Component of the vector field to compute the divergence of, that
        aligned along the x dimension and located on the U-grid (e.g., UVEL).

    v : array, shape (nt, ny, nx)
        Component of the vector field to compute the divergence of, that
        aligned along the y dimension and located on the U-grid (e.g., VVEL).
        Must be in the same units as u.


    Returns
    -------
    div_u : array, shape (nt, ny, nx)
        Divergence of (u, v) on the T-grid in units of [u]/[x].


    Notes
    -----
    The velocity field (u, v) should not contain missing values: for land, set
    missing points to zero.

    The divergence of (u, v) is found by computing its integral flux across
    T-cell boundaries, then dividing by T-cell areas, which follows from the
    divergence theorem. Consistently with CICE, velocity components are assumed
    to vary linearly from one point to the next, e.g., from u[j-1,i] to u[j,i]
    (see CICE grid documentation).

    Inputs u and v can also be 2D (i.e., no time coordinate). In this case the
    return value div_u is also 2D.

    """

    if np.ndim(u) == 2:
        u = np.array([u])
        v = np.array([v])
        reshape = True
    else:
        reshape = False

    nt, ny, nx = np.shape(u)

    # Arrays for the north (N), east (E), south (S), and west (W) fluxes (F)
    # across each T-cell boundary:
    NF = np.zeros((nt, ny, nx))
    EF = np.zeros((nt, ny, nx))
    SF = np.zeros((nt, ny, nx))
    WF = np.zeros((nt, ny, nx))

    NF[:,:,1:]  = .5 * (v[:,:,1:] + v[:,:,:-1]) * dx[np.newaxis,:,1:]
    NF[:,:,0]   = .5 * v[:,:,0] * dx[np.newaxis,:,0]

    EF[:,1:,:]  = .5 * (u[:,1:,:] + u[:,:-1,:]) * dy[np.newaxis,1:,:]
    EF[:,0,:]   = .5 * u[:,0,:] * dy[np.newaxis,0,:]

    SF[:,1:,1:] = .5 * (v[:,:-1,1:] + v[:,:-1,:-1]) * dx[np.newaxis,:-1,1:]
    SF[:,1:,0]  = .5 * v[:,:-1,0] * dx[np.newaxis,:-1,0]

    WF[:,1:,1:] = .5 * (u[:,1:,:-1] + u[:,:-1,:-1]) * dy[np.newaxis,1:,:-1]
    WF[:,0,1:]  = .5 * u[:,0,:-1] * dy[np.newaxis,0,:-1]

    div_u = (NF - SF + EF - WF) / areacello[np.newaxis,:,:]

    if reshape:
        u = np.squeeze(u)
        v = np.squeeze(v)
        div_u = np.squeeze(div_u)

    return div_u


def curl(dx, dy, areacello, u, v):
    """Compute the 2D curl of a vector field defined on the U-grid of CICE
    (points located at northeast corners of cells, i.e., the Arakawa B grid) as
    a function of time and space, consistent with the internal dynamics and
    transport scheme of CICE.


    Parameters
    ----------
    dx : array, shape (ny, nx)
        T-cell widths on the north side (CICE grid variable HTN)

    dy : array, shape (ny, nx)
        T-cell heights on the east side (CICE grid variable HTE).
        These must be in the same units as dx.

    areacello : array, shape (ny, nx)
        T-cell areas (CICE grid variable tarea).
        These must be in units of [dx]*[dy] == [dx]**2 == [dy]**2.

    u : array, shape (nt, ny, nx)
        Component of the vector field to compute the curl of, that aligned
        along the x dimension and located on the U-grid (e.g., UVEL).

    v : array, shape (nt, ny, nx)
        Component of the vector field to compute the curl of, that aligned
        along the y dimension and located on the U-grid (e.g., VVEL).


    Returns
    -------
    curl_u : array, shape (nt, ny, nx)
        Curl of (u, v) on the T-grid in units of [u]/[x].


    Notes
    -----
    The velocity field (u, v) should not contain missing values: for land, set
    missing points to zero.

    The curl of (u, v) is found by computing the line integral (circulation) of
    (u, v) anticlockwise around T-cell boundaries, then dividing by T-cell
    areas, which follows from Stokes' theorem. Consistently with CICE, velocity
    components are assumed to vary linearly from one point to the next, e.g.,
    from u[j-1,i] to u[j,i] (see CICE grid documentation).

    """

    if np.ndim(u) == 2:
        u = np.array([u])
        v = np.array([v])
        reshape = True
    else:
        reshape = False

    nt, ny, nx = np.shape(u)

    # Arrays for the magnitudes of the north (N), east (E), south (S), and west (W)
    # components (C) of the line integral around each T-cell boundary:
    NC = np.zeros((nt, ny, nx))
    EC = np.zeros((nt, ny, nx))
    SC = np.zeros((nt, ny, nx))
    WC = np.zeros((nt, ny, nx))
    
    NC[:,:,1:]  = .5 * (u[:,:,1:] + u[:,:,:-1]) * dx[np.newaxis,:,1:]
    NC[:,:,0]   = .5 * u[:,:,0] * dx[np.newaxis,:,0]

    EC[:,1:,:]  = .5 * (v[:,1:,:] + v[:,:-1,:]) * dy[np.newaxis,1:,:]
    EC[:,0,:]   = .5 * v[:,0,:] * dy[np.newaxis,0,:]

    SC[:,1:,1:] = .5 * (u[:,:-1,1:] + u[:,:-1,:-1]) * dx[np.newaxis,:-1,1:]
    SC[:,1:,0]  = .5 * u[:,:-1,0] * dx[np.newaxis,:-1,0]

    WC[:,1:,1:] = .5 * (v[:,1:,:-1] + v[:,:-1,:-1]) * dy[np.newaxis,1:,:-1]
    WC[:,0,1:]  = .5 * v[:,0,:-1] * dy[np.newaxis,0,:-1]

    curl_u = (-NC + SC + EC - WC) / areacello[np.newaxis,:,:]

    if reshape:
        u = np.squeeze(u)
        v = np.squeeze(v)
        curl_u = np.squeeze(curl_u)

    return curl_u

