"""Sub-package for reading in the various sources of data and carrying out any
pre-processing. Also provides the function 'rotate_vectors' to transform vector
components defined on the native CICE grid to regular longitude/latitude
components. This is needed for both CICE history outputs (module cice.py), such
as ice drift or atmosphere-ice strain, and the atmospheric wind field (read in
by functions in module atmo.py).
"""

import netCDF4 as nc
import numpy as np

from ..io import config as cfg

def rotate_vectors(u_in, v_in):
    """Rotate vectors with components (u_in, v_in) on the native CICE grid to
    regular lon/lat-components (u_out, v_out) using the sine and cosine of the
    CICE grid coordinate angle.
    """

    if np.ndim(u_in) == 2:
        ny = np.shape(u_in)[0]  # could be 129 or 131
    elif np.ndim(u_in) == 3:
        ny = np.shape(u_in)[1]  # could be 129 or 131
    else:
        raise ValueError("src.data.rotate_vectors(): inputs must be 2D or 3D")

    with nc.Dataset(cfg.data_path[f"cosa{ny}"], "r") as ncdat:
        cosa = np.array(ncdat.variables["cosa"])

    with nc.Dataset(cfg.data_path[f"sina{ny}"], "r") as ncdat:
        sina = np.array(ncdat.variables["sina"])

    u_out = u_in*cosa - v_in*sina
    v_out = u_in*sina + v_in*cosa

    return u_out, v_out

