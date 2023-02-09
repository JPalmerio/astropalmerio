import numpy as np
import logging

log = logging.getLogger(__name__)


def extract_1D_from_2D(wvlg, spatial, flux, spatial_bounds, error=None):
    """Summary

    Parameters
    ----------
    wvlg : 1D array-like
        Value of the spectral dimension
    spatial : 1D array-like
        Value of the spatial dimension
    flux : 2D array-like
        2D flux spectrum of size (len(spatial), len(wvlg)).
        First dimension is assumed to be the spatial
        dimension and second dimension the spectral one.
    spatial_bounds : tuple (float, float)
        Lower and upper bound of the spatial extraction.
    error : 2D array-like, optional, Default None
        2D error spectrum. If None, will return 0 for the extracted error

    Returns
    -------
    wvlg, extracted_flux, extracted_error : 3-tuple of array-like of size len(flux).
        Tuple of the spectral dimension, extracted flux and error.
    """

    # Finds the index corresponding to the min and max spatial positions
    index_min = spatial.searchsorted(spatial_bounds[0]) - 1
    index_max = spatial.searchsorted(spatial_bounds[1]) - 1
    if index_min >= index_max:
        temp = index_max
        index_max = index_min
        index_min = temp

    # Extract the 1D data
    extracted_flux = np.zeros(flux.shape[1])
    # Sum along the spatial direction
    for i in range(index_max - index_min):
        extracted_flux += flux[index_min + i]

    # Same for error
    if error is None:
        extracted_error = np.zeros(flux.shape[1])
    else:
        extracted_error = np.zeros(error.shape[1])
        for i in range(index_max - index_min):
            extracted_error += (
                error[index_min + i] ** 2
            )  # quadratic sum for error propagation
        extracted_error = np.sqrt(
            extracted_error
        )  # quadratic sum for error propagation

    return wvlg, extracted_flux, extracted_error
