# -*- coding: utf-8 -*-

__all__ = [
    "format_to_string",
    "get_errorbars",
    "quantiles",
    "binned_CDFs_from_realizations",
    "unbinned_empirical_cdf",
    "get_corresponding_y_value",
    "log_to_lin",
    "lin_to_log",
]

import logging
import numpy as np

log = logging.getLogger(__name__)


def format_to_string(
    value,
    uncertainty=None,
    lim_type=None,
    exponent=None,
    value_precision=2,
    uncertainty_precision=2,
):
    """Summary

    Parameters
    ----------
    value : TYPE
        Description
    uncertainty : None, optional
        Description
    lim_type : None, optional
        Description
    value_precision : int, optional
        Description
    uncertainty_precision : int, optional
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    ValueError
        Description
    """
    if uncertainty is not None and lim_type is not None:
        log.warning(
            "Both uncertainty and limits are not None, "
            "make sure you know what you are doing."
        )

    if isinstance(exponent, int):
        value = value / 10 ** int(exponent)
        if isinstance(uncertainty, (float, int)):
            uncertainty = float(uncertainty) / 10 ** int(exponent)
        elif isinstance(uncertainty, (list, np.ndarray)):
            uncertainty = np.array(uncertainty).astype(float) / 10 ** int(exponent)

    val_str, exponent = f"{value:.{value_precision:d}e}".split("e")

    # Don't print exponent if its only 0.1, 1, or 10
    if int(exponent) in [-1, 0, 1]:
        exponent_str = ""
        val_str = f"{float(val_str)*10**int(exponent):.{value_precision:d}f}"
    else:
        exponent_str = rf" \times 10^{{{int(exponent):d}}}"

    # Error
    if uncertainty is None:
        unc_str = ""
    elif isinstance(uncertainty, (float, int)):
        uncertainty = float(uncertainty)
        # Don't print exponent if its only 0.1, 1, or 10
        if int(exponent) in [-1, 0, 1]:
            uncertainty = uncertainty * 10 ** int(exponent)
        uncp = f"{uncertainty/10**int(exponent):.{uncertainty_precision:d}f}"
        uncm = uncp
        unc_str = rf"^{{+{uncp}}}_{{-{uncm}}}"
    elif isinstance(uncertainty, (list, np.ndarray)):
        uncertainty = np.array(uncertainty).astype(float)
        # Don't print exponent if its only 0.1, 1, or 10
        if int(exponent) in [-1, 0, 1]:
            uncertainty = uncertainty * 10 ** int(exponent)
        uncm = f"{uncertainty[0]/10**int(exponent):.{uncertainty_precision:d}f}"
        uncp = f"{uncertainty[1]/10**int(exponent):.{uncertainty_precision:d}f}"
        unc_str = f"^{{+{uncp}}}_{{-{uncm}}}"
    else:
        raise ValueError("Invalid input for uncertainty")

    # Limit
    if lim_type is None:
        lim_str = ""
    elif lim_type == "upper":
        lim_str = "< "
    elif lim_type == "lower":
        lim_str = "> "
    else:
        raise ValueError("Invalid input for lim_type")

    formatted_string = rf"${lim_str}{val_str}{unc_str}{exponent_str}$"

    return formatted_string


def get_errorbars(data, confidence=0.68, axis=0):
    """Summary

    Parameters
    ----------
    data : TYPE
        Description
    confidence : float, optional
        Description
    axis : int, optional
        Description

    Returns
    -------
    TYPE
        Description

    Deleted Parameters
    ------------------
    data_MC : TYPE
        Description
    """
    value, low, upp = quantiles(data, confidence=confidence, axis=axis)
    uncm = value - low
    uncp = upp - value
    return value, uncm, uncp


def quantiles(array, confidence=0.68, axis=0):
    """
    Convenience function to quickly calculate the quantiles defined
    by the confidence level, along a given axis of an array.
    This is meant to be used with a 2D array of CDF or histogram of
    realizations of a given sample of size (N_real, N_bins), where:

    - N_real stands for the number of realizations (MC or bootstrap)
    - N_bins stands for the number of bins within the histogram/CDF

    If the input array is of the size (N_bins, N_real) instead,
    either provide the transpose of that array or use axis=1.

    Returns the median, lower and upper bounds of the histogram/CDF
    (for each bin) at the desired confidence level.
    """

    # Create percentiles:
    lower_percentile = (1.0 - confidence) / 2.0
    upper_percentile = 1.0 - lower_percentile
    # Compute the percentiles for each bin
    median = np.quantile(array, 0.5, axis=axis)
    lower = np.quantile(array, lower_percentile, axis=axis)
    upper = np.quantile(array, upper_percentile, axis=axis)
    return median, lower, upper


def binned_CDFs_from_realizations(realizations, bins=1000, weights=None):
    """
    Takes in a 2D array of size (N_real, N_sample) and computes the
    CDF for each realization. The CDFs are sampled given a certain
    precision determined from the bins; this is necessary for the
    different CDFs of each realization to be defined on the same
    grid/domain. This allows to create confidence intervals on each
    bin/element of the grid.

    - N_real stands for the number of realizations (MC or bootstrap)
    - N_sample stands for the number of objects in the sample

    If the input array is of the size (N_sample, N_real) instead,
    provide the transpose of that array.
    """
    N_real = realizations.shape[0]

    if isinstance(bins, (int, float)):
        bins = np.linspace(realizations.min(), realizations.max(), bins + 1)

    precision = bins.shape[0] - 1

    # Compute the CDF for each subsample realization
    CDFs = np.zeros((N_real, precision))
    for i in range(N_real):
        if weights is None:
            hist, bins_ = np.histogram(realizations[i], bins=bins, weights=None)
        else:
            hist, bins_ = np.histogram(realizations[i], bins=bins, weights=weights[i])

        CDFs[i, :] = np.cumsum(hist).astype(float) / float(np.sum(hist))

    return bins, CDFs


def unbinned_empirical_cdf(data, weights=None):
    """
    From the answer of Dave at http://stackoverflow.com/questions/3209362/how-to-plot-empirical-cdf-in-matplotlib-in-python
    Note : if you wish to plot, use arg drawstyle='steps-post', I found it is the most accurate
    description of the data.

    Parameters
    ----------
    data : array
        The data to convert into a Cumulative Distribution Function.
    weights : array, optional
        The weights for the data.

    Returns:
    ---------
    sorted_data : array
        The array of the data sorted.
    CDF : array
        The cumulative distribution function that follows the formal
        definition of CDF(x) = "nb samples <= x" / "nb samples"
    """
    sorted_data = np.sort(data)
    N = len(data)
    if weights is None:
        CDF = np.arange(1, N + 1) / float(N)
    else:
        # create 2D array with data and weights
        arr = np.column_stack((data, weights))
        # Sort them by ascending data, need to use this method and not np.sort()
        arr = arr[arr[:, 0].argsort()]
        CDF = np.cumsum(arr[:, 1]).astype(float)
        CDF /= CDF[-1]
    return sorted_data, CDF


def get_corresponding_y_value(x_val, x, y):
    """
    Find the y value for a given x value for a pair of finite (x,y) arrays.
    If the x value is not in the x array, a linear interpolation is assumed.
    """

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if isinstance(x_val, np.ndarray):
        invalid_values = any(x_val < np.min(x)) or any(x_val > np.max(x))
    elif isinstance(x_val, (float, int)):
        invalid_values = (x_val < np.min(x)) or (x_val > np.max(x))
    if invalid_values:
        raise ValueError(
            "Values searched must be within bounds of the array, cannot extrapolate."
        )

    y_step = y[1:] - y[:-1]
    i = x.searchsorted(x_val, side="left")
    # if x values are in x, return corresponding y values
    # otherwise, linearly interpolate between the bounding values
    # Beware of special case if index of x value is 0
    # to avoid unwanted behavior
    y_val = np.where(
        i == 0, y[i], y[i - 1] + y_step[i - 1] * (x_val - x[i - 1]) / (x[i] - x[i - 1])
    )
    return y_val


def log_to_lin(log_x, log_x_uncp, log_x_uncm=None):
    """
    Takes logscale data with uncertainties and converts to linear scale with correct uncertainty propagation.
    If log_x_uncm is not provided, uncertainties are assumed symmetric.
    Returns : x, x_uncp, x_uncm

    Parameters
    ----------
    log_x : int, float, array-like
        The logarithmic value or array to convert to linear.
    log_x_uncp : int, float, array-like
        The positive uncertainty in logscale.
    log_x_uncm : int, float, array-like, optional
        The negative uncertainty in logscale.

    Returns
    -------
    TYPE
        x, x_uncp, x_uncm
    """
    if log_x_uncm is None:
        log_x_uncm = log_x_uncp
    x = 10**log_x
    x_uncp = x * (10**log_x_uncp - 1.0)
    x_uncm = x * (1.0 - 10 ** (-log_x_uncm))

    return x, x_uncp, x_uncm


def lin_to_log(x, x_uncp, x_uncm=None):
    """
    Takes linear data with uncertainties and converts to logscale with correct uncertainty propagation.
    If x_uncm is not provided, uncertainties are assumed symmetric.
    Returns : log_x, log_x_uncp, log_x_uncm
    """
    if x_uncm is None:
        x_uncm = x_uncp
    log_x = np.log10(x)
    log_x_uncp = np.log10((x + x_uncp) / x)
    log_x_uncm = np.log10(x / (x - x_uncm))

    return log_x, log_x_uncp, log_x_uncm
