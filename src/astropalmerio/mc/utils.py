import logging
import numpy as np

log = logging.getLogger(__name__)


def format_to_string(
    value,
    error=None,
    lim_type=None,
    exponent=None,
    value_precision=2,
    error_precision=2,
):
    """Summary

    Parameters
    ----------
    value : TYPE
        Description
    error : None, optional
        Description
    lim_type : None, optional
        Description
    value_precision : int, optional
        Description
    error_precision : int, optional
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
    if error is not None and lim_type is not None:
        log.warning(
            "Both error and limits are not None, "
            "make sure you know what you are doing."
        )

    if isinstance(exponent, int):
        value = value / 10 ** int(exponent)
        if isinstance(error, (float, int)):
            error = float(error) / 10 ** int(exponent)
        elif isinstance(error, (list, np.ndarray)):
            error = np.array(error).astype(float) / 10 ** int(exponent)

    val_str, exponent = f"{value:.{value_precision:d}e}".split("e")

    # Don't print exponent if its only 0.1, 1, or 10
    if int(exponent) in [-1, 0, 1]:
        exponent_str = ""
        val_str = f"{float(val_str)*10**int(exponent):.{value_precision:d}f}"
    else:
        exponent_str = rf" \times 10^{{{int(exponent):d}}}"

    # Error
    if error is None:
        err_str = ""
    elif isinstance(error, (float, int)):
        error = float(error)
        # Don't print exponent if its only 0.1, 1, or 10
        if int(exponent) in [-1, 0, 1]:
            error = error * 10 ** int(exponent)
        errp = f"{error/10**int(exponent):.{error_precision:d}f}"
        errm = errp
        err_str = rf"^{{+{errp}}}_{{-{errm}}}"
    elif isinstance(error, (list, np.ndarray)):
        error = np.array(error).astype(float)
        # Don't print exponent if its only 0.1, 1, or 10
        if int(exponent) in [-1, 0, 1]:
            error = error * 10 ** int(exponent)
        errm = f"{error[0]/10**int(exponent):.{error_precision:d}f}"
        errp = f"{error[1]/10**int(exponent):.{error_precision:d}f}"
        err_str = f"^{{+{errp}}}_{{-{errm}}}"
    else:
        raise ValueError("Invalid input for error")

    # Limit
    if lim_type is None:
        lim_str = ""
    elif lim_type == "upper":
        lim_str = "< "
    elif lim_type == "lower":
        lim_str = "> "
    else:
        raise ValueError("Invalid input for lim_type")

    formatted_string = rf"${lim_str}{val_str}{err_str}{exponent_str}$"

    return formatted_string


def get_errorbars(data_MC):
    """Summary

    Parameters
    ----------
    data_MC : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    value, low, upp = quantiles(data_MC)
    errm = value - low
    errp = upp - value
    return value, errm, errp


def quantiles(array_2D, confidence=95.0, axis=0):
    """
    Convenience function to quickly calculate the quantiles defined
    by the confidence level, along a given axis of a 2D array.
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
    lower_percentile = (1.0 - confidence / 100.0) / 2.0
    upper_percentile = 1.0 - lower_percentile
    # Compute the percentiles for each bin
    median = np.quantile(array_2D, 0.5, axis=axis)
    lower = np.quantile(array_2D, lower_percentile, axis=axis)
    upper = np.quantile(array_2D, upper_percentile, axis=axis)
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


def log_to_lin(log_x, log_x_errp, log_x_errm=None):
    """
    Takes logscale data with errors and converts to linear scale with correct error propagation.
    If log_x_errm is not provided, errors are assumed symmetric.
    Returns : x, x_errp, x_errm

    Parameters
    ----------
    log_x : int, float, array-like
        The logarithmic value or array to convert to linear.
    log_x_errp : int, float, array-like
        The positive error in logscale.
    log_x_errm : int, float, array-like, optional
        The negative error in logscale.

    Returns
    -------
    TYPE
        x, x_errp, x_errm
    """
    if log_x_errm is None:
        log_x_errm = log_x_errp
    x = 10**log_x
    x_errp = x * (10**log_x_errp - 1.0)
    x_errm = x * (1.0 - 10 ** (-log_x_errm))

    return x, x_errp, x_errm


def lin_to_log(x, x_errp, x_errm=None):
    """
    Takes linear data with errors and converts to logscale with correct error propagation.
    If x_errm is not provided, errors are assumed symmetric.
    Returns : log_x, log_x_errp, log_x_errm
    """
    if x_errm is None:
        x_errm = x_errp
    log_x = np.log10(x)
    log_x_errp = np.log10((x + x_errp) / x)
    log_x_errm = np.log10(x / (x - x_errm))

    return log_x, log_x_errp, log_x_errm
