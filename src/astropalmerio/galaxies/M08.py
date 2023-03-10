# -*- coding: utf-8 -*-

__all__ = [
    "R23",
    "OIIIb_Hbeta",
    "OII_Hbeta",
    "OIIIb_OII",
    "NeIII_OII",
    "NII_Ha",
    "OIIIb_NII",
    "R23_rec",
    "R23_scatter_rec",
    "OIIIb_Hbeta_rec",
    "OIIIb_Hbeta_scatter_rec",
    "OII_Hbeta_rec",
    "OII_Hbeta_scatter_rec",
    "OIIIb_OII_rec",
    "OIIIb_OII_scatter_rec",
    "NeIII_OII_rec",
    "NeIII_OII_scatter_rec",
    "NII_Halpha_rec",
    "NII_Halpha_scatter_rec",
    "OIIIb_NII_rec",
    "OIIIb_NII_scatter_rec",
    "plot_relations",
]

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


Maio_file = "./data/met_cal_for_M08_method.txt"
Z_sun = 8.69
font = {"size": 15}


def read_column(
    filename,
    column_nb,
    dtype=float,
    array=True,
    splitter=None,
    stripper=None,
    verbose=False,
):
    """
    Function used to read ASCII (or fits) files.
    It will skip lines starting with '#', '!' or '%'.

    Parameters
    ----------
    filename : [str]
        Name of the file containing the data

    column_nb: [int]
        Number of the column for the data.

    dtype : [data-type]
        Type of the returned data. Default is float.

    array : [bool]
        If True returns xdata as an array rather than a list. Default is True (arrays are faster).

    splitter : [str]
        String to use as a delimiter between columns. Default is None (uses default for str.split() which is a whitespace).

    stripper : [str]
        String to strip at the beginning and end of each line. Default is None (uses default for str.strip() which is a whitespace).

    Returns
    -------
    xdata : [array/list]
        The data read by the function as a 1D array or list.
    """

    nan = False

    xdata = []
    f = open(filename, "r")
    for line in f:
        if len(line) != 0:
            if line[0] != "#" and line[0] != "!" and line[0] != "%":
                if stripper is not None:
                    line = line.strip(stripper)
                else:
                    line = line.strip()
                if splitter is not None:
                    columns = line.split(splitter)
                else:
                    columns = line.split()
                try:
                    xdata.append(dtype(columns[column_nb]))
                except ValueError:
                    nan = True
                    xdata.append(np.nan)
    if array:
        xdata = np.asarray(xdata, dtype=dtype)
    f.close()
    if nan:
        if verbose:
            print(
                "[Warning] in read_column for %s. Could not convert string to %s, so added NaN."
                % (filename, dtype)
            )

    return xdata


def R23(F_OII_3727, F_OIII_4959, F_OIII_5007, F_Hbeta, verbose=True):
    """
    R23 method. Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [0.7462, -0.7149, -0.9401, -0.6154, -0.2524]
    log_R23 = np.log10((F_OII_3727 + F_OIII_4959 + F_OIII_5007) / F_Hbeta)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = (
        c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 + c[4] * x1**4 - log_R23
    )

    # Find solution by iteration
    while polyn <= 0.0:
        x1 += 0.01
        polyn = (
            c[0]
            + c[1] * x1
            + c[2] * x1**2
            + c[3] * x1**3
            + c[4] * x1**4
            - log_R23
        )
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in R23 lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = (
        c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 + c[4] * x2**4 - log_R23
    )
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = (
            c[0]
            + c[1] * x2
            + c[2] * x2**2
            + c[3] * x2**3
            + c[4] * x2**4
            - log_R23
        )
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in R23 upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def OIIIb_Hbeta(F_OIII_5007, F_Hbeta, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [0.1549, -1.5031, -0.9790, -0.0297]
    log_R = np.log10(F_OIII_5007 / F_Hbeta)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 - log_R

    # Find solution by iteration
    while polyn <= 0.0:
        x1 += 0.01
        polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 - log_R
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in OIIIb_Hbeta lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 - log_R
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in OIIIb_Hbeta upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def OII_Hbeta(F_OII_3727, F_Hbeta, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [0.5603, 0.0450, -1.8017, -1.8434, -0.6549]
    log_R = np.log10(F_OII_3727 / F_Hbeta)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 + c[4] * x1**4 - log_R

    # Find solution by iteration
    while polyn <= 0.0:
        x1 += 0.01
        polyn = (
            c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 + c[4] * x1**4 - log_R
        )
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in OII_Hbeta lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 + c[4] * x2**4 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = (
            c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 + c[4] * x2**4 - log_R
        )
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in OII_Hbeta upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def OIIIb_OII(F_OIII_5007, F_OII_3727, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [-0.2893, -1.3881, -0.3172]
    log_R = np.log10(F_OIII_5007 / F_OII_3727)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 - log_R

    # Find solution by iteration
    while (
        polyn >= 0.0
    ):  # careful of the sign of inequality here ! Depends on monotony of the polynome
        x1 += 0.01
        polyn = c[0] + c[1] * x1 + c[2] * x1**2 - log_R
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in OIIIb_OII3727 lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = c[0] + c[1] * x2 + c[2] * x2**2 - log_R
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in OIIIb_OII3727 upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def NeIII_OII(F_NeIII_3870, F_OII_3727, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [-1.2608, -1.0861, -0.1470]
    log_R = np.log10(F_NeIII_3870 / F_OII_3727)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 - log_R

    # Find solution by iteration
    while (
        polyn >= 0.0
    ):  # careful of the sign of inequality here ! Depends on monotony of the polynome
        x1 += 0.01
        polyn = c[0] + c[1] * x1 + c[2] * x1**2 - log_R
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in NeIII3870_OII3727 lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = c[0] + c[1] * x2 + c[2] * x2**2 - log_R
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in NeIII3870_OII3727 upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def NII_Ha(F_NII_6583, F_Halpha_6528, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [-0.7732, 1.2357, -0.2811, -0.7201, -0.3330]
    log_R = np.log10(F_NII_6583 / F_Halpha_6528)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 + c[4] * x1**4 - log_R

    # Find solution by iteration
    while (
        polyn >= 0.0
    ):  # careful of the sign of inequality here ! Depends on monotony of the polynome
        x1 += 0.01
        polyn = (
            c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 + c[4] * x1**4 - log_R
        )
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in NII6583_Halpha6528 lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = (
            c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 + c[4] * x2**4 - log_R
        )
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in NII6583_Halpha6528 upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def OIIIb_NII(F_OIII_5007, F_NII_6583, verbose=True):
    """
    Flux is expected in units of erg/s/cm2 (although it doesn't matter because this function calculates a ratio of fluxes).
    Returns lower branch and upper branch metallicities in units of 12 + log(O/H).
    """

    c = [0.4520, -2.6096, -0.7170, 0.1347]
    log_R = np.log10(F_OIII_5007 / F_NII_6583)

    x1 = 7.0 - Z_sun
    x2 = 9.3 - Z_sun  # Should probably use 9.1 instead of 9.3 here (limit of the fit)
    polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 - log_R

    # Find solution by iteration
    while polyn <= 0.0:
        x1 += 0.01
        polyn = c[0] + c[1] * x1 + c[2] * x1**2 + c[3] * x1**3 - log_R
        if x1 > 1:  # Equivalent to metallicity > 9.69
            if verbose:
                print(
                    "Warning in OIIIb_NII lower branch : no solution found for metallicity."
                )
            # polyn = 0.
            x1 = np.nan
            break

    polyn = c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 - log_R
    while polyn <= 0.0:
        x2 -= 0.01
        polyn = c[0] + c[1] * x2 + c[2] * x2**2 + c[3] * x2**3 - log_R
        if x2 < -2:  # Equivalent to metallicity < 6.69
            if verbose:
                print(
                    "Warning in OIIIb_NII upper branch : no solution found for metallicity."
                )
            # polyn = 0.
            x2 = np.nan
            break

    lower_branch = x1 + Z_sun
    upper_branch = x2 + Z_sun

    return lower_branch, upper_branch


def R23_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [0.7462, -0.7149, -0.9401, -0.6154, -0.2524]
    x = Z - Z_sun

    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


def R23_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    R23_unc = read_column(Maio_file, 8)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = R23_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = R23_unc[j]
    return scatter


def OIIIb_Hbeta_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [0.1549, -1.5031, -0.9790, -0.0297]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3


def OIIIb_Hbeta_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    OIIIb_Hb_unc = read_column(Maio_file, 4)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = OIIIb_Hb_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = OIIIb_Hb_unc[j]
    return scatter


def OII_Hbeta_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [0.5603, 0.0450, -1.8017, -1.8434, -0.6549]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


def OII_Hbeta_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    OII_Hb_unc = read_column(Maio_file, 2)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = OII_Hb_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = OII_Hb_unc[j]
    return scatter


def OIIIb_OII_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [-0.2893, -1.3881, -0.3172]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2


def OIIIb_OII_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    OIIIb_OII_unc = read_column(Maio_file, 6)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = OIIIb_OII_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = OIIIb_OII_unc[j]
    return scatter


def NeIII_OII_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [-1.2608, -1.0861, -0.1470]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2


def NeIII_OII_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    NeIII_OII_unc = read_column(Maio_file, 10)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = NeIII_OII_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = NeIII_OII_unc[j]
    return scatter


def NII_Halpha_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [-0.7732, 1.2357, -0.2811, -0.7201, -0.3330]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


def NII_Halpha_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    NII_Ha_unc = read_column(Maio_file, 12)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = NII_Ha_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = NII_Ha_unc[j]
    return scatter


def OIIIb_NII_rec(Z):
    """
    Returns the expected LogR value for the given metallicity.
    """
    c = [0.4520, -2.6096, -0.7170, 0.1347]
    x = Z - Z_sun
    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3


def OIIIb_NII_scatter_rec(Z):
    """
    Returns the expected scatter on LogR value for the given metallicity.
    """

    # all of these are in log scale
    Z_to_plot = read_column(Maio_file, 0)
    OIIIb_NII_unc = read_column(Maio_file, 14)

    if isinstance(Z, np.ndarray):
        scatter = np.zeros(len(Z))
        for i in range(len(Z)):
            j = Z_to_plot.searchsorted(Z[i])
            scatter[i] = OIIIb_NII_unc[j]
    else:
        j = Z_to_plot.searchsorted(Z)
        scatter = OIIIb_NII_unc[j]
    return scatter


def plot_relations(ratio_list=None, ax=None):
    """
        Helper function to easily plot the various calibrators from Maiolino et al. 2008.

    Parameters:
    -----------
    ratio_list : [list]
        List to include a certain calibrator or not.
        If the value "i" is None, the calibrator "i" is not plotted.
        The order is : [R23, OIIIb_Hb, OII_Hb, OIIIb_OII, NeIII_OII, NII_Ha, OIIIb_NII]
        Default is None, in which case all are plotted.

    ax : [axes]
        Axes instance from matplotlib on which to draw the plots.

    Returns:
    --------

    art_list : [list]
        List of artists drawn on ax.

    """

    if ax is None:
        fig = plt.figure(tight_layout=True, figsize=(7, 6))
        ax = fig.add_subplot(111)

    if ratio_list is None:
        ratio_list = [True for i in range(7)]

    Z_to_plot = read_column(Maio_file, 0)
    OII_Hb = read_column(Maio_file, 1)
    OII_Hb_unc = read_column(Maio_file, 2)
    OIIIb_Hb = read_column(Maio_file, 3)
    OIIIb_Hb_unc = read_column(Maio_file, 4)
    OIIIb_OII = read_column(Maio_file, 5)
    OIIIb_OII_unc = read_column(Maio_file, 6)
    R23 = read_column(Maio_file, 7)
    R23_unc = read_column(Maio_file, 8)
    NeIII_OII = read_column(Maio_file, 9)
    NeIII_OII_unc = read_column(Maio_file, 10)
    NII_Ha = read_column(Maio_file, 11)
    NII_Ha_unc = read_column(Maio_file, 12)
    OIIIb_NII = read_column(Maio_file, 13)
    OIIIb_NII_unc = read_column(Maio_file, 14)

    colors = ["C{:1d}".format(i) for i in range(7)]
    labels = [
        "R23",
        "OIIIb/Hb",
        "OII/Hb",
        "OIIIb/OII",
        "NeIII/OII",
        "NII/Ha",
        "OIIIb/NII",
    ]
    obs_calib = [R23, OIIIb_Hb, OII_Hb, OIIIb_OII, NeIII_OII, NII_Ha, OIIIb_NII]
    obs_calib_unc = [
        R23_unc,
        OIIIb_Hb_unc,
        OII_Hb_unc,
        OIIIb_OII_unc,
        NeIII_OII_unc,
        NII_Ha_unc,
        OIIIb_NII_unc,
    ]
    art_list = []

    for index, ratio in enumerate(ratio_list):
        if ratio is not None:
            (plot,) = ax.plot(
                Z_to_plot, obs_calib[index], color=colors[index], label=labels[index]
            )
            ax.fill_between(
                Z_to_plot,
                obs_calib[index] - obs_calib_unc[index],
                obs_calib[index] + obs_calib_unc[index],
                color=plt.getp(plot, "color"),
                alpha=0.1,
            )
            art_list.append(plot)

    ax.legend(loc="best")
    ax.set_xlabel("12 + log(O/H)", **font)
    ax.set_ylabel("Ratio", **font)

    return art_list
