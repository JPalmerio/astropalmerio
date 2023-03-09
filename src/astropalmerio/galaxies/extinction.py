# -*- coding: utf-8 -*-

__all__ = [
    "Cardelli89",
    "Pei92_MW",
    "Pei92_SMC",
    "Calzetti_SB",
    "get_extinction",
    "extinguish_line",
    "correct_line_for_extinction",
]

import logging
import numpy as np
from astropy.units.quantity import Quantity

log = logging.getLogger(__name__)


def Cardelli89(wvlg, Rv=3.1):
    """
    Computes the extinction from [Cardelli+89]
    (https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract),
    equation (1), (2a), (2b), (3a), (3b).
    Valid from 3000 Angstrom to 33000 Angstrom.

    Parameters
    ----------
    wvlg : float, array-like, astropy Quantity
        Wavelength at which to estimate the extinction. If not using an
        astropy `Quantity` object, units are assumed to be Angstrom.
    Rv : float, optional, Default=3.1
        The Rv value to use.

    Returns
    -------
    extinction : array-like
        The value of the extinction calculated.
    """

    # Convert to microns^-1
    wvlg = _convert_to_microns(wvlg)
    x = 1.0 / wvlg

    if not isinstance(Rv, (float, int)):
        raise ValueError("Rv should be a single float or int.")

    if any(wvlg < 3300.0) or any(wvlg > 33000):
        log.warning(
            "Invalid wavelength (must be between 3300 A and 33 000 A)!"
            " Extinction returned will be zero outside these bounds."
        )

    y = x - 1.82

    # Infrared
    a = np.where((0.3 <= x) & (x <= 1.1), 0.574 * x**1.61, 0)
    # Optical
    a = np.where(
        (1.1 <= x) & (x <= 3.3),
        (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        ),
        a,
    )

    # Infrared
    b = np.where((0.3 <= x) & (x <= 1.1), -0.527 * x**1.61, 0)
    # Optical
    b = np.where(
        (1.1 <= x) & (x <= 3.3),
        (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        ),
        b,
    )

    extinct = a + b / Rv

    return extinct


def Pei92_MW(wvlg):
    """
    Milky Way extinction curve from Pei+92.
    wvlg is expected in Angstrom.
    Valid from 1000 Angstroms.
    """

    x = _convert_to_microns(wvlg)

    extinct = 1.32 * (
        165.0 / ((x / 0.047) ** 2 + (0.047 / x) ** 2 + 90.0)
        + 14.0 / ((x / 0.08) ** 6.5 + (0.08 / x) ** 6.5 + 4.0)
        + 0.045 / ((x / 0.22) ** 2 + (0.22 / x) ** 2 - 1.95)
        + 0.002 / ((x / 9.7) ** 2 + (9.7 / x) ** 2 - 1.95)
        + 0.002 / ((x / 18.0) ** 2 + (18.0 / x) ** 2 - 1.80)
        + 0.012 / ((x / 25.0) ** 2 + (25.0 / x) ** 2)
    )
    return extinct


def Pei92_SMC(wvlg):
    """
    Small Magellanic Cloud extinction curve from Pei+92.
    wvlg is expected in Angstrom.
    Valid from 1000 Angstroms.
    """

    x = _convert_to_microns(wvlg)

    extinct = 1.34 * (
        185.0 / ((x / 0.042) ** 2 + (0.042 / x) ** 2 + 90.0)
        + 27.0 / ((x / 0.08) ** 4 + (0.08 / x) ** 4 + 5.5)
        + 0.005 / ((x / 0.22) ** 2 + (0.22 / x) ** 2 - 1.95)
        + 0.010 / ((x / 9.7) ** 2 + (9.7 / x) ** 2 - 1.95)
        + 0.012 / ((x / 18.0) ** 2 + (18.0 / x) ** 2 - 1.80)
        + 0.030 / ((x / 25.0) ** 2 + (25.0 / x) ** 2)
    )
    return extinct


def Calzetti_SB(wvlg):
    """Summary

    Parameters
    ----------
    wvlg : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    wvlg = _convert_to_microns(wvlg)
    x = 1.0 / wvlg

    extinct = np.where(
        x < 0.64,
        2.659 * (-2.156 + 1.509 * x - 0.198 * x**2.0 + 0.011 * x**3.0) + 4.05,
        2.659 * (-1.857 + 1.040 * x) + 4.05,
    )
    return extinct


def _convert_to_microns(wvlg_AA):
    """
    Takes a wavelength array or Quantity and converts it to microns

    Parameters
    ----------
    wvlg_AA : float, array-like, astropy Quantity
        Wavelength values to convert. If not using an astropy `Quantity`
        object, units are assumed to be Angstrom.

    Returns
    -------
    wvlg : numpy ndarray
        Wavelength converted to microns and into a numpy array.
    """

    if isinstance(wvlg_AA, Quantity):
        wvlg = np.atleast_1d(wvlg_AA.to("micron").value)
    else:
        wvlg = np.atleast_1d(wvlg_AA * 1e-4)
    return wvlg


def get_extinction(ext_str):
    """Summary

    Parameters
    ----------
    ext_str : str
        Description

    Returns
    -------
    extinction_function : function
        Description

    """
    # Select extinction
    if ext_str == "Pei+92 MW":
        extinction_func = Pei92_MW
    elif ext_str == "Pei+92 SMC":
        extinction_func = Pei92_SMC
    elif ext_str == "Cardelli+89":
        extinction_func = Cardelli89
    elif ext_str == "Calzetti+":
        raise NotImplementedError
        # extinction_func = Calzetti
    else:
        raise ValueError(
            "Extinction must be one of ['Pei+92 MW', 'Pei+92 SMC', 'Cardelli+89', 'Calzetti+']"
        )
    return extinction_func


def extinguish_line(flux, wvlg, Av, extinction="Pei+92 MW"):
    """
    Helper function to easily apply extinction on a given line.
    """

    extinction_func = get_extinction(extinction)

    flux_extinguished = flux * 10 ** (-0.4 * Av * extinction_func(wvlg))

    if isinstance(flux_extinguished, np.ndarray):
        flux_extinguished = _check_if_scalar(flux_extinguished)

    return flux_extinguished


def correct_line_for_extinction(flux, wvlg, Av, extinction="Pei+92 MW"):
    """
    Helper function to easily apply extinction on a given line.
    """

    extinction_func = get_extinction(extinction)

    flux_corrected = flux * 10 ** (0.4 * Av * extinction_func(wvlg))

    if isinstance(flux_corrected, np.ndarray):
        flux_corrected = _check_if_scalar(flux_corrected)

    return flux_corrected


def _check_if_scalar(array):
    if (array.ndim == 1) and (array.shape[0] == 1):
        return float(array)
    else:
        return array
