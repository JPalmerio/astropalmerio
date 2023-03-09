# -*- coding: utf-8 -*-

__all__ = [
    "anpdf",
    "ancdf",
    "flat_pdf",
    "asym_normal_pdf",
    "flat_pdf_for_limits",
]

import logging
import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid

log = logging.getLogger(__name__)


def anpdf(x, mu, s1, s2):
    """
    Asymmetric normal probability density function.
    In the form :   { exp( -0.5*(x-mu)**2 / s1**2)     if x < mu
                    { exp( -0.5*(x-mu)**2 / s2**2)     if x >= mu

    Parameters
    ----------
    x : int, float, or array-like
        Value(s) at which to evaluate the PDF
    mu : int, float
        Location parameter (mean) of the asymmetric normal distribution.
    s1 : int, float
        Scale parameter at x < mu
    s2 : int, optional
        Scale parameter at x >= mu

    Returns
    -------
    pdf : int, float or array-like
        The PDF evaluated at x.
    """
    if (s1 == 0) and (s2 == 0):
        raise ValueError(
            "Sigma 1 and Sigma 2 cannot be equal to zero at the same time."
        )
    if (s1 < 0) or (s2 < 0):
        raise ValueError("Sigma 1 and Sigma 2 must be positive.")

    a = 2.0 / (np.sqrt(2.0 * np.pi) * (s1 + s2))
    if isinstance(x, np.ndarray):
        pdf = np.where(
            x < mu,
            np.exp(-0.5 * (x - mu) ** 2 / s1**2),
            np.exp(-0.5 * (x - mu) ** 2 / s2**2),
        )
    else:
        if x < mu:
            pdf = np.exp(-0.5 * (x - mu) ** 2 / s1**2)
        else:
            pdf = np.exp(-0.5 * (x - mu) ** 2 / s2**2)

    # Control for case where s2=0, which will return NaN at x=mu
    if s2 == 0:
        pdf[np.isnan(pdf)] = 1
    return a * pdf


def ancdf(x, mu, s1, s2):
    """
    Asymmetric normal cumulative distribution function for a pdf of the
    form :   { exp( -0.5*(x-mu)**2 / s1**2)     if x < mu
             { exp( -0.5*(x-mu)**2 / s2**2)     if x >= mu

    Parameters
    ----------
    x : int, float, or array-like
        Value(s) at which to evaluate the PDF
    mu : int, float
        Location parameter (mean) of the asymmetric normal distribution.
    s1 : int, float
        Scale parameter at x < mu
    s2 : int, optional
        Scale parameter at x >= mu

    Returns
    -------
    pdf : int, float or array-like
        The CDF evaluated at x.
    """

    if (s1 == 0) and (s2 == 0):
        raise ValueError(
            "Sigma 1 and Sigma 2 cannot be equal to zero at the same time."
        )
    if (s1 < 0) or (s2 < 0):
        raise ValueError("Sigma 1 and Sigma 2 must be positive.")

    a = 2.0 / (s1 + s2)
    if isinstance(x, np.ndarray):
        cdf = np.where(
            x < mu, s1 * norm.cdf(x, mu, s1), 0.5 * (s1 - s2) + s2 * norm.cdf(x, mu, s2)
        )
    else:
        if x < mu:
            cdf = s1 * norm.cdf(x, mu, s1)
        else:
            cdf = 0.5 * (s1 - s2) + s2 * norm.cdf(x, mu, s2)

    # Control for case where s2=0, which will return NaN at x=mu
    if (s1 == 0) or (s2 == 0):
        log.warning(
            "Replacing NaN values by 0 in Asymetric Normal CDF. "
            "This was caused by sigma1 or sigma2 being equal to 0."
        )
        cdf[np.isnan(cdf)] = 0

    return a * cdf


def flat_pdf(x, lower, upper):
    """
    Returns a flat probability density function (PDF) evaluated at x.

    Parameters
    ----------
    x : int, float, or array-like
        Value(s) at which to evaluate the PDF
    lower : int, float
        Lower bound of the flat PDF
    upper : int, float
        Upper bound of the flat PDF

    Returns
    -------
    pdf : int, float or array-like
        The PDF evaluated at x.

    """
    if lower >= upper:
        raise ValueError(
            "The lower bound should be strictly less than the upper bound."
        )

    a = 1.0 / np.abs(upper - lower)
    pdf = np.where((x >= lower) & (x <= upper), a, 0)
    return pdf


def asym_normal_pdf(x, mu, sigma1, sigma2, val_min=None, val_max=None):
    if sigma1 == sigma2:
        pdf = norm.pdf(x, loc=mu, scale=sigma1)
    else:
        pdf = anpdf(x, mu=mu, s1=sigma1, s2=sigma2)
    # if bounds, cut the PDF
    if val_min is not None:
        pdf = np.where(x < val_min, 0, pdf)
    if val_max is not None:
        pdf = np.where(x > val_max, 0, pdf)
    # And renormalize PDF
    if (val_min is not None) or (val_max is not None):
        renorm = trapezoid(pdf, x)
        pdf /= renorm

    return pdf


def flat_pdf_for_limits(x, value, uplim=False, lolim=False, val_min=None, val_max=None):
    """
    Return a flat pdf in the case of upper limits (uplim) or lower limits (lolim).
    This flat pdf is based on the principle of least information.
    In the case where there are both upper and lower limits on the data, a uniform pdf between
    those values is returned.
    """
    if lolim and val_max is None:
        raise ValueError(
            "If using a lower-limit (lolim=True), you must provide a maximum value "
            "to set a bound on the flat pdf with val_max."
        )

    if uplim and val_min is None:
        raise ValueError(
            "If using an upper-limit (uplim=True), you must provide a minimum value "
            "to set a bound on the flat pdf with val_min."
        )

    if lolim and not uplim:
        pdf = flat_pdf(x, value, val_max)
    elif uplim and not lolim:
        pdf = flat_pdf(x, val_min, value)
    elif lolim and uplim:
        pdf = flat_pdf(x, val_min, val_max)
    else:
        raise ValueError(
            "If you don't have limits, don't use this function. "
            "Perhaps you forgot to set uplim or lolim to True."
        )
    return pdf
