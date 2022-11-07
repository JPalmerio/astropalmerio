import numpy as np
import logging
from scipy.stats import norm

log = logging.getLogger(__name__)


def anpdf(x, mu=0, s1=1, s2=1):
    """
        Asymmetric normal pdf.
    """
    a = 2. / (np.sqrt(2. * np.pi) * (s1 + s2))
    if isinstance(x, np.ndarray):
        pdf = a * np.exp(-0.5*(x - mu)**2 / s1**2)
        w = np.where(x > mu)
        pdf[w] = a * np.exp(-0.5*(x[w] - mu)**2 / s2**2)
    else:
        if x < mu:
            pdf = a * np.exp(-0.5*(x - mu)**2 / s1**2)
        else:
            pdf = a * np.exp(-0.5*(x - mu)**2 / s2**2)
    return pdf


def ancdf(x, mu=0, s1=1, s2=1):
    """
        Asymmetric normal cdf.
    """
    a = 2. / (s1 + s2)
    if isinstance(x, np.ndarray):
        cdf = a * s1 * norm.cdf(x, mu, s1)
        w = np.where(x > mu)
        cdf[w] = a * (0.5*(s1 - s2) + s2*norm.cdf(x[w], mu, s2))
    else:
        if x < mu:
            cdf = a * s1 * norm.cdf(x, mu, s1)
        else:
            cdf = a * (0.5*(s1 - s2) + s2*norm.cdf(x, mu, s2))
    return cdf


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
        raise ValueError("The lower bound should be strictly less than "
                         " the upper bound.")

    a = 1./np.abs(upper - lower)
    pdf = np.where((x >= lower) & (x <= upper), a, 0)
    return pdf


def flat_pdf_for_limits(x, data, ul=False, ll=False, ul_min_val=None, ll_max_val=None):
    """
        Return a flat pdf in the case of upper limits (ul) or lower limits (ll).
        This flat pdf is based on the principle of least information.
        In the case where there are both upper and lower limits on the data, a uniform pdf between
        those values is returned.
    """
    if ll and ll_max_val is None:
        raise ValueError("If using a lower-limit (ll=True), you must provide a maximum value "
                         "to set a bound on the flat pdf with ll_max_val.")

    if ul and ul_min_val is None:
        raise ValueError("If using an upper-limit (ul=True), you must provide a minimum value "
                         "to set a bound on the flat pdf with ul_min_val.")

    if ll and not ul:
        pdf = flat_pdf(x, data, ll_max_val)
    elif ul and not ll:
        pdf = flat_pdf(x, ul_min_val, data)
    elif ll and ul:
        pdf = flat_pdf(x, ul_min_val, ll_max_val)
    else:
        raise ValueError("If you don't have limits, don't use this function. "
                         "Perhaps you forgot to set ul or ll to True.")
    return pdf
