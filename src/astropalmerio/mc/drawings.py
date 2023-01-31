import logging
import numpy as np
from .distributions import ancdf
from .utils import get_corresponding_y_value

log = logging.getLogger(__name__)


def sample_uniform_between(val_min, val_max, nb_draws=1000):
    """
    Sample uniformly between maximum and minimum.
    """

    drawings = (val_max - val_min) * np.random.rand(nb_draws) + val_min

    return drawings


def sample_from_CDF(x, Fx, nb_draws, val_min=None, val_max=None):
    """
    Sample from Fx.
    """

    rand_min = np.min(Fx)
    rand_max = np.max(Fx)

    # Check for min bound
    if val_min is not None:
        if val_min < np.min(x):
            pass
        elif val_min > np.max(x):
            raise ValueError(
                "The minimum allowed value is greater than the maximum value of the provided domain."
            )
        else:
            rand_min = get_corresponding_y_value(val_min, x, Fx)

    # Check for max bound
    if val_max is not None:
        if val_max > np.max(x):
            pass
        elif val_max < np.min(x):
            raise ValueError(
                "The maximum allowed value is greater than the minimum value of the provided domain."
            )
        else:
            rand_max = get_corresponding_y_value(val_max, x, Fx)

    rand = (rand_max - rand_min) * np.random.rand(nb_draws) + rand_min
    drawings = get_corresponding_y_value(rand, Fx, x)

    return drawings


def sample_asym_norm(
    mu, sigma1, sigma2, nb_draws=1000, precision=500, val_min=None, val_max=None
):
    """
    Function that draws randomly in a asymmetric normal (Gaussian) distribution.
    in the form :   { exp( -(x-mu)**2/(2*sigma1**2) )     if x < mu
                    { exp( -(x-mu)**2/(2*sigma2**2) )     if x >= mu
    Returns an array of the drawings
    """

    if (sigma1 == 0) and (sigma2 == 0):
        return mu * np.ones(nb_draws)

    if (sigma1 < 0) or (sigma2 < 0):
        raise ValueError("sigma1 or sigma2 can not be negative, check your input.")

    if sigma1 == 0:
        if mu == 0:
            sigma1 = 1e-9
            log.warning("Sigma1 and mu are equal to zero, replacing sigma1 by 1e-9")
        else:
            sigma1 = 1e-9 * mu
            log.warning("Sigma1 is equal to zero, replacing sigma1 by mu * 1e-9")
    elif sigma2 == 0:
        if mu == 0:
            sigma2 = 1e-9
            log.warning("Sigma2 and mu are equal to zero, replacing sigma2 by 1e-9")
        else:
            sigma2 = 1e-9 * mu
            log.warning("Sigma2 is equal to zero, replacing sigma2 by mu * 1e-9")

    # limits
    x_min = mu - 10.0 * sigma1
    x_max = mu + 10.0 * sigma2

    # create Cumulative distribution
    x = np.linspace(x_min, x_max, precision)
    Fx = ancdf(x, mu, sigma1, sigma2)

    # draw from generated distribution
    draw = sample_from_CDF(x, Fx, nb_draws=nb_draws, val_min=val_min, val_max=val_max)

    return draw
