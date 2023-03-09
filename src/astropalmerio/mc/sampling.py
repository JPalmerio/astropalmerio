# -*- coding: utf-8 -*-

__all__ = [
    "bootstrap",
    "sample_uniform_between",
    "sample_from_CDF",
    "sample_asym_norm",
]
import logging
import numpy as np
from numpy.random import default_rng, PCG64
from .distributions import ancdf
from .utils import get_corresponding_y_value

log = logging.getLogger(__name__)

rng = default_rng()


def bootstrap(sample, weights=None, random_state=None):
    """
    Bootstrap from a 2D array of size (N_real, N_samp) and return the bootstrapped array
    with the RNG state (for reproductibility).
    N_real stands for number of realizations
    N_samp stands for size of the sample
    """
    N_real, sample_len = sample.shape
    ind = rng.randint(sample_len, size=sample.shape)
    sample_bootstrapped = np.zeros(sample.shape)
    for i in range(N_real):
        sample_bootstrapped[i] = sample[i][ind[i]]

    if weights is None:
        weights_bootstrapped = None
    else:
        weights_bootstrapped = np.zeros(sample.shape)
        for i in range(N_real):
            weights_bootstrapped[i] = weights[i][ind[i]]

    return sample_bootstrapped, weights_bootstrapped


def sample_uniform_between(val_min, val_max, nb_draws=1000, seed=None):
    """
    Sample uniformly between maximum and minimum.
    """

    # Fix seed for reproducibility
    if seed is not None:
        rng.bit_generator.state = PCG64(seed).state

    drawings = (val_max - val_min) * rng.random(nb_draws) + val_min

    return drawings


def sample_from_CDF(x, Fx, nb_draws, val_min=None, val_max=None, seed=None):
    """
    Sample from Fx.
    """

    rand_min = np.min(Fx)
    rand_max = np.max(Fx)

    # Fix seed for reproducibility
    if seed is not None:
        rng.bit_generator.state = PCG64(seed).state

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

    rand = (rand_max - rand_min) * rng.random(nb_draws) + rand_min
    drawings = get_corresponding_y_value(rand, Fx, x)

    return drawings


def sample_asym_norm(
    mu,
    sigma1,
    sigma2,
    nb_draws=1000,
    precision=500,
    val_min=None,
    val_max=None,
    seed=None,
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
    draw = sample_from_CDF(
        x, Fx, nb_draws=nb_draws, val_min=val_min, val_max=val_max, seed=seed
    )

    return draw
