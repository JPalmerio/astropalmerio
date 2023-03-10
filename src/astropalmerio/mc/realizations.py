# -*- coding: utf-8 -*-

__all__ = "MC_realization"

import logging
import numpy as np
from numpy.random import default_rng, PCG64
from .sampling import sample_asym_norm, sample_uniform_between

rng = default_rng()

log = logging.getLogger(__name__)


def MC_realization(
    data,
    uncp=None,
    uncm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):
    """
    Function to create a realization of a sample with uncertainties and upper limits.
    This assumes the value's PDF's can be represented by asymmetric gaussians whose
    sigmas are the plus and minus uncertainty.
    For limits, assumes a flat prior (uniform draw) between the limit and the array
    specified by val_max/val_min.
    val_max and val_min can also be used to specify (in the case of non-limits) the
    allowed domain.
    This can be used to avoid unphysical values, for instance if you are creating
    realizations for a distance,
    which cannot be negative, you can provide val_min=np.zeros(len(data))

    """
    if isinstance(data, (int, float)):
        realizations = _MC_realization_scalar(
            data=data,
            uncp=uncp,
            uncm=uncm,
            lolim=lolim,
            uplim=uplim,
            val_max=val_max,
            val_min=val_min,
            N_MC=N_MC,
            seed=seed,
        )
    elif isinstance(data, (np.ndarray, list)):
        realizations = _MC_realization_array(
            data=data,
            uncp=uncp,
            uncm=uncm,
            lolim=lolim,
            uplim=uplim,
            val_max=val_max,
            val_min=val_min,
            N_MC=N_MC,
            seed=seed,
        )
    else:
        raise ValueError("data must be a float, int, list or numpy array")

    return realizations


def _MC_realization_array(
    data,
    uncp=None,
    uncm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):
    if all(v is None for v in [uncp, uncm, lolim, uplim, val_max, val_min]):
        raise ValueError(
            "All inputs are None. You must specify either "
            "an uncertainty with uncp/uncm or a upper/lower limit "
            "with uplim/lolim."
        )

    N_data = len(data)

    if (uncp is None) and (uncm is None):
        uncp = np.array([None for i in range(N_data)])
        uncm = np.array([None for i in range(N_data)])
    # If positive but no negative uncertainty is specified assume uncertainties are symetric
    if (uncm is None) and (uncp is not None):
        uncm = uncp

    # Check for limits
    # Lower-limits
    if lolim is None:
        lolim = np.array([False for i in range(N_data)])
    # Upper limits
    if uplim is None:
        uplim = np.array([False for i in range(N_data)])

    # If a scalar is specified for val_min or val_max,
    # turn it into an array of length N_data
    if val_min is None:
        val_min = np.array([None for i in range(N_data)])
    elif isinstance(val_min, (float, int)):
        val_min = val_min * np.ones(N_data)

    if val_max is None:
        val_max = np.array([None for i in range(N_data)])
    elif isinstance(val_max, (float, int)):
        val_max = val_max * np.ones(N_data)

    # Check if inputs are ok
    for var in [uncp, uncm, uplim, lolim, val_max, val_min]:
        if not isinstance(var, (np.ndarray, list)):
            raise ValueError("All inputs must be list or numpy array")
        if len(var) != N_data:
            raise ValueError("All inputs must have same length")

    # Create realizations
    realizations = np.zeros((N_MC, N_data))
    for i in range(N_data):
        if lolim[i] and (val_max[i] is None):
            val_max[i] = data.max() + 5 * uncp.max()
            log.warning(
                f"You have a lower limit at i={i} but did not specify what "
                "the maximum value should be for uniform drawings. "
                "I will use the maximum value of your data plus "
                f"5 times the maximum plus uncertainty : {val_max[i]:.3e}"
            )
        if uplim[i] and (val_min[i] is None):
            val_min[i] = data.min() - 5 * uncm.max()
            log.warning(
                f"You have an upper limit at i={i} but did not specify what "
                "the minimum value should be for uniform drawings. "
                "I will use the minimum value of your data minus "
                f"5 times the maximum minus uncertainty : {val_min[i]:.3e}"
            )

        log.debug(
            "Passing the following to scalar MC sampling:\n"
            f"{i=}\n"
            f"{data[i]=}\n"
            f"{uncp[i]=}\n"
            f"{uncm[i]=}\n"
            f"{lolim[i]=}\n"
            f"{uplim[i]=}\n"
            f"{val_min[i]=}\n"
            f"{val_max[i]=}\n"
            f"{N_MC=}"
        )
        realizations[:, i] = _MC_realization_scalar(
            data=data[i],
            uncp=uncp[i],
            uncm=uncm[i],
            lolim=lolim[i],
            uplim=uplim[i],
            val_max=val_max[i],
            val_min=val_min[i],
            N_MC=N_MC,
            seed=seed,
        )

    return realizations


def _MC_realization_scalar(
    data,
    uncp=None,
    uncm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):
    if all(v is None for v in [uncp, uncm, lolim, uplim, val_max, val_min]):
        raise ValueError(
            "All inputs are None. You must specify either "
            "an uncertainty with uncp/uncm or a upper/lower limit "
            "with uplim/lolim."
        )

    # If no negative uncertainty is specified, assume uncertainties are symetric
    if uncm is None:
        uncm = uncp

    if lolim is None:
        lolim = False

    if uplim is None:
        uplim = False

    # Check for limits
    # Lower-limits
    if lolim and (val_max is None):
        raise ValueError(
            "You have a lower limits but did not specify what "
            "the maximum value should be for uniform drawings. "
            "Please specify the val_max argument."
        )

    # Upper limits
    if uplim and (val_min is None):
        raise ValueError(
            "You have an upper limit but did not specify what "
            "the minimum value should be for uniform drawings. "
            "Please specify the val_min argument."
        )

    # Check if inputs are ok
    for var in [uncp, uncm, val_max, val_min]:
        if (var is not None) and not isinstance(var, (float, int)):
            raise ValueError("Errors and max/min values must be float or int")
    for var in [uplim, lolim]:
        if not isinstance(var, (bool, np.bool_)):
            raise ValueError(
                f"Uplim and lolim must be booleans but {uplim=} and {lolim=}"
            )

    log.debug(
        "About to attempt sampling for following inputs:\n"
        f"{data=}\n"
        f"{uncp=}\n"
        f"{uncm=}\n"
        f"{lolim=}\n"
        f"{uplim=}\n"
        f"{val_min=}\n"
        f"{val_max=}\n"
        f"{N_MC=}"
    )

    # Create realizations
    realizations = np.zeros(N_MC)

    # If limits, draw uniform
    if uplim and lolim:
        realizations = sample_uniform_between(
            val_min=val_min,
            val_max=val_max,
            nb_draws=N_MC,
            seed=seed,
        )
    elif uplim and not lolim:
        realizations = sample_uniform_between(
            val_min=val_min,
            val_max=data,
            nb_draws=N_MC,
            seed=seed,
        )
    elif not uplim and lolim:
        realizations = sample_uniform_between(
            val_min=data,
            val_max=val_max,
            nb_draws=N_MC,
            seed=seed,
        )
    # If no limits draw asymmetric gaussian
    else:
        # If symmetric uncertainty and no bounding values
        # simply draw from numpy gaussian
        if (uncm == uncp) and (val_min is None) and (val_max is None):
            # Fix seed for reproducibility
            if seed is not None:
                rng.bit_generator.state = PCG64(seed).state
            realizations = rng.normal(loc=data, scale=uncp, size=N_MC)
        else:
            realizations = sample_asym_norm(
                data,
                sigma1=uncm,
                sigma2=uncp,
                nb_draws=N_MC,
                val_min=val_min,
                val_max=val_max,
                seed=seed,
            )

    return realizations
