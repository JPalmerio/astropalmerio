import logging
import numpy as np
from .drawings import sample_asym_norm, sample_uniform_between

log = logging.getLogger(__name__)


def MC_realization(
    data,
    errp=None,
    errm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):
    """
    Function to create a realization of a sample with errors and upper limits.
    This assumes the value's PDF's can be represented by asymmetric gaussians whose sigmas are the plus and minus error.
    For limits, assumes a flat prior (uniform draw) between the limit and the array specified by val_max/val_min.
    val_max and val_min can also be used to specify (in the case of non-limits) the allowed domain.
    This can be used to avoid unphysical values, for instance if you are creating realizations for a distance,
    which cannot be negative, you can provide val_min=np.zeros(len(data))

    """
    if isinstance(data, (int, float)):
        realizations = _MC_realization_scalar(
            data=data,
            errp=errp,
            errm=errm,
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
            errp=errp,
            errm=errm,
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
    errp=None,
    errm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):

    if all(v is None for v in [errp, errm, lolim, uplim, val_max, val_min]):
        raise ValueError(
            "All inputs are None. You must specify either "
            "an error with errp/errm or a upper/lower limit "
            "with uplim/lolim."
        )

    N_data = len(data)

    if (errp is None) and (errm is None):
        errp = np.array([None for i in range(N_data)])
        errm = np.array([None for i in range(N_data)])
    # If positive but no negative error is specified assume errors are symetric
    if (errm is None) and (errp is not None):
        errm = errp

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
    for var in [errp, errm, uplim, lolim, val_max, val_min]:
        if not isinstance(var, (np.ndarray, list)):
            raise ValueError("All inputs must be list or numpy array")
        if len(var) != N_data:
            raise ValueError("All inputs must have same length")

    # Create realizations
    realizations = np.zeros((N_MC, N_data))
    for i in range(N_data):
        if lolim[i] and (val_max[i] is None):
            val_max[i] = data.max() + 5 * errp.max()
            log.warning(
                f"You have a lower limit at i={i} but did not specify what "
                "the maximum value should be for uniform drawings. "
                "I will use the maximum value of your data plus "
                f"5 times the maximum plus error : {val_max[i]:.3e}"
            )
        if uplim[i] and (val_min[i] is None):
            val_min[i] = data.min() - 5 * errm.max()
            log.warning(
                f"You have an upper limit at i={i} but did not specify what "
                "the minimum value should be for uniform drawings. "
                "I will use the minimum value of your data minus "
                f"5 times the maximum minus error : {val_min[i]:.3e}"
            )

        log.debug(
            "Passing the following to scalar MC sampling:\n"
            f"{i=}\n"
            f"{data[i]=}\n"
            f"{errp[i]=}\n"
            f"{errm[i]=}\n"
            f"{lolim[i]=}\n"
            f"{uplim[i]=}\n"
            f"{val_min[i]=}\n"
            f"{val_max[i]=}\n"
            f"{N_MC=}"
        )
        realizations[:, i] = _MC_realization_scalar(
            data=data[i],
            errp=errp[i],
            errm=errm[i],
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
    errp=None,
    errm=None,
    lolim=None,
    uplim=None,
    val_max=None,
    val_min=None,
    N_MC=1000,
    seed=None,
):

    if all(v is None for v in [errp, errm, lolim, uplim, val_max, val_min]):
        raise ValueError(
            "All inputs are None. You must specify either "
            "an error with errp/errm or a upper/lower limit "
            "with uplim/lolim."
        )

    # If no negative error is specified, assume errors are symetric
    if errm is None:
        errm = errp

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
    for var in [errp, errm, val_max, val_min]:
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
        f"{errp=}\n"
        f"{errm=}\n"
        f"{lolim=}\n"
        f"{uplim=}\n"
        f"{val_min=}\n"
        f"{val_max=}\n"
        f"{N_MC=}"
    )

    # Create realizations
    realizations = np.zeros(N_MC)

    # Fix seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # If limits, draw uniform
    if uplim and lolim:
        realizations = sample_uniform_between(
            val_min=val_min, val_max=val_max, nb_draws=N_MC
        )
    elif uplim and not lolim:
        realizations = sample_uniform_between(
            val_min=val_min, val_max=data, nb_draws=N_MC
        )
    elif not uplim and lolim:
        realizations = sample_uniform_between(
            val_min=data, val_max=val_max, nb_draws=N_MC
        )
    # If no limits draw asymmetric gaussian
    else:
        # If symmetric error and no bounding values, simply draw from numpy gaussian
        if (errm == errp) and (val_min is None) and (val_max is None):
            realizations = np.random.normal(data, errp, N_MC)
        else:
            realizations = sample_asym_norm(
                data,
                sigma1=errm,
                sigma2=errp,
                nb_draws=N_MC,
                val_min=val_min,
                val_max=val_max,
            )

    return realizations
