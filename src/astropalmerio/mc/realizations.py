import logging
import numpy as np

log = logging.getLogger(__name__)


def MC_realization(
    sample,
    sample_errp,
    sample_errm=None,
    sample_ll=None,
    sample_ul=None,
    val_max=None,
    val_min=None,
    N_real=1000,
):
    """
    Function to create a realization of a sample with errors and upper limits.
    This assumes the value's PDF's can be represented by asymmetric gaussians whose sigmas are the plus and minus error.
    For limits, assumes a flat prior (uniform draw) between the limit and the array specified by val_max/val_min.
    val_max and val_min can also be used to specify (in the case of non-limits) the allowed domain.
    This can be used to avoid unphysical values, for instance if you are creating realizations for a distance,
    which cannot be negative, you can provide val_min=np.zeros(len(sample))

    """

    # If no negative error is specified, assume errors are symetric
    if sample_errm is None:
        sample_errm = sample_errp

    # Check for limits
    # Lower-limits
    if sample_ll is None:
        sample_ll = np.zeros(len(sample))
    else:
        if len(sample_ll) != len(sample):
            raise IOError("sample_ll and sample must have same length")
        if val_max is None:
            val_max = (sample.max() + 5 * sample_errp.max()) * np.ones(len(sample))
            log.warning(
                "You have lower limits but did not specify what "
                "the maximum value should be for uniform drawings. "
                "I will use the maximum value of your sample plus "
                f"5 times the maximum plus error : {val_max[0]:.3e}"
            )

    # Upper limits
    if sample_ul is None:
        sample_ul = np.zeros(len(sample))
    else:
        if len(sample_ul) != len(sample):
            raise IOError("sample_ul and sample must have same length")
        if val_min is None:
            val_min = (sample.min() - 5 * sample_errm.max()) * np.ones(len(sample))
            log.warning(
                "You have upper limits but did not specify what "
                "the minimum value should be for uniform drawings. "
                "I will use the minimum value of your sample minus "
                f"5 times the maximum minus error : {val_min[0]:.3e}"
            )

    # Create realizations
    sample_real = np.zeros((N_real, len(sample)))
    for i in range(len(sample)):
        # If no limits draw asymmetric gaussian
        if (sample_ll[i] == 0) & (sample_ul[i] == 0):
            sample_real[:, i] = asym_gaussian_draw(
                sample[i], sigma1=sample_errm[i], sigma2=sample_errp[i], nb_draws=N_real
            )
        # Otherwise, draw uniform
        elif (sample_ll[i] == 1) & (sample_ul[i] == 0):
            sample_real[:, i] = (
                np.random.rand(N_real) * (val_max[i] - sample[i]) + sample[i]
            )
        elif (sample_ll[i] == 0) & (sample_ul[i] == 1):
            if positive & (val_min[i] < 0.0):
                sample_real[:, i] = np.random.rand(N_real) * sample[i]
            else:
                sample_real[:, i] = (
                    np.random.rand(N_real) * (sample[i] - val_min[i]) + val_min[i]
                )
        else:
            sample_real[:, i] = (
                np.random.rand(N_real) * (val_max[i] - val_min[i]) + val_min[i]
            )

    return sample_real
