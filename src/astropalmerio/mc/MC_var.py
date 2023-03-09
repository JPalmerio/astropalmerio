# -*- coding: utf-8 -*-

__all__ = [
    "MC_var",
]

import logging
import numpy as np
from .realizations import MC_realization
from .distributions import asym_normal_pdf, flat_pdf_for_limits

log = logging.getLogger(__name__)


class MC_var(object):
    """
    Class to propagate errors using MonteCarlo sampling.


    error : float or array-like of shape(2)

            - scalar: symmetric +/- error values
            - array-like of shape(2): Separate - and + error values.
              First value contains the lower errors, the second value
              contains the upper errors.
            - *None*: No error, used in case of limits. (Default)

            All values must be >= 0.
    """

    def __init__(
        self,
        value,
        error=None,
        lolim=None,
        uplim=None,
        val_max=None,
        val_min=None,
        N_MC=10000,
        seed=None,
    ):
        if all(v is None for v in [error, lolim, uplim, val_max, val_min]):
            raise ValueError(
                "You must specify either an error or an upper/lower limit, "
                "otherwise you wouldn't need Monte Carlo sampling."
            )

        self.value = float(value)

        # Errors
        if isinstance(error, (float, int)):
            self.error = {"minus": float(error), "plus": float(error)}
        elif isinstance(error, (list, np.ndarray)):
            self.error = {"minus": float(error[0]), "plus": float(error[1])}
        elif isinstance(error, dict):
            self.error = error.copy()
        elif error is None:
            self.error = {"minus": None, "plus": None}
        for k, err in self.error.items():
            if err is not None:
                if err < 0:
                    log.warning(
                        "Errors should be positive, I'm assuming this is a mistake and removing the minus sign."
                    )
                    self.error[k] = -err

        # Limits
        if lolim and uplim:
            raise ValueError("Cannot be upper and lower limit at the same time.")
        self.lim = {"lower": bool(lolim), "upper": bool(uplim)}
        if any(self.lim.values()) and any(self.error.values()):
            raise ValueError("Either limits or errors should be specified, not both.")
        if uplim and (val_min is None):
            raise ValueError("Please provide a minimum value if using upper limits")
        if lolim and (val_max is None):
            raise ValueError("Please provide a maximum value if using lower limits")

        # Bounds
        if val_min is not None:
            val_min = float(val_min)
            if val_min > self.value:
                raise ValueError("Data value is below minimum allowed value.")
        if val_max is not None:
            val_max = float(val_max)
            if val_max < self.value:
                raise ValueError("Data value is above maximum allowed value.")
        self.bounds = {"max": val_max, "min": val_min}

        # Monte Carlo sampling parameters
        self.N_MC = int(N_MC)
        self.seed = seed

        self.realizations = None

    def __str__(self):
        if all(err is not None for err in self.error.values()):
            string = "{:.3e} +/- [{:.2e}, {:.2e}]".format(
                self.value, self.error["plus"], self.error["minus"]
            )
        elif self.lim["upper"]:
            string = "<{:.3e}".format(self.value)
        elif self.lim["lower"]:
            string = ">{:.3e}".format(self.value)

        if self.bounds["min"] is not None:
            string += ", minimum allowed value is {:.3e}".format(self.bounds["min"])
        if self.bounds["max"] is not None:
            string += ", maximum allowed value is {:.3e}".format(self.bounds["max"])
        return string

    def copy(self):
        """
        Returns a copy of this MC_var object.

        Returns
        -------
        copy : `MC_var`
            The copied MC_var object
        """
        copy = MC_var(
            value=self.value,
            error=self.error,
            lolim=self.lim["lower"],
            uplim=self.lim["upper"],
            val_max=self.bounds["max"],
            val_min=self.bounds["min"],
            N_MC=self.N_MC,
            seed=self.seed,
        )
        if self.realizations is not None:
            copy.realizations = self.realizations.copy()

        return copy

    def sample(self, N_MC=None, seed=None, force=False):
        """
        Sample the Monte Carlo realizations associated with this MC_var.

        Parameters
        ----------
        N_MC : int, optional
            Number of Monte Carlo realizations.
        seed : int, optional
            Seed passed to `BitGenerator` before sampling; used for
            reproducibility.
        force : bool, optional, Default `False`
            Force resampling if a realization of this MC_var already
            exists.

        Returns
        -------
        realizations : array-like
            The array of the realizations of size N_MC.
        """
        if N_MC is None:
            N_MC = self.N_MC

        if seed is None:
            seed = self.seed

        if force or (self.realizations is None):
            self.realizations = MC_realization(
                data=self.value,
                errp=self.error["plus"],
                errm=self.error["minus"],
                lolim=self.lim["lower"],
                uplim=self.lim["upper"],
                val_max=self.bounds["max"],
                val_min=self.bounds["min"],
                N_MC=N_MC,
                seed=seed,
            )
        else:
            log.warning(
                "Realizations for this MC_var already exist, if you wish to force resampling, use force=True."
            )
        return self.realizations.copy()

    def show_pdf(self, x=None, precision=1000, ax=None, **kwargs):
        """
        Plot the probability density function of this MC_var.

        Parameters
        ----------
        x : array-like, optional
            Array of values at which to evaluate the pdf.
        precision : int, optional
            Number of points at which to evaluate the pdf.
            If `x` is specified, this argument is ignored.
        ax : axes, optional
            Matplotlib axes, if `None` will use `plt.gca()`.
        **kwargs
            Additional arguments passed to `plt.plot()`
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        x, pdf = self.get_pdf(x=x, precision=precision)
        ax.plot(x, pdf, **kwargs)

        plt.draw()

    def show_realizations(self, N_MC=None, ax=None, **kwargs):
        """
        Plot a histogram of the realizations of this MC_var.

        Parameters
        ----------
        N_MC : int, optional
            Number of Monte Carlo realizations.
        ax : axes, optional
            Matplotlib axes, if `None` will use `plt.gca()`.
        **kwargs
            Additional arguments passed to `plt.hist()`
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if self.realizations is None:
            self.sample(N_MC=N_MC)

        ax.hist(self.realizations, **kwargs)

        plt.draw()

    def get_domain(self, precision=1000):
        """
        Get the domain over which this MC_var is defined.
        This is defined by the `bounds` attribute if it exists,
        otherwise the min/max values of the domain are calculated
        using the `value` attribute -/+ 10 times the error
        'minus'/'plus' respectively.

        Parameters
        ----------
        precision : int, optional
            Number of points in the domain.

        Returns
        -------
        domain : array-like
            The derived domain.

        """
        if any(lim for lim in self.lim.values()):
            if self.lim["upper"]:
                xmin = self.bounds["min"]
                xmax = self.lim["upper"]
            elif self.lim["lower"]:
                xmin = self.lim["lower"]
                xmax = self.bounds["max"]
        elif all(err is not None for err in self.error.values()):
            if self.bounds["min"] is None:
                xmin = self.value - 10 * self.error["minus"]
            else:
                xmin = self.bounds["min"]
            if self.bounds["max"] is None:
                xmax = self.value + 10 * self.error["plus"]
            else:
                xmax = self.bounds["max"]
        else:
            raise ValueError(
                "You should not see this error message unless you've "
                "tinkered with the MC_var object in a forbidden way."
            )

        domain = np.linspace(xmin, xmax, precision)

        return domain

    def get_pdf(self, x=None, precision=1000):
        """
        Get the probability density function.

        Parameters
        ----------
        x : array-like, optional
            Array of values at which to evaluate the pdf.
        precision : int, optional
            Number of points at which to evaluate the pdf.
            If `x` is specified, this argument is ignored.

        Returns
        -------
        x : array-like
            Array of values used to evaluate the pdf.
        pdf : array-like
            Probability density function evaluated at `x`.
        """
        if x is None:
            x = self.get_domain(precision=precision)

        if any(lim for lim in self.lim.values()):
            pdf = flat_pdf_for_limits(
                x,
                value=self.value,
                uplim=self.lim["upper"],
                lolim=self.lim["lower"],
                val_min=self.bounds["min"],
                val_max=self.bounds["max"],
            )
        else:
            pdf = asym_normal_pdf(
                x,
                mu=self.value,
                sigma1=self.error["minus"],
                sigma2=self.error["plus"],
                val_min=self.bounds["min"],
                val_max=self.bounds["max"],
            )
        return x, pdf


# NOTE: defining basic arithmetic operations has proven too difficult
# I leave my best attempt at the code here for now but eventually it should be deleted
# def __add__(self, other):

#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self + real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value + float(other)
#         # Apply to bounds as well
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val + float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations + float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __sub__(self, other):

#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self - real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value - float(other)
#         # Apply to bounds as well
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val - float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations - float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __mul__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self * real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value * float(other)
#         # Apply to bounds
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val * float(other)
#         # Apply to errors
#         for e, val in result.error.items():
#             if val is not None:
#                 result.error[e] = val * float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations * float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __div__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self / real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value / float(other)
#         # Apply to bounds
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val / float(other)
#         # Apply to errors
#         for e, val in result.error.items():
#             if val is not None:
#                 result.error[e] = val / float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations / float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __pow__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_other ** real_self
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         real_result = other ** real_self
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     else:
#         raise TypeError("Unsupported operand type")

# def log10(self):

#     if self.seed is not None:
#         np.random.seed(self.seed)
#     draw = self.asym_gaussian_draw(
#         self.value, self.errm, self.errp, nb_draws=self.N_MC
#     )
#     result = np.log10(draw)
#     q = stats.mstats.mquantiles(result, prob=[0.16, 0.5, 0.84])
#     value = q[1]
#     errp = q[2] - q[1]
#     errm = q[1] - q[0]
#     return MC_var(value, errp, errm, N=self.N_MC)

# def exp(self):

#     if self.seed is not None:
#         np.random.seed(self.seed)
#     draw = self.asym_gaussian_draw(
#         self.value, self.errm, self.errp, nb_draws=self.N_MC
#     )
#     result = np.exp(draw)
#     q = stats.mstats.mquantiles(result, prob=[0.16, 0.5, 0.84])
#     value = q[1]
#     errp = q[2] - q[1]
#     errm = q[1] - q[0]
#     return MC_var(value, errp, errm, N=self.N_MC)

# def _check_sample_sizes(self, other):
#     if other.N_MC != self.N_MC:
#         log.warning(
#             "Trying to perform an operation on two MC_var objects with different number of samples, "
#             "falling back to the smallest sample size."
#         )
#         newN = min(other.N_MC, self.N_MC)
#     else:
#         newN = self.N_MC
#     return newN

# def _check_seeds(self, other):
#     if (other.seed == self.seed) and (self.seed is not None):
#         log.warning(
#             "Performing an operation on two MC_var objects with the same seed "
#             "will result in correlated realizations."
#         )
