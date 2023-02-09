import logging
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.units.quantity import Quantity
from astropy.nddata import StdDevUncertainty
from astropy.modeling.models import Gaussian1D
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting.continuum import fit_continuum
from specutils.fitting import fit_lines

from astropalmerio import DIRS
from .utils import measure_noise, integrate_flux
from .visualization import plot_spectrum, show_regions
from .conversions import sigma2FWHM, FWHM_w2v
from .utils import gaussian_infinite_integral


log = logging.getLogger(__name__)


line_list = pd.read_csv(
    DIRS["data"] / "emission_lines.csv",
    sep=",",
    comment="#",
)
line_list.set_index("name", inplace=True)


class EmissionLine(object):
    def __init__(
        self,
        name,
        rest_awav=None,
        z_guess=None,
        detected=True,
    ):

        self.name = name
        self.properties = {}
        self.spectrum = {}
        self.fit = {}

        if rest_awav is None:
            try:
                self.properties["rest_awav"] = line_list.loc[name]["awav"] * u.AA
            except KeyError:
                raise ValueError(
                    f"Could not find {name} line in {DIRS['data']/'emission_lines.csv'} "
                    "No rest (air) wavelength could be set. "
                    "You may set it manually during the initialization with `rest_awav`."
                )
        elif isinstance(rest_awav, Quantity):
            self.properties["rest_awav"] = rest_awav
        else:
            raise TypeError("rest_awav must be an astropy Quantity")

        if z_guess is not None:
            self.properties["obs_awav_guess"] = self.properties["rest_awav"] * (
                1 + float(z_guess)
            )

        self.properties["detected"] = bool(detected)

    def extract_spectrum_from(self, spectrum, bounds=None):
        """Extract a region of a Spectrum1D object containing the line.

        Parameters
        ----------
        spectrum : Spectrum1D
            Description
        bounds : array-like, optional
            Tuple or list of 2 values: lower and upper bounds.

        Returns
        -------
        TYPE
            Description
        """

        if bounds is None:
            # If no bounds and a redshift guess is provided
            # try to define reasonable bounds of ~ 100 Angstrom around
            # the expected line center
            if "z_guess" in self.properties.keys():
                wmin = (
                    self.properties["obs_awav_guess"]
                    - 50 * u.AA * (1 + self.properties["z_guess"]),
                )
                wmax = (
                    self.properties["obs_awav_guess"]
                    + 50 * u.AA * (1 + self.properties["z_guess"]),
                )
            else:
                raise ValueError(
                    "Please provide bounds or a guess redshift to extract a spectrum."
                )
        else:
            wmin, wmax = bounds

        imin = spectrum.spectral_axis.searchsorted(wmin)
        imax = spectrum.spectral_axis.searchsorted(wmax)

        self.spectrum["wvlg"] = spectrum.spectral_axis[imin:imax]
        self.spectrum["flux"] = spectrum.flux[imin:imax]

        if spectrum.uncertainty is not None:
            self.spectrum["error"] = spectrum.uncertainty.quantity[imin:imax]
        else:
            log.warning(
                "No error spectrum, determining average noise from spectrum "
                "and using that value throughout the spectrum... "
                "You should really be providing uncertainties with your measurements!"
            )
            mean, noise, snr = measure_noise(
                self.spectrum["wvlg"],
                self.spectrum["flux"],
                wvlg_min=self.spectrum["wvlg"].min(),
                wvlg_max=self.spectrum["wvlg"].max(),
            )
            self.spectrum["error"] = noise * np.ones(len(self.spectrum["flux"]))

        return self.spectrum

    def _get_bounds(self, bounds, default=None):
        """Summary

        Parameters
        ----------
        bounds : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        if bounds is None:
            if default is None:
                wmin = self.spectrum["wvlg"].min()
                wmax = self.spectrum["wvlg"].max()
            else:
                wmin, wmax = default
        else:
            wmin, wmax = bounds
        return wmin, wmax

    def fit_continuum(self, regions, **args):
        """Summary

        Parameters
        ----------
        regions : TYPE
            List of 2-tuple containing the lower and upper bounds of the
        region.
        **args
            Arguments to pass to `fit_continuum` function of specutils.

        Returns
        -------
        TYPE
            Description
        """
        continuum_fit = fit_continuum(
            spectrum=Spectrum1D(
                spectral_axis=self.spectrum["wvlg"],
                flux=self.spectrum["flux"],
                uncertainty=StdDevUncertainty(self.spectrum["error"]),
            ),
            window=regions,
            **args,
        )
        self.fit["continuum"] = {
            "regions": regions,
            "flux": continuum_fit(self.spectrum["wvlg"]),
        }
        return continuum_fit

    def measure_flux(self, bounds=None):

        wmin, wmax = self._get_bounds(
            bounds,
            default=(
                self.fit["bounds"]["min"],
                self.fit["bounds"]["max"],
            ),
        )

        flux, error = integrate_flux(
            wvlg=self.spectrum["wvlg"],
            flux=self.spectrum["flux"],
            error=self.spectrum["error"],
            wvlg_min=wmin,
            wvlg_max=wmax,
            continuum=self.fit["continuum"]["flux"],
        )

        self.properties["flux_int_bounds"] = (wmin, wmax)
        self.properties["flux_int"] = flux
        self.properties["flux_int_err"] = error

        return flux, error

    def fit_single_gaussian(self, bounds=None, **args):
        """Summary

        Parameters
        ----------
        bounds : None, optional
            Description
        """

        wmin, wmax = self._get_bounds(bounds)

        initial_guess = {
            "mean": args.get("mean", self.properties["obs_wvlg"]),
            "stddev": args.get("stddev", 1 * u.AA),
            "amplitude": args.get(
                "amplitude",
                np.max(self.spectrum["flux"] - self.fit["continuum"]["flux"]),
            ),
        }

        self.fit["bounds"] = (wmin, wmax)
        self.fit["initial_guess"] = initial_guess

        g_init = Gaussian1D(**initial_guess)

        g_fit = fit_lines(
            spectrum=Spectrum1D(
                spectral_axis=self.spectrum["wvlg"],
                flux=self.spectrum["flux"] - self.fit["continuum"]["flux"],
                uncertainty=StdDevUncertainty(self.spectrum["error"]),
            ),
            model=g_init,
            get_fit_info=True,
            window=SpectralRegion(self.fit["bounds"]),
        )

        # This is a bit contrived but necessary to get the units
        # if only g_fit could return the units of the parameters...
        self.fit["results"] = {
            name: value * initial_guess[name].unit
            for name, value in zip(g_fit.param_names, g_fit.parameters)
        }

        self.fit["model"] = Gaussian1D(**self.fit["results"])

        self.fit["flux"] = (
            self.fit["model"](x=self.spectrum["wvlg"]) + self.fit["continuum"]["flux"]
        )

        self.fit["residuals"] = (
            self.spectrum["flux"] - self.fit["model"]
        ) / self.spectrum["error"]

    def derive_properties_from_fit(self):
        self.properties['obs_awav_fit'] = self.fit['results']['mean']
        self.properties['FWHM_lam'] = sigma2FWHM(self.fit['results']['stddev'])
        self.properties['FWHM_vel'] = FWHM_w2v(
            wvlg=self.properties['FWHM_lam'],
            w0=self.properties['obs_awav_fit']
        )
        self.properties['flux_fit'] = gaussian_infinite_integral(
                amplitude=self.fit['results']['amplitude'],
                stddev=self.fit['results']['stddev'],
        )
        self.properties['z_fit'] = self.properties['obs_awav_fit'] / self.properties['rest_awav'] - 1

    # Visualization stuff
    def show_continuum(self, ax=None, show_fitting_regions=True, **kwargs):
        """Summary

        Parameters
        ----------
        ax : None, optional
            Description
        show_fitting_regions : bool, optional
            Description
        **kwargs
            Description
        """
        self.show_spectrum(ax=ax)
        ax.plot(
            self.spectrum["wvlg"],
            self.fit["continuum"]["flux"],
            label=kwargs.get("label", "Continuum fit"),
            color=kwargs.get("color", "C1"),
            **kwargs,
        )

        if show_fitting_regions:
            show_regions(
                ax=ax,
                regions=self.fit["continuum"]["regions"],
                color=kwargs.get("color", "C1"),
            )

    def show_spectrum(self, ax=None, **kwargs):

        plot_spectrum(
            wvlg=self.spectrum["wvlg"], flux=self.spectrum["flux"], ax=ax, **kwargs
        )
