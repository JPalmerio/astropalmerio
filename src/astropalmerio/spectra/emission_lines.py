import logging
from pathlib import Path
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
from astropalmerio.mc.utils import format_to_string
from astropalmerio.io.fits import read_fits_1D_spectrum, read_fits_2D_spectrum
from .utils import measure_noise, integrate_flux, gaussian_infinite_integral
from .visualization import plot_spectrum, plot_continuum, plot_fit
from .conversions import sigma_to_fwhm, fwhm_w2v, ergscm2AA
from .reduction import extract_1D_from_2D

log = logging.getLogger(__name__)

fname_em_line = DIRS["DATA"] / "emission_lines.tsv"
line_list = pd.read_csv(
    fname_em_line,
    sep="\t",
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
                    f"Could not find {name} line in {fname_em_line}. "
                    "No rest (air) wavelength could be set. "
                    "You may set it manually to avoid this error during "
                    "the instanciation with `rest_awav`."
                )
        elif isinstance(rest_awav, Quantity):
            self.properties["rest_awav"] = rest_awav
        else:
            raise TypeError("rest_awav must be an astropy Quantity")

        if z_guess is not None:
            self.properties["z_guess"] = float(z_guess) * u.dimensionless_unscaled
            self.properties["obs_awav_guess"] = self.properties["rest_awav"] * (
                1 + float(z_guess)
            )

        self.properties["detected"] = bool(detected)

    def to_latex_string(self):
        return line_list.loc[self.name]["latex_str"]

    def load_from_1D_file(self, fname, **args):
        """Summary

        Parameters
        ----------
        fname : str or Path
            Name of the file containing data to load.
        **args
            Arguments to pass to `pd.read_csv()`

        """
        if not isinstance(fname, Path):
            fname = Path(fname)

        if fname.suffix == ".fits":
            wvlg, flux, error = read_fits_1D_spectrum(fname)

        elif fname.suffix in [".txt", ".dat"]:
            df = pd.read_csv(fname, **args)
            wvlg = df["wave"].to_numpy()
            flux = df["flux"].to_numpy()
            error = df["error"].to_numpy()
        else:
            raise ValueError(
                "Cannot read this type of file. It must be "
                "a '.fits', '.dat' or '.txt' file."
            )

        spectrum = Spectrum1D(
            spectral_axis=wvlg * u.AA,
            flux=flux * ergscm2AA,
            uncertainty=StdDevUncertainty(error),
        )

        self.spectrum["1D"] = {
            "fname": fname,
            "data": spectrum,
        }

    def load_from_2D_file(self, fname, spatial_bounds):
        """Summary

        Parameters
        ----------
        fname : str or Path
            Name of the file containing data to load.
        spatial_bounds : tuple
            2-tuple of the lower and upper bounds to extract 1D from
            (in arcsec). Example: (-1.8, -3.2)
        """
        if not isinstance(fname, Path):
            fname = Path(fname)

        if fname.suffix != ".fits":
            raise ValueError(
                "Cannot read this type of file. It must be " "a '.fits' file."
            )

        wvlg, spatial, flux_2D, error_2D = read_fits_2D_spectrum(fname)

        wvlg, flux, error = extract_1D_from_2D(
            wvlg, spatial, spatial_bounds=spatial_bounds, flux=flux_2D, error=error_2D
        )

        spectrum = Spectrum1D(
            spectral_axis=wvlg * u.AA,
            flux=flux * ergscm2AA,
            uncertainty=StdDevUncertainty(error),
        )

        self.spectrum["2D"] = {
            "fname": fname,
            "wave": wvlg * u.AA,
            "spatial": spatial * u.arcsec,
            "flux": flux * ergscm2AA,
            "error": error * ergscm2AA,
        }

        self.spectrum["1D"] = {
            "fname_2D": fname,
            "extraction_bounds": spatial_bounds,
            "data": spectrum,
        }

    def extract_line_region(self, spectrum=None, bounds=None):
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

        if spectrum is None:
            spectrum = self.spectrum["1D"]["data"]

        if bounds is None:
            # If no bounds and a redshift guess is provided
            # try to define reasonable bounds of ~ 100 Angstrom around
            # the expected line center
            if "z_guess" in self.properties.keys():
                wmin = self.properties["obs_awav_guess"] - 50 * u.AA * (
                    1 + self.properties["z_guess"]
                )
                wmax = self.properties["obs_awav_guess"] + 50 * u.AA * (
                    1 + self.properties["z_guess"]
                )
            else:
                raise ValueError(
                    "Please provide bounds or a guess redshift to extract a spectrum."
                )
        else:
            wmin, wmax = bounds

        log.debug(f"Attempting to extract the following bounds: ({wmin}, {wmax})")

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

    def _get_bounds(self, bounds=None, default=None):
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
                wmin = np.min(self.spectrum["wvlg"]).value * self.spectrum["wvlg"].unit
                wmax = np.max(self.spectrum["wvlg"]).value * self.spectrum["wvlg"].unit
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
        try:
            default_bounds = self.fit["bounds"]
        except KeyError:
            default_bounds = None

        try:
            continuum = self.fit["continuum"]["flux"]
        except KeyError:
            raise ValueError(
                "You must fit the continuum before you can "
                "measure a flux by integration."
            )

        wmin, wmax = self._get_bounds(bounds, default=default_bounds)

        flux, error = integrate_flux(
            wvlg=self.spectrum["wvlg"],
            flux=self.spectrum["flux"],
            error=self.spectrum["error"],
            wvlg_min=wmin,
            wvlg_max=wmax,
            continuum=continuum,
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

        try:
            mean_guess = self.properties["obs_wvlg"]
        except KeyError:
            mean_guess = np.median(self.spectrum["wvlg"])

        initial_guess = {
            "mean": args.get("mean", mean_guess),
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
            window=SpectralRegion(*self.fit["bounds"]),
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
            self.spectrum["flux"] - self.fit["flux"]
        ) / self.spectrum["error"]

    def derive_upper_limit(self, line_center, line_width, bounds=None):
        try:
            default_bounds = (
                line_center + 3 * line_width,
                line_center - 3 * line_width,
            )
        except KeyError:
            default_bounds = None

        wmin, wmax = self._get_bounds(bounds, default=default_bounds)

        mean, noise, snr = measure_noise(
            self.spectrum["wvlg"],
            self.spectrum["flux"] - self.spectrum["fit"]["continuum"]["flux"],
            wvlg_min=wmin,
            wvlg_max=wmax,
        )

        self.fit["results"]["mean"] = line_center
        self.fit["results"]["stddev"] = line_width
        self.fit["results"]["amplitude"] = noise

        self.properties["flux_lim"] = gaussian_infinite_integral(
            stddev=self.fit["results"]["stddev"],
            amplitude=self.fit["results"]["amplitude"],
        )

    def derive_properties_from_fit(self):
        self.properties["obs_awav_fit"] = self.fit["results"]["mean"]
        self.properties["FWHM_lam"] = sigma_to_fwhm(self.fit["results"]["stddev"])
        self.properties["FWHM_vel"] = fwhm_w2v(
            fwhm_w=self.properties["FWHM_lam"], w0=self.properties["obs_awav_fit"]
        )
        self.properties["flux_fit"] = gaussian_infinite_integral(
            amplitude=self.fit["results"]["amplitude"],
            stddev=self.fit["results"]["stddev"],
        )
        self.properties["z_fit"] = (
            self.properties["obs_awav_fit"] / self.properties["rest_awav"] - 1
        )

    # Visualization stuff
    def show_spectrum(self, ax=None, **kwargs):
        plot_spectrum(
            wvlg=self.spectrum["wvlg"],
            flux=self.spectrum["flux"],
            error=self.spectrum["error"],
            ax=ax,
            **kwargs,
        )

    def show_continuum(
        self, ax=None, show_fitting_regions=True, show_spec_kwargs={}, **kwargs
    ):
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

        self.show_spectrum(ax=ax, **show_spec_kwargs)
        plot_continuum(
            wvlg=self.spectrum["wvlg"],
            flux=self.fit["continuum"]["flux"],
            regions=self.fit["continuum"]["regions"] if show_fitting_regions else None,
            ax=ax,
            **kwargs,
        )

    def show_fit(self, model_plot_kw={}, spec_plot_kw={}, resid_plot_kw={}, show_legend=True):
        if self.properties["detected"]:
            flux_str = format_to_string(
                gaussian_infinite_integral(
                    amplitude=self.fit["results"]["amplitude"],
                    stddev=self.fit["results"]["stddev"],
                ).value
            )
        else:
            flux_str = format_to_string(
                self.properties["flux_lim"].value, lim_type="upper"
            )

        z = self.fit["results"]["mean"] / self.properties["rest_awav"] - 1
        fwhm_lam = sigma_to_fwhm(self.fit["results"]["stddev"])
        fwhm_vel = fwhm_w2v(fwhm_w=fwhm_lam, w0=self.fit["results"]["mean"])

        default_label = (
            r"$z=$"
            + f" {z.value:.4f} "
            + "\nFWHM ="
            + f" {fwhm_lam.value:.1f} "
            + r"$\rm \AA$"
            + f" ({fwhm_vel.value:.0f} km/s)"
            + "\nFlux = "
            + flux_str
            + r" $\rm erg/s/cm^{2}$"
        )

        model_plot_kw['label'] = model_plot_kw.pop('label', default_label)

        plot_fit(
            wvlg=self.spectrum["wvlg"],
            flux=self.spectrum["flux"],
            error=self.spectrum["error"],
            model=self.fit["flux"],
            residuals=self.fit['residuals'],
            fit_bounds=self.fit['bounds'],
            spec_plot_kw=spec_plot_kw,
            model_plot_kw=model_plot_kw,
            resid_plot_kw=resid_plot_kw,
            show_legend=show_legend,
        )
