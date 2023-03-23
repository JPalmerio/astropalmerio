# -*- coding: utf-8 -*-

__all__ = [
    "air_to_vac",
    "vac_to_air",
    "vel_to_wave",
    "wave_to_vel",
    "fwhm_v2w",
    "fwhm_w2v",
    "fwhm_to_sigma",
    "sigma_to_fwhm",
    "ergscm2AA",
    "ergscm2",
    "EmissionLine",
    "extract_1D_from_2D",
    "integrate_flux",
    "measure_noise",
    "gaussian_fct",
    "gaussian_infinite_integral",
    "correct_lambda_for_radial_velocity",
    "calc_vel_corr",
    "show_flux_integration_bounds",
    "plot_fit",
    "plot_model",
    "plot_continuum",
    "plot_spectrum",
    "show_regions",
    "set_standard_spectral_labels",
    "fig_resid",
]

from astropalmerio.spectra.conversions import (
    air_to_vac,
    vac_to_air,
    vel_to_wave,
    wave_to_vel,
    fwhm_v2w,
    fwhm_w2v,
    fwhm_to_sigma,
    sigma_to_fwhm,
    ergscm2AA,
    ergscm2,
)
from astropalmerio.spectra.emission_lines import EmissionLine
from astropalmerio.spectra.reduction import extract_1D_from_2D
from astropalmerio.spectra.utils import (
    integrate_flux,
    measure_noise,
    gaussian_fct,
    gaussian_infinite_integral,
    correct_lambda_for_radial_velocity,
    calc_vel_corr,
)
from astropalmerio.spectra.visualization import (
    show_flux_integration_bounds,
    plot_fit,
    plot_model,
    plot_continuum,
    plot_spectrum,
    show_regions,
    set_standard_spectral_labels,
    fig_resid,
)
