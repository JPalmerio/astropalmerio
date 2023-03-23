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
]

import logging
import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity

log = logging.getLogger(__name__)

ergscm2AA = u.def_unit(
    s="erg/s/cm2/AA",
    represents=u.Unit("erg s^-1 cm^-2 AA^-1"),
    format={"latex": r"\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}"},
)
ergscm2 = u.def_unit(
    s="erg/s/cm2",
    represents=u.Unit("erg s^-1 cm^-2"),
    format={"latex": r"\mathrm{erg\,s^{-1}\,cm^{-2}}"},
)


def _quantity_to_array(x):
    if isinstance(x, Quantity):
        x = np.atleast_1d(x.value)
    else:
        x = np.atleast_1d(x)
    return x


def air_to_vac(awav):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006
    """
    try:
        wlum = awav.to(u.um).value
    except AttributeError:
        log.debug(
            "Unitless value passed to air_to_vac, make sure you "
            " know what you are doing (units should be microns)."
        )
        wlum = awav
    return (1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)) * awav


def vac_to_air(wave):
    """
    Griesen 2006 reports that the error in naively inverting Eqn 65 is less
    than 10^-9 and therefore acceptable.  This is therefore eqn 67
    """
    try:
        wlum = wave.to(u.um).value
    except AttributeError:
        log.debug(
            "Unitless value passed to vac_to_air, make sure you "
            " know what you are doing (units should be microns)."
        )
        wlum = wave

    nl = 1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)
    return wave / nl


def vel_to_wave(vel, w0):
    """Summary

    Parameters
    ----------
    vel : TYPE
        Description
    w0 : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(vel, Quantity):
        vel = vel.to(u.km / u.s)
    else:
        vel = vel * u.km / u.s
        log.info(
            "Input velocity is not a quantity so it has no units." " Assuming km/s."
        )

    if isinstance(w0, Quantity):
        w0 = w0.to(u.AA)
    else:
        log.info(
            "Input center wavelength is not a quantity so it has no units."
            " Assuming Angstrom."
        )
        w0 = w0 * u.AA

    wave = vel.to(u.AA, equivalencies=u.doppler_optical(w0))

    return wave if isinstance(vel, Quantity) else wave.value


def wave_to_vel(wave, w0):
    """Summary

    Parameters
    ----------
    wave : TYPE
        Description
    w0 : TYPE
        Description

    Returns
    -------
    TYPE
        Description

    """
    if isinstance(wave, Quantity):
        wave = wave.to(u.AA)
    else:
        wave = wave * u.AA
        log.info(
            "Input wavelength is not a quantity so it has no units."
            " Assuming Angstrom."
        )
    if isinstance(w0, Quantity):
        w0 = w0.to(u.AA)
    else:
        log.info(
            "Input center wavelength is not a quantity so it has no units."
            " Assuming Angstrom."
        )
        w0 = w0 * u.AA

    vel = wave.to(u.km / u.s, equivalencies=u.doppler_optical(w0))

    return vel if isinstance(wave, Quantity) else vel.value


def fwhm_v2w(fwhm_v, w0):
    # Calculate Half Width at Half Max in velocity space
    # This is necessary because velocity to wavelength is symmetric but
    # not linear
    hwhm_v = fwhm_v / 2.0
    fwhm_w = 2.0 * hwhm_v.to(w0.unit, equivalencies=u.doppler_optical(w0)) - w0
    return fwhm_w


def fwhm_w2v(fwhm_w, w0):
    # Calculate Half Width at Half Max in wavelength space
    # This is necessary because wavelength to velocity is symmetric but
    # not linear
    hwhm_w = w0 + fwhm_w / 2.0
    fwhm_v = 2.0 * hwhm_w.to(u.Unit("km/s"), equivalencies=u.doppler_optical(w0))
    return fwhm_v


def fwhm_to_sigma(fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma


def sigma_to_fwhm(sigma):
    fwhm = sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return fwhm
