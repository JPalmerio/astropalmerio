import logging
import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity

log = logging.getLogger(__name__)

ergscm2AA = u.Unit("erg s^-1 cm^-2 AA^-1")


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
    try:
        vel = vel.to(u.km / u.s)
    except AttributeError:
        vel = vel * u.km / u.s
    try:
        w0 = w0.to(u.AA)
    except AttributeError:
        w0 = w0 * u.AA

    w = vel.to(u.AA, equivalencies=u.doppler_optical(w0))
    return w


def wave_to_vel(w, w0):
    w = w * u.AA
    w0 = w0 * u.AA
    v = w.to(u.km / u.s, equivalencies=u.doppler_optical(w0))
    return v


def FWHM_v2w(vel, w0):
    wvlg = vel.to(w0.unit, equivalencies=u.doppler_optical(w0)) - w0
    return wvlg


def FWHM_w2v(wvlg, w0):
    wvlg = wvlg + w0
    vel = wvlg.to(u.km / u.s, equivalencies=u.doppler_optical(w0))
    return vel


def FWHM2sigma(FWHM):
    sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma


def sigma2FWHM(sigma):
    FWHM = sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return FWHM
