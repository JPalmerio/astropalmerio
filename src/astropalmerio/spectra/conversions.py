import logging
import numpy as np
import astropy.units as u

log = logging.getLogger(__name__)

ergscm2AA = u.Unit("erg s^-1 cm^-2 AA^-1")


def air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006
    """
    wlum = wavelength.to(u.um).value
    return (
        1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)
    ) * wavelength


def vac_to_air(wavelength):
    """
    Griesen 2006 reports that the error in naively inverting Eqn 65 is less
    than 10^-9 and therefore acceptable.  This is therefore eqn 67
    """
    wlum = wavelength.to(u.um).value
    nl = 1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)
    return wavelength / nl


def v2w(v, w0):
    v = v * u.km / u.s
    w0 = w0 * u.AA
    w = v.to(u.AA, equivalencies=u.doppler_optical(w0))
    return w


def w2v(w, w0):
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
