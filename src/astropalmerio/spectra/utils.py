# -*- coding: utf-8 -*-

__all__ = [
    "integrate_flux",
    "measure_noise",
    "gaussian_fct",
    "gaussian_infinite_integral",
    "correct_lambda_for_radial_velocity",
    "calc_vel_corr",
]

import logging
import numpy as np
import astropy.constants as cst
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
import astropalmerio.io as io
from astropalmerio.mc.realizations import MC_realization

log = logging.getLogger(__name__)


def integrate_flux(
    wvlg,
    flux,
    wvlg_min,
    wvlg_max,
    uncertainty=None,
    continuum=None,
    MC=False,
    N_MC=1000,
):
    """
    Calculate the integrated flux over a given wavelength range.
    This can be used to estimated the flux of a line.
    Continuum can be specified and will be subtracted to the
    flux density array.
    Returns the uncertainty on the summed flux if uncertainty is provided
    (assumes the uncertainty is the standard deviation).
    """
    imin = wvlg.searchsorted(wvlg_min)
    imax = wvlg.searchsorted(wvlg_max)

    if continuum is None:
        continuum = np.zeros(flux.shape[0])

    wvlg_step = wvlg[1] - wvlg[0]
    sub_flux = flux[imin:imax] - continuum[imin:imax]
    flux_summed = np.sum(sub_flux * wvlg_step)

    if uncertainty is not None:
        sub_unc = uncertainty[imin:imax]
        if MC:
            sub_flux_real = MC_realization(
                sample=sub_flux, sample_uncp=sub_unc, sample_uncm=sub_unc, N_real=N_MC
            )
            flux_summed_real = np.sum(sub_flux_real * wvlg_step, axis=1)
            flux_summed = np.quantile(flux_summed_real, 0.5)
            uncm = flux_summed - np.quantile(flux_summed_real, 0.16)
            uncp = np.quantile(flux_summed_real, 0.84) - flux_summed
            uncertainty_summed = (uncm, uncp)
        else:
            uncertainty_summed = np.sqrt(np.sum(sub_unc**2 * wvlg_step**2))
    else:
        if MC:
            log.warning(
                "You asked for Monte Carlo uncertainties but uncertainty is None. "
                "Returning 0 for the uncertainty."
            )
        uncertainty_summed = 0

    return flux_summed, uncertainty_summed


def measure_noise(wvlg, flux, wvlg_min, wvlg_max):
    """
    Calculate the noise over a given wavelength range.
    Return the mean, the noise and the S/N.
    """

    imin = wvlg.searchsorted(wvlg_min)
    imax = wvlg.searchsorted(wvlg_max)
    noise = np.std(flux[imin:imax])
    mean = np.mean(flux[imin:imax])
    snr = mean / noise
    return mean, noise, snr


def gaussian_fct(x, mean, stddev, amplitude=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    mean : TYPE
        Description
    stddev : TYPE
        Description
    amp : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if amplitude is None:
        amplitude = 1.0 / (np.sqrt(2.0 * np.pi) * stddev)
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * stddev**2))


def gaussian_infinite_integral(stddev, amplitude=None):
    """Summary

    Parameters
    ----------
    mean : TYPE
        Description
    sddev : TYPE
        Description
    amp : None, optional
        Description
    """
    if amplitude is None:
        amplitude = 1.0 / (np.sqrt(2.0 * np.pi) * stddev)
    return np.sqrt(2 * np.pi) * amplitude * stddev


def correct_lambda_for_radial_velocity(fname, kind="barycentric", mode="1D"):
    """
    Correct lambda for heliocentric or barycentric radial velocity shift.
    Return the corrected wavelength
    """
    c = cst.c.to("km/s")
    if mode == "1D":
        wvlg, _, _ = io.read_fits_1D_spectrum(fname)
    elif mode == "2D":
        wvlg, _, _, _ = io.read_fits_2D_spectrum(fname)
    else:
        raise ValueError
    vcorr = calc_vel_corr(header=fits.getheader(fname), kind=kind)
    wvlg_corr = wvlg * (1.0 + vcorr / c)
    return wvlg_corr


def calc_vel_corr(header, kind="barycentric"):
    """
    Calculates the radial velocity correction given an observing date, a telescope position
    and an object's RA and DEC in the sky (along with the reference frame).
    Returns the velocity correction for barycentric or heliocentric motion in km/s
    """
    ra2000 = header["RA"] * u.deg
    dec2000 = header["DEC"] * u.deg
    tel_lat = header["HIERARCH ESO TEL GEOLAT"] * u.deg
    tel_long = header["HIERARCH ESO TEL GEOLON"] * u.deg
    tel_alt = header["HIERARCH ESO TEL GEOELEV"] * u.m
    frame = header["RADECSYS"].lower()
    mjd = header["MJD-OBS"]
    exptime = header["EXPTIME"]

    coord = SkyCoord(ra2000, dec2000, frame=frame)
    date_obs = Time(
        mjd + exptime / (2.0 * 86400.0), format="mjd"
    )  # midpoint of observation
    tel_pos = EarthLocation.from_geodetic(lat=tel_lat, lon=tel_long, height=tel_alt)
    vel_corr = coord.radial_velocity_correction(
        kind=kind, obstime=date_obs, location=tel_pos
    )
    vel_corr = vel_corr.to("km/s")
    log.debug(
        "Velocity correction calculated: {:.3f} {:s}".format(
            vel_corr.value, vel_corr.unit
        )
    )

    return vel_corr
