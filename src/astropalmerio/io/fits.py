# -*- coding: utf-8 -*-

__all__ = [
    "read_fits_1D_spectrum",
    "read_fits_2D_spectrum",
]

import logging
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

log = logging.getLogger(__name__)


def read_fits_1D_spectrum(filename):
    """
    A function to read data from a 1D spectrum fits file.
    Returns wavelength in angstroms, flux and uncertainties in erg/s/cm2/A .
    """
    if isinstance(filename, Path):
        filename = str(filename)
    hdu_list = fits.open(filename)

    # Start by looking for BinTableHDU
    try:
        hdr = hdu_list[1].header
        data = Table.read(filename, hdu=1)
        cols = [c.lower() for c in data.columns]
        wvlg_key = [c for c in cols if "wave" in c or "wvlg" in c][0]
        flux_key = [c for c in cols if "flux" in c][0]
        uncertainty_key = [c for c in cols if "err" in c][0]
        wvlg = np.array(data[wvlg_key])
        flux = np.array(data[flux_key])
        uncertainty = np.array(data[uncertainty_key])

        return wvlg, flux, uncertainty

    except ValueError:
        log.debug("No BinTableHDU found, trying different method")
    except IndexError:
        log.debug("No extension 1 found")

    # If no Table found, look for an HDU extension named 'FLUX'
    try:
        data_index = hdu_list.index_of("FLUX")
        hdr = hdu_list[data_index].header
        flux = fits.getdata(filename, ext=data_index)
        log.debug("Found FLUX extension")
    except KeyError:
        log.debug(
            "No BinTableHDU or FLUX extension found, falling back to default index"
        )
        try:
            hdr = hdu_list[0].header
            flux = fits.getdata(filename, ext=0)
            log.debug("Found data in extension 0")
        except Exception as e:
            log.error(e)
            raise ValueError(
                f"Could not understand FITS file format for {filename}."
            ) from e

    # Look for uncertainties as well
    try:
        unc_index = hdu_list.index_of("ERRS")
        uncertainty = fits.getdata(filename, ext=unc_index)
        log.debug("Found ERRS extension")
    except KeyError:
        log.warning("No ERRS extension found; setting uncertainties to 0")
        uncertainty = 0.0 * flux

    # Check for wavelength units
    try:
        cunit1 = hdr["CUNIT1"].strip().lower()
        if cunit1 == "angstroms":
            cunit1 = "angstrom"
        wvlg_unit = u.Unit(cunit1)
    except KeyError:
        log.warning("No unit found in header for wavelength, assuming angstroms")
        wvlg_unit = u.AA
    wvlg_step = (
        (hdr["CDELT1"] * wvlg_unit).to("AA").value
    )  # Make sure units are Angstrom
    wvlg_init = (
        (hdr["CRVAL1"] * wvlg_unit).to("AA").value
    )  # Make sure units are Angstrom
    wvlg = np.array([wvlg_init + i * wvlg_step for i in range(flux.shape[0])])

    return wvlg, flux, uncertainty


def read_fits_2D_spectrum(filename, verbose=False):
    """
    A function to read data from a 2D spectrum.
    Returns wavelength in angstroms, spatial position in arcsec, flux and uncertainties in erg/s/cm2/A.
    """

    hdu_list = fits.open(filename)
    hdu_names = [hdu.name for hdu in hdu_list]
    try:
        # Look for an extension name that contains the letters 'flux'
        flux_hdu_name = [n for n in hdu_names if "flux" in n.lower()][0]
        log.info(f"Found flux extension named: {flux_hdu_name}")
        data_index = hdu_list.index_of(flux_hdu_name)
        data = fits.getdata(filename, ext=data_index)
    except IndexError:
        data_index = 0
        data = fits.getdata(filename, ext=data_index)

    try:
        # Look for an extension name that contains the letters 'err'
        unc_hdu_name = [n for n in hdu_names if "err" in n.lower()][0]
        log.info(f"Found uncertainty extension named: {unc_hdu_name}")
        unc_index = hdu_list.index_of(unc_hdu_name)
        uncertainty = fits.getdata(filename, ext=unc_index)
    except IndexError:
        log.warning("No uncertainty extension found in file %s", filename)
        uncertainty = np.zeros(data.shape)

    hdr = hdu_list[data_index].header
    if verbose:
        print(repr(hdr))

    # Check for wavelength units
    try:
        cunit1 = hdr["CUNIT1"].strip().lower()
        if cunit1 == "angstroms":
            cunit1 = "angstrom"
        wvlg_unit = u.Unit(cunit1)
    except KeyError:
        log.warning("No unit found in header for wavelength, assuming angstroms")
        wvlg_unit = u.AA
    wvlg_step = (
        (hdr["CDELT1"] * wvlg_unit).to("AA").value
    )  # Make sure units are Angstrom
    wvlg_init = (
        (hdr["CRVAL1"] * wvlg_unit).to("AA").value
    )  # Make sure units are Angstrom
    wvlg = np.array([wvlg_init + i * wvlg_step for i in range(data.shape[1])])

    # Assumes the units are arcseconds
    try:
        spatial_init = hdr["CRVAL2"]
    except KeyError:
        spatial_init = 0
    spatial_step = hdr["CDELT2"]
    spatial = np.array([spatial_init + i * spatial_step for i in range(data.shape[0])])

    return wvlg, spatial, data, uncertainty
