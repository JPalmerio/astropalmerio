import logging
import numpy as np

log = logging.getLogger(__name__)


def star_formation_rate(flux, z, line="Ha", normalization="Chabrier", cosmo=None):
    """
    Computes the SFR using H_alpha or [OII]3727 with different normalizations.
    Flux is interpreted as [erg/cm2/s].
    Note : if using OII, use combined flux of both lines in the doublet.
    Returns SFR in [M_sun/yr].
    SFR from H_alpha is from Kennicutt+98.

    Parameters
    ----------
    flux : float or array-like
        Integrated flux of the line in units of erg/cm2/s.
    z : float or array-like
        Redshift of the source
    line : str, optional, Default 'Ha'
        Line to use for the determination of the SFR. Can be one of
        'Ha', 'OII'.
    normalization : str, optional, Default 'Chabrier'
        Stellar Initial Mass Function (IMF) normalization to use. Can be
        one of ['Baldry', 'Chabrier', 'Kroupa', 'Salpeter'].
    cosmo : astropy cosmology object, optional
        The cosmology to use to calculate the luminosity distance from
        the redshift. Defaults to the values of Planck 2018.

    Returns
    -------
    SFR : float or array-like
        The star formation rate calculated in units of solar masses per
        year (Msun/yr).
    """

    norm = 0.0

    if normalization == "Chabrier":
        if line == "Ha":
            norm = 4.6  # Chabrier
        elif line == "OII":
            norm = 5.54 * 4.6 / 4.39  # Chabrier/Baldry
        else:
            raise ValueError(
                "Wrong argument for 'line' in star_formation_rate. Use 'Ha' or 'OII'."
            )
    elif normalization == "Baldry":
        if line == "Ha":
            norm = 4.39  # Baldry
        elif line == "OII":
            norm = 5.54 * 4.39
        else:
            raise ValueError(
                "Wrong argument for 'line' in star_formation_rate. Use 'Ha' or 'OII'."
            )
    elif normalization == "Kroupa":
        if line == "Ha":
            norm = 5.37  # Kroupa
        elif line == "OII":
            norm = 5.54 * 5.37 / 4.39  # Kroupa/Baldry
        else:
            raise ValueError(
                "Wrong argument for 'line' in star_formation_rate. Use 'Ha' or 'OII'."
            )
    elif normalization == "Salpeter":
        if line == "Ha":
            norm = 7.9  # Salpeter
        elif line == "OII":
            norm = 5.54 * 7.9 / 4.39  # Salpeter/Baldry
        else:
            raise ValueError(
                "Wrong argument for 'line' in star_formation_rate. Use 'Ha' or 'OII'."
            )
    else:
        raise ValueError(
            "Wrong argument for 'normalization' in star_formation_rate.\n"
            "Options are: ('Chabrier', 'Baldry', 'Kroupa', 'Salpeter')."
        )

    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo

        log.info("Using Planck 2018 cosmology from astropy : {}".format(cosmo))
    D_L = cosmo.luminosity_distance(z=z).to("cm").value
    Lum = flux * 4.0 * np.pi * D_L**2
    SFR = Lum * norm * 10 ** (-42)

    return SFR
