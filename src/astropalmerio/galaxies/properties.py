# -*- coding: utf-8 -*-


__all__ = ["star_formation_rate", "Av_from_Balmer_decrement", "H_RATIOS"]

import logging
import numpy as np
from itertools import combinations
from .extinction import get_extinction

log = logging.getLogger(__name__)

# Assuming HII regions in a case B recombination with
# electron temperature and density : 10 000 K and 10^4 cm^-3
# (Osterbrock & Ferland 2006)
H_RATIOS = {
    "Hb": 1.0,
    "Ha": 2.847,
    "Hg": 0.469,
    "Hd": 0.260,
    "Pa": 0.332,
    "Pb": 0.162,
    "Pg": 0.0901,
    "Pd": 0.0554,
    "Bra": 0.0778,
    "Brb": 0.0447,
    "Brg": 0.0275,
    "Brd": 0.0181,
}


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


def Av_from_Balmer_decrement(
    F_Ha=None,
    F_Hb=None,
    F_Hg=None,
    F_Pa=None,
    F_Hd=None,
    extinction="Pei+92 MW",
):
    """
    Calculates the Av from the Balmer decrement method, assuming HII
    regions in a case B recombination whith electron temperature and
    density : 10 000 K and 10^4 cm^-3 (Osterbrock & Ferland 2006).
    Returns Av
    """

    extinction_func = get_extinction(extinction)

    if (
        (F_Ha is None)
        & (F_Hb is None)
        & (F_Hg is None)
        & (F_Pa is None)
        & (F_Hd is None)
    ):
        raise ValueError(
            "Please provide at least two of ['F_Ha', 'F_Hb', 'F_Hg', 'F_Pa', 'F_Hd']."
        )

    lines = {
        "Ha": {
            "flux": F_Ha,
            # "err": F_Halpha[1] if F_Halpha is not None else None,
            "wvlg": 6562.819,  # in Angstrom in air
        },
        "Hb": {
            "flux": F_Hb,
            # "err": F_Hbeta[1] if F_Hbeta is not None else None,
            "wvlg": 4861.333,  # in Angstrom in air
        },
        "Hg": {
            "flux": F_Hg,
            # "err": F_Hgamma[1] if F_Hgamma is not None else None,
            "wvlg": 4340.471,  # in Angstrom in air
        },
        "Pa": {
            "flux": F_Pa,
            # "err": F_Palpha[1] if F_Hgamma is not None else None,
            "wvlg": 18751.3,  # in Angstrom in air
        },
        "Hd": {
            "flux": F_Hd,
            # "err": F_Hdelta[1] if F_Hdelta is not None else None,
            "wvlg": 4101.742,  # in Angstrom in air
        },
    }

    Avs = {}
    # Iterate over the set of pairs of lines that can be combined
    for l1, l2 in combinations(lines.keys(), 2):
        line1 = lines[l1]
        line2 = lines[l2]
        if (line1["flux"] is not None) and (line2["flux"] is not None):
            delta_ext = extinction_func(line1["wvlg"]) - extinction_func(line2["wvlg"])
            Av = _calc_Av(
                observed_ratio=line1["flux"] / line2["flux"],
                theoretical_ratio=H_RATIOS[f"{l1}"] / H_RATIOS[f"{l2}"],
                delta_extinction=delta_ext,
            )
            Avs[f"{l1}/{l2}"] = Av
            log.debug(
                "Estimating Av from:\n"
                f"Observed {l1}/{l2} = {line1['flux'] / line2['flux']}\n"
                f"Theoretical {l1}/{l2} = {H_RATIOS[f'{l1}']/H_RATIOS[f'{l2}']}\n"
                f"Av = {Av}"
            )
    # Av = np.mean(np.array(Avs), axis=0)

    return Avs


def _calc_Av(observed_ratio, theoretical_ratio, delta_extinction):
    Av = -2.5 * np.log10(observed_ratio / theoretical_ratio) / delta_extinction
    return Av
