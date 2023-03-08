# -*- coding: utf-8 -*-

__all__ = [
    'Cardelli89',
    'Pei92_MW',
    'Pei92_SMC',
    'Calzetti_SB',
    'get_extinction',
    'extinguish_line',
    'correct_line_for_extinction',
    'star_formation_rate',
    'Av_from_Balmer_decrement',
    'H_RATIOS',
    'R23',
    'OIIIb_Hbeta',
    'OII_Hbeta',
    'OIIIb_OII',
    'NeIII_OII',
    'NII_Ha',
    'OIIIb_NII',
    'R23_rec',
    'R23_scatter_rec',
    'OIIIb_Hbeta_rec',
    'OIIIb_Hbeta_scatter_rec',
    'OII_Hbeta_rec',
    'OII_Hbeta_scatter_rec',
    'OIIIb_OII_rec',
    'OIIIb_OII_scatter_rec',
    'NeIII_OII_rec',
    'NeIII_OII_scatter_rec',
    'NII_Halpha_rec',
    'NII_Halpha_scatter_rec',
    'OIIIb_NII_rec',
    'OIIIb_NII_scatter_rec',
    'plot_relations',
]

from astropalmerio.galaxies.extinction import Cardelli89, Pei92_MW, Pei92_SMC, Calzetti_SB, get_extinction, extinguish_line, correct_line_for_extinction
from astropalmerio.galaxies.properties import star_formation_rate, Av_from_Balmer_decrement, H_RATIOS
from astropalmerio.galaxies.M08 import R23, OIIIb_Hbeta, OII_Hbeta, OIIIb_OII, NeIII_OII, NII_Ha, OIIIb_NII, R23_rec, R23_scatter_rec, OIIIb_Hbeta_rec, OIIIb_Hbeta_scatter_rec, OII_Hbeta_rec, OII_Hbeta_scatter_rec, OIIIb_OII_rec, OIIIb_OII_scatter_rec, NeIII_OII_rec, NeIII_OII_scatter_rec, NII_Halpha_rec, NII_Halpha_scatter_rec, OIIIb_NII_rec, OIIIb_NII_scatter_rec, plot_relations
