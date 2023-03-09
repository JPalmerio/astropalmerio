# -*- coding: utf-8 -*-

__all__ = [
    "read_nddata",
    "read_fits_1D_spectrum",
    "read_fits_2D_spectrum",
]

from .ascii import read_nddata
from .fits import read_fits_1D_spectrum, read_fits_2D_spectrum
