
import numpy as np
import unittest
from astropalmerio.mc.utils import get_corresponding_y_value, log_to_lin, lin_to_log


class TestUtils(unittest.TestCase):

    def test_log_to_lin(self):
        x, x_errp, x_errm = log_to_lin(1, 1)
        assert (x, x_errp, x_errm) == (10, 90, 9)

    def test_get_corresponding_y_value(self):





