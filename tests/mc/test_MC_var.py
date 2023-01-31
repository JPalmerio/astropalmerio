import numpy as np
import unittest
from astropalmerio.mc.MC_var import MC_var
import logging

log = logging.getLogger(__name__)


class TestMCvar(unittest.TestCase):

    def test_MC_var_init(self):
        np.random.seed(0)
        # Empty assignment
        with self.assertRaises(ValueError):
            MC_var(value=0)

        # Check that floats and ints are well behaved
        MC_var(value=0, error=1)
        MC_var(value=0., error=1)
        MC_var(value=0, error=1.)
        x = MC_var(value=0., error=1.)

        # Check that other types raise error
        with self.assertRaises(ValueError):
            MC_var(value='a string', error=' another string')
            MC_var(value=False, error=False)

        # Check error plus and minus
        x = MC_var(value=10., error=[3, 3])
        assert x.error['plus'] == 3
        assert x.error['minus'] == 3
        # Check correct conversion if negative error
        x = MC_var(value=10., error=[-3, 3])
        assert x.error['plus'] == 3
        assert x.error['minus'] == 3
        # Check wrong error
        with self.assertRaises(TypeError):
            MC_var(value=10., error=[None])

        # Check limits
        with self.assertRaises(ValueError):
            x = MC_var(value=10., lolim=True)
        with self.assertRaises(ValueError):
            x = MC_var(value=10., uplim=True)

        # Check min and max value
        MC_var(value=10., lolim=True, val_max=11.)
        MC_var(value=10., uplim=True, val_min=9.)
        # Check error if max val smaller than lower limit
        with self.assertRaises(ValueError):
            MC_var(value=10., lolim=True, val_max=9.)
        # Check error if min val larger than upper limit
        with self.assertRaises(ValueError):
            MC_var(value=10., uplim=True, val_min=11.)
        # Check error if both upper and lower limit are provided
        with self.assertRaises(ValueError):
            MC_var(value=10., uplim=True, lolim=True)
        # Check error if both limits and erros are provided
        with self.assertRaises(ValueError):
            MC_var(value=10., error=[1,2], uplim=True)

    def test_MC_var_str(self):
        x = MC_var(value=10., error=3)
        print(x)

