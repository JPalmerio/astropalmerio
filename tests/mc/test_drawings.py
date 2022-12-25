
import numpy as np
import unittest
from astropalmerio.mc.drawings import sample_uniform_between, sample_from_CDF


class TestDrawings(unittest.TestCase):

    def test_uniform_sample(self):
        np.random.seed(0)
        draw = sample_uniform_between(0, 1, nb_draws=1)
        assert (draw >= 0) and (draw <= 1)
        draw = sample_uniform_between(-1., 1., nb_draws=100)
        assert len(draw) == 100
        assert all(draw >= -1) and all(draw <= 1)

    def test_sample_CDF(self):
        np.random.seed(0)
        x = np.asarray([0,1,2,3,4,5])
        Fx = x*x
        # test no bounds and number of draws
        draw = sample_from_CDF(x, Fx, nb_draws=100)
        assert all(draw >= x.min()) and all(draw <= x.max())
        assert len(draw) == 100
        # test val_min
        draw = sample_from_CDF(x, Fx, nb_draws=100, val_min=3)
        assert all(draw >= 3) and all(draw <= x.max())
        # test val_max
        draw = sample_from_CDF(x, Fx, nb_draws=100, val_max=4)
        assert all(draw >= x.min()) and all(draw <= 4)
        # val_min and val_max bounds are larger than span of x
        draw = sample_from_CDF(x, Fx, nb_draws=100, val_min=-1, val_max=6)
        assert all(draw >= x.min()) and all(draw <= x.max())
        # val_min and val_max bounds are smaller than span of x
        draw = sample_from_CDF(x, Fx, nb_draws=100, val_min=1, val_max=3)
        assert all(draw >= 1) and all(draw <= 3)
        # Invalid value for val_min and val_max
        with self.assertRaises(ValueError):
            draw = sample_from_CDF(x, Fx, nb_draws=100, val_min=10)
        with self.assertRaises(ValueError):
            draw = sample_from_CDF(x, Fx, nb_draws=100, val_max=-10)




