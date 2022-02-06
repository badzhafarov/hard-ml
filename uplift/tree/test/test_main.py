from unittest import TestCase
import numpy as np

from src.main import UpliftTreeRegressor


class Test(TestCase):

    def test_uplift(self):
        regressor = UpliftTreeRegressor()
        result = regressor.uplift(np.array([1, 1, 0, 0]), np.array([100, 10, 5, 5]))
        assert result == 50

    def test_get_delta_delta_p(self):
        regressor = UpliftTreeRegressor()

        delta_delta_p = regressor.get_delta_delta_p(
            X=np.array([0.085, 0.087, 0.192, 0.202, 0.248, 0.283, 0.318, 0.356, 0.359, 0.381,
                        0.583, 0.642, 0.689, 0.692, 0.771, 0.777, 0.779, 0.813, 0.918, 0.985]),
            y=np.array([0.170, 0.192, 0.423, 0.445, 0.496, 0.622, 0.700, 0.784, 0.789, 0.838,
                        1.166, 1.413, 1.378, 1.522, 1.541, 1.553, 1.558, 1.788, 2.019, 1.971]),
            treatment=np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]),
            threshold=0.7)

        assert round(delta_delta_p, 3) == 0.277
