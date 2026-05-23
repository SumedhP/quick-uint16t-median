import numpy as np

from sumedh_quick_uint16_median import median


class TestAll:
    def test_all(self):
        array = np.array([[10, 1, 3], [4, 2, 9]], dtype=np.uint16)

        assert 3 == median(array)

    def test_ignores_zeros(self):
        array = np.array([[0, 0, 0], [5, 6, 7]], dtype=np.uint16)

        assert 6 == median(array)

    def test_returns_nan_for_zero_heavy_data(self):
        array = np.array([[0, 0, 0], [0, 0, 7]], dtype=np.uint16)

        assert np.isnan(median(array, non_zero_threshold=0.5))
