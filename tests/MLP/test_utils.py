import unittest
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from src.TimeSeries.TimeSeries import DiffOperation


class TestDiff(unittest.TestCase):

    def setUp(self) -> None:
        length = 40
        values = [i + np.random.rand()*i/3 for i in range(length)]
        start_date = dt(1984, 1, 1)
        indexes = [start_date + timedelta(days=i) for i in range(length)]
        self._serie = pd.Series(data=values, index=indexes)

        external_serie_indexes = [start_date + timedelta(days=i+1) for i in range(10)]
        external_serie_values = [1 for _ in range(10)]
        self._external_serie = pd.Series(data=external_serie_values, index=external_serie_indexes)

    def test_diff(self):
        diff_operator = DiffOperation()
        diff = diff_operator.fit_transform(data=self._serie)
        inverted = diff_operator.invert(diff)
        self.assertEqual(len(self._serie), len(inverted), msg="The length of the serie after inverse is not "
                                                              "equal to the original temporal serie.")
        difference = (self._serie.values - inverted.values).sum()
        self.assertAlmostEqual(difference, 0)

    def test_external_diff(self):
        diff_operator = DiffOperation()
        diff_operator.fit_transform(data=self._serie)
        inverted = diff_operator.partial_invert(self._external_serie)
        # TODO: Check values and finish test


if __name__ == '__main__':
    unittest.main()
