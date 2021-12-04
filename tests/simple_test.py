import shutil
import unittest

import numpy as np

import deann


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(10000, 30)

    def test_percentile(self):
        model = deann.DEANN(metric="cosine")
        model.fit(self.X)
        density = model.percentile(self.X)


if __name__ == "__main__":
    unittest.main()
