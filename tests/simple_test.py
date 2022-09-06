import shutil
import unittest

import numpy as np

import deann


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(21, 30)

    def test_percentile(self):
        model = deann.DEANN(k=30, metric="cosine", exact = True)
        model.fit(self.X)
        density = model.percentile(self.X)
        print(density)
    
    def test_gradient(self):
        self.X = np.random.randn(1000, 2)
        model = deann.DEANN(k=30, metric="cosine", exact = True)
        model.fit(self.X)
        grad = model.gradient(self.X)
        print(grad, self.X)


if __name__ == "__main__":
    unittest.main()
