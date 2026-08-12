import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from leak_free.preprocess import QuantileClipTransformer, build_preprocessor


class PreprocessTest(unittest.TestCase):
    def test_clip_bounds_come_from_train(self):
        rng = np.random.default_rng(1)
        train = rng.normal(0, 1, (200, 5))
        test = rng.normal(0, 1, (50, 5))
        clip = QuantileClipTransformer()
        clip.fit(train)
        out = clip.transform(test)
        self.assertTrue(np.all(out <= clip.upper_bounds_))
        self.assertTrue(np.all(out >= clip.lower_bounds_))

    def test_pipeline_shape(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (300, 30))
        y = np.array([0, 1] * 150)
        pipe = build_preprocessor(k_best=20)
        pipe.fit(X, y)
        self.assertEqual(pipe.transform(X).shape[1], 20)


if __name__ == "__main__":
    unittest.main()
