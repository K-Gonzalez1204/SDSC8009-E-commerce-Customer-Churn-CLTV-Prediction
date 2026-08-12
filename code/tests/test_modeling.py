import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from leak_free.modeling import (
    bootstrap_auc,
    build_pipelines,
    evaluate_holdout,
    run_cv_with_oof,
    select_best,
    select_threshold,
)


class ModelingTest(unittest.TestCase):
    def test_five_pipelines(self):
        pipes = build_pipelines(pos_weight=1.0, k_best=10)
        self.assertEqual(
            set(pipes),
            {
                "LogisticRegression",
                "RandomForest",
                "XGBoost",
                "LightGBM",
                "Voting",
            },
        )

    def test_oof_cv_and_threshold(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (400, 10))
        y = np.array([0, 1] * 200)
        pipes = build_pipelines(pos_weight=1.0, k_best=10)
        scores, oof = run_cv_with_oof(pipes, X, y)
        self.assertEqual(set(scores), set(pipes))
        self.assertEqual(set(oof), set(pipes))
        self.assertEqual(len(oof["LogisticRegression"]), len(y))
        best = select_best(scores)
        self.assertIn(best, pipes)
        thr = select_threshold(y, oof[best])
        self.assertTrue(0.10 <= thr <= 0.90)

    def test_evaluate_holdout_and_bootstrap(self):
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (400, 10))
        y = np.array([0, 1] * 200)
        pipes = build_pipelines(pos_weight=1.0, k_best=10)
        metrics, prob = evaluate_holdout(
            pipes["LogisticRegression"], X[:300], y[:300], X[300:], y[300:], 0.5
        )
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        low, high = bootstrap_auc(y[300:], prob, seed=42, n=100)
        self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
