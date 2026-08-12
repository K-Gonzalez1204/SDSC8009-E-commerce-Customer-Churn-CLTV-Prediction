import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

from leak_free.features import prepare_customer_frame
from leak_free.main import run_pipeline_from_frame
from tests.test_features import small_frame


class MainTest(unittest.TestCase):
    def test_smoke_pipeline_writes_artifacts(self):
        frame = prepare_customer_frame(small_frame())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = run_pipeline_from_frame(frame, output_dir=output)
            self.assertEqual(manifest["rows"], 60)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "cv_results.csv").is_file())
            self.assertTrue((output / "final_metrics.json").is_file())
            self.assertTrue((output / "best_model.pkl").is_file())
            self.assertTrue((output / "threshold_selection.csv").is_file())


if __name__ == "__main__":
    unittest.main()
