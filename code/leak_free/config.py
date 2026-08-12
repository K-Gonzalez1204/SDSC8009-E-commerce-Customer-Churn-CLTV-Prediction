from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "leak_free_churn"
INPUT_CSV = DATA_DIR / "ecommerce_customer_behavior_dataset_v2.csv"
INPUT_SHA256 = "1588736FA27A1AC5C0C4BA1162C390AEC7D1735EA4A4B9417275EE86CC173285"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
K_BEST = 20

FEATURE_PERIOD_START = date(2023, 1, 1)
CUTOFF_DATE = date(2023, 10, 1)
LABEL_PERIOD_END = date(2024, 3, 1)
CHURN_RECENCY_DAYS = 90

TARGET = "is_churn"
