import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import polars as pl

from leak_free.features import prepare_customer_frame


def small_frame():
    rows = []
    for cid, dates in {
        "C1": ["2023-02-01", "2023-05-01", "2023-09-01", "2023-11-01"],
        "C2": ["2023-03-01", "2023-08-01", "2024-01-01"],
        "C3": ["2023-01-15", "2023-04-01", "2023-12-15"],
        "C4": ["2023-06-01", "2023-09-15", "2024-02-01"],
        "C5": ["2023-01-01", "2023-04-01"],
        "C6": ["2023-02-10"],
    }.items():
        for date in dates:
            rows.append({
                "Customer_ID": cid,
                "Date": date,
                "Age": 30,
                "Gender": "Male",
                "City": "Istanbul",
                "Product_Category": "Electronics",
                "Unit_Price": 100.0,
                "Quantity": 1,
                "Discount_Amount": 0.0,
                "Total_Amount": 100.0,
                "Payment_Method": "Credit Card",
                "Device_Type": "Mobile",
                "Session_Duration_Minutes": 15,
                "Pages_Viewed": 8,
                "Is_Returning_Customer": True,
                "Delivery_Time_Days": 3,
                "Customer_Rating": 4,
            })
    return pl.DataFrame(rows)


class FeaturesTest(unittest.TestCase):
    def test_frame_shape_and_columns(self):
        frame = prepare_customer_frame(small_frame())
        self.assertEqual(frame.height, 6)
        self.assertIn("is_churn", frame.columns)
        self.assertIn("recency", frame.columns)
        self.assertEqual(
            len([c for c in frame.columns if c not in ("Customer_ID", "is_churn")]),
            30,
        )

    def test_churn_labels(self):
        frame = prepare_customer_frame(small_frame())
        labels = dict(
            zip(frame["Customer_ID"].to_list(), frame["is_churn"].to_list())
        )
        self.assertEqual(labels["C1"], 0)
        self.assertEqual(labels["C2"], 0)
        self.assertEqual(labels["C3"], 0)
        self.assertEqual(labels["C4"], 0)
        self.assertEqual(labels["C5"], 1)
        self.assertEqual(labels["C6"], 1)


if __name__ == "__main__":
    unittest.main()
