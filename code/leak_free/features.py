from datetime import timedelta

import numpy as np
import polars as pl

from .config import (
    CHURN_RECENCY_DAYS,
    CUTOFF_DATE,
    FEATURE_PERIOD_START,
    LABEL_PERIOD_END,
)

FEATURE_NAMES = [
    "recency", "frequency", "monetary", "customer_lifetime_days",
    "avg_days_between_orders", "n_categories", "avg_rating",
    "avg_delivery_days", "avg_session", "avg_pages", "discount_rate",
    "orders_last_30days", "electronics_ratio", "fashion_ratio",
    "home_ratio", "beauty_ratio", "mobile_ratio", "credit_card_ratio",
    "is_returning_ratio", "age", "rfm_score", "avg_order_value",
    "purchase_intensity", "monetary_trend", "recency_ratio",
    "purchase_regularity", "recent_3m_trend", "category_concentration",
    "price_sensitivity", "engagement_score",
]


def prepare_customer_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("Date").str.to_date("%Y-%m-%d"))
    feature = df.filter(
        (pl.col("Date") >= FEATURE_PERIOD_START) & (pl.col("Date") < CUTOFF_DATE)
    )
    label = df.filter(
        (pl.col("Date") >= CUTOFF_DATE) & (pl.col("Date") < LABEL_PERIOD_END)
    )
    feature_customers = feature["Customer_ID"].unique().to_list()
    label_customers = set(label["Customer_ID"].unique().to_list())
    last_dates = {
        row["Customer_ID"]: row["Date"]
        for row in feature.group_by("Customer_ID")
        .agg(pl.col("Date").max().alias("Date"))
        .iter_rows(named=True)
    }
    rows = [
        _customer_row(cid, feature.filter(pl.col("Customer_ID") == cid), last_dates[cid], label_customers)
        for cid in feature_customers
    ]
    return pl.DataFrame(rows)


def _customer_row(cid, cust, last_date, label_customers):
    n_orders = cust.height
    recency = (CUTOFF_DATE - last_date).days
    frequency = n_orders
    monetary = cust["Total_Amount"].sum()
    first_date = cust["Date"].min()
    lifetime = (last_date - first_date).days + 1
    avg_gap = lifetime / max(frequency - 1, 1)
    n_categories = cust["Product_Category"].n_unique()
    avg_rating = cust["Customer_Rating"].mean()
    avg_delivery = cust["Delivery_Time_Days"].mean()
    avg_session = cust["Session_Duration_Minutes"].mean()
    avg_pages = cust["Pages_Viewed"].mean()
    discount_rate = cust["Discount_Amount"].sum() / max(monetary, 1)
    cutoff_30 = CUTOFF_DATE - timedelta(days=30)
    orders_last_30 = cust.filter(pl.col("Date") >= cutoff_30).height
    total = max(n_orders, 1)

    def ratio(name, values):
        return cust.filter(pl.col(name) == values).height / total

    electronics = ratio("Product_Category", "Electronics")
    fashion = ratio("Product_Category", "Fashion")
    home = ratio("Product_Category", "Home & Garden")
    beauty = ratio("Product_Category", "Beauty & Personal Care")
    mobile = ratio("Device_Type", "Mobile")
    credit = ratio("Payment_Method", "Credit Card")
    returning = ratio("Is_Returning_Customer", True)
    age = cust["Age"][0]

    r_score = 5 if recency < 30 else 4 if recency < 60 else 3 if recency < 90 else 2 if recency < 180 else 1
    f_score = 5 if frequency >= 8 else 4 if frequency >= 6 else 3 if frequency >= 4 else 2 if frequency >= 2 else 1
    m_score = 5 if monetary >= 5000 else 4 if monetary >= 3000 else 3 if monetary >= 1500 else 2 if monetary >= 500 else 1
    rfm = r_score * 100 + f_score * 10 + m_score
    aov = monetary / max(frequency, 1)
    intensity = frequency / max(lifetime / 30, 1)

    mid = first_date + timedelta(days=lifetime // 2)
    first_half = cust.filter(pl.col("Date") < mid)["Total_Amount"].sum()
    second_half = cust.filter(pl.col("Date") >= mid)["Total_Amount"].sum()
    trend = (second_half - first_half) / max(first_half, 1)
    recency_ratio = recency / max(lifetime, 1)

    if frequency > 1:
        dates = sorted(cust["Date"].to_list())
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        regularity = np.std(gaps) / (np.mean(gaps) + 1)
    else:
        regularity = 0.0

    cutoff_90 = CUTOFF_DATE - timedelta(days=90)
    recent_3m = cust.filter(pl.col("Date") >= cutoff_90)["Total_Amount"].sum()
    monthly = monetary / max(lifetime / 30, 1)
    recent_trend = (recent_3m / 3) / max(monthly, 1) - 1

    counts = cust.group_by("Product_Category").agg(pl.len().alias("count"))
    concentration = sum((c / total) ** 2 for c in counts["count"].to_list())

    if frequency > 1:
        fh = cust.filter(pl.col("Date") < mid)
        sh = cust.filter(pl.col("Date") >= mid)
        fh_rate = fh["Discount_Amount"].sum() / max(fh["Total_Amount"].sum(), 1)
        sh_rate = sh["Discount_Amount"].sum() / max(sh["Total_Amount"].sum(), 1)
        price_sensitivity = sh_rate - fh_rate
    else:
        price_sensitivity = 0.0

    engagement = (
        (avg_session / 30) * 0.3
        + (avg_pages / 10) * 0.3
        + (avg_rating / 5) * 0.4
    )
    is_churn = 1 if (cid not in label_customers and recency > CHURN_RECENCY_DAYS) else 0

    values = [
        recency, frequency, monetary, lifetime, avg_gap, n_categories,
        avg_rating, avg_delivery, avg_session, avg_pages, discount_rate,
        orders_last_30, electronics, fashion, home, beauty, mobile, credit,
        returning, age, rfm, aov, intensity, trend, recency_ratio,
        regularity, recent_trend, concentration, price_sensitivity, engagement,
    ]
    row = dict(zip(FEATURE_NAMES, values))
    row["Customer_ID"] = cid
    row["is_churn"] = is_churn
    return row
