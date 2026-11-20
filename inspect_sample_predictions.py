
from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# 1. Project paths and model settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "olist_low_satisfaction_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivered_order_level.csv"

THRESHOLD = 0.5453

# Module 4 chronological validation boundaries
VALIDATION_START = pd.Timestamp("2018-03-20 13:03:41")
VALIDATION_END = pd.Timestamp("2018-05-27 12:07:47")

# --------------------------------------------------
# 2. Load model and dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. Recreate Module 4 time features
# --------------------------------------------------

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"],
    errors="coerce"
)

df["purchase_year"] = df["order_purchase_timestamp"].dt.year
df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_day_of_week"] = (
    df["order_purchase_timestamp"].dt.dayofweek
)
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

# --------------------------------------------------
# 4. Define the original model features
# --------------------------------------------------

features = [
    "customer_state",
    "item_spending",
    "freight_spending",
    "total_items",
    "distinct_products",
    "category_diversity",
    "preferred_payment_type",
    "purchase_year",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour"
]

# --------------------------------------------------
# 5. Reconstruct the validation period
# --------------------------------------------------

validation_df = df[
    (df["order_purchase_timestamp"] > VALIDATION_START)
    & (df["order_purchase_timestamp"] <= VALIDATION_END)
].copy()

# Keep records with available target and model features
validation_df = validation_df.dropna(
    subset=features + ["average_review_score"]
).copy()

# Actual low satisfaction = average review score <= 2
validation_df["actual_low_satisfaction"] = (
    validation_df["average_review_score"] <= 2
).astype(int)

X_validation = validation_df[features]

y_actual = validation_df["actual_low_satisfaction"]

print("Validation records:", len(validation_df))

# --------------------------------------------------
# 6. Generate model probabilities and predictions
# --------------------------------------------------

probabilities = model.predict_proba(X_validation)[:, 1]

validation_df["predicted_probability"] = probabilities

validation_df["predicted_low_satisfaction"] = (
    probabilities >= THRESHOLD
).astype(int)

# --------------------------------------------------
# 7. Identify the three examples
# --------------------------------------------------

true_positives = validation_df[
    (validation_df["actual_low_satisfaction"] == 1)
    & (validation_df["predicted_low_satisfaction"] == 1)
]

false_negatives = validation_df[
    (validation_df["actual_low_satisfaction"] == 1)
    & (validation_df["predicted_low_satisfaction"] == 0)
]

true_negatives = validation_df[
    (validation_df["actual_low_satisfaction"] == 0)
    & (validation_df["predicted_low_satisfaction"] == 0)
]

# Select examples close to the classification threshold
examples = []

if not true_positives.empty:
    examples.append(
        ("True Positive", true_positives.sort_values(
            "predicted_probability"
        ).iloc[0])
    )

if not false_negatives.empty:
    examples.append(
        ("False Negative", false_negatives.sort_values(
            "predicted_probability",
            ascending=False
        ).iloc[0])
    )

if not true_negatives.empty:
    examples.append(
        ("True Negative", true_negatives.sort_values(
            "predicted_probability",
            ascending=False
        ).iloc[0])
    )

# --------------------------------------------------
# 8. Display selected examples
# --------------------------------------------------

display_columns = [
    "order_id",
    "customer_state",
    "average_review_score",
    "actual_low_satisfaction",
    "predicted_probability",
    "predicted_low_satisfaction",
    "item_spending",
    "freight_spending",
    "total_items",
    "preferred_payment_type"
]

print("\nSELECTED SAMPLE PREDICTIONS")
print("=" * 70)

for example_type, example in examples:

    print(f"\n{example_type}")
    print("-" * 50)

    print(
        example[display_columns].to_string()
    )

print("\nExample extraction completed.")
