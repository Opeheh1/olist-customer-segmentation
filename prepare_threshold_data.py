
from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "olist_low_satisfaction_model.joblib"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delivered_order_level.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_threshold_data.csv"
)

# Module 4 classification threshold
THRESHOLD = 0.5453

# Module 4 chronological validation boundaries
VALIDATION_START = pd.Timestamp("2018-03-20 13:03:41")
VALIDATION_END = pd.Timestamp("2018-05-27 12:07:47")

# --------------------------------------------------
# 2. Load the saved model and dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. Recreate Module 4 time-based features
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
# 4. Define the 11 model features
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
# 5. Reconstruct the validation dataset
# --------------------------------------------------

validation_df = df[
    (df["order_purchase_timestamp"] > VALIDATION_START)
    & (df["order_purchase_timestamp"] <= VALIDATION_END)
].copy()

validation_df = validation_df.dropna(
    subset=features + ["average_review_score"]
).copy()

# Actual outcome: review score <= 2
validation_df["actual"] = (
    validation_df["average_review_score"] <= 2
).astype(int)

X_validation = validation_df[features]

# --------------------------------------------------
# 6. Generate predicted probabilities
# --------------------------------------------------

validation_df["predicted_probability"] = (
    model.predict_proba(X_validation)[:, 1]
)

validation_df["predicted"] = (
    validation_df["predicted_probability"] >= THRESHOLD
).astype(int)

# --------------------------------------------------
# 7. Save the validation results
# --------------------------------------------------

output_columns = [
    "order_id",
    "actual",
    "predicted_probability",
    "predicted"
]

threshold_data = validation_df[output_columns].copy()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

threshold_data.to_csv(
    OUTPUT_PATH,
    index=False
)

# --------------------------------------------------
# 8. Display verification results
# --------------------------------------------------

print("VALIDATION THRESHOLD DATA GENERATED")
print("=" * 50)

print("Number of validation orders:", len(threshold_data))
print("Actual low-satisfaction orders:", threshold_data["actual"].sum())
print("Predicted low-satisfaction orders:", threshold_data["predicted"].sum())

print("\nClassification threshold:", THRESHOLD)

print("\nOutput file:")
print(OUTPUT_PATH)

print("\nFirst five records:")
print(threshold_data.head())

print("\nFile saved successfully.")
