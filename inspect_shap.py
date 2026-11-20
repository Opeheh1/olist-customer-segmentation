
from pathlib import Path

import joblib
import pandas as pd
import shap

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "olist_low_satisfaction_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivered_order_level.csv"

# --------------------------------------------------
# 2. Load model and dataset
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
df["purchase_day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

# --------------------------------------------------
# 4. Select the original 11 model features
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

# Keep only complete observations
X = df[features].dropna()

# Sample up to 300 actual orders
X = X.sample(
    n=min(300, len(X)),
    random_state=42
)

print("Original sample shape:", X.shape)

# --------------------------------------------------
# 5. Access the trained pipeline components
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Apply the existing preprocessing
X_transformed = preprocessor.transform(X)

# Convert sparse output to dense if necessary
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

transformed_features = preprocessor.get_feature_names_out()

print("Transformed sample shape:", X_transformed.shape)
print("Number of transformed features:", len(transformed_features))

# --------------------------------------------------
# 6. Calculate SHAP values
# --------------------------------------------------

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_transformed)

# Extract SHAP values for low satisfaction (class 1)
if isinstance(shap_values, list):
    positive_class_shap = shap_values[1]
elif len(shap_values.shape) == 3:
    positive_class_shap = shap_values[:, :, 1]
else:
    positive_class_shap = shap_values

print("\nSHAP OUTPUT SHAPE:", positive_class_shap.shape)

# --------------------------------------------------
# 7. Calculate global feature importance
# --------------------------------------------------

importance = pd.Series(
    abs(positive_class_shap).mean(axis=0),
    index=transformed_features
).sort_values(ascending=False)

print("\nTOP 10 FEATURES:")
print(importance.head(10).to_string())

print("\nSHAP inspection completed successfully.")
