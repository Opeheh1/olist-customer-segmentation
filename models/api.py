
# ============================================================
# OLIST LOW CUSTOMER SATISFACTION PREDICTION API
# Final integrated Random Forest model + prediction monitoring
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import logging

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------
# 1. PROJECT PATHS AND LOGGING CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "baseline_rf_integrated_model.joblib"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = RESULTS_DIR / "api_monitoring.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 2. LOAD THE FINAL MODEL ARTIFACT
# ------------------------------------------------------------

MODEL_ARTIFACT = joblib.load(MODEL_PATH)

MODEL = MODEL_ARTIFACT["model"]
PREPROCESSOR = MODEL_ARTIFACT["preprocessor"]
THRESHOLD = float(MODEL_ARTIFACT["threshold"])
MODEL_NAME = MODEL_ARTIFACT["model_name"]
TARGET = MODEL_ARTIFACT["target"]

FEATURES = list(PREPROCESSOR.feature_names_in_)

POSITIVE_CLASS_INDEX = list(MODEL.classes_).index(1)


# ------------------------------------------------------------
# 3. INITIALISE THE MONITORING DATABASE
# ------------------------------------------------------------

def initialise_monitoring_database():

    with sqlite3.connect(DATABASE_PATH, timeout=10) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                prediction_timestamp TEXT NOT NULL,

                model_name TEXT NOT NULL,

                predicted_probability REAL NOT NULL,

                predicted_class INTEGER NOT NULL,

                classification_threshold REAL NOT NULL

            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_timestamp
            ON prediction_logs(prediction_timestamp)
        """)

        connection.commit()


# Create the database and table when the API starts.
initialise_monitoring_database()


# ------------------------------------------------------------
# 4. SAVE SUCCESSFUL PREDICTIONS
# ------------------------------------------------------------

def log_prediction(probability, prediction):

    timestamp = datetime.now(timezone.utc).isoformat()

    try:

        with sqlite3.connect(
            DATABASE_PATH,
            timeout=10
        ) as connection:

            connection.execute("""
                INSERT INTO prediction_logs (
                    prediction_timestamp,
                    model_name,
                    predicted_probability,
                    predicted_class,
                    classification_threshold
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                MODEL_NAME,
                probability,
                prediction,
                THRESHOLD
            ))

            connection.commit()

        logger.info("Prediction successfully recorded.")

    except sqlite3.Error:

        # A monitoring failure should not prevent a valid
        # prediction from being returned to the API user.
        logger.exception("Could not save prediction to monitoring database.")


# ------------------------------------------------------------
# 5. INITIALISE FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="Olist Low Customer Satisfaction Prediction API",
    description=(
        "Predicts the probability that an order will receive "
        "a low customer satisfaction score."
    ),
    version="2.1.0"
)


# ------------------------------------------------------------
# 6. DEFINE THE MODEL INPUT FEATURES
# ------------------------------------------------------------

class OrderFeatures(BaseModel):

    model_config = ConfigDict(extra="forbid")

    # Delivery features
    delivery_days: float | None = None
    delivery_delay_days: float | None = None
    late_delivery: float | None = None

    # Order and product features
    item_count: float | None = None
    unique_products: float | None = None
    unique_sellers: float | None = None
    total_product_price: float | None = None
    total_freight_value: float | None = None
    average_item_price: float | None = None
    average_freight_value: float | None = None

    # Payment features
    payment_count: float | None = None
    total_payment_value: float | None = None
    maximum_installments: float | None = None
    average_installments: float | None = None
    payment_method_count: float | None = None

    # Customer location
    customer_state: str | None = None

    # Product characteristics
    product_category_count: float | None = None
    average_product_weight_g: float | None = None
    average_product_length_cm: float | None = None
    average_product_height_cm: float | None = None
    average_product_width_cm: float | None = None

    # Seller and delivery distance
    seller_state_count: float | None = None
    average_seller_distance_km: float | None = None
    maximum_seller_distance_km: float | None = None
    sellers_with_coordinates: float | None = None

    # Purchase date and time
    purchase_year: float | None = None
    purchase_month: float | None = None
    purchase_day: float | None = None
    purchase_day_of_week: float | None = None
    purchase_hour: float | None = None
    purchase_is_weekend: float | None = None

    # Payment method indicators
    used_credit_card: float | None = None
    used_boleto: float | None = None
    used_voucher: float | None = None
    used_debit_card: float | None = None

    # Historical seller and product performance
    average_seller_previous_completed_orders: float | None = None
    average_seller_previous_delay: float | None = None
    average_product_previous_completed_orders: float | None = None
    average_product_previous_delay: float | None = None


# ------------------------------------------------------------
# 7. ROOT ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Olist prediction API is running",
        "model": MODEL_NAME,
        "target": TARGET,
        "classification_threshold": THRESHOLD,
        "prediction_unit": "Individual order"
    }


# ------------------------------------------------------------
# 8. HEALTH-CHECK ENDPOINT
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None
    }


# ------------------------------------------------------------
# 9. MODEL INFORMATION ENDPOINT
# ------------------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "classification_threshold": THRESHOLD,
        "positive_class": 1,
        "number_of_input_features": len(FEATURES),
        "input_features": FEATURES
    }


# ------------------------------------------------------------
# 10. PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
def predict_order(features: OrderFeatures):

    try:

        input_data = features.model_dump()

        input_frame = pd.DataFrame([input_data])

        input_frame = input_frame.reindex(columns=FEATURES)

        transformed_data = PREPROCESSOR.transform(input_frame)

        probability = float(
            MODEL.predict_proba(transformed_data)[
                0, POSITIVE_CLASS_INDEX
            ]
        )

        prediction = int(probability >= THRESHOLD)

        # Record the prediction for monitoring.
        log_prediction(probability, prediction)

        return {
            "model": MODEL_NAME,
            "prediction_target": TARGET,
            "low_satisfaction_prediction": prediction,
            "low_satisfaction_probability": round(probability, 4),
            "classification_threshold": THRESHOLD,
            "risk_category": (
                "Higher predicted risk"
                if prediction == 1
                else "Lower predicted risk"
            ),
            "interpretation": (
                "The model flags this order as having a higher "
                "predicted likelihood of low customer satisfaction."
                if prediction == 1
                else "The model does not flag this order as likely "
                "to have low customer satisfaction."
            ),
            "disclaimer": (
                "This is a model prediction, not a confirmed "
                "customer outcome."
            )
        }

    except Exception as error:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
