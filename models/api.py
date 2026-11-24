
# ============================================================
# OLIST LOW CUSTOMER SATISFACTION PREDICTION API
# Random Forest - 40 Trees + Neon PostgreSQL Monitoring
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import os
import logging

import joblib
import pandas as pd
import psycopg2

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------
# 1. PROJECT PATHS AND LOGGING CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "baseline_rf_40trees_api.joblib"
)

DATABASE_URL = os.getenv("DATABASE_URL")

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
# 3. INITIALISE THE NEON POSTGRESQL DATABASE
# ------------------------------------------------------------

DATABASE_READY = False


def get_database_connection():

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing.")

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=10
    )


def initialise_monitoring_database():

    global DATABASE_READY

    try:

        with get_database_connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_logs (

                        id BIGSERIAL PRIMARY KEY,

                        prediction_timestamp TIMESTAMPTZ NOT NULL,

                        model_name VARCHAR(150) NOT NULL,

                        predicted_probability DOUBLE PRECISION NOT NULL,

                        predicted_class INTEGER NOT NULL,

                        classification_threshold DOUBLE PRECISION NOT NULL

                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prediction_timestamp
                    ON prediction_logs(prediction_timestamp)
                """)

        DATABASE_READY = True

        logger.info(
            "Neon PostgreSQL monitoring database initialised successfully."
        )

    except Exception:

        DATABASE_READY = False

        logger.exception(
            "Could not initialise Neon monitoring database."
        )


# Initialise the database when the API starts.
initialise_monitoring_database()


# ------------------------------------------------------------
# 4. SAVE SUCCESSFUL PREDICTIONS
# ------------------------------------------------------------

def log_prediction(probability, prediction):

    global DATABASE_READY

    timestamp = datetime.now(timezone.utc)

    try:

        with get_database_connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute("""
                    INSERT INTO prediction_logs (
                        prediction_timestamp,
                        model_name,
                        predicted_probability,
                        predicted_class,
                        classification_threshold
                    )
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    MODEL_NAME,
                    probability,
                    prediction,
                    THRESHOLD
                ))

        DATABASE_READY = True

        logger.info("Prediction successfully recorded in Neon.")

    except Exception:

        DATABASE_READY = False

        # A monitoring failure should not prevent a valid
        # prediction from being returned to the API user.
        logger.exception(
            "Could not save prediction to Neon monitoring database."
        )


# ------------------------------------------------------------
# 5. INITIALISE FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="Olist Low Customer Satisfaction Prediction API",
    description=(
        "Predicts the probability that an order will receive "
        "a low customer satisfaction score."
    ),
    version="2.2.0"
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
        "preprocessor_loaded": PREPROCESSOR is not None,
        "monitoring_database_ready": DATABASE_READY
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

        # Record the prediction in Neon PostgreSQL.
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
