
# ============================================================
# OLIST LOW CUSTOMER SATISFACTION PREDICTION API
# ============================================================

from pathlib import Path
import json
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# 1. LOAD THE SAVED MODEL AND CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_FILE = MODEL_DIR / "olist_low_satisfaction_model.joblib"
CONFIG_FILE = MODEL_DIR / "model_config.json"

model = joblib.load(MODEL_FILE)

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    config = json.load(file)

THRESHOLD = config["classification_threshold"]


# ------------------------------------------------------------
# 2. INITIALISE FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="Olist Customer Satisfaction Prediction API",
    description="Predicts the likelihood of low customer satisfaction for an order.",
    version="1.0.0"
)


# ------------------------------------------------------------
# 3. DEFINE THE INPUT DATA STRUCTURE
# ------------------------------------------------------------

class OrderFeatures(BaseModel):

    customer_state: str

    item_spending: float = Field(ge=0)
    freight_spending: float = Field(ge=0)

    total_items: float = Field(ge=0)
    distinct_products: float = Field(ge=0)
    category_diversity: float = Field(ge=0)

    preferred_payment_type: str

    purchase_year: int
    purchase_month: int = Field(ge=1, le=12)
    purchase_day_of_week: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)


# ------------------------------------------------------------
# 4. CREATE A BASIC HEALTH-CHECK ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Olist Low Satisfaction Prediction API is running",
        "model": config["model_name"],
        "threshold": THRESHOLD
    }


# ------------------------------------------------------------
# 5. CREATE THE PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
def predict_order(features: OrderFeatures):

    input_data = features.model_dump()

    probability = float(
        model.predict_proba(
            __import__("pandas").DataFrame([input_data])
        )[0, 1]
    )

    prediction = int(probability >= THRESHOLD)

    return {
        "low_satisfaction_prediction": prediction,
        "low_satisfaction_probability": round(probability, 4),
        "classification_threshold": THRESHOLD,
        "risk_category": (
            "High risk" if prediction == 1 else "Lower risk"
        ),
        "interpretation": (
            "The order is predicted to have low customer satisfaction."
            if prediction == 1
            else "The order is not predicted to have low customer satisfaction."
        )
    }
