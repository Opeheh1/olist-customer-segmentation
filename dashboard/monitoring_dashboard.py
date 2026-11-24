
# ============================================================
# OLIST MODEL MONITORING DASHBOARD
# Monitors API prediction activity using Neon PostgreSQL
# ============================================================

import os
from contextlib import closing

import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Olist Model Monitoring",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------------------
# 2. DATABASE CONNECTION
# ------------------------------------------------------------

def get_database_url():

    # First, try Streamlit Cloud secrets.
    try:
        database_url = st.secrets["DATABASE_URL"]

        if database_url:
            return database_url

    except (KeyError, FileNotFoundError):
        pass

    # If secrets are unavailable, try the environment variable.
    return os.getenv("DATABASE_URL")


# ------------------------------------------------------------
# 3. DASHBOARD HEADER
# ------------------------------------------------------------

st.title("Olist Model Monitoring Dashboard")

st.markdown(
    """
    This dashboard monitors prediction activity from the
    Olist Low Customer Satisfaction Prediction API.

    It tracks prediction volumes, predicted risk categories,
    probability distributions and model configuration.
    """
)

st.divider()


# ------------------------------------------------------------
# 4. LOAD PREDICTION RECORDS FROM NEON
# ------------------------------------------------------------

@st.cache_data(ttl=10)
def load_prediction_data():

    database_url = get_database_url()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Please configure it in Streamlit secrets."
        )

    query = """
        SELECT
            id,
            prediction_timestamp,
            model_name,
            predicted_probability,
            predicted_class,
            classification_threshold

        FROM public.prediction_logs

        ORDER BY prediction_timestamp DESC
    """

    with closing(
        psycopg2.connect(
            database_url,
            connect_timeout=10
        )
    ) as connection:

        data = pd.read_sql_query(
            query,
            connection
        )

    if not data.empty:

        data["prediction_timestamp"] = pd.to_datetime(
            data["prediction_timestamp"],
            utc=True,
            errors="coerce"
        )

        numeric_columns = [
            "predicted_class",
            "predicted_probability",
            "classification_threshold"
        ]

        for column in numeric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


# ------------------------------------------------------------
# 5. LOAD DATA
# ------------------------------------------------------------

try:

    data = load_prediction_data()

except Exception as error:

    st.error(
        f"Could not load monitoring records from Neon: {error}"
    )

    st.info(
        "Check that DATABASE_URL is configured correctly "
        "in your Streamlit application secrets."
    )

    st.stop()


# ------------------------------------------------------------
# 6. CHECK DATA AVAILABILITY
# ------------------------------------------------------------

if data.empty:

    st.warning(
        "No prediction records are currently available. "
        "Submit a prediction through the API to populate "
        "this dashboard."
    )

    st.stop()


# ------------------------------------------------------------
# 7. PREPARE MONITORING VARIABLES
# ------------------------------------------------------------

total_predictions = len(data)

higher_risk = int(
    (data["predicted_class"] == 1).sum()
)

lower_risk = int(
    (data["predicted_class"] == 0).sum()
)

average_probability = data[
    "predicted_probability"
].mean()

classification_threshold = data[
    "classification_threshold"
].iloc[0]

latest_prediction = data[
    "prediction_timestamp"
].max()

model_name = data["model_name"].iloc[0]


# ------------------------------------------------------------
# 8. MODEL INFORMATION
# ------------------------------------------------------------

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    model_name
)

col2.metric(
    "Classification Threshold",
    f"{classification_threshold:.2f}"
)

col3.metric(
    "Prediction Target",
    "Low Satisfaction"
)

st.divider()


# ------------------------------------------------------------
# 9. KEY MONITORING INDICATORS
# ------------------------------------------------------------

st.subheader("Prediction Activity")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    f"{total_predictions:,}"
)

col2.metric(
    "Higher-Risk Predictions",
    f"{higher_risk:,}"
)

col3.metric(
    "Lower-Risk Predictions",
    f"{lower_risk:,}"
)

col4.metric(
    "Average Predicted Probability",
    f"{average_probability:.2%}"
)

st.caption(
    f"Latest recorded prediction: {latest_prediction}"
)

st.divider()


# ------------------------------------------------------------
# 10. PREDICTION RISK DISTRIBUTION
# ------------------------------------------------------------

st.subheader("Predicted Risk Distribution")

risk_data = pd.DataFrame({
    "Risk Category": [
        "Higher predicted risk",
        "Lower predicted risk"
    ],
    "Number of Predictions": [
        higher_risk,
        lower_risk
    ]
})

risk_figure = px.pie(
    risk_data,
    names="Risk Category",
    values="Number of Predictions",
    hole=0.45,
    title="Distribution of Predicted Risk Categories"
)

st.plotly_chart(
    risk_figure,
    use_container_width=True
)


# ------------------------------------------------------------
# 11. PREDICTED PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

st.subheader("Predicted Probability Distribution")

probability_figure = px.histogram(
    data,
    x="predicted_probability",
    nbins=20,
    title="Distribution of Low Satisfaction Probabilities",
    labels={
        "predicted_probability": "Predicted Probability",
        "count": "Number of Predictions"
    }
)

probability_figure.add_vline(
    x=classification_threshold,
    line_dash="dash",
    annotation_text="Classification Threshold"
)

st.plotly_chart(
    probability_figure,
    use_container_width=True
)


# ------------------------------------------------------------
# 12. PREDICTION ACTIVITY OVER TIME
# ------------------------------------------------------------

st.subheader("Prediction Activity Over Time")

activity_data = data.copy()

activity_data["prediction_date"] = (
    activity_data["prediction_timestamp"].dt.date
)

daily_activity = (
    activity_data
    .groupby("prediction_date")
    .size()
    .reset_index(name="Number of Predictions")
)

activity_figure = px.line(
    daily_activity,
    x="prediction_date",
    y="Number of Predictions",
    markers=True,
    title="Daily API Prediction Volume"
)

st.plotly_chart(
    activity_figure,
    use_container_width=True
)


# ------------------------------------------------------------
# 13. PREDICTION RECORDS
# ------------------------------------------------------------

st.subheader("Recent Prediction Records")

display_data = data.copy()

display_data["risk_category"] = (
    display_data["predicted_class"]
    .map({
        1: "Higher predicted risk",
        0: "Lower predicted risk"
    })
)

display_data = display_data.rename(
    columns={
        "id": "Prediction ID",
        "prediction_timestamp": "Timestamp",
        "model_name": "Model",
        "predicted_probability": "Predicted Probability",
        "predicted_class": "Predicted Class",
        "classification_threshold": "Threshold",
        "risk_category": "Risk Category"
    }
)

st.dataframe(
    display_data[
        [
            "Prediction ID",
            "Timestamp",
            "Model",
            "Predicted Probability",
            "Predicted Class",
            "Threshold",
            "Risk Category"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# 14. MONITORING LIMITATIONS
# ------------------------------------------------------------

st.divider()

st.subheader("Monitoring Scope and Limitations")

st.info(
    """
    **Current monitoring scope**

    - Tracks API prediction volume.
    - Monitors predicted risk categories.
    - Displays predicted probability distributions.
    - Records prediction timestamps and model configuration.
    - Retrieves monitoring records from Neon PostgreSQL.

    **Current limitations**

    - Actual customer review outcomes are not linked to API predictions.
    - Accuracy, precision, recall and other performance metrics
      cannot be recalculated from prediction logs alone.
    - Data drift and fairness monitoring are not yet automated.

    These areas require additional data and monitoring procedures.
    """
)


# ------------------------------------------------------------
# 15. REFRESH CONTROL
# ------------------------------------------------------------

if st.button("Refresh Monitoring Data"):

    st.cache_data.clear()

    st.rerun()
