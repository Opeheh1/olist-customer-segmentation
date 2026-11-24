from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# ============================================================

st.set_page_config(
    page_title="Olist Customer Experience Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# app.py is inside the dashboard folder; the project root is one level above.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_rf_integrated_model.joblib"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_integrated_data_model_ready.csv"
)


# ============================================================
# 2. LOAD THE CORRECTED MODEL AND INTEGRATED DATASET
# ============================================================

@st.cache_resource
def load_model_artifact():
    """Load the integrated model and its fitted preprocessing pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    """Load the corrected, integrated order-level modelling dataset."""
    return pd.read_csv(DATA_PATH, low_memory=False)


try:
    MODEL_ARTIFACT = load_model_artifact()
    DATASET = load_dataset()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Check that the corrected model and integrated dataset are saved "
        "in the models and data/processed folders."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the project files: {error}")
    st.stop()

if not isinstance(MODEL_ARTIFACT, dict):
    st.error("The model artifact must contain the model and preprocessor dictionary.")
    st.stop()

MODEL = MODEL_ARTIFACT.get("model")
PREPROCESSOR = MODEL_ARTIFACT.get("preprocessor")
TARGET = MODEL_ARTIFACT.get("target", "low_satisfaction")
THRESHOLD = float(MODEL_ARTIFACT.get("threshold", 0.35))
POSITIVE_CLASS = 1
MODEL_NAME = MODEL_ARTIFACT.get("model_name", "Baseline Random Forest")

if MODEL is None or PREPROCESSOR is None:
    st.error(
        "The saved artifact does not contain both 'model' and 'preprocessor'. "
        "Use models/baseline_rf_integrated_model.joblib."
    )
    st.stop()

if not hasattr(PREPROCESSOR, "feature_names_in_"):
    st.error("The saved preprocessor does not expose its original input feature names.")
    st.stop()

FEATURES = list(PREPROCESSOR.feature_names_in_)
TRANSFORMED_FEATURES = list(
    MODEL_ARTIFACT.get("feature_names", PREPROCESSOR.get_feature_names_out())
)

missing_columns = [
    column for column in FEATURES + [TARGET, "order_purchase_timestamp"]
    if column not in DATASET.columns
]

if missing_columns:
    st.error(
        "The integrated dataset is missing columns required by the saved model: "
        + ", ".join(missing_columns)
    )
    st.stop()

if POSITIVE_CLASS not in list(MODEL.classes_):
    st.error("The loaded model does not contain the expected positive class (1).")
    st.stop()

POSITIVE_INDEX = list(MODEL.classes_).index(POSITIVE_CLASS)


# ============================================================
# 3. CURRENT MODEL RESULTS
# ============================================================

# Final chronological test results for the selected integrated Random Forest
# at the validation-selected classification threshold of 0.35.
TEST_METRICS = {
    "Accuracy": 89.68,
    "Precision": 44.45,
    "Recall": 26.59,
    "F1 Score": 33.28,
    "ROC-AUC": 67.02,
    "PR-AUC": 28.78
}

CONFUSION_MATRIX = pd.DataFrame(
    [[16696, 616], [1361, 493]],
    index=["Actual: Not low satisfaction", "Actual: Low satisfaction"],
    columns=[
        "Predicted: Not low satisfaction",
        "Predicted: Low satisfaction"
    ]
)

# This is the earlier model setup, retained only to show the overall project
# improvement. It is not a controlled, like-for-like experiment.
PREVIOUS_MODEL_METRICS = {
    "Accuracy": 81.64,
    "Precision": 18.21,
    "Recall": 25.73,
    "F1 Score": 21.33,
    "ROC-AUC": 58.17,
    "PR-AUC": 16.98
}

# The second candidate was evaluated during the final model comparison.
BALANCED_MODEL_TEST_METRICS = {
    "Accuracy": 89.15,
    "Precision": 40.95,
    "Recall": 27.45,
    "F1 Score": 32.87,
    "ROC-AUC": 67.56,
    "PR-AUC": 28.57
}


# ============================================================
# 4. DATA SPLITS AND PREDICTION HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def get_chronological_splits():
    """Recreate the 64% / 16% / 20% chronological project split."""
    frame = load_dataset().copy()
    frame["order_purchase_timestamp"] = pd.to_datetime(
        frame["order_purchase_timestamp"], errors="coerce"
    )
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(
        subset=["order_purchase_timestamp", TARGET]
    ).copy()
    frame[TARGET] = frame[TARGET].astype(int)
    frame = frame.sort_values(
        "order_purchase_timestamp", kind="mergesort"
    ).reset_index(drop=True)

    x = frame[FEATURES].copy()
    y = frame[TARGET].copy()

    train_end = int(len(frame) * 0.64)
    validation_end = train_end + int(len(frame) * 0.16)

    x_train = x.iloc[:train_end].reset_index(drop=True)
    x_validation = x.iloc[train_end:validation_end].reset_index(drop=True)
    x_test = x.iloc[validation_end:].reset_index(drop=True)

    y_train = y.iloc[:train_end].reset_index(drop=True)
    y_validation = y.iloc[train_end:validation_end].reset_index(drop=True)
    y_test = y.iloc[validation_end:].reset_index(drop=True)

    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test
    )


def predict_positive_probability(input_frame):
    """Apply the saved preprocessor and return positive-class probabilities."""
    if isinstance(input_frame, pd.Series):
        input_frame = input_frame.to_frame().T
    elif isinstance(input_frame, dict):
        input_frame = pd.DataFrame([input_frame])

    input_frame = input_frame.reindex(columns=FEATURES)
    transformed = PREPROCESSOR.transform(input_frame)
    probabilities = MODEL.predict_proba(transformed)
    return probabilities[:, POSITIVE_INDEX]


def get_prediction(values):
    """Generate a class prediction and probability for one order profile."""
    if isinstance(values, pd.Series):
        input_frame = values.to_frame().T
    elif isinstance(values, dict):
        input_frame = pd.DataFrame([values])
    else:
        input_frame = values.copy()

    probability = float(predict_positive_probability(input_frame)[0])
    prediction = int(probability >= THRESHOLD)
    return prediction, probability


@st.cache_data(show_spinner=False)
def get_validation_predictions():
    """Score the chronological validation set for sample and threshold pages."""
    _, x_validation, _, _, y_validation, _ = get_chronological_splits()
    probabilities = predict_positive_probability(x_validation)

    return pd.DataFrame({
        "actual": y_validation.astype(int),
        "predicted_probability": probabilities
    })


@st.cache_data(show_spinner=False)
def get_test_prediction_data():
    """Score the chronological test set for the state-level fairness review."""
    _, _, x_test, _, _, y_test = get_chronological_splits()
    probabilities = predict_positive_probability(x_test)
    predictions = (probabilities >= THRESHOLD).astype(int)

    state_values = (
        x_test["customer_state"].fillna("Unknown").astype(str)
        if "customer_state" in x_test.columns
        else pd.Series(["Unknown"] * len(x_test))
    )

    return pd.DataFrame({
        "customer_state": state_values.reset_index(drop=True),
        "actual": y_test.astype(int),
        "predicted_probability": probabilities,
        "predicted": predictions
    })


def calculate_classification_metrics(actual, predicted):
    """Calculate threshold-dependent classification measures."""
    actual = np.asarray(actual).astype(int)
    predicted = np.asarray(predicted).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    accuracy = (tp + tn) / len(actual) if len(actual) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    }


def display_prediction(prediction, probability):
    """Display one order-level prediction in plain language."""
    st.metric(
        "Predicted probability of low satisfaction",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.warning(
            "The model flags this order as having a higher predicted "
            "likelihood of low customer satisfaction."
        )
    else:
        st.success(
            "The model does not flag this order as likely to receive "
            "a low satisfaction score at the selected threshold."
        )

    st.caption(
        f"Classification threshold: {THRESHOLD:.2f}. "
        "This is a model prediction, not a confirmed customer outcome."
    )


def add_feature_input(feature, current_value, key):
    """Create a suitable Streamlit input for a selected raw model feature."""
    series = DATASET[feature]

    if feature == "customer_state" or not pd.api.types.is_numeric_dtype(series):
        options = sorted(series.dropna().astype(str).unique().tolist())
        if not options:
            return current_value
        default_value = (
            str(current_value)
            if pd.notna(current_value) and str(current_value) in options
            else options[0]
        )
        return st.selectbox(
            feature.replace("_", " ").title(),
            options,
            index=options.index(default_value),
            key=key
        )

    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return current_value

    unique_values = sorted(clean_series.unique().tolist())
    if len(unique_values) <= 2:
        options = unique_values
        default_value = (
            current_value if pd.notna(current_value) and current_value in options
            else options[0]
        )
        return st.selectbox(
            feature.replace("_", " ").title(),
            options,
            index=options.index(default_value),
            key=key
        )

    minimum = float(clean_series.min())
    maximum = float(clean_series.max())
    default_value = (
        float(current_value)
        if pd.notna(current_value)
        else float(clean_series.median())
    )
    default_value = min(max(default_value, minimum), maximum)

    if pd.api.types.is_integer_dtype(series):
        return st.number_input(
            feature.replace("_", " ").title(),
            min_value=int(np.floor(minimum)),
            max_value=int(np.ceil(maximum)),
            value=int(round(default_value)),
            step=1,
            key=key
        )

    return st.number_input(
        feature.replace("_", " ").title(),
        min_value=minimum,
        max_value=maximum,
        value=default_value,
        step=0.1,
        format="%.3f",
        key=key
    )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Olist Analytics")
st.sidebar.caption("Customer Experience Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Interactive Predictions",
        "Sample Predictions",
        "Threshold Analysis",
        "What-If Analysis",
        "Model Explainability",
        "Ethical Compliance",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Active model")
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write(f"**Algorithm:** Random Forest")
st.sidebar.write(f"**Threshold:** {THRESHOLD:.2f}")
st.sidebar.write("**Prediction unit:** Order")
st.sidebar.caption(
    "Uses the corrected integrated dataset and the fitted preprocessing "
    "pipeline saved with the model."
)


# ============================================================
# 6. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("Olist Customer Experience Analytics")

    st.markdown(
        """
        **Identifying order characteristics associated with low customer satisfaction.**

        This dashboard presents an integrated Random Forest model that estimates
        whether an order is likely to receive a review score of 2 or below. It
        supports the examination of order-level customer experience risks,
        model behaviour and geographic performance differences.
        """
    )

    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        get_chronological_splits()
    )

    st.divider()
    st.subheader("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Integrated Orders", f"{len(DATASET):,}")
    col2.metric("Raw Model Features", f"{len(FEATURES)}")
    col3.metric("Test F1 Score", "33.28%")
    col4.metric("Test Recall", "26.59%")

    st.caption(
        "The modelling dataset contains reviewed orders retained after the "
        "project's integration and data-quality checks."
    )

    st.divider()
    st.subheader("The Business Problem")

    st.write(
        "Olist connects customers with sellers through an online marketplace. "
        "Poor delivery experiences and other order characteristics may be "
        "associated with low customer reviews. The project examines whether "
        "integrated order, product, seller and delivery information can help "
        "identify orders associated with low satisfaction."
    )

    st.subheader("What the Model Examines")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Customer experience**
            - Which orders have a higher predicted likelihood of low satisfaction?
            - Which delivery and order characteristics contribute most to predictions?
            - How does the predicted risk change under different scenarios?
            """
        )
    with col2:
        st.markdown(
            """
            **Model oversight**
            - How well does the model identify low-satisfaction orders?
            - How does the classification threshold affect missed cases and alerts?
            - Does performance vary across customer states?
            """
        )

    st.divider()
    st.subheader("Main Finding")

    st.write(
        "Delivery performance is the strongest contributor to the model's "
        "predictions. Delivery delay, late-delivery status and delivery duration "
        "rank among the leading SHAP features. This suggests that delivery "
        "experience deserves attention when investigating low customer ratings."
    )

    st.warning(
        "Several leading predictors, including actual delivery duration and "
        "delivery delay, are only known after an order has been delivered. "
        "The current model is therefore more suitable for retrospective order "
        "review than for predicting dissatisfaction immediately after purchase."
    )

    st.divider()
    st.subheader("Current Test Performance")

    metric_columns = st.columns(3)
    for index, (metric, value) in enumerate(TEST_METRICS.items()):
        with metric_columns[index % 3]:
            st.metric(metric, f"{value:.2f}%")

    st.caption(
        "Final chronological test results at the validation-selected threshold "
        "of 0.35. The model detects some low-satisfaction orders but still misses "
        "most actual cases."
    )

    st.divider()
    st.subheader("Explore the Dashboard")

    navigation_descriptions = [
        ("Model Performance", "Review final test results and classification errors."),
        ("Interactive Predictions", "Start with a validation order and adjust selected inputs."),
        ("Sample Predictions", "Inspect validation examples of correct and incorrect predictions."),
        ("Threshold Analysis", "Explore the precision-recall trade-off on validation data."),
        ("What-If Analysis", "Change one model feature and compare predicted probabilities."),
        ("Model Explainability", "Explore global and individual SHAP contributions."),
        ("Ethical Compliance", "Review test-set performance differences across customer states."),
        ("Model Information", "Review the data, features, model selection and limitations.")
    ]

    for title, description in navigation_descriptions:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.divider()
    st.subheader("Important Limitation")

    st.warning(
        "The model achieved 33.28% test F1 and 26.59% recall. It correctly "
        "identified 493 of 1,854 actual low-satisfaction test orders and missed "
        "1,361. Predictions should support investigation only; they should not "
        "be treated as confirmed outcomes or used as the sole basis for customer decisions."
    )


# ============================================================
# 7. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "This page evaluates how accurately the selected integrated Random "
        "Forest identifies orders associated with low customer satisfaction."
    )

    st.caption(
        "All final metrics below are from the 19,166-order chronological test "
        "dataset, using the threshold selected on validation data (0.35)."
    )

    metric_columns = st.columns(3)
    for index, (metric, value) in enumerate(TEST_METRICS.items()):
        with metric_columns[index % 3]:
            st.metric(metric, f"{value:.2f}%")

    st.divider()
    st.subheader("Actual vs Predicted Results")

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=CONFUSION_MATRIX.values,
            x=["Predicted: Not Low", "Predicted: Low"],
            y=["Actual: Not Low", "Actual: Low"],
            text=CONFUSION_MATRIX.values,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )
    fig_cm.update_layout(
        xaxis_title="Model prediction",
        yaxis_title="Actual outcome",
        height=420
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        """
        - **16,696 True Negatives:** Correctly identified orders without low satisfaction.
        - **493 True Positives:** Correctly identified low-satisfaction orders.
        - **616 False Positives:** Orders incorrectly flagged as low satisfaction.
        - **1,361 False Negatives:** Actual low-satisfaction orders missed by the model.
        """
    )

    st.divider()
    st.subheader("Understanding the Metrics")

    st.markdown(
        """
        - **Accuracy (89.68%):** Percentage of all test orders classified correctly.
        - **Precision (44.45%):** Of the orders flagged, about 44 in every 100
          actually had low satisfaction.
        - **Recall (26.59%):** The model identified about one in four actual
          low-satisfaction orders.
        - **F1 Score (33.28%):** A combined measure of precision and recall.
        - **ROC-AUC (67.02%):** Indicates better-than-random, but still limited,
          separation between the two outcome classes.
        - **PR-AUC (28.78%):** Summarises precision-recall performance and is
          useful when the positive class is relatively uncommon.
        """
    )

    st.divider()
    st.subheader("Class Imbalance")

    col1, col2 = st.columns(2)
    col1.metric("Actual low-satisfaction orders", "1,854", "9.67%")
    col2.metric("Actual non-low-satisfaction orders", "17,312", "90.33%")

    st.warning(
        "A model that classified every test order as not low satisfaction would "
        "achieve 90.33% accuracy, which is higher than the current model's "
        "89.68%. The current model is more useful than that trivial rule for "
        "identifying some dissatisfied orders, but its recall remains limited. "
        "Accuracy alone is therefore not evidence of useful performance."
    )

    st.divider()
    st.subheader("Comparison of the Two Final Random Forest Candidates")

    candidate_comparison = pd.DataFrame([
        ["Baseline Random Forest (selected)", 89.68, 44.45, 26.59, 33.28, 67.02, 28.78],
        ["Balanced Random Forest", 89.15, 40.95, 27.45, 32.87, 67.56, 28.57]
    ], columns=[
        "Model", "Accuracy (%)", "Precision (%)", "Recall (%)",
        "F1 Score (%)", "ROC-AUC (%)", "PR-AUC (%)"
    ])

    st.dataframe(
        candidate_comparison.style.format({
            column: "{:.2f}" for column in candidate_comparison.columns
            if column != "Model"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.write(
        "The baseline Random Forest was retained because it achieved slightly "
        "higher test precision, F1, accuracy and PR-AUC. The balanced model had "
        "slightly higher recall and ROC-AUC. The differences are small, so the "
        "selection should not be interpreted as evidence that one model is "
        "universally superior."
    )

    st.caption(
        "Both prespecified candidates were compared on the final test set after "
        "threshold selection on validation data. Further model tuning should use "
        "new validation data rather than repeatedly consulting this test set."
    )

    st.divider()
    st.subheader("Business Interpretation")

    st.write(
        "The corrected integrated model performs better than the earlier model "
        "setup on several reported measures, particularly precision, F1, ROC-AUC "
        "and PR-AUC. However, it still misses most actual low-satisfaction orders. "
        "It can support retrospective investigation, but its current recall does "
        "not justify using it as a standalone customer alert system."
    )


# ============================================================
# 8. INTERACTIVE PREDICTIONS PAGE
# ============================================================

elif page == "Interactive Predictions":

    st.title("Interactive Order Prediction")

    st.write(
        "Select an order from the validation data, retain its existing "
        "characteristics and modify selected features to explore the model's "
        "predicted probability of low satisfaction."
    )

    st.info(
        "The form starts from a real validation observation so that all 39 "
        "model inputs are populated consistently. Only selected inputs are "
        "editable; all other features retain their original values."
    )

    _, x_validation, _, _, _, _ = get_chronological_splits()
    observation_number = st.slider(
        "Validation observation",
        min_value=1,
        max_value=len(x_validation),
        value=1,
        help="This is the row position in the validation split, not an Olist order ID."
    )
    reference_row = x_validation.iloc[observation_number - 1].copy()

    editable_features = [
        feature for feature in [
            "customer_state",
            "delivery_delay_days",
            "delivery_days",
            "late_delivery",
            "total_freight_value",
            "average_seller_previous_delay"
        ]
        if feature in FEATURES
    ]

    st.subheader("Modify Selected Order Characteristics")

    with st.form("interactive_prediction_form"):
        first_column, second_column = st.columns(2)
        edited_values = reference_row.copy()

        for index, feature in enumerate(editable_features):
            target_column = first_column if index % 2 == 0 else second_column
            with target_column:
                edited_values[feature] = add_feature_input(
                    feature,
                    reference_row[feature],
                    key=f"interactive_{feature}"
                )

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary"
        )

    st.caption(
        "All non-edited features are carried over from the selected validation "
        "observation. The prediction is hypothetical when an input is changed."
    )

    if submitted:
        try:
            prediction, probability = get_prediction(edited_values)
            st.divider()
            st.subheader("Prediction Result")
            display_prediction(prediction, probability)

            comparison_rows = []
            for feature in editable_features:
                comparison_rows.append({
                    "Feature": feature,
                    "Reference value": reference_row[feature],
                    "Prediction input": edited_values[feature]
                })

            st.subheader("Reference Values and Prediction Inputs")
            st.dataframe(
                pd.DataFrame(comparison_rows),
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "This model includes delivery outcomes that become available "
                "after fulfilment. The output should be interpreted as an "
                "exploratory order assessment, not an early warning at purchase. "
                "The model also misses most actual low-satisfaction cases on the test set."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 9. SAMPLE PREDICTIONS PAGE
# ============================================================

elif page == "Sample Predictions":

    st.title("Sample Model Predictions")

    st.write(
        "Examine individual validation observations to see where the model "
        "correctly identifies low satisfaction and where it makes errors."
    )

    with st.spinner("Preparing validation predictions..."):
        validation_results = get_validation_predictions().copy()

    validation_results["predicted"] = (
        validation_results["predicted_probability"] >= THRESHOLD
    ).astype(int)

    st.info(
        f"These examples use all {len(validation_results):,} chronological "
        f"validation observations. The classification threshold is {THRESHOLD:.2f}."
    )

    validation_metrics = calculate_classification_metrics(
        validation_results["actual"],
        validation_results["predicted"]
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Validation Accuracy", f"{validation_metrics['accuracy']:.2%}")
    metric_columns[1].metric("Validation Precision", f"{validation_metrics['precision']:.2%}")
    metric_columns[2].metric("Validation Recall", f"{validation_metrics['recall']:.2%}")
    metric_columns[3].metric("Validation F1", f"{validation_metrics['f1']:.2%}")

    example_masks = {
        "True Positive": (
            (validation_results["actual"] == 1)
            & (validation_results["predicted"] == 1)
        ),
        "False Negative": (
            (validation_results["actual"] == 1)
            & (validation_results["predicted"] == 0)
        ),
        "True Negative": (
            (validation_results["actual"] == 0)
            & (validation_results["predicted"] == 0)
        ),
        "False Positive": (
            (validation_results["actual"] == 0)
            & (validation_results["predicted"] == 1)
        )
    }

    examples = {}
    for name, mask in example_masks.items():
        candidates = validation_results.loc[mask]
        if candidates.empty:
            continue
        if name in ["True Positive", "False Negative"]:
            selected_index = candidates["predicted_probability"].idxmax()
        else:
            selected_index = candidates["predicted_probability"].idxmin()
        examples[name] = int(selected_index)

    if not examples:
        st.warning("No validation examples could be selected.")
    else:
        st.divider()
        st.subheader("Illustrative Validation Cases")

        selected_example = st.selectbox(
            "Select a prediction case",
            list(examples.keys())
        )
        selected_index = examples[selected_example]
        example = validation_results.loc[selected_index]

        actual_label = (
            "Low satisfaction" if int(example["actual"]) == 1
            else "Not low satisfaction"
        )
        predicted_label = (
            "Low satisfaction" if int(example["predicted"]) == 1
            else "Not low satisfaction"
        )

        if selected_example in ["True Positive", "True Negative"]:
            st.success(f"{selected_example}: The model classified this validation observation correctly.")
        else:
            st.error(f"{selected_example}: The model classified this validation observation incorrectly.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Probability", f"{example['predicted_probability']:.2%}")
        col2.metric("Classification Threshold", f"{THRESHOLD:.2%}")
        col3.metric("Actual Outcome", actual_label)

        st.subheader("Actual vs Predicted Outcome")
        st.dataframe(
            pd.DataFrame({
                "Measure": ["Actual outcome", "Model prediction"],
                "Classification": [actual_label, predicted_label]
            }),
            hide_index=True,
            use_container_width=True
        )

        _, x_validation, _, _, _, _ = get_chronological_splits()
        selected_order = x_validation.iloc[selected_index]
        detail_features = [
            feature for feature in [
                "customer_state",
                "delivery_delay_days",
                "delivery_days",
                "late_delivery",
                "total_freight_value",
                "average_seller_previous_delay",
                "average_seller_distance_km",
                "total_payment_value"
            ]
            if feature in FEATURES
        ]

        st.subheader("Selected Order Characteristics")
        st.dataframe(
            pd.DataFrame({
                "Feature": detail_features,
                "Value": [selected_order[feature] for feature in detail_features]
            }),
            hide_index=True,
            use_container_width=True
        )

        explanations = {
            "True Positive": (
                "The observation had a low-satisfaction outcome and the model "
                "correctly flagged it. This is an example of a low-satisfaction "
                "case the model was able to identify."
            ),
            "False Negative": (
                "The observation had a low-satisfaction outcome, but the model "
                "did not flag it. False negatives are important because the model "
                "misses some orders associated with poor customer experiences."
            ),
            "True Negative": (
                "The observation did not have a low-satisfaction outcome and "
                "the model correctly did not flag it."
            ),
            "False Positive": (
                "The observation did not have a low-satisfaction outcome, but "
                "the model flagged it. False positives can lead to unnecessary "
                "investigation if predictions are used operationally."
            )
        }

        st.subheader("Business Interpretation")
        st.write(explanations[selected_example])

        st.metric(
            "Distance from Classification Threshold",
            f"{(example['predicted_probability'] - THRESHOLD) * 100:+.2f} percentage points"
        )

        st.caption(
            "The validation observation number is its position in the saved "
            "chronological validation split, not an Olist order ID. Individual "
            "examples are illustrative and are not representative of all orders."
        )


# ============================================================
# 10. THRESHOLD ANALYSIS PAGE
# ============================================================

elif page == "Threshold Analysis":

    st.title("Classification Threshold Analysis")

    st.write(
        "Explore how changing the classification threshold affects the model's "
        "ability to identify orders associated with low customer satisfaction."
    )

    with st.spinner("Preparing validation probabilities..."):
        validation_results = get_validation_predictions()

    actual = validation_results["actual"].astype(int)
    probabilities = validation_results["predicted_probability"].astype(float)

    total_orders = len(validation_results)
    actual_low = int((actual == 1).sum())
    actual_not_low = int((actual == 0).sum())
    low_rate = actual_low / total_orders if total_orders else 0.0

    st.info(
        "Threshold analysis uses the chronological validation dataset. "
        "Changing the threshold changes the classification decision, "
        "not the model's predicted probabilities."
    )

    st.subheader("Validation Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation Orders", f"{total_orders:,}")
    col2.metric("Low-Satisfaction Orders", f"{actual_low:,}")
    col3.metric("Low-Satisfaction Rate", f"{low_rate:.2%}")

    st.divider()
    st.subheader("Adjust the Classification Threshold")

    selected_threshold = st.slider(
        "Select classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(THRESHOLD),
        step=0.01,
        format="%.2f"
    )

    st.caption(
        f"The deployed model threshold is {THRESHOLD:.2f}. It was selected "
        "using validation data to maximise F1 Score. Final test results are "
        "reported separately on the Model Performance page."
    )

    predicted = (probabilities >= selected_threshold).astype(int)
    metrics = calculate_classification_metrics(actual, predicted)
    flagged = metrics["tp"] + metrics["fp"]

    st.divider()
    st.subheader("Performance at Selected Threshold")

    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    columns[1].metric("Precision", f"{metrics['precision']:.2%}")
    columns[2].metric("Recall", f"{metrics['recall']:.2%}")
    columns[3].metric("F1 Score", f"{metrics['f1']:.2%}")

    columns = st.columns(3)
    columns[0].metric("False Positive Rate", f"{metrics['fpr']:.2%}")
    columns[1].metric("Orders Flagged", f"{flagged:,}")
    columns[2].metric("Low-Satisfaction Orders Missed", f"{metrics['fn']:,}")

    threshold_cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=threshold_cm,
            x=["Predicted: Not Low", "Predicted: Low"],
            y=["Actual: Not Low", "Actual: Low"],
            text=threshold_cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )
    fig_cm.update_layout(
        title="Validation Confusion Matrix",
        xaxis_title="Predicted outcome",
        yaxis_title="Actual outcome",
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.subheader("Business Interpretation")

    st.write(
        f"At a threshold of **{selected_threshold:.2f}**, the model flags "
        f"**{flagged:,} orders** for potential low customer satisfaction. It "
        f"correctly identifies **{metrics['tp']:,}** low-satisfaction orders "
        f"and misses **{metrics['fn']:,}**. Of the flagged orders, "
        f"**{metrics['precision']:.2%}** are actually low satisfaction."
    )

    if metrics["recall"] < 0.50:
        st.warning(
            "At this threshold, the model misses more than half of actual "
            "low-satisfaction orders. A lower threshold may identify more cases, "
            "but it generally increases false alerts."
        )
    else:
        st.warning(
            "This threshold identifies more than half of actual low-satisfaction "
            "orders. The additional false alerts should still be considered."
        )

    st.divider()
    st.subheader("How Performance Changes Across Thresholds")

    threshold_values = np.round(np.arange(0.05, 0.951, 0.05), 2)
    sensitivity_results = []

    for value in threshold_values:
        threshold_predictions = (probabilities >= value).astype(int)
        threshold_metrics = calculate_classification_metrics(
            actual, threshold_predictions
        )
        sensitivity_results.append({
            "Threshold": value,
            "Precision": threshold_metrics["precision"],
            "Recall": threshold_metrics["recall"],
            "F1 Score": threshold_metrics["f1"]
        })

    sensitivity_df = pd.DataFrame(sensitivity_results)
    fig_sensitivity = px.line(
        sensitivity_df,
        x="Threshold",
        y=["Precision", "Recall", "F1 Score"],
        markers=True,
        title="Validation Precision, Recall and F1 by Threshold"
    )
    fig_sensitivity.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_sensitivity, use_container_width=True)

    st.dataframe(
        sensitivity_df.style.format({
            "Threshold": "{:.2f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All threshold comparisons on this page are validation results. "
        "Threshold changes do not resolve the model's underlying predictive "
        "limitations, and validation results should not be presented as test results."
    )


# ============================================================
# 11. WHAT-IF ANALYSIS PAGE
# ============================================================

elif page == "What-If Analysis":

    st.title("What-If Analysis")

    st.write(
        "Explore how changing one order characteristic affects the model's "
        "predicted probability while all other features remain fixed."
    )

    st.info(
        "This is a sensitivity exercise using an existing validation order. "
        "It demonstrates the model's response to hypothetical input changes; "
        "it does not establish cause and effect."
    )

    x_train, x_validation, _, _, _, _ = get_chronological_splits()

    numeric_drivers = [
        feature for feature in [
            "delivery_delay_days",
            "delivery_days",
            "average_seller_previous_delay",
            "total_freight_value",
            "total_payment_value",
            "average_item_price"
        ]
        if feature in FEATURES
        and pd.api.types.is_numeric_dtype(DATASET[feature])
    ]

    if not numeric_drivers:
        st.warning("No suitable numeric features are available for what-if analysis.")
        st.stop()

    observation_number = st.slider(
        "Validation observation",
        min_value=1,
        max_value=len(x_validation),
        value=1,
        key="whatif_observation"
    )
    reference_values = x_validation.iloc[observation_number - 1].copy()

    selected_feature = st.selectbox(
        "Feature to change",
        numeric_drivers,
        index=(
            numeric_drivers.index("delivery_delay_days")
            if "delivery_delay_days" in numeric_drivers else 0
        )
    )

    baseline_prediction, baseline_probability = get_prediction(reference_values)

    st.metric("Baseline Predicted Probability", f"{baseline_probability:.2%}")
    st.caption(
        f"Baseline {selected_feature.replace('_', ' ')} value: "
        f"{reference_values[selected_feature]}"
    )

    training_values = pd.to_numeric(
        x_train[selected_feature], errors="coerce"
    ).dropna()

    scenario_values = [
        float(training_values.quantile(q))
        for q in [0.10, 0.30, 0.50, 0.70, 0.90]
    ]
    scenario_values.append(float(reference_values[selected_feature]))
    scenario_values = sorted(set(scenario_values))

    scenario_results = []
    for scenario_value in scenario_values:
        modified_values = reference_values.copy()
        modified_values[selected_feature] = scenario_value
        scenario_prediction, scenario_probability = get_prediction(modified_values)

        scenario_results.append({
            selected_feature: scenario_value,
            "Predicted Probability": scenario_probability,
            "Change from Baseline (pp)": (
                scenario_probability - baseline_probability
            ) * 100,
            "Classification": (
                "Low satisfaction" if scenario_prediction == 1
                else "Not low satisfaction"
            )
        })

    results_df = pd.DataFrame(scenario_results).sort_values(selected_feature)

    st.divider()
    st.subheader("Predicted Probability Across Scenarios")

    chart = px.line(
        results_df,
        x=selected_feature,
        y="Predicted Probability",
        markers=True,
        title=f"Model Response to Changes in {selected_feature.replace('_', ' ').title()}"
    )
    chart.add_hline(
        y=THRESHOLD,
        line_dash="dot",
        annotation_text="Classification threshold"
    )
    chart.update_yaxes(tickformat=".0%")
    chart.update_layout(height=430)
    st.plotly_chart(chart, use_container_width=True)

    display_df = results_df.copy()
    display_df["Predicted Probability"] = display_df["Predicted Probability"].map(
        lambda value: f"{value:.2%}"
    )
    display_df["Change from Baseline (pp)"] = display_df[
        "Change from Baseline (pp)"
    ].map(lambda value: f"{value:+.2f}")

    st.subheader("Scenario Comparison")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    probability_range = (
        results_df["Predicted Probability"].max()
        - results_df["Predicted Probability"].min()
    ) * 100

    st.divider()
    st.subheader("Business Interpretation")
    st.write(
        f"Across the displayed values, predicted probability ranges from "
        f"**{results_df['Predicted Probability'].min():.2%}** to "
        f"**{results_df['Predicted Probability'].max():.2%}**. The difference "
        f"between the highest and lowest probabilities is "
        f"**{probability_range:.2f} percentage points**."
    )
    st.caption(
        "Scenario values are based on the training-data distribution. Other "
        "features remain unchanged. A change in model probability does not "
        "show that changing the feature would cause satisfaction to change."
    )


# ============================================================
# 12. MODEL EXPLAINABILITY PAGE
# ============================================================

elif page == "Model Explainability":

    st.title("Model Explainability")

    st.write(
        "Explore the order characteristics that contribute to the model's "
        "low-satisfaction predictions using SHAP explanations."
    )

    st.info(
        "SHAP values describe how features influence the model's output. "
        "They do not prove that a feature causes customer dissatisfaction."
    )

    if st.button("Generate SHAP Explanations", type="primary"):
        with st.spinner("Calculating SHAP values for 300 validation observations..."):
            try:
                import shap

                _, x_validation, _, _, y_validation, _ = get_chronological_splits()
                sample_size = min(50, len(x_validation))
                x_shap = x_validation.iloc[:sample_size].copy()
                y_shap = y_validation.iloc[:sample_size].copy()

                transformed = PREPROCESSOR.transform(x_shap)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                transformed = np.asarray(transformed)

                explainer = shap.TreeExplainer(MODEL)
                raw_shap_values = explainer.shap_values(transformed)

                if isinstance(raw_shap_values, list):
                    positive_shap = np.asarray(raw_shap_values[POSITIVE_INDEX])
                else:
                    raw_shap_values = np.asarray(raw_shap_values)
                    if raw_shap_values.ndim == 3:
                        if raw_shap_values.shape[-1] == len(MODEL.classes_):
                            positive_shap = raw_shap_values[:, :, POSITIVE_INDEX]
                        elif raw_shap_values.shape[0] == len(MODEL.classes_):
                            positive_shap = raw_shap_values[POSITIVE_INDEX, :, :]
                        else:
                            raise ValueError(
                                "Unexpected SHAP output dimensions for the classifier."
                            )
                    else:
                        positive_shap = raw_shap_values

                transformed_features = list(PREPROCESSOR.get_feature_names_out())
                if positive_shap.shape[1] != len(transformed_features):
                    raise ValueError(
                        "The SHAP output does not match the transformed feature names."
                    )

                importance = pd.Series(
                    np.abs(positive_shap).mean(axis=0),
                    index=transformed_features
                ).sort_values(ascending=False)

                probabilities = predict_positive_probability(x_shap)

                st.session_state["integrated_shap_results"] = {
                    "X_shap": x_shap,
                    "y_shap": y_shap,
                    "positive_shap": positive_shap,
                    "transformed_features": transformed_features,
                    "importance": importance,
                    "probabilities": probabilities
                }

                st.success(
                    f"SHAP explanations generated for {sample_size:,} validation observations."
                )
            except Exception as error:
                st.error(f"Unable to generate SHAP explanations: {error}")

    if "integrated_shap_results" in st.session_state:
        results = st.session_state["integrated_shap_results"]
        x_shap = results["X_shap"]
        y_shap = results["y_shap"]
        positive_shap = results["positive_shap"]
        transformed_features = results["transformed_features"]
        importance = results["importance"]
        probabilities = results["probabilities"]

        st.divider()
        st.subheader("1. Global Feature Importance")
        st.write(
            "Features are ranked by their average absolute SHAP contribution "
            "across the sampled validation observations."
        )

        top_importance = importance.head(15).sort_values(ascending=True)
        global_fig = px.bar(
            x=top_importance.values,
            y=[
                name.replace("numerical__", "").replace("categorical__", "")
                for name in top_importance.index
            ],
            orientation="h",
            labels={
                "x": "Mean Absolute SHAP Value",
                "y": "Model Feature"
            },
            title="Top 15 Features by Average SHAP Importance"
        )
        global_fig.update_layout(
            showlegend=False,
            yaxis_title="",
            xaxis_title="Mean Absolute SHAP Value"
        )
        st.plotly_chart(global_fig, use_container_width=True)

        top_feature_names = [
            name.replace("numerical__", "").replace("categorical__", "")
            for name in importance.head(10).index
        ]
        st.caption(
            "Higher absolute SHAP values indicate stronger average influence "
            "on model output in this sample. Absolute values do not show direction."
        )

        st.divider()
        st.subheader("2. Individual Prediction Explanation")

        selected_order = st.slider(
            "Select a validation observation from the SHAP sample",
            min_value=1,
            max_value=len(x_shap),
            value=1
        )
        row_index = selected_order - 1
        probability = float(probabilities[row_index])
        prediction = int(probability >= THRESHOLD)
        actual_outcome = int(y_shap.iloc[row_index])

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Probability", f"{probability:.2%}")
        col2.metric(
            "Model Classification",
            "Low satisfaction" if prediction == 1 else "Not low satisfaction"
        )
        col3.metric(
            "Actual Outcome",
            "Low satisfaction" if actual_outcome == 1 else "Not low satisfaction"
        )

        st.caption(
            f"Classification threshold: {THRESHOLD:.2f}. "
            "The actual outcome is shown for this validation observation only."
        )

        individual_shap = pd.Series(
            positive_shap[row_index],
            index=transformed_features
        )
        individual_shap = individual_shap.reindex(
            individual_shap.abs().sort_values(ascending=False).head(10).index
        ).sort_values()

        individual_labels = [
            name.replace("numerical__", "").replace("categorical__", "")
            for name in individual_shap.index
        ]
        contribution_directions = [
            "Pushes towards low satisfaction" if value > 0
            else "Pushes away from low satisfaction"
            for value in individual_shap.values
        ]

        individual_fig = px.bar(
            x=individual_shap.values,
            y=individual_labels,
            orientation="h",
            color=contribution_directions,
            color_discrete_map={
                "Pushes towards low satisfaction": "#D95F5F",
                "Pushes away from low satisfaction": "#4C9F70"
            },
            labels={
                "x": "SHAP Contribution",
                "y": "Feature",
                "color": "Contribution Direction"
            },
            title="Top 10 Contributions to the Selected Prediction"
        )
        individual_fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(individual_fig, use_container_width=True)

        st.divider()
        st.subheader("3. Selected Order Characteristics")
        selected_raw_row = x_shap.iloc[row_index]
        raw_details = [
            feature for feature in [
                "customer_state",
                "delivery_delay_days",
                "delivery_days",
                "late_delivery",
                "item_count",
                "total_freight_value",
                "average_seller_previous_delay",
                "total_payment_value"
            ]
            if feature in FEATURES
        ]
        st.dataframe(
            pd.DataFrame({
                "Feature": raw_details,
                "Value": [selected_raw_row[feature] for feature in raw_details]
            }),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("4. How to Interpret These Results")
        st.markdown(
            """
            - **Positive SHAP contribution:** Pushes the model output towards
              the low-satisfaction class.
            - **Negative SHAP contribution:** Pushes the model output away
              from the low-satisfaction class.
            - **Larger absolute values:** Indicate stronger contributions
              to the model output.

            **Important limitations**
            - Global explanations use the first 300 observations in the
              chronological validation split, not the complete dataset.
            - SHAP values describe model behaviour; they do not establish causality.
            - Customer state is a geographic variable, not a direct measure
              of a protected demographic characteristic.
            - Explanations do not remove the model's limited test recall.
            - Predictions require human interpretation and review.
            """
        )

    else:
        st.warning(
            "Select 'Generate SHAP Explanations' to calculate the global and "
            "individual explanations."
        )


# ============================================================
# 13. ETHICAL COMPLIANCE AND FAIRNESS PAGE
# ============================================================

elif page == "Ethical Compliance":

    st.title("Ethical Compliance and Geographic Fairness")

    st.write(
        "This section examines whether the selected model performs differently "
        "across customer states in the final chronological test dataset."
    )

    st.info(
        "Customer state is a geographic variable, not a protected demographic "
        "attribute. Differences across states are monitoring signals; they do "
        "not by themselves prove discrimination or establish fairness."
    )

    with st.spinner("Calculating test-set geographic performance..."):
        fairness_source = get_test_prediction_data().copy()

    minimum_group_size = 100
    state_counts = fairness_source["customer_state"].value_counts()
    small_states = state_counts[state_counts < minimum_group_size].index
    fairness_source["State Group"] = fairness_source["customer_state"].where(
        ~fairness_source["customer_state"].isin(small_states),
        "Other"
    )

    group_rows = []
    for state, group in fairness_source.groupby("State Group", dropna=False):
        actual = group["actual"].to_numpy(dtype=int)
        predicted = group["predicted"].to_numpy(dtype=int)

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())

        actual_positive = tp + fn
        actual_negative = fp + tn
        group_rows.append({
            "State Group": str(state),
            "Test Orders": len(group),
            "Actual Low Satisfaction (%)": (
                actual_positive / len(group) * 100 if len(group) else np.nan
            ),
            "Prediction Rate (%)": (
                (tp + fp) / len(group) * 100 if len(group) else np.nan
            ),
            "True Positive Rate (%)": (
                tp / actual_positive * 100 if actual_positive else np.nan
            ),
            "False Positive Rate (%)": (
                fp / actual_negative * 100 if actual_negative else np.nan
            )
        })

    fairness_data = pd.DataFrame(group_rows).sort_values(
        "Test Orders", ascending=False
    )

    st.subheader("Group-Level Performance")
    st.caption(
        f"States with fewer than {minimum_group_size} test orders are combined "
        "into 'Other' to avoid over-interpreting very small groups."
    )

    tpr_values = fairness_data["True Positive Rate (%)"].dropna()
    fpr_values = fairness_data["False Positive Rate (%)"].dropna()
    tpr_gap = (
        tpr_values.max() - tpr_values.min() if len(tpr_values) else np.nan
    )
    fpr_gap = (
        fpr_values.max() - fpr_values.min() if len(fpr_values) else np.nan
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Groups Reviewed", f"{len(fairness_data):,}")
    col2.metric("TPR Gap", f"{tpr_gap:.2f} pp" if pd.notna(tpr_gap) else "N/A")
    col3.metric("FPR Gap", f"{fpr_gap:.2f} pp" if pd.notna(fpr_gap) else "N/A")

    st.dataframe(
        fairness_data.style.format({
            "Actual Low Satisfaction (%)": "{:.2f}",
            "Prediction Rate (%)": "{:.2f}",
            "True Positive Rate (%)": "{:.2f}",
            "False Positive Rate (%)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.subheader("True Positive Rate by State Group")

    tpr_fig = px.bar(
        fairness_data.sort_values("True Positive Rate (%)", ascending=False),
        x="State Group",
        y="True Positive Rate (%)",
        hover_data=["Test Orders", "Actual Low Satisfaction (%)"],
        title="Share of Actual Low-Satisfaction Orders Identified"
    )
    tpr_fig.update_layout(height=450)
    st.plotly_chart(tpr_fig, use_container_width=True)

    st.divider()
    st.subheader("False Positive Rate by State Group")

    fpr_fig = px.bar(
        fairness_data.sort_values("False Positive Rate (%)", ascending=False),
        x="State Group",
        y="False Positive Rate (%)",
        hover_data=["Test Orders", "Prediction Rate (%)"],
        title="Share of Actual Non-Low-Satisfaction Orders Incorrectly Flagged"
    )
    fpr_fig.update_layout(height=450)
    st.plotly_chart(fpr_fig, use_container_width=True)

    st.divider()
    st.subheader("Groups with the Lowest Recall")

    lowest_recall = fairness_data.dropna(
        subset=["True Positive Rate (%)"]
    ).nsmallest(5, "True Positive Rate (%)")
    st.dataframe(
        lowest_recall[[
            "State Group", "Test Orders", "Actual Low Satisfaction (%)",
            "True Positive Rate (%)", "False Positive Rate (%)"
        ]].style.format({
            "Actual Low Satisfaction (%)": "{:.2f}",
            "True Positive Rate (%)": "{:.2f}",
            "False Positive Rate (%)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.subheader("Ethical Interpretation and Controls")

    st.markdown(
        """
        **Observed risks**
        - Recall may differ across geographic groups, meaning the model may
          miss a different share of low-satisfaction orders in different states.
        - Small group sizes can make performance estimates unstable.
        - Customer state may reflect differences in delivery distance,
          logistics coverage or other structural conditions.

        **Responsible-use controls**
        - Investigate low-recall groups before considering operational use.
        - Repeat the review across additional time periods and larger samples.
        - Use protected-group analysis only where lawful, appropriate and
          supported by suitable data.
        - Monitor model performance and geographic differences over time.
        - Retain human review; do not make automatic customer decisions from
          the model's output.
        - Restrict access to order-level data and minimise any information
          displayed to users who do not need it.
        """
    )

    st.warning(
        "This page reports geographic performance differences for the selected "
        "model. It does not include the earlier Equalized Odds experiment because "
        "those results were produced under the previous modelling setup and are "
        "not directly comparable with this corrected integrated model."
    )


# ============================================================
# 14. MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.title("Model Information and Transparency")

    st.write(
        "This page summarises the corrected dataset, model inputs, training "
        "approach, evaluation results and responsible-use limitations."
    )

    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        get_chronological_splits()
    )

    transformed_feature_count = len(TRANSFORMED_FEATURES)
    categorical_features = [
        feature for feature in FEATURES
        if not pd.api.types.is_numeric_dtype(DATASET[feature])
    ]
    numerical_features = [
        feature for feature in FEATURES
        if pd.api.types.is_numeric_dtype(DATASET[feature])
    ]

    st.subheader("Model Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model name:** {MODEL_NAME}")
        st.write("**Algorithm:** Random Forest Classifier")
        st.write(f"**Number of trees:** {getattr(MODEL, 'n_estimators', 'Not available')}")
        st.write("**Prediction unit:** Individual order")
    with col2:
        st.write(f"**Classification threshold:** {THRESHOLD:.2f}")
        st.write("**Positive outcome:** Review score of 2 or below")
        st.write("**Evaluation:** Chronological test dataset")
        st.write(f"**Target field:** {TARGET}")

    st.write(
        "The model estimates whether an order is associated with a low "
        "customer review score. A probability at or above the threshold is "
        "classified as low satisfaction predicted."
    )

    st.divider()
    st.subheader("Dataset and Chronological Split")

    split_columns = st.columns(4)
    split_columns[0].metric("Integrated Orders", f"{len(DATASET):,}")
    split_columns[1].metric("Training Orders", f"{len(x_train):,}")
    split_columns[2].metric("Validation Orders", f"{len(x_validation):,}")
    split_columns[3].metric("Test Orders", f"{len(x_test):,}")

    target_counts = DATASET[TARGET].value_counts(dropna=False).sort_index()
    positive_count = int((pd.to_numeric(DATASET[TARGET], errors="coerce") == 1).sum())
    negative_count = int((pd.to_numeric(DATASET[TARGET], errors="coerce") == 0).sum())
    positive_rate = positive_count / (positive_count + negative_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Low-Satisfaction Orders", f"{positive_count:,}")
    col2.metric("Other Orders", f"{negative_count:,}")
    col3.metric("Overall Positive Rate", f"{positive_rate:.2%}")

    st.caption(
        "The chronological split keeps earlier orders for training, the next "
        "period for validation and the latest period for testing."
    )

    st.divider()
    st.subheader("Model Input Features")

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Predictors", f"{len(FEATURES)}")
    col2.metric("Numerical Predictors", f"{len(numerical_features)}")
    col3.metric("Transformed Predictors", f"{transformed_feature_count}")

    st.write(
        "The model uses the 39 original predictors retained by the fitted "
        "preprocessing pipeline. Customer state is one-hot encoded, producing "
        "65 transformed features for the Random Forest."
    )

    with st.expander("View all original model input features"):
        st.dataframe(
            pd.DataFrame({"Model input feature": FEATURES}),
            hide_index=True,
            use_container_width=True
        )

    st.divider()
    st.subheader("Preprocessing and Leakage Controls")

    st.markdown(
        """
        - Numerical missing values are handled using median imputation.
        - Numerical features are standardised using a scaler fitted on training data.
        - Categorical missing values are handled using most-frequent imputation.
        - Customer state is one-hot encoded, with unseen categories handled by
          the fitted encoder.
        - The fitted preprocessor is saved together with the selected model,
          ensuring interactive predictions use the same transformations as training.
        - Review information is excluded from the predictors to avoid using the
          target source as an input.
        - Historical seller and product features use prior completed orders;
          orders with the same delivery timestamp are excluded from one another's
          historical calculations.
        """
    )

    st.divider()
    st.subheader("Final Test Performance")

    metric_columns = st.columns(3)
    for index, (metric, value) in enumerate(TEST_METRICS.items()):
        with metric_columns[index % 3]:
            st.metric(metric, f"{value:.2f}%")

    st.warning(
        "The model's recall is 26.59%, so it misses most actual low-satisfaction "
        "orders. Test accuracy is also below the 90.33% majority-class baseline. "
        "The model is an academic prototype for analysis, not an approved "
        "production decision system."
    )

    st.divider()
    st.subheader("Comparison with the Earlier Model Setup")

    previous_comparison = pd.DataFrame([
        ["Earlier model setup", 81.64, 18.21, 25.73, 21.33, 58.17, 16.98],
        ["Corrected integrated model", 89.68, 44.45, 26.59, 33.28, 67.02, 28.78]
    ], columns=[
        "Model setup", "Accuracy (%)", "Precision (%)", "Recall (%)",
        "F1 Score (%)", "ROC-AUC (%)", "PR-AUC (%)"
    ])

    st.dataframe(
        previous_comparison.style.format({
            column: "{:.2f}" for column in previous_comparison.columns
            if column != "Model setup"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "The earlier and corrected results come from different data integration, "
        "target construction and feature setups. This is a project-level comparison, "
        "not a controlled experiment; the full improvement cannot be attributed "
        "to any single change."
    )

    st.divider()
    st.subheader("Random Forest Feature Importance")

    try:
        feature_importances = np.asarray(MODEL.feature_importances_)
        if len(feature_importances) == len(TRANSFORMED_FEATURES):
            importance_df = pd.DataFrame({
                "Feature": TRANSFORMED_FEATURES,
                "Importance": feature_importances
            }).sort_values("Importance", ascending=False).head(15)

            importance_fig = px.bar(
                importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Random Forest Features"
            )
            st.plotly_chart(importance_fig, use_container_width=True)
    except Exception:
        st.caption("Built-in feature importance is not available for this saved model.")

    st.divider()
    st.subheader("Important Limitations")

    st.markdown(
        """
        1. **Missed low-satisfaction orders:** Recall of 26.59% means most actual
           low-satisfaction cases are not identified.
        2. **Class imbalance:** Low-satisfaction orders form a minority of the
           test data, so accuracy must be interpreted alongside precision,
           recall, F1 and PR-AUC.
        3. **Geographic differences:** Customer-state performance gaps require
           investigation and do not independently prove unfair treatment.
        4. **Limited demographic information:** The public dataset does not
           provide sufficient protected demographic information for a full
           demographic fairness assessment.
        5. **Prediction versus causation:** Model patterns do not establish why
           a customer may be dissatisfied.
        6. **Prediction-time availability:** Delivery duration, delay and
           late-delivery status are observed after fulfilment. Their inclusion
           means the current model should not be presented as an early warning
           model available immediately after purchase.
        7. **Generalisation:** Performance should be monitored on later periods
           and other datasets before any operational use.
        """
    )

    st.divider()
    st.subheader("Responsible Use Statement")

    st.info(
        "This model is an exploratory decision-support tool. Its outputs are "
        "estimates, not confirmed customer outcomes. Human review, appropriate "
        "data protection, further temporal validation and ongoing performance "
        "and fairness monitoring are necessary before operational use."
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()
st.caption(
    "Olist Brazilian E-Commerce Analytics Project | "
    "Integrated Low-Satisfaction Prediction | Academic prototype"
)
