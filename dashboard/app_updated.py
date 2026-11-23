from pathlib import Path
import json

import joblib
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# ============================================================

st.set_page_config(
    page_title="Olist Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# app.py is inside the dashboard folder; the project root is one level above.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "olist_low_satisfaction_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivered_order_level.csv"
THRESHOLD_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation_threshold_data.csv"
)


# ============================================================
# 2. MODEL FEATURES
# ============================================================

FEATURES = [
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

CATEGORICAL_FEATURES = [
    "customer_state",
    "preferred_payment_type"
]


# ============================================================
# 3. LOAD PROJECT FILES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_threshold_data():
    return pd.read_csv(THRESHOLD_DATA_PATH)


try:
    model = load_model()
    model_config = load_config()
    dataset = load_dataset()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()
except Exception as error:
    st.error(f"An error occurred while loading project files: {error}")
    st.stop()


# ============================================================
# 4. MODEL CONFIGURATION AND VERIFIED TEST RESULTS
# ============================================================

MODEL_NAME = model_config.get(
    "model_name", "Olist Low Satisfaction Prediction"
)
MODEL_TYPE = model_config.get("model_type", "Tuned Random Forest")
THRESHOLD = float(model_config.get("classification_threshold", 0.5047))
POSITIVE_CLASS = model_config.get("positive_class", 1)
TARGET_DEFINITION = model_config.get(
    "positive_class_definition", "Average review score <= 2"
)

# Final chronological test results at threshold 0.5047.
TEST_METRICS = {
    "Accuracy": 81.64,
    "Precision": 18.21,
    "Recall": 25.73,
    "F1 Score": 21.33,
    "ROC-AUC": 58.17,
    "PR-AUC": 16.98
}

CONFUSION_MATRIX = pd.DataFrame(
    [[15171, 2142], [1377, 477]],
    index=["Actual: Not low satisfaction", "Actual: Low satisfaction"],
    columns=[
        "Predicted: Not low satisfaction",
        "Predicted: Low satisfaction"
    ]
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
st.sidebar.markdown("### Model details")
st.sidebar.write(f"**Model:** {MODEL_TYPE}")
st.sidebar.write(f"**Threshold:** {THRESHOLD:.4f}")
st.sidebar.write("**Prediction unit:** Order")
st.sidebar.caption(
    "This dashboard presents an analytical model for identifying "
    "orders that may receive low customer review scores."
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def prepare_model_input(values):
    """Convert user inputs into the DataFrame expected by the pipeline."""
    input_df = pd.DataFrame([values], columns=FEATURES)
    for column in CATEGORICAL_FEATURES:
        input_df[column] = input_df[column].astype(str)
    return input_df


def get_prediction(values):
    """Generate a prediction and positive-class probability."""
    input_df = prepare_model_input(values)
    probabilities = model.predict_proba(input_df)[0]
    positive_index = list(model.classes_).index(POSITIVE_CLASS)
    probability = float(probabilities[positive_index])
    prediction = int(probability >= THRESHOLD)
    return prediction, probability


def display_prediction(prediction, probability):
    """Display an order prediction in plain language."""
    st.metric(
        "Predicted probability of low satisfaction",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.warning(
            "The model identifies this order as having a higher "
            "predicted likelihood of low customer satisfaction."
        )
    else:
        st.success(
            "The model does not classify this order as likely "
            "to receive a low satisfaction score at the selected threshold."
        )

    st.caption(
        f"Classification threshold: {THRESHOLD:.4f}. "
        "This is a model prediction, not a confirmed customer outcome."
    )


def get_states_and_payments():
    """Return valid category options from the loaded dataset."""
    states = sorted(
        dataset["customer_state"].dropna().astype(str).unique().tolist()
    )
    payment_types = sorted(
        dataset["preferred_payment_type"].dropna().astype(str).unique().tolist()
    )
    return states, payment_types


# ============================================================
# 7. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("Olist Customer Experience Analytics")

    st.markdown(
        """
        **Understanding and investigating low customer satisfaction
        in the Olist e-commerce marketplace.**

        This dashboard presents an analytical model that estimates
        the likelihood of an order receiving a low customer review score.
        It helps business stakeholders explore customer experience
        patterns, examine model predictions and understand potential
        areas requiring further investigation.
        """
    )

    st.divider()
    st.subheader("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Delivered Orders", f"{len(dataset):,}")
    col2.metric("Model Algorithm", "Random Forest")
    col3.metric("Test Accuracy", "81.64%")
    col4.metric("Test Recall", "25.73%")

    st.caption(
        "Performance figures are from the final Module 4 chronological test dataset."
    )

    st.divider()
    st.subheader("The Business Problem")

    st.write(
        """
        Olist connects customers with sellers through an online marketplace.
        Customer satisfaction is important to maintaining customer
        relationships and supporting long-term business value.

        This project explores whether order characteristics and purchasing
        patterns can help identify orders associated with low customer
        review scores.
        """
    )

    st.subheader("Business Questions Supported by the Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Customer Experience**
            - Which orders are predicted to have low satisfaction?
            - How many potentially dissatisfied orders are identified?
            - What order characteristics influence predictions?
            """
        )
    with col2:
        st.markdown(
            """
            **Model Oversight**
            - How reliable are the model's predictions?
            - How does the classification threshold affect results?
            - Are there differences in performance across states?
            """
        )

    st.divider()
    st.subheader("How the Analytical Model Works")

    st.write(
        """
        The model uses 11 order-level characteristics, including customer
        state, item spending, freight spending, product information,
        payment type and purchase timing. It estimates the probability
        that an order will receive an average review score of 2 or below.
        """
    )

    st.info(
        "The original project vision focuses on customer segmentation. "
        "Low-satisfaction prediction extends that work by examining "
        "customer experience risks at order level."
    )

    st.divider()
    st.subheader("Explore the Dashboard")

    navigation = [
        ("Model Performance", "Review final test-set metrics and classification errors."),
        ("Interactive Predictions", "Enter order characteristics and generate a prediction."),
        ("Sample Predictions", "Examine actual validation examples of correct and incorrect predictions."),
        ("Threshold Analysis", "Explore how the classification threshold affects validation results."),
        ("What-If Analysis", "Explore how changing order characteristics affects predictions."),
        ("Model Explainability", "Examine features influencing predictions using SHAP."),
        ("Ethical Compliance", "Review geographic fairness results and mitigation trade-offs."),
        ("Model Information", "Review model inputs, experiments, limitations and responsible-use conditions.")
    ]

    for title, description in navigation:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.divider()
    st.subheader("Important Model Limitation")

    st.warning(
        """
        The model achieved 81.64% accuracy on the final test set, but its
        recall was 25.73%, precision was 18.21%, and ROC-AUC was 58.17%.
        It identified approximately one in four actual low-satisfaction
        orders and missed most of them. It is exploratory and should not
        be treated as a production-ready alert system or used as the sole
        basis for customer decisions.
        """
    )


# ============================================================
# 8. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "This page evaluates how accurately the Random Forest model "
        "identifies orders associated with low customer satisfaction."
    )

    st.caption(
        "Evaluation is based on 19,167 orders in the final chronological "
        "test dataset. All metrics below are test results."
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
            texttemplate="%{text:,}",
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
        **What the results mean:**
        - **15,171 True Negatives:** Correctly identified orders without low satisfaction.
        - **477 True Positives:** Correctly identified low-satisfaction orders.
        - **2,142 False Positives:** Orders incorrectly flagged as low satisfaction.
        - **1,377 False Negatives:** Actual low-satisfaction orders missed by the model.
        """
    )

    st.divider()
    st.subheader("Understanding the Performance Metrics")

    st.markdown(
        """
        - **Accuracy (81.64%):** Percentage of all orders classified correctly.
        - **Precision (18.21%):** Of the orders flagged as low satisfaction,
          approximately 18 in every 100 actually had low satisfaction.
        - **Recall (25.73%):** The model identified approximately one in four
          actual low-satisfaction orders.
        - **F1 Score (21.33%):** A combined measure of precision and recall.
        - **ROC-AUC (58.17%):** Indicates limited ability to distinguish
          between low- and non-low-satisfaction orders.
        - **PR-AUC (16.98%):** Measures precision-recall performance,
          particularly relevant when the positive class is relatively rare.
        """
    )

    st.divider()
    st.subheader("Why Accuracy Can Be Misleading")

    st.write(
        "Only 9.67% of the test orders had low satisfaction outcomes. "
        "This means the dataset is imbalanced, with substantially more "
        "non-low-satisfaction orders than low-satisfaction orders."
    )

    col1, col2 = st.columns(2)
    col1.metric("Actual low-satisfaction orders", "1,854", "9.67%")
    col2.metric("Actual non-low-satisfaction orders", "17,313", "90.33%")

    st.warning(
        "A model that classified every order as non-low-satisfaction "
        "would achieve 90.33% accuracy, despite identifying no "
        "low-satisfaction orders. The current model achieves 81.64% "
        "accuracy and detects 25.73% of actual low-satisfaction orders. "
        "Accuracy alone therefore does not demonstrate useful performance."
    )

    st.divider()
    st.subheader("Business Interpretation")

    st.write(
        """
        The model demonstrates limited ability to identify orders that may
        result in low customer satisfaction. It identifies some problematic
        orders, but misses most actual low-satisfaction cases and produces
        false alerts. Relying on it alone could lead to missed opportunities
        to investigate customer complaints and unnecessary follow-up.
        """
    )

    st.error(
        "The current test results do not support using this model as a "
        "standalone operational alert system. Further investigation, "
        "improvement and evaluation are needed before operational use."
    )


# ============================================================
# 9. INTERACTIVE PREDICTIONS PAGE
# ============================================================

elif page == "Interactive Predictions":

    st.title("Interactive Order Prediction")

    st.write(
        "Explore how different order characteristics relate to "
        "the estimated likelihood of receiving a low customer "
        "satisfaction score."
    )

    st.info(
        "Enter hypothetical order details and select Generate Prediction. "
        "The prediction is exploratory and is not a confirmed customer outcome."
    )

    states, payment_types = get_states_and_payments()

    st.subheader("Order Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Customer and Purchase Details**")
        customer_state = st.selectbox(
            "Customer state",
            states,
            index=states.index("SP") if "SP" in states else 0
        )
        item_spending = st.number_input(
            "Item spending (R$)", 0.0, 10000.0, 150.0, step=10.0
        )
        freight_spending = st.number_input(
            "Freight spending (R$)", 0.0, 3000.0, 25.0, step=5.0
        )
        total_items = st.number_input(
            "Total items", 1, 100, 2, step=1
        )
        distinct_products = st.number_input(
            "Distinct products", 1, 100, 2, step=1
        )
        category_diversity = st.number_input(
            "Product category diversity", 1, 100, 2, step=1
        )

    with col2:
        st.markdown("**Payment and Purchase Timing**")
        preferred_payment_type = st.selectbox(
            "Preferred payment type", payment_types
        )
        purchase_year = st.selectbox(
            "Purchase year", [2016, 2017, 2018], index=2
        )
        purchase_month = st.slider("Purchase month", 1, 12, 5)

        days = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
        purchase_day_of_week = st.selectbox(
            "Purchase day of week",
            list(range(7)),
            index=2,
            format_func=lambda x: days[x]
        )
        purchase_hour = st.slider(
            "Purchase hour", 0, 23, 14,
            help="Hour of purchase using the 24-hour clock."
        )

    st.divider()

    if distinct_products > total_items:
        st.error("Distinct products cannot exceed total items.")
    elif category_diversity > distinct_products:
        st.error(
            "Product category diversity cannot exceed the number "
            "of distinct products."
        )
    elif st.button("Generate Prediction", type="primary"):

        values = {
            "customer_state": customer_state,
            "item_spending": item_spending,
            "freight_spending": freight_spending,
            "total_items": total_items,
            "distinct_products": distinct_products,
            "category_diversity": category_diversity,
            "preferred_payment_type": preferred_payment_type,
            "purchase_year": purchase_year,
            "purchase_month": purchase_month,
            "purchase_day_of_week": purchase_day_of_week,
            "purchase_hour": purchase_hour
        }

        try:
            prediction, probability = get_prediction(values)
            st.divider()
            st.subheader("Prediction Result")
            display_prediction(prediction, probability)

            st.caption(
                "This estimate is based on the entered order characteristics. "
                "It does not establish that a customer will actually give "
                "a low review score."
            )

            st.warning(
                "The model has limited test performance, including 25.73% "
                "recall. Its predictions should be interpreted with caution "
                "and should not be used as the sole basis for customer decisions."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 10. SAMPLE PREDICTIONS PAGE
# ============================================================

elif page == "Sample Predictions":

    st.title("Sample Model Predictions")

    st.write(
        "This page demonstrates how the model classifies individual "
        "validation observations, including correct predictions "
        "and classification errors."
    )

    st.info(
        f"Examples are selected from the Module 4 validation dataset. "
        f"The classification threshold is {THRESHOLD:.4f}. "
        "Low satisfaction is defined as an average review score of 2 or below."
    )

    try:
        sample_df = load_threshold_data().copy()
    except FileNotFoundError:
        st.error(
            "The validation prediction file could not be found. "
            "Check data/processed/validation_threshold_data.csv."
        )
        st.stop()

    required_columns = {"actual", "predicted_probability"}
    if not required_columns.issubset(sample_df.columns):
        st.error(
            "The validation file must contain 'actual' and "
            "'predicted_probability' columns."
        )
        st.stop()

    sample_df = sample_df[
        ["actual", "predicted_probability"]
    ].copy()

    sample_df["actual"] = pd.to_numeric(
        sample_df["actual"], errors="coerce"
    )
    sample_df["predicted_probability"] = pd.to_numeric(
        sample_df["predicted_probability"], errors="coerce"
    )
    sample_df = sample_df.dropna().reset_index(drop=True)
    sample_df["actual"] = sample_df["actual"].astype(int)
    sample_df["predicted"] = (
        sample_df["predicted_probability"] >= THRESHOLD
    ).astype(int)

    example_masks = {
        "True Positive": (
            (sample_df["actual"] == 1) & (sample_df["predicted"] == 1)
        ),
        "False Negative": (
            (sample_df["actual"] == 1) & (sample_df["predicted"] == 0)
        ),
        "True Negative": (
            (sample_df["actual"] == 0) & (sample_df["predicted"] == 0)
        )
    }

    examples = {}
    for name, mask in example_masks.items():
        candidates = sample_df.loc[mask].copy()
        if not candidates.empty:
            if name == "True Positive":
                selected = candidates.sort_values(
                    "predicted_probability", ascending=True
                ).iloc[0]
            else:
                selected = candidates.sort_values(
                    "predicted_probability", ascending=False
                ).iloc[0]
            examples[name] = selected

    if not examples:
        st.warning("No suitable examples could be selected from the validation file.")
    else:
        st.subheader("Illustrative Validation Cases")
        selected_example = st.selectbox(
            "Select an example", list(examples.keys())
        )
        example = examples[selected_example]

        actual_outcome = (
            "Low satisfaction" if example["actual"] == 1
            else "Not low satisfaction"
        )
        predicted_outcome = (
            "Low satisfaction" if example["predicted"] == 1
            else "Not low satisfaction"
        )

        if selected_example == "True Positive":
            st.success(
                "True Positive: The model correctly identified an "
                "actual low-satisfaction validation observation."
            )
        elif selected_example == "False Negative":
            st.error(
                "False Negative: The model failed to identify an "
                "actual low-satisfaction validation observation."
            )
        else:
            st.success(
                "True Negative: The model correctly identified an "
                "observation without low satisfaction."
            )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Predicted Probability",
            f"{example['predicted_probability']:.2%}"
        )
        col2.metric("Classification Threshold", f"{THRESHOLD:.2%}")
        col3.metric("Actual Outcome", actual_outcome)

        st.divider()
        st.subheader("Actual vs Predicted Outcome")

        comparison = pd.DataFrame({
            "Measure": ["Actual outcome", "Model prediction"],
            "Classification": [actual_outcome, predicted_outcome]
        })
        st.dataframe(
            comparison, hide_index=True, use_container_width=True
        )

        st.subheader("Validation Observation Details")
        observation_details = pd.DataFrame({
            "Characteristic": [
                "Validation row",
                "Predicted probability",
                "Classification threshold",
                "Actual outcome",
                "Predicted outcome"
            ],
            "Value": [
                int(example.name) + 1,
                f"{example['predicted_probability']:.2%}",
                f"{THRESHOLD:.4f}",
                actual_outcome,
                predicted_outcome
            ]
        })
        st.dataframe(
            observation_details, hide_index=True, use_container_width=True
        )

        interpretations = {
            "True Positive": (
                "The observation had an actual low-satisfaction outcome, "
                "and the model correctly classified it. This illustrates "
                "a case where the model identified an order associated "
                "with a poor customer experience."
            ),
            "False Negative": (
                "The observation had an actual low-satisfaction outcome, "
                "but the model classified it as not low satisfaction. "
                "This demonstrates the risk of overlooking dissatisfied "
                "customers when relying only on model predictions."
            ),
            "True Negative": (
                "The observation did not have a low-satisfaction outcome, "
                "and the model correctly classified it as not low satisfaction."
            )
        }

        st.divider()
        st.subheader("Business Interpretation")
        st.write(interpretations[selected_example])

        st.subheader("Understanding the Classification Threshold")
        probability_difference = (
            example["predicted_probability"] - THRESHOLD
        ) * 100
        st.metric(
            "Distance from Classification Threshold",
            f"{probability_difference:+.4f} percentage points"
        )

        st.caption(
            "The validation row number identifies the observation's position "
            "in the saved validation prediction file; it is not an Olist order ID. "
            "The examples are not representative of all validation orders."
        )

        st.warning(
            "Individual examples do not establish overall model reliability. "
            "The model's limited test performance and its missed "
            "low-satisfaction cases must be considered."
        )


# ============================================================
# 11. THRESHOLD ANALYSIS PAGE
# ============================================================

elif page == "Threshold Analysis":

    st.title("Classification Threshold Analysis")

    st.write(
        "Explore how changing the classification threshold affects "
        "the model's ability to identify orders associated with "
        "low customer satisfaction."
    )

    st.info(
        "This analysis uses the 15,333-order Module 4 chronological "
        "validation dataset. Changing the threshold changes the "
        "classification decision, not the underlying model."
    )

    try:
        threshold_df = load_threshold_data().copy()
    except FileNotFoundError:
        st.error(
            "Could not find data/processed/validation_threshold_data.csv."
        )
        st.stop()

    required_columns = {"actual", "predicted_probability"}
    if not required_columns.issubset(threshold_df.columns):
        st.error(
            "The validation file must contain 'actual' and "
            "'predicted_probability' columns."
        )
        st.stop()

    actual = threshold_df["actual"].astype(int)
    probabilities = threshold_df["predicted_probability"].astype(float)

    total_orders = len(threshold_df)
    actual_low = int((actual == 1).sum())
    actual_not_low = int((actual == 0).sum())
    low_rate = actual_low / total_orders if total_orders else 0

    st.subheader("Validation Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Low-Satisfaction Orders", f"{actual_low:,}")
    col3.metric("Low-Satisfaction Rate", f"{low_rate:.2%}")

    st.divider()
    st.subheader("Adjust the Classification Threshold")

    threshold = st.slider(
        "Select classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=THRESHOLD,
        step=0.0001,
        format="%.4f"
    )

    st.caption(
        f"The selected model threshold is {THRESHOLD:.4f}. It was selected "
        "using validation data to maximise F1 Score. Final test metrics "
        "are reported separately on the Model Performance page."
    )

    predicted = (probabilities >= threshold).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0
    flagged = tp + fp
    missed = fn

    st.divider()
    st.subheader("Performance at Selected Threshold")

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy:.2%}")
    cols[1].metric("Precision", f"{precision:.2%}")
    cols[2].metric("Recall", f"{recall:.2%}")
    cols[3].metric("F1 Score", f"{f1:.2%}")

    cols = st.columns(3)
    cols[0].metric("False Positive Rate", f"{fpr:.2%}")
    cols[1].metric("Orders Flagged", f"{flagged:,}")
    cols[2].metric("Low-Satisfaction Orders Missed", f"{missed:,}")

    st.divider()
    st.subheader("Actual vs Predicted Results")

    threshold_cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Not Low Satisfaction", "Actual: Low Satisfaction"],
        columns=["Predicted: Not Low Satisfaction", "Predicted: Low Satisfaction"]
    )

    fig_cm = px.imshow(
        threshold_cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted outcome",
            "y": "Actual outcome",
            "color": "Number of orders"
        }
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        "True positives are correctly identified low-satisfaction orders. "
        "False positives are incorrectly flagged orders. False negatives "
        "are actual low-satisfaction orders missed by the model."
    )

    st.divider()
    st.subheader("Business Interpretation")

    st.write(
        f"At a threshold of **{threshold:.4f}**, the model flags "
        f"**{flagged:,} orders** for potential low customer satisfaction."
    )
    st.write(
        f"It correctly identifies **{tp:,}** low-satisfaction orders "
        f"but misses **{missed:,}** actual low-satisfaction orders."
    )
    st.write(
        f"Of the flagged orders, **{precision:.2%}** are actually "
        f"low satisfaction. The model identifies **{recall:.2%}** "
        "of all actual low-satisfaction orders."
    )

    if recall > 0.50:
        st.warning(
            "This threshold identifies more than half of actual "
            "low-satisfaction orders, but false alerts should also be considered."
        )
    else:
        st.warning(
            "At this threshold, the model misses more than half of actual "
            "low-satisfaction orders. This limits its usefulness for "
            "identifying dissatisfied customers."
        )

    st.divider()
    st.subheader("How Performance Changes Across Thresholds")

    threshold_values = [round(x / 100, 2) for x in range(10, 96, 5)]
    sensitivity_results = []

    for value in threshold_values:
        temp_predicted = (probabilities >= value).astype(int)
        temp_tp = int(((actual == 1) & (temp_predicted == 1)).sum())
        temp_fp = int(((actual == 0) & (temp_predicted == 1)).sum())
        temp_fn = int(((actual == 1) & (temp_predicted == 0)).sum())

        temp_precision = (
            temp_tp / (temp_tp + temp_fp)
            if (temp_tp + temp_fp) else 0
        )
        temp_recall = (
            temp_tp / (temp_tp + temp_fn)
            if (temp_tp + temp_fn) else 0
        )
        temp_f1 = (
            2 * temp_precision * temp_recall /
            (temp_precision + temp_recall)
            if (temp_precision + temp_recall) else 0
        )

        sensitivity_results.append({
            "Threshold": value,
            "Precision": temp_precision,
            "Recall": temp_recall,
            "F1 Score": temp_f1
        })

    sensitivity_df = pd.DataFrame(sensitivity_results)

    fig_sensitivity = px.line(
        sensitivity_df,
        x="Threshold",
        y=["Precision", "Recall", "F1 Score"],
        markers=True,
        title="Precision, Recall and F1 Score by Threshold"
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
        "These are validation results and should not be confused with "
        "the final test performance. Threshold adjustment does not resolve "
        "the model's underlying predictive limitations."
    )


# ============================================================
# 12. WHAT-IF ANALYSIS PAGE
# ============================================================

elif page == "What-If Analysis":

    st.title("What-If Analysis")

    st.write(
        "Explore how changes in order characteristics affect the "
        "model's predicted probability of low customer satisfaction. "
        "This analysis changes item spending while keeping all other "
        "order characteristics constant."
    )

    st.info(
        "This is an exploratory sensitivity exercise. It demonstrates "
        "how the model responds to hypothetical changes, not what would "
        "necessarily happen to actual customers."
    )

    states, payment_types = get_states_and_payments()

    st.subheader("Baseline Order Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(
            "Customer state",
            states,
            index=states.index("SP") if "SP" in states else 0,
            key="whatif_state"
        )
        spending = st.slider(
            "Item spending (R$)", 0.0, 2000.0, 150.0,
            step=10.0, key="whatif_spending"
        )
        freight = st.slider(
            "Freight spending (R$)", 0.0, 500.0, 25.0,
            step=5.0, key="whatif_freight"
        )
        items = st.slider(
            "Total items", 1, 20, 2, key="whatif_items"
        )

    with col2:
        products = st.slider(
            "Distinct products", 1, items, min(2, items),
            key="whatif_products"
        )
        categories = st.slider(
            "Category diversity", 1, products, min(2, products),
            key="whatif_categories"
        )
        payment = st.selectbox(
            "Payment type", payment_types, key="whatif_payment"
        )
        month = st.slider(
            "Purchase month", 1, 12, 5, key="whatif_month"
        )

    baseline_values = {
        "customer_state": state,
        "item_spending": spending,
        "freight_spending": freight,
        "total_items": items,
        "distinct_products": products,
        "category_diversity": categories,
        "preferred_payment_type": payment,
        "purchase_year": 2018,
        "purchase_month": month,
        "purchase_day_of_week": 2,
        "purchase_hour": 14
    }

    try:
        baseline_prediction, baseline_probability = get_prediction(
            baseline_values
        )

        st.divider()
        st.subheader("Baseline Prediction")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Probability", f"{baseline_probability:.2%}")
        col2.metric("Classification Threshold", f"{THRESHOLD:.2%}")

        if baseline_prediction == 1:
            st.warning("Baseline classification: Potential low satisfaction")
        else:
            st.success(
                "Baseline classification: Not classified as low satisfaction"
            )

        st.divider()
        st.subheader("Explore Alternative Spending Scenarios")

        st.write(
            "The following scenarios change only item spending. "
            "All other baseline characteristics remain unchanged."
        )

        spending_values = sorted(set([
            max(0, spending - 100),
            spending,
            spending + 100,
            spending + 200,
            spending + 300
        ]))

        what_if_results = []

        for amount in spending_values:
            modified_values = baseline_values.copy()
            modified_values["item_spending"] = amount
            prediction, probability = get_prediction(modified_values)

            what_if_results.append({
                "Item Spending (R$)": amount,
                "Predicted Probability": probability,
                "Change from Baseline (pp)": (
                    probability - baseline_probability
                ) * 100,
                "Classification": (
                    "Low satisfaction" if prediction == 1
                    else "Not low satisfaction"
                )
            })

        results_df = pd.DataFrame(what_if_results)

        st.subheader("Effect of Item Spending on Predicted Probability")

        fig = px.line(
            results_df,
            x="Item Spending (R$)",
            y="Predicted Probability",
            markers=True,
            title="Predicted Probability Across Spending Scenarios"
        )
        fig.add_vline(
            x=spending,
            line_dash="dash",
            annotation_text="Baseline spending"
        )
        fig.add_hline(
            y=THRESHOLD,
            line_dash="dot",
            annotation_text="Classification threshold"
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scenario Comparison")
        display_results = results_df.copy()
        display_results["Predicted Probability"] = (
            display_results["Predicted Probability"].map(lambda x: f"{x:.2%}")
        )
        display_results["Change from Baseline (pp)"] = (
            display_results["Change from Baseline (pp)"].map(
                lambda x: f"{x:+.2f}"
            )
        )
        st.dataframe(
            display_results, hide_index=True, use_container_width=True
        )

        st.divider()
        st.subheader("Business Interpretation")

        min_probability = results_df["Predicted Probability"].min()
        max_probability = results_df["Predicted Probability"].max()
        probability_range = (max_probability - min_probability) * 100

        st.write(
            f"Across the spending scenarios, the model's predicted "
            f"probability ranges from **{min_probability:.2%}** "
            f"to **{max_probability:.2%}**."
        )
        st.write(
            f"The difference between the highest and lowest predicted "
            f"probabilities is **{probability_range:.2f} percentage points**."
        )
        st.caption(
            "A change in predicted probability reflects the model's response "
            "to the modified input. It does not establish that changing item "
            "spending causes customer satisfaction to change."
        )
        st.warning(
            "The model has limited predictive performance. These hypothetical "
            "results should be interpreted cautiously and should not be treated "
            "as reliable forecasts of individual customer experiences."
        )

    except Exception as error:
        st.error(f"What-If analysis failed: {error}")


# ============================================================
# 13. MODEL EXPLAINABILITY PAGE
# ============================================================

elif page == "Model Explainability":

    st.title("Model Explainability")

    st.write(
        "Explore which order characteristics contribute to "
        "the model's low-satisfaction predictions."
    )

    st.info(
        "SHAP explains how individual features contribute to model outputs. "
        "These contributions describe model behaviour, not causal relationships."
    )

    if st.button("Generate SHAP Explanations", type="primary"):

        with st.spinner("Calculating SHAP values..."):

            try:
                df_shap = load_dataset().copy()

                df_shap["order_purchase_timestamp"] = pd.to_datetime(
                    df_shap["order_purchase_timestamp"],
                    errors="coerce"
                )

                df_shap["purchase_year"] = (
                    df_shap["order_purchase_timestamp"].dt.year
                )
                df_shap["purchase_month"] = (
                    df_shap["order_purchase_timestamp"].dt.month
                )
                df_shap["purchase_day_of_week"] = (
                    df_shap["order_purchase_timestamp"].dt.dayofweek
                )
                df_shap["purchase_hour"] = (
                    df_shap["order_purchase_timestamp"].dt.hour
                )

                X_shap = df_shap[FEATURES].dropna()
                X_shap = X_shap.sample(
                    n=min(300, len(X_shap)),
                    random_state=42
                )

                preprocessor = model.named_steps["preprocessor"]
                classifier = model.named_steps["classifier"]

                X_transformed = preprocessor.transform(X_shap)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                transformed_features = preprocessor.get_feature_names_out()
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)

                if isinstance(shap_values, list):
                    positive_shap = shap_values[1]
                elif len(shap_values.shape) == 3:
                    positive_shap = shap_values[:, :, 1]
                else:
                    positive_shap = shap_values

                importance = pd.Series(
                    abs(positive_shap).mean(axis=0),
                    index=transformed_features
                ).sort_values(ascending=False)

                positive_index = list(model.classes_).index(POSITIVE_CLASS)
                probabilities = model.predict_proba(X_shap)[:, positive_index]

                st.session_state["shap_results"] = {
                    "X_shap": X_shap,
                    "positive_shap": positive_shap,
                    "transformed_features": transformed_features,
                    "importance": importance,
                    "probabilities": probabilities
                }

                st.success(
                    f"SHAP explanations generated for {len(X_shap):,} sampled orders."
                )

            except Exception as error:
                st.error(f"Unable to generate SHAP explanations: {error}")

    if "shap_results" in st.session_state:

        results = st.session_state["shap_results"]
        X_shap = results["X_shap"]
        positive_shap = results["positive_shap"]
        transformed_features = results["transformed_features"]
        importance = results["importance"]
        probabilities = results["probabilities"]

        st.divider()
        st.subheader("1. Global Feature Importance")

        st.write(
            "This chart ranks features according to their average "
            "absolute SHAP contribution across the sampled orders."
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

        st.caption(
            "Higher values indicate greater average influence on model outputs "
            "within the 300 sampled orders. Absolute values show magnitude, "
            "not direction."
        )

        st.divider()
        st.subheader("2. Individual Prediction Explanation")

        selected_order = st.slider(
            "Select an order from the SHAP sample",
            min_value=1,
            max_value=len(X_shap),
            value=1
        )

        row_index = selected_order - 1
        probability = probabilities[row_index]
        prediction = (
            "Low satisfaction predicted"
            if probability >= THRESHOLD
            else "Low satisfaction not predicted"
        )

        st.write(f"**Selected sample order:** {selected_order}")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Probability", f"{probability:.2%}")
        col2.metric("Classification", prediction)

        st.caption(
            f"Classification threshold: {THRESHOLD:.2%}. "
            "This is a model prediction, not a confirmed customer outcome."
        )

        individual_shap = pd.Series(
            positive_shap[row_index],
            index=transformed_features
        )

        individual_shap = individual_shap.reindex(
            individual_shap.abs().sort_values(
                ascending=False
            ).head(10).sort_values().index
        )

        individual_labels = [
            name.replace("numerical__", "").replace("categorical__", "")
            for name in individual_shap.index
        ]

        individual_fig = px.bar(
            x=individual_shap.values,
            y=individual_labels,
            orientation="h",
            color=[
                "Pushes towards low satisfaction" if value > 0
                else "Pushes away from low satisfaction"
                for value in individual_shap.values
            ],
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
        individual_fig.add_vline(
            x=0, line_dash="dash", line_color="gray"
        )
        st.plotly_chart(individual_fig, use_container_width=True)

        st.divider()
        st.subheader("3. Selected Order Characteristics")
        st.dataframe(
            X_shap.iloc[[row_index]],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("4. How to Interpret These Results")
        st.markdown(
            """
            - **Positive SHAP contribution:** Pushes the model output
              towards the low-satisfaction class.
            - **Negative SHAP contribution:** Pushes the model output
              away from the low-satisfaction class.
            - **Larger absolute values:** Indicate stronger contributions
              to the model output.

            **Important limitations**
            - The global explanation uses a reproducible sample of
              300 delivered orders, not the complete dataset.
            - SHAP values describe model behaviour and do not prove
              that a feature causes customer dissatisfaction.
            - Customer state is a geographical feature, not a direct
              measure of a protected demographic characteristic.
            - Explanations do not overcome weak test performance.
              Predictions require human oversight.
            """
        )

    else:
        st.warning(
            "Select 'Generate SHAP Explanations' to calculate and "
            "explore the model explanations."
        )


# ============================================================
# 14. ETHICAL COMPLIANCE AND FAIRNESS PAGE
# ============================================================

elif page == "Ethical Compliance":

    st.title("Ethical Compliance and Fairness")

    st.write(
        "This section examines geographic differences in model "
        "performance and evaluates an experimental fairness "
        "mitigation approach."
    )

    st.info(
        "Customer state is a geographic variable, not a protected "
        "demographic attribute. These results are exploratory and "
        "do not establish demographic discrimination."
    )

    # Test-set geographic results across the 16 groups.
    fairness_data = pd.DataFrame([
        ["BA", 634, 10.73, 34.54, 0.95, 45.59, 33.22, 2.94, 0.71],
        ["CE", 215, 7.91, 27.44, 1.40, 17.65, 28.28, 5.88, 1.01],
        ["DF", 466, 9.66, 8.58, 1.07, 20.00, 7.36, 2.22, 0.95],
        ["ES", 358, 10.06, 7.26, 0.84, 30.56, 4.66, 2.78, 0.62],
        ["GO", 346, 10.69, 9.83, 1.73, 16.22, 9.06, 2.70, 1.62],
        ["MA", 115, 16.52, 33.91, 0.00, 26.32, 35.42, 0.00, 0.00],
        ["MG", 2117, 9.26, 8.69, 0.47, 23.98, 7.13, 1.53, 0.36],
        ["MT", 150, 8.67, 15.33, 1.33, 53.85, 11.68, 7.69, 0.73],
        ["Other", 710, 10.42, 21.41, 0.85, 32.43, 20.13, 2.70, 0.63],
        ["PA", 159, 7.55, 61.64, 0.63, 58.33, 61.90, 0.00, 0.68],
        ["PE", 322, 9.01, 56.83, 0.93, 44.83, 58.02, 6.90, 0.34],
        ["PR", 969, 8.26, 10.11, 0.52, 23.75, 8.89, 2.50, 0.34],
        ["RJ", 2186, 11.12, 26.94, 0.41, 39.09, 25.42, 0.82, 0.36],
        ["RS", 933, 8.68, 12.33, 0.54, 33.33, 10.33, 1.23, 0.47],
        ["SC", 618, 8.74, 11.17, 0.32, 24.07, 9.93, 0.00, 0.35],
        ["SP", 8869, 9.58, 7.79, 0.58, 18.82, 6.62, 2.35, 0.39]
    ], columns=[
        "State Group", "Test Orders", "Actual Low Satisfaction (%)",
        "Original Prediction Rate (%)", "Mitigated Prediction Rate (%)",
        "Original TPR (%)", "Original FPR (%)",
        "Mitigated TPR (%)", "Mitigated FPR (%)"
    ])

    st.subheader("Original vs Fairness-Mitigated Model")

    st.write(
        "The Equalized Odds experiment used Fairlearn's ThresholdOptimizer "
        "as a post-processing approach. It was fitted using validation data "
        "and evaluated on the separate test dataset."
    )

    model_comparison = pd.DataFrame([
        ["Original Random Forest", 81.64, 18.21, 25.73, 21.33],
        ["Equalized Odds Mitigation", 90.12, 33.33, 2.10, 3.96]
    ], columns=[
        "Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"
    ])

    st.dataframe(
        model_comparison.style.format({
            "Accuracy (%)": "{:.2f}",
            "Precision (%)": "{:.2f}",
            "Recall (%)": "{:.2f}",
            "F1 Score (%)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Both rows show test-set results. The original Random Forest remains "
        "the model used by interactive prediction features. Mitigated results "
        "are an experimental comparison."
    )

    cols = st.columns(4)
    cols[0].metric("Original Recall", "25.73%")
    cols[1].metric("Mitigated Recall", "2.10%")
    cols[2].metric("Original F1 Score", "21.33%")
    cols[3].metric("Mitigated F1 Score", "3.96%")

    st.warning(
        "Although the mitigated model achieved higher test accuracy and "
        "precision, its recall fell to 2.10%. It identified very few actual "
        "low-satisfaction orders. Its high accuracy is not evidence of "
        "better overall usefulness because the test dataset is imbalanced."
    )

    st.divider()
    st.subheader("Geographic Fairness Gaps")

    fairness_gaps = pd.DataFrame([
        ["True Positive Rate Difference", 42.12, 7.69, 34.43],
        ["False Positive Rate Difference", 57.25, 1.62, 55.63]
    ], columns=[
        "Measure", "Original Model (%)", "Mitigated Model (%)",
        "Reduction (percentage points)"
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("TPR Gap: Original Model", "42.12 pp")
        st.metric("TPR Gap: Mitigated Model", "7.69 pp", delta="-34.43 pp")
    with col2:
        st.metric("FPR Gap: Original Model", "57.25 pp")
        st.metric("FPR Gap: Mitigated Model", "1.62 pp", delta="-55.63 pp")

    st.dataframe(fairness_gaps, hide_index=True, use_container_width=True)

    st.caption(
        "Gaps represent the difference between the highest and lowest "
        "group-level rates across the 16 test groups."
    )

    st.divider()
    st.subheader("Positive Prediction Rates by State Group")

    prediction_rates = fairness_data.melt(
        id_vars=["State Group"],
        value_vars=[
            "Original Prediction Rate (%)",
            "Mitigated Prediction Rate (%)"
        ],
        var_name="Model",
        value_name="Positive Prediction Rate (%)"
    )

    prediction_fig = px.bar(
        prediction_rates,
        x="State Group",
        y="Positive Prediction Rate (%)",
        color="Model",
        barmode="group",
        title="Share of Test Orders Predicted as Low Satisfaction"
    )
    prediction_fig.update_layout(height=480)
    st.plotly_chart(prediction_fig, use_container_width=True)

    st.caption(
        "The mitigated model's positive prediction rate ranged from "
        "0.00% to 1.73% across the 16 groups. This very low rate helps "
        "explain its low overall recall."
    )

    st.divider()
    st.subheader("True Positive Rate by State Group")

    tpr_data = fairness_data.melt(
        id_vars=["State Group"],
        value_vars=["Original TPR (%)", "Mitigated TPR (%)"],
        var_name="Model",
        value_name="True Positive Rate (%)"
    )

    tpr_fig = px.bar(
        tpr_data,
        x="State Group",
        y="True Positive Rate (%)",
        color="Model",
        barmode="group",
        title="Share of Actual Low-Satisfaction Orders Identified"
    )
    tpr_fig.update_layout(height=480)
    st.plotly_chart(tpr_fig, use_container_width=True)

    st.divider()
    st.subheader("False Positive Rate by State Group")

    fpr_data = fairness_data.melt(
        id_vars=["State Group"],
        value_vars=["Original FPR (%)", "Mitigated FPR (%)"],
        var_name="Model",
        value_name="False Positive Rate (%)"
    )

    fpr_fig = px.bar(
        fpr_data,
        x="State Group",
        y="False Positive Rate (%)",
        color="Model",
        barmode="group",
        title="Share of Actual Non-Low-Satisfaction Orders Incorrectly Flagged"
    )
    fpr_fig.update_layout(height=480)
    st.plotly_chart(fpr_fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Geographic Results")

    st.dataframe(
        fairness_data.style.format({
            "Actual Low Satisfaction (%)": "{:.2f}",
            "Original Prediction Rate (%)": "{:.2f}",
            "Mitigated Prediction Rate (%)": "{:.2f}",
            "Original TPR (%)": "{:.2f}",
            "Original FPR (%)": "{:.2f}",
            "Mitigated TPR (%)": "{:.2f}",
            "Mitigated FPR (%)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "States with at least 500 training records were retained separately; "
        "states below that threshold were combined into Other. Test-set "
        "group counts are displayed in the table."
    )

    st.divider()
    st.subheader("What the Fairness Experiment Shows")

    st.markdown(
        """
        **Observed improvement**
        - Equalized Odds substantially reduced the observed differences
          in true positive and false positive rates across geographic groups.
        - The true positive rate gap declined from 42.12 to 7.69 percentage points.
        - The false positive rate gap declined from 57.25 to 1.62 percentage points.

        **Important performance trade-off**
        - The mitigated model's recall fell from 25.73% to 2.10%.
        - Its F1 Score fell from 21.33% to 3.96%.
        - Its positive prediction rates were very low across all groups.
        - Consequently, it missed almost all actual low-satisfaction test orders.
        """
    )

    st.error(
        "The Equalized Odds experiment is not selected for deployment. "
        "Reducing geographic error-rate differences came with an "
        "unacceptable loss of low-satisfaction detection. The original "
        "Random Forest remains the active model in this dashboard."
    )

    st.divider()
    st.subheader("Ethical Compliance Assessment")

    st.markdown(
        """
        **Fairness and representation**
        - Geographic fairness comparisons were completed on the final test dataset.
        - Equalized Odds post-processing was experimentally implemented and evaluated.
        - Mitigation was not adopted because of the substantial recall reduction.
        - The dataset does not provide sufficient protected demographic information
          for a formal assessment across demographic groups.
        - Geographic comparisons do not prove discrimination or establish
          complete fairness.

        **Privacy and data protection**
        - The project uses the public Olist Brazilian E-Commerce dataset.
        - This page presents aggregated geographic results rather than
          individual customer records.
        - Operational use would require appropriate access restrictions,
          data minimisation and safeguards against re-identification.

        **Transparency and explainability**
        - The dashboard reports model performance and SHAP explanations.
        - Predictions are estimates, not confirmed customer outcomes.
        - SHAP contributions describe model behaviour and do not establish causality.
        """
    )

    st.divider()
    st.subheader("Further Fairness and Risk Controls")

    st.markdown(
        """
        Before operational use, Olist would need to:
        1. Investigate the geographic differences observed in the original model.
        2. Evaluate whether differences persist across additional time periods
           and larger group samples.
        3. Assess protected-group fairness only where appropriate, lawful
           and supported by suitable data.
        4. Compare the benefits and performance costs of alternative
           mitigation methods.
        5. Monitor model performance and fairness indicators over time.
        6. Retain human review rather than using predictions as automatic
           decisions about customers.
        """
    )

    st.caption(
        "These are exploratory test-set findings for 16 geographic groups. "
        "They do not establish that the model is fair or production-ready."
    )


# ============================================================
# 15. MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.title("Model Information and Transparency")

    st.write(
        "This page explains how the model works, what information it uses, "
        "how well it performed and the limitations that should guide its use."
    )

    st.subheader("Model Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model name:** {MODEL_NAME}")
        st.write(f"**Algorithm:** {MODEL_TYPE}")
        st.write("**Prediction unit:** Individual order")
    with col2:
        st.write(f"**Classification threshold:** {THRESHOLD:.4f}")
        st.write(f"**Positive outcome:** {TARGET_DEFINITION}")
        st.write("**Evaluation:** Chronological test dataset")

    st.write(
        "The model estimates whether an order is likely to receive "
        "an average review score of 2 or below. It uses order information "
        "to identify patterns associated with low customer satisfaction."
    )

    st.caption(
        "A prediction at or above the classification threshold is classified "
        "as low satisfaction predicted. A prediction below the threshold "
        "is classified as low satisfaction not predicted."
    )

    st.divider()
    st.subheader("Final Test Performance")

    st.write(
        "The following results were obtained from 19,167 orders in the "
        f"final chronological test dataset at threshold {THRESHOLD:.4f}."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "81.64%")
        st.metric("Precision", "18.21%")
    with col2:
        st.metric("Recall", "25.73%")
        st.metric("F1 Score", "21.33%")
    with col3:
        st.metric("ROC-AUC", "58.17%")
        st.metric("PR-AUC", "16.98%")

    st.warning(
        "The model has limited ability to identify low-satisfaction orders. "
        "Its recall is 25.73%, meaning it misses most actual low-satisfaction "
        "cases. Accuracy is affected by class imbalance and should not be "
        "considered on its own."
    )

    st.divider()
    st.subheader("Input Variables")

    st.write(
        "The model uses 11 order-level characteristics, grouped "
        "into four business categories:"
    )

    st.markdown(
        """
        **Customer location**
        - Customer state

        **Order and spending characteristics**
        - Item spending
        - Freight spending
        - Total items
        - Distinct products
        - Category diversity

        **Payment preference**
        - Preferred payment type

        **Purchase timing**
        - Purchase year
        - Purchase month
        - Purchase day of the week
        - Purchase hour
        """
    )

    st.divider()
    st.subheader("Feature Engineering and Model Selection")

    st.write(
        "Additional order-level features were developed using information "
        "from the Olist order items and product tables to investigate "
        "whether product and seller characteristics could improve prediction."
    )

    feature_comparison = pd.DataFrame([
        ["Original Random Forest", 11, 0.5047, 25.00, 45.44, 32.25, 68.48, 28.91],
        ["17-Feature Random Forest", 17, 0.5400, 19.40, 49.62, 27.90, 63.08, 23.28],
        ["13-Feature Random Forest", 13, 0.5450, 19.24, 49.79, 27.76, 62.88, 22.17]
    ], columns=[
        "Model", "Features", "Best Validation Threshold",
        "Validation Precision (%)", "Validation Recall (%)",
        "Validation F1 (%)", "Validation ROC-AUC (%)",
        "Validation PR-AUC (%)"
    ])

    st.dataframe(
        feature_comparison.style.format({
            "Best Validation Threshold": "{:.4f}",
            "Validation Precision (%)": "{:.2f}",
            "Validation Recall (%)": "{:.2f}",
            "Validation F1 (%)": "{:.2f}",
            "Validation ROC-AUC (%)": "{:.2f}",
            "Validation PR-AUC (%)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These are validation results used for model comparison and selection. "
        "They are not final test results. The best validation F1 threshold "
        "is shown for each model."
    )

    st.markdown(
        """
        **Additional features investigated**

        The 17-feature experiment added six features:
        - Number of distinct sellers
        - Variation in item prices
        - Average product weight
        - Average product length
        - Average product width
        - Freight-to-item spending ratio

        A reduced 13-feature experiment retained average product weight
        and the freight-to-item spending ratio alongside the original
        11 features.

        **Model selection decision**

        The engineered models achieved somewhat higher validation recall
        at their best F1 thresholds, but their F1, ROC-AUC and PR-AUC
        results were lower than those of the original model. The original
        11-feature Random Forest was therefore retained as the primary model.
        """
    )

    st.info(
        "The live dashboard continues to use the original 11-feature "
        "Random Forest. The engineered models are validation experiments "
        "and are not used for interactive predictions."
    )

    st.divider()
    st.subheader("Important Limitations")

    st.markdown(
        """
        1. **Limited predictive performance:** The model achieved relatively
           low recall and ROC-AUC on the final test dataset.

        2. **Class imbalance:** Low-satisfaction orders represent a minority
           of observations, affecting how performance should be interpreted.

        3. **Geographic representation:** Customer state is included as an
           input variable. Differences across states require further
           investigation and do not independently establish unfairness.

        4. **Limited demographic information:** The dataset does not provide
           sufficient protected demographic information for formal
           demographic fairness testing.

        5. **Fairness mitigation trade-off:** An Equalized Odds post-processing
           experiment was implemented and evaluated. Although it reduced
           geographic differences in error rates, recall fell to 2.10%.
           It was therefore not selected for the live prediction model.

        6. **Prediction versus causation:** Predictions identify statistical
           patterns. They do not establish why a customer may be dissatisfied.

        7. **Prediction-time availability:** The availability of all input
           variables immediately after an order is placed requires further
           operational verification.
        """
    )

    st.divider()
    st.subheader("Responsible Use Statement")

    st.info(
        """
        This model is an exploratory decision-support tool. Its outputs
        should not be treated as confirmed customer outcomes or used as
        the sole basis for customer-level decisions.

        Human review, further validation, appropriate data protection
        and ongoing performance and fairness monitoring are necessary
        before operational use.
        """
    )

    st.divider()
    st.subheader("Transparency and Accountability")

    st.markdown(
        """
        - **Data basis:** Olist Brazilian E-Commerce Public Dataset.
        - **Model type:** Original 11-feature Random Forest classification.
        - **Feature experiments:** 13-feature and 17-feature alternatives
          evaluated using validation data.
        - **Explainability:** SHAP-based global and individual explanations.
        - **Fairness review:** Geographic test-set comparisons and an
          experimental Equalized Odds mitigation assessment.
        - **Model selection:** Original Random Forest retained because it
          performed better on the main validation metrics.
        - **Oversight:** Predictions require human interpretation.
        - **Project status:** Academic analytics prototype, not a
          production-approved customer decision system.
        """)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "Olist Brazilian E-Commerce Analytics Project | "
    "Machine Learning and Customer Experience | Academic project"
)
