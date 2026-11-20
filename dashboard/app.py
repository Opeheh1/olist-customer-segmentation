
from pathlib import Path
import json

import joblib
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Olist Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# app.py is inside the dashboard folder.
# The project root is therefore one folder above it.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "olist_low_satisfaction_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivered_order_level.csv"


# ============================================================
# 3. MODEL INPUT FEATURES
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
# 4. LOAD MODEL, CONFIGURATION AND DATA
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


try:
    model = load_model()
    model_config = load_config()
    dataset = load_dataset()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"An error occurred while loading the project files: {error}")
    st.stop()


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = model_config.get(
    "model_name", "Olist Low Satisfaction Prediction"
)

MODEL_TYPE = model_config.get(
    "model_type", "Tuned Random Forest"
)

THRESHOLD = float(
    model_config.get("classification_threshold", 0.5453)
)

POSITIVE_CLASS = model_config.get("positive_class", 1)

TARGET_DEFINITION = model_config.get(
    "positive_class_definition",
    "Average review score <= 2"
)


# ============================================================
# 6. VERIFIED MODULE 4 TEST RESULTS
# ============================================================

# These are the previously evaluated final test-set results.
# They are not recalculated from the full delivered-order dataset.

TEST_METRICS = {
    "Accuracy": 86.42,
    "Precision": 24.61,
    "Recall": 19.58,
    "F1 Score": 21.81,
    "ROC-AUC": 58.17,
    "PR-AUC": 16.98
}

CONFUSION_MATRIX = pd.DataFrame(
    [[16201, 1112], [1491, 363]],
    index=["Actual: Not low satisfaction", "Actual: Low satisfaction"],
    columns=["Predicted: Not low satisfaction", "Predicted: Low satisfaction"]
)


# ============================================================
# 7. SIDEBAR
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
# 8. HELPER FUNCTIONS
# ============================================================

def prepare_model_input(values):
    """
    Convert user inputs into the DataFrame format
    expected by the trained model.
    """

    input_df = pd.DataFrame([values], columns=FEATURES)

    for column in CATEGORICAL_FEATURES:
        input_df[column] = input_df[column].astype(str)

    return input_df


def get_prediction(values):
    """
    Generate a prediction and probability using
    the saved Module 4 model.
    """

    input_df = prepare_model_input(values)

    probabilities = model.predict_proba(input_df)[0]

    # Locate the probability belonging to the positive class.
    positive_index = list(model.classes_).index(POSITIVE_CLASS)

    probability = float(probabilities[positive_index])

    prediction = int(probability >= THRESHOLD)

    return prediction, probability


def display_prediction(prediction, probability):
    """
    Display prediction results in plain language.
    """

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



# ============================================================
# 9. OVERVIEW PAGE
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

    # Key project indicators

    st.subheader("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Delivered Orders", f"{len(dataset):,}")
    col2.metric("Model Algorithm", "Random Forest")
    col3.metric("Test Accuracy", "86.42%")
    col4.metric("Test Recall", "19.58%")

    st.caption(
        "Performance figures are from the final Module 4 test dataset."
    )

    st.divider()

    # Business problem

    st.subheader("The Business Problem")

    st.write(
        """
        Olist connects customers with sellers through an online
        marketplace. Customer satisfaction is important to the
        marketplace's ability to maintain customer relationships
        and support long-term business value.

        However, identifying potentially dissatisfied customers
        before or during customer experience investigations can
        be challenging.

        This project explores whether order characteristics and
        purchasing patterns can help identify orders associated
        with low customer review scores.
        """
    )

    # Decision support

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
            - Are there differences in prediction performance across states?
            """
        )

    st.divider()

    # How the model works

    st.subheader("How the Analytical Model Works")

    st.write(
        """
        The model uses 11 order-level characteristics, including
        customer state, item spending, freight spending, product
        information, payment type and purchase timing.

        It estimates the probability that an order will receive
        an average review score of 2 or below.
        """
    )

    st.info(
        "The original project vision focuses on customer segmentation. "
        "Low-satisfaction prediction extends that work by examining "
        "customer experience risks at order level."
    )

    st.divider()

    # Dashboard navigation

    st.subheader("Explore the Dashboard")

    navigation = [
        (
            "Model Performance",
            "Review test-set metrics and classification errors."
        ),
        (
            "Interactive Predictions",
            "Enter order characteristics and generate a prediction."
        ),
        (
            "Sample Predictions",
            "Examine actual examples of correct and incorrect predictions."
        ),
        (
            "Threshold Analysis",
            "Explore how changing the classification threshold "
            "affects model performance."
        ),
        (
            "What-If Analysis",
            "Explore how changing order characteristics affects predictions."
        ),
        (
            "Model Explainability",
            "Examine the features influencing model predictions using SHAP."
        ),
        (
            "Ethical Compliance",
            "Review geographic performance differences and ethical limitations."
        ),
        (
            "Model Information",
            "Review model inputs, limitations and responsible-use conditions."
        )
    ]

    for title, description in navigation:

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    st.divider()

    # Model limitation notice

    st.subheader("Important Model Limitation")

    st.warning(
        """
        Although the model achieved 86.42% accuracy on the final test set,
        its recall was only 19.58% and ROC-AUC was 58.17%.

        This indicates limited ability to identify actual low-satisfaction
        orders. The model is therefore exploratory and should not be
        treated as a production-ready customer alert system or used
        as the sole basis for customer decisions.
        """
    )



# ============================================================
# 10. MODEL PERFORMANCE PAGE
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

    # Performance indicators

    metric_columns = st.columns(3)

    for index, (metric, value) in enumerate(TEST_METRICS.items()):
        with metric_columns[index % 3]:
            st.metric(metric, f"{value:.2f}%")

    st.divider()

    # Confusion matrix

    st.subheader("Actual vs Predicted Results")

    st.write(
        "The confusion matrix shows how many orders the model "
        "classified correctly and incorrectly."
    )

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

        - **16,201 True Negatives:** Correctly identified orders that
          did not receive low satisfaction scores.
        - **363 True Positives:** Correctly identified low-satisfaction orders.
        - **1,112 False Positives:** Orders flagged as low satisfaction
          that did not have a low satisfaction outcome.
        - **1,491 False Negatives:** Actual low-satisfaction orders
          that the model failed to identify.
        """
    )

    st.divider()

    # Metric interpretation

    st.subheader("Understanding the Performance Metrics")

    st.markdown(
        """
        - **Accuracy (86.42%):** Percentage of all orders classified correctly.
        - **Precision (24.61%):** Of the orders flagged as low satisfaction,
          approximately one in four actually had low satisfaction.
        - **Recall (19.58%):** The model identified only about one in five
          actual low-satisfaction orders.
        - **F1 Score (21.81%):** A combined measure of precision and recall.
        - **ROC-AUC (58.17%):** Indicates limited ability to distinguish
          between low- and non-low-satisfaction orders.
        - **PR-AUC (16.98%):** Measures precision-recall performance,
          particularly relevant when the positive class is relatively rare.
        """
    )

    st.divider()

    # Class imbalance

    st.subheader("Why Accuracy Can Be Misleading")

    st.write(
        "Only 9.67% of the test orders had low satisfaction outcomes. "
        "This means the dataset is imbalanced, with substantially more "
        "non-low-satisfaction orders than low-satisfaction orders."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Actual low-satisfaction orders", "1,854", "9.67%")

    with col2:
        st.metric("Actual non-low-satisfaction orders", "17,313", "90.33%")

    st.warning(
        "A model that simply classified every order as non-low-satisfaction "
        "would achieve 90.33% accuracy, despite identifying no low-satisfaction "
        "orders. The current model achieves 86.42% accuracy and detects "
        "19.58% of actual low-satisfaction orders. Therefore, accuracy alone "
        "does not demonstrate useful predictive performance."
    )

    st.divider()

    # Business interpretation

    st.subheader("Business Interpretation")

    st.markdown(
        """
        The model demonstrates limited ability to identify orders that may
        result in low customer satisfaction. Although it identifies some
        problematic orders, it misses most actual low-satisfaction cases
        and produces a considerable number of false alerts.

        From a business perspective, relying on this model alone could
        result in missed opportunities to investigate customer complaints
        and unnecessary follow-up on orders that would not receive low
        satisfaction scores.
        """
    )

    st.error(
        "**Model limitation:** The current test results do not support "
        "using this model as a standalone operational alert system. "
        "Further investigation, improvement and evaluation are needed "
        "before operational use."
    )


# ============================================================
# 11. INTERACTIVE PREDICTIONS PAGE
# ============================================================


elif page == "Interactive Predictions":

    st.title("Interactive Order Prediction")

    st.write(
        "Explore how different order characteristics relate to "
        "the estimated likelihood of receiving a low customer "
        "satisfaction score."
    )

    st.info(
        "Enter hypothetical order details and select Generate Prediction "
        "to see the model's estimate. The prediction is exploratory "
        "and should not be treated as a confirmed customer outcome."
    )

    # Get available categories from the actual dataset.

    states = sorted(
        dataset["customer_state"].dropna().astype(str).unique().tolist()
    )

    payment_types = sorted(
        dataset["preferred_payment_type"].dropna().astype(str).unique().tolist()
    )

    st.subheader("Order Characteristics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Customer and Purchase Details**")

        customer_state = st.selectbox(
            "Customer state",
            states,
            index=states.index("SP") if "SP" in states else 0,
            help="Select the Brazilian state associated with the customer."
        )

        item_spending = st.number_input(
            "Item spending (R$)",
            min_value=0.0,
            max_value=10000.0,
            value=150.0,
            step=10.0,
            help="Total spending on items in the order."
        )

        freight_spending = st.number_input(
            "Freight spending (R$)",
            min_value=0.0,
            max_value=3000.0,
            value=25.0,
            step=5.0,
            help="Total freight cost associated with the order."
        )

        total_items = st.number_input(
            "Total items",
            min_value=1,
            max_value=100,
            value=2,
            step=1
        )

        distinct_products = st.number_input(
            "Distinct products",
            min_value=1,
            max_value=100,
            value=2,
            step=1
        )

        category_diversity = st.number_input(
            "Product category diversity",
            min_value=1,
            max_value=100,
            value=2,
            step=1,
            help="Number of different product categories in the order."
        )

    with col2:

        st.markdown("**Payment and Purchase Timing**")

        preferred_payment_type = st.selectbox(
            "Preferred payment type",
            payment_types
        )

        purchase_year = st.selectbox(
            "Purchase year",
            [2016, 2017, 2018],
            index=2
        )

        purchase_month = st.slider(
            "Purchase month",
            1,
            12,
            5
        )

        days = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        purchase_day_of_week = st.selectbox(
            "Purchase day of week",
            list(range(7)),
            index=2,
            format_func=lambda x: days[x]
        )

        purchase_hour = st.slider(
            "Purchase hour",
            0,
            23,
            14,
            help="Hour of purchase using the 24-hour clock."
        )

    st.divider()

    # Validate the order inputs.

    if distinct_products > total_items:

        st.error("Distinct products cannot exceed total items.")

    elif category_diversity > distinct_products:

        st.error(
            "Product category diversity cannot exceed "
            "the number of distinct products."
        )

    else:

        if st.button("Generate Prediction", type="primary"):

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
                    "This is a model-generated estimate based on the "
                    "entered order characteristics. It does not establish "
                    "that a customer will actually give a low review score."
                )

                st.warning(
                    "The model has limited test performance, including "
                    "19.58% recall. Its predictions should be interpreted "
                    "with caution and should not be used as the sole basis "
                    "for customer-related decisions."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")



# ============================================================
# 12. WHAT-IF ANALYSIS PAGE
# ============================================================


elif page == "What-If Analysis":

    st.title("What-If Analysis")

    st.write(
        "Explore how changes in order characteristics affect the "
        "model's predicted probability of low customer satisfaction. "
        "The analysis changes item spending while keeping all other "
        "order characteristics constant."
    )

    st.info(
        "This is an exploratory model sensitivity exercise. "
        "It demonstrates how the model responds to hypothetical "
        "changes, not what would necessarily happen to actual customers."
    )

    states = sorted(
        dataset["customer_state"].dropna().astype(str).unique().tolist()
    )

    payment_types = sorted(
        dataset["preferred_payment_type"].dropna().astype(str).unique().tolist()
    )

    # Baseline order characteristics

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
            "Item spending (R$)",
            0.0,
            2000.0,
            150.0,
            step=10.0,
            key="whatif_spending"
        )

        freight = st.slider(
            "Freight spending (R$)",
            0.0,
            500.0,
            25.0,
            step=5.0,
            key="whatif_freight"
        )

        items = st.slider(
            "Total items",
            1,
            20,
            2,
            key="whatif_items"
        )

    with col2:

        products = st.slider(
            "Distinct products",
            1,
            items,
            min(2, items),
            key="whatif_products"
        )

        categories = st.slider(
            "Category diversity",
            1,
            products,
            min(2, products),
            key="whatif_categories"
        )

        payment = st.selectbox(
            "Payment type",
            payment_types,
            key="whatif_payment"
        )

        month = st.slider(
            "Purchase month",
            1,
            12,
            5,
            key="whatif_month"
        )

    # Build baseline input

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

        # Baseline prediction

        st.subheader("Baseline Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Probability",
                f"{baseline_probability:.2%}"
            )

        with col2:
            st.metric(
                "Classification Threshold",
                f"{THRESHOLD:.2%}"
            )

        if baseline_prediction == 1:
            st.warning(
                "Baseline classification: Potential low satisfaction"
            )
        else:
            st.success(
                "Baseline classification: Not classified as low satisfaction"
            )

        st.divider()

        # Alternative spending scenarios

        st.subheader("Explore Alternative Spending Scenarios")

        st.write(
            "The following scenarios change only item spending. "
            "All other baseline characteristics remain unchanged."
        )

        spending_values = sorted(
            set([
                max(0, spending - 100),
                spending,
                spending + 100,
                spending + 200,
                spending + 300
            ])
        )

        what_if_results = []

        for amount in spending_values:

            modified_values = baseline_values.copy()
            modified_values["item_spending"] = amount

            prediction, probability = get_prediction(modified_values)

            probability_change = (
                probability - baseline_probability
            ) * 100

            what_if_results.append({
                "Item Spending (R$)": amount,
                "Predicted Probability": probability,
                "Change from Baseline (pp)": probability_change,
                "Classification": (
                    "Low satisfaction"
                    if prediction == 1
                    else "Not low satisfaction"
                )
            })

        results_df = pd.DataFrame(what_if_results)

        # Probability comparison chart

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Scenario comparison table

        st.subheader("Scenario Comparison")

        display_results = results_df.copy()

        display_results["Predicted Probability"] = (
            display_results["Predicted Probability"].map(
                lambda x: f"{x:.2%}"
            )
        )

        display_results["Change from Baseline (pp)"] = (
            display_results["Change from Baseline (pp)"].map(
                lambda x: f"{x:+.2f}"
            )
        )

        st.dataframe(
            display_results,
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        # Business interpretation

        st.subheader("Business Interpretation")

        min_probability = results_df["Predicted Probability"].min()
        max_probability = results_df["Predicted Probability"].max()

        probability_range = (
            max_probability - min_probability
        ) * 100

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
            "A change in predicted probability reflects the model's "
            "response to the modified input. It does not establish that "
            "changing item spending causes customer satisfaction to change."
        )

        st.warning(
            "The model has limited predictive performance. These "
            "hypothetical results should be interpreted cautiously "
            "and should not be treated as reliable forecasts of "
            "individual customer experiences."
        )

    except Exception as error:

        st.error(f"What-If analysis failed: {error}")


# ============================================================
# 13. MODEL INFORMATION PAGE
# ============================================================



elif page == "Sample Predictions":

    st.title("Sample Model Predictions")

    st.write(
        "This page demonstrates how the model classifies individual "
        "orders and illustrates examples of correct predictions "
        "and classification errors."
    )

    st.info(
        "The three examples are actual orders from the Module 4 "
        "chronological validation dataset. The classification "
        "threshold is 54.53%. Low satisfaction is defined as "
        "an average review score of 2 or below."
    )

    # Verified examples from inspect_sample_predictions.py

    examples = pd.DataFrame([
        {
            "Example": "True Positive",
            "Order ID": "ab18ff09f55ca3ccc4dbf5bed2c9036d",
            "State": "MG",
            "Review Score": 1.0,
            "Probability": 0.545467,
            "Actual Outcome": "Low satisfaction",
            "Predicted Outcome": "Low satisfaction",
            "Interpretation": "Correctly identified"
        },
        {
            "Example": "False Negative",
            "Order ID": "0e05c0027e31a0a9cf1074ec1b95c8da",
            "State": "MG",
            "Review Score": 1.0,
            "Probability": 0.544985,
            "Actual Outcome": "Low satisfaction",
            "Predicted Outcome": "Not low satisfaction",
            "Interpretation": "Missed low-satisfaction order"
        },
        {
            "Example": "True Negative",
            "Order ID": "90e1fe92be310b125e91833c44a23970",
            "State": "SC",
            "Review Score": 5.0,
            "Probability": 0.545232,
            "Actual Outcome": "Not low satisfaction",
            "Predicted Outcome": "Not low satisfaction",
            "Interpretation": "Correctly identified"
        }
    ])

    threshold = 0.5453

    # Example selection

    st.subheader("Illustrative Prediction Cases")

    selected_example = st.selectbox(
        "Select an example",
        examples["Example"].tolist()
    )

    example = examples[
        examples["Example"] == selected_example
    ].iloc[0]

    # Prediction outcome

    if selected_example == "True Positive":

        st.success(
            "True Positive: The model correctly identified "
            "a low-satisfaction order."
        )

    elif selected_example == "False Negative":

        st.error(
            "False Negative: The model failed to identify "
            "an actual low-satisfaction order."
        )

    else:

        st.success(
            "True Negative: The model correctly identified "
            "an order without low satisfaction."
        )

    # Prediction indicators

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Probability",
            f"{example['Probability']:.2%}"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.2%}"
        )

    with col3:
        st.metric(
            "Actual Review Score",
            f"{example['Review Score']:.0f}/5"
        )

    st.divider()

    # Actual versus predicted outcome

    st.subheader("Actual vs Predicted Outcome")

    comparison = pd.DataFrame({
        "Measure": ["Actual Outcome", "Model Prediction"],
        "Classification": [
            example["Actual Outcome"],
            example["Predicted Outcome"]
        ]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    # Order characteristics

    st.subheader("Order Characteristics")

    order_details = pd.DataFrame({
        "Characteristic": [
            "Order ID",
            "Customer State",
            "Item Spending (R$)",
            "Freight Spending (R$)",
            "Total Items",
            "Payment Type"
        ],
        "Value": [
            example["Order ID"],
            example["State"],
            {
                "True Positive": "59.90",
                "False Negative": "39.95",
                "True Negative": "49.99"
            }[selected_example],
            {
                "True Positive": "18.30",
                "False Negative": "17.93",
                "True Negative": "18.23"
            }[selected_example],
            "1",
            "Credit card"
        ]
    })

    st.dataframe(
        order_details,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # Business interpretation

    st.subheader("Business Interpretation")

    interpretations = {
        "True Positive": (
            "The customer gave a review score of 1, and the model "
            "correctly classified the order as low satisfaction. "
            "This illustrates a case where the model successfully "
            "identified an order associated with a poor customer experience."
        ),

        "False Negative": (
            "The customer gave a review score of 1, but the model "
            "classified the order as not low satisfaction. This "
            "demonstrates how relying exclusively on model predictions "
            "could cause customer experience teams to overlook "
            "some dissatisfied customers."
        ),

        "True Negative": (
            "The customer gave a review score of 5, and the model "
            "correctly classified the order as not low satisfaction. "
            "This represents a correct negative prediction."
        )
    }

    st.write(interpretations[selected_example])

    # Classification threshold

    st.subheader("Understanding the Classification Threshold")

    st.write(
        "The model classifies an order as low satisfaction when its "
        "predicted probability is equal to or greater than 54.53%. "
        "Probabilities below this threshold are classified as "
        "not low satisfaction."
    )

    probability_difference = (
        example["Probability"] - threshold
    ) * 100

    st.metric(
        "Distance from Classification Threshold",
        f"{probability_difference:+.4f} percentage points"
    )

    st.caption(
        "The examples were deliberately selected close to the "
        "classification threshold to demonstrate how small probability "
        "differences can result in different classifications. "
        "They are not representative of all validation orders."
    )

    st.warning(
        "These individual examples do not establish overall model "
        "reliability. The model's limited validation performance "
        "and the possibility of missed low-satisfaction orders "
        "must be considered before interpreting its predictions."
    )



elif page == "Threshold Analysis":

    st.title("Classification Threshold Analysis")

    st.write(
        "Explore how changing the classification threshold affects "
        "the model's ability to identify orders associated with "
        "low customer satisfaction."
    )

    st.info(
        "This analysis uses 15,333 actual orders from the Module 4 "
        "chronological validation dataset. Changing the threshold "
        "changes the classification decision, not the underlying model."
    )

    # Load saved validation predictions

    threshold_data_path = (
        PROJECT_ROOT / "data" / "processed" / "validation_threshold_data.csv"
    )

    threshold_df = pd.read_csv(threshold_data_path)

    actual = threshold_df["actual"].astype(int)
    probabilities = threshold_df["predicted_probability"]

    total_orders = len(threshold_df)
    actual_low = int((actual == 1).sum())
    actual_not_low = int((actual == 0).sum())
    low_rate = actual_low / total_orders

    # Dataset overview

    st.subheader("Validation Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Low-Satisfaction Orders", f"{actual_low:,}")
    col3.metric("Low-Satisfaction Rate", f"{low_rate:.2%}")

    st.divider()

    # Interactive threshold control

    st.subheader("Adjust the Classification Threshold")

    threshold = st.slider(
        "Select classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5453,
        step=0.0001,
        format="%.4f"
    )

    st.caption(
        "An order is classified as low satisfaction when its predicted "
        "probability is equal to or greater than the selected threshold."
    )

    # Generate predictions

    predicted = (probabilities >= threshold).astype(int)

    # Calculate confusion matrix values

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = tp + tn + fp + fn

    # Calculate performance metrics

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

    # Performance at selected threshold

    st.divider()

    st.subheader("Performance at Selected Threshold")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    col5, col6, col7 = st.columns(3)

    col5.metric("False Positive Rate", f"{fpr:.2%}")
    col6.metric("Orders Flagged", f"{flagged:,}")
    col7.metric("Low-Satisfaction Orders Missed", f"{missed:,}")

    st.divider()

    # Confusion matrix

    st.subheader("Actual vs Predicted Results")

    confusion_matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=[
            "Actual: Not Low Satisfaction",
            "Actual: Low Satisfaction"
        ],
        columns=[
            "Predicted: Not Low Satisfaction",
            "Predicted: Low Satisfaction"
        ]
    )

    fig_cm = px.imshow(
        confusion_matrix,
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
        "True positives: correctly identified low-satisfaction orders. "
        "False positives: orders incorrectly flagged as low satisfaction. "
        "False negatives: actual low-satisfaction orders missed by the model. "
        "True negatives: correctly identified orders without low satisfaction."
    )

    st.divider()

    # Business interpretation

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
            "low-satisfaction orders, but the number of false alerts "
            "should also be considered."
        )

    else:
        st.warning(
            "At this threshold, the model misses more than half of "
            "actual low-satisfaction orders. This limits its usefulness "
            "for identifying dissatisfied customers."
        )

    st.divider()

    # Threshold sensitivity analysis

    st.subheader("How Performance Changes Across Thresholds")

    st.write(
        "The chart below shows how precision, recall and F1 Score "
        "change as the classification threshold increases."
    )

    threshold_values = [
        round(x / 100, 2) for x in range(10, 96, 5)
    ]

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

    st.plotly_chart(
        fig_sensitivity,
        use_container_width=True
    )

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
        "The appropriate threshold depends on the business consequences "
        "of missed low-satisfaction orders versus unnecessary alerts. "
        "These results are validation results and should not be confused "
        "with the final test performance."
    )

    st.warning(
        "Threshold adjustment does not resolve the model's underlying "
        "predictive limitations. The validation analysis is exploratory "
        "and does not establish that a particular threshold is suitable "
        "for operational deployment."
    )


elif page == "Model Explainability":

    st.title("Model Explainability")

    st.write(
        "Explore which order characteristics contribute to "
        "the model's low-satisfaction predictions."
    )

    st.info(
        "SHAP explains how individual features contribute to model "
        "outputs. These contributions show model behaviour, not "
        "causal relationships."
    )

    # Generate and store SHAP explanations
    if st.button("Generate SHAP Explanations", type="primary"):

        with st.spinner("Calculating SHAP values..."):

            try:
                # Load processed order data
                df_shap = load_dataset().copy()

                # Recreate the time-based features used in Module 4
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

                # Select the original model features
                X_shap = df_shap[FEATURES].dropna()

                # Use a reproducible sample of up to 300 orders
                X_shap = X_shap.sample(
                    n=min(300, len(X_shap)),
                    random_state=42
                )

                # Access the trained pipeline components
                preprocessor = model.named_steps["preprocessor"]
                classifier = model.named_steps["classifier"]

                # Apply the existing preprocessing
                X_transformed = preprocessor.transform(X_shap)

                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                transformed_features = (
                    preprocessor.get_feature_names_out()
                )

                # Calculate SHAP values
                explainer = shap.TreeExplainer(classifier)

                shap_values = explainer.shap_values(
                    X_transformed
                )

                # Extract contributions for the low-satisfaction class
                if isinstance(shap_values, list):
                    positive_shap = shap_values[1]

                elif len(shap_values.shape) == 3:
                    positive_shap = shap_values[:, :, 1]

                else:
                    positive_shap = shap_values

                # Calculate global feature importance
                importance = pd.Series(
                    abs(positive_shap).mean(axis=0),
                    index=transformed_features
                ).sort_values(ascending=False)

                # Calculate individual prediction probabilities
                probabilities = model.predict_proba(X_shap)[:, 1]

                # Store results to preserve them during slider interaction
                st.session_state["shap_results"] = {
                    "X_shap": X_shap,
                    "positive_shap": positive_shap,
                    "transformed_features": transformed_features,
                    "importance": importance,
                    "probabilities": probabilities
                }

                st.success(
                    f"SHAP explanations generated for "
                    f"{len(X_shap):,} sampled orders."
                )

            except Exception as e:
                st.error(
                    f"Unable to generate SHAP explanations: {e}"
                )

    # Display stored explanations
    if "shap_results" in st.session_state:

        results = st.session_state["shap_results"]

        X_shap = results["X_shap"]
        positive_shap = results["positive_shap"]
        transformed_features = results["transformed_features"]
        importance = results["importance"]
        probabilities = results["probabilities"]

        # -----------------------------------------
        # 1. GLOBAL FEATURE IMPORTANCE
        # -----------------------------------------

        st.subheader("1. Global Feature Importance")

        st.write(
            "This chart ranks the features according to their average "
            "absolute SHAP contribution across the sampled orders."
        )

        top_importance = importance.head(15).sort_values(
            ascending=True
        )

        global_fig = px.bar(
            x=top_importance.values,
            y=[
                name.replace("numerical__", "")
                    .replace("categorical__", "")
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

        st.plotly_chart(
            global_fig,
            use_container_width=True
        )

        st.caption(
            "Higher values indicate greater average influence on model "
            "outputs within the 300 sampled orders. Absolute values "
            "show contribution magnitude, not direction."
        )

        # -----------------------------------------
        # 2. INDIVIDUAL PREDICTION EXPLANATION
        # -----------------------------------------

        st.subheader("2. Individual Prediction Explanation")

        selected_order = st.slider(
            "Select an order from the SHAP sample",
            min_value=1,
            max_value=len(X_shap),
            value=1
        )

        row_index = selected_order - 1

        probability = probabilities[row_index]
        threshold = 0.5453

        prediction = (
            "Low satisfaction predicted"
            if probability >= threshold
            else "Low satisfaction not predicted"
        )

        st.write(f"**Selected sample order:** {selected_order}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Probability",
                f"{probability:.2%}"
            )

        with col2:
            st.metric(
                "Classification",
                prediction
            )

        st.caption(
            f"Classification threshold: {threshold:.2%}. "
            "This is a model prediction, not a confirmed customer outcome."
        )

        # Extract the selected order's SHAP contributions
        individual_shap = pd.Series(
            positive_shap[row_index],
            index=transformed_features
        )

        # Display the ten largest contributions by magnitude
        individual_shap = individual_shap.reindex(
            individual_shap.abs().sort_values(
                ascending=False
            ).head(10).sort_values().index
        )

        individual_labels = [
            name.replace("numerical__", "")
                .replace("categorical__", "")
            for name in individual_shap.index
        ]

        individual_fig = px.bar(
            x=individual_shap.values,
            y=individual_labels,
            orientation="h",
            color=[
                "Pushes towards low satisfaction"
                if value > 0
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
            x=0,
            line_dash="dash",
            line_color="gray"
        )

        st.plotly_chart(
            individual_fig,
            use_container_width=True
        )

        # -----------------------------------------
        # 3. SELECTED ORDER CHARACTERISTICS
        # -----------------------------------------

        st.subheader("3. Selected Order Characteristics")

        st.dataframe(
            X_shap.iloc[[row_index]],
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------
        # 4. INTERPRETATION AND LIMITATIONS
        # -----------------------------------------

        st.subheader("How to Interpret These Results")

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
            - Explanations do not overcome the model's weak test
              performance. Predictions require human oversight.
            """
        )

    else:
        st.warning(
            "Select 'Generate SHAP Explanations' to calculate and "
            "explore the model explanations."
        )


elif page == "Ethical Compliance":

    st.title("Ethical Compliance and Fairness")
    st.write(
        "This section examines geographic differences in model predictions "
        "and highlights the ethical limitations of the current analysis."
    )

    st.info(
        "Important: Customer state is a geographic variable, not a protected "
        "demographic attribute. These comparisons are exploratory and do not "
        "establish demographic discrimination."
    )

    # --------------------------------------------------
    # 1. Verified Module 4 validation findings
    # --------------------------------------------------

    fairness_data = pd.DataFrame({
        "State": ["SP", "RJ", "PA", "MT"],
        "Orders": [6945, 1814, 109, 139],
        "Positive Prediction Rate": [6.44, 26.41, 38.53, 17.99],
        "Precision": [27.29, 30.69, 52.38, 16.00],
        "Recall": [17.91, 50.52, 70.97, 44.44],
        "False Positive Rate": [5.19, 21.80, 25.64, 16.15]
    })

    # --------------------------------------------------
    # 2. Geographic comparison overview
    # --------------------------------------------------

    st.subheader("Geographic Prediction Differences")

    col1, col2, col3 = st.columns(3)

    prediction_gap = (
        fairness_data["Positive Prediction Rate"].max()
        - fairness_data["Positive Prediction Rate"].min()
    )

    recall_gap = (
        fairness_data["Recall"].max()
        - fairness_data["Recall"].min()
    )

    fpr_gap = (
        fairness_data["False Positive Rate"].max()
        - fairness_data["False Positive Rate"].min()
    )

    with col1:
        st.metric(
            "Prediction Rate Gap",
            f"{prediction_gap:.2f} pp"
        )

    with col2:
        st.metric(
            "Recall Gap",
            f"{recall_gap:.2f} pp"
        )

    with col3:
        st.metric(
            "False Positive Rate Gap",
            f"{fpr_gap:.2f} pp"
        )

    st.caption(
        "Percentage-point gaps represent the difference between the highest "
        "and lowest observed state-level values among the four groups."
    )

    # --------------------------------------------------
    # 3. Positive prediction rates
    # --------------------------------------------------

    st.subheader("Positive Prediction Rate by State")

    prediction_fig = px.bar(
        fairness_data,
        x="State",
        y="Positive Prediction Rate",
        text="Positive Prediction Rate",
        labels={
            "State": "Customer State",
            "Positive Prediction Rate": "Positive Prediction Rate (%)"
        },
        title="Share of Orders Predicted as Low Satisfaction"
    )

    prediction_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    prediction_fig.update_layout(
        yaxis_title="Positive Prediction Rate (%)",
        xaxis_title="Customer State",
        yaxis_range=[
            0,
            fairness_data["Positive Prediction Rate"].max() + 10
        ]
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 4. Recall comparison
    # --------------------------------------------------

    st.subheader("Recall by State")

    recall_fig = px.bar(
        fairness_data,
        x="State",
        y="Recall",
        text="Recall",
        labels={
            "State": "Customer State",
            "Recall": "Recall (%)"
        },
        title="Share of Actual Low-Satisfaction Orders Identified"
    )

    recall_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    recall_fig.update_layout(
        yaxis_title="Recall (%)",
        xaxis_title="Customer State",
        yaxis_range=[0, 85]
    )

    st.plotly_chart(
        recall_fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 5. False positive rate comparison
    # --------------------------------------------------

    st.subheader("False Positive Rate by State")

    fpr_fig = px.bar(
        fairness_data,
        x="State",
        y="False Positive Rate",
        text="False Positive Rate",
        labels={
            "State": "Customer State",
            "False Positive Rate": "False Positive Rate (%)"
        },
        title="Share of Actual Non-Low-Satisfaction Orders Incorrectly Flagged"
    )

    fpr_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fpr_fig.update_layout(
        yaxis_title="False Positive Rate (%)",
        xaxis_title="Customer State",
        yaxis_range=[0, 35]
    )

    st.plotly_chart(
        fpr_fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 6. Detailed geographic results
    # --------------------------------------------------

    st.subheader("Geographic Fairness Results")

    st.dataframe(
        fairness_data,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These are Module 4 validation results. Groups with fewer orders "
        "may have less stable estimates. The analysis included states "
        "with at least 100 validation orders."
    )

    # --------------------------------------------------
    # 7. Interpretation
    # --------------------------------------------------

    st.subheader("What the Results Indicate")

    st.markdown("""
    - **São Paulo (SP):** Has the lowest positive prediction rate (6.44%)
      and recall (17.91%) among the four states.
    - **Rio de Janeiro (RJ):** Has a higher positive prediction rate (26.41%)
      and false positive rate (21.80%) than São Paulo.
    - **Pará (PA):** Has the highest observed positive prediction rate
      (38.53%) and recall (70.97%), but its validation sample is relatively
      small (109 orders).
    - **Mato Grosso (MT):** Has a positive prediction rate of 17.99%
      and recall of 44.44%.
    """)

    st.warning(
        "These differences require further investigation. They may reflect "
        "differences in customer/order patterns, data representation, or "
        "model performance. They should not automatically be interpreted "
        "as evidence of unfair treatment."
    )

    # --------------------------------------------------
    # 8. Ethical compliance assessment
    # --------------------------------------------------

    st.subheader("Ethical Compliance Assessment")

    st.markdown("""
    **Fairness assessment**

    - Geographic performance comparisons: Completed on the validation data.
    - Protected demographic fairness assessment: Not possible with the
      available dataset.
    - Demographic Parity and Equalized Odds: Not formally assessed.
    - Formal bias mitigation: Not implemented in Module 4.

    **Privacy and data protection**

    - The project uses the public Olist Brazilian e-commerce dataset.
    - The fairness page presents aggregated state-level results rather
      than individual customer records.
    - Customer-level information should be handled with appropriate
      access restrictions and safeguards against re-identification.

    **Transparency and explainability**

    - The dashboard provides model performance results and SHAP explanations.
    - The model's limitations and prediction uncertainty should be
      communicated to business users.
    - SHAP explanations describe model behaviour; they do not establish
      causal relationships.
    """)

    # --------------------------------------------------
    # 9. Further validation and mitigation
    # --------------------------------------------------

    st.subheader("Further Fairness and Risk Controls")

    st.markdown("""
    The following activities remain important before any operational use:

    1. Investigate why prediction and error rates differ across states.
    2. Examine data representation and the stability of results across
       larger samples and time periods.
    3. Obtain appropriate demographic data, where lawful and justified,
       before conducting protected-group fairness assessments.
    4. Evaluate mitigation approaches and compare model performance
       before and after any changes.
    5. Monitor model performance and fairness indicators over time.
    6. Retain human oversight rather than using model predictions as
       automatic decisions about customers.
    """)

    st.caption(
        "These are Module 4 validation results for the four states with "
        "at least 100 validation orders. The results do not represent "
        "all Brazilian states. Estimates for smaller groups may be less stable."
    )


elif page == "Model Information":

    st.title("Model Information and Transparency")

    st.write(
        "This page explains how the model works, what information it uses, "
        "how well it performed and the limitations that should guide its use."
    )

    # --------------------------------------------------
    # 1. MODEL OVERVIEW
    # --------------------------------------------------

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
        "A prediction above the classification threshold is classified "
        "as low satisfaction predicted. A prediction below the threshold "
        "is classified as low satisfaction not predicted."
    )

    st.divider()

    # --------------------------------------------------
    # 2. MODEL PERFORMANCE
    # --------------------------------------------------

    st.subheader("Final Test Performance")

    st.write(
        "The following results were obtained from 19,167 orders "
        "in the final chronological test dataset."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "86.42%")
        st.metric("Precision", "24.61%")

    with col2:
        st.metric("Recall", "19.58%")
        st.metric("F1 Score", "21.81%")

    with col3:
        st.metric("ROC-AUC", "58.17%")
        st.metric("PR-AUC", "16.98%")

    st.warning(
        "The model has limited ability to identify low-satisfaction "
        "orders. Its accuracy is influenced by class imbalance, so "
        "accuracy alone gives an incomplete picture of performance."
    )

    st.divider()

    # --------------------------------------------------
    # 3. INPUT VARIABLES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 4. IMPORTANT LIMITATIONS
    # --------------------------------------------------

    st.subheader("Important Limitations")

    st.markdown(
        """
        1. **Limited predictive performance:** The model achieved
           relatively low recall and ROC-AUC on the final test dataset.

        2. **Class imbalance:** Low-satisfaction orders represent
           a minority of observations, affecting how performance
           should be interpreted.

        3. **Geographic representation:** Customer state is included
           as an input variable. Differences across states require
           further investigation and do not independently establish
           unfairness.

        4. **Limited demographic information:** The dataset does not
           provide sufficient protected demographic information for
           formal demographic fairness testing.

        5. **No formal bias mitigation:** No formal bias mitigation
           intervention was implemented in Module 4.

        6. **Prediction versus causation:** Predictions identify
           statistical patterns. They do not establish why a customer
           may be dissatisfied.

        7. **Prediction-time availability:** The availability of all
           input variables immediately after an order is placed requires
           further operational verification.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 5. RESPONSIBLE USE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 6. TRANSPARENCY
    # --------------------------------------------------

    st.subheader("Transparency and Accountability")

    st.markdown(
        """
        - **Data basis:** Olist Brazilian E-Commerce Public Dataset.
        - **Model type:** Random Forest classification.
        - **Explainability:** SHAP-based global and individual explanations.
        - **Fairness review:** Exploratory geographic comparisons.
        - **Oversight:** Predictions require human interpretation.
        - **Project status:** Academic analytics prototype, not a
          production-approved customer decision system.
        """
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Olist Brazilian E-Commerce Analytics Project | "
    "Machine Learning and Customer Experience | "
    "Academic project"
)
