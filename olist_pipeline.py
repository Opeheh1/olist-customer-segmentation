
# ============================================================
# OLIST END-TO-END DATA PIPELINE
# Extracted from the tested project notebook
# ============================================================

import os
from pathlib import Path

import pandas as pd
import great_expectations as gx

from prefect import flow, task

# Project directory: location of this Python script
PROJECT_DIR = Path(__file__).resolve().parent

# Raw data directory can be configured through an environment
# variable. Docker will mount the source data into /app/data/raw.
RAW_DATA_DIR = Path(
    os.environ.get(
        "OLIST_RAW_DATA_DIR",
        str(PROJECT_DIR / "data" / "raw")
    )
)

# Processed datasets and Great Expectations configuration
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
GX_PROJECT_DIR = PROJECT_DIR / "gx"


@task(name="Ingest and Clean Raw Data")
def ingest_and_clean_raw_data():

    # 1. Define the six raw data sources
    data_sources = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv"
    }

    cleaned_raw = {}

    # 2. Load and standardize each dataset
    for dataset_name, filename in data_sources.items():

        file_path = RAW_DATA_DIR / filename

        # Check that the source file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing raw dataset: {file_path}"
            )

        # Load the raw CSV file
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            low_memory=False
        )

        # Work on a copy
        df = df.copy()

        # 3. Standardise identifier columns
        identifier_columns = {
            "customers": [
                "customer_id",
                "customer_unique_id"
            ],
            "orders": [
                "order_id",
                "customer_id"
            ],
            "order_items": [
                "order_id",
                "product_id",
                "seller_id"
            ],
            "payments": [
                "order_id"
            ],
            "reviews": [
                "review_id",
                "order_id"
            ],
            "products": [
                "product_id"
            ]
        }

        for column in identifier_columns[dataset_name]:
            df[column] = df[column].astype("string")

        # 4. Standardize customer ZIP prefixes
        if dataset_name == "customers":
            df["customer_zip_code_prefix"] = (
                df["customer_zip_code_prefix"]
                .astype("string")
                .str.zfill(5)
            )

        # 5. Convert date columns into datetime format
        date_columns = {
            "orders": [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ],
            "reviews": [
                "review_creation_date",
                "review_answer_timestamp"
            ]
        }

        for column in date_columns.get(dataset_name, []):
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

        # 6. Store the cleaned dataset
        cleaned_raw[dataset_name] = df

        print(
            f"{dataset_name}: {df.shape} | "
            f"Standardisation completed"
        )

    print("\nRAW DATA INGESTION AND CLEANING COMPLETED.")

    return cleaned_raw



@task(name="Aggregate Order Transactions")
def aggregate_order_data(cleaned_raw):

    # Retrieve the cleaned raw datasets
    items = cleaned_raw["order_items"].copy()
    products = cleaned_raw["products"].copy()
    payments = cleaned_raw["payments"].copy()
    reviews = cleaned_raw["reviews"].copy()

    # --------------------------------------------------------
    # 1. ITEM AGGREGATION
    # --------------------------------------------------------

    # Attach product categories to order items
    items_enriched = items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Preserve original missing categories while using
    # "Unknown" for analytical category counting
    items_enriched["product_category_analytical"] = (
        items_enriched["product_category_name"]
        .fillna("Unknown")
    )

    # Summarize item transactions by order
    item_features = (
        items_enriched.groupby("order_id")
        .agg(
            item_spending=("price", "sum"),
            freight_spending=("freight_value", "sum"),
            total_items=("order_item_id", "count"),
            distinct_products=("product_id", "nunique"),
            category_diversity=(
                "product_category_analytical", "nunique"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 2. PAYMENT AGGREGATION
    # --------------------------------------------------------

    # Select the most frequent payment type.
    # Alphabetical ordering resolves ties consistently.
    def preferred_payment_type(series):
        counts = series.value_counts()
        highest_count = counts.max()
        tied_types = sorted(
            counts[counts == highest_count].index.tolist()
        )
        return tied_types[0]

    payment_features = (
        payments.groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            payment_record_count=("payment_value", "count"),
            preferred_payment_type=(
                "payment_type", preferred_payment_type
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 3. REVIEW AGGREGATION
    # --------------------------------------------------------

    # Group by order_id rather than review_id because
    # review IDs are reused across some orders.
    review_features = (
        reviews.groupby("order_id")
        .agg(
            average_review_score=("review_score", "mean"),
            review_record_count=("review_score", "count")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 4. BASIC INTEGRITY CHECKS
    # --------------------------------------------------------

    summaries = {
        "item_features": item_features,
        "payment_features": payment_features,
        "review_features": review_features
    }

    for name, dataframe in summaries.items():

        if dataframe["order_id"].duplicated().any():
            raise ValueError(
                f"Duplicate order IDs found in {name}."
            )

        print(
            f"{name}: {dataframe.shape} | "
            "Unique order IDs confirmed"
        )

    print("\nTRANSACTION AGGREGATION COMPLETED.")

    return summaries



@task(name="Integrate Order and Customer Data")
def integrate_order_data(cleaned_raw, aggregated_data):

    # Retrieve cleaned source datasets
    orders = cleaned_raw["orders"].copy()
    customers = cleaned_raw["customers"].copy()

    # Retrieve aggregated transaction datasets
    item_features = aggregated_data["item_features"]
    payment_features = aggregated_data["payment_features"]
    review_features = aggregated_data["review_features"]

    # --------------------------------------------------------
    # 1. Attach customer information to orders
    # --------------------------------------------------------

    customer_lookup = customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state"
        ]
    ].copy()

    unified_orders = orders.merge(
        customer_lookup,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------------
    # 2. Integrate item, payment and review summaries
    # --------------------------------------------------------

    unified_orders = unified_orders.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    unified_orders = unified_orders.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    unified_orders = unified_orders.merge(
        review_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # 3. Create missing-date indicators
    # --------------------------------------------------------

    unified_orders["approval_date_missing"] = (
        unified_orders["order_approved_at"].isna()
    )

    unified_orders["carrier_date_missing"] = (
        unified_orders["order_delivered_carrier_date"].isna()
    )

    unified_orders["delivery_date_missing"] = (
        unified_orders["order_delivered_customer_date"].isna()
    )

    # --------------------------------------------------------
    # 4. Create date-quality flags
    # --------------------------------------------------------

    unified_orders["approval_before_purchase"] = (
        unified_orders["order_approved_at"]
        < unified_orders["order_purchase_timestamp"]
    )

    unified_orders["carrier_before_approval"] = (
        unified_orders["order_delivered_carrier_date"]
        < unified_orders["order_approved_at"]
    )

    unified_orders["delivery_before_carrier"] = (
        unified_orders["order_delivered_customer_date"]
        < unified_orders["order_delivered_carrier_date"]
    )

    unified_orders["delivery_after_estimate"] = (
        unified_orders["order_delivered_customer_date"]
        > unified_orders["order_estimated_delivery_date"]
    )

    # Calculate time gaps without changing original timestamps
    unified_orders["carrier_approval_gap_hours"] = (
        unified_orders["order_delivered_carrier_date"]
        - unified_orders["order_approved_at"]
    ).dt.total_seconds() / 3600

    unified_orders["delivery_carrier_gap_hours"] = (
        unified_orders["order_delivered_customer_date"]
        - unified_orders["order_delivered_carrier_date"]
    ).dt.total_seconds() / 3600

    # --------------------------------------------------------
    # 5. Validate the integrated dataset
    # --------------------------------------------------------

    if unified_orders["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found.")

    if unified_orders["customer_unique_id"].isna().any():
        raise ValueError("Some orders have no customer identity.")

    # Create delivered-order dataset
    delivered_orders = unified_orders[
        unified_orders["order_status"] == "delivered"
    ].copy()

    print("INTEGRATION COMPLETED")
    print("-" * 45)
    print("Unified dataset:", unified_orders.shape)
    print("Delivered dataset:", delivered_orders.shape)
    print(
        "Unique unified orders:",
        unified_orders["order_id"].nunique()
    )
    print(
        "Unique delivered orders:",
        delivered_orders["order_id"].nunique()
    )

    print("\nDelivered order status check:")
    print(delivered_orders["order_status"].value_counts())

    return unified_orders, delivered_orders



@task(name="Save Integrated Datasets")
def save_integrated_data(unified_orders, delivered_orders):

    # Create the processed-data directory if necessary
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Define output file paths
    unified_path = (
        PROCESSED_DIR / "unified_order_level.csv"
    )

    delivered_path = (
        PROCESSED_DIR / "delivered_order_level.csv"
    )

    # Save both integrated datasets
    unified_orders.to_csv(
        unified_path,
        index=False
    )

    delivered_orders.to_csv(
        delivered_path,
        index=False
    )

    # Confirm that the files can be reloaded
    unified_check = pd.read_csv(unified_path)
    delivered_check = pd.read_csv(delivered_path)

    if unified_check["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs found in saved unified data."
        )

    if delivered_check["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs found in saved delivered data."
        )

    print("DATASETS SAVED AND VERIFIED")
    print("-" * 45)
    print("Unified file:", unified_path)
    print("Unified shape:", unified_check.shape)
    print("Delivered file:", delivered_path)
    print("Delivered shape:", delivered_check.shape)

    return {
        "unified_path": str(unified_path),
        "delivered_path": str(delivered_path)
    }



@task(name="Validate Final Unified Dataset")
def validate_final_dataset():

    # Load the saved Great Expectations context
    gx_context = gx.get_context(
        context_root_dir=str(GX_PROJECT_DIR)
    )

    # Retrieve the clean nine-expectation validation definition
    validation_definition = (
        gx_context.validation_definitions.get(
            "olist_order_quality_validation_v2"
        )
    )

    # Load the newly generated unified dataset
    unified_data = pd.read_csv(
        PROCESSED_DIR / "unified_order_level.csv"
    )

    # Execute validation
    validation = validation_definition.run(
        batch_parameters={"dataframe": unified_data}
    )

    print("\nFINAL DATA QUALITY VALIDATION")
    print("-" * 45)
    print("Validation successful:", validation.success)
    print(
        "Total checks:",
        validation.statistics["evaluated_expectations"]
    )
    print(
        "Successful checks:",
        validation.statistics["successful_expectations"]
    )
    print(
        "Failed checks:",
        validation.statistics["unsuccessful_expectations"]
    )

    # Stop the pipeline if validation fails
    if not validation.success:
        raise ValueError(
            "Final data quality validation failed."
        )

    return True



@flow(name="Olist End-to-End Data Pipeline")
def olist_end_to_end_pipeline():

    print("=" * 60)
    print("STARTING OLIST END-TO-END DATA PIPELINE")
    print("=" * 60)

    # Stage A: Load and standardize raw datasets
    cleaned_raw = ingest_and_clean_raw_data()

    # Stage B: Aggregate item, payment and review records
    aggregated = aggregate_order_data(cleaned_raw)

    # Stage C: Integrate orders and customer information
    unified, delivered = integrate_order_data(
        cleaned_raw,
        aggregated
    )

    # Stage D: Save the integrated datasets
    saved_paths = save_integrated_data(
        unified,
        delivered
    )

    # Stage E: Validate the final unified dataset
    validation_passed = validate_final_dataset()

    # Final pipeline summary
    print("\n" + "=" * 60)
    print("OLIST PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    print("Raw ingestion and standardisation: SUCCESS")
    print("Transaction aggregation: SUCCESS")
    print("Data integration: SUCCESS")
    print("Dataset saving: SUCCESS")
    print("Great Expectations validation:",
          "SUCCESS" if validation_passed else "FAILED")

    print("\nUnified dataset:", unified.shape)
    print("Delivered dataset:", delivered.shape)

    print("\nOVERALL PIPELINE: SUCCESS")

    return {
        "unified_shape": unified.shape,
        "delivered_shape": delivered.shape,
        "saved_paths": saved_paths,
        "validation_passed": validation_passed
    }




if __name__ == "__main__":

    pipeline_results = olist_end_to_end_pipeline()

    print("\nPIPELINE RESULTS")
    print(pipeline_results)
    