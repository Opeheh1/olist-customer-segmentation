# Olist Customer Segmentation

A customer analytics project using the **Olist Brazilian E-Commerce Public Dataset (2016–2018)**. The project is designed to help Olist understand customer purchasing behaviour and customer experience, with the longer-term goal of developing actionable customer segments.

> **Current achievement:** The data engineering stage has been implemented and tested. A Prefect pipeline integrates and validates order-level data. 

## 1. Project Overview

Olist is a Brazilian e-commerce marketplace that connects small businesses with online sales channels. This project uses Olist's historical marketplace data to develop a customer segmentation solution based on purchasing behaviour, spending patterns, payment preferences and customer experience.

The intended business value is to support more relevant marketing, improve customer retention and help decision-makers understand differences between customer groups.

## 2. Business Problem

Olist needs a structured way to understand how customers differ in their purchasing activity and experience. Without meaningful customer groups, marketing and retention activities may treat customers too uniformly and miss opportunities to tailor engagement.

The project addresses this problem through customer-level data preparation, RFM analysis and machine-learning-based segmentation. The resulting segments will be interpreted in business terms and translated into practical recommendations.

## 3. Project Objectives

The project aims to:

- Develop approximately **3–6 interpretable customer segments**.
- Calculate customer-level **Recency, Frequency and Monetary (RFM)** measures.
- Use K-Means clustering as a candidate segmentation method.
- Evaluate clustering quality using the Silhouette Score alongside business interpretability.
- Examine payment preferences and review-based customer experience where data coverage permits.
- Identify customers with relatively low or declining purchasing activity for further retention analysis.
- Develop a Power BI dashboard and actionable recommendations.

**Interpretation note:** Recency and frequency describe historical purchasing behaviour. They can help identify customers who may warrant further retention attention, but they do not, by themselves, establish that a customer has churned or predict future churn. Historical monetary value is not the same as predicted customer lifetime value.

## 4. Dataset and Data Sources

The project uses the real, anonymised **Brazilian E-Commerce Public Dataset by Olist**, covering transactions from 2016 to 2018.

**Dataset source:**  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset contains related CSV tables covering areas such as:

- Customers
- Orders
- Order items
- Payments
- Reviews
- Products
- Product category translations

The source tables are integrated at order level for the data pipeline. Customer-level feature engineering is performed separately for the segmentation analysis.

The dataset is historical and anonymised. Customer and order identifiers are used to link records; they should not be treated as direct personal identifiers or published unnecessarily.

## 5. Project Methodology and Current Progress

The work is organised into connected stages:

1. **Business understanding:** Define the customer segmentation problem, project vision, objectives, stakeholders and ethical principles.
2. **Data understanding:** Inspect the source tables, assess data quality and identify issues relevant to analysis.
3. **Data engineering:** Clean, aggregate and integrate the source tables into order-level analytical datasets.
4. **Data validation:** Apply Great Expectations checks to assess the quality of the integrated output.
5. **Customer feature engineering:** Prepare customer-level RFM and supporting behavioural features.
6. **Segmentation:** Apply and assess clustering methods, including K-Means.
7. **Business interpretation:** Profile the resulting segments and develop targeted recommendations.
8. **Communication:** Present findings through a Power BI dashboard and project report.

### Implemented data pipeline

The current automated pipeline is implemented in `olist_pipeline.py` using **Prefect 3**. It contains the following main stages:

- `ingest_and_clean_raw_data` – loads and standardises the source tables.
- `aggregate_order_data` – aggregates order-item, payment and review information.
- `integrate_order_data` – combines the prepared data into order-level records.
- `save_integrated_data` – saves the integrated datasets.
- `validate_final_dataset` – runs Great Expectations validation on the final dataset.

The pipeline produces two order-level outputs:

| Output | Rows | Columns | Unique orders |
|---|---:|---:|---:|
| Unified order-level dataset | 99,441 | 31 | 99,441 |
| Delivered-order dataset | 96,478 | 31 | 96,478 |

The delivered-order dataset provides the completed-order population for analyses such as RFM, where an order must have been delivered to qualify as a completed purchase.

### Data quality validation

The implemented Great Expectations validation run completed with:

- **27 checks passed**
- **0 checks failed**

Validation results are documented in `reports/validation/`. These results relate to the tested pipeline output; they should not be interpreted as proof that every possible data issue has been eliminated.

### Customer-level features

Customer-level feature engineering is maintained separately from the Prefect order-level pipeline. The prepared customer feature dataset contains **93,358 customer records and 14 features**. It supports the next segmentation stage and is not an output created by the Prefect pipeline described above.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Data processing and pipeline implementation |
| Pandas and NumPy | Data cleaning, aggregation and transformation |
| Prefect 3 | Workflow orchestration |
| Great Expectations | Data quality validation |
| Scikit-learn | Planned clustering and model evaluation |
| Jupyter Notebook | Exploration and analytical development |
| Power BI | Planned segmentation dashboard |
| Docker | Reproducible pipeline execution |
| Git and GitHub | Version control and project collaboration |

## 7. Repository Structure

The main project components are organised as follows:

```text
olist-customer-segmentation/
│
├── data/
│   ├── raw/                 # Local source data location (not for new commits)
│   └── processed/           # Generated analytical datasets
│
├── notebooks/
│   └── data_cleaning.ipynb  # Data preparation and exploration
│
├── gx/
│   ├── expectations/        # Great Expectations suites
│   ├── validation_definitions/
│   └── great_expectations.yml
│
├── reports/
│   ├── governance/          # Governance and audit documentation
│   └── validation/          # Data validation results
│
├── src/                     # Reusable source code (for future modules)
├── models/                  # Saved models (for future modelling)
├── dashboard/               # Power BI dashboard (planned)
│
├── olist_pipeline.py        # Prefect data pipeline
├── requirements.txt         # Python dependencies
├── Dockerfile               # Pipeline container definition
├── .dockerignore
├── .gitignore
└── README.md
```

Generated data files and local working outputs should remain outside version control unless there is a clear project requirement to share them. Do not commit customer-level feature exports, credentials, local environment files or other sensitive/generated artefacts.

## 8. Setup and Execution

### Prerequisites

- Python 3.12
- Git
- Access to the Olist dataset
- Docker Desktop (optional, for container execution)

### Clone the repository

```bash
git clone https://github.com/Opeheh1/olist-customer-segmentation.git
cd olist-customer-segmentation
```

### Install dependencies

Create and activate a virtual environment, then install the project requirements:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare the source data

Download the Olist dataset from Kaggle and place the required source CSV files in the local raw-data directory:

```text
data/raw/
```

The pipeline supports an alternative raw-data location through the `OLIST_RAW_DATA_DIR` environment variable. For example, in Windows PowerShell:

```powershell
$env:OLIST_RAW_DATA_DIR = "C:\path\to\your\olist\raw\data"
```

Use the folder containing the source CSV files as the value. Keep local data and generated outputs out of new Git commits.

### Run the pipeline locally

From the repository root:

```bash
python olist_pipeline.py
```

The pipeline writes the integrated order-level outputs to the project's processed-data directory and runs the configured Great Expectations validation.

### Run with Docker

Build the image:

```bash
docker build -t olist-customer-segmentation:1.0 .
```

Example run on Windows PowerShell, mounting the folder containing the downloaded source CSV files and a local output folder:

```powershell
docker run --rm `
  -v "C:\path\to\olist\raw\data:/app/data/raw:ro" `
  -v "C:\path\to\processed\output:/app/data/processed" `
  olist-customer-segmentation:1.0
```

Replace the example Windows paths with the appropriate local folders. The raw-data mount is read-only; the processed-data mount allows the container to write its outputs to the host machine.


## 9. Project Status and Next Steps

### Completed

- Defined the Olist customer segmentation business problem, vision and objectives.
- Reviewed and prepared the source tables for integration.
- Implemented the Prefect order-level data pipeline.
- Created Great Expectations validation suites and validation documentation.
- Tested the pipeline in Docker.
- Verified the integrated and delivered-order dataset outputs.
- Documented governance and data-quality audit artefacts.

### Next steps

- Finalise and review customer-level RFM and supporting features.
- Perform exploratory analysis of customer purchasing behaviour.
- Apply K-Means and compare candidate segmentation solutions.
- Evaluate cluster quality, stability and business interpretability.
- Profile the segments and develop practical marketing and retention recommendations.
- Build the Power BI dashboard.
- Document findings, limitations and recommendations in the final project report.

**Project status:** Data engineering and validation are implemented; customer segmentation and business-facing outputs are still in progress.

## 10. Project Repository

GitHub: https://github.com/Opeheh1/olist-customer-segmentation

## 11. References

Olist. (n.d.). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle.  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
