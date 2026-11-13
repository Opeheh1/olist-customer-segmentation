# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from creating unnecessary cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure the raw data directory inside Docker
ENV OLIST_RAW_DATA_DIR=/app/data/raw

# Copy and install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the standalone pipeline and Great Expectations configuration
COPY olist_pipeline.py .
COPY gx/ ./gx/

# Create the processed-data output directory
RUN mkdir -p /app/data/processed

# Execute the Olist data pipeline
CMD ["python", "olist_pipeline.py"]
