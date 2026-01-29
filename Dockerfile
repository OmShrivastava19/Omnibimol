# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY api_client.py .
COPY cache_manager.py .
COPY chembl_method.py .
COPY data_processor.py .
COPY visualizations.py .

# Copy data directory
COPY data/ ./data/

# Copy icons directory
COPY icons/ ./icons/

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py"]
