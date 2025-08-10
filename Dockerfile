# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:./mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-lock.txt ./

# Install Python dependencies with better caching and dependency resolution
RUN pip install --no-cache-dir --progress-bar on --timeout 300 --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --progress-bar on --timeout 300 --use-pep517 -r requirements-lock.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs mlruns

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 