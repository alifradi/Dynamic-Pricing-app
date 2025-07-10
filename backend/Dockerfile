# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .
COPY backend/main.py .
COPY ../data /app/data

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Copy data models and optimization modules
COPY data_models ./data_models/
COPY optimization ./optimization/

# Create necessary directories
RUN mkdir -p /app/data_models /app/optimization

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run the application
CMD ["python", "-c", "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=8001)"]

