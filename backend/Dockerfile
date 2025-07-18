# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
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

# Copy MiniZinc models and runners
COPY backend/stochastic_optimization.mzn .
COPY backend/deterministic_optimization.mzn .
COPY backend/run_stochastic_optimization.py .
COPY backend/run_deterministic_optimization.py .
COPY backend/test_minizinc.py .
COPY backend/requirements_optimization.txt .

# Create necessary directories
RUN mkdir -p /app/data_models /app/optimization

# Install optimization dependencies
RUN pip install --no-cache-dir -r requirements_optimization.txt

# Set Python path and MiniZinc path
ENV PYTHONPATH=/app
ENV PATH="/root/.local/bin:${PATH}"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run the application
CMD ["python", "-c", "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=8001)"]

