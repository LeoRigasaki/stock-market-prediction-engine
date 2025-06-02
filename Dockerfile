# Stock Market Prediction Engine - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy essential model files (only what's needed)
COPY models/ensemble/ ./models/ensemble/
COPY models/advanced/ ./models/advanced/
COPY models/regression_random_forest.joblib ./models/
COPY models/feature_scaler.joblib ./models/

# Copy essential data files (not raw datasets)
COPY data/processed/target_stocks.txt ./data/processed/
COPY data/processed/day11_risk_summary.csv ./data/processed/
COPY data/processed/day10_validation_summary.csv ./data/processed/
COPY data/processed/day11_risk_analysis.json ./data/processed/
COPY data/features/selected_features.csv ./data/features/
COPY data/features/selected_features_list.txt ./data/features/
COPY data/features/model_ready_features.txt ./data/features/

# Create necessary directories
RUN mkdir -p logs plots data/processed data/features

# Copy configuration
COPY src/config.py ./src/

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$SERVICE_TYPE" = "api" ]; then\n\
    echo "🚀 Starting FastAPI Server..."\n\
    cd /app && python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000\n\
elif [ "$SERVICE_TYPE" = "dashboard" ]; then\n\
    echo "📊 Starting Streamlit Dashboard..."\n\
    cd /app && streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
else\n\
    echo "❌ Please specify SERVICE_TYPE as api or dashboard"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501 || exit 1

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 8000 8501

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]