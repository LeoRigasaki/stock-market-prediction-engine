# User Guide

## Local Run

1. Activate the repository `venv`.
2. Start the backend:
   `venv/bin/python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000`
3. Start the dashboard:
   `venv/bin/streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 127.0.0.1`

## Dashboard Pages

- `Overview`: health, saved model metrics, validation curve, artifact provenance.
- `Live Predictions`: real prediction scan using the backend detailed endpoint.
- `Performance Analytics`: saved evaluation metrics from validation and risk analysis.
- `Portfolio Optimizer`: Markowitz, risk parity, or equal weight allocation.
- `Alert Center`: live alert feed and retraining trigger display.
- `Model Insights`: model roster, validation health, and real feature importance.

## API Notes

- Docs: `http://127.0.0.1:8000/docs`
- Default demo key: `demo_key_12345`
- Key live endpoints:
  - `POST /predict`
  - `POST /predict/detailed`
  - `POST /portfolio/optimize`
  - `GET /models/performance`
  - `GET /alerts/active`

## Data Provenance

- Evaluation metrics are loaded from `data/processed/day11_risk_summary.csv`.
- Validation metrics are loaded from `data/processed/day10_validation_results.json`.
- Feature importance is loaded from `data/processed/feature_importance_analysis.csv`.
- Live prediction cycles use `yfinance` plus the production model files in `models/`.
