# Scalability Assessment

## Current Strengths

- FastAPI and Streamlit are separated into independent services.
- Model artifacts are loaded once per process.
- The API already uses caching for repeated prediction and optimization requests.
- Docker packaging exists for multi-service deployment.

## Current Bottlenecks

- Live predictions depend on sequential `yfinance` fetches.
- The dashboard can trigger full prediction cycles, which are network-bound.
- Streamlit is optimized for operator workflows rather than high-traffic public usage.

## Next Scaling Steps

- Replace sequential market fetches with batched or provider-backed data ingestion.
- Move prediction jobs to a background queue if request volume grows.
- Add Redis for shared caching across API replicas.
- Persist metrics to a time-series store for historical monitoring instead of JSON files.
- Place the dashboard behind auth if exposed publicly.

## Recommended Production Shape

- 1 API service
- 1 dashboard service
- shared object or volume storage for models and processed artifacts
- centralized logging and metrics collection
- optional worker for scheduled retraining and artifact refresh
