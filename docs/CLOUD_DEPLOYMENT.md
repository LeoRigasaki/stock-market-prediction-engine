# Cloud Deployment

## Docker Entry Points

- API container:
  `SERVICE_TYPE=api`
- Dashboard container:
  `SERVICE_TYPE=dashboard`

The existing `Dockerfile`, `docker-compose.yml`, and `docker-compose-public.yml` are already suitable for container-based deployment.

## Recommended Hosting Paths

### Option 1: Single VM

- Run both containers behind Caddy or Nginx.
- Route `/api` to the FastAPI service and `/` to Streamlit.
- Persist `data/`, `models/`, and `logs/` on attached storage.

### Option 2: Container Platforms

- Deploy the API container to Render, Railway, Fly.io, ECS, or Azure Container Apps.
- Deploy the dashboard container separately and expose `STOCK_ENGINE_API_URL` to point at the API service.
- Use secrets for `STOCK_ENGINE_API_KEY`, Kaggle credentials, and logging configuration.

## Deployment Checklist

- Confirm `models/**/*.joblib` is included in the image or mounted at runtime.
- Expose ports `8000` and `8501`.
- Set health checks for:
  - API: `GET /health`
  - Dashboard: HTTP 200 on `/`
- Configure log aggregation for the `logs/` directory.
- Run the load test tool before public release:
  `venv/bin/python tools/load_test_api.py --requests 20 --concurrency 5`
