# Stock Market Prediction Engine

Research-oriented stock prediction and portfolio analytics system built with FastAPI, Streamlit, and classical/ensemble ML models.

This repository is best understood as an end-to-end applied data science project, not proof of deployable trading alpha. It includes:
- live inference from current market data
- saved walk-forward validation artifacts
- risk and portfolio analysis
- a dashboard and API for inspection

## Reality Check

The strongest improvement in this revision is methodological honesty:
- performance metrics are now annualized on the correct 5-day forecast horizon
- dashboard and API performance tiles now use cost-adjusted risk metrics
- live inference no longer hardcodes one "best" model and instead ranks models from the saved evaluation summary
- validation pages now surface predictive quality metrics directly instead of hiding behind Sharpe alone

If you are reviewing this as a hiring or client project, the right takeaway is:
- good engineering depth
- real artifact pipeline
- useful ML and risk tooling
- still a research system with meaningful predictive limitations

## Verified Evaluation Snapshot

The current saved evaluation reflects a 5-day forecast horizon with a 10 bps transaction-cost assumption per evaluation period.

### Cost-Adjusted Risk Summary

| Metric | Value | Source |
| --- | ---: | --- |
| Best net Sharpe | 3.72 | `data/processed/day11_risk_summary.csv` |
| Best gross Sharpe | 4.00 | `data/processed/day11_risk_summary.csv` |
| Best model | `Ensemble_SimpleAverage` | `data/processed/day11_risk_summary.csv` |
| Best benchmark | `EqualWeightLongOnly` | `data/processed/day11_benchmark_summary.csv` |
| Best benchmark Sharpe | 0.74 | `data/processed/day11_benchmark_summary.csv` |
| Model edge vs best benchmark | +2.98 Sharpe | `data/processed/day11_risk_summary.csv`, `data/processed/day11_benchmark_summary.csv` |
| Net annual return | 69.96% | `data/processed/day11_risk_summary.csv` |
| Best benchmark annual return | 21.03% | `data/processed/day11_benchmark_summary.csv` |
| Trade accuracy | 62.40% | `data/processed/day11_risk_summary.csv` |
| Trade rate | 87.87% | `data/processed/day11_risk_summary.csv` |
| Signal accuracy | 54.86% | `data/processed/day11_risk_summary.csv` |
| Max drawdown | -41.35% | `data/processed/day11_risk_summary.csv` |

### Predictive Validation Summary

| Metric | Value | Source |
| --- | ---: | --- |
| Walk-forward R2 | 0.140 | `data/processed/day10_validation_results.json` |
| Mean fold R2 | -0.273 | `data/processed/day10_validation_results.json` |
| Out-of-sample R2 | -0.013 | `data/processed/day10_validation_results.json` |
| Stability rating | `Excellent` | `data/processed/day10_validation_results.json` |
| Stability score | 3.278 | `data/processed/day10_validation_results.json` |

These numbers are intentionally presented together because they tell a more truthful story:
- the strategy backtest is materially stronger than the pure predictive fit
- the model can still be useful for ranking or signal generation even when out-of-sample regression fit is weak
- the project should be discussed as a research and signal-engineering system, not as a proven production trading business

## What The System Does

### Backend
- FastAPI service for prediction, model performance, portfolio optimization, and alerting
- authenticated endpoints for `/predict`, `/predict/detailed`, `/portfolio/optimize`, `/models/performance`, and `/alerts/active`
- real-time market fetches via `yfinance`

### Frontend
- Streamlit dashboard for:
  - overview and provenance
  - live predictions
  - performance analytics
  - portfolio optimization
  - alert review
  - model insights

### Modeling
- ensemble models: simple average, voting regressor, stacked ensemble
- individual models: XGBoost, LightGBM, Random Forest
- 73 engineered features from technical and statistical signals
- walk-forward validation and out-of-sample evaluation artifacts

## Why This Version Is Better

This revision addresses the main credibility issues that usually make quant portfolio projects look fake:

1. It does not present a backtest Sharpe ratio without context.
- the README now pairs Sharpe with walk-forward and out-of-sample predictive metrics
- the dashboard labels cost-adjusted metrics and forecast horizon assumptions

2. It does not annualize 5-day returns as if they were daily.
- the risk layer now uses `252 / 5` periods per year

3. It does not pretend one model is best because its filename says so.
- the live engine now loads multiple production models and uses saved evaluation rankings to weight predictions

4. It does not silently mix live data, saved artifacts, and placeholder values.
- the dashboard now exposes provenance and avoids fake defaults for "last cycle" and performance summaries

5. It does not benchmark itself against only one weak baseline.
- the saved benchmark suite now includes equal-weight, momentum, and mean-reversion baselines on the same dates and horizon
- the dashboard and API expose the strongest naive benchmark separately from the model ranking

6. It now uses a confidence-aware neutral band instead of forcing every tiny prediction into a trade.
- low-conviction signals are held flat rather than automatically converted into short exposure
- the saved risk summary now reports both `Signal_Accuracy` and `Trade_Accuracy`

## Running Locally

### 1. Create and activate the virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the backend

```bash
venv/bin/python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000
```

Open:
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Start the frontend

```bash
venv/bin/streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Open:
- Dashboard: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## API Example

```python
import requests

headers = {"Authorization": "Bearer demo_key_12345"}

prediction = requests.post(
    "http://127.0.0.1:8000/predict/detailed",
    json={"symbols": ["AAPL", "NVDA", "MSFT"]},
    headers=headers,
    timeout=120,
).json()

performance = requests.get(
    "http://127.0.0.1:8000/models/performance",
    headers=headers,
    timeout=30,
).json()
```

## Methodology Notes

### Target
- the main regression target is `return_5d`
- that means all risk and annualization logic should be interpreted on a 5-day horizon, not a 1-day horizon

### Benchmarking
- model metrics still use the equal-weight realized universe return over the same walk-forward dates as the direct benchmark reference
- `data/processed/day11_benchmark_summary.csv` now adds a benchmark suite with:
  - `EqualWeightLongOnly`
  - `TopQuartileMomentum5D`
  - `TopQuartileMomentum20D`
  - `CrossSectionMomentum5D`
  - `MeanReversion1D`
- the README, API, and dashboard now report the best naive benchmark separately so the model has to clear a stronger hurdle

### Transaction Costs
- cost-adjusted metrics use a 10 bps per-period assumption in the saved risk summary
- this is still a simplification and should not be treated as a full execution model

### Live Inference
- live predictions use a score-weighted ensemble across loaded production models
- score weights come from the saved risk summary rather than a hardcoded model choice

## Limitations

- `yfinance` is fine for prototyping but fragile for production use
- saved validation artifacts and live inference are not a substitute for broker-connected live PnL
- out-of-sample regression fit is weak, so this should not be sold as a fully proven forecasting edge
- transaction costs are simplified and do not model slippage, spread, market impact, or turnover exactly
- portfolio optimization still depends on model outputs whose economic interpretation should be stress-tested further

## Good Portfolio Talking Points

If you present this project in interviews or to clients, lead with:
- end-to-end ML system design
- time-aware validation and artifact management
- ensemble inference with live API and dashboard delivery
- risk analysis, portfolio construction, and signal monitoring
- willingness to correct inflated metrics instead of hiding them

That last point matters. A realistic and defensible evaluation story is stronger than a flashy but fragile one.

## Project Structure

```text
src/
  api_server.py
  realtime_prediction.py
  risk_management.py
  streamlit_dashboard.py
  validation_framework.py
data/
  features/
  processed/
models/
plots/
tests/
tools/
```

## Current Gaps Worth Improving Next

- replace `yfinance` with a more reliable market-data source
- add explicit turnover-based transaction cost modeling
- track experiments and model versions formally
- add external SPY or sector-ETF benchmarks alongside the internal feature-driven benchmark suite
- add a persistent prediction history store for live monitoring

## Verification

Recent local verification:
- `venv/bin/python -m unittest discover -s tests`
- backend health and authenticated API checks
- Streamlit runtime checks including live-scan path
- regenerated `day11_risk_summary.csv`, `day11_benchmark_summary.csv`, and related artifacts after correcting the evaluation math and expanding the benchmark suite
