import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def ensure_test_artifacts(root: Path = ROOT) -> None:
    """Create deterministic smoke-test fixtures when CI runs on a clean checkout."""
    features_dir = root / "data" / "features"
    processed_dir = root / "data" / "processed"
    features_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_path = features_dir / "selected_features.csv"
    if not feature_path.exists():
        _write_selected_features(feature_path)

    validation_path = processed_dir / "day10_validation_results.json"
    if not validation_path.exists():
        _write_validation_results(validation_path)

    risk_summary_path = processed_dir / "day11_risk_summary.csv"
    if not risk_summary_path.exists():
        _write_risk_summary(risk_summary_path)

    benchmark_summary_path = processed_dir / "day11_benchmark_summary.csv"
    if not benchmark_summary_path.exists():
        _write_benchmark_summary(benchmark_summary_path)

    feature_importance_path = processed_dir / "feature_importance_analysis.csv"
    if not feature_importance_path.exists():
        _write_feature_importance(feature_importance_path)

    target_stocks_path = processed_dir / "target_stocks.txt"
    if not target_stocks_path.exists():
        target_stocks_path.write_text("AAPL\nNVDA\nMSFT\nAMD\nAMZN\n", encoding="utf-8")


def _write_selected_features(path: Path) -> None:
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    tickers = ["AAPL", "NVDA", "MSFT"]
    ticker_bias = {"AAPL": -0.0007, "NVDA": 0.0016, "MSFT": 0.0005}
    ticker_code = {"AAPL": 0.0, "NVDA": 1.0, "MSFT": 2.0}
    rows = []

    for day_index, date in enumerate(dates):
        for stock_index, ticker in enumerate(tickers):
            phase = day_index / 9.0 + stock_index * 0.7
            trend = day_index / 180.0
            daily_return = 0.0012 * np.cos(phase / 1.3) + (stock_index - 1) * 0.00025
            return_5d = (
                0.0034 * np.sin(phase)
                + 0.0011 * np.cos(phase / 2.0)
                + ticker_bias[ticker]
            )
            close = 120 + stock_index * 18 + day_index * 0.22 + 2.4 * np.sin(phase / 1.5)
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Close": round(close, 4),
                    "daily_return": round(daily_return, 6),
                    "momentum_1d": round(daily_return, 6),
                    "momentum_5d": round(return_5d * 1.6 + stock_index * 0.0004, 6),
                    "momentum_20d": round(return_5d * 2.3 + trend * 0.002, 6),
                    "feature_alpha": round(np.sin(phase) + stock_index * 0.15, 6),
                    "feature_beta": round(np.cos(phase / 1.7) - stock_index * 0.08, 6),
                    "feature_gamma": round(((day_index % 10) / 10.0) + stock_index * 0.05, 6),
                    "ticker_code": ticker_code[ticker],
                    "target_1d": round(daily_return, 6),
                    "target_5d": round(return_5d, 6),
                    "return_1d": round(daily_return, 6),
                    "return_5d": round(return_5d, 6),
                    "sharpe_5d": round(return_5d / (0.0045 + stock_index * 0.0004), 6),
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_validation_results(path: Path) -> None:
    dates = pd.date_range("2024-06-03", periods=40, freq="B")
    predictions = []
    actuals = []
    for idx, _ in enumerate(dates):
        prediction = 0.0045 * np.sin(idx / 4.5) + 0.0012
        actual = 0.0038 * np.sin((idx + 1) / 4.7) + 0.0006 * np.cos(idx / 6.0)
        predictions.append(round(float(prediction), 6))
        actuals.append(round(float(actual), 6))

    payload = {
        "walk_forward": {
            "model_name": "FixtureValidationModel",
            "predictions": predictions,
            "actuals": actuals,
            "dates": [date.strftime("%Y-%m-%d") for date in dates],
            "total_predictions": len(predictions),
            "total_folds": 5,
            "overall_r2": 0.14,
            "overall_rmse": 0.031,
            "overall_mse": 0.00096,
            "overall_mae": 0.022,
            "mean_fold_r2": -0.273,
            "std_fold_r2": 0.19,
            "mean_fold_rmse": 0.034,
            "stability_score": 3.278,
            "fold_results": [
                {"fold": 1, "r2": -0.31, "rmse": 0.035},
                {"fold": 2, "r2": -0.25, "rmse": 0.034},
                {"fold": 3, "r2": -0.21, "rmse": 0.033},
            ],
        },
        "out_of_sample": {
            "r2": -0.013,
            "rmse": 0.036,
            "mae": 0.024,
        },
        "robustness": {
            "directional_accuracy": 0.5486,
            "prediction_bias": 0.0002,
        },
        "risk_metrics": {
            "annual_return": 0.6996,
            "max_drawdown": -0.4135,
            "sharpe_ratio": 3.72,
        },
        "stability": {
            "rating": "Excellent",
            "score": 3.278,
        },
        "attribution": {
            "forecast_horizon_days": 5,
            "source": "test-fixture",
        },
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_risk_summary(path: Path) -> None:
    rows = [
        {
            "Model": "Ensemble_SimpleAverage",
            "Sharpe_Ratio": 3.72,
            "Gross_Sharpe_Ratio": 4.00,
            "Sortino_Ratio": 5.10,
            "Calmar_Ratio": 1.69,
            "Annual_Return": 0.6996,
            "Gross_Annual_Return": 0.755,
            "Annual_Volatility": 0.188,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": 0.4893,
            "Information_Ratio": 1.94,
            "VaR_95": -0.061,
            "CVaR_95": -0.084,
            "Max_Drawdown": -0.4135,
            "Max_Drawdown_Percent": -41.35,
            "Win_Rate": 0.624,
            "Signal_Accuracy": 0.5486,
            "Trade_Accuracy": 0.624,
            "Trade_Rate": 0.8787,
            "Kelly_Position_Size": 0.10,
            "Total_Return": 0.561,
            "Gross_Total_Return": 0.603,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 10,
        },
        {
            "Model": "Ensemble_VotingRegressor",
            "Sharpe_Ratio": 3.21,
            "Gross_Sharpe_Ratio": 3.46,
            "Sortino_Ratio": 4.52,
            "Calmar_Ratio": 1.41,
            "Annual_Return": 0.611,
            "Gross_Annual_Return": 0.648,
            "Annual_Volatility": 0.190,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": 0.4007,
            "Information_Ratio": 1.62,
            "VaR_95": -0.058,
            "CVaR_95": -0.081,
            "Max_Drawdown": -0.432,
            "Max_Drawdown_Percent": -43.2,
            "Win_Rate": 0.598,
            "Signal_Accuracy": 0.541,
            "Trade_Accuracy": 0.603,
            "Trade_Rate": 0.861,
            "Kelly_Position_Size": 0.09,
            "Total_Return": 0.487,
            "Gross_Total_Return": 0.523,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 10,
        },
        {
            "Model": "ConsensusMetaEnsemble",
            "Sharpe_Ratio": 2.84,
            "Gross_Sharpe_Ratio": 3.05,
            "Sortino_Ratio": 4.18,
            "Calmar_Ratio": 1.29,
            "Annual_Return": 0.552,
            "Gross_Annual_Return": 0.589,
            "Annual_Volatility": 0.194,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": 0.3417,
            "Information_Ratio": 1.38,
            "VaR_95": -0.056,
            "CVaR_95": -0.078,
            "Max_Drawdown": -0.428,
            "Max_Drawdown_Percent": -42.8,
            "Win_Rate": 0.584,
            "Signal_Accuracy": 0.533,
            "Trade_Accuracy": 0.589,
            "Trade_Rate": 0.842,
            "Kelly_Position_Size": 0.08,
            "Total_Return": 0.451,
            "Gross_Total_Return": 0.482,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 10,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_benchmark_summary(path: Path) -> None:
    rows = [
        {
            "Benchmark": "EqualWeightLongOnly",
            "Description": "Passive equal-weight long-only universe benchmark",
            "Strategy_Type": "long_only",
            "Signal_Source": "equal_weight",
            "Sharpe_Ratio": 0.74,
            "Sortino_Ratio": 1.10,
            "Calmar_Ratio": 0.51,
            "Annual_Return": 0.2103,
            "Annual_Volatility": 0.182,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": 0.0,
            "Information_Ratio": 0.0,
            "Max_Drawdown": -0.412,
            "Max_Drawdown_Percent": -41.2,
            "Win_Rate": 0.53,
            "Signal_Coverage": 1.0,
            "Observations": 120,
            "Total_Return": 0.191,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 0,
        },
        {
            "Benchmark": "TopQuartileMomentum5D",
            "Description": "Long-only top quartile by 5-day momentum",
            "Strategy_Type": "long_only",
            "Signal_Source": "momentum_5d",
            "Sharpe_Ratio": 0.62,
            "Sortino_Ratio": 0.95,
            "Calmar_Ratio": 0.44,
            "Annual_Return": 0.184,
            "Annual_Volatility": 0.176,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": -0.0263,
            "Information_Ratio": -0.14,
            "Max_Drawdown": -0.391,
            "Max_Drawdown_Percent": -39.1,
            "Win_Rate": 0.516,
            "Signal_Coverage": 0.34,
            "Observations": 120,
            "Total_Return": 0.167,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 10,
        },
        {
            "Benchmark": "CrossSectionMomentum5D",
            "Description": "Long top quartile and short bottom quartile by 5-day momentum",
            "Strategy_Type": "long_short",
            "Signal_Source": "momentum_5d",
            "Sharpe_Ratio": 0.51,
            "Sortino_Ratio": 0.81,
            "Calmar_Ratio": 0.37,
            "Annual_Return": 0.143,
            "Annual_Volatility": 0.169,
            "Benchmark_Annual_Return": 0.2103,
            "Benchmark_Sharpe_Ratio": 0.74,
            "Excess_Annual_Return": -0.0673,
            "Information_Ratio": -0.25,
            "Max_Drawdown": -0.384,
            "Max_Drawdown_Percent": -38.4,
            "Win_Rate": 0.507,
            "Signal_Coverage": 0.67,
            "Observations": 120,
            "Total_Return": 0.131,
            "Forecast_Horizon_Days": 5,
            "Transaction_Cost_Bps": 20,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_feature_importance(path: Path) -> None:
    rows = [
        {"model": "Ensemble_SimpleAverage", "feature": "momentum_5d", "importance": 0.24, "rank": 1},
        {"model": "Ensemble_SimpleAverage", "feature": "feature_alpha", "importance": 0.18, "rank": 2},
        {"model": "ConsensusMetaEnsemble", "feature": "momentum_20d", "importance": 0.22, "rank": 1},
        {"model": "ConsensusMetaEnsemble", "feature": "feature_beta", "importance": 0.16, "rank": 2},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
