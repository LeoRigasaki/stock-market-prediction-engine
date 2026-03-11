#!/usr/bin/env python3
"""
Stock Market Prediction Engine
Dynamic Streamlit dashboard driven by real backend and saved model artifacts.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests import RequestException

# Ensure the project root is importable when Streamlit executes this file from src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.realtime_prediction import RealTimePredictionEngine
from src.risk_management import RiskManagementFramework

st.set_page_config(
    page_title="Stock Market Prediction Engine",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_theme() -> None:
    """Apply a distinct market-terminal visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #06110f;
            --bg-soft: rgba(11, 28, 24, 0.78);
            --panel: rgba(8, 22, 19, 0.84);
            --panel-border: rgba(112, 173, 147, 0.18);
            --text: #edf8f4;
            --muted: #98b6ab;
            --line: rgba(120, 175, 152, 0.16);
            --bull: #62f7a6;
            --bear: #ff7a63;
            --accent: #f4b56a;
            --accent-soft: rgba(244, 181, 106, 0.14);
            --shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 15% 18%, rgba(84, 247, 166, 0.12), transparent 28%),
                radial-gradient(circle at 82% 0%, rgba(244, 181, 106, 0.14), transparent 24%),
                linear-gradient(135deg, #03100e 0%, #081916 46%, #07130f 100%);
            background-attachment: fixed;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
            background-size: 32px 32px;
            mask-image: linear-gradient(to bottom, rgba(255,255,255,0.85), rgba(255,255,255,0.15));
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 20, 18, 0.96), rgba(4, 14, 12, 0.94));
            border-right: 1px solid rgba(104, 157, 136, 0.16);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }

        h1, h2, h3 {
            font-family: "Syne", sans-serif;
            letter-spacing: -0.02em;
        }

        .hero-shell, .panel-shell {
            border: 1px solid var(--panel-border);
            background: linear-gradient(180deg, rgba(11, 29, 25, 0.9), rgba(5, 16, 14, 0.92));
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.8rem 1.8rem 1.5rem 1.8rem;
            margin-bottom: 1.2rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.06) 40%, transparent 60%);
            transform: translateX(-100%);
            animation: sweep 8s linear infinite;
            pointer-events: none;
        }

        @keyframes sweep {
            from { transform: translateX(-100%); }
            to { transform: translateX(100%); }
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: clamp(2.3rem, 5vw, 4.3rem);
            line-height: 0.94;
            margin: 0;
            max-width: 900px;
        }

        .hero-subtitle {
            max-width: 760px;
            color: var(--muted);
            margin-top: 0.8rem;
            font-size: 1rem;
        }

        .ticker-strip {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 1.2rem;
        }

        .ticker-chip, .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.55rem 0.78rem;
            border-radius: 999px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            border: 1px solid rgba(130, 191, 165, 0.18);
            background: rgba(255, 255, 255, 0.03);
        }

        .status-online { color: var(--bull); }
        .status-offline { color: var(--bear); }

        .metric-tile {
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            background:
                linear-gradient(180deg, rgba(13, 36, 31, 0.78), rgba(7, 18, 16, 0.94));
            border: 1px solid rgba(128, 187, 161, 0.12);
            min-height: 140px;
        }

        .metric-label {
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            font-size: 0.74rem;
            color: var(--muted);
            letter-spacing: 0.12em;
        }

        .metric-value {
            font-family: "Syne", sans-serif;
            font-size: 2.2rem;
            line-height: 1;
            margin-top: 0.7rem;
        }

        .metric-meta {
            color: var(--muted);
            margin-top: 0.6rem;
            font-size: 0.92rem;
        }

        .signal-card {
            border-radius: 24px;
            padding: 1rem 1rem 0.95rem 1rem;
            border: 1px solid rgba(127, 189, 163, 0.16);
            box-shadow: var(--shadow);
            min-height: 220px;
        }

        .signal-long {
            background: linear-gradient(180deg, rgba(13, 44, 33, 0.96), rgba(5, 20, 15, 0.98));
        }

        .signal-short {
            background: linear-gradient(180deg, rgba(52, 21, 18, 0.96), rgba(20, 8, 8, 0.98));
        }

        .signal-neutral {
            background: linear-gradient(180deg, rgba(30, 30, 24, 0.94), rgba(14, 14, 11, 0.98));
        }

        .signal-symbol {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.95rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }

        .signal-direction {
            font-family: "Syne", sans-serif;
            font-size: 2rem;
            margin-top: 0.5rem;
        }

        .signal-stat {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.86rem;
            color: var(--muted);
            margin-top: 0.45rem;
        }

        .panel-shell {
            border-radius: 24px;
            padding: 1rem 1rem 0.25rem 1rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            color: var(--accent);
            margin-bottom: 0.8rem;
        }

        .source-callout {
            border-left: 3px solid var(--accent);
            background: rgba(255, 255, 255, 0.035);
            padding: 0.85rem 1rem;
            border-radius: 0 18px 18px 0;
            margin: 0.8rem 0 1rem 0;
            color: var(--muted);
        }

        div[data-testid="stMetric"] {
            border-radius: 18px;
            border: 1px solid rgba(124, 182, 157, 0.14);
            background: rgba(9, 24, 21, 0.72);
            padding: 0.85rem 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(124, 182, 157, 0.12);
        }

        .stButton button {
            border-radius: 999px;
            padding: 0.68rem 1.35rem;
            border: 1px solid rgba(244, 181, 106, 0.28);
            background: linear-gradient(135deg, rgba(244, 181, 106, 0.24), rgba(82, 246, 166, 0.12));
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stCheckbox label {
            color: var(--muted) !important;
        }

        .footer-note {
            color: var(--muted);
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            margin-top: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value: Any) -> float:
    """Convert values to float without propagating exceptions."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def percent_value(value: Any) -> float:
    """Treat ratios and already-scaled percentages consistently."""
    numeric = safe_float(value)
    return numeric * 100 if abs(numeric) <= 1.5 else numeric


def format_percent(value: Any, digits: int = 1) -> str:
    """Format ratios or percentage-like values for UI display."""
    return f"{percent_value(value):.{digits}f}%"


def normalize_return_series(values: np.ndarray) -> np.ndarray:
    """Normalize returns that may be stored either as ratios or percentages."""
    if len(values) == 0:
        return values

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr)

    scale = 100.0 if np.nanpercentile(np.abs(finite), 95) > 1.5 else 1.0
    return arr / scale


def strategy_curve_from_validation(validation_results: Dict[str, Any]) -> pd.DataFrame:
    """Build a cumulative strategy curve from saved validation outputs."""
    walk_forward = validation_results.get('walk_forward', {})
    predictions = np.asarray(walk_forward.get('predictions', []), dtype=float)
    actuals = normalize_return_series(np.asarray(walk_forward.get('actuals', []), dtype=float))
    dates = pd.to_datetime(walk_forward.get('dates', []), errors='coerce')

    length = min(len(predictions), len(actuals), len(dates))
    if length == 0:
        return pd.DataFrame()

    curve_df = pd.DataFrame({
        'Date': dates[:length],
        'Prediction': predictions[:length],
        'ActualReturn': actuals[:length],
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if curve_df.empty:
        return pd.DataFrame()

    curve_df['StrategyReturn'] = np.where(
        curve_df['Prediction'] >= 0,
        curve_df['ActualReturn'],
        -curve_df['ActualReturn'],
    )

    daily_curve = (
        curve_df.groupby('Date', as_index=False)['StrategyReturn']
        .mean()
        .sort_values('Date')
    )
    daily_curve['StrategyReturn'] = daily_curve['StrategyReturn'].clip(-0.35, 0.35)

    cumulative_log_returns = np.log1p(daily_curve['StrategyReturn']).cumsum()
    daily_curve['StrategyValue'] = 100 * np.exp(cumulative_log_returns)
    daily_curve = daily_curve.replace([np.inf, -np.inf], np.nan).dropna(subset=['StrategyValue'])

    return daily_curve[['Date', 'StrategyValue']]


def prediction_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a prediction table without assuming optional columns exist."""
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop(columns=['ModelBreakdown'], errors='ignore')


class DashboardApp:
    """Interactive dashboard driven by live API responses and saved artifacts."""

    def __init__(self) -> None:
        self.config = Config()
        self.api_base_url = os.getenv("STOCK_ENGINE_API_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("STOCK_ENGINE_API_KEY", "demo_key_12345")
        self.prediction_engine: Optional[RealTimePredictionEngine] = None
        self.risk_framework: Optional[RiskManagementFramework] = None
        self._bootstrap_session_state()

    def _bootstrap_session_state(self) -> None:
        """Initialize Streamlit session storage."""
        defaults = {
            'latest_cycle': None,
            'latest_cycle_symbols': [],
            'latest_portfolio': None,
            'latest_portfolio_request': {},
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def ensure_local_runtime(self) -> None:
        """Lazy-load local runtime for offline fallback paths."""
        if self.prediction_engine is None:
            self.prediction_engine = RealTimePredictionEngine()
            self.prediction_engine.load_production_models()

        if self.risk_framework is None:
            self.risk_framework = RiskManagementFramework()

    def api_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = 45,
        quiet: bool = False,
    ) -> Optional[Any]:
        """Perform authenticated API requests to the backend."""
        try:
            response = requests.request(
                method,
                f"{self.api_base_url}{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            if not quiet:
                st.warning(f"Backend request failed for `{path}`: {exc}")
            return None

    def backend_health(self) -> Dict[str, Any]:
        """Fetch backend health with a quiet failure mode."""
        return self.api_request("GET", "/health", timeout=5, quiet=True) or {}

    def market_status(self) -> Dict[str, Any]:
        """Fetch market status from the backend."""
        return self.api_request("GET", "/market/status", timeout=5, quiet=True) or {}

    def load_performance_data(self) -> pd.DataFrame:
        """Load saved performance metrics from the API or artifact file."""
        api_payload = self.api_request("GET", "/models/performance", timeout=10, quiet=True)
        if api_payload and api_payload.get('models'):
            df = pd.DataFrame(api_payload['models'])
        else:
            risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
            df = pd.read_csv(risk_summary_path) if risk_summary_path.exists() else pd.DataFrame()

        if df.empty:
            return df

        normalized = df.copy()
        normalized['Annual_Return_Pct'] = normalized['Annual_Return'].apply(percent_value)
        normalized['Annual_Volatility_Pct'] = normalized['Annual_Volatility'].apply(percent_value)
        normalized['Win_Rate_Pct'] = normalized['Win_Rate'].apply(percent_value)
        if 'Benchmark_Annual_Return' in normalized.columns:
            normalized['Benchmark_Annual_Return_Pct'] = normalized['Benchmark_Annual_Return'].apply(percent_value)
        if 'Excess_Annual_Return' in normalized.columns:
            normalized['Excess_Annual_Return_Pct'] = normalized['Excess_Annual_Return'].apply(percent_value)
        if 'Max_Drawdown_Percent' in normalized.columns:
            normalized['Drawdown_Pct'] = normalized['Max_Drawdown_Percent'].apply(percent_value)
        else:
            normalized['Drawdown_Pct'] = normalized['Max_Drawdown'].apply(percent_value)

        return normalized.sort_values('Sharpe_Ratio', ascending=False).reset_index(drop=True)

    def load_benchmark_data(self) -> pd.DataFrame:
        """Load saved benchmark metrics from the API or artifact file."""
        api_payload = self.api_request("GET", "/models/performance", timeout=10, quiet=True)
        if api_payload and api_payload.get('benchmarks'):
            df = pd.DataFrame(api_payload['benchmarks'])
        else:
            benchmark_path = self.config.PROCESSED_DATA_PATH / "day11_benchmark_summary.csv"
            df = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()

        if df.empty:
            return df

        normalized = df.copy()
        normalized['Annual_Return_Pct'] = normalized['Annual_Return'].apply(percent_value)
        normalized['Annual_Volatility_Pct'] = normalized['Annual_Volatility'].apply(percent_value)
        normalized['Win_Rate_Pct'] = normalized['Win_Rate'].apply(percent_value)
        if 'Benchmark_Annual_Return' in normalized.columns:
            normalized['Benchmark_Annual_Return_Pct'] = normalized['Benchmark_Annual_Return'].apply(percent_value)
        if 'Excess_Annual_Return' in normalized.columns:
            normalized['Excess_Annual_Return_Pct'] = normalized['Excess_Annual_Return'].apply(percent_value)
        if 'Max_Drawdown_Percent' in normalized.columns:
            normalized['Drawdown_Pct'] = normalized['Max_Drawdown_Percent'].apply(percent_value)
        else:
            normalized['Drawdown_Pct'] = normalized['Max_Drawdown'].apply(percent_value)

        return normalized.sort_values('Sharpe_Ratio', ascending=False).reset_index(drop=True)

    def load_validation_results(self) -> Dict[str, Any]:
        """Load saved validation outputs."""
        path = self.config.PROCESSED_DATA_PATH / "day10_validation_results.json"
        if not path.exists():
            return {}

        with open(path, 'r') as handle:
            return json.load(handle)

    def load_feature_importance(self) -> pd.DataFrame:
        """Load saved feature importance outputs."""
        path = self.config.PROCESSED_DATA_PATH / "feature_importance_analysis.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        return df.sort_values(['model', 'importance'], ascending=[True, False]).reset_index(drop=True)

    def load_risk_analysis(self) -> Dict[str, Any]:
        """Load saved risk analysis artifacts."""
        path = self.config.PROCESSED_DATA_PATH / "day11_risk_analysis.json"
        if not path.exists():
            return {}

        with open(path, 'r') as handle:
            return json.load(handle)

    def load_retraining_triggers(self) -> List[Dict[str, Any]]:
        """Load recorded retraining triggers if any exist."""
        path = self.config.PROCESSED_DATA_PATH / "retraining_triggers.json"
        if not path.exists():
            return []

        with open(path, 'r') as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, list) else []

    def get_target_stocks(self) -> List[str]:
        """Load the stock universe from the processed artifact."""
        stocks_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
        if stocks_path.exists():
            with open(stocks_path, 'r') as handle:
                return [line.strip() for line in handle.readlines() if line.strip()][:10]

        return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD']

    def build_file_audit(self) -> pd.DataFrame:
        """Summarize live artifact availability and provenance."""
        required = [
            ("Performance Summary", self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"),
            ("Benchmark Summary", self.config.PROCESSED_DATA_PATH / "day11_benchmark_summary.csv"),
            ("Validation Results", self.config.PROCESSED_DATA_PATH / "day10_validation_results.json"),
            ("Feature Importance", self.config.PROCESSED_DATA_PATH / "feature_importance_analysis.csv"),
            ("Target Stocks", self.config.PROCESSED_DATA_PATH / "target_stocks.txt"),
            ("Risk Analysis", self.config.PROCESSED_DATA_PATH / "day11_risk_analysis.json"),
        ]

        rows: List[Dict[str, Any]] = []
        for label, path in required:
            exists = path.exists()
            rows.append({
                'Artifact': label,
                'Status': 'Live' if exists else 'Missing',
                'Source': str(path.relative_to(self.config.PROJECT_ROOT)),
                'Modified': datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else "N/A"
            })

        rows.append({
            'Artifact': 'Model Files',
            'Status': 'Live',
            'Source': 'models/**/*.joblib',
            'Modified': str(len(list((self.config.PROJECT_ROOT / "models").rglob("*.joblib"))))
        })

        return pd.DataFrame(rows)

    def run_local_cycle(self, symbols: List[str]) -> Dict[str, Any]:
        """Fallback to local runtime if the backend is unavailable."""
        self.ensure_local_runtime()

        async def _execute() -> Dict[str, Any]:
            original_method = self.prediction_engine.get_target_stocks
            self.prediction_engine.get_target_stocks = lambda: symbols
            try:
                return await self.prediction_engine.run_realtime_cycle()
            finally:
                self.prediction_engine.get_target_stocks = original_method

        return asyncio.run(_execute())

    def run_prediction_cycle(self, symbols: List[str]) -> Dict[str, Any]:
        """Run detailed predictions through the backend, with a local fallback."""
        payload = {'symbols': symbols, 'include_confidence': True, 'include_alerts': True}
        results = self.api_request("POST", "/predict/detailed", payload=payload, timeout=120, quiet=True)

        if results:
            results['_source'] = 'backend_api'
            return results

        fallback = self.run_local_cycle(symbols)
        fallback['_source'] = 'local_runtime'
        fallback['requested_symbols'] = symbols
        fallback['models_loaded'] = len(self.prediction_engine.models) if self.prediction_engine else 0
        return fallback

    def run_portfolio_optimization(
        self,
        symbols: List[str],
        optimization_method: str,
        target_return: float,
    ) -> Dict[str, Any]:
        """Run portfolio optimization via backend or local fallback."""
        if optimization_method == "equal_weight":
            weights = np.array([1.0 / len(symbols)] * len(symbols))
            return {
                'weights': dict(zip(symbols, weights)),
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'optimization_method': 'equal_weight',
                '_source': 'dashboard_equal_weight'
            }

        payload = {
            'symbols': symbols,
            'optimization_method': optimization_method,
            'target_return': target_return,
            'risk_tolerance': 'medium'
        }
        result = self.api_request("POST", "/portfolio/optimize", payload=payload, timeout=120, quiet=True)
        if result:
            result['_source'] = 'backend_api'
            return result

        self.ensure_local_runtime()
        features_df = self.risk_framework.load_feature_data()
        filtered = features_df[features_df['Ticker'].isin(symbols)].copy()
        X, _, _ = self.risk_framework.prepare_portfolio_data(filtered)
        predictions_df = self.risk_framework.generate_predictions(self.risk_framework.load_best_models(), X, filtered)

        if optimization_method == "markowitz":
            result = self.risk_framework.portfolio_optimization_markowitz(predictions_df, target_return=target_return)
            if not result or not result.get('success', False):
                result = self.risk_framework.risk_parity_portfolio(predictions_df)
                result['optimization_method'] = 'markowitz_fallback_risk_parity'
        else:
            result = self.risk_framework.risk_parity_portfolio(predictions_df)

        result['_source'] = 'local_runtime'
        result['weights'] = dict(zip(result['stocks'], result['weights'])) if 'stocks' in result else {}
        result['expected_return'] = result.get('expected_return', result.get('portfolio_return', 0))
        result['volatility'] = result.get('volatility', result.get('portfolio_volatility', 0))
        return result

    def render_metric_tile(self, label: str, value: str, meta: str) -> None:
        """Render a branded metric tile."""
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-meta">{meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_signal_card(self, row: Dict[str, Any]) -> None:
        """Render a directional signal card."""
        direction = row['Direction']
        tone = "signal-neutral"
        if direction == "BUY":
            tone = "signal-long"
        elif direction == "SELL":
            tone = "signal-short"

        st.markdown(
            f"""
            <div class="signal-card {tone}">
                <div class="signal-symbol">{row['Symbol']}</div>
                <div class="signal-direction">{direction}</div>
                <div class="signal-stat">Signal {row['Prediction']:+.4f}</div>
                <div class="signal-stat">Confidence band {row['Lower']:+.4f} to {row['Upper']:+.4f}</div>
                <div class="signal-stat">Model agreement {row['ModelAgreement']:.2f}</div>
                <div class="signal-stat">Models used {row['ModelsUsed']}</div>
                <div class="signal-stat">Position bias {row['PositionDirection']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_hero(self, selected_stocks: List[str]) -> None:
        """Render top-level dashboard branding and runtime state."""
        health = self.backend_health()
        market = self.market_status()
        performance = self.load_performance_data()
        best_model = performance.iloc[0] if not performance.empty else None

        connection_state = "API Live" if health else "Local Fallback"
        connection_class = "status-online" if health else "status-offline"
        market_open = market.get('is_open', False)

        ticker_markup = "".join(
            f'<span class="ticker-chip">{symbol}</span>' for symbol in selected_stocks
        )

        subtitle = (
            "Live signal generation, portfolio construction, drift surveillance, and saved validation "
            "artifacts, presented as a single stock-engine control room."
        )

        best_model_text = best_model['Model'] if best_model is not None else "Awaiting artifacts"
        sharpe_text = f"{best_model['Sharpe_Ratio']:.2f}" if best_model is not None else "N/A"
        market_text = "Open" if market_open else "Closed"

        st.markdown(
            f"""
            <section class="hero-shell">
                <div class="hero-kicker">Live Quant Stack</div>
                <h1 class="hero-title">Stock Market<br>Prediction Engine</h1>
                <div class="hero-subtitle">{subtitle}</div>
                <div class="ticker-strip">
                    <span class="status-pill {connection_class}">{connection_state}</span>
                    <span class="status-pill">{market_text}</span>
                    <span class="status-pill">Top model {best_model_text}</span>
                    <span class="status-pill">Net Sharpe {sharpe_text}</span>
                </div>
                <div class="ticker-strip">{ticker_markup}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self) -> Dict[str, Any]:
        """Render dashboard controls."""
        health = self.backend_health()
        available_stocks = self.get_target_stocks()

        with st.sidebar:
            st.markdown("## Control Deck")
            st.caption("Configure the live scan and portfolio construction path.")

            page = st.selectbox(
                "Workspace",
                [
                    "Overview",
                    "Live Predictions",
                    "Performance Analytics",
                    "Portfolio Optimizer",
                    "Alert Center",
                    "Model Insights",
                ],
            )

            selected_stocks = st.multiselect(
                "Signal Universe",
                available_stocks,
                default=available_stocks[:4],
            )

            target_return = st.number_input(
                "Target Return (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.5,
            ) / 100

            optimization_mode = st.selectbox(
                "Portfolio Mode",
                ["markowitz", "risk_parity", "equal_weight"],
                format_func=lambda item: item.replace("_", " ").title(),
            )

            st.markdown("---")
            st.caption(f"Backend URL: `{self.api_base_url}`")
            st.caption(f"Backend health: {'online' if health else 'offline'}")
            st.caption("Frontend uses real API responses when available and local runtime only as fallback.")

        return {
            'page': page,
            'selected_stocks': selected_stocks,
            'target_return': target_return,
            'optimization_mode': optimization_mode,
        }

    def render_overview_page(self, selected_stocks: List[str]) -> None:
        """Render the overview command center."""
        performance = self.load_performance_data()
        benchmarks = self.load_benchmark_data()
        validation = self.load_validation_results()
        market = self.market_status()
        audit_df = self.build_file_audit()
        curve_df = strategy_curve_from_validation(validation)
        feature_df = self.load_feature_importance()
        latest_cycle = st.session_state.get('latest_cycle') or {}
        latest_predictions = latest_cycle.get('predictions', {})

        cols = st.columns(4)
        with cols[0]:
            if not performance.empty:
                self.render_metric_tile("Best Net Sharpe", f"{performance.iloc[0]['Sharpe_Ratio']:.2f}", performance.iloc[0]['Model'])
        with cols[1]:
            best_win = performance['Win_Rate_Pct'].max() if not performance.empty else 0
            best_win_model = (
                performance.loc[performance['Win_Rate_Pct'].idxmax(), 'Model']
                if not performance.empty else "No artifact"
            )
            self.render_metric_tile("Best Win Rate", f"{best_win:.1f}%", best_win_model)
        with cols[2]:
            stability = validation.get('stability', {}).get('stability_rating', 'N/A')
            stability_score = validation.get('stability', {}).get('overall_stability_score', 0)
            self.render_metric_tile("Stability", stability, f"Score {stability_score:.2f}")
        with cols[3]:
            session = market.get('trading_session', 'closed')
            cycle_metrics = latest_cycle.get('performance_metrics', {})
            cycle_meta = (
                f"Last cycle {cycle_metrics.get('cycle_time_seconds', 0):.2f}s"
                if cycle_metrics else "No live scan yet"
            )
            self.render_metric_tile("Market Session", session.title(), cycle_meta)

        st.markdown(
            f"""
            <div class="source-callout">
                Metrics below are sourced from live backend endpoints and saved artifacts. Performance tiles use a
                {self.config.FORECAST_HORIZON_DAYS}-day forecast horizon and assume {self.config.DEFAULT_TRANSACTION_COST_BPS:.0f} bps
                transaction cost per evaluation period.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not performance.empty and not benchmarks.empty:
            model_row = performance.iloc[0]
            benchmark_row = benchmarks.iloc[0]
            sharpe_edge = model_row['Sharpe_Ratio'] - benchmark_row['Sharpe_Ratio']
            return_edge = model_row['Annual_Return_Pct'] - benchmark_row['Annual_Return_Pct']
            st.markdown(
                f"""
                <div class="source-callout">
                    Benchmark hurdle: top model <strong>{model_row['Model']}</strong> Sharpe {model_row['Sharpe_Ratio']:.2f}
                    versus best naive benchmark <strong>{benchmark_row['Benchmark']}</strong> Sharpe {benchmark_row['Sharpe_Ratio']:.2f}.
                    Current edge: {sharpe_edge:+.2f} Sharpe and {return_edge:+.1f}% annual return.
                </div>
                """,
                unsafe_allow_html=True,
            )

        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown('<div class="panel-title">Model Performance Engine</div>', unsafe_allow_html=True)
            if not performance.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=performance['Model'],
                    y=performance['Sharpe_Ratio'],
                    marker_color=performance['Sharpe_Ratio'],
                    marker_colorscale='Viridis',
                    name='Sharpe Ratio',
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown('<div class="panel-title">Artifact Provenance</div>', unsafe_allow_html=True)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="panel-title">Walk-Forward Strategy Curve</div>', unsafe_allow_html=True)
            if not curve_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df['Date'],
                    y=curve_df['StrategyValue'],
                    mode='lines',
                    line=dict(color='#62f7a6', width=2.5),
                    name='Strategy value',
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=360,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(type='log', title='Strategy Index'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Validation curve unavailable.")

        with right:
            st.markdown('<div class="panel-title">Feature Drivers</div>', unsafe_allow_html=True)
            if not feature_df.empty:
                top_features = (
                    feature_df.groupby('feature', as_index=False)['importance']
                    .mean()
                    .sort_values('importance', ascending=True)
                    .tail(10)
                )
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker_color='#f4b56a',
                    name='Avg importance',
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=360,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance artifact unavailable.")

        if latest_predictions:
            st.markdown('<div class="panel-title">Latest Signal Snapshot</div>', unsafe_allow_html=True)
            snapshot_rows = self.prediction_rows(latest_cycle)
            preview_cols = st.columns(min(3, len(snapshot_rows)))
            for index, row in enumerate(snapshot_rows[:3]):
                with preview_cols[index]:
                    self.render_signal_card(row)

    def prediction_rows(self, cycle_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten prediction results into dashboard-friendly rows."""
        portfolio_impact = cycle_results.get('portfolio_impact', {}).get('recommendations', {})
        rows: List[Dict[str, Any]] = []

        for symbol, data in cycle_results.get('predictions', {}).items():
            primary = data.get('primary', {})
            prediction_value = safe_float(primary.get('prediction'))
            interval = data.get('confidence_interval', {})
            position = portfolio_impact.get(symbol, {})
            rows.append({
                'Symbol': symbol,
                'Prediction': prediction_value,
                'Direction': "BUY" if prediction_value > 0.001 else "SELL" if prediction_value < -0.001 else "HOLD",
                'Confidence': primary.get('confidence', 'medium').upper(),
                'Lower': safe_float(interval.get('lower', prediction_value)),
                'Upper': safe_float(interval.get('upper', prediction_value)),
                'ModelAgreement': safe_float(data.get('model_agreement')),
                'ModelDispersion': safe_float(data.get('model_dispersion')),
                'ModelsUsed': int(data.get('models_used', 1)),
                'Timestamp': data.get('timestamp'),
                'PositionDirection': position.get('direction', 'N/A'),
                'PositionSize': safe_float(position.get('position_size')),
                'ModelBreakdown': {
                    key: safe_float(value)
                    for key, value in data.items()
                    if key not in {'primary', 'timestamp', 'symbol', 'confidence_interval', 'model_dispersion', 'model_agreement', 'models_used'}
                    and isinstance(value, (int, float))
                }
            })

        return sorted(rows, key=lambda item: abs(item['Prediction']), reverse=True)

    def render_predictions_page(self, selected_stocks: List[str]) -> None:
        """Render live prediction generation and inspection."""
        st.markdown('<div class="panel-title">Real-Time Signal Deck</div>', unsafe_allow_html=True)

        if not selected_stocks:
            st.warning("Select at least one stock in the sidebar.")
            return

        refresh_cols = st.columns([1, 1, 4])
        with refresh_cols[0]:
            run_scan = st.button("Run Live Scan")
        with refresh_cols[1]:
            reuse_latest = st.button("Reuse Latest")

        if run_scan:
            with st.spinner("Scanning live market data and generating predictions..."):
                st.session_state.latest_cycle = self.run_prediction_cycle(selected_stocks)
                st.session_state.latest_cycle_symbols = selected_stocks
        elif reuse_latest and st.session_state.get('latest_cycle_symbols') != selected_stocks:
            st.info("Latest cycle was run for a different stock set. Run a fresh scan for these symbols.")

        cycle = st.session_state.get('latest_cycle') or {}
        if not cycle:
            st.info("Run a live scan to generate signals.")
            return

        source = cycle.get('_source', 'unknown')
        st.caption(f"Prediction source: `{source}`")

        metrics = cycle.get('performance_metrics', {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("Cycle Time", f"{metrics.get('cycle_time_seconds', 0):.2f}s")
        with cols[1]:
            st.metric("Success Rate", format_percent(metrics.get('success_rate', 0)))
        with cols[2]:
            st.metric("Alerts Triggered", str(metrics.get('alerts_triggered', 0)))
        with cols[3]:
            st.metric("Retraining Triggers", str(metrics.get('retraining_triggers', 0)))

        rows = self.prediction_rows(cycle)
        cycle_errors = cycle.get('errors', [])
        if cycle_errors:
            st.warning("Live scan returned partial data for some symbols.")
            st.code("\n".join(cycle_errors))

        prediction_df = prediction_dataframe(rows)
        if prediction_df.empty:
            st.info("No predictions were generated for the selected symbols.")
            return

        card_cols = st.columns(min(3, len(rows)) or 1)
        for index, row in enumerate(rows[:3]):
            with card_cols[index]:
                self.render_signal_card(row)

        st.dataframe(prediction_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prediction_df['Symbol'],
            y=prediction_df['Prediction'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=prediction_df['Upper'] - prediction_df['Prediction'],
                arrayminus=prediction_df['Prediction'] - prediction_df['Lower'],
            ),
            marker_color=[
                '#62f7a6' if value > 0 else '#ff7a63' if value < 0 else '#f4b56a'
                for value in prediction_df['Prediction']
            ],
            name='Primary signal',
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=430,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_symbol = st.selectbox("Model Breakdown", prediction_df['Symbol'])
        symbol_row = next((row for row in rows if row['Symbol'] == selected_symbol), None)
        if symbol_row and symbol_row['ModelBreakdown']:
            breakdown = pd.DataFrame(
                list(symbol_row['ModelBreakdown'].items()),
                columns=['Model', 'Prediction']
            ).sort_values('Prediction')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=breakdown['Prediction'],
                y=breakdown['Model'],
                orientation='h',
                marker_color='#f4b56a',
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=360,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        portfolio_impact = cycle.get('portfolio_impact', {})
        if portfolio_impact:
            st.markdown('<div class="panel-title">Portfolio Impact</div>', unsafe_allow_html=True)
            st.caption(
                f"Risk level: `{portfolio_impact.get('risk_level', 'unknown')}` | "
                f"Exposure utilization: {format_percent(portfolio_impact.get('utilization', 0))}"
            )

    def render_performance_page(self) -> None:
        """Render saved performance analytics and validation metrics."""
        performance = self.load_performance_data()
        benchmarks = self.load_benchmark_data()
        validation = self.load_validation_results()
        curve_df = strategy_curve_from_validation(validation)

        if performance.empty:
            st.warning("Performance artifacts are unavailable.")
            return

        left, right = st.columns([0.95, 1.05])
        with left:
            st.markdown('<div class="panel-title">Model Ranking</div>', unsafe_allow_html=True)
            ranking_columns = ['Model', 'Sharpe_Ratio', 'Annual_Return_Pct', 'Win_Rate_Pct', 'Drawdown_Pct']
            rename_map = {
                'Annual_Return_Pct': 'Annual Return (%)',
                'Win_Rate_Pct': 'Win Rate (%)',
                'Drawdown_Pct': 'Max Drawdown (%)'
            }
            if 'Benchmark_Annual_Return_Pct' in performance.columns:
                ranking_columns.append('Benchmark_Annual_Return_Pct')
                rename_map['Benchmark_Annual_Return_Pct'] = 'Benchmark Return (%)'
            if 'Signal_Accuracy' in performance.columns:
                ranking_columns.append('Signal_Accuracy')
                rename_map['Signal_Accuracy'] = 'Signal Accuracy (%)'
            ranking_df = performance[ranking_columns].rename(columns=rename_map)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown('<div class="panel-title">Risk vs Return Surface</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=performance['Drawdown_Pct'].abs(),
                y=performance['Annual_Return_Pct'],
                mode='markers+text',
                text=performance['Model'],
                textposition='top center',
                marker=dict(
                    size=np.clip(performance['Sharpe_Ratio'] * 9, 10, 38),
                    color=performance['Sharpe_Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                )
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=430,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title='Absolute Max Drawdown (%)',
                yaxis_title='Annual Return (%)',
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
            <div class="source-callout">
                Performance metrics are cost-adjusted and annualized using a {self.config.FORECAST_HORIZON_DAYS}-day forecast horizon.
                The benchmark suite uses the same evaluation dates and includes equal-weight, momentum, and simple mean-reversion baselines.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not benchmarks.empty:
            left, right = st.columns([1.0, 1.0])
            with left:
                st.markdown('<div class="panel-title">Benchmark Hurdle Table</div>', unsafe_allow_html=True)
                benchmark_table = benchmarks[[
                    'Benchmark',
                    'Strategy_Type',
                    'Sharpe_Ratio',
                    'Annual_Return_Pct',
                    'Win_Rate_Pct',
                    'Drawdown_Pct',
                ]].rename(columns={
                    'Strategy_Type': 'Type',
                    'Annual_Return_Pct': 'Annual Return (%)',
                    'Win_Rate_Pct': 'Win Rate (%)',
                    'Drawdown_Pct': 'Max Drawdown (%)',
                })
                st.dataframe(benchmark_table, use_container_width=True, hide_index=True)

            with right:
                st.markdown('<div class="panel-title">Models vs Benchmarks</div>', unsafe_allow_html=True)
                comparison_df = pd.concat([
                    performance[['Model', 'Sharpe_Ratio']].rename(columns={'Model': 'Name'}).assign(Group='Model'),
                    benchmarks[['Benchmark', 'Sharpe_Ratio']].rename(columns={'Benchmark': 'Name'}).assign(Group='Benchmark'),
                ], ignore_index=True)
                comparison_df = comparison_df.sort_values('Sharpe_Ratio', ascending=False)
                fig = go.Figure()
                palette = comparison_df['Group'].map({'Model': '#62f7a6', 'Benchmark': '#f4b56a'})
                fig.add_trace(go.Bar(
                    x=comparison_df['Name'],
                    y=comparison_df['Sharpe_Ratio'],
                    marker_color=palette,
                    text=comparison_df['Group'],
                    textposition='outside',
                    name='Sharpe ratio',
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=420,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_tickangle=-30,
                    yaxis_title='Sharpe Ratio',
                )
                st.plotly_chart(fig, use_container_width=True)

        if not curve_df.empty:
            st.markdown('<div class="panel-title">Walk-Forward Equity Curve</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df['Date'],
                y=curve_df['StrategyValue'],
                mode='lines',
                line=dict(color='#62f7a6', width=2.5),
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(type='log', title='Strategy Index'),
            )
            st.plotly_chart(fig, use_container_width=True)

        walk_forward = validation.get('walk_forward', {})
        out_of_sample = validation.get('out_of_sample', {})
        stability = validation.get('stability', {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("Walk-Forward R2", f"{walk_forward.get('overall_r2', 0):.3f}")
        with cols[1]:
            st.metric("Mean Fold R2", f"{walk_forward.get('mean_fold_r2', 0):.3f}")
        with cols[2]:
            st.metric("Out-of-Sample R2", f"{out_of_sample.get('r2', 0):.3f}")
        with cols[3]:
            st.metric("Stability Rating", str(stability.get('stability_rating', 'N/A')))

    def render_portfolio_page(
        self,
        selected_stocks: List[str],
        optimization_method: str,
        target_return: float,
    ) -> None:
        """Render portfolio optimization results."""
        st.markdown('<div class="panel-title">Portfolio Construction</div>', unsafe_allow_html=True)

        if len(selected_stocks) < 2:
            st.warning("Select at least two stocks in the sidebar.")
            return

        if st.button("Optimize Portfolio"):
            with st.spinner("Optimizing allocation using real model outputs..."):
                st.session_state.latest_portfolio = self.run_portfolio_optimization(
                    selected_stocks, optimization_method, target_return
                )
                st.session_state.latest_portfolio_request = {
                    'symbols': selected_stocks,
                    'optimization_method': optimization_method,
                    'target_return': target_return,
                }

        portfolio = st.session_state.get('latest_portfolio') or {}
        if not portfolio:
            st.info("Run portfolio optimization to inspect weights and risk metrics.")
            return

        st.caption(
            f"Portfolio source: `{portfolio.get('_source', 'unknown')}` | "
            f"Method: `{portfolio.get('optimization_method', optimization_method)}`"
        )

        cols = st.columns(3)
        with cols[0]:
            st.metric("Expected Return", format_percent(portfolio.get('expected_return', 0)))
        with cols[1]:
            st.metric("Volatility", format_percent(portfolio.get('volatility', 0)))
        with cols[2]:
            st.metric("Sharpe Ratio", f"{safe_float(portfolio.get('sharpe_ratio')):.2f}")

        weights = portfolio.get('weights', {})
        weights_df = pd.DataFrame({
            'Symbol': list(weights.keys()),
            'Weight': [safe_float(value) for value in weights.values()]
        }).sort_values('Weight', ascending=False)
        weights_df['WeightPct'] = weights_df['Weight'] * 100

        left, right = st.columns([0.9, 1.1])
        with left:
            fig = go.Figure(data=[go.Pie(
                labels=weights_df['Symbol'],
                values=weights_df['WeightPct'],
                hole=0.58,
                marker=dict(colors=['#62f7a6', '#f4b56a', '#62c3ff', '#ff7a63', '#dce38f', '#7fd6bb']),
            )])
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=430,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.dataframe(
                weights_df.rename(columns={'WeightPct': 'Weight (%)'}),
                use_container_width=True,
                hide_index=True,
            )

    def render_alerts_page(self, selected_stocks: List[str]) -> None:
        """Render live alert feed and retraining triggers."""
        st.markdown('<div class="panel-title">Alert Scanner</div>', unsafe_allow_html=True)

        if st.button("Refresh Alert Feed"):
            with st.spinner("Refreshing alert feed from the live signal engine..."):
                st.session_state.latest_cycle = self.run_prediction_cycle(selected_stocks or self.get_target_stocks()[:5])
                st.session_state.latest_cycle_symbols = selected_stocks

        cycle = st.session_state.get('latest_cycle') or {}
        alerts = cycle.get('alerts', [])
        triggers = cycle.get('retraining_triggers', [])

        if not alerts and not triggers:
            st.info("No live alerts in session. Run a live scan from the Predictions page or refresh here.")
            return

        if alerts:
            alerts_df = pd.DataFrame(alerts)
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Alerts", str(len(alerts_df)))
            with cols[1]:
                st.metric("High Confidence", str((alerts_df['type'] == 'high_confidence_signal').sum()))
            with cols[2]:
                st.metric("Risk Limit", str((alerts_df['type'] == 'risk_limit_exceeded').sum()))

            st.dataframe(alerts_df, use_container_width=True, hide_index=True)

            counts = alerts_df.groupby(['symbol', 'type']).size().reset_index(name='count')
            fig = go.Figure()
            for alert_type in counts['type'].unique():
                subset = counts[counts['type'] == alert_type]
                fig.add_trace(go.Bar(x=subset['symbol'], y=subset['count'], name=alert_type))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=360,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        if triggers:
            st.markdown('<div class="panel-title">Automated Retraining Triggers</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(triggers), use_container_width=True, hide_index=True)

    def render_model_insights_page(self) -> None:
        """Render model lineup, feature importance, and validation quality signals."""
        performance = self.load_performance_data()
        feature_df = self.load_feature_importance()
        validation = self.load_validation_results()
        triggers = self.load_retraining_triggers()
        health = self.backend_health()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Models Available", str(len(performance)))
        with cols[1]:
            st.metric("Backend Loaded", str(health.get('models_loaded', 0)))
        with cols[2]:
            st.metric("WF Predictions", str(validation.get('walk_forward', {}).get('total_predictions', 0)))
        with cols[3]:
            st.metric("Saved Drift Triggers", str(len(triggers)))

        st.markdown(
            """
            <div class="source-callout">
                This page is wired to live API metadata, saved feature importance artifacts, validation outputs,
                and retraining trigger files. The static placeholder explanation content was removed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([0.95, 1.05])
        with left:
            st.markdown('<div class="panel-title">Model Roster</div>', unsafe_allow_html=True)
            roster = performance[['Model', 'Sharpe_Ratio', 'Annual_Return_Pct', 'Win_Rate_Pct']].rename(columns={
                'Annual_Return_Pct': 'Annual Return (%)',
                'Win_Rate_Pct': 'Win Rate (%)',
            })
            st.dataframe(roster, use_container_width=True, hide_index=True)

        with right:
            st.markdown('<div class="panel-title">Validation Quality</div>', unsafe_allow_html=True)
            stability = validation.get('stability', {})
            walk_forward = validation.get('walk_forward', {})
            st.write(
                {
                    'stability_rating': stability.get('stability_rating'),
                    'overall_stability_score': stability.get('overall_stability_score'),
                    'walk_forward_r2': walk_forward.get('overall_r2'),
                    'mean_fold_r2': walk_forward.get('mean_fold_r2'),
                    'out_of_sample_r2': validation.get('out_of_sample', {}).get('r2'),
                }
            )

        if not feature_df.empty:
            model_options = sorted(feature_df['model'].unique())
            selected_model = st.selectbox("Feature Importance Model", model_options)
            top_features = (
                feature_df[feature_df['model'] == selected_model]
                .sort_values('importance', ascending=True)
                .tail(12)
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='#62c3ff',
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self) -> None:
        """Render the full dashboard."""
        inject_theme()

        sidebar_state = self.render_sidebar()
        selected_stocks = sidebar_state['selected_stocks']
        self.render_hero(selected_stocks)

        page = sidebar_state['page']
        if page == "Overview":
            self.render_overview_page(selected_stocks)
        elif page == "Live Predictions":
            self.render_predictions_page(selected_stocks)
        elif page == "Performance Analytics":
            self.render_performance_page()
        elif page == "Portfolio Optimizer":
            self.render_portfolio_page(
                selected_stocks,
                sidebar_state['optimization_mode'],
                sidebar_state['target_return'],
            )
        elif page == "Alert Center":
            self.render_alerts_page(selected_stocks)
        elif page == "Model Insights":
            self.render_model_insights_page()

        st.markdown("---")
        st.markdown(
            '<div class="footer-note">Stock Market Prediction Engine | Dynamic data only | Streamlit control room</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    """Main dashboard entry point."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
