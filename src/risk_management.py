import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import joblib
from datetime import datetime, timedelta
import json

from .config import Config
from .consensus_model import ConsensusMetaEnsemble, load_model_scores

class RiskManagementFramework:
    """Comprehensive Risk Management and Portfolio Optimization Framework"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.portfolio_weights = {}
        self.risk_metrics = {}
        self.portfolio_returns = []
        self.position_sizes = {}

    def periods_per_year(self) -> float:
        """Number of forecast periods in a trading year."""
        return self.config.periods_per_year()

    def period_risk_free_rate(self) -> float:
        """Risk-free rate aligned to the forecast horizon."""
        return self.config.period_risk_free_rate()

    def aggregate_period_returns(
        self,
        dates: pd.Series,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        transaction_cost_bps: Optional[float] = None,
    ) -> pd.DataFrame:
        """Aggregate row-level 5-day returns into a per-date evaluation series."""
        cost_bps = (
            self.config.DEFAULT_TRANSACTION_COST_BPS
            if transaction_cost_bps is None else transaction_cost_bps
        )
        cost_rate = cost_bps / 10000

        period_df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'StrategyReturn': strategy_returns,
            'BenchmarkReturn': benchmark_returns,
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if period_df.empty:
            return period_df

        period_df = (
            period_df.groupby('Date', as_index=False)[['StrategyReturn', 'BenchmarkReturn']]
            .mean()
            .sort_values('Date')
        )
        period_df['NetStrategyReturn'] = period_df['StrategyReturn'] - cost_rate
        period_df['TransactionCostBps'] = cost_bps
        return period_df

    def normalize_return_array(self, returns: np.ndarray) -> np.ndarray:
        """Normalize percentage-like return arrays into ratio space."""
        arr = np.asarray(returns, dtype=float)
        finite = arr[np.isfinite(arr)]

        if len(finite) == 0:
            return np.zeros_like(arr, dtype=float)

        scale = 100.0 if np.nanpercentile(np.abs(finite), 95) > 1.5 else 1.0
        return arr / scale

    def prediction_positions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert continuous predictions into long/short/flat positions."""
        normalized_predictions = self.normalize_return_array(predictions)
        threshold = self.config.signal_threshold_ratio()
        positions = np.where(
            normalized_predictions > threshold,
            1.0,
            np.where(normalized_predictions < -threshold, -1.0, 0.0),
        )
        return positions
        
    def load_validation_results(self) -> Dict:
        """Load validation results from Day 10"""
        logger.info("Loading validation results from Day 10...")
        
        try:
            results_path = self.config.PROCESSED_DATA_PATH / "day10_validation_results.json"
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    validation_data = json.load(f)
                
                # Handle nested structure
                if 'validation_results' in validation_data:
                    validation_results = validation_data['validation_results']
                else:
                    validation_results = validation_data
                
                logger.info(f"Loaded validation results for {len(validation_results)} models")
                return validation_results
            else:
                logger.error(f"Validation results file not found: {results_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load validation results: {e}")
            return {}
    
    def load_best_models(self) -> Dict[str, Any]:
        """Load the best performing models from validation"""
        logger.info("Loading best performing models...")
        
        models = {}
        
        # Load ensemble models (typically best performers)
        ensemble_dir = self.config.PROJECT_ROOT / "models" / "ensemble"
        ensemble_files = {
            'SimpleAverage': ensemble_dir / "simple_average_ensemble.joblib",
            'VotingRegressor': ensemble_dir / "voting_regressor_ensemble.joblib", 
            'StackedEnsemble': ensemble_dir / "stacked_ensemble_ensemble.joblib"
        }
        
        # Load individual models as backup
        models_dir = self.config.PROJECT_ROOT / "models"
        advanced_dir = models_dir / "advanced"
        individual_files = {
            'XGBoost': advanced_dir / "regression_xgboost_optimized.joblib",
            'LightGBM': advanced_dir / "regression_lightgbm_optimized.joblib",
            'RandomForest': models_dir / "regression_random_forest.joblib"
        }
        
        # Try loading ensemble models first
        for name, path in ensemble_files.items():
            if path.exists():
                try:
                    models[f"Ensemble_{name}"] = joblib.load(path)
                    logger.info(f"Loaded ensemble model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load ensemble {name}: {e}")
        
        # Load individual models
        for name, path in individual_files.items():
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                    logger.info(f"Loaded individual model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load individual {name}: {e}")

        consensus_components = {
            name: model for name, model in models.items()
            if name in {
                'Ensemble_VotingRegressor',
                'Ensemble_SimpleAverage',
                'XGBoost',
                'LightGBM',
                'RandomForest',
            }
        }
        if len(consensus_components) >= 2:
            models['ConsensusMetaEnsemble'] = ConsensusMetaEnsemble(
                consensus_components,
                model_scores=load_model_scores(self.config),
            )
            logger.info("Loaded derived model: ConsensusMetaEnsemble")
        
        logger.info(f"Loaded {len(models)} models for portfolio optimization")
        return models
    
    def load_feature_data(self) -> pd.DataFrame:
        """Load feature data for portfolio analysis"""
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        
        if not features_path.exists():
            logger.error(f"Feature data not found: {features_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        logger.info(f"Loaded feature data: {len(df)} records, {df.shape[1]} features")
        return df
    
    def prepare_portfolio_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Prepare data for portfolio analysis"""
        logger.info("Preparing portfolio data...")
        
        # Prepare features (same as validation)
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['return_5d'].fillna(df['return_5d'].median())
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_cols
    
    def generate_predictions(self, models: Dict, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from all models"""
        logger.info("Generating predictions from all models...")
        
        benchmark_cols = [
            col for col in ['momentum_1d', 'momentum_5d', 'momentum_10d', 'momentum_20d', 'daily_return']
            if col in df.columns
        ]
        predictions_df = df[['Date', 'Ticker', 'Close', 'return_5d', *benchmark_cols]].copy()
        
        for model_name, model in models.items():
            logger.info(f"Generating predictions with {model_name}...")
            
            try:
                if hasattr(model, 'predict'):
                    # Standard sklearn-like model
                    y_pred = model.predict(X)
                elif isinstance(model, tuple) and len(model) == 2:
                    # Model with scaler
                    model_obj, scaler = model
                    X_scaled = scaler.transform(X)
                    y_pred = model_obj.predict(X_scaled)
                else:
                    logger.warning(f"Unknown model type for {model_name}")
                    continue
                
                predictions_df[f'pred_{model_name}'] = y_pred
                logger.info(f"✅ {model_name} predictions generated")
                
            except Exception as e:
                logger.error(f"❌ {model_name} prediction failed: {e}")
                continue
        
        return predictions_df
    
    def calculate_value_at_risk(self, returns: np.ndarray, confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        logger.info(f"Calculating VaR at {(1-confidence_level)*100}% confidence level...")
        
        # Remove extreme outliers
        returns_clean = returns[~np.isnan(returns)]
        returns_clean = returns_clean[np.abs(returns_clean) < np.percentile(np.abs(returns_clean), 99)]
        
        if len(returns_clean) == 0:
            logger.warning("No valid returns for VaR calculation")
            return {'var': 0, 'cvar': 0, 'expected_shortfall': 0}
        
        # Historical VaR
        var_historical = np.percentile(returns_clean, confidence_level * 100)
        
        # Conditional VaR (Expected Shortfall)
        cvar_returns = returns_clean[returns_clean <= var_historical]
        cvar_historical = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_historical
        
        # Parametric VaR (assuming normal distribution)
        mu = np.mean(returns_clean)
        sigma = np.std(returns_clean)
        var_parametric = mu + sigma * stats.norm.ppf(confidence_level)
        
        var_metrics = {
            'var_historical': var_historical,
            'var_parametric': var_parametric,
            'cvar': cvar_historical,
            'expected_shortfall': cvar_historical,
            'confidence_level': confidence_level,
            'sample_size': len(returns_clean)
        }
        
        logger.info(f"VaR calculated: Historical={var_historical:.4f}, Parametric={var_parametric:.4f}")
        return var_metrics
    
    def calculate_maximum_drawdown(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        logger.info("Calculating maximum drawdown...")
        
        returns_clean = np.asarray(returns, dtype=float)
        returns_clean = returns_clean[np.isfinite(returns_clean)]

        if len(returns_clean) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'avg_drawdown': 0.0,
                'recovery_time': 0,
                'drawdown_periods': 0,
                'max_drawdown_index': 0
            }

        # Calculate compounded equity curve from periodic returns
        cumulative_returns = np.cumprod(1 + np.clip(returns_clean, -0.95, None))
        
        # Calculate running maximum
        peak = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns / peak) - 1
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_drawdown_idx = np.argmin(drawdowns)
        
        # Find recovery time
        recovery_time = 0
        if max_drawdown_idx < len(drawdowns) - 1:
            post_drawdown = drawdowns[max_drawdown_idx:]
            recovery_indices = np.where(post_drawdown >= 0)[0]
            if len(recovery_indices) > 0:
                recovery_time = recovery_indices[0]
        
        # Calculate average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        drawdown_metrics = {
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'avg_drawdown': avg_drawdown,
            'recovery_time': recovery_time,
            'drawdown_periods': len(negative_drawdowns),
            'max_drawdown_index': max_drawdown_idx
        }
        
        logger.info(f"Max Drawdown: {max_drawdown:.4f} ({drawdown_metrics['max_drawdown_percent']:.2f}%)")
        return drawdown_metrics
    
    def calculate_sharpe_sortino_ratios(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate horizon-aware Sharpe, Sortino, and benchmark-relative metrics."""
        logger.info("Calculating Sharpe and Sortino ratios...")

        periods_per_year = self.periods_per_year()
        annual_risk_free_rate = (
            self.config.DEFAULT_RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
        )
        period_rf = annual_risk_free_rate / periods_per_year

        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'period_rf_rate': period_rf,
                'benchmark_annual_return': 0.0,
                'benchmark_sharpe_ratio': 0.0,
                'excess_annual_return': 0.0,
                'information_ratio': 0.0,
                'forecast_horizon_days': self.config.FORECAST_HORIZON_DAYS,
                'periods_per_year': periods_per_year,
            }

        # Calculate excess returns
        excess_returns = returns - period_rf
        
        # Sharpe ratio
        return_std = np.std(returns)
        sharpe_ratio = (
            np.mean(excess_returns) / return_std * np.sqrt(periods_per_year)
            if return_std != 0 else 0
        )
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else return_std
        sortino_ratio = (
            np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
            if downside_deviation != 0 else 0
        )
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = np.mean(returns) * periods_per_year
        max_dd = self.calculate_maximum_drawdown(returns)['max_drawdown']
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0

        benchmark_annual_return = 0.0
        benchmark_sharpe_ratio = 0.0
        excess_annual_return = annual_return
        information_ratio = 0.0

        if benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns, dtype=float)
            benchmark_returns = benchmark_returns[np.isfinite(benchmark_returns)]
            min_length = min(len(returns), len(benchmark_returns))

            if min_length > 1:
                benchmark_returns = benchmark_returns[:min_length]
                truncated_returns = returns[:min_length]
                benchmark_excess = benchmark_returns - period_rf
                benchmark_std = np.std(benchmark_returns)
                benchmark_annual_return = np.mean(benchmark_returns) * periods_per_year
                benchmark_sharpe_ratio = (
                    np.mean(benchmark_excess) / benchmark_std * np.sqrt(periods_per_year)
                    if benchmark_std != 0 else 0
                )
                excess_annual_return = annual_return - benchmark_annual_return

                active_returns = truncated_returns - benchmark_returns
                tracking_error = np.std(active_returns)
                information_ratio = (
                    np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)
                    if tracking_error != 0 else 0
                )
        
        ratios = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annual_return': annual_return,
            'annual_volatility': return_std * np.sqrt(periods_per_year),
            'period_rf_rate': period_rf,
            'benchmark_annual_return': benchmark_annual_return,
            'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
            'excess_annual_return': excess_annual_return,
            'information_ratio': information_ratio,
            'forecast_horizon_days': self.config.FORECAST_HORIZON_DAYS,
            'periods_per_year': periods_per_year,
        }
        
        logger.info(f"Sharpe: {sharpe_ratio:.4f}, Sortino: {sortino_ratio:.4f}, Calmar: {calmar_ratio:.4f}")
        return ratios
    
    def position_sizing_kelly_criterion(self, predictions: np.ndarray, actuals: np.ndarray, 
                                      max_position: float = 0.1) -> Dict[str, float]:
        """Calculate optimal position sizes using Kelly Criterion"""
        logger.info("Calculating position sizes using Kelly Criterion...")
        
        # Create binary win/loss based on predictions
        predicted_direction = self.prediction_positions(predictions)
        actual_direction = np.sign(actuals)
        
        # Calculate win rate and average win/loss
        correct_predictions = (predicted_direction == actual_direction)
        win_rate = np.mean(correct_predictions)
        
        # Calculate average returns for wins and losses
        wins = actuals[correct_predictions & (actuals > 0)]
        losses = actuals[~correct_predictions & (actuals < 0)]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
        
        # Kelly fraction: f = (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = 1-p
        if avg_loss > 0:
            b = avg_win / avg_loss  # odds
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_fraction = 0
        
        # Cap position size for risk management
        optimal_position = min(max(kelly_fraction, 0), max_position)
        
        position_metrics = {
            'kelly_fraction': kelly_fraction,
            'optimal_position': optimal_position,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
            'max_position_cap': max_position
        }
        
        logger.info(f"Kelly Criterion: {kelly_fraction:.4f}, Optimal Position: {optimal_position:.4f}")
        return position_metrics
    
    def portfolio_optimization_markowitz(self, predictions_df: pd.DataFrame, 
                                       target_return: Optional[float] = None) -> Dict[str, Any]:
        """Markowitz mean-variance portfolio optimization"""
        logger.info("Performing Markowitz portfolio optimization...")
        
        # Create returns matrix for each stock
        stocks = predictions_df['Ticker'].unique()
        
        if len(stocks) < 2:
            logger.warning("Need at least 2 stocks for portfolio optimization")
            return {}
        
        # Get prediction columns
        pred_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
        if not pred_cols:
            logger.error("No prediction columns found")
            return {}
        
        # Use the first (best) prediction model
        best_pred_col = pred_cols[0]
        logger.info(f"Using {best_pred_col} for portfolio optimization")
        
        # Create stock returns and predictions matrix
        stock_data = {}
        for stock in stocks:
            stock_df = predictions_df[predictions_df['Ticker'] == stock].copy()
            if len(stock_df) >= 100:  # Increased minimum data requirement
                stock_data[stock] = {
                    'returns': self.normalize_return_array(stock_df['return_5d'].values),
                    'predictions': self.normalize_return_array(stock_df[best_pred_col].values),
                }
        
        if len(stock_data) < 2:
            logger.warning("Insufficient stocks with adequate data for portfolio optimization")
            return {'success': False, 'message': 'Need at least 2 stocks with sufficient data'}
        
        stocks = list(stock_data.keys())
        n_assets = len(stocks)
        
        # Calculate expected returns (using predictions)
        expected_returns = np.array([np.mean(stock_data[stock]['predictions']) for stock in stocks])
        
        # Calculate covariance matrix (using actual returns)
        # Ensure all stocks have the same length by finding minimum length
        min_length = min(len(stock_data[stock]['returns']) for stock in stocks)
        returns_matrix = np.array([stock_data[stock]['returns'][:min_length] for stock in stocks]).T
        
        if returns_matrix.shape[0] < 10:  # Need minimum data for covariance
            logger.warning("Insufficient data for covariance matrix calculation")
            return {'success': False, 'message': 'Insufficient data for portfolio optimization'}
        
        cov_matrix = np.cov(returns_matrix.T)
        
        # Set target return if not provided and keep it in the feasible range
        feasible_min = float(np.min(expected_returns))
        feasible_max = float(np.max(expected_returns))

        if target_return is None:
            target_return = float(np.median(expected_returns))
        else:
            target_return = float(np.clip(target_return, feasible_min, feasible_max))
        
        # Portfolio optimization objective function (minimize variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Portfolio return constraint
        def portfolio_return_constraint(weights):
            return np.dot(weights, expected_returns) - target_return
        
        # Sum of weights = 1 constraint
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': weight_sum_constraint},
            {'type': 'ineq', 'fun': portfolio_return_constraint}
        ]
        
        # Bounds (0 <= weight <= 1 for long-only portfolio)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        # Add light diagonal regularization to keep the covariance matrix numerically stable
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6

        # Optimize
        try:
            result = minimize(portfolio_variance, initial_guess, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_volatility = np.sqrt(portfolio_variance(optimal_weights))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
                
                optimization_results = {
                    'stocks': stocks,
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'target_return': target_return,
                    'feasible_return_range': [feasible_min, feasible_max],
                    'success': True,
                    'optimization_method': 'markowitz',
                    'optimization_message': result.message
                }
                
                logger.info(f"Portfolio optimization successful:")
                logger.info(f"  Expected Return: {portfolio_return:.4f}")
                logger.info(f"  Volatility: {portfolio_volatility:.4f}")
                logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
                
                return optimization_results
                
            else:
                logger.warning(f"Primary Markowitz optimization failed: {result.message}")

                def negative_sharpe(weights):
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_volatility = np.sqrt(max(portfolio_variance(weights), 1e-12))
                    return -(portfolio_return / portfolio_volatility)

                fallback_result = minimize(
                    negative_sharpe,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'eq', 'fun': weight_sum_constraint}]
                )

                if fallback_result.success:
                    optimal_weights = fallback_result.x
                    portfolio_return = np.dot(optimal_weights, expected_returns)
                    portfolio_volatility = np.sqrt(portfolio_variance(optimal_weights))
                    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0

                    logger.info("Markowitz fallback to max-Sharpe optimization succeeded")
                    return {
                        'stocks': stocks,
                        'weights': optimal_weights,
                        'expected_return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'target_return': target_return,
                        'feasible_return_range': [feasible_min, feasible_max],
                        'success': True,
                        'optimization_method': 'markowitz_max_sharpe_fallback',
                        'optimization_message': fallback_result.message,
                        'fallback_reason': result.message
                    }

                logger.error(f"Portfolio optimization failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {'success': False, 'message': str(e)}
    
    def risk_parity_portfolio(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Create risk parity portfolio (equal risk contribution)"""
        logger.info("Creating risk parity portfolio...")
        
        # Get stocks and prediction data
        stocks = predictions_df['Ticker'].unique()
        pred_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
        
        if len(stocks) < 2 or not pred_cols:
            logger.warning("Insufficient data for risk parity portfolio")
            return {}
        
        best_pred_col = pred_cols[0]
        
        # Calculate individual stock volatilities
        stock_volatilities = {}
        stock_returns_data = {}
        for stock in stocks:
            stock_df = predictions_df[predictions_df['Ticker'] == stock]
            if len(stock_df) >= 100:  # Increased minimum requirement
                returns = self.normalize_return_array(stock_df['return_5d'].values)
                volatility = np.std(returns)
                if volatility > 0:  # Only include stocks with non-zero volatility
                    stock_volatilities[stock] = volatility
                    stock_returns_data[stock] = returns
        
        if len(stock_volatilities) < 2:
            logger.warning("Insufficient data for risk parity")
            return {}
        
        # Risk parity weights (inverse volatility)
        inv_volatilities = {stock: 1/vol if vol > 0 else 0 for stock, vol in stock_volatilities.items()}
        total_inv_vol = sum(inv_volatilities.values())
        
        risk_parity_weights = {stock: inv_vol/total_inv_vol for stock, inv_vol in inv_volatilities.items()}
        
        # Calculate portfolio metrics
        stocks_list = list(risk_parity_weights.keys())
        weights_array = np.array([risk_parity_weights[stock] for stock in stocks_list])
        
        # Expected returns using predictions
        expected_returns = []
        for stock in stocks_list:
            stock_df = predictions_df[predictions_df['Ticker'] == stock]
            expected_returns.append(np.mean(self.normalize_return_array(stock_df[best_pred_col].values)))
        
        portfolio_return = np.dot(weights_array, expected_returns)
        min_length = min(len(stock_returns_data[stock]) for stock in stocks_list)
        returns_matrix = np.array([stock_returns_data[stock][:min_length] for stock in stocks_list]).T
        cov_matrix = np.cov(returns_matrix.T) + np.eye(len(stocks_list)) * 1e-6
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        risk_parity_results = {
            'stocks': stocks_list,
            'weights': weights_array,
            'weight_dict': risk_parity_weights,
            'individual_volatilities': stock_volatilities,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0,
            'success': True,
            'optimization_method': 'risk_parity'
        }
        
        logger.info(f"Risk parity portfolio created with {len(stocks_list)} stocks")
        logger.info(f"Portfolio return: {portfolio_return:.4f}, volatility: {portfolio_volatility:.4f}")
        
        return risk_parity_results
    
    def transaction_cost_modeling(self, weights_old: np.ndarray, weights_new: np.ndarray,
                                portfolio_value: float = 100000, transaction_cost: float = 0.001) -> Dict[str, float]:
        """Model transaction costs for portfolio rebalancing"""
        logger.info("Calculating transaction costs...")
        
        # Calculate position changes
        weight_changes = np.abs(weights_new - weights_old)
        
        # Calculate dollar amounts traded
        trades = weight_changes * portfolio_value
        
        # Calculate transaction costs
        total_costs = np.sum(trades) * transaction_cost
        cost_percentage = total_costs / portfolio_value * 100
        
        # Calculate turnover
        turnover = np.sum(weight_changes) / 2  # Half the sum of absolute changes
        
        transaction_metrics = {
            'total_transaction_cost': total_costs,
            'cost_percentage': cost_percentage,
            'portfolio_turnover': turnover,
            'total_trades': np.sum(trades),
            'transaction_cost_rate': transaction_cost,
            'portfolio_value': portfolio_value
        }
        
        logger.info(f"Transaction costs: ${total_costs:.2f} ({cost_percentage:.4f}% of portfolio)")
        return transaction_metrics

    def analysis_model_names(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Return only model result keys from the analysis payload."""
        excluded = {'portfolios', 'best_strategy', 'benchmarks'}
        return [name for name in analysis_results.keys() if name not in excluded]

    def _benchmark_period_returns(
        self,
        predictions_df: pd.DataFrame,
        ranking_col: Optional[str] = None,
        *,
        top_quantile: float = 0.25,
        long_short: bool = False,
        reverse: bool = False,
        transaction_cost_bps: float = 0.0,
    ) -> pd.DataFrame:
        """Build per-date benchmark returns from cross-sectional rules."""
        benchmark_df = predictions_df[['Date', 'Ticker', 'return_5d']].copy()
        benchmark_df['ActualReturn'] = self.normalize_return_array(benchmark_df['return_5d'].values)

        if ranking_col is not None:
            benchmark_df[ranking_col] = predictions_df[ranking_col].fillna(0).values

        period_rows: List[Dict[str, Any]] = []
        cost_rate = transaction_cost_bps / 10000.0

        for date, group in benchmark_df.groupby('Date'):
            if group.empty:
                continue

            universe_return = float(group['ActualReturn'].mean())
            universe_size = len(group)
            bucket_size = max(1, int(np.ceil(universe_size * top_quantile)))

            if ranking_col is None:
                strategy_return = universe_return
                signal_coverage = 100.0
            else:
                ranked = group.sort_values(ranking_col, ascending=reverse)
                long_slice = ranked.head(bucket_size)
                long_return = float(long_slice['ActualReturn'].mean())

                if long_short and universe_size > 1:
                    short_ranked = group.sort_values(ranking_col, ascending=not reverse)
                    short_slice = short_ranked.head(bucket_size)
                    short_return = float(short_slice['ActualReturn'].mean())
                    strategy_return = long_return - short_return
                    signal_coverage = min(100.0, 200.0 * bucket_size / universe_size)
                else:
                    strategy_return = long_return
                    signal_coverage = 100.0 * bucket_size / universe_size

            period_rows.append({
                'Date': pd.to_datetime(date),
                'StrategyReturn': strategy_return,
                'BenchmarkReturn': universe_return,
                'NetStrategyReturn': strategy_return - cost_rate,
                'SignalCoverage': signal_coverage,
                'TransactionCostBps': transaction_cost_bps,
            })

        if not period_rows:
            return pd.DataFrame()

        return pd.DataFrame(period_rows).sort_values('Date').reset_index(drop=True)

    def _evaluate_benchmark(
        self,
        benchmark_name: str,
        description: str,
        period_returns: pd.DataFrame,
        *,
        strategy_type: str,
        signal_source: str,
    ) -> Dict[str, Any]:
        """Convert benchmark return series into the common evaluation schema."""
        net_returns = period_returns['NetStrategyReturn'].values
        benchmark_returns = period_returns['BenchmarkReturn'].values

        return {
            'name': benchmark_name,
            'description': description,
            'strategy_type': strategy_type,
            'signal_source': signal_source,
            'performance_ratios': self.calculate_sharpe_sortino_ratios(
                net_returns,
                benchmark_returns=benchmark_returns,
            ),
            'drawdown_metrics': self.calculate_maximum_drawdown(net_returns),
            'win_rate': float(np.mean(net_returns > 0) * 100),
            'signal_coverage': float(period_returns['SignalCoverage'].mean()),
            'transaction_cost_bps': float(period_returns['TransactionCostBps'].iloc[0]),
            'observations': int(len(period_returns)),
            'total_return': float(np.sum(net_returns)),
        }

    def build_benchmark_suite(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Build defensible cross-sectional benchmarks on the same dates and horizon."""
        logger.info("Building benchmark suite...")
        benchmark_results: Dict[str, Any] = {}

        equal_weight_periods = self._benchmark_period_returns(predictions_df)
        if not equal_weight_periods.empty:
            benchmark_results['EqualWeightLongOnly'] = self._evaluate_benchmark(
                'EqualWeightLongOnly',
                'Passive equal-weight long-only universe benchmark',
                equal_weight_periods,
                strategy_type='long_only',
                signal_source='equal_weight',
            )

        benchmark_specs = [
            (
                'TopQuartileMomentum5D',
                'Long-only top quartile by 5-day momentum',
                'momentum_5d',
                False,
                False,
                self.config.DEFAULT_TRANSACTION_COST_BPS,
            ),
            (
                'TopQuartileMomentum20D',
                'Long-only top quartile by 20-day momentum',
                'momentum_20d',
                False,
                False,
                self.config.DEFAULT_TRANSACTION_COST_BPS,
            ),
            (
                'CrossSectionMomentum5D',
                'Long top quartile and short bottom quartile by 5-day momentum',
                'momentum_5d',
                True,
                False,
                self.config.DEFAULT_TRANSACTION_COST_BPS * 2,
            ),
            (
                'MeanReversion1D',
                'Long worst 1-day losers and short best 1-day winners',
                'daily_return',
                True,
                True,
                self.config.DEFAULT_TRANSACTION_COST_BPS * 2,
            ),
        ]

        for benchmark_name, description, column, long_short, reverse, cost_bps in benchmark_specs:
            if column not in predictions_df.columns:
                continue

            period_returns = self._benchmark_period_returns(
                predictions_df,
                ranking_col=column,
                long_short=long_short,
                reverse=reverse,
                transaction_cost_bps=cost_bps,
            )
            if period_returns.empty:
                continue

            benchmark_results[benchmark_name] = self._evaluate_benchmark(
                benchmark_name,
                description,
                period_returns,
                strategy_type='long_short' if long_short else 'long_only',
                signal_source=column,
            )

        return benchmark_results
    
    def comprehensive_risk_analysis(self, predictions_df: pd.DataFrame, models: Dict) -> Dict[str, Any]:
        """Run comprehensive risk analysis for all models and strategies"""
        logger.info("Running comprehensive risk analysis...")
        
        analysis_results = {}
        
        # 1. Individual Model Risk Analysis
        logger.info("1. Analyzing individual model risks...")
        pred_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
        
        for pred_col in pred_cols:
            model_name = pred_col.replace('pred_', '')
            predictions = self.normalize_return_array(predictions_df[pred_col].values)
            actuals = self.normalize_return_array(predictions_df['return_5d'].values)
            
            # Calculate strategy returns (simple long/short based on predictions)
            positions = self.prediction_positions(predictions)
            strategy_returns = positions * actuals
            period_returns = self.aggregate_period_returns(
                predictions_df['Date'],
                strategy_returns,
                actuals,
            )

            if period_returns.empty:
                logger.warning(f"No valid period returns for {model_name}")
                continue

            gross_period_returns = period_returns['StrategyReturn'].values
            net_period_returns = period_returns['NetStrategyReturn'].values
            benchmark_period_returns = period_returns['BenchmarkReturn'].values
            
            # Risk metrics
            var_metrics = self.calculate_value_at_risk(net_period_returns)
            drawdown_metrics = self.calculate_maximum_drawdown(net_period_returns)
            ratio_metrics = self.calculate_sharpe_sortino_ratios(
                net_period_returns,
                benchmark_returns=benchmark_period_returns,
            )
            gross_ratio_metrics = self.calculate_sharpe_sortino_ratios(
                gross_period_returns,
                benchmark_returns=benchmark_period_returns,
            )
            position_metrics = self.position_sizing_kelly_criterion(predictions, actuals)
            traded_mask = positions != 0
            trade_accuracy = (
                float(np.mean(np.sign(positions[traded_mask]) == np.sign(actuals[traded_mask])) * 100)
                if np.any(traded_mask) else 0.0
            )
            
            analysis_results[model_name] = {
                'var_metrics': var_metrics,
                'drawdown_metrics': drawdown_metrics,
                'performance_ratios': ratio_metrics,
                'gross_performance_ratios': gross_ratio_metrics,
                'position_sizing': position_metrics,
                'total_return': float(np.sum(net_period_returns)),
                'gross_total_return': float(np.sum(gross_period_returns)),
                'win_rate': float(np.mean(net_period_returns > 0) * 100),
                'avg_return': float(np.mean(net_period_returns)),
                'signal_accuracy': float(np.mean(np.sign(positions) == np.sign(actuals)) * 100),
                'trade_accuracy': trade_accuracy,
                'trade_rate': float(np.mean(positions != 0) * 100),
                'transaction_cost_bps': self.config.DEFAULT_TRANSACTION_COST_BPS,
                'forecast_horizon_days': self.config.FORECAST_HORIZON_DAYS,
            }
        
        # 2. Portfolio Optimization
        logger.info("2. Performing portfolio optimization...")
        markowitz_portfolio = self.portfolio_optimization_markowitz(predictions_df)
        risk_parity_portfolio = self.risk_parity_portfolio(predictions_df)
        benchmark_results = self.build_benchmark_suite(predictions_df)

        analysis_results['portfolios'] = {
            'markowitz': markowitz_portfolio,
            'risk_parity': risk_parity_portfolio
        }
        analysis_results['benchmarks'] = benchmark_results
        
        # 3. Best Strategy Selection
        logger.info("3. Identifying best risk-adjusted strategy...")
        best_strategy = None
        best_sharpe = -999
        
        for model_name, metrics in analysis_results.items():
            if model_name not in ['portfolios', 'benchmarks']:
                sharpe = metrics['performance_ratios']['sharpe_ratio']
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_strategy = model_name
        
        analysis_results['best_strategy'] = {
            'name': best_strategy,
            'sharpe_ratio': best_sharpe
        }
        
        logger.info(f"Comprehensive risk analysis completed")
        logger.info(f"Best strategy: {best_strategy} (Sharpe: {best_sharpe:.4f})")
        
        return analysis_results
    
    def create_risk_dashboard(self, analysis_results: Dict) -> go.Figure:
        """Create comprehensive risk management dashboard"""
        logger.info("Creating risk management dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Risk-Adjusted Returns (Sharpe Ratios)', 'Value at Risk (95% Confidence)',
                'Maximum Drawdown Analysis', 'Portfolio Weights (Markowitz)',
                'Win Rates by Strategy', 'Kelly Criterion Position Sizes',
                'Return vs Risk Scatter', 'Cumulative Strategy Performance',
                'Risk Metrics Summary'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Extract model data
        model_names = self.analysis_model_names(analysis_results)
        
        # 1. Sharpe Ratios
        sharpe_ratios = [analysis_results[name]['performance_ratios']['sharpe_ratio'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=sharpe_ratios,
                  name='Sharpe Ratios',
                  marker_color='blue'),
            row=1, col=1
        )
        
        # 2. Value at Risk
        var_values = [analysis_results[name]['var_metrics']['var_historical'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=var_values,
                  name='VaR (95%)',
                  marker_color='red'),
            row=1, col=2
        )
        
        # 3. Maximum Drawdown
        max_drawdowns = [analysis_results[name]['drawdown_metrics']['max_drawdown'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=max_drawdowns,
                  name='Max Drawdown',
                  marker_color='orange'),
            row=1, col=3
        )
        
        # 4. Portfolio Weights (if Markowitz successful)
        if 'portfolios' in analysis_results and 'markowitz' in analysis_results['portfolios']:
            markowitz = analysis_results['portfolios']['markowitz']
            if markowitz.get('success', False):
                stocks = markowitz['stocks']
                weights = markowitz['weights']
                
                fig.add_trace(
                    go.Bar(x=stocks, y=weights,
                          name='Portfolio Weights',
                          marker_color='green'),
                    row=2, col=1
                )
        
        # 5. Win Rates
        win_rates = [analysis_results[name]['win_rate'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=win_rates,
                  name='Win Rate (%)',
                  marker_color='purple'),
            row=2, col=2
        )
        
        # 6. Kelly Criterion Position Sizes
        kelly_positions = [analysis_results[name]['position_sizing']['optimal_position'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=kelly_positions,
                  name='Optimal Position Size',
                  marker_color='brown'),
            row=2, col=3
        )
        
        # 7. Risk vs Return Scatter
        returns = [analysis_results[name]['performance_ratios']['annual_return'] for name in model_names]
        volatilities = [analysis_results[name]['performance_ratios']['annual_volatility'] for name in model_names]
        
        fig.add_trace(
            go.Scatter(x=volatilities, y=returns,
                      mode='markers+text',
                      text=model_names,
                      textposition="top center",
                      name='Risk vs Return',
                      marker=dict(size=10)),
            row=3, col=1
        )
        
        # 8. Total Returns by Model
        if model_names:
            total_returns = [analysis_results[name].get('total_return', 0) for name in model_names]
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=total_returns,
                    name='Total Return',
                    marker_color='teal'
                ),
                row=3, col=2
            )
        
        # 9. Risk Metrics Summary Table
        summary_text = ["RISK METRICS SUMMARY:", ""]
        if model_names:
            best_strategy = analysis_results.get('best_strategy', {})
            summary_text.append(f"Best Strategy: {best_strategy.get('name', 'N/A')}")
            summary_text.append(f"Best Sharpe: {best_strategy.get('sharpe_ratio', 0):.3f}")
            summary_text.append("")

            benchmark_results = analysis_results.get('benchmarks', {})
            if benchmark_results:
                best_benchmark_name, best_benchmark = max(
                    benchmark_results.items(),
                    key=lambda item: item[1]['performance_ratios']['sharpe_ratio']
                )
                summary_text.append(f"Best Benchmark: {best_benchmark_name}")
                summary_text.append(
                    f"  Sharpe: {best_benchmark['performance_ratios']['sharpe_ratio']:.3f}"
                )
                summary_text.append("")
            
            for model in model_names[:5]:  # Top 5 models
                metrics = analysis_results[model]
                summary_text.append(f"{model}:")
                summary_text.append(f"  Sharpe: {metrics['performance_ratios']['sharpe_ratio']:.3f}")
                summary_text.append(f"  VaR: {metrics['var_metrics']['var_historical']:.3f}")
                summary_text.append(f"  Max DD: {metrics['drawdown_metrics']['max_drawdown']:.3f}")
                summary_text.append("")
        
        fig.add_annotation(
            text="<br>".join(summary_text),
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=10, family="monospace"),
            align="left",
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Risk Management & Portfolio Optimization Dashboard - Day 11",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update x-axis labels rotation
        for i in range(1, 4):
            for j in range(1, 4):
                if i < 3:  # Skip last row for scatter plot
                    fig.update_xaxes(tickangle=-45, row=i, col=j)
        
        return fig
    
    def save_risk_analysis_results(self, analysis_results: Dict, models: Dict) -> Dict[str, str]:
        """Save all risk analysis results"""
        logger.info("Saving risk analysis results...")
        
        saved_files = {}
        
        # 1. Save detailed risk analysis
        analysis_path = self.config.PROCESSED_DATA_PATH / "day11_risk_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        saved_files['risk_analysis'] = str(analysis_path)
        
        # 2. Save risk metrics summary
        model_names = self.analysis_model_names(analysis_results)
        
        risk_summary_data = []
        for model_name in model_names:
            metrics = analysis_results[model_name]
            
            summary_row = {
                'Model': model_name,
                'Sharpe_Ratio': metrics['performance_ratios']['sharpe_ratio'],
                'Gross_Sharpe_Ratio': metrics['gross_performance_ratios']['sharpe_ratio'],
                'Sortino_Ratio': metrics['performance_ratios']['sortino_ratio'],
                'Calmar_Ratio': metrics['performance_ratios']['calmar_ratio'],
                'Annual_Return': metrics['performance_ratios']['annual_return'],
                'Gross_Annual_Return': metrics['gross_performance_ratios']['annual_return'],
                'Annual_Volatility': metrics['performance_ratios']['annual_volatility'],
                'Benchmark_Annual_Return': metrics['performance_ratios']['benchmark_annual_return'],
                'Benchmark_Sharpe_Ratio': metrics['performance_ratios']['benchmark_sharpe_ratio'],
                'Excess_Annual_Return': metrics['performance_ratios']['excess_annual_return'],
                'Information_Ratio': metrics['performance_ratios']['information_ratio'],
                'VaR_95': metrics['var_metrics']['var_historical'],
                'CVaR_95': metrics['var_metrics']['cvar'],
                'Max_Drawdown': metrics['drawdown_metrics']['max_drawdown'],
                'Max_Drawdown_Percent': metrics['drawdown_metrics']['max_drawdown_percent'],
                'Win_Rate': metrics['win_rate'],
                'Signal_Accuracy': metrics.get('signal_accuracy', 0),
                'Trade_Accuracy': metrics.get('trade_accuracy', 0),
                'Trade_Rate': metrics.get('trade_rate', 0),
                'Kelly_Position_Size': metrics['position_sizing']['optimal_position'],
                'Total_Return': metrics['total_return'],
                'Gross_Total_Return': metrics.get('gross_total_return', metrics['total_return']),
                'Forecast_Horizon_Days': metrics.get('forecast_horizon_days', self.config.FORECAST_HORIZON_DAYS),
                'Transaction_Cost_Bps': metrics.get('transaction_cost_bps', self.config.DEFAULT_TRANSACTION_COST_BPS),
            }
            risk_summary_data.append(summary_row)
        
        risk_summary_df = pd.DataFrame(risk_summary_data)
        risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
        risk_summary_df.to_csv(risk_summary_path, index=False)
        saved_files['risk_summary'] = str(risk_summary_path)

        benchmark_results = analysis_results.get('benchmarks', {})
        if benchmark_results:
            benchmark_rows = []
            for benchmark_name, metrics in benchmark_results.items():
                benchmark_rows.append({
                    'Benchmark': benchmark_name,
                    'Description': metrics.get('description', ''),
                    'Strategy_Type': metrics.get('strategy_type', ''),
                    'Signal_Source': metrics.get('signal_source', ''),
                    'Sharpe_Ratio': metrics['performance_ratios']['sharpe_ratio'],
                    'Sortino_Ratio': metrics['performance_ratios']['sortino_ratio'],
                    'Calmar_Ratio': metrics['performance_ratios']['calmar_ratio'],
                    'Annual_Return': metrics['performance_ratios']['annual_return'],
                    'Annual_Volatility': metrics['performance_ratios']['annual_volatility'],
                    'Benchmark_Annual_Return': metrics['performance_ratios']['benchmark_annual_return'],
                    'Benchmark_Sharpe_Ratio': metrics['performance_ratios']['benchmark_sharpe_ratio'],
                    'Excess_Annual_Return': metrics['performance_ratios']['excess_annual_return'],
                    'Information_Ratio': metrics['performance_ratios']['information_ratio'],
                    'Max_Drawdown': metrics['drawdown_metrics']['max_drawdown'],
                    'Max_Drawdown_Percent': metrics['drawdown_metrics']['max_drawdown_percent'],
                    'Win_Rate': metrics['win_rate'],
                    'Signal_Coverage': metrics.get('signal_coverage', 0),
                    'Observations': metrics.get('observations', 0),
                    'Total_Return': metrics.get('total_return', 0),
                    'Forecast_Horizon_Days': metrics['performance_ratios']['forecast_horizon_days'],
                    'Transaction_Cost_Bps': metrics.get('transaction_cost_bps', 0),
                })

            benchmark_summary_df = pd.DataFrame(benchmark_rows).sort_values('Sharpe_Ratio', ascending=False)
            benchmark_summary_path = self.config.PROCESSED_DATA_PATH / "day11_benchmark_summary.csv"
            benchmark_summary_df.to_csv(benchmark_summary_path, index=False)
            saved_files['benchmark_summary'] = str(benchmark_summary_path)
        
        # 3. Save portfolio optimization results
        if 'portfolios' in analysis_results:
            portfolio_results = analysis_results['portfolios']
            
            # Markowitz portfolio
            if 'markowitz' in portfolio_results and portfolio_results['markowitz'].get('success', False):
                markowitz_data = portfolio_results['markowitz']
                markowitz_df = pd.DataFrame({
                    'Stock': markowitz_data['stocks'],
                    'Weight': markowitz_data['weights']
                })
                markowitz_path = self.config.PROCESSED_DATA_PATH / "day11_markowitz_portfolio.csv"
                markowitz_df.to_csv(markowitz_path, index=False)
                saved_files['markowitz_portfolio'] = str(markowitz_path)
            
            # Risk parity portfolio
            if 'risk_parity' in portfolio_results and portfolio_results['risk_parity']:
                rp_data = portfolio_results['risk_parity']
                if 'stocks' in rp_data and 'weights' in rp_data:
                    rp_df = pd.DataFrame({
                        'Stock': rp_data['stocks'],
                        'Weight': rp_data['weights'],
                        'Individual_Volatility': [rp_data['individual_volatilities'][stock] for stock in rp_data['stocks']]
                    })
                    rp_path = self.config.PROCESSED_DATA_PATH / "day11_risk_parity_portfolio.csv"
                    rp_df.to_csv(rp_path, index=False)
                    saved_files['risk_parity_portfolio'] = str(rp_path)
        
        # 4. Save position sizing recommendations
        position_data = []
        for model_name in model_names:
            pos_metrics = analysis_results[model_name]['position_sizing']
            position_data.append({
                'Model': model_name,
                'Kelly_Fraction': pos_metrics['kelly_fraction'],
                'Optimal_Position': pos_metrics['optimal_position'],
                'Win_Rate': pos_metrics['win_rate'],
                'Win_Loss_Ratio': pos_metrics['win_loss_ratio'],
                'Average_Win': pos_metrics['avg_win'],
                'Average_Loss': pos_metrics['avg_loss']
            })
        
        position_df = pd.DataFrame(position_data)
        position_path = self.config.PROCESSED_DATA_PATH / "day11_position_sizing.csv"
        position_df.to_csv(position_path, index=False)
        saved_files['position_sizing'] = str(position_path)
        
        # 5. Create comprehensive report
        report = {
            'analysis_date': datetime.now().isoformat(),
            'models_analyzed': len(model_names),
            'best_strategy': analysis_results.get('best_strategy', {}),
            'risk_management_summary': {
                'highest_sharpe': max([analysis_results[name]['performance_ratios']['sharpe_ratio'] for name in model_names]),
                'highest_gross_sharpe': max([analysis_results[name]['gross_performance_ratios']['sharpe_ratio'] for name in model_names]),
                'lowest_var': min([analysis_results[name]['var_metrics']['var_historical'] for name in model_names]),
                'lowest_drawdown': max([analysis_results[name]['drawdown_metrics']['max_drawdown'] for name in model_names]),  # max because drawdowns are negative
                'highest_win_rate': max([analysis_results[name]['win_rate'] for name in model_names]),
                'forecast_horizon_days': self.config.FORECAST_HORIZON_DAYS,
                'transaction_cost_bps': self.config.DEFAULT_TRANSACTION_COST_BPS,
            },
            'benchmark_summary': {
                'benchmarks_analyzed': len(benchmark_results),
                'best_benchmark': (
                    max(
                        benchmark_results.items(),
                        key=lambda item: item[1]['performance_ratios']['sharpe_ratio']
                    )[0]
                    if benchmark_results else None
                ),
                'best_benchmark_sharpe': (
                    max(
                        metrics['performance_ratios']['sharpe_ratio']
                        for metrics in benchmark_results.values()
                    )
                    if benchmark_results else None
                ),
            },
            'portfolio_optimization': {
                'markowitz_successful': 'markowitz' in analysis_results.get('portfolios', {}) and analysis_results['portfolios']['markowitz'].get('success', False),
                'risk_parity_successful': 'risk_parity' in analysis_results.get('portfolios', {}) and bool(analysis_results['portfolios']['risk_parity'])
            },
            'files_generated': list(saved_files.keys()),
            'risk_management_recommendations': self._generate_risk_recommendations(analysis_results)
        }
        
        report_path = self.config.PROCESSED_DATA_PATH / "day11_risk_management_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        saved_files['comprehensive_report'] = str(report_path)
        
        logger.info(f"Risk analysis results saved: {len(saved_files)} files")
        return saved_files
    
    def _generate_risk_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate practical risk management recommendations"""
        recommendations = []
        
        model_names = self.analysis_model_names(analysis_results)
        
        if not model_names:
            return ["Insufficient data for recommendations"]
        
        # Best strategy recommendation
        best_strategy = analysis_results.get('best_strategy', {})
        if best_strategy:
            recommendations.append(f"Primary Strategy: Use {best_strategy.get('name', 'N/A')} model (Sharpe: {best_strategy.get('sharpe_ratio', 0):.3f})")

        benchmark_results = analysis_results.get('benchmarks', {})
        if benchmark_results:
            best_benchmark_name, best_benchmark = max(
                benchmark_results.items(),
                key=lambda item: item[1]['performance_ratios']['sharpe_ratio']
            )
            benchmark_sharpe = best_benchmark['performance_ratios']['sharpe_ratio']
            edge = best_strategy.get('sharpe_ratio', 0) - benchmark_sharpe
            recommendations.append(
                f"Benchmark hurdle: best naive baseline is {best_benchmark_name} (Sharpe: {benchmark_sharpe:.3f}); current model edge is {edge:.3f}"
            )
        
        # Risk level assessment
        avg_sharpe = np.mean([analysis_results[name]['performance_ratios']['sharpe_ratio'] for name in model_names])
        avg_var = np.mean([analysis_results[name]['var_metrics']['var_historical'] for name in model_names])
        
        if avg_sharpe > 1.0:
            recommendations.append("Strong risk-adjusted returns detected - suitable for moderate to aggressive allocation")
        elif avg_sharpe > 0.5:
            recommendations.append("Moderate risk-adjusted returns - consider conservative position sizing")
        else:
            recommendations.append("Low risk-adjusted returns - focus on risk management over return maximization")
        
        # VaR-based recommendations
        if avg_var < -0.05:  # VaR more than 5%
            recommendations.append("High daily VaR detected - implement strict stop-loss orders at 3-5% levels")
        else:
            recommendations.append("Moderate daily VaR - standard 2-3% stop-loss levels appropriate")
        
        # Position sizing recommendations
        avg_kelly = np.mean([analysis_results[name]['position_sizing']['optimal_position'] for name in model_names])
        if avg_kelly > 0.15:
            recommendations.append("Kelly Criterion suggests large positions - cap at 10% per trade for safety")
        elif avg_kelly > 0.05:
            recommendations.append(f"Optimal position size: {avg_kelly*100:.1f}% of portfolio per trade")
        else:
            recommendations.append("Small position sizes recommended - focus on diversification")
        
        # Portfolio recommendations
        if 'portfolios' in analysis_results:
            portfolios = analysis_results['portfolios']
            if portfolios.get('markowitz', {}).get('success', False):
                recommendations.append("Markowitz optimization successful - consider mean-variance portfolio allocation")
            if portfolios.get('risk_parity', {}):
                recommendations.append("Risk parity portfolio available - suitable for risk-conscious investors")
        
        # Diversification recommendations
        if len(model_names) > 1:
            sharpe_std = np.std([analysis_results[name]['performance_ratios']['sharpe_ratio'] for name in model_names])
            if sharpe_std > 0.3:
                recommendations.append("High performance variation between models - consider ensemble approach")
            else:
                recommendations.append("Consistent model performance - single best model approach acceptable")
        
        return recommendations

    def run_comprehensive_risk_management(self) -> Dict[str, Any]:
        """Run the complete risk management analysis"""
        logger.info("Starting comprehensive risk management analysis...")
        
        # 1. Load all necessary data
        logger.info("1. Loading validation results and models...")
        validation_results = self.load_validation_results()
        models = self.load_best_models()
        df = self.load_feature_data()
        
        if df.empty:
            logger.error("Failed to load feature data")
            return {}
        
        if not models:
            logger.error("No models loaded for analysis")
            return {}
        
        # 2. Prepare data and generate predictions
        logger.info("2. Preparing data and generating predictions...")
        X, y, feature_cols = self.prepare_portfolio_data(df)
        predictions_df = self.generate_predictions(models, X, df)
        
        if predictions_df.empty or not any(col.startswith('pred_') for col in predictions_df.columns):
            logger.error("Failed to generate predictions")
            return {}
        
        # 3. Run comprehensive risk analysis
        logger.info("3. Running comprehensive risk analysis...")
        analysis_results = self.comprehensive_risk_analysis(predictions_df, models)
        
        # 4. Create visualizations
        logger.info("4. Creating risk management dashboard...")
        try:
            dashboard_fig = self.create_risk_dashboard(analysis_results)
            
            # Save dashboard
            plots_dir = self.config.PROJECT_ROOT / "plots"
            plots_dir.mkdir(exist_ok=True)
            dashboard_path = plots_dir / "day11_risk_dashboard.html"
            dashboard_fig.write_html(str(dashboard_path))
            logger.info(f"Risk dashboard saved: {dashboard_path}")
        except Exception as e:
            logger.warning(f"Dashboard creation failed: {e}")
        
        # 5. Save all results
        logger.info("5. Saving risk analysis results...")
        saved_files = self.save_risk_analysis_results(analysis_results, models)
        
        # 6. Generate final summary
        final_results = {
            'analysis_results': analysis_results,
            'saved_files': saved_files,
            'summary': {
                'models_analyzed': len(self.analysis_model_names(analysis_results)),
                'benchmarks_analyzed': len(analysis_results.get('benchmarks', {})),
                'best_strategy': analysis_results.get('best_strategy', {}),
                'portfolios_created': len(analysis_results.get('portfolios', {})),
                'risk_recommendations': self._generate_risk_recommendations(analysis_results)
            }
        }
        
        logger.info("Comprehensive risk management analysis completed!")
        return final_results
