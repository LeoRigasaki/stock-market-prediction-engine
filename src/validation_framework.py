import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import jarque_bera, kstest
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
import json
from datetime import datetime, timedelta
from pathlib import Path
import itertools

from .config import Config

class ValidationFramework:
    """Comprehensive model validation and backtesting framework"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.ensemble_models = {}
        self.validation_results = {}
        self.backtest_results = {}
        self.risk_metrics = {}
        
    def load_trained_models(self) -> Dict[str, Any]:
        """Load all trained models from previous days"""
        logger.info("Loading trained models for validation...")
        
        models = {}
        
        # Load individual models
        models_dir = self.config.PROJECT_ROOT / "models"
        advanced_dir = models_dir / "advanced"
        ensemble_dir = models_dir / "ensemble"
        
        # Individual model files (from Days 6, 7-8)
        individual_files = {
            'XGBoost': advanced_dir / "regression_xgboost_optimized.joblib",
            'LightGBM': advanced_dir / "regression_lightgbm_optimized.joblib",
            'RandomForest': models_dir / "regression_random_forest.joblib",
            'GradientBoosting': advanced_dir / "regression_gradient_boosting.joblib",
            'NeuralNetwork': advanced_dir / "regression_neural_network_advanced.joblib",
            'LinearRegression': models_dir / "regression_linear_regression.joblib",
            'Ridge': models_dir / "regression_ridge_regression.joblib",
            'Lasso': models_dir / "regression_lasso_regression.joblib",
            'SVM': models_dir / "regression_svm.joblib"
        }
        
        # Ensemble model files (from Day 9)
        ensemble_files = {
            'VotingRegressor': ensemble_dir / "voting_regressor_ensemble.joblib",
            'StackedEnsemble': ensemble_dir / "stacked_ensemble_ensemble.joblib",
            'SimpleAverage': ensemble_dir / "simple_average_ensemble.joblib"
        }
        
        # Load individual models
        loaded_individual = 0
        for name, path in individual_files.items():
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                    logger.info(f"Loaded individual model: {name}")
                    loaded_individual += 1
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        # Load ensemble models (these are the key ones from Day 9!)
        loaded_ensemble = 0
        for name, path in ensemble_files.items():
            if path.exists():
                try:
                    models[f"Ensemble_{name}"] = joblib.load(path)
                    logger.info(f"Loaded ensemble model: {name}")
                    loaded_ensemble += 1
                except Exception as e:
                    logger.warning(f"Failed to load ensemble {name}: {e}")
        
        logger.info(f"Loaded {loaded_individual} individual models and {loaded_ensemble} ensemble models")
        logger.info(f"Total models for validation: {len(models)}")
        
        return models
    
    def load_feature_data(self) -> pd.DataFrame:
        """Load feature data for validation"""
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"Loaded feature data: {len(df)} records, {df.shape[1]} features")
        return df
    
    def prepare_validation_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Prepare data for validation"""
        # Prepare features and target
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['return_5d'].fillna(df['return_5d'].median())
        
        return X, y, feature_cols
    
    def walk_forward_validation(self, df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, 
                              model: Any, model_name: str, window_size: int = 252, 
                              step_size: int = 21) -> Dict[str, Any]:
        """Implement walk-forward validation for time series"""
        logger.info(f"Performing walk-forward validation for {model_name}")
        logger.info(f"Window size: {window_size} days, Step size: {step_size} days")
        
        # Sort by date to ensure proper temporal order
        df_sorted = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index] if isinstance(y, pd.Series) else y[df_sorted.index]
        
        predictions = []
        actuals = []
        dates = []
        fold_results = []
        
        start_idx = window_size
        max_idx = len(df_sorted) - step_size
        
        fold_count = 0
        while start_idx < max_idx:
            # Define training and testing windows
            train_end = start_idx
            train_start = max(0, train_end - window_size)
            test_start = train_end
            test_end = min(len(df_sorted), test_start + step_size)
            
            # Extract training and testing data
            X_train = X_sorted.iloc[train_start:train_end]
            y_train = y_sorted.iloc[train_start:train_end] if isinstance(y_sorted, pd.Series) else y_sorted[train_start:train_end]
            X_test = X_sorted.iloc[test_start:test_end]
            y_test = y_sorted.iloc[test_start:test_end] if isinstance(y_sorted, pd.Series) else y_sorted[test_start:test_end]
            
            # Get corresponding dates
            test_dates = df_sorted.iloc[test_start:test_end]['Date'].tolist()
            
            try:
                # Handle different model types
                if hasattr(model, 'predict'):
                    # Standard sklearn-like model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif isinstance(model, tuple) and len(model) == 2:
                    # Model with scaler (Neural Network)
                    model_obj, scaler = model
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model_obj.fit(X_train_scaled, y_train)
                    y_pred = model_obj.predict(X_test_scaled)
                else:
                    logger.warning(f"Unknown model type for {model_name}")
                    start_idx += step_size
                    continue
                
                # Calculate fold metrics
                fold_mse = mean_squared_error(y_test, y_pred)
                fold_mae = mean_absolute_error(y_test, y_pred)
                fold_r2 = r2_score(y_test, y_pred)
                
                fold_results.append({
                    'fold': fold_count,
                    'train_start': df_sorted.iloc[train_start]['Date'],
                    'train_end': df_sorted.iloc[train_end-1]['Date'],
                    'test_start': df_sorted.iloc[test_start]['Date'],
                    'test_end': df_sorted.iloc[test_end-1]['Date'],
                    'mse': fold_mse,
                    'mae': fold_mae,
                    'r2': fold_r2,
                    'rmse': np.sqrt(fold_mse),
                    'samples': len(y_test)
                })
                
                # Store predictions
                predictions.extend(y_pred)
                actuals.extend(y_test)
                dates.extend(test_dates)
                
                fold_count += 1
                
                if fold_count % 10 == 0:
                    logger.info(f"Completed {fold_count} folds for {model_name}")
                
            except Exception as e:
                logger.warning(f"Fold {fold_count} failed for {model_name}: {e}")
            
            start_idx += step_size
        
        # Calculate overall metrics
        overall_results = {
            'model_name': model_name,
            'total_folds': fold_count,
            'total_predictions': len(predictions),
            'overall_mse': mean_squared_error(actuals, predictions),
            'overall_mae': mean_absolute_error(actuals, predictions),
            'overall_r2': r2_score(actuals, predictions),
            'overall_rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'fold_results': fold_results,
            'mean_fold_r2': np.mean([f['r2'] for f in fold_results]),
            'std_fold_r2': np.std([f['r2'] for f in fold_results]),
            'mean_fold_rmse': np.mean([f['rmse'] for f in fold_results]),
            'stability_score': 1 - (np.std([f['r2'] for f in fold_results]) / np.mean([f['r2'] for f in fold_results])) if np.mean([f['r2'] for f in fold_results]) != 0 else 0
        }
        
        logger.info(f"Walk-forward validation completed for {model_name}")
        logger.info(f"Overall R²: {overall_results['overall_r2']:.4f}, "
                   f"Stability: {overall_results['stability_score']:.4f}")
        
        return overall_results
    
    def out_of_sample_testing(self, df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                            model: Any, model_name: str, test_ratio: float = 0.2) -> Dict[str, Any]:
        """Perform out-of-sample testing with latest data"""
        logger.info(f"Performing out-of-sample testing for {model_name}")
        
        # Sort by date and use latest data for out-of-sample testing
        df_sorted = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        
        # Split data
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx] if isinstance(y, pd.Series) else y[:split_idx]
        y_test = y.iloc[split_idx:] if isinstance(y, pd.Series) else y[split_idx:]
        
        # Get date ranges
        train_dates = df_sorted.iloc[:split_idx]['Date']
        test_dates = df_sorted.iloc[split_idx:]['Date']
        
        logger.info(f"Training period: {train_dates.min().date()} to {train_dates.max().date()}")
        logger.info(f"Testing period: {test_dates.min().date()} to {test_dates.max().date()}")
        
        try:
            # Train and predict
            if hasattr(model, 'predict'):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif isinstance(model, tuple) and len(model) == 2:
                model_obj, scaler = model
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model_obj.fit(X_train_scaled, y_train)
                y_pred = model_obj.predict(X_test_scaled)
            else:
                raise ValueError(f"Unknown model type for {model_name}")
            
            # Calculate metrics
            oos_results = {
                'model_name': model_name,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_start': train_dates.min(),
                'train_end': train_dates.max(),
                'test_start': test_dates.min(),
                'test_end': test_dates.max(),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'predictions': y_pred.tolist(),
                'actuals': y_test.tolist() if isinstance(y_test, pd.Series) else y_test.tolist(),
                'test_dates': test_dates.tolist()
            }
            
            logger.info(f"Out-of-sample R² for {model_name}: {oos_results['r2']:.4f}")
            return oos_results
            
        except Exception as e:
            logger.error(f"Out-of-sample testing failed for {model_name}: {e}")
            return {}
    
    def statistical_significance_testing(self, results1: Dict, results2: Dict, 
                                       test_name1: str, test_name2: str) -> Dict[str, Any]:
        """Test statistical significance between two model results"""
        logger.info(f"Testing statistical significance between {test_name1} and {test_name2}")
        
        # Get prediction errors
        errors1 = np.array(results1['predictions']) - np.array(results1['actuals'])
        errors2 = np.array(results2['predictions']) - np.array(results2['actuals'])
        
        # Ensure same length
        min_len = min(len(errors1), len(errors2))
        errors1 = errors1[:min_len]
        errors2 = errors2[:min_len]
        
        # Diebold-Mariano test (simplified)
        diff_errors = errors1**2 - errors2**2
        
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(errors1**2, errors2**2)
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(errors1**2, errors2**2, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(errors1) - 1) * np.var(errors1, ddof=1) + 
                             (len(errors2) - 1) * np.var(errors2, ddof=1)) / 
                            (len(errors1) + len(errors2) - 2))
        cohens_d = (np.mean(errors1) - np.mean(errors2)) / pooled_std if pooled_std != 0 else 0
        
        significance_results = {
            'comparison': f"{test_name1} vs {test_name2}",
            'samples_compared': min_len,
            'mean_error_1': np.mean(np.abs(errors1)),
            'mean_error_2': np.mean(np.abs(errors2)),
            'paired_t_test': {
                'statistic': t_stat,
                'p_value': t_pvalue,
                'significant': t_pvalue < 0.05
            },
            'wilcoxon_test': {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_pvalue,
                'significant': wilcoxon_pvalue < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'better_model': test_name1 if np.mean(np.abs(errors1)) < np.mean(np.abs(errors2)) else test_name2
        }
        
        logger.info(f"Statistical significance: t-test p={t_pvalue:.4f}, "
                   f"Wilcoxon p={wilcoxon_pvalue:.4f}")
        
        return significance_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def robustness_testing(self, df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                         model: Any, model_name: str) -> Dict[str, Any]:
        """Test model robustness across different market conditions"""
        logger.info(f"Testing robustness for {model_name}")
        
        robustness_results = {}
        
        # 1. Performance across different volatility regimes
        if 'volatility_20d' in df.columns:
            df_with_vol = df.copy()
            df_with_vol['volatility_regime'] = pd.cut(df_with_vol['volatility_20d'], 
                                                     bins=3, labels=['Low', 'Medium', 'High'])
            
            vol_results = {}
            for regime in ['Low', 'Medium', 'High']:
                regime_mask = df_with_vol['volatility_regime'] == regime
                if regime_mask.sum() > 50:  # Minimum samples
                    X_regime = X[regime_mask]
                    y_regime = y[regime_mask] if isinstance(y, pd.Series) else y[regime_mask.values]
                    
                    # Simple train-test split for regime
                    split_idx = int(len(X_regime) * 0.8)
                    X_train_regime = X_regime.iloc[:split_idx]
                    X_test_regime = X_regime.iloc[split_idx:]
                    y_train_regime = y_regime.iloc[:split_idx] if isinstance(y_regime, pd.Series) else y_regime[:split_idx]
                    y_test_regime = y_regime.iloc[split_idx:] if isinstance(y_regime, pd.Series) else y_regime[split_idx:]
                    
                    try:
                        if hasattr(model, 'predict'):
                            model.fit(X_train_regime, y_train_regime)
                            y_pred_regime = model.predict(X_test_regime)
                        elif isinstance(model, tuple):
                            model_obj, scaler = model
                            X_train_scaled = scaler.fit_transform(X_train_regime)
                            X_test_scaled = scaler.transform(X_test_regime)
                            model_obj.fit(X_train_scaled, y_train_regime)
                            y_pred_regime = model_obj.predict(X_test_scaled)
                        else:
                            continue
                        
                        vol_results[regime] = {
                            'r2': r2_score(y_test_regime, y_pred_regime),
                            'rmse': np.sqrt(mean_squared_error(y_test_regime, y_pred_regime)),
                            'samples': len(y_test_regime)
                        }
                    except Exception as e:
                        logger.warning(f"Volatility regime {regime} testing failed: {e}")
            
            robustness_results['volatility_regimes'] = vol_results
        
        # 2. Performance across different return regimes
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        return_regime = pd.cut(y_series, bins=3, labels=['Bear', 'Neutral', 'Bull'])
        
        return_results = {}
        for regime in ['Bear', 'Neutral', 'Bull']:
            regime_mask = return_regime == regime
            if regime_mask.sum() > 50:
                X_regime = X[regime_mask]
                y_regime = y_series[regime_mask]
                
                split_idx = int(len(X_regime) * 0.8)
                X_train_regime = X_regime.iloc[:split_idx]
                X_test_regime = X_regime.iloc[split_idx:]
                y_train_regime = y_regime.iloc[:split_idx]
                y_test_regime = y_regime.iloc[split_idx:]
                
                try:
                    if hasattr(model, 'predict'):
                        model.fit(X_train_regime, y_train_regime)
                        y_pred_regime = model.predict(X_test_regime)
                    elif isinstance(model, tuple):
                        model_obj, scaler = model
                        X_train_scaled = scaler.fit_transform(X_train_regime)
                        X_test_scaled = scaler.transform(X_test_regime)
                        model_obj.fit(X_train_scaled, y_train_regime)
                        y_pred_regime = model_obj.predict(X_test_scaled)
                    else:
                        continue
                    
                    return_results[regime] = {
                        'r2': r2_score(y_test_regime, y_pred_regime),
                        'rmse': np.sqrt(mean_squared_error(y_test_regime, y_pred_regime)),
                        'samples': len(y_test_regime)
                    }
                except Exception as e:
                    logger.warning(f"Return regime {regime} testing failed: {e}")
        
        robustness_results['return_regimes'] = return_results
        
        # 3. Temporal stability (performance over time)
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        n_periods = 4
        period_size = len(df_sorted) // n_periods
        
        temporal_results = {}
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_sorted)
            
            period_mask = df_sorted.index[start_idx:end_idx]
            X_period = X.iloc[period_mask]
            y_period = y.iloc[period_mask] if isinstance(y, pd.Series) else y[period_mask]
            
            if len(X_period) > 50:
                split_idx = int(len(X_period) * 0.8)
                X_train_period = X_period.iloc[:split_idx]
                X_test_period = X_period.iloc[split_idx:]
                y_train_period = y_period.iloc[:split_idx] if isinstance(y_period, pd.Series) else y_period[:split_idx]
                y_test_period = y_period.iloc[split_idx:] if isinstance(y_period, pd.Series) else y_period[split_idx:]
                
                try:
                    if hasattr(model, 'predict'):
                        model.fit(X_train_period, y_train_period)
                        y_pred_period = model.predict(X_test_period)
                    elif isinstance(model, tuple):
                        model_obj, scaler = model
                        X_train_scaled = scaler.fit_transform(X_train_period)
                        X_test_scaled = scaler.transform(X_test_period)
                        model_obj.fit(X_train_scaled, y_train_period)
                        y_pred_period = model_obj.predict(X_test_scaled)
                    else:
                        continue
                    
                    period_start = df_sorted.iloc[start_idx]['Date']
                    period_end = df_sorted.iloc[end_idx-1]['Date']
                    
                    temporal_results[f'Period_{i+1}'] = {
                        'start_date': period_start,
                        'end_date': period_end,
                        'r2': r2_score(y_test_period, y_pred_period),
                        'rmse': np.sqrt(mean_squared_error(y_test_period, y_pred_period)),
                        'samples': len(y_test_period)
                    }
                except Exception as e:
                    logger.warning(f"Temporal period {i+1} testing failed: {e}")
        
        robustness_results['temporal_stability'] = temporal_results
        
        # Calculate overall robustness score
        all_r2_scores = []
        for category in robustness_results.values():
            for result in category.values():
                if 'r2' in result:
                    all_r2_scores.append(result['r2'])
        
        if all_r2_scores:
            robustness_score = 1 - (np.std(all_r2_scores) / np.mean(all_r2_scores)) if np.mean(all_r2_scores) != 0 else 0
            robustness_results['overall_robustness_score'] = max(0, robustness_score)  # Ensure non-negative
        else:
            robustness_results['overall_robustness_score'] = 0
        
        logger.info(f"Robustness testing completed for {model_name}")
        logger.info(f"Overall robustness score: {robustness_results.get('overall_robustness_score', 0):.4f}")
        
        return robustness_results
    
    def calculate_risk_adjusted_returns(self, predictions: List[float], actuals: List[float],
                                      dates: List[datetime], model_name: str) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        logger.info(f"Calculating risk-adjusted returns for {model_name}")
        
        # Convert to numpy arrays
        pred_array = np.array(predictions)
        actual_array = np.array(actuals)
        
        # Calculate prediction-based strategy returns
        # Simple strategy: long when prediction > 0, short when prediction < 0
        strategy_returns = np.where(pred_array > 0, actual_array, -actual_array)
        
        # Remove extreme outliers
        strategy_returns = np.clip(strategy_returns, 
                                 np.percentile(strategy_returns, 1), 
                                 np.percentile(strategy_returns, 99))
        
        # Risk-free rate (assumed 2% annually, converted to daily)
        risk_free_rate = 0.02 / 252
        
        # Performance metrics
        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        volatility = np.std(strategy_returns)
        
        # Sharpe ratio
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < risk_free_rate]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calmar ratio
        annualized_return = mean_return * 252
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (vs buy-and-hold)
        benchmark_returns = actual_array  # Buy and hold actual returns
        active_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(active_returns)
        information_ratio = np.mean(active_returns) / tracking_error if tracking_error != 0 else 0
        
        # Value at Risk (VaR) - 5% VaR
        var_95 = np.percentile(strategy_returns, 5)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(strategy_returns[strategy_returns <= var_95]) if np.any(strategy_returns <= var_95) else var_95
        
        # Win rate
        win_rate = np.mean(strategy_returns > 0) * 100
        
        risk_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility * np.sqrt(252),  # Annualized
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
            'sortino_ratio': sortino_ratio * np.sqrt(252),  # Annualized
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio * np.sqrt(252),  # Annualized
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'total_trades': len(strategy_returns),
            'profitable_trades': np.sum(strategy_returns > 0),
            'loss_making_trades': np.sum(strategy_returns < 0)
        }
        
        logger.info(f"Risk-adjusted metrics for {model_name}:")
        logger.info(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Max Drawdown: {risk_metrics['max_drawdown']:.4f}")
        logger.info(f"  Win Rate: {risk_metrics['win_rate']:.1f}%")
        
        return risk_metrics
    
    def performance_attribution_analysis(self, df: pd.DataFrame, predictions: List[float], 
                                       actuals: List[float], dates: List[datetime],
                                       model_name: str) -> Dict[str, Any]:
        """Analyze performance attribution across different factors"""
        logger.info(f"Performing attribution analysis for {model_name}")
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'Date': dates,
            'Prediction': predictions,
            'Actual': actuals,
            'Error': np.array(predictions) - np.array(actuals),
            'AbsError': np.abs(np.array(predictions) - np.array(actuals))
        })
        
        # Merge with original data to get additional features
        df_merged = pd.merge(analysis_df, df[['Date', 'Ticker']], on='Date', how='left')
        
        attribution_results = {}
        
        # 1. Performance by stock
        if 'Ticker' in df_merged.columns:
            stock_performance = df_merged.groupby('Ticker').agg({
                'Error': ['mean', 'std', 'count'],
                'AbsError': 'mean'
            }).round(4)
            
            stock_performance.columns = ['Mean_Error', 'Std_Error', 'Count', 'MAE']
            stock_performance = stock_performance.sort_values('MAE')
            
            attribution_results['by_stock'] = stock_performance.to_dict('index')
        
        # 2. Performance by time period
        analysis_df['Year'] = pd.to_datetime(analysis_df['Date']).dt.year
        analysis_df['Month'] = pd.to_datetime(analysis_df['Date']).dt.month
        analysis_df['Quarter'] = pd.to_datetime(analysis_df['Date']).dt.quarter
        
        # Yearly performance
        yearly_performance = analysis_df.groupby('Year').agg({
            'Error': ['mean', 'std', 'count'],
            'AbsError': 'mean'
        }).round(4)
        yearly_performance.columns = ['Mean_Error', 'Std_Error', 'Count', 'MAE']
        attribution_results['by_year'] = yearly_performance.to_dict('index')
        
        # Monthly performance
        monthly_performance = analysis_df.groupby('Month').agg({
            'Error': ['mean', 'std', 'count'],
            'AbsError': 'mean'
        }).round(4)
        monthly_performance.columns = ['Mean_Error', 'Std_Error', 'Count', 'MAE']
        attribution_results['by_month'] = monthly_performance.to_dict('index')
        
        # 3. Performance by prediction confidence
        analysis_df['PredictionAbs'] = np.abs(analysis_df['Prediction'])
        analysis_df['ConfidenceLevel'] = pd.cut(analysis_df['PredictionAbs'], 
                                               bins=3, labels=['Low', 'Medium', 'High'])
        
        confidence_performance = analysis_df.groupby('ConfidenceLevel').agg({
            'Error': ['mean', 'std', 'count'],
            'AbsError': 'mean'
        }).round(4)
        confidence_performance.columns = ['Mean_Error', 'Std_Error', 'Count', 'MAE']
        attribution_results['by_confidence'] = confidence_performance.to_dict('index')
        
        # 4. Error distribution analysis
        attribution_results['error_statistics'] = {
            'mean_error': float(analysis_df['Error'].mean()),
            'std_error': float(analysis_df['Error'].std()),
            'skewness': float(analysis_df['Error'].skew()),
            'kurtosis': float(analysis_df['Error'].kurtosis()),
            'mae': float(analysis_df['AbsError'].mean()),
            'rmse': float(np.sqrt(np.mean(analysis_df['Error']**2))),
            'median_error': float(analysis_df['Error'].median()),
            'q25_error': float(analysis_df['Error'].quantile(0.25)),
            'q75_error': float(analysis_df['Error'].quantile(0.75))
        }
        
        logger.info(f"Attribution analysis completed for {model_name}")
        
        return attribution_results
    
    def model_stability_assessment(self, walk_forward_results: Dict, 
                                 robustness_results: Dict, model_name: str) -> Dict[str, Any]:
        """Assess overall model stability"""
        logger.info(f"Assessing model stability for {model_name}")
        
        stability_metrics = {}
        
        # 1. Temporal stability from walk-forward validation
        if 'fold_results' in walk_forward_results:
            fold_r2_scores = [fold['r2'] for fold in walk_forward_results['fold_results']]
            
            stability_metrics['temporal_stability'] = {
                'mean_r2': np.mean(fold_r2_scores),
                'std_r2': np.std(fold_r2_scores),
                'min_r2': np.min(fold_r2_scores),
                'max_r2': np.max(fold_r2_scores),
                'coefficient_of_variation': np.std(fold_r2_scores) / np.mean(fold_r2_scores) if np.mean(fold_r2_scores) != 0 else 0,
                'stability_score': walk_forward_results.get('stability_score', 0)
            }
        
        # 2. Robustness across market conditions
        if robustness_results:
            robustness_score = robustness_results.get('overall_robustness_score', 0)
            stability_metrics['market_robustness'] = {
                'overall_score': robustness_score,
                'volatility_regimes': robustness_results.get('volatility_regimes', {}),
                'return_regimes': robustness_results.get('return_regimes', {}),
                'temporal_periods': robustness_results.get('temporal_stability', {})
            }
        
        # 3. Overall stability score (combination of temporal and market robustness)
        temporal_score = stability_metrics.get('temporal_stability', {}).get('stability_score', 0)
        robustness_score = stability_metrics.get('market_robustness', {}).get('overall_score', 0)
        
        overall_stability = (temporal_score + robustness_score) / 2
        stability_metrics['overall_stability_score'] = overall_stability
        
        # 4. Stability rating
        if overall_stability >= 0.8:
            stability_rating = "Excellent"
        elif overall_stability >= 0.6:
            stability_rating = "Good"
        elif overall_stability >= 0.4:
            stability_rating = "Fair"
        elif overall_stability >= 0.2:
            stability_rating = "Poor"
        else:
            stability_rating = "Very Poor"
        
        stability_metrics['stability_rating'] = stability_rating
        
        logger.info(f"Model stability for {model_name}: {stability_rating} ({overall_stability:.4f})")
        
        return stability_metrics
    
    def create_validation_visualizations(self, validation_results: Dict) -> go.Figure:
        """Create comprehensive validation visualizations"""
        logger.info("Creating validation visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Walk-Forward Validation Performance', 'Out-of-Sample Performance',
                'Model Stability Scores', 'Risk-Adjusted Returns',
                'Performance Attribution by Time', 'Robustness Across Regimes',
                'Error Distribution Analysis', 'Prediction vs Actual Scatter',
                'Cumulative Performance Over Time'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        model_names = list(validation_results.keys())
        colors = px.colors.qualitative.Set1[:len(model_names)]
        
        # 1. Walk-Forward Validation Performance
        for i, (model_name, results) in enumerate(validation_results.items()):
            if 'walk_forward' in results:
                wf_results = results['walk_forward']
                if 'fold_results' in wf_results:
                    fold_data = wf_results['fold_results']
                    fold_numbers = [f['fold'] for f in fold_data]
                    fold_r2 = [f['r2'] for f in fold_data]
                    
                    fig.add_trace(
                        go.Scatter(x=fold_numbers, y=fold_r2, 
                                  name=f'{model_name} (WF)',
                                  line=dict(color=colors[i % len(colors)])),
                        row=1, col=1
                    )
        
        # 2. Out-of-Sample Performance
        oos_r2_scores = []
        oos_model_names = []
        for model_name, results in validation_results.items():
            if 'out_of_sample' in results and 'r2' in results['out_of_sample']:
                oos_r2_scores.append(results['out_of_sample']['r2'])
                oos_model_names.append(model_name)
        
        if oos_r2_scores:
            fig.add_trace(
                go.Bar(x=oos_model_names, y=oos_r2_scores,
                      name='Out-of-Sample R²',
                      marker_color=colors[:len(oos_r2_scores)]),
                row=1, col=2
            )
        
        # 3. Model Stability Scores
        stability_scores = []
        stability_model_names = []
        for model_name, results in validation_results.items():
            if 'stability' in results and 'overall_stability_score' in results['stability']:
                stability_scores.append(results['stability']['overall_stability_score'])
                stability_model_names.append(model_name)
        
        if stability_scores:
            fig.add_trace(
                go.Bar(x=stability_model_names, y=stability_scores,
                      name='Stability Score',
                      marker_color='lightblue'),
                row=1, col=3
            )
        
        # 4. Risk-Adjusted Returns (Sharpe Ratios)
        sharpe_ratios = []
        sharpe_model_names = []
        for model_name, results in validation_results.items():
            if 'risk_metrics' in results and 'sharpe_ratio' in results['risk_metrics']:
                sharpe_ratios.append(results['risk_metrics']['sharpe_ratio'])
                sharpe_model_names.append(model_name)
        
        if sharpe_ratios:
            fig.add_trace(
                go.Bar(x=sharpe_model_names, y=sharpe_ratios,
                      name='Sharpe Ratio',
                      marker_color='gold'),
                row=2, col=1
            )
        
        # 5. Performance Attribution by Year (first model as example)
        if validation_results:
            first_model = list(validation_results.keys())[0]
            if 'attribution' in validation_results[first_model]:
                attr_data = validation_results[first_model]['attribution']
                if 'by_year' in attr_data:
                    years = list(attr_data['by_year'].keys())
                    mae_by_year = [attr_data['by_year'][year]['MAE'] for year in years]
                    
                    fig.add_trace(
                        go.Scatter(x=years, y=mae_by_year,
                                  mode='lines+markers',
                                  name=f'{first_model} MAE by Year'),
                        row=2, col=2
                    )
        
        # 6. Robustness Across Regimes (volatility regimes for first model)
        if validation_results:
            first_model = list(validation_results.keys())[0]
            if 'robustness' in validation_results[first_model]:
                rob_data = validation_results[first_model]['robustness']
                if 'volatility_regimes' in rob_data:
                    regimes = list(rob_data['volatility_regimes'].keys())
                    regime_r2 = [rob_data['volatility_regimes'][regime]['r2'] for regime in regimes]
                    
                    fig.add_trace(
                        go.Bar(x=regimes, y=regime_r2,
                              name=f'{first_model} by Vol Regime',
                              marker_color='orange'),
                        row=2, col=3
                    )
        
        # 7. Error Distribution (first model)
        if validation_results:
            first_model = list(validation_results.keys())[0]
            if 'walk_forward' in validation_results[first_model]:
                wf_data = validation_results[first_model]['walk_forward']
                if 'predictions' in wf_data and 'actuals' in wf_data:
                    errors = np.array(wf_data['predictions']) - np.array(wf_data['actuals'])
                    
                    fig.add_trace(
                        go.Histogram(x=errors, nbinsx=50,
                                    name=f'{first_model} Errors',
                                    opacity=0.7),
                        row=3, col=1
                    )
        
        # 8. Prediction vs Actual Scatter (first model)
        if validation_results:
            first_model = list(validation_results.keys())[0]
            if 'walk_forward' in validation_results[first_model]:
                wf_data = validation_results[first_model]['walk_forward']
                if 'predictions' in wf_data and 'actuals' in wf_data:
                    predictions = wf_data['predictions'][:1000]  # Limit for performance
                    actuals = wf_data['actuals'][:1000]
                    
                    fig.add_trace(
                        go.Scatter(x=actuals, y=predictions,
                                  mode='markers',
                                  name=f'{first_model} Predictions',
                                  marker=dict(size=3, opacity=0.6)),
                        row=3, col=2
                    )
                    
                    # Perfect prediction line
                    min_val = min(min(actuals), min(predictions))
                    max_val = max(max(actuals), max(predictions))
                    fig.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                  mode='lines',
                                  name='Perfect Prediction',
                                  line=dict(dash='dash', color='red')),
                        row=3, col=2
                    )
        
        # 9. Cumulative Performance Over Time (risk-adjusted returns)
        for i, (model_name, results) in enumerate(validation_results.items()):
            if 'walk_forward' in results:
                wf_data = results['walk_forward']
                if 'predictions' in wf_data and 'actuals' in wf_data:
                    # Simple strategy returns
                    predictions = np.array(wf_data['predictions'])
                    actuals = np.array(wf_data['actuals'])
                    strategy_returns = np.where(predictions > 0, actuals, -actuals)
                    cumulative_returns = np.cumsum(strategy_returns)
                    
                    # Sample every 10th point for performance
                    sample_indices = range(0, len(cumulative_returns), 10)
                    sampled_returns = cumulative_returns[sample_indices]
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(len(sampled_returns))), y=sampled_returns,
                                  name=f'{model_name} Cumulative',
                                  line=dict(color=colors[i % len(colors)])),
                        row=3, col=3
                    )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Model Validation Dashboard - Day 10",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation for all models"""
        logger.info("Starting comprehensive model validation...")
        
        # Load models and data
        models = self.load_trained_models()
        df = self.load_feature_data()
        X, y, feature_cols = self.prepare_validation_data(df)
        
        if not models:
            logger.error("No models loaded for validation")
            return {}
        
        validation_results = {}
        
        # Process each model
        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Validating {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                model_results = {}
                
                # 1. Walk-forward validation
                logger.info(f"1. Walk-forward validation for {model_name}")
                wf_results = self.walk_forward_validation(df, X, y, model, model_name)
                model_results['walk_forward'] = wf_results
                
                # 2. Out-of-sample testing
                logger.info(f"2. Out-of-sample testing for {model_name}")
                oos_results = self.out_of_sample_testing(df, X, y, model, model_name)
                if oos_results:
                    model_results['out_of_sample'] = oos_results
                
                # 3. Robustness testing
                logger.info(f"3. Robustness testing for {model_name}")
                robustness_results = self.robustness_testing(df, X, y, model, model_name)
                model_results['robustness'] = robustness_results
                
                # 4. Risk-adjusted returns
                if 'predictions' in wf_results and 'actuals' in wf_results:
                    logger.info(f"4. Risk-adjusted returns for {model_name}")
                    risk_metrics = self.calculate_risk_adjusted_returns(
                        wf_results['predictions'], wf_results['actuals'], 
                        wf_results['dates'], model_name
                    )
                    model_results['risk_metrics'] = risk_metrics
                
                # 5. Performance attribution
                if 'predictions' in wf_results and 'actuals' in wf_results:
                    logger.info(f"5. Performance attribution for {model_name}")
                    attribution_results = self.performance_attribution_analysis(
                        df, wf_results['predictions'], wf_results['actuals'],
                        wf_results['dates'], model_name
                    )
                    model_results['attribution'] = attribution_results
                
                # 6. Model stability assessment
                logger.info(f"6. Model stability assessment for {model_name}")
                stability_results = self.model_stability_assessment(
                    wf_results, robustness_results, model_name
                )
                model_results['stability'] = stability_results
                
                validation_results[model_name] = model_results
                
                logger.info(f"✅ {model_name} validation completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {model_name} validation failed: {e}")
                continue
        
        # Statistical significance testing between models
        logger.info("\n7. Statistical significance testing...")
        significance_results = {}
        model_pairs = list(itertools.combinations(validation_results.keys(), 2))
        
        for model1, model2 in model_pairs:
            if ('walk_forward' in validation_results[model1] and 
                'walk_forward' in validation_results[model2]):
                try:
                    sig_test = self.statistical_significance_testing(
                        validation_results[model1]['walk_forward'],
                        validation_results[model2]['walk_forward'],
                        model1, model2
                    )
                    significance_results[f"{model1}_vs_{model2}"] = sig_test
                except Exception as e:
                    logger.warning(f"Significance test failed for {model1} vs {model2}: {e}")
        
        # Store results
        self.validation_results = validation_results
        self.significance_results = significance_results
        
        logger.info(f"\n🎯 Comprehensive validation completed for {len(validation_results)} models")
        
        return {
            'validation_results': validation_results,
            'significance_results': significance_results
        }
    
    def save_validation_results(self, results: Dict) -> Dict[str, str]:
        """Save all validation results"""
        logger.info("Saving validation results...")
        
        saved_files = {}
        
        # Save detailed validation results
        validation_path = self.config.PROCESSED_DATA_PATH / "day10_validation_results.json"
        with open(validation_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        saved_files['validation_results'] = str(validation_path)
        
        # Handle both possible data structures
        validation_results = results.get('validation_results', results)
        
        # Save summary metrics
        summary_data = []
        for model_name, model_results in validation_results.items():
            summary_row = {
                'Model': model_name,
                'WalkForward_R2': model_results.get('walk_forward', {}).get('overall_r2', 0),
                'OutOfSample_R2': model_results.get('out_of_sample', {}).get('r2', 0),
                'Stability_Score': model_results.get('stability', {}).get('overall_stability_score', 0),
                'Sharpe_Ratio': model_results.get('risk_metrics', {}).get('sharpe_ratio', 0),
                'Max_Drawdown': model_results.get('risk_metrics', {}).get('max_drawdown', 0),
                'Win_Rate': model_results.get('risk_metrics', {}).get('win_rate', 0),
                'Robustness_Score': model_results.get('robustness', {}).get('overall_robustness_score', 0)
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.config.PROCESSED_DATA_PATH / "day10_validation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        saved_files['validation_summary'] = str(summary_path)
        
        # Save significance test results
        significance_results = results.get('significance_results', {})
        if significance_results:
            sig_data = []
            for comparison, sig_results in significance_results.items():
                sig_row = {
                    'Comparison': comparison,
                    'T_Test_PValue': sig_results.get('paired_t_test', {}).get('p_value', 1),
                    'T_Test_Significant': sig_results.get('paired_t_test', {}).get('significant', False),
                    'Wilcoxon_PValue': sig_results.get('wilcoxon_test', {}).get('p_value', 1),
                    'Wilcoxon_Significant': sig_results.get('wilcoxon_test', {}).get('significant', False),
                    'Effect_Size': sig_results.get('effect_size', {}).get('cohens_d', 0),
                    'Better_Model': sig_results.get('better_model', 'Unknown')
                }
                sig_data.append(sig_row)
            
            sig_df = pd.DataFrame(sig_data)
            sig_path = self.config.PROCESSED_DATA_PATH / "day10_significance_tests.csv"
            sig_df.to_csv(sig_path, index=False)
            saved_files['significance_tests'] = str(sig_path)
        
        logger.info(f"Validation results saved: {len(saved_files)} files")
        return saved_files