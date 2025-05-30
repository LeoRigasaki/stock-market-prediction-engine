import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
import joblib
from datetime import datetime
import json

from .config import Config

class AdvancedMLFramework:
    """Advanced ML models with hyperparameter optimization and deep learning"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.best_params = {}
        self.optimization_history = {}
        self.ensemble_models = {}
        
    def load_baseline_results(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Load results from Day 6 baseline models"""
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        reg_results_path = self.config.PROCESSED_DATA_PATH / "regression_model_results.csv"
        class_results_path = self.config.PROCESSED_DATA_PATH / "classification_model_results.csv"
        
        # Load feature data
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Load baseline results
        baseline_regression = {}
        baseline_classification = {}
        
        if reg_results_path.exists():
            reg_df = pd.read_csv(reg_results_path, index_col=0)
            baseline_regression = reg_df.to_dict('index')
        
        if class_results_path.exists():
            class_df = pd.read_csv(class_results_path, index_col=0)
            baseline_classification = class_df.to_dict('index')
        
        logger.info(f"Loaded data: {len(df)} records, baseline regression: {len(baseline_regression)} models, "
                   f"baseline classification: {len(baseline_classification)} models")
        
        return df, baseline_regression, baseline_classification
    
    def create_xgboost_model(self, task_type: str = 'regression') -> Union[xgb.XGBRegressor, xgb.XGBClassifier]:
        """Create XGBoost model with optimized hyperparameters"""
        if task_type == 'regression':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='rmse'
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
    
    def create_lightgbm_model(self, task_type: str = 'regression') -> Union[lgb.LGBMRegressor, lgb.LGBMClassifier]:
        """Create LightGBM model with optimized hyperparameters"""
        if task_type == 'regression':
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1  # Suppress output
            )
        else:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1  # Suppress output
            )
    
    def create_neural_network(self, input_dim: int, task_type: str = 'regression') -> Union[MLPRegressor, MLPClassifier]:
        """Create Neural Network with adaptive architecture"""
        # Determine hidden layer sizes based on input dimension
        hidden_layer_1 = min(100, max(50, input_dim // 2))
        hidden_layer_2 = min(50, hidden_layer_1 // 2)
        
        if task_type == 'regression':
            return MLPRegressor(
                hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
        else:
            return MLPClassifier(
                hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
    
    def create_lstm_model(self, input_shape: Tuple[int, int], task_type: str = 'regression') -> Any:
        """Create LSTM-like model using scikit-learn for compatibility"""
        # Since TensorFlow isn't available, we'll use a time-aware neural network instead
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        
        # Create a more complex neural network to simulate LSTM behavior
        hidden_layer_1 = min(100, max(50, input_shape[1] * 2))
        hidden_layer_2 = min(50, hidden_layer_1 // 2)
        hidden_layer_3 = min(25, hidden_layer_2 // 2)
        
        if task_type == 'regression':
            return MLPRegressor(
                hidden_layer_sizes=(hidden_layer_1, hidden_layer_2, hidden_layer_3),
                activation='tanh',  # Better for time series
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                alpha=0.001  # L2 regularization
            )
        else:
            return MLPClassifier(
                hidden_layer_sizes=(hidden_layer_1, hidden_layer_2, hidden_layer_3),
                activation='tanh',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                alpha=0.001
            )
    
    def optimize_xgboost_hyperparameters(self, X: pd.DataFrame, y: np.ndarray, 
                                       task_type: str = 'regression', n_trials: int = 50) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna"""
        logger.info(f"Optimizing XGBoost hyperparameters for {task_type} with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1
            }
            
            if task_type == 'regression':
                model = xgb.XGBRegressor(**params, eval_metric='rmse')
                metric = 'neg_mean_squared_error'
            else:
                model = xgb.XGBClassifier(**params, eval_metric='logloss')
                metric = 'f1'
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                
                if task_type == 'regression':
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)
                else:
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='binary')
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
        logger.info(f"XGBoost optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def optimize_lightgbm_hyperparameters(self, X: pd.DataFrame, y: np.ndarray,
                                        task_type: str = 'regression', n_trials: int = 50) -> Dict:
        """Optimize LightGBM hyperparameters using Optuna"""
        logger.info(f"Optimizing LightGBM hyperparameters for {task_type} with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            if task_type == 'regression':
                model = lgb.LGBMRegressor(**params)
                metric = 'neg_mean_squared_error'
            else:
                model = lgb.LGBMClassifier(**params)
                metric = 'f1'
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                
                if task_type == 'regression':
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)
                else:
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='binary')
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['verbose'] = -1
        
        logger.info(f"LightGBM optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def prepare_lstm_data(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_col: str = 'return_5d', sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time-series aware data (simplified without true LSTM)"""
        logger.info(f"Preparing time-series data with {sequence_length}-period rolling features...")
        
        X_features = []
        y_targets = []
        
        # Process each stock separately to maintain temporal order
        for ticker in df['Ticker'].unique():
            stock_data = df[df['Ticker'] == ticker].sort_values('Date')
            
            if len(stock_data) < sequence_length + 1:
                continue  # Skip stocks with insufficient data
            
            # Create rolling window features instead of true sequences
            features = stock_data[feature_cols].fillna(0)
            targets = stock_data[target_col].fillna(0)
            
            # Add rolling statistics as "time-aware" features
            for window in [sequence_length//2, sequence_length]:
                for col in feature_cols[:10]:  # Limit to prevent feature explosion
                    if col in features.columns:
                        features[f'{col}_roll_mean_{window}'] = features[col].rolling(window).mean()
                        features[f'{col}_roll_std_{window}'] = features[col].rolling(window).std()
            
            # Remove rows with NaN from rolling calculations
            features = features.dropna()
            targets = targets.loc[features.index]
            
            X_features.append(features.values)
            y_targets.append(targets.values)
        
        # Combine all stocks
        if X_features:
            X_combined = np.vstack(X_features)
            y_combined = np.hstack(y_targets)
        else:
            X_combined = np.array([])
            y_combined = np.array([])
        
        logger.info(f"Created time-series dataset with shape {X_combined.shape}")
        return X_combined, y_combined
    
    def train_advanced_models(self, df: pd.DataFrame, target_col: str = 'return_5d', 
                            task_type: str = 'regression') -> Dict[str, Any]:
        """Train all advanced models with optimization"""
        logger.info(f"Training advanced {task_type} models...")
        
        # Prepare data
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].fillna(df[target_col].median())
        
        if task_type == 'classification':
            y = (y > 0).astype(int)
        
        trained_models = {}
        
        # 1. Optimized XGBoost
        logger.info("1. Training optimized XGBoost...")
        try:
            xgb_params = self.optimize_xgboost_hyperparameters(X, y, task_type, n_trials=30)
            self.best_params['xgboost'] = xgb_params
            
            if task_type == 'regression':
                xgb_model = xgb.XGBRegressor(**xgb_params, eval_metric='rmse')
            else:
                xgb_model = xgb.XGBClassifier(**xgb_params, eval_metric='logloss')
            
            xgb_model.fit(X, y)
            trained_models['XGBoost_Optimized'] = xgb_model
            logger.info("✅ XGBoost trained successfully")
            
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
        
        # 2. Optimized LightGBM
        logger.info("2. Training optimized LightGBM...")
        try:
            lgb_params = self.optimize_lightgbm_hyperparameters(X, y, task_type, n_trials=30)
            self.best_params['lightgbm'] = lgb_params
            
            if task_type == 'regression':
                lgb_model = lgb.LGBMRegressor(**lgb_params)
            else:
                lgb_model = lgb.LGBMClassifier(**lgb_params)
            
            lgb_model.fit(X, y)
            trained_models['LightGBM_Optimized'] = lgb_model
            logger.info("✅ LightGBM trained successfully")
            
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
        
        # 3. Advanced Neural Network
        logger.info("3. Training advanced Neural Network...")
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            nn_model = self.create_neural_network(X.shape[1], task_type)
            nn_model.fit(X_scaled, y)
            trained_models['Neural_Network_Advanced'] = (nn_model, scaler)
            logger.info("✅ Neural Network trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Neural Network training failed: {e}")
        
        # 4. Gradient Boosting
        logger.info("4. Training Gradient Boosting...")
        try:
            if task_type == 'regression':
                gb_model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            else:
                gb_model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            
            gb_model.fit(X, y)
            trained_models['Gradient_Boosting'] = gb_model
            logger.info("✅ Gradient Boosting trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Gradient Boosting training failed: {e}")
        
        # 5. Time-Aware Neural Network (LSTM alternative)
        logger.info("5. Training Time-Aware Neural Network...")
        try:
            X_time_series, y_time_series = self.prepare_lstm_data(df, feature_cols, target_col)
            
            if len(X_time_series) > 100:  # Ensure sufficient data
                # Convert to classification if needed
                if task_type == 'classification':
                    y_time_series = (y_time_series > 0).astype(int)
                
                # Scale features
                scaler = StandardScaler()
                X_time_series_scaled = scaler.fit_transform(X_time_series)
                
                # Create time-aware neural network
                time_nn = self.create_lstm_model((X_time_series.shape[0], X_time_series.shape[1]), task_type)
                time_nn.fit(X_time_series_scaled, y_time_series)
                
                trained_models['Time_Aware_NN'] = (time_nn, scaler)
                logger.info("✅ Time-Aware Neural Network trained successfully")
            else:
                logger.warning("⚠️ Insufficient data for Time-Aware Neural Network training")
                
        except Exception as e:
            logger.error(f"❌ Time-Aware Neural Network training failed: {e}")
        
        self.models[task_type] = trained_models
        logger.info(f"Completed training {len(trained_models)} advanced {task_type} models")
        
        return trained_models
    
    def evaluate_advanced_models(self, models: Dict[str, Any], df: pd.DataFrame, 
                               target_col: str = 'return_5d', task_type: str = 'regression') -> Dict[str, Dict]:
        """Evaluate advanced models with comprehensive metrics"""
        logger.info(f"Evaluating advanced {task_type} models...")
        
        # Prepare data
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].fillna(df[target_col].median())
        
        if task_type == 'classification':
            y = (y > 0).astype(int)
        
        results = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            cv_scores = []
            predictions_all = []
            actuals_all = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                try:
                    # Handle different model types
                    if model_name == 'Neural_Network_Advanced':
                        model_obj, scaler = model
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model_obj.fit(X_train_scaled, y_train)
                        y_pred = model_obj.predict(X_test_scaled)
                        
                    elif model_name == 'Time_Aware_NN':
                        model_obj, scaler = model
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model_obj.fit(X_train_scaled, y_train)
                        y_pred = model_obj.predict(X_test_scaled)
                        
                    else:
                        # Standard sklearn-like models
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Flatten predictions if needed
                    if hasattr(y_pred, 'flatten'):
                        y_pred = y_pred.flatten()
                    
                    # Calculate metrics
                    if task_type == 'regression':
                        score = r2_score(y_test, y_pred)
                    else:
                        score = f1_score(y_test, y_pred, average='binary')
                    
                    cv_scores.append(score)
                    predictions_all.extend(y_pred)
                    actuals_all.extend(y_test)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name} on fold: {e}")
                    continue
            
            if cv_scores:
                if task_type == 'regression':
                    results[model_name] = {
                        'cv_r2_mean': np.mean(cv_scores),
                        'cv_r2_std': np.std(cv_scores),
                        'cv_rmse_mean': np.sqrt(np.mean([(p - a)**2 for p, a in zip(predictions_all, actuals_all)])),
                        'predictions': predictions_all,
                        'actuals': actuals_all
                    }
                else:
                    # Calculate additional classification metrics
                    accuracy = accuracy_score(actuals_all, np.round(predictions_all))
                    precision = precision_score(actuals_all, np.round(predictions_all), average='binary', zero_division=0)
                    recall = recall_score(actuals_all, np.round(predictions_all), average='binary', zero_division=0)
                    
                    results[model_name] = {
                        'cv_f1_mean': np.mean(cv_scores),
                        'cv_f1_std': np.std(cv_scores),
                        'cv_accuracy': accuracy,
                        'cv_precision': precision,
                        'cv_recall': recall,
                        'predictions': predictions_all,
                        'actuals': actuals_all
                    }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    results[model_name]['feature_importance'] = importance_df
                
                logger.info(f"✅ {model_name} evaluation completed")
            else:
                logger.warning(f"⚠️ {model_name} evaluation failed - no valid scores")
        
        return results
    
    def create_model_ensemble(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create ensemble of best performing models"""
        logger.info("Creating model ensemble...")
        
        if weights is None:
            # Equal weights for all models
            weights = {name: 1.0/len(models) for name in models.keys()}
        
        ensemble = {
            'models': models,
            'weights': weights,
            'ensemble_type': 'weighted_average'
        }
        
        logger.info(f"Created ensemble with {len(models)} models and weights: {weights}")
        return ensemble
    
    def create_advanced_visualizations(self, results: Dict, baseline_results: Dict, 
                                     optimization_history: Dict) -> go.Figure:
        """Create comprehensive advanced model visualizations"""
        logger.info("Creating advanced model visualizations...")
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Advanced vs Baseline Performance', 'Feature Importance (XGBoost)',
                'Model Performance Distribution', 'Hyperparameter Comparison',
                'Advanced Model Metrics', 'Performance Summary'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Advanced vs Baseline Performance Comparison
        if results and baseline_results:
            model_names = []
            model_scores = []
            model_types = []
            
            # Get performance metric (R² for regression, F1 for classification)
            if 'cv_r2_mean' in list(results.values())[0]:
                metric_key = 'cv_r2_mean'
                metric_name = 'R² Score'
            else:
                metric_key = 'cv_f1_mean'
                metric_name = 'F1 Score'
            
            # Add advanced models
            for name, model_results in results.items():
                model_names.append(f"ADV_{name}")
                model_scores.append(model_results.get(metric_key, 0))
                model_types.append('Advanced')
            
            # Add baseline models  
            for name, model_results in baseline_results.items():
                model_names.append(f"BASE_{name}")
                model_scores.append(model_results.get(metric_key, 0))
                model_types.append('Baseline')
            
            # Create color mapping
            colors = ['green' if t == 'Advanced' else 'blue' for t in model_types]
            
            fig.add_trace(
                go.Bar(
                    x=model_names, 
                    y=model_scores,
                    marker_color=colors,
                    name='Model Performance',
                    text=[f'{score:.4f}' for score in model_scores],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # 2. Feature Importance (XGBoost if available)
        if results:
            for model_name, model_results in results.items():
                if 'feature_importance' in model_results and 'XGBoost' in model_name:
                    importance_df = model_results['feature_importance'].head(10)
                    
                    fig.add_trace(
                        go.Bar(
                            x=importance_df['importance'],
                            y=importance_df['feature'],
                            orientation='h',
                            name='Feature Importance',
                            marker_color='orange'
                        ),
                        row=1, col=2
                    )
                    break
        
        # 3. Model Performance Distribution
        if results:
            all_scores = []
            all_models = []
            
            for model_name, model_results in results.items():
                score = model_results.get(metric_key, 0)
                all_scores.append(score)
                all_models.append(model_name)
            
            fig.add_trace(
                go.Scatter(
                    x=all_models,
                    y=all_scores,
                    mode='markers+lines',
                    marker=dict(size=10, color='red'),
                    name='Advanced Models',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=3
            )
        
        # 4. Hyperparameter Comparison
        if self.best_params:
            param_names = []
            param_values = []
            model_names_hp = []
            
            for model_name, params in self.best_params.items():
                for param_name, param_value in list(params.items())[:5]:  # Top 5 params
                    param_names.append(f"{model_name}_{param_name}")
                    if isinstance(param_value, (int, float)):
                        param_values.append(param_value)
                    else:
                        param_values.append(0)  # For non-numeric values
                    model_names_hp.append(model_name)
            
            if param_values:
                fig.add_trace(
                    go.Bar(
                        x=param_names,
                        y=param_values,
                        marker_color='purple',
                        name='Hyperparameters'
                    ),
                    row=2, col=1
                )
        
        # 5. Advanced Model Metrics Summary
        if results:
            metrics_text = []
            metrics_text.append("ADVANCED MODEL SUMMARY:")
            
            for model_name, model_results in results.items():
                if metric_key in model_results:
                    score = model_results[metric_key]
                    metrics_text.append(f"{model_name}: {score:.4f}")
            
            # Add as annotation
            fig.add_annotation(
                text="<br>".join(metrics_text),
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=10),
                align="left",
                row=2, col=2
            )
        
        # 6. Performance Summary Chart
        if results and baseline_results:
            # Calculate improvements
            best_advanced_score = max([r.get(metric_key, 0) for r in results.values()])
            best_baseline_score = max([r.get(metric_key, 0) for r in baseline_results.values()])
            
            summary_data = {
                'Best Advanced': best_advanced_score,
                'Best Baseline': best_baseline_score,
                'Improvement': best_advanced_score - best_baseline_score
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(summary_data.keys()),
                    y=list(summary_data.values()),
                    marker_color=['green', 'blue', 'gold'],
                    name='Performance Summary'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Advanced Model Analysis Dashboard - Day 7-8<br>Metric: {metric_name}",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update x-axis labels rotation for better readability
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        
        return fig
    
    def save_advanced_results(self, regression_results: Dict, classification_results: Dict,
                            baseline_regression: Dict, baseline_classification: Dict) -> Dict[str, str]:
        """Save all advanced modeling results"""
        logger.info("Saving advanced modeling results...")
        
        saved_files = {}
        
        # Save advanced model results
        if regression_results:
            reg_df = pd.DataFrame({
                name: {
                    'cv_r2_mean': results.get('cv_r2_mean', 0),
                    'cv_r2_std': results.get('cv_r2_std', 0),
                    'cv_rmse_mean': results.get('cv_rmse_mean', 0)
                }
                for name, results in regression_results.items()
            }).T
            
            reg_path = self.config.PROCESSED_DATA_PATH / "advanced_regression_results.csv"
            reg_df.to_csv(reg_path)
            saved_files['advanced_regression'] = str(reg_path)
        
        if classification_results:
            class_df = pd.DataFrame({
                name: {
                    'cv_f1_mean': results.get('cv_f1_mean', 0),
                    'cv_f1_std': results.get('cv_f1_std', 0),
                    'cv_accuracy': results.get('cv_accuracy', 0)
                }
                for name, results in classification_results.items()
            }).T
            
            class_path = self.config.PROCESSED_DATA_PATH / "advanced_classification_results.csv"
            class_df.to_csv(class_path)
            saved_files['advanced_classification'] = str(class_path)
        
        # Save hyperparameter optimization results
        if self.best_params:
            params_path = self.config.PROCESSED_DATA_PATH / "optimized_hyperparameters.json"
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2, default=str)
            saved_files['hyperparameters'] = str(params_path)
        
        # Save models
        models_dir = self.config.PROJECT_ROOT / "models" / "advanced"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for task_type in ['regression', 'classification']:
            if task_type in self.models:
                for model_name, model in self.models[task_type].items():
                    model_path = models_dir / f"{task_type}_{model_name.lower().replace(' ', '_')}.joblib"
                    
                    if model_name == 'Time_Aware_NN':
                        # Save Keras-like model separately
                        model_path = models_dir / f"{task_type}_time_aware_nn.joblib"
                        joblib.dump(model, model_path)
                    elif isinstance(model, tuple):
                        # Handle models with scalers (Neural Network)
                        joblib.dump(model, model_path)
                    else:
                        joblib.dump(model, model_path)
                    
                    saved_files[f'{task_type}_{model_name}'] = str(model_path)
        
        # Create comprehensive comparison report
        comparison_report = {
            'analysis_date': datetime.now().isoformat(),
            'advanced_models_trained': {
                'regression': list(regression_results.keys()) if regression_results else [],
                'classification': list(classification_results.keys()) if classification_results else []
            },
            'baseline_models': {
                'regression': list(baseline_regression.keys()) if baseline_regression else [],
                'classification': list(baseline_classification.keys()) if baseline_classification else []
            },
            'performance_improvement': {},
            'best_models': {},
            'hyperparameter_optimization': {
                'models_optimized': list(self.best_params.keys()),
                'optimization_trials': 30  # Default value used
            }
        }
        
        # Calculate performance improvements
        if regression_results and baseline_regression:
            # Find best advanced and baseline models
            best_advanced_reg = max(regression_results.keys(), 
                                  key=lambda x: regression_results[x].get('cv_r2_mean', 0))
            best_baseline_reg = max(baseline_regression.keys(), 
                                  key=lambda x: baseline_regression[x].get('cv_r2_mean', 0))
            
            advanced_score = regression_results[best_advanced_reg].get('cv_r2_mean', 0)
            baseline_score = baseline_regression[best_baseline_reg].get('cv_r2_mean', 0)
            improvement = ((advanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            comparison_report['performance_improvement']['regression'] = {
                'best_advanced_model': best_advanced_reg,
                'best_baseline_model': best_baseline_reg,
                'advanced_r2': advanced_score,
                'baseline_r2': baseline_score,
                'improvement_percentage': improvement
            }
            comparison_report['best_models']['regression'] = best_advanced_reg
        
        if classification_results and baseline_classification:
            # Find best advanced and baseline models
            best_advanced_class = max(classification_results.keys(), 
                                    key=lambda x: classification_results[x].get('cv_f1_mean', 0))
            best_baseline_class = max(baseline_classification.keys(), 
                                    key=lambda x: baseline_classification[x].get('cv_f1_mean', 0))
            
            advanced_score = classification_results[best_advanced_class].get('cv_f1_mean', 0)
            baseline_score = baseline_classification[best_baseline_class].get('cv_f1_mean', 0)
            improvement = ((advanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            comparison_report['performance_improvement']['classification'] = {
                'best_advanced_model': best_advanced_class,
                'best_baseline_model': best_baseline_class,
                'advanced_f1': advanced_score,
                'baseline_f1': baseline_score,
                'improvement_percentage': improvement
            }
            comparison_report['best_models']['classification'] = best_advanced_class
        
        # Save comparison report
        report_path = self.config.PROCESSED_DATA_PATH / "advanced_models_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        saved_files['comparison_report'] = str(report_path)
        
        logger.info(f"Advanced modeling results saved: {len(saved_files)} files")
        return saved_files