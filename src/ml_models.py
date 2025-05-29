import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import joblib
from datetime import datetime

from .config import Config

class MLModelFramework:
    """Comprehensive ML model development and evaluation framework"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_feature_data(self) -> pd.DataFrame:
        """Load the selected features from Day 4"""
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        
        if not features_path.exists():
            logger.error(f"Feature dataset not found at {features_path}")
            logger.error("Please run Day 4 first to generate engineered features")
            return pd.DataFrame()
        
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"Loaded feature dataset: {len(df)} records, {df.shape[1]} features, {df['Ticker'].nunique()} stocks")
        return df
    
    def prepare_modeling_data(self, df: pd.DataFrame, target_col: str = 'return_5d', 
                            prediction_type: str = 'regression') -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Prepare data for machine learning modeling"""
        logger.info(f"Preparing data for {prediction_type} modeling with target: {target_col}")
        
        # Identify feature columns (exclude metadata and target columns)
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Check if target column exists
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            available_targets = [col for col in df.columns if col.startswith(('target_', 'return_'))]
            logger.info(f"Available target columns: {available_targets}")
            if available_targets:
                target_col = available_targets[0]
                logger.info(f"Using {target_col} as target instead")
            else:
                raise ValueError("No suitable target column found")
        
        # Prepare features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # For classification, convert returns to binary labels
        if prediction_type == 'classification':
            # Create binary target: 1 if return > 0, 0 otherwise
            y_binary = (y > 0).astype(int)
            logger.info(f"Classification target distribution: {np.bincount(y_binary)}")
            return X, y_binary, feature_cols
        else:
            logger.info(f"Regression target stats: mean={y.mean():.4f}, std={y.std():.4f}")
            return X, y, feature_cols
    
    def create_time_series_splits(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-series aware train/test splits"""
        logger.info(f"Creating time-series splits with {n_splits} folds")
        
        # Sort by date to ensure temporal order
        df_sorted = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(df_sorted))
        
        logger.info(f"Created {len(splits)} time-series splits")
        for i, (train_idx, test_idx) in enumerate(splits):
            train_start = df_sorted.iloc[train_idx[0]]['Date']
            train_end = df_sorted.iloc[train_idx[-1]]['Date']
            test_start = df_sorted.iloc[test_idx[0]]['Date']
            test_end = df_sorted.iloc[test_idx[-1]]['Date']
            logger.info(f"Split {i+1}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
        
        return splits
    
    def train_baseline_models(self, X: pd.DataFrame, y: np.ndarray, 
                            prediction_type: str = 'regression') -> Dict[str, Any]:
        """Train baseline machine learning models"""
        logger.info(f"Training baseline {prediction_type} models...")
        
        models = {}
        
        if prediction_type == 'regression':
            # Regression models
            models['Linear_Regression'] = LinearRegression()
            models['Ridge_Regression'] = Ridge(alpha=1.0, random_state=42)
            models['Lasso_Regression'] = Lasso(alpha=0.1, random_state=42)
            models['Random_Forest'] = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            models['SVM'] = SVR(kernel='rbf', C=1.0, gamma='scale')
            
        else:  # classification
            models['Random_Forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            models['SVM'] = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        
        # Scale features for models that need it
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                if name in ['SVM', 'Ridge_Regression', 'Lasso_Regression']:
                    # These models benefit from scaling
                    model.fit(X_scaled, y)
                else:
                    # Tree-based models don't need scaling
                    model.fit(X, y)
                
                trained_models[name] = model
                logger.info(f"✅ {name} trained successfully")
                
            except Exception as e:
                logger.error(f"❌ {name} training failed: {e}")
        
        self.models[prediction_type] = trained_models
        logger.info(f"Completed training {len(trained_models)} {prediction_type} models")
        
        return trained_models
    
    def evaluate_regression_models(self, models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                                cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Dict]:
        """Comprehensive evaluation of regression models"""
        logger.info("Evaluating regression models...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Cross-validation metrics
            cv_scores = {
                'mse': [],
                'mae': [],
                'r2': []
            }
            
            predictions_all = []
            actuals_all = []
            
            # Evaluate on each time series split
            for train_idx, test_idx in cv_splits:
                # Handle both DataFrame and numpy array cases
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                else:
                    X_train, X_test = X[train_idx], X[test_idx]
                
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_train, y_test = y[train_idx], y[test_idx]
                
                # Handle scaling for models that need it
                if name in ['SVM', 'Ridge_Regression', 'Lasso_Regression']:
                    fold_scaler = RobustScaler()
                    X_train_scaled = fold_scaler.fit_transform(X_train)
                    X_test_scaled = fold_scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                cv_scores['mse'].append(mse)
                cv_scores['mae'].append(mae)
                cv_scores['r2'].append(r2)
                
                predictions_all.extend(y_pred)
                actuals_all.extend(y_test)
            
            # Calculate average metrics
            results[name] = {
                'cv_mse_mean': np.mean(cv_scores['mse']),
                'cv_mse_std': np.std(cv_scores['mse']),
                'cv_mae_mean': np.mean(cv_scores['mae']),
                'cv_mae_std': np.std(cv_scores['mae']),
                'cv_r2_mean': np.mean(cv_scores['r2']),
                'cv_r2_std': np.std(cv_scores['r2']),
                'rmse_mean': np.sqrt(np.mean(cv_scores['mse'])),
                'predictions': predictions_all,
                'actuals': actuals_all
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                if isinstance(X, pd.DataFrame):
                    feature_cols = X.columns
                else:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
            
            logger.info(f"✅ {name}: R²={results[name]['cv_r2_mean']:.4f}±{results[name]['cv_r2_std']:.4f}, "
                    f"RMSE={results[name]['rmse_mean']:.4f}")
        
        return results

    def evaluate_classification_models(self, models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                                    cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Dict]:
        """Comprehensive evaluation of classification models"""
        logger.info("Evaluating classification models...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            cv_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
            
            predictions_all = []
            actuals_all = []
            probabilities_all = []
            
            # Evaluate on each time series split
            for train_idx, test_idx in cv_splits:
                # Handle both DataFrame and numpy array cases
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                else:
                    X_train, X_test = X[train_idx], X[test_idx]
                
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_train, y_test = y[train_idx], y[test_idx]
                
                # Handle scaling for SVM
                if name == 'SVM':
                    fold_scaler = RobustScaler()
                    X_train_scaled = fold_scaler.fit_transform(X_train)
                    X_test_scaled = fold_scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                
                cv_scores['accuracy'].append(accuracy)
                cv_scores['precision'].append(precision)
                cv_scores['recall'].append(recall)
                cv_scores['f1'].append(f1)
                
                predictions_all.extend(y_pred)
                actuals_all.extend(y_test)
                probabilities_all.extend(y_prob)
            
            # Calculate average metrics
            results[name] = {
                'cv_accuracy_mean': np.mean(cv_scores['accuracy']),
                'cv_accuracy_std': np.std(cv_scores['accuracy']),
                'cv_precision_mean': np.mean(cv_scores['precision']),
                'cv_precision_std': np.std(cv_scores['precision']),
                'cv_recall_mean': np.mean(cv_scores['recall']),
                'cv_recall_std': np.std(cv_scores['recall']),
                'cv_f1_mean': np.mean(cv_scores['f1']),
                'cv_f1_std': np.std(cv_scores['f1']),
                'predictions': predictions_all,
                'actuals': actuals_all,
                'probabilities': probabilities_all
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                if isinstance(X, pd.DataFrame):
                    feature_cols = X.columns
                else:
                    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
            
            logger.info(f"✅ {name}: Accuracy={results[name]['cv_accuracy_mean']:.4f}±{results[name]['cv_accuracy_std']:.4f}, "
                    f"F1={results[name]['cv_f1_mean']:.4f}±{results[name]['cv_f1_std']:.4f}")
        
        return results
    
    def create_model_comparison_plots(self, regression_results: Dict, classification_results: Dict) -> go.Figure:
        """Create comprehensive model comparison visualizations"""
        logger.info("Creating model comparison visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Regression Model Performance (R²)', 'Classification Model Performance (F1)',
                'Regression Predictions vs Actual', 'Classification ROC Curves',
                'Feature Importance (Random Forest)', 'Model Metrics Comparison'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Regression R² comparison
        if regression_results:
            reg_names = list(regression_results.keys())
            reg_r2_means = [regression_results[name]['cv_r2_mean'] for name in reg_names]
            reg_r2_stds = [regression_results[name]['cv_r2_std'] for name in reg_names]
            
            fig.add_trace(
                go.Bar(x=reg_names, y=reg_r2_means, 
                      error_y=dict(type='data', array=reg_r2_stds),
                      name='R² Score', marker_color='blue'),
                row=1, col=1
            )
        
        # 2. Classification F1 comparison
        if classification_results:
            class_names = list(classification_results.keys())
            class_f1_means = [classification_results[name]['cv_f1_mean'] for name in class_names]
            class_f1_stds = [classification_results[name]['cv_f1_std'] for name in class_names]
            
            fig.add_trace(
                go.Bar(x=class_names, y=class_f1_means,
                      error_y=dict(type='data', array=class_f1_stds),
                      name='F1 Score', marker_color='green'),
                row=1, col=2
            )
        
        # 3. Regression predictions vs actual (best model)
        if regression_results:
            best_reg_model = max(regression_results.keys(), 
                               key=lambda x: regression_results[x]['cv_r2_mean'])
            best_reg_results = regression_results[best_reg_model]
            
            fig.add_trace(
                go.Scatter(x=best_reg_results['actuals'][:1000],  # Limit points for performance
                          y=best_reg_results['predictions'][:1000],
                          mode='markers', name=f'{best_reg_model} Predictions',
                          marker=dict(size=3, opacity=0.6)),
                row=1, col=3
            )
            
            # Add perfect prediction line
            min_val = min(min(best_reg_results['actuals']), min(best_reg_results['predictions']))
            max_val = max(max(best_reg_results['actuals']), max(best_reg_results['predictions']))
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', name='Perfect Prediction',
                          line=dict(dash='dash', color='red')),
                row=1, col=3
            )
        
        # 4. Classification probability distribution
        if classification_results:
            best_class_model = max(classification_results.keys(),
                                 key=lambda x: classification_results[x]['cv_f1_mean'])
            best_class_results = classification_results[best_class_model]
            
            # Separate probabilities by actual class
            probs = np.array(best_class_results['probabilities'])
            actuals = np.array(best_class_results['actuals'])
            
            fig.add_trace(
                go.Histogram(x=probs[actuals == 0], name='Negative Class',
                           opacity=0.7, nbinsx=30),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=probs[actuals == 1], name='Positive Class',
                           opacity=0.7, nbinsx=30),
                row=2, col=1
            )
        
        # 5. Feature importance (Random Forest)
        if regression_results and 'Random_Forest' in regression_results:
            rf_importance = regression_results['Random_Forest'].get('feature_importance')
            if rf_importance is not None:
                top_features = rf_importance.head(10)
                fig.add_trace(
                    go.Bar(x=top_features['importance'], y=top_features['feature'],
                          orientation='h', name='Feature Importance'),
                    row=2, col=2
                )
        
        # 6. Model metrics comparison table
        metrics_text = []
        if regression_results:
            metrics_text.append("REGRESSION MODELS:")
            for name, results in regression_results.items():
                metrics_text.append(f"{name}: R²={results['cv_r2_mean']:.4f}")

        if classification_results:
            metrics_text.append("CLASSIFICATION MODELS:")
            for name, results in classification_results.items():
                metrics_text.append(f"{name}: F1={results['cv_f1_mean']:.4f}")

        # Add as text annotation instead of table
        fig.add_annotation(
            text="<br>".join(metrics_text),
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Baseline Model Performance Comparison - Day 6",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def save_models_and_results(self, regression_results: Dict, classification_results: Dict,
                              feature_cols: List[str]) -> Dict[str, str]:
        """Save trained models and evaluation results"""
        logger.info("Saving models and evaluation results...")
        
        saved_files = {}
        
        # Create models directory
        models_dir = self.config.PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        for prediction_type in ['regression', 'classification']:
            if prediction_type in self.models:
                for model_name, model in self.models[prediction_type].items():
                    model_path = models_dir / f"{prediction_type}_{model_name.lower().replace(' ', '_')}.joblib"
                    joblib.dump(model, model_path)
                    logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scaler
        if 'main' in self.scalers:
            scaler_path = models_dir / "feature_scaler.joblib"
            joblib.dump(self.scalers['main'], scaler_path)
            saved_files['scaler'] = str(scaler_path)
        
        # Save regression results
        if regression_results:
            reg_results_df = pd.DataFrame({
                name: {
                    'cv_r2_mean': results['cv_r2_mean'],
                    'cv_r2_std': results['cv_r2_std'],
                    'cv_mae_mean': results['cv_mae_mean'],
                    'cv_mae_std': results['cv_mae_std'],
                    'rmse_mean': results['rmse_mean']
                }
                for name, results in regression_results.items()
            }).T
            
            reg_path = self.config.PROCESSED_DATA_PATH / "regression_model_results.csv"
            reg_results_df.to_csv(reg_path)
            saved_files['regression_results'] = str(reg_path)
            logger.info(f"Regression results saved to {reg_path}")
        
        # Save classification results
        if classification_results:
            class_results_df = pd.DataFrame({
                name: {
                    'cv_accuracy_mean': results['cv_accuracy_mean'],
                    'cv_accuracy_std': results['cv_accuracy_std'],
                    'cv_f1_mean': results['cv_f1_mean'],
                    'cv_f1_std': results['cv_f1_std'],
                    'cv_precision_mean': results['cv_precision_mean'],
                    'cv_precision_std': results['cv_precision_std']
                }
                for name, results in classification_results.items()
            }).T
            
            class_path = self.config.PROCESSED_DATA_PATH / "classification_model_results.csv"
            class_results_df.to_csv(class_path)
            saved_files['classification_results'] = str(class_path)
            logger.info(f"Classification results saved to {class_path}")
        
        # Save feature importance
        feature_importance_data = {}
        
        for prediction_type, results in [('regression', regression_results), ('classification', classification_results)]:
            if results:
                for model_name, model_results in results.items():
                    if 'feature_importance' in model_results:
                        feature_importance_data[f"{prediction_type}_{model_name}"] = model_results['feature_importance']
        
        if feature_importance_data:
            # Combine all feature importance results
            combined_importance = pd.DataFrame()
            for model_name, importance_df in feature_importance_data.items():
                temp_df = importance_df.copy()
                temp_df['model'] = model_name
                combined_importance = pd.concat([combined_importance, temp_df], ignore_index=True)
            
            importance_path = self.config.PROCESSED_DATA_PATH / "feature_importance_analysis.csv"
            combined_importance.to_csv(importance_path, index=False)
            saved_files['feature_importance'] = str(importance_path)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Save comprehensive summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': {
                'regression': list(regression_results.keys()) if regression_results else [],
                'classification': list(classification_results.keys()) if classification_results else []
            },
            'best_models': {
                'regression': max(regression_results.keys(), key=lambda x: regression_results[x]['cv_r2_mean']) if regression_results else None,
                'classification': max(classification_results.keys(), key=lambda x: classification_results[x]['cv_f1_mean']) if classification_results else None
            },
            'feature_count': len(feature_cols),
            'features_used': feature_cols[:20]  # First 20 features for summary
        }
        
        import json
        summary_path = self.config.PROCESSED_DATA_PATH / "day6_model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = str(summary_path)
        logger.info(f"Model summary saved to {summary_path}")
        
        return saved_files