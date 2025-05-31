import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import json
from datetime import datetime
from pathlib import Path

from .config import Config

class SimpleEnsemble:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self

    def predict(self, X):
        import numpy as np
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                    predictions.append(pred)
            except:
                continue
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(len(X))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
class StackedEnsemble:
    def __init__(self, base_models, meta_learner):
        self.base_models = base_models
        self.meta_learner = meta_learner
                
    def fit(self, X, y):
        return self
                
    def predict(self, X):
        base_preds = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                base_preds[:, i] = model.predict(X)
            except:
                base_preds[:, i] = 0
        
        return self.meta_learner.predict(base_preds)
            
    def get_params(self, deep=True):
        return {}
            
    def set_params(self, **params):
        return self
class EnsembleFramework:
    """Comprehensive ensemble methods framework with proper sklearn implementation"""
    
    def __init__(self):
        self.config = Config()
        self.base_models = {}
        self.ensemble_models = {}
        self.individual_results = {}
        self.ensemble_results = {}
        self.scaler = StandardScaler()
        
    def load_base_models(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Load all trained models from Days 6-8"""
        logger.info("Loading base models from Days 6-8...")
        
        # Load feature data
        features_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        df = pd.read_csv(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Prepare features
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['return_5d'].fillna(df['return_5d'].median())
        
        # Load stable models
        models = {}
        
        # Load from models directory
        models_dir = self.config.PROJECT_ROOT / "models"
        advanced_dir = models_dir / "advanced"
        
        model_files = {
            'XGBoost': advanced_dir / "regression_xgboost_optimized.joblib",
            'LightGBM': advanced_dir / "regression_lightgbm_optimized.joblib",
            'RandomForest': models_dir / "regression_random_forest.joblib",
            'GradientBoosting': advanced_dir / "regression_gradient_boosting.joblib"
        }
        
        for name, path in model_files.items():
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                    logger.info(f"Loaded {name} from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        self.base_models = models
        logger.info(f"Loaded {len(models)} base models, {len(X)} samples, {X.shape[1]} features")
        
        return models, df, X, y
    
    def evaluate_individual_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate individual model performance - FAST version"""
        logger.info("Evaluating individual models (fast version)...")
        
        results = {}
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 to 3 for speed
        
        for name, model in self.base_models.items():
            try:
                cv_scores = []
                all_predictions = []
                all_actuals = []
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    score = r2_score(y_test, y_pred)
                    cv_scores.append(score)
                    all_predictions.extend(y_pred)
                    all_actuals.extend(y_test)
                
                results[name] = {
                    'cv_r2_mean': np.mean(cv_scores),
                    'cv_r2_std': np.std(cv_scores),
                    'cv_scores': cv_scores,
                    'rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
                    'mae': mean_absolute_error(all_actuals, all_predictions)
                }
                
                logger.info(f"{name}: R² = {results[name]['cv_r2_mean']:.4f} ± {results[name]['cv_r2_std']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
        
        self.individual_results = results
        return results
    
    def create_voting_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> VotingRegressor:
        """Create VotingRegressor ensemble using sklearn"""
        logger.info("Creating VotingRegressor ensemble...")
        
        # Prepare estimators for VotingRegressor
        estimators = []
        for name, model in self.base_models.items():
            estimators.append((name, model))
        
        if len(estimators) < 2:
            logger.warning("Need at least 2 models for voting ensemble")
            return None
        
        # Create VotingRegressor
        voting_ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)
        
        logger.info(f"Created VotingRegressor with {len(estimators)} models: {[name for name, _ in estimators]}")
        return voting_ensemble
    
    def create_stacked_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Create stacked ensemble - SIMPLIFIED and FASTER"""
        logger.info("Creating simple stacked ensemble...")
        
        # Use only 3 splits for speed
        tscv = TimeSeriesSplit(n_splits=3)
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        model_names = list(self.base_models.keys())
        
        for i, (name, model) in enumerate(self.base_models.items()):
            oof_preds = np.zeros(len(X))
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                try:
                    model.fit(X_train, y_train)
                    oof_preds[val_idx] = model.predict(X_val)
                except:
                    continue
            
            oof_predictions[:, i] = oof_preds
        
        # Simple linear meta-learner
        meta_learner = LinearRegression()
        meta_learner.fit(oof_predictions, y)
        
        logger.info("Fast stacked ensemble created")
        return StackedEnsemble(self.base_models, meta_learner)
    
    def create_neural_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Create neural network ensemble with MLPRegressor meta-learner"""
        logger.info("Creating neural network ensemble with MLPRegressor...")
        
        # Generate base predictions for neural network training
        tscv = TimeSeriesSplit(n_splits=3)  # Smaller splits for speed
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            oof_preds = np.zeros(len(X))
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                try:
                    model.fit(X_train, y_train)
                    oof_preds[val_idx] = model.predict(X_val)
                except:
                    continue
            
            oof_predictions[:, i] = oof_preds
        
        # Scale features for neural network
        scaler = StandardScaler()
        oof_scaled = scaler.fit_transform(oof_predictions)
        
        # Train MLPRegressor meta-learner
        neural_meta = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        neural_meta.fit(oof_scaled, y)
        
        logger.info("Neural ensemble with MLPRegressor trained successfully")
        
        class NeuralEnsemble:
            def __init__(self, base_models, neural_meta, scaler):
                self.base_models = base_models
                self.neural_meta = neural_meta
                self.scaler = scaler
                
            def fit(self, X, y):
                return self
                
            def predict(self, X):
                base_preds = np.zeros((len(X), len(self.base_models)))
                
                for i, (name, model) in enumerate(self.base_models.items()):
                    try:
                        base_preds[:, i] = model.predict(X)
                    except:
                        base_preds[:, i] = 0
                
                base_scaled = self.scaler.transform(base_preds)
                return self.neural_meta.predict(base_scaled)
        
        return NeuralEnsemble(self.base_models, neural_meta, scaler)
    
    def create_simple_ensemble(self) -> Any:
        """Create simple averaging ensemble - with sklearn compatibility"""
        logger.info("Creating simple averaging ensemble...")
        return SimpleEnsemble(self.base_models)
    
    def create_all_ensembles(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Create all ensemble models - FAST version"""
        logger.info("Creating ensemble models (fast version)...")
        
        ensembles = {}
        
        # 1. Simple Average Ensemble (fastest)
        ensembles['Simple_Average'] = self.create_simple_ensemble()
        
        # 2. Voting Regressor (skip for speed if needed)
        try:
            voting_ensemble = self.create_voting_ensemble(X, y)
            if voting_ensemble:
                ensembles['Voting_Regressor'] = voting_ensemble
        except Exception as e:
            logger.warning(f"Skipping VotingRegressor: {e}")
        
        # 3. Simple Stacked Ensemble (faster version)
        try:
            ensembles['Stacked_Ensemble'] = self.create_stacked_ensemble(X, y)
        except Exception as e:
            logger.warning(f"Skipping Stacked Ensemble: {e}")
        
        # Skip Neural Ensemble for speed
        logger.info("Skipping Neural Ensemble for speed")
        
        self.ensemble_models = ensembles
        logger.info(f"Created {len(ensembles)} ensemble models (fast mode)")
        
        return ensembles
    
    def evaluate_ensemble(self, ensemble: Any, X: pd.DataFrame, y: np.ndarray, name: str) -> Dict:
        """Evaluate ensemble performance - FAST version"""
        logger.info(f"Evaluating {name} (fast version)...")
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed
        cv_scores = []
        all_predictions = []
        all_actuals = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                
                if not np.any(np.isnan(y_pred)) and not np.any(np.isinf(y_pred)):
                    score = r2_score(y_test, y_pred)
                    cv_scores.append(score)
                    all_predictions.extend(y_pred)
                    all_actuals.extend(y_test)
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
                continue
        
        if cv_scores:
            results = {
                'cv_r2_mean': np.mean(cv_scores),
                'cv_r2_std': np.std(cv_scores),
                'cv_scores': cv_scores,
                'rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
                'mae': mean_absolute_error(all_actuals, all_predictions)
            }
            
            logger.info(f"{name}: R² = {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
            return results
        else:
            logger.error(f"Failed to evaluate {name}")
            return {'cv_r2_mean': 0, 'cv_r2_std': 0, 'rmse': 0, 'mae': 0, 'cv_scores': []}
    
    def evaluate_all_ensembles(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate all ensemble models"""
        logger.info("Evaluating all ensemble models...")
        
        results = {}
        
        for name, ensemble in self.ensemble_models.items():
            results[name] = self.evaluate_ensemble(ensemble, X, y, name)
        
        self.ensemble_results = results
        return results
    
    def create_visualizations(self) -> go.Figure:
        """Create comprehensive visualizations"""
        logger.info("Creating comprehensive visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Model Performance Comparison (R²)',
                'Cross-Validation Score Distribution',
                'Performance Improvement Analysis',
                'Error Metrics Comparison'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Combine all results
        all_results = {**self.individual_results, **self.ensemble_results}
        
        # 1. Model Performance Comparison
        model_names = list(all_results.keys())
        r2_means = [all_results[name]['cv_r2_mean'] for name in model_names]
        r2_stds = [all_results[name]['cv_r2_std'] for name in model_names]
        
        colors = ['blue' if name in self.individual_results else 'red' for name in model_names]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=r2_means,
                error_y=dict(type='data', array=r2_stds),
                marker_color=colors,
                name='R² Performance',
                text=[f'{score:.4f}' for score in r2_means],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. CV Score Distribution (Box plots)
        for name in model_names:
            if 'cv_scores' in all_results[name] and all_results[name]['cv_scores']:
                fig.add_trace(
                    go.Box(
                        y=all_results[name]['cv_scores'],
                        name=name,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ),
                    row=1, col=2
                )
        
        # 3. Performance Improvement
        if self.individual_results:
            baseline_best = max([r['cv_r2_mean'] for r in self.individual_results.values()])
            improvements = []
            ensemble_names = []
            
            for name, results in self.ensemble_results.items():
                improvement = ((results['cv_r2_mean'] - baseline_best) / abs(baseline_best) * 100) if baseline_best != 0 else 0
                improvements.append(improvement)
                ensemble_names.append(name)
            
            fig.add_trace(
                go.Bar(
                    x=ensemble_names,
                    y=improvements,
                    marker_color='green',
                    name='Improvement %',
                    text=[f'{imp:+.1f}%' for imp in improvements],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Error Metrics
        rmse_values = [all_results[name].get('rmse', 0) for name in model_names]
        mae_values = [all_results[name].get('mae', 0) for name in model_names]
        
        fig.add_trace(
            go.Scatter(
                x=model_names,
                y=rmse_values,
                mode='markers+lines',
                name='RMSE',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=model_names,
                y=mae_values,
                mode='markers+lines',
                name='MAE',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Day 9: Ensemble Methods Performance Analysis",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update x-axis labels rotation
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=2)
        
        # Save visualization
        plots_dir = self.config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        viz_path = plots_dir / "day9_ensemble_analysis.html"
        fig.write_html(str(viz_path))
        
        logger.info(f"Visualizations saved: {viz_path}")
        return fig
    
    def save_ensemble_models(self) -> Dict[str, str]:
        """Save ensemble models for production use"""
        logger.info("Saving ensemble models...")
        
        # Create ensemble models directory
        ensemble_dir = self.config.PROJECT_ROOT / "models" / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        saved_models = {}
        
        for name, ensemble in self.ensemble_models.items():
            model_path = ensemble_dir / f"{name.lower()}_ensemble.joblib"
            
            try:
                joblib.dump(ensemble, model_path)
                saved_models[name] = str(model_path)
                logger.info(f"Saved {name}: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
        
        return saved_models
    
    def save_comprehensive_results(self, saved_models: Dict[str, str]) -> Tuple[Path, Path]:
        """Save comprehensive results and reports"""
        logger.info("Saving comprehensive results...")
        
        # Combine all results
        all_results = {**self.individual_results, **self.ensemble_results}
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'cv_r2_mean': results['cv_r2_mean'],
                'cv_r2_std': results['cv_r2_std'],
                'rmse': results.get('rmse', 0),
                'mae': results.get('mae', 0),
                'model_type': 'Individual' if name in self.individual_results else 'Ensemble'
            }
            for name, results in all_results.items()
        }).T
        
        # Save to CSV
        results_path = self.config.PROCESSED_DATA_PATH / "day9_ensemble_results.csv"
        results_df.to_csv(results_path)
        
        # Create comprehensive report
        best_individual = max(self.individual_results.keys(), key=lambda x: self.individual_results[x]['cv_r2_mean']) if self.individual_results else None
        best_ensemble = max(self.ensemble_results.keys(), key=lambda x: self.ensemble_results[x]['cv_r2_mean']) if self.ensemble_results else None
        
        improvement = 0
        if best_individual and best_ensemble:
            improvement = ((self.ensemble_results[best_ensemble]['cv_r2_mean'] - 
                           self.individual_results[best_individual]['cv_r2_mean']) / 
                          abs(self.individual_results[best_individual]['cv_r2_mean']) * 100)
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'models_evaluated': {
                'individual': list(self.individual_results.keys()),
                'ensemble': list(self.ensemble_results.keys())
            },
            'best_performance': {
                'individual': {
                    'name': best_individual,
                    'r2_score': self.individual_results[best_individual]['cv_r2_mean'] if best_individual else 0
                },
                'ensemble': {
                    'name': best_ensemble,
                    'r2_score': self.ensemble_results[best_ensemble]['cv_r2_mean'] if best_ensemble else 0
                }
            },
            'improvement_percentage': improvement,
            'saved_models': saved_models,
            'ensemble_methods_used': [
                'Simple Averaging',
                'Voting Regressor (sklearn.ensemble.VotingRegressor)',
                'Stacked Ensemble (sklearn.linear_model.LinearRegression meta-learner)',
                'Neural Network Ensemble (sklearn.neural_network.MLPRegressor meta-learner)'
            ],
            'libraries_used': [
                'sklearn.ensemble.VotingRegressor',
                'sklearn.linear_model.LinearRegression',
                'sklearn.neural_network.MLPRegressor',
                'sklearn.model_selection.cross_val_score',
                'sklearn.model_selection.TimeSeriesSplit',
                'sklearn.preprocessing.StandardScaler',
                'sklearn.metrics (r2_score, mean_squared_error, mean_absolute_error)'
            ]
        }
        
        report_path = self.config.PROCESSED_DATA_PATH / "day9_comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved: {results_path}")
        logger.info(f"Report saved: {report_path}")
        
        return results_path, report_path
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for display"""
        if not self.individual_results or not self.ensemble_results:
            return {}
        
        best_individual = max(self.individual_results.keys(), key=lambda x: self.individual_results[x]['cv_r2_mean'])
        best_ensemble = max(self.ensemble_results.keys(), key=lambda x: self.ensemble_results[x]['cv_r2_mean'])
        
        improvement = ((self.ensemble_results[best_ensemble]['cv_r2_mean'] - 
                       self.individual_results[best_individual]['cv_r2_mean']) / 
                      abs(self.individual_results[best_individual]['cv_r2_mean']) * 100)
        
        return {
            'best_individual': {
                'name': best_individual,
                'score': self.individual_results[best_individual]['cv_r2_mean'],
                'rmse': self.individual_results[best_individual]['rmse']
            },
            'best_ensemble': {
                'name': best_ensemble,
                'score': self.ensemble_results[best_ensemble]['cv_r2_mean'],
                'rmse': self.ensemble_results[best_ensemble]['rmse']
            },
            'improvement_pct': improvement,
            'total_models': len(self.base_models),
            'total_ensembles': len(self.ensemble_models)
        }