import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Any
from loguru import logger
import time
from pathlib import Path
import ta

from .config import Config
from .consensus_model import ConsensusMetaEnsemble

class RealTimePredictionEngine:
    """High-performance real-time prediction system"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.best_model = None
        self.primary_model_name = ""
        self.model_scores = {}
        self.feature_columns = []
        self.scalers = {}
        self.last_predictions = {}
        self.alert_thresholds = {
            'high_confidence': 0.01,  # Lower threshold for better signals
            'risk_limit': 0.05        # Lower risk threshold
        }
        self.performance_cache = {}
        self.drift_detector = ModelDriftDetector(self.config)

    def _load_model_scores(self) -> Dict[str, float]:
        """Load model ranking scores from the latest saved risk summary."""
        risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
        if not risk_summary_path.exists():
            return {}

        try:
            risk_df = pd.read_csv(risk_summary_path)
            if risk_df.empty or 'Model' not in risk_df.columns:
                return {}

            score_column = 'Sharpe_Ratio'
            if 'Gross_Sharpe_Ratio' in risk_df.columns:
                score_column = 'Sharpe_Ratio'

            scores = {}
            for _, row in risk_df.iterrows():
                raw_score = float(row.get(score_column, 0.0))
                scores[row['Model']] = max(raw_score, 0.05)
            return scores
        except Exception as exc:
            logger.warning(f"Failed to load model scores from risk summary: {exc}")
            return {}
        
    def load_production_models(self) -> bool:
        """Load best performing models for production"""
        logger.info("Loading production models...")
        
        try:
            ensemble_dir = self.config.PROJECT_ROOT / "models" / "ensemble"
            model_files = {
                'Ensemble_SimpleAverage': ensemble_dir / "simple_average_ensemble.joblib",
                'Ensemble_VotingRegressor': ensemble_dir / "voting_regressor_ensemble.joblib",
                'Ensemble_StackedEnsemble': ensemble_dir / "stacked_ensemble_ensemble.joblib",
                'XGBoost': self.config.PROJECT_ROOT / "models" / "advanced" / "regression_xgboost_optimized.joblib",
                'LightGBM': self.config.PROJECT_ROOT / "models" / "advanced" / "regression_lightgbm_optimized.joblib",
                'RandomForest': self.config.PROJECT_ROOT / "models" / "regression_random_forest.joblib"
            }

            for name, path in model_files.items():
                if path.exists():
                    self.models[name] = joblib.load(path)
                    logger.info(f"✅ Loaded production model: {name}")

            self.model_scores = self._load_model_scores()
            consensus_components = {
                name: model for name, model in self.models.items()
                if name in {
                    'Ensemble_VotingRegressor',
                    'Ensemble_SimpleAverage',
                    'XGBoost',
                    'LightGBM',
                    'RandomForest',
                }
            }
            if len(consensus_components) >= 2:
                self.models['ConsensusMetaEnsemble'] = ConsensusMetaEnsemble(
                    consensus_components,
                    model_scores=self.model_scores,
                )
                logger.info("✅ Loaded derived production model: ConsensusMetaEnsemble")

            if self.models:
                ranked_models = sorted(
                    self.models.keys(),
                    key=lambda model_name: self.model_scores.get(model_name, 0.0),
                    reverse=True,
                )
                self.primary_model_name = ranked_models[0]
                self.best_model = self.models[self.primary_model_name]
                logger.info(f"✅ Selected primary production model: {self.primary_model_name}")
            
            # Load EXACT feature columns from training data
            self.feature_columns = self._get_exact_training_features()
            
            if not self.feature_columns:
                logger.error("Failed to load feature columns")
                return False
            
            logger.info(f"✅ Loaded {len(self.feature_columns)} feature columns")
            return len(self.models) > 0 and len(self.feature_columns) > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
    
    def _get_exact_training_features(self) -> List[str]:
        """Get exact feature list that models expect (73 features)"""
        try:
            # First try the model-ready features (if created by diagnostics)
            model_ready_path = self.config.FEATURES_DATA_PATH / "model_ready_features.txt"
            if model_ready_path.exists():
                with open(model_ready_path, 'r') as f:
                    features = [line.strip() for line in f.readlines() if line.strip()]
                if len(features) == 73:
                    logger.info(f"Using model-ready features: {len(features)}")
                    return features
            
            # Load training data to get features and exclude target-related ones
            training_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
            if training_path.exists():
                df = pd.read_csv(training_path)
                
                # Exclude metadata, targets, AND sharpe_5d (which is target-related)
                exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d', 'sharpe_5d']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                logger.info(f"Extracted {len(feature_cols)} features (excluded sharpe_5d)")
                
                # Save for future use
                if len(feature_cols) == 73:
                    model_ready_path = self.config.FEATURES_DATA_PATH / "model_ready_features.txt"
                    with open(model_ready_path, 'w') as f:
                        for feature in feature_cols:
                            f.write(f"{feature}\n")
                    logger.info(f"Saved model-ready features to {model_ready_path}")
                
                return feature_cols
            else:
                # Fallback to feature list file
                features_path = self.config.FEATURES_DATA_PATH / "selected_features_list.txt"
                if features_path.exists():
                    with open(features_path, 'r') as f:
                        features = [line.strip() for line in f.readlines() if line.strip()]
                    
                    # If we have 74 features, remove sharpe_5d
                    if len(features) == 74 and 'sharpe_5d' in features:
                        features.remove('sharpe_5d')
                        logger.info("Removed sharpe_5d from feature list")
                    
                    return features
                else:
                    logger.error("No feature source found")
                    return []
        except Exception as e:
            logger.error(f"Failed to get training features: {e}")
            return []
    
    def get_target_stocks(self) -> List[str]:
        """Get target stocks for real-time monitoring"""
        try:
            stocks_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
            if stocks_path.exists():
                with open(stocks_path, 'r') as f:
                    stocks = [line.strip() for line in f.readlines()]
                return stocks[:5]  # Top 5 for real-time efficiency
            else:
                # Fallback to popular stocks
                return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD']
        except Exception as e:
            logger.warning(f"Using fallback stocks: {e}")
            return ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'AMD']
    
    async def fetch_realtime_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch real-time market data efficiently - need more data for proper feature engineering"""
        logger.info(f"Fetching real-time data for {len(symbols)} symbols...")
        
        market_data = {}
        
        try:
            # Use longer period to ensure sufficient data for all indicators
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                
                # Get more historical data for proper indicator calculation
                hist = ticker.history(period=period, interval="1d")
                
                if not hist.empty and len(hist) >= 200:  # Ensure sufficient data
                    # Ensure we have required columns
                    hist = hist.reset_index()
                    hist['Ticker'] = symbol
                    
                    # Handle datetime column
                    if 'Date' not in hist.columns and 'Datetime' in hist.columns:
                        hist['Date'] = hist['Datetime']
                    elif hist.index.name == 'Date':
                        hist['Date'] = hist.index
                    
                    # Add Stock_Splits column if missing
                    if 'Stock Splits' in hist.columns:
                        hist['Stock_Splits'] = hist['Stock Splits']
                    else:
                        hist['Stock_Splits'] = 0
                    
                    market_data[symbol] = hist
                    logger.info(f"✅ Fetched {len(hist)} records for {symbol}")
                else:
                    logger.warning(f"⚠️ Insufficient data for {symbol} ({len(hist) if not hist.empty else 0} records)")
                    
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
        
        return market_data
    
    def engineer_realtime_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precisely replicate training feature engineering"""
        logger.info(f"Engineering features for {symbol}...")
        
        try:
            # Ensure we have enough data for all indicators
            if len(df) < 200:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} records")
                return pd.DataFrame()
            
            # Sort by date and reset index
            df = df.sort_values('Date').reset_index(drop=True)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # === EXACT REPLICATION OF TRAINING FEATURES ===
            
            # Basic price features
            df['price_change'] = df['Close'] - df['Open']
            df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['daily_return'] = df['Close'].pct_change() * 100
            
            # Technical indicators - use TA library for consistency
            # Moving averages
            df['sma_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['sma_100'] = ta.trend.sma_indicator(df['Close'], window=100)
            df['sma_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # Exponential moving averages
            df['ema_5'] = ta.trend.ema_indicator(df['Close'], window=5)
            df['ema_10'] = ta.trend.ema_indicator(df['Close'], window=10)
            df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            df['ema_100'] = ta.trend.ema_indicator(df['Close'], window=100)
            df['ema_200'] = ta.trend.ema_indicator(df['Close'], window=200)
            
            # Price relative to moving averages
            df['close_to_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20'] * 100
            df['close_to_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50'] * 100
            
            # RSI and momentum
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            
            # Momentum features
            df['momentum_1d'] = df['Close'].pct_change(1) * 100
            df['momentum_5d'] = df['Close'].pct_change(5) * 100
            df['momentum_10d'] = df['Close'].pct_change(10) * 100
            df['momentum_20d'] = df['Close'].pct_change(20) * 100
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            
            # ATR and volatility
            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['atr_ratio'] = df['atr'] / df['Close'] * 100
            
            # Rolling statistics
            df['close_mean_5'] = df['Close'].rolling(5).mean()
            df['close_mean_10'] = df['Close'].rolling(10).mean()
            df['close_mean_20'] = df['Close'].rolling(20).mean()
            df['close_std_20'] = df['Close'].rolling(20).std()
            df['volatility_20d'] = df['close_std_20'] / df['close_mean_20'] * 100
            
            # Min/Max rolling windows
            df['close_min_5'] = df['Close'].rolling(5).min()
            df['close_min_10'] = df['Close'].rolling(10).min()
            df['close_min_20'] = df['Close'].rolling(20).min()
            df['close_max_5'] = df['Close'].rolling(5).max()
            df['close_max_10'] = df['Close'].rolling(10).max()
            df['close_max_20'] = df['Close'].rolling(20).max()
            
            # Lag features
            df['close_lag_1'] = df['Close'].shift(1)
            df['close_lag_2'] = df['Close'].shift(2)
            df['close_lag_3'] = df['Close'].shift(3)
            df['close_lag_5'] = df['Close'].shift(5)
            df['close_lag_10'] = df['Close'].shift(10)
            
            # Volume features
            df['volume_sma20'] = df['Volume'].rolling(20).mean()
            df['volume_mean_5'] = df['Volume'].rolling(5).mean()
            df['volume_mean_10'] = df['Volume'].rolling(10).mean()
            df['volume_mean_20'] = df['Volume'].rolling(20).mean()
            df['volume_std_5'] = df['Volume'].rolling(5).std()
            df['volume_std_10'] = df['Volume'].rolling(10).std()
            df['volume_std_20'] = df['Volume'].rolling(20).std()
            
            # Volume lags
            df['volume_lag_1'] = df['Volume'].shift(1)
            df['volume_lag_2'] = df['Volume'].shift(2)
            df['volume_lag_3'] = df['Volume'].shift(3)
            df['volume_lag_5'] = df['Volume'].shift(5)
            df['volume_lag_10'] = df['Volume'].shift(10)
            
            # Additional technical indicators
            df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Price range and candlestick features
            df['price_range'] = df['High'] - df['Low']
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            
            # VWAP
            df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            
            # Price position
            df['price_position_10'] = (df['Close'] - df['close_min_10']) / (df['close_max_10'] - df['close_min_10'] + 1e-8)
            
            # Time features
            df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            
            # Volatility regime
            vol_threshold = df['volatility_20d'].rolling(100, min_periods=20).quantile(0.2)
            df['low_volatility'] = (df['volatility_20d'] < vol_threshold).fillna(False).astype(int)
            
            # Special target feature (set to 0 for real-time)
            df['sharpe_5d'] = 0.0
            
            # === CRITICAL: Fill NaN values systematically ===
            
            # First pass: forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Second pass: fill remaining NaN with appropriate defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], 0)
            
            # Ensure boolean/integer columns are properly typed
            int_columns = ['rsi_oversold', 'low_volatility', 'Stock_Splits']
            for col in int_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            
            logger.info(f"✅ Engineered {df.shape[1]} features for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def prepare_prediction_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features EXACTLY as expected by trained models"""
        try:
            if df.empty:
                return None
            
            # Get the latest record (most recent data)
            latest_data = df.iloc[-1:].copy()
            
            # Create feature vector in EXACT order expected by model
            feature_vector = []
            missing_features = []
            
            for feature in self.feature_columns:
                if feature in latest_data.columns:
                    value = latest_data[feature].iloc[0]
                    # Handle any remaining NaN or inf values
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    feature_vector.append(float(value))
                else:
                    # Missing feature - use 0 as default
                    feature_vector.append(0.0)
                    missing_features.append(feature)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:3]}...")
            
            # Convert to numpy array with correct shape
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Final validation - CRITICAL CHECK
            expected_features = len(self.feature_columns)
            actual_features = feature_array.shape[1]
            
            if actual_features != expected_features:
                logger.error(f"FEATURE MISMATCH: Expected {expected_features}, got {actual_features}")
                
                # Debug output
                logger.error(f"Expected features: {self.feature_columns[:5]}...")
                logger.error(f"Available features: {list(latest_data.columns)[:5]}...")
                return None
            
            assert not np.any(np.isnan(feature_array)), "NaN values in feature array"
            assert not np.any(np.isinf(feature_array)), "Inf values in feature array"
            
            logger.info(f"✅ Prepared exactly {expected_features} features for prediction")
            return feature_array
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_predictions(self, feature_data: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Generate predictions using loaded models"""
        predictions = {}
        
        try:
            if self.models:
                model_predictions = {}
                for model_name, model in self.models.items():
                    try:
                        model_pred = model.predict(feature_data)[0]
                        if pd.isna(model_pred) or np.isinf(model_pred):
                            continue
                        model_predictions[model_name] = float(model_pred)
                        predictions[model_name] = float(model_pred)
                    except Exception as model_error:
                        logger.warning(f"Model {model_name} prediction failed: {model_error}")

                if not model_predictions:
                    return predictions

                consensus_prediction = model_predictions.get('ConsensusMetaEnsemble')
                base_prediction_names = [
                    name for name in model_predictions.keys()
                    if name != 'ConsensusMetaEnsemble'
                ]
                available_names = base_prediction_names or list(model_predictions.keys())
                raw_scores = np.array([self.model_scores.get(name, 1.0) for name in available_names], dtype=float)
                normalized_scores = raw_scores / raw_scores.sum() if raw_scores.sum() > 0 else np.ones(len(raw_scores)) / len(raw_scores)
                all_preds = np.array([model_predictions[name] for name in available_names], dtype=float)

                weighted_pred = float(np.average(all_preds, weights=normalized_scores))
                prediction_dispersion = float(np.std(all_preds)) if len(all_preds) > 1 else 0.0
                interval_width = max(prediction_dispersion, self.alert_thresholds['high_confidence'] / 2)
                top_model_name = max(available_names, key=lambda name: self.model_scores.get(name, 0.0))
                primary_prediction = float(consensus_prediction) if consensus_prediction is not None else weighted_pred
                threshold = self.config.signal_threshold_pct()
                confidence_label = (
                    'high' if abs(primary_prediction) > max(self.alert_thresholds['high_confidence'], threshold * 2)
                    else 'medium' if abs(primary_prediction) > threshold
                    else 'low'
                )

                predictions['primary'] = {
                    'model': 'ConsensusMetaEnsemble' if consensus_prediction is not None else 'ScoreWeightedEnsemble',
                    'prediction': primary_prediction,
                    'confidence': confidence_label,
                    'top_component_model': top_model_name,
                }
                predictions['ensemble_average'] = float(np.mean(all_preds))
                predictions['weighted_ensemble'] = weighted_pred
                if consensus_prediction is not None:
                    predictions['consensus_meta_ensemble'] = primary_prediction
                predictions['model_dispersion'] = prediction_dispersion
                predictions['model_agreement'] = float(max(0.0, 1.0 - min(prediction_dispersion / 0.1, 1.0)))
                predictions['models_used'] = len(all_preds)
                predictions['confidence_interval'] = {
                    'lower': float(primary_prediction - interval_width),
                    'upper': float(primary_prediction + interval_width)
                }
                predictions['timestamp'] = datetime.now().isoformat()
                predictions['symbol'] = symbol

                logger.info(f"✅ Generated predictions for {symbol}: {primary_prediction:.4f}")
                
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        return predictions
    
    def check_alert_conditions(self, predictions: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        try:
            primary_pred = predictions.get('primary', {}).get('prediction', 0)
            
            # High confidence prediction alert
            if abs(primary_pred) > self.alert_thresholds['high_confidence']:
                direction = "BUY" if primary_pred > 0 else "SELL"
                alerts.append({
                    'type': 'high_confidence_signal',
                    'symbol': symbol,
                    'direction': direction,
                    'prediction': primary_pred,
                    'confidence': predictions.get('primary', {}).get('confidence', 'medium'),
                    'timestamp': datetime.now().isoformat(),
                    'message': f"{direction} signal for {symbol}: {primary_pred:.4f}"
                })
            
            # Risk limit alert
            if abs(primary_pred) > self.alert_thresholds['risk_limit']:
                alerts.append({
                    'type': 'risk_limit_exceeded',
                    'symbol': symbol,
                    'prediction': primary_pred,
                    'risk_threshold': self.alert_thresholds['risk_limit'],
                    'timestamp': datetime.now().isoformat(),
                    'message': f"HIGH RISK: {symbol} prediction {primary_pred:.4f} exceeds risk limit"
                })
            
            # Model agreement alert (if multiple models available)
            model_preds = [v for k, v in predictions.items() 
                          if k not in ['timestamp', 'symbol', 'primary'] and isinstance(v, (int, float))]
            if len(model_preds) > 1:
                pred_std = np.std(model_preds)
                if pred_std < 0.001:  # Very high agreement
                    alerts.append({
                        'type': 'model_consensus',
                        'symbol': symbol,
                        'agreement': 'high',
                        'std_deviation': pred_std,
                        'timestamp': datetime.now().isoformat(),
                        'message': f"High model consensus for {symbol}"
                    })
                        
        except Exception as e:
            logger.error(f"❌ Alert check failed for {symbol}: {e}")
        
        return alerts
    
    def calculate_portfolio_impact(self, all_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall portfolio impact from predictions"""
        try:
            total_exposure = 0
            position_recommendations = {}
            
            # Load risk management settings from Day 11
            risk_summary_path = self.config.PROCESSED_DATA_PATH / "day11_risk_summary.csv"
            if risk_summary_path.exists():
                risk_df = pd.read_csv(risk_summary_path)
                if 'Sharpe_Ratio' in risk_df.columns:
                    best_model_row = risk_df.loc[risk_df['Sharpe_Ratio'].idxmax()]
                    best_model_name = best_model_row['Model']
                    best_model_risk = risk_df[risk_df['Model'] == best_model_name]
                else:
                    best_model_risk = risk_df
                if not best_model_risk.empty:
                    optimal_position = best_model_risk['Kelly_Position_Size'].iloc[0]
                else:
                    optimal_position = 0.1  # 10% default
            else:
                optimal_position = 0.1
            
            for symbol, preds in all_predictions.items():
                primary_pred = preds.get('primary', {}).get('prediction', 0)
                
                # Calculate position size based on prediction strength
                if abs(primary_pred) > 0.001:  # Only if meaningful prediction
                    position_size = min(abs(primary_pred) * optimal_position * 10, optimal_position)  # Scale up
                    direction = 1 if primary_pred > 0 else -1
                else:
                    position_size = 0
                    direction = 0
                
                position_recommendations[symbol] = {
                    'direction': 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'NEUTRAL',
                    'position_size': position_size,
                    'prediction': primary_pred,
                    'kelly_optimal': optimal_position
                }
                
                total_exposure += abs(position_size)
            
            portfolio_impact = {
                'total_exposure': total_exposure,
                'max_exposure_limit': 1.0,  # 100% max
                'utilization': total_exposure,
                'recommendations': position_recommendations,
                'risk_level': 'HIGH' if total_exposure > 0.8 else 'MEDIUM' if total_exposure > 0.5 else 'LOW',
                'timestamp': datetime.now().isoformat()
            }
            
            return portfolio_impact
            
        except Exception as e:
            logger.error(f"❌ Portfolio impact calculation failed: {e}")
            return {}

    def evaluate_drift_monitoring(self, all_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Update drift tracking and emit retraining triggers when thresholds are breached."""
        if not all_predictions:
            return {'reports': {}, 'triggers': []}

        try:
            self.drift_detector.update_performance_metrics(all_predictions)
            return self.drift_detector.evaluate_symbols(list(all_predictions.keys()))
        except Exception as e:
            logger.error(f"❌ Drift monitoring failed: {e}")
            return {'reports': {}, 'triggers': [], 'error': str(e)}
    
    async def run_realtime_cycle(self) -> Dict[str, Any]:
        """Run a complete real-time prediction cycle"""
        logger.info("🚀 Starting real-time prediction cycle...")
        
        cycle_start = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'alerts': [],
            'portfolio_impact': {},
            'performance_metrics': {},
            'drift_monitoring': {},
            'retraining_triggers': [],
            'errors': []
        }
        
        try:
            # 1. Get target stocks
            symbols = self.get_target_stocks()
            logger.info(f"📊 Monitoring {len(symbols)} symbols: {symbols}")
            
            # 2. Fetch real-time data
            market_data = await self.fetch_realtime_data(symbols, period="1y")
            
            # 3. Generate predictions for each stock
            all_predictions = {}
            all_alerts = []
            
            for symbol in symbols:
                if symbol in market_data:
                    # Engineer features
                    df_features = self.engineer_realtime_features(market_data[symbol], symbol)
                    
                    if not df_features.empty:
                        # Prepare features for prediction
                        feature_data = self.prepare_prediction_features(df_features)
                        
                        if feature_data is not None:
                            # Generate predictions
                            predictions = self.generate_predictions(feature_data, symbol)
                            if predictions:
                                all_predictions[symbol] = predictions
                                
                                # Check for alerts
                                alerts = self.check_alert_conditions(predictions, symbol)
                                all_alerts.extend(alerts)
                        else:
                            results['errors'].append(f"Feature preparation failed for {symbol}")
                    else:
                        results['errors'].append(f"Feature engineering failed for {symbol}")
                else:
                    results['errors'].append(f"No market data for {symbol}")
            
            # 4. Calculate portfolio impact
            if all_predictions:
                portfolio_impact = self.calculate_portfolio_impact(all_predictions)
                results['portfolio_impact'] = portfolio_impact
            
            # 5. Compile results
            results['predictions'] = all_predictions
            results['alerts'] = all_alerts
            drift_report = self.evaluate_drift_monitoring(all_predictions)
            results['drift_monitoring'] = drift_report.get('reports', {})
            results['retraining_triggers'] = drift_report.get('triggers', [])
            
            # 6. Performance metrics
            cycle_time = time.time() - cycle_start
            results['performance_metrics'] = {
                'cycle_time_seconds': cycle_time,
                'symbols_processed': len(all_predictions),
                'primary_signals_generated': len(all_predictions),
                'model_outputs_generated': sum(int(pred.get('models_used', 1)) for pred in all_predictions.values()),
                'alerts_triggered': len(all_alerts),
                'retraining_triggers': len(results['retraining_triggers']),
                'success_rate': len(all_predictions) / len(symbols) if symbols else 0
            }
            
            logger.info(f"✅ Real-time cycle completed in {cycle_time:.2f}s")
            logger.info(f"📈 Generated predictions for {len(all_predictions)} stocks")
            logger.info(f"🚨 Triggered {len(all_alerts)} alerts")
            
        except Exception as e:
            logger.error(f"❌ Real-time cycle failed: {e}")
            results['errors'].append(f"Cycle failure: {str(e)}")
        
        return results
    
    def save_realtime_results(self, results: Dict[str, Any]) -> str:
        """Save real-time results for monitoring"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.config.PROCESSED_DATA_PATH / f"realtime_results_{timestamp}.json"
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Saved results: {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return ""
    
    async def continuous_monitoring(self, interval_minutes: int = 15, max_cycles: int = 96):
        """Run continuous monitoring for a trading day"""
        logger.info(f"🔄 Starting continuous monitoring (every {interval_minutes} min)")
        
        cycle_count = 0
        while cycle_count < max_cycles:
            try:
                # Run prediction cycle
                results = await self.run_realtime_cycle()
                
                # Save results
                self.save_realtime_results(results)
                
                # Display summary
                predictions_count = len(results.get('predictions', {}))
                alerts_count = len(results.get('alerts', []))
                cycle_time = results.get('performance_metrics', {}).get('cycle_time_seconds', 0)
                
                print(f"🔄 Cycle {cycle_count + 1}: {predictions_count} predictions, "
                      f"{alerts_count} alerts, {cycle_time:.1f}s")
                
                # Check for high-priority alerts
                for alert in results.get('alerts', []):
                    if alert.get('type') == 'risk_limit_exceeded':
                        print(f"🚨 HIGH RISK ALERT: {alert['message']}")
                    elif alert.get('type') == 'high_confidence_signal':
                        print(f"📈 SIGNAL: {alert['message']}")
                
                cycle_count += 1
                
                # Wait for next cycle
                if cycle_count < max_cycles:
                    await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
        
        logger.info(f"✅ Monitoring completed after {cycle_count} cycles")

class ModelDriftDetector:
    """Detect model performance drift in real-time"""
    
    def __init__(self, config: Config):
        self.config = config
        self.historical_performance = {}
        self.drift_threshold = 0.1  # 10% performance degradation
        
    def update_performance_metrics(self, predictions: Dict[str, Any], 
                                 actual_returns: Dict[str, float] = None):
        """Update performance tracking"""
        timestamp = datetime.now()
        
        # In production, this would compare predictions with actual returns
        # For now, we'll track prediction variance as a proxy
        for symbol, pred_data in predictions.items():
            if symbol not in self.historical_performance:
                self.historical_performance[symbol] = []
            
            primary_pred = pred_data.get('primary', {}).get('prediction', 0)
            self.historical_performance[symbol].append({
                'timestamp': timestamp,
                'prediction': primary_pred,
                'actual': actual_returns.get(symbol) if actual_returns else None
            })
            
            # Keep only recent history (last 100 predictions)
            if len(self.historical_performance[symbol]) > 100:
                self.historical_performance[symbol] = self.historical_performance[symbol][-100:]
    
    def detect_drift(self, symbol: str) -> Dict[str, Any]:
        """Detect if model performance is drifting"""
        if symbol not in self.historical_performance:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        history = self.historical_performance[symbol]
        if len(history) < 20:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Calculate recent vs historical variance
        recent_preds = [h['prediction'] for h in history[-10:]]
        historical_preds = [h['prediction'] for h in history[-50:-10]]
        
        if len(historical_preds) == 0:
            return {'drift_detected': False, 'reason': 'insufficient_historical_data'}
        
        recent_variance = np.var(recent_preds)
        historical_variance = np.var(historical_preds)
        
        # Detect significant variance change
        if historical_variance > 0:
            variance_ratio = abs(recent_variance - historical_variance) / historical_variance
            drift_detected = variance_ratio > self.drift_threshold
            
            return {
                'drift_detected': drift_detected,
                'variance_ratio': variance_ratio,
                'threshold': self.drift_threshold,
                'recent_variance': recent_variance,
                'historical_variance': historical_variance,
                'recommendation': 'retrain_model' if drift_detected else 'continue_monitoring'
            }
        
        return {'drift_detected': False, 'reason': 'zero_historical_variance'}

    def save_retraining_trigger(self, symbol: str, drift_report: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a retraining trigger so monitoring and automation can pick it up."""
        trigger = {
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'reason': 'model_drift_detected',
            'variance_ratio': drift_report.get('variance_ratio'),
            'threshold': drift_report.get('threshold'),
            'recommendation': drift_report.get('recommendation', 'retrain_model')
        }

        try:
            trigger_path = self.config.PROCESSED_DATA_PATH / "retraining_triggers.json"
            existing_triggers = []

            if trigger_path.exists():
                with open(trigger_path, 'r') as f:
                    existing_triggers = json.load(f)
                    if not isinstance(existing_triggers, list):
                        existing_triggers = []

            existing_triggers.append(trigger)
            existing_triggers = existing_triggers[-100:]

            with open(trigger_path, 'w') as f:
                json.dump(existing_triggers, f, indent=2)
        except Exception as e:
            trigger['persistence_error'] = str(e)

        return trigger

    def evaluate_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Evaluate drift across multiple symbols and create retraining triggers where needed."""
        reports = {}
        triggers = []

        for symbol in symbols:
            report = self.detect_drift(symbol)
            reports[symbol] = report

            if report.get('drift_detected'):
                triggers.append(self.save_retraining_trigger(symbol, report))

        return {'reports': reports, 'triggers': triggers}
