import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import Config

class FeatureEngineer:
    """Advanced feature engineering for stock market prediction"""
    
    def __init__(self):
        self.config = Config()
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load the cleaned dataset from Day 3"""
        cleaned_path = self.config.PROCESSED_DATA_PATH / "cleaned_world_stocks.csv"
        
        if not cleaned_path.exists():
            logger.error(f"Cleaned dataset not found at {cleaned_path}")
            logger.error("Please run Day 3 first to generate cleaned data")
            return pd.DataFrame()
        
        df = pd.read_csv(cleaned_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"Loaded cleaned dataset: {len(df)} records, {df['Ticker'].nunique()} stocks")
        return df
    
    def load_target_stocks(self) -> List[str]:
        """Load target stocks from Day 3"""
        target_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
        
        if not target_path.exists():
            logger.error("Target stocks file not found. Using fallback stocks.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        with open(target_path, 'r') as f:
            stocks = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Loaded {len(stocks)} target stocks: {', '.join(stocks)}")
        return stocks
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price and volume features"""
        logger.info("Creating basic price and volume features...")
        
        # Price features
        df['price_range'] = df['High'] - df['Low']
        df['price_change'] = df['Close'] - df['Open']
        df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_pct'] = df['gap'] / df['Close'].shift(1) * 100
        
        # Volume features
        df['volume_change'] = df['Volume'] - df['Volume'].shift(1)
        df['volume_change_pct'] = df['Volume'].pct_change() * 100
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = (df['price_volume'].rolling(window=20).sum() / 
                     df['Volume'].rolling(window=20).sum())
        
        # Body and shadow features (candlestick analysis)
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['body_to_range_ratio'] = df['body_size'] / (df['price_range'] + 1e-8)
        
        logger.info("Basic features created: price_range, price_change, volume features, candlestick features")
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators using TA library"""
        logger.info("Creating technical indicators...")
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # Price relative to moving averages
        df['close_to_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20'] * 100
        df['close_to_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50'] * 100
        df['sma20_to_sma50'] = (df['sma_20'] - df['sma_50']) / df['sma_50'] * 100
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Average True Range (Volatility)
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close'] * 100
        
        # Volume indicators
        df['volume_sma20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma20']
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Price momentum indicators
        df['momentum_1d'] = df['Close'].pct_change(1) * 100
        df['momentum_5d'] = df['Close'].pct_change(5) * 100
        df['momentum_10d'] = df['Close'].pct_change(10) * 100
        df['momentum_20d'] = df['Close'].pct_change(20) * 100
        
        logger.info("Technical indicators created: MA, MACD, RSI, Bollinger Bands, Stochastic, Williams %R, CCI, ATR, Volume indicators")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features...")
        
        # Basic time features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['quarter'] = df['Date'].dt.quarter
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market regime features (simple volatility-based)
        df['volatility_20d'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean() * 100
        df['high_volatility'] = (df['volatility_20d'] > df['volatility_20d'].rolling(100).quantile(0.8)).astype(int)
        df['low_volatility'] = (df['volatility_20d'] < df['volatility_20d'].rolling(100).quantile(0.2)).astype(int)
        
        logger.info("Time features created: date components, cyclical encoding, volatility regimes")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series prediction"""
        logger.info("Creating lag features...")
        
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'close_max_{window}'] = df['Close'].rolling(window).max()
            df[f'close_min_{window}'] = df['Close'].rolling(window).min()
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['Volume'].rolling(window).std()
        
        # Price position in recent range
        for window in [10, 20, 50]:
            rolling_max = df['Close'].rolling(window).max()
            rolling_min = df['Close'].rolling(window).min()
            df[f'price_position_{window}'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        logger.info("Lag features created: price/volume lags, rolling statistics, price positions")
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        logger.info("Creating target variables...")
        
        # Future price movements (classification targets)
        df['target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['target_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        df['target_10d'] = (df['Close'].shift(-10) > df['Close']).astype(int)
        
        # Future returns (regression targets)
        df['return_1d'] = df['Close'].pct_change(-1) * 100
        df['return_5d'] = (df['Close'].shift(-5) / df['Close'] - 1) * 100
        df['return_10d'] = (df['Close'].shift(-10) / df['Close'] - 1) * 100
        
        # Risk-adjusted targets
        future_volatility_5d = df['Close'].pct_change().shift(-5).rolling(5).std() * np.sqrt(5) * 100
        df['sharpe_5d'] = df['return_5d'] / (future_volatility_5d + 1e-8)
        
        logger.info("Target variables created: classification and regression targets for 1d, 5d, 10d horizons")
        return df
    
    def process_single_stock(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process features for a single stock"""
        logger.info(f"Processing features for {ticker}...")
        
        stock_data = stock_data.copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        # Apply all feature engineering steps
        stock_data = self.create_basic_features(stock_data)
        stock_data = self.create_technical_indicators(stock_data)
        stock_data = self.create_time_features(stock_data)
        stock_data = self.create_lag_features(stock_data)
        stock_data = self.create_target_variables(stock_data)
        
        # Add stock-specific features
        stock_data['ticker'] = ticker
        
        # Remove rows with insufficient data for features
        stock_data = stock_data.dropna()
        
        logger.info(f"Completed {ticker}: {len(stock_data)} records with {stock_data.shape[1]} features")
        return stock_data
    
    def analyze_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations and importance"""
        logger.info("Analyzing feature correlations and importance...")
        
        # Get numerical features (exclude dates and categorical)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        excluded_cols = ['Date', 'year', 'month', 'day_of_week', 'day_of_month', 'quarter']
        feature_cols = [col for col in numeric_features if col not in excluded_cols]
        
        if 'return_5d' in df.columns:
            target_col = 'return_5d'
        else:
            logger.warning("Target variable not found, using Close price")
            target_col = 'Close'
        
        # Calculate correlations with target
        correlations = []
        for feature in feature_cols:
            if feature != target_col and not feature.startswith('target_') and not feature.startswith('return_'):
                try:
                    corr = df[feature].corr(df[target_col])
                    if not np.isnan(corr):
                        correlations.append({
                            'feature': feature,
                            'correlation': abs(corr),
                            'correlation_raw': corr
                        })
                except:
                    continue
        
        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            corr_df = corr_df.sort_values('correlation', ascending=False)
            logger.info(f"Feature correlation analysis completed for {len(corr_df)} features")
        else:
            logger.warning("No valid correlations calculated")
        
        return corr_df
    
    def select_features(self, df: pd.DataFrame, correlation_threshold: float = 0.01) -> List[str]:
        """Select features based on correlation and importance"""
        logger.info(f"Selecting features with correlation threshold {correlation_threshold}...")
        
        corr_analysis = self.analyze_feature_importance(df)
        
        if corr_analysis.empty:
            logger.warning("No correlation analysis available, using default features")
            default_features = ['Close', 'Volume', 'rsi', 'macd', 'bb_position', 
                              'momentum_5d', 'sma_20', 'atr_ratio', 'volume_ratio']
            return [f for f in default_features if f in df.columns]
        
        # Select features above threshold
        selected_features = corr_analysis[
            corr_analysis['correlation'] >= correlation_threshold
        ]['feature'].tolist()
        
        # Always include basic OHLCV if available
        essential_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        for feature in essential_features:
            if feature in df.columns and feature not in selected_features:
                selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features above correlation threshold")
        logger.info(f"Top 10 features by correlation: {selected_features[:10]}")
        
        return selected_features
    
    def save_engineered_features(self, df: pd.DataFrame, selected_features: List[str], 
                               correlation_analysis: pd.DataFrame):
        """Save all engineered features and analysis"""
        logger.info("Saving engineered features and analysis...")
        
        # Save complete feature dataset
        features_path = self.config.FEATURES_DATA_PATH / "engineered_features.csv"
        df.to_csv(features_path, index=False)
        logger.info(f"Complete feature dataset saved: {features_path}")
        
        # Save selected features dataset
        feature_columns = ['Date', 'Ticker'] + selected_features
        if 'return_5d' in df.columns:
            feature_columns.extend(['target_1d', 'target_5d', 'return_1d', 'return_5d'])
        
        available_columns = [col for col in feature_columns if col in df.columns]
        selected_df = df[available_columns]
        
        selected_path = self.config.FEATURES_DATA_PATH / "selected_features.csv"
        selected_df.to_csv(selected_path, index=False)
        logger.info(f"Selected features dataset saved: {selected_path}")
        
        # Save correlation analysis
        corr_path = self.config.FEATURES_DATA_PATH / "feature_correlations.csv"
        correlation_analysis.to_csv(corr_path, index=False)
        logger.info(f"Feature correlation analysis saved: {corr_path}")
        
        # Save feature list
        features_list_path = self.config.FEATURES_DATA_PATH / "selected_features_list.txt"
        with open(features_list_path, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        logger.info(f"Selected features list saved: {features_list_path}")
        
        # Create feature engineering summary
        summary = {
            'processing_date': pd.Timestamp.now().isoformat(),
            'total_features_created': df.shape[1],
            'selected_features_count': len(selected_features),
            'total_records': len(df),
            'stocks_processed': df['Ticker'].nunique() if 'Ticker' in df.columns else 'Unknown',
            'feature_categories': {
                'basic_features': len([f for f in df.columns if any(x in f for x in ['price_', 'volume_', 'body_'])]),
                'technical_indicators': len([f for f in df.columns if any(x in f for x in ['sma_', 'ema_', 'rsi', 'macd', 'bb_'])]),
                'time_features': len([f for f in df.columns if any(x in f for x in ['month', 'day_', 'quarter'])]),
                'lag_features': len([f for f in df.columns if 'lag_' in f or 'mean_' in f or 'std_' in f]),
                'target_variables': len([f for f in df.columns if f.startswith('target_') or f.startswith('return_')])
            },
            'top_10_features': selected_features[:10] if selected_features else [],
            'correlation_threshold_used': 0.01
        }
        
        import json
        summary_path = self.config.FEATURES_DATA_PATH / "feature_engineering_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Feature engineering summary saved: {summary_path}")
        
        return {
            'engineered_features': features_path,
            'selected_features': selected_path,
            'correlations': corr_path,
            'features_list': features_list_path,
            'summary': summary_path
        }