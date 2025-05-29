import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the Stock Market Prediction Engine"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_PATH / "raw"
    PROCESSED_DATA_PATH = DATA_PATH / "processed"
    FEATURES_DATA_PATH = DATA_PATH / "features"
    LOGS_PATH = PROJECT_ROOT / "logs"
    
    # Kaggle settings
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME', '')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY', '')
    
    # Data sources
    DATASETS = {
        'world_stocks': 'nelgiriyewithana/world-stock-prices-daily-updating',
        'nasdaq_stocks': 'svaningelgem/nasdaq-daily-stock-prices',
        'sp500_stocks': 'paultimothymooney/stock-market-data'
    }
    
    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    # Technical indicators periods
    TECHNICAL_INDICATORS = {
        'sma_short': 5,
        'sma_long': 20,
        'ema_short': 12,
        'ema_long': 26,
        'rsi_period': 14,
        'bb_period': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = LOGS_PATH / 'stock_engine.log'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for path in [cls.RAW_DATA_PATH, cls.PROCESSED_DATA_PATH, 
                    cls.FEATURES_DATA_PATH, cls.LOGS_PATH]:
            path.mkdir(parents=True, exist_ok=True)