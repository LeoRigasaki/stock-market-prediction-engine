#!/usr/bin/env python3
"""
Debug Real-time Prediction Issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
from src.config import Config
from src.realtime_prediction import RealTimePredictionEngine
import asyncio

async def debug_realtime_predictions():
    """Debug why predictions are zero"""
    
    print("🔍 DEBUGGING REAL-TIME PREDICTIONS")
    print("=" * 50)
    
    config = Config()
    engine = RealTimePredictionEngine()
    
    # 1. Load models and features
    print("1. Loading models and feature columns...")
    if not engine.load_production_models():
        print("❌ Failed to load models")
        return
    
    print(f"✅ Loaded {len(engine.models)} models")
    print(f"✅ Feature columns: {len(engine.feature_columns)}")
    
    # 2. Load sample training data for comparison
    print("\n2. Loading training data sample...")
    training_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    if training_path.exists():
        training_df = pd.read_csv(training_path)
        print(f"✅ Training data: {training_df.shape}")
        print(f"   Columns: {list(training_df.columns)}")
        
        # Get a sample for testing
        sample_features = training_df.iloc[-1][engine.feature_columns].values.reshape(1, -1)
        print(f"✅ Sample features shape: {sample_features.shape}")
        
        # Test with training data
        print("\n3. Testing with training data...")
        if engine.best_model:
            train_pred = engine.best_model.predict(sample_features)[0]
            print(f"✅ Training data prediction: {train_pred:.6f}")
        
    # 3. Test real-time data
    print("\n4. Testing real-time data...")
    symbols = ['AAPL']
    market_data = await engine.fetch_realtime_data(symbols, period="2y")
    
    if 'AAPL' in market_data:
        print(f"✅ Fetched {len(market_data['AAPL'])} records for AAPL")
        
        # Engineer features
        df_features = engine.engineer_realtime_features(market_data['AAPL'], 'AAPL')
        print(f"✅ Engineered features: {df_features.shape}")
        print(f"   Columns: {list(df_features.columns)}")
        
        # Check feature alignment
        print("\n5. Feature alignment check...")
        missing_features = []
        extra_features = []
        
        for feature in engine.feature_columns:
            if feature not in df_features.columns:
                missing_features.append(feature)
        
        for feature in df_features.columns:
            if feature not in engine.feature_columns and feature not in ['Date', 'Ticker']:
                extra_features.append(feature)
        
        print(f"   Missing features: {len(missing_features)}")
        if missing_features[:5]:  # Show first 5
            print(f"   Examples: {missing_features[:5]}")
        
        print(f"   Extra features: {len(extra_features)}")
        if extra_features[:5]:  # Show first 5
            print(f"   Examples: {extra_features[:5]}")
        
        # Prepare features
        feature_data = engine.prepare_prediction_features(df_features)
        if feature_data is not None:
            print(f"✅ Prepared features: {feature_data.shape}")
            print(f"   Sample values: {feature_data[0][:5]}")
            print(f"   Min: {np.min(feature_data):.6f}, Max: {np.max(feature_data):.6f}")
            print(f"   NaN count: {np.isnan(feature_data).sum()}")
            print(f"   Inf count: {np.isinf(feature_data).sum()}")
            
            # Test prediction
            if engine.best_model:
                pred = engine.best_model.predict(feature_data)[0]
                print(f"✅ Real-time prediction: {pred:.6f}")
        else:
            print("❌ Feature preparation failed")

if __name__ == "__main__":
    asyncio.run(debug_realtime_predictions())