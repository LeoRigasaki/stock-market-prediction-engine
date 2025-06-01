#!/usr/bin/env python3
"""
Find which feature to exclude to match model expectations
"""

import pandas as pd
import joblib
import numpy as np
from src.config import Config

def find_model_feature_requirements():
    """Test each model to find which features they expect"""
    
    print("🔍 FINDING MODEL FEATURE REQUIREMENTS")
    print("=" * 50)
    
    config = Config()
    
    # Load training data
    training_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    df = pd.read_csv(training_path)
    
    # Get all potential features (74 total)
    exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d']
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"✅ Found {len(all_features)} potential features")
    
    # Load models and test feature requirements
    models_to_test = {
        'XGBoost': config.PROJECT_ROOT / "models" / "advanced" / "regression_xgboost_optimized.joblib",
        'RandomForest': config.PROJECT_ROOT / "models" / "regression_random_forest.joblib",
        'VotingRegressor': config.PROJECT_ROOT / "models" / "ensemble" / "voting_regressor_ensemble.joblib"
    }
    
    # Create test data with all 74 features
    test_sample = df[all_features].iloc[-1:].fillna(0).values
    print(f"✅ Test sample shape: {test_sample.shape}")
    
    model_requirements = {}
    
    for model_name, model_path in models_to_test.items():
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print(f"\n🧪 Testing {model_name}...")
                
                # Test with all 74 features
                try:
                    _ = model.predict(test_sample)
                    model_requirements[model_name] = 74
                    print(f"   ✅ Accepts 74 features")
                except Exception as e:
                    print(f"   ❌ Rejects 74 features: {str(e)}")
                    
                    # Test with 73 features (try removing different features)
                    features_to_try_removing = [
                        'sharpe_5d',    # Most likely candidate (target-related)
                        'Stock_Splits', # Might be added later
                        'obv',          # Last in the list
                        'stoch_d'       # Second to last
                    ]
                    
                    for feature_to_remove in features_to_try_removing:
                        if feature_to_remove in all_features:
                            test_features = [f for f in all_features if f != feature_to_remove]
                            test_data_73 = df[test_features].iloc[-1:].fillna(0).values
                            
                            try:
                                _ = model.predict(test_data_73)
                                model_requirements[model_name] = 73
                                print(f"   ✅ Works with 73 features (removed: {feature_to_remove})")
                                
                                # Save the working feature list
                                working_features_path = config.FEATURES_DATA_PATH / f"working_features_{model_name.lower()}.txt"
                                with open(working_features_path, 'w') as f:
                                    for feature in test_features:
                                        f.write(f"{feature}\n")
                                print(f"   💾 Saved working features to: {working_features_path}")
                                break
                            except Exception as e2:
                                print(f"   ❌ Still fails without {feature_to_remove}: {str(e2)}")
                                continue
                    else:
                        print(f"   ❌ Could not find working 73-feature combination")
                        model_requirements[model_name] = "unknown"
                        
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        else:
            print(f"❌ {model_name} not found at {model_path}")
    
    # Summary
    print(f"\n📊 MODEL REQUIREMENTS SUMMARY:")
    print("=" * 40)
    for model_name, requirement in model_requirements.items():
        print(f"{model_name}: {requirement} features")
    
    # Determine the correct feature count
    feature_counts = [req for req in model_requirements.values() if isinstance(req, int)]
    if feature_counts:
        most_common = max(set(feature_counts), key=feature_counts.count)
        print(f"\n🎯 RECOMMENDATION: Use {most_common} features")
        
        if most_common == 73:
            print("💡 Likely need to remove 'sharpe_5d' (target-related feature)")
            
            # Create the correct 73-feature list
            correct_features = [f for f in all_features if f != 'sharpe_5d']
            correct_path = config.FEATURES_DATA_PATH / "model_ready_features.txt"
            with open(correct_path, 'w') as f:
                for feature in correct_features:
                    f.write(f"{feature}\n")
            print(f"✅ Created model-ready feature list: {correct_path}")
            return correct_features
    
    return all_features

if __name__ == "__main__":
    features = find_model_feature_requirements()
    print(f"\n✅ Analysis complete. Recommended features: {len(features)}")