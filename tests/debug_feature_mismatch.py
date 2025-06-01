#!/usr/bin/env python3
"""
Debug Feature Count Mismatch
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.config import Config

def analyze_feature_mismatch():
    """Analyze exact feature requirements"""
    
    print("🔍 FEATURE MISMATCH ANALYSIS")
    print("=" * 50)
    
    config = Config()
    
    # 1. Load the actual training data to see what features were used
    print("1. Loading training data...")
    training_path = config.FEATURES_DATA_PATH / "selected_features.csv"
    
    if training_path.exists():
        df = pd.read_csv(training_path)
        print(f"✅ Training data shape: {df.shape}")
        
        # Exclude metadata and target columns
        exclude_cols = ['Date', 'Ticker', 'target_1d', 'target_5d', 'return_1d', 'return_5d']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"✅ Total columns: {len(df.columns)}")
        print(f"✅ Feature columns: {len(feature_cols)}")
        print(f"✅ Excluded columns: {len(exclude_cols)}")
        
        print(f"\n📋 ACTUAL TRAINING FEATURES ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        # Save the correct feature list
        correct_features_path = config.FEATURES_DATA_PATH / "correct_features_list.txt"
        with open(correct_features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        print(f"\n💾 Saved correct features to: {correct_features_path}")
        
        # Check what's in the current feature list file
        current_features_path = config.FEATURES_DATA_PATH / "selected_features_list.txt"
        if current_features_path.exists():
            with open(current_features_path, 'r') as f:
                current_features = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"\n📋 CURRENT FEATURE LIST ({len(current_features)}):")
            print(f"From file: {current_features_path}")
            
            # Find differences
            training_set = set(feature_cols)
            current_set = set(current_features)
            
            missing_in_current = training_set - current_set
            extra_in_current = current_set - training_set
            
            if missing_in_current:
                print(f"\n❌ MISSING in current list ({len(missing_in_current)}):")
                for feature in missing_in_current:
                    print(f"   • {feature}")
            
            if extra_in_current:
                print(f"\n⚠️ EXTRA in current list ({len(extra_in_current)}):")
                for feature in extra_in_current:
                    print(f"   • {feature}")
            
            if len(missing_in_current) == 0 and len(extra_in_current) == 0:
                print("✅ Feature lists match perfectly!")
            else:
                print(f"\n🔧 RECOMMENDED ACTION:")
                print(f"Replace {current_features_path} with the correct feature list")
        
        return feature_cols
    else:
        print("❌ Training data not found!")
        return []

if __name__ == "__main__":
    correct_features = analyze_feature_mismatch()
    if correct_features:
        print(f"\n✅ Analysis complete. Found {len(correct_features)} correct features.")
    else:
        print("❌ Analysis failed!")