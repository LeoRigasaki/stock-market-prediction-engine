#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 4
Advanced Feature Engineering with Technical Indicators
"""

from typing import List
from src.config import Config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from loguru import logger
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_feature_visualizations(df: pd.DataFrame, selected_features: List[str], 
                                correlation_analysis: pd.DataFrame):
    """Create comprehensive feature analysis visualizations"""
    logger.info("Creating feature analysis visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Feature Engineering Analysis - Day 4', fontsize=16, fontweight='bold')
    
    # 1. Feature correlation heatmap (top 15 features)
    if not correlation_analysis.empty and len(selected_features) > 5:
        top_features = selected_features[:15]
        if 'return_5d' in df.columns:
            top_features.append('return_5d')
        
        available_features = [f for f in top_features if f in df.columns]
        if len(available_features) > 3:
            corr_matrix = df[available_features].corr()
            
            im = axes[0,0].imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[0,0].set_xticks(range(len(corr_matrix.columns)))
            axes[0,0].set_yticks(range(len(corr_matrix.index)))
            axes[0,0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            axes[0,0].set_yticklabels(corr_matrix.index)
            axes[0,0].set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=axes[0,0])
    
    # 2. Feature importance (correlation with target)
    if not correlation_analysis.empty:
        top_10_corr = correlation_analysis.head(10)
        axes[0,1].barh(range(len(top_10_corr)), top_10_corr['correlation'])
        axes[0,1].set_yticks(range(len(top_10_corr)))
        axes[0,1].set_yticklabels(top_10_corr['feature'])
        axes[0,1].set_xlabel('Absolute Correlation with Target')
        axes[0,1].set_title('Top 10 Features by Correlation')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Technical indicator example (RSI)
    if 'rsi' in df.columns:
        sample_stock = df[df['Ticker'] == df['Ticker'].iloc[0]].tail(100)
        axes[0,2].plot(sample_stock.index, sample_stock['rsi'], label='RSI', color='purple')
        axes[0,2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[0,2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[0,2].set_title('RSI Technical Indicator')
        axes[0,2].set_ylabel('RSI Value')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
    
    # 4. Price vs Moving Averages
    if all(col in df.columns for col in ['Close', 'sma_20', 'sma_50']):
        sample_stock = df[df['Ticker'] == df['Ticker'].iloc[0]].tail(200)
        axes[1,0].plot(sample_stock.index, sample_stock['Close'], label='Close Price', alpha=0.8)
        axes[1,0].plot(sample_stock.index, sample_stock['sma_20'], label='SMA 20', alpha=0.7)
        axes[1,0].plot(sample_stock.index, sample_stock['sma_50'], label='SMA 50', alpha=0.7)
        axes[1,0].set_title('Price vs Moving Averages')
        axes[1,0].set_ylabel('Price ($)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Volume analysis
    if all(col in df.columns for col in ['Volume', 'volume_ratio']):
        axes[1,1].scatter(df['Volume'], df['volume_ratio'], alpha=0.5, s=1)
        axes[1,1].set_xlabel('Volume')
        axes[1,1].set_ylabel('Volume Ratio')
        axes[1,1].set_title('Volume vs Volume Ratio')
        axes[1,1].set_xscale('log')
        axes[1,1].grid(True, alpha=0.3)
    
    # 6. Bollinger Bands position distribution
    if 'bb_position' in df.columns:
        axes[1,2].hist(df['bb_position'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1,2].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Lower Band')
        axes[1,2].axvline(x=1, color='r', linestyle='--', alpha=0.7, label='Upper Band')
        axes[1,2].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='Middle Band')
        axes[1,2].set_xlabel('Bollinger Band Position')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Distribution of BB Positions')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    # 7. Feature count by category
    feature_categories = {
        'Basic': len([f for f in df.columns if any(x in f for x in ['price_', 'volume_', 'body_'])]),
        'Technical': len([f for f in df.columns if any(x in f for x in ['sma_', 'ema_', 'rsi', 'macd', 'bb_'])]),
        'Time': len([f for f in df.columns if any(x in f for x in ['month', 'day_', 'quarter'])]),
        'Lag': len([f for f in df.columns if 'lag_' in f or 'mean_' in f or 'std_' in f]),
        'Target': len([f for f in df.columns if f.startswith('target_') or f.startswith('return_')])
    }
    
    categories = list(feature_categories.keys())
    counts = list(feature_categories.values())
    
    axes[2,0].bar(categories, counts, color=['skyblue', 'lightgreen', 'orange', 'pink', 'yellow'])
    axes[2,0].set_title('Features by Category')
    axes[2,0].set_ylabel('Number of Features')
    axes[2,0].tick_params(axis='x', rotation=45)
    axes[2,0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[2,0].text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    # 8. Target variable distribution
    if 'return_5d' in df.columns:
        returns = df['return_5d'].dropna()
        axes[2,1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[2,1].axvline(x=returns.mean(), color='r', linestyle='--', 
                         label=f'Mean: {returns.mean():.2f}%')
        axes[2,1].set_xlabel('5-Day Return (%)')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Distribution of 5-Day Returns')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
    
    # 9. Feature completeness analysis
    feature_completeness = []
    for col in selected_features[:20]:  # Top 20 features
        if col in df.columns:
            completeness = (1 - df[col].isnull().sum() / len(df)) * 100
            feature_completeness.append({'feature': col, 'completeness': completeness})
    
    if feature_completeness:
        comp_df = pd.DataFrame(feature_completeness).sort_values('completeness')
        axes[2,2].barh(range(len(comp_df)), comp_df['completeness'])
        axes[2,2].set_yticks(range(len(comp_df)))
        axes[2,2].set_yticklabels(comp_df['feature'])
        axes[2,2].set_xlabel('Data Completeness (%)')
        axes[2,2].set_title('Feature Data Completeness')
        axes[2,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Config.PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "day4_feature_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Feature analysis visualizations saved to {plots_dir / 'day4_feature_analysis.png'}")

def engineer_features_for_target_stocks():
    """Main feature engineering workflow"""
    print("\n🔧 Starting Advanced Feature Engineering...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load cleaned data and target stocks
    print("\n📊 Loading cleaned data from Day 3...")
    df = engineer.load_cleaned_data()
    if df.empty:
        print("❌ Failed to load cleaned data. Please run Day 3 first.")
        return None
    
    target_stocks = engineer.load_target_stocks()
    print(f"✅ Loaded data: {len(df):,} records from {df['Ticker'].nunique()} stocks")
    print(f"🎯 Target stocks: {', '.join(target_stocks)}")
    
    # Process features for target stocks only
    print(f"\n⚙️ Processing features for {len(target_stocks)} target stocks...")
    processed_stocks = []
    
    for i, ticker in enumerate(target_stocks, 1):
        print(f"   Processing {ticker} ({i}/{len(target_stocks)})...")
        stock_data = df[df['Ticker'] == ticker].copy()
        
        if len(stock_data) < 100:  # Skip stocks with insufficient data
            print(f"   ⚠️ Skipping {ticker}: insufficient data ({len(stock_data)} records)")
            continue
        
        try:
            processed_stock = engineer.process_single_stock(stock_data, ticker)
            if not processed_stock.empty:
                processed_stocks.append(processed_stock)
                print(f"   ✅ {ticker}: {len(processed_stock)} records, {processed_stock.shape[1]} features")
            else:
                print(f"   ⚠️ {ticker}: No valid records after feature engineering")
        except Exception as e:
            print(f"   ❌ {ticker}: Error in processing - {str(e)}")
            continue
    
    if not processed_stocks:
        print("❌ No stocks were successfully processed")
        return None
    
    # Combine all processed stocks
    print(f"\n📊 Combining features from {len(processed_stocks)} stocks...")
    combined_df = pd.concat(processed_stocks, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df):,} records, {combined_df.shape[1]} features")
    
    # Analyze feature importance
    print("\n🔍 Analyzing feature correlations and importance...")
    correlation_analysis = engineer.analyze_feature_importance(combined_df)
    
    if not correlation_analysis.empty:
        print("🏆 Top 10 Features by Correlation:")
        for idx, row in correlation_analysis.head(10).iterrows():
            print(f"   {row['feature']}: {row['correlation']:.4f}")
    
    # Select best features
    print("\n🎯 Selecting optimal features for modeling...")
    selected_features = engineer.select_features(combined_df, correlation_threshold=0.01)
    print(f"✅ Selected {len(selected_features)} features for modeling")
    
    # Create visualizations
    print("\n📊 Creating comprehensive feature analysis visualizations...")
    create_feature_visualizations(combined_df, selected_features, correlation_analysis)
    
    # Save all results
    print("\n💾 Saving engineered features and analysis...")
    saved_files = engineer.save_engineered_features(combined_df, selected_features, correlation_analysis)
    
    print("✅ Saved files:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return {
        'combined_data': combined_df,
        'selected_features': selected_features,
        'correlation_analysis': correlation_analysis,
        'saved_files': saved_files
    }

def main():
    """Main execution function for Day 4"""
    
    print("🚀 Stock Market Prediction Engine - Day 4")
    print("Advanced Feature Engineering with Technical Indicators")
    print("=" * 60)
    
    # Check dependencies from previous days
    config = Config()
    cleaned_data_path = config.PROCESSED_DATA_PATH / "cleaned_world_stocks.csv"
    
    if not cleaned_data_path.exists():
        print("\n❌ Cleaned data not found!")
        print("Please run Day 3 first to generate cleaned datasets")
        print("Command: python main.py  # (from Day 3)")
        return
    
    # Run feature engineering
    results = engineer_features_for_target_stocks()
    
    if results is None:
        print("\n❌ Feature engineering failed!")
        return
    
    # Display final results
    combined_df = results['combined_data']
    selected_features = results['selected_features']
    
    print("\n🎯 Day 4 Completed Successfully!")
    print("=" * 60)
    print("✅ Advanced feature engineering completed")
    print("✅ Technical indicators calculated")
    print("✅ Feature correlation analysis performed")
    print("✅ Optimal features selected")
    print("✅ Comprehensive visualizations created")
    print("✅ Feature datasets saved")
    
    print(f"\n📊 Final Feature Engineering Summary:")
    print(f"   Total records processed: {len(combined_df):,}")
    print(f"   Stocks successfully processed: {combined_df['Ticker'].nunique()}")
    print(f"   Total features created: {combined_df.shape[1]}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Date range: {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}")
    
    print(f"\n🔧 Feature Categories Created:")
    feature_categories = {
        'Basic Features': len([f for f in combined_df.columns if any(x in f for x in ['price_', 'volume_', 'body_'])]),
        'Technical Indicators': len([f for f in combined_df.columns if any(x in f for x in ['sma_', 'ema_', 'rsi', 'macd', 'bb_'])]),
        'Time Features': len([f for f in combined_df.columns if any(x in f for x in ['month', 'day_', 'quarter'])]),
        'Lag Features': len([f for f in combined_df.columns if 'lag_' in f or 'mean_' in f or 'std_' in f]),
        'Target Variables': len([f for f in combined_df.columns if f.startswith('target_') or f.startswith('return_')])
    }
    
    for category, count in feature_categories.items():
        print(f"   {category}: {count}")
    
    print(f"\n🏆 Top 10 Selected Features:")
    for i, feature in enumerate(selected_features[:10], 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n📋 Ready for Day 5:")
    print("1. Exploratory data analysis with engineered features")
    print("2. Market pattern recognition and regime detection")
    print("3. Statistical analysis and hypothesis testing")
    print("4. Interactive visualization dashboard creation")

if __name__ == "__main__":
    main()