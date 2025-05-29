#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 3
Data Preprocessing and Cleaning
"""

from src.config import Config
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from loguru import logger
import sys
import pandas as pd
from datetime import datetime

def process_and_clean_data():
    """Process and clean the downloaded data"""
    print("\n🔧 Starting Data Processing and Cleaning...")
    
    processor = DataProcessor()
    
    # Load and clean the world stocks data
    print("\n📊 Loading and cleaning world stocks dataset...")
    cleaned_df = processor.load_and_clean_world_stocks()
    
    if cleaned_df.empty:
        print("❌ Failed to load or clean the dataset")
        return None
    
    print(f"✅ Dataset cleaned: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    
    # Analyze stock coverage
    print("\n🔍 Analyzing stock data coverage...")
    analysis_df = processor.analyze_stock_coverage(cleaned_df)
    
    print("\n📈 Stock Analysis Summary:")
    print(f"   Total stocks analyzed: {len(analysis_df)}")
    print(f"   Average records per stock: {analysis_df['total_records'].mean():.0f}")
    print(f"   Date range: {analysis_df['date_range_start'].min()} to {analysis_df['date_range_end'].max()}")
    
    # Show top stocks
    print("\n🏆 Top 5 Stocks by Data Volume:")
    top_5 = analysis_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"   {row['ticker']} ({row['brand_name']}): {row['total_records']:,} records")
    
    # Select target stocks for modeling
    print("\n🎯 Selecting target stocks for prediction modeling...")
    target_stocks = processor.select_target_stocks(analysis_df)
    
    if not target_stocks:
        print("⚠️ No stocks met the minimum criteria")
        return None
    
    print(f"✅ Selected {len(target_stocks)} target stocks for modeling")
    
    # Create comprehensive visualizations
    print("\n📊 Creating comprehensive visualizations...")
    processor.create_comprehensive_visualizations(cleaned_df, target_stocks)
    
    # Save all processed data
    print("\n💾 Saving processed data...")
    saved_files = processor.save_cleaned_data(cleaned_df, analysis_df, target_stocks)
    
    print("✅ Saved files:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return {
        'cleaned_data': cleaned_df,
        'analysis': analysis_df,
        'target_stocks': target_stocks,
        'saved_files': saved_files
    }

def main():
    """Main execution function for Day 3"""
    
    print("🚀 Stock Market Prediction Engine - Day 3")
    print("Data Preprocessing and Cleaning")
    print("=" * 50)
    
    # Initialize components
    loader = DataLoader()
    
    # Check if we have data from Day 2
    downloaded_datasets = loader.list_downloaded_datasets()
    
    if 'world_stocks' not in downloaded_datasets:
        print("\n❌ World stocks dataset not found!")
        print("Please run Day 2 first: python main.py --download-data")
        return
    
    # Process and clean the data
    results = process_and_clean_data()
    
    if results is None:
        print("\n❌ Data processing failed!")
        return
    
    # Display final results
    cleaned_df = results['cleaned_data']
    target_stocks = results['target_stocks']
    
    print("\n🎯 Day 3 Completed Successfully!")
    print("=" * 50)
    print("✅ Data cleaned and validated")
    print("✅ Stock coverage analyzed")
    print("✅ Target stocks selected")
    print("✅ Comprehensive visualizations created")
    print("✅ Processed data saved")
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"   Records: {len(cleaned_df):,}")
    print(f"   Stocks: {cleaned_df['Ticker'].nunique()}")
    print(f"   Date range: {cleaned_df['Date'].min().date()} to {cleaned_df['Date'].max().date()}")
    print(f"   Target stocks: {', '.join(target_stocks[:5])}{'...' if len(target_stocks) > 5 else ''}")
    
    print("\n📋 Ready for Day 4:")
    print("1. Feature engineering with technical indicators")
    print("2. Advanced time series features")
    print("3. Market sentiment indicators")
    print("4. Feature selection and validation")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download-data":
        print("Data already downloaded in Day 2. Running Day 3 processing...")
    
    main()