#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 2
Data Download and Exploration
"""

from src.config import Config
from src.data_loader import DataLoader
from loguru import logger
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def download_datasets(loader):
    """Download all configured datasets"""
    print("\n📥 Starting dataset downloads...")
    
    success_count = 0
    datasets_to_download = ['world_stocks', 'nasdaq_stocks']  # Start with these two
    
    for dataset_name in datasets_to_download:
        print(f"\n⏳ Downloading {dataset_name}...")
        if loader.download_dataset(dataset_name):
            success_count += 1
            print(f"✅ {dataset_name} downloaded successfully")
            
            # Get dataset info
            info = loader.get_dataset_info(dataset_name)
            if info:
                print(f"   📊 Files: {info['csv_files']} CSV files")
                print(f"   💾 Size: {info['total_size_mb']:.2f} MB")
                if info.get('csv_preview'):
                    for csv_info in info['csv_preview']:
                        print(f"   📄 {csv_info['filename']}: {csv_info['column_count']} columns")
        else:
            print(f"❌ Failed to download {dataset_name}")
    
    print(f"\n🎯 Download Summary: {success_count}/{len(datasets_to_download)} datasets downloaded")
    return success_count > 0

def explore_world_stocks(loader):
    """Explore the world stocks dataset"""
    print("\n🔍 Exploring World Stocks Dataset...")
    
    dataset_path = Config.RAW_DATA_PATH / "world_stocks"
    
    if not dataset_path.exists():
        print("❌ World stocks dataset not found. Please download first.")
        return None
    
    # Load CSV files
    dataframes = loader.load_csv_files(dataset_path)
    
    if not dataframes:
        print("❌ No CSV files found in world stocks dataset")
        return None
    
    # Find the main dataset (usually the largest one)
    main_df_name = max(dataframes.keys(), key=lambda k: dataframes[k].shape[0])
    main_df = dataframes[main_df_name]
    
    print(f"📊 Main dataset: {main_df_name}")
    print(f"   Shape: {main_df.shape}")
    print(f"   Columns: {list(main_df.columns)}")
    
    # Basic exploration
    print("\n📈 Dataset Overview:")
    print(f"   Date range: {main_df.iloc[:, 0].min()} to {main_df.iloc[:, 0].max()}")
    
    # Check for stock symbols/tickers
    potential_ticker_cols = [col for col in main_df.columns if 'ticker' in col.lower() or 'symbol' in col.lower() or 'stock' in col.lower()]
    if potential_ticker_cols:
        ticker_col = potential_ticker_cols[0]
        print(f"   Unique stocks: {main_df[ticker_col].nunique()}")
        print(f"   Sample tickers: {main_df[ticker_col].unique()[:10]}")
    
    # Show first few rows
    print("\n📋 Sample Data:")
    print(main_df.head())
    
    # Data types
    print("\n🔍 Data Types:")
    print(main_df.dtypes)
    
    # Check for missing values
    missing_data = main_df.isnull().sum()
    if missing_data.any():
        print("\n⚠️ Missing Data:")
        print(missing_data[missing_data > 0])
    else:
        print("\n✅ No missing data found")
    
    return main_df

def analyze_data_quality(df):
    """Perform data quality analysis"""
    print("\n🔍 Data Quality Analysis:")
    
    # Basic statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"   Numeric columns: {len(numeric_cols)}")
        print("\n📊 Numeric Data Summary:")
        print(df[numeric_cols].describe())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {duplicates}")
    
    # Data completeness
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    print("\n✅ Data Completeness (%):")
    for col, pct in completeness.items():
        status = "✅" if pct > 95 else "⚠️" if pct > 80 else "❌"
        print(f"   {status} {col}: {pct:.1f}%")

def identify_target_stocks(df):
    """Identify stocks for prediction modeling"""
    print("\n🎯 Identifying Target Stocks for Prediction:")
    
    # Try to find ticker column
    ticker_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['ticker', 'symbol', 'stock', 'name'])]
    
    if not ticker_cols:
        print("❌ Could not identify ticker column")
        return []
    
    ticker_col = ticker_cols[0]
    print(f"   Using column: {ticker_col}")
    
    # Get stock counts
    stock_counts = df[ticker_col].value_counts()
    print(f"   Total unique stocks: {len(stock_counts)}")
    
    # Focus on stocks with substantial data
    min_records = 100  # Minimum records per stock
    qualified_stocks = stock_counts[stock_counts >= min_records]
    
    print(f"   Stocks with ≥{min_records} records: {len(qualified_stocks)}")
    
    # Target popular stocks (if we can identify them)
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    available_popular = [ticker for ticker in popular_tickers if ticker in qualified_stocks.index]
    
    print(f"   Popular stocks available: {available_popular}")
    
    # Select top 10 stocks by data volume
    top_stocks = qualified_stocks.head(10).index.tolist()
    print(f"   Top 10 stocks by data volume: {top_stocks}")
    
    return top_stocks

def create_initial_visualizations(df, target_stocks):
    """Create initial data visualizations"""
    print("\n📊 Creating Initial Visualizations...")
    
    try:
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Market Data - Initial Analysis', fontsize=16, fontweight='bold')
        
        # Find date and price columns
        date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
        price_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['close', 'price', 'adj_close'])]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        ticker_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['ticker', 'symbol', 'stock'])]
        
        if date_cols and ticker_cols:
            date_col = date_cols[0]
            ticker_col = ticker_cols[0]
            
            # Convert date column
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Plot 1: Data availability over time
            monthly_counts = df.groupby([df[date_col].dt.to_period('M'), ticker_col]).size().reset_index()
            monthly_counts['date'] = monthly_counts[date_col].dt.to_timestamp()
            
            if len(target_stocks) > 0:
                target_data = monthly_counts[monthly_counts[ticker_col].isin(target_stocks[:5])]
                for stock in target_stocks[:5]:
                    stock_data = target_data[target_data[ticker_col] == stock]
                    if not stock_data.empty:
                        axes[0,0].plot(stock_data['date'], stock_data[0], label=stock, alpha=0.7)
            
            axes[0,0].set_title('Data Availability Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Records per Month')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Top stocks by data volume
            stock_counts = df[ticker_col].value_counts().head(10)
            axes[0,1].bar(range(len(stock_counts)), stock_counts.values)
            axes[0,1].set_title('Top 10 Stocks by Data Volume')
            axes[0,1].set_xlabel('Stock Rank')
            axes[0,1].set_ylabel('Number of Records')
            axes[0,1].set_xticks(range(len(stock_counts)))
            axes[0,1].set_xticklabels(stock_counts.index, rotation=45)
            
            # Plot 3: Price distribution (if price column exists)
            if price_cols:
                price_col = price_cols[0]
                numeric_prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
                axes[1,0].hist(numeric_prices, bins=50, alpha=0.7, edgecolor='black')
                axes[1,0].set_title(f'{price_col} Distribution')
                axes[1,0].set_xlabel('Price')
                axes[1,0].set_ylabel('Frequency')
            
            # Plot 4: Volume distribution (if volume column exists)
            if volume_cols:
                volume_col = volume_cols[0]
                numeric_volumes = pd.to_numeric(df[volume_col], errors='coerce').dropna()
                if len(numeric_volumes) > 0:
                    axes[1,1].hist(numeric_volumes, bins=50, alpha=0.7, edgecolor='black')
                    axes[1,1].set_title(f'{volume_col} Distribution')
                    axes[1,1].set_xlabel('Volume')
                    axes[1,1].set_ylabel('Frequency')
                    axes[1,1].set_yscale('log')  # Log scale for volume
        
        plt.tight_layout()
        
        # Save the plot
        plots_dir = Config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "day2_initial_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {plots_dir / 'day2_initial_analysis.png'}")
        
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")

def main():
    """Main execution function for Day 2"""
    
    print("🚀 Stock Market Prediction Engine - Day 2")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--download-data":
        # Download datasets
        if download_datasets(loader):
            print("\n🎉 Data download completed!")
        else:
            print("\n❌ Data download failed!")
            return
    
    # Check if we have data to explore
    downloaded_datasets = loader.list_downloaded_datasets()
    
    if 'world_stocks' not in downloaded_datasets:
        print("\n📥 No data found. Run with --download-data to download datasets:")
        print("python main.py --download-data")
        return
    
    # Explore the data
    main_df = explore_world_stocks(loader)
    
    if main_df is not None:
        # Analyze data quality
        analyze_data_quality(main_df)
        
        # Identify target stocks
        target_stocks = identify_target_stocks(main_df)
        
        # Create visualizations
        create_initial_visualizations(main_df, target_stocks)
        
        # Save summary info
        summary = {
            'dataset_shape': main_df.shape,
            'columns': list(main_df.columns),
            'target_stocks': target_stocks,
            'date_processed': datetime.now().isoformat()
        }
        
        import json
        summary_path = Config.PROCESSED_DATA_PATH / "day2_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to {summary_path}")
        
        print("\n🎯 Day 2 Completed Successfully!")
        print("=" * 50)
        print("✅ Dataset downloaded and explored")
        print("✅ Data quality assessed")
        print("✅ Target stocks identified")
        print("✅ Initial visualizations created")
        
        print("\n📋 Ready for Day 3:")
        print("1. Data preprocessing and cleaning")
        print("2. Feature engineering setup")
        print("3. Technical indicators calculation")
        print("4. Data split for modeling")

if __name__ == "__main__":
    main()