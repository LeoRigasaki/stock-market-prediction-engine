import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config

class DataProcessor:
    """Data preprocessing and cleaning utilities"""
    
    def __init__(self):
        self.config = Config()
        
    def load_and_clean_world_stocks(self) -> pd.DataFrame:
        """Load and clean the world stocks dataset"""
        logger.info("Loading and cleaning world stocks dataset...")
        
        # Load the dataset
        dataset_path = self.config.RAW_DATA_PATH / "world_stocks" / "World-Stock-Prices-Dataset.csv"
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return pd.DataFrame()
        
        # Read CSV
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Fix date parsing - handle timezone issues
        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        df['Date'] = df['Date'].dt.tz_convert(None)  # Remove timezone for easier handling
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        logger.info(f"After date cleaning: {df.shape[0]} rows")
        
        # Clean ticker symbols - remove any whitespace
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        
        # Remove rows with missing essential data
        essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        df = df.dropna(subset=essential_cols)
        logger.info(f"After essential data cleaning: {df.shape[0]} rows")
        
        # Remove obvious data errors
        # Prices should be positive
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Volume should be non-negative
        df = df[df['Volume'] >= 0]
        
        # High >= Low, and Close/Open should be between Low and High
        df = df[df['High'] >= df['Low']]
        df = df[df['Close'] >= df['Low']]
        df = df[df['Close'] <= df['High']]
        df = df[df['Open'] >= df['Low']]
        df = df[df['Open'] <= df['High']]
        
        logger.info(f"After price validation: {df.shape[0]} rows")
        
        # Sort by date and ticker
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Drop the problematic Capital Gains column (99.9% missing)
        if 'Capital_Gains' in df.columns:
            df = df.drop('Capital_Gains', axis=1)
        
        logger.info(f"Final cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def analyze_stock_coverage(self, df: pd.DataFrame) -> Dict:
        """Analyze data coverage for each stock"""
        logger.info("Analyzing stock data coverage...")
        
        stock_analysis = []
        
        for ticker in df['Ticker'].unique():
            stock_data = df[df['Ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('Date')
            
            # Calculate metrics
            analysis = {
                'ticker': ticker,
                'brand_name': stock_data['Brand_Name'].iloc[0] if 'Brand_Name' in stock_data.columns else 'Unknown',
                'industry': stock_data['Industry_Tag'].iloc[0] if 'Industry_Tag' in stock_data.columns else 'Unknown',
                'country': stock_data['Country'].iloc[0] if 'Country' in stock_data.columns else 'Unknown',
                'total_records': len(stock_data),
                'date_range_start': stock_data['Date'].min(),
                'date_range_end': stock_data['Date'].max(),
                'trading_days': (stock_data['Date'].max() - stock_data['Date'].min()).days,
                'avg_volume': stock_data['Volume'].mean(),
                'avg_price': stock_data['Close'].mean(),
                'price_volatility': stock_data['Close'].std(),
                'min_price': stock_data['Close'].min(),
                'max_price': stock_data['Close'].max(),
                'data_completeness': len(stock_data) / max(1, stock_data['Date'].nunique())
            }
            
            stock_analysis.append(analysis)
        
        # Convert to DataFrame for easier analysis
        analysis_df = pd.DataFrame(stock_analysis)
        analysis_df = analysis_df.sort_values('total_records', ascending=False)
        
        logger.info(f"Analyzed {len(analysis_df)} stocks")
        return analysis_df
    
    def select_target_stocks(self, analysis_df: pd.DataFrame, min_records: int = 1000) -> List[str]:
        """Select the best stocks for prediction modeling"""
        logger.info(f"Selecting target stocks with minimum {min_records} records...")
        
        # Filter stocks with sufficient data
        qualified_stocks = analysis_df[analysis_df['total_records'] >= min_records].copy()
        
        logger.info(f"Stocks with ≥{min_records} records: {len(qualified_stocks)}")
        
        if len(qualified_stocks) == 0:
            # Lower the threshold if no stocks meet criteria
            min_records = 500
            qualified_stocks = analysis_df[analysis_df['total_records'] >= min_records].copy()
            logger.warning(f"Lowered threshold to {min_records} records: {len(qualified_stocks)} stocks")
        
        # Prefer stocks with longer trading history and higher volume
        qualified_stocks['score'] = (
            qualified_stocks['total_records'] * 0.4 +
            qualified_stocks['trading_days'] * 0.3 +
            qualified_stocks['avg_volume'].rank() * 0.2 +
            qualified_stocks['data_completeness'] * 0.1
        )
        
        # Select top stocks
        top_stocks = qualified_stocks.nlargest(10, 'score')
        
        logger.info("Selected target stocks:")
        for idx, row in top_stocks.iterrows():
            logger.info(f"  {row['ticker']} ({row['brand_name']}): {row['total_records']} records, "
                       f"{row['trading_days']} days, avg_price=${row['avg_price']:.2f}")
        
        return top_stocks['ticker'].tolist()
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame, target_stocks: List[str]):
        """Create comprehensive data visualizations"""
        logger.info("Creating comprehensive visualizations...")
        
        # Set up the plotting environment
        plt.style.use('default')  # Use default style to avoid seaborn issues
        
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Stock data volume comparison
        plt.subplot(3, 3, 1)
        stock_counts = df['Ticker'].value_counts().head(10)
        plt.barh(range(len(stock_counts)), stock_counts.values)
        plt.yticks(range(len(stock_counts)), stock_counts.index)
        plt.xlabel('Number of Records')
        plt.title('Top 10 Stocks by Data Volume')
        plt.grid(True, alpha=0.3)
        
        # 2. Price distribution
        plt.subplot(3, 3, 2)
        plt.hist(df['Close'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Closing Price ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Closing Prices')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 3. Volume distribution
        plt.subplot(3, 3, 3)
        plt.hist(df['Volume'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Trading Volume')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trading Volume')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # 4. Data timeline
        plt.subplot(3, 3, 4)
        monthly_counts = df.groupby(df['Date'].dt.to_period('M')).size()
        monthly_counts.plot(kind='line')
        plt.xlabel('Date')
        plt.ylabel('Records per Month')
        plt.title('Data Availability Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Industry distribution
        plt.subplot(3, 3, 5)
        if 'Industry_Tag' in df.columns:
            industry_counts = df['Industry_Tag'].value_counts()
            plt.pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%')
            plt.title('Distribution by Industry')
        
        # 6. Country distribution
        plt.subplot(3, 3, 6)
        if 'Country' in df.columns:
            country_counts = df['Country'].value_counts()
            plt.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%')
            plt.title('Distribution by Country')
        
        # 7. Price ranges for top stocks
        plt.subplot(3, 3, 7)
        if target_stocks:
            for i, stock in enumerate(target_stocks[:5]):
                stock_data = df[df['Ticker'] == stock]
                if not stock_data.empty:
                    plt.plot(stock_data['Date'], stock_data['Close'], 
                            label=stock, alpha=0.8, linewidth=1)
            plt.xlabel('Date')
            plt.ylabel('Closing Price ($)')
            plt.title('Price Trends for Top 5 Stocks')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 8. Volatility analysis
        plt.subplot(3, 3, 8)
        df['daily_return'] = df.groupby('Ticker')['Close'].pct_change()
        volatility_by_stock = df.groupby('Ticker')['daily_return'].std().sort_values(ascending=False).head(10)
        plt.barh(range(len(volatility_by_stock)), volatility_by_stock.values)
        plt.yticks(range(len(volatility_by_stock)), volatility_by_stock.index)
        plt.xlabel('Daily Return Volatility')
        plt.title('Most Volatile Stocks')
        plt.grid(True, alpha=0.3)
        
        # 9. Correlation heatmap for top stocks
        plt.subplot(3, 3, 9)
        if len(target_stocks) >= 3:
            pivot_data = df[df['Ticker'].isin(target_stocks[:5])].pivot_table(
                index='Date', columns='Ticker', values='Close'
            )
            correlation_matrix = pivot_data.corr()
            
            # Create heatmap manually since seaborn might have issues
            im = plt.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
            plt.title('Stock Price Correlations')
            
            # Add correlation values
            for i in range(len(correlation_matrix.index)):
                for j in range(len(correlation_matrix.columns)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plots_dir = self.config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "day3_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Comprehensive visualizations saved to {plots_dir / 'day3_comprehensive_analysis.png'}")
    
    def save_cleaned_data(self, df: pd.DataFrame, analysis_df: pd.DataFrame, target_stocks: List[str]):
        """Save cleaned data and analysis results"""
        logger.info("Saving cleaned data and analysis...")
        
        # Save cleaned dataset
        cleaned_path = self.config.PROCESSED_DATA_PATH / "cleaned_world_stocks.csv"
        df.to_csv(cleaned_path, index=False)
        logger.info(f"Cleaned dataset saved to {cleaned_path}")
        
        # Save stock analysis
        analysis_path = self.config.PROCESSED_DATA_PATH / "stock_analysis.csv"
        analysis_df.to_csv(analysis_path, index=False)
        logger.info(f"Stock analysis saved to {analysis_path}")
        
        # Save target stocks for future use
        target_stocks_path = self.config.PROCESSED_DATA_PATH / "target_stocks.txt"
        with open(target_stocks_path, 'w') as f:
            for stock in target_stocks:
                f.write(f"{stock}\n")
        logger.info(f"Target stocks saved to {target_stocks_path}")
        
        # Create summary report
        summary = {
            'processing_date': datetime.now().isoformat(),
            'original_records': len(df),
            'cleaned_records': len(df),
            'unique_stocks': df['Ticker'].nunique(),
            'date_range': {
                'start': df['Date'].min().isoformat(),
                'end': df['Date'].max().isoformat()
            },
            'target_stocks': target_stocks,
            'data_quality': {
                'completeness': (1 - df.isnull().sum() / len(df)).to_dict(),
                'duplicates': df.duplicated().sum()
            }
        }
        
        import json
        summary_path = self.config.PROCESSED_DATA_PATH / "day3_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Processing summary saved to {summary_path}")
        
        return {
            'cleaned_dataset': cleaned_path,
            'stock_analysis': analysis_path,
            'target_stocks': target_stocks_path,
            'summary': summary_path
        }