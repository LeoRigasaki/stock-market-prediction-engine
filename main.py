#!/usr/bin/env python3
"""
Stock Market Prediction Engine - Day 1
Data Loading and Initial Setup
"""

from src.config import Config
from src.data_loader import DataLoader
from loguru import logger
import sys

def main():
    """Main execution function for Day 1"""
    
    print("🚀 Stock Market Prediction Engine - Day 1")
    print("=" * 50)
    
    # Create directories
    Config.create_directories()
    print("✅ Project directories created")
    
    # Initialize data loader
    loader = DataLoader()
    print("✅ Data loader initialized")
    
    # List available datasets
    available_datasets = loader.list_available_datasets()
    print(f"\n📊 Available datasets: {', '.join(available_datasets)}")
    
    # Check downloaded datasets
    downloaded_datasets = loader.list_downloaded_datasets()
    print(f"📥 Downloaded datasets: {', '.join(downloaded_datasets) if downloaded_datasets else 'None'}")
    
    # For Day 1, we'll just verify setup is working
    print("\n🎯 Day 1 Goals:")
    print("1. ✅ Virtual environment setup")
    print("2. ✅ Dependencies installed") 
    print("3. ✅ Project structure created")
    print("4. ✅ Configuration system setup")
    print("5. ✅ Data loader module created")
    print("6. ⏳ Kaggle API setup (Next: Add your credentials)")
    
    print("\n📋 Next Steps for Day 2:")
    print("1. Setup Kaggle API credentials")
    print("2. Download world stocks dataset") 
    print("3. Initial data exploration")
    print("4. Data quality assessment")
    
    print("\n🔧 To setup Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Create new API token")
    print("3. Add credentials to .env file")
    print("4. Run: python main.py --download-data")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download-data":
        # This will be used in Day 2
        print("Data download functionality will be implemented in Day 2")
    else:
        main()