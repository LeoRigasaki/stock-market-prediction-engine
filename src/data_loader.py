import os
import pandas as pd
import kaggle
from pathlib import Path
from loguru import logger
from typing import Dict, Optional, List
import zipfile

from .config import Config

class DataLoader:
    """Data loading utilities for stock market data"""
    
    def __init__(self):
        self.config = Config()
        self.setup_kaggle_auth()
        self.setup_logging()
    
    def setup_kaggle_auth(self):
        """Setup Kaggle API authentication"""
        try:
            # Check if credentials exist
            if not self.config.KAGGLE_USERNAME or not self.config.KAGGLE_KEY:
                logger.warning("Kaggle credentials not found in environment variables")
                return False
            
            # Set environment variables for Kaggle API
            os.environ['KAGGLE_USERNAME'] = self.config.KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = self.config.KAGGLE_KEY
            
            # Test authentication
            kaggle.api.authenticate()
            logger.info("Kaggle API authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Kaggle authentication failed: {e}")
            return False
    
    def setup_logging(self):
        """Setup logging configuration"""
        logger.add(
            self.config.LOG_FILE,
            rotation="10 MB",
            retention="30 days",
            level=self.config.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}"
        )
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """
        Download a dataset from Kaggle
        
        Args:
            dataset_name: Name of the dataset (key from Config.DATASETS)
            force_download: Whether to re-download if already exists
            
        Returns:
            bool: Success status
        """
        try:
            if dataset_name not in self.config.DATASETS:
                logger.error(f"Dataset '{dataset_name}' not found in configuration")
                return False
            
            dataset_id = self.config.DATASETS[dataset_name]
            download_path = self.config.RAW_DATA_PATH / dataset_name
            
            # Check if already downloaded
            if download_path.exists() and not force_download:
                logger.info(f"Dataset '{dataset_name}' already exists. Use force_download=True to re-download.")
                return True
            
            # Create download directory
            download_path.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            logger.info(f"Downloading dataset: {dataset_id}")
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(download_path),
                unzip=True
            )
            
            logger.info(f"Successfully downloaded {dataset_name} to {download_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset '{dataset_name}': {e}")
            return False
    
    def load_csv_files(self, dataset_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from a dataset directory
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dict of DataFrames with filename as key
        """
        dataframes = {}
        
        try:
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {dataset_path}")
                return dataframes
            
            for csv_file in csv_files:
                try:
                    logger.info(f"Loading {csv_file.name}")
                    df = pd.read_csv(csv_file)
                    
                    # Basic data info
                    logger.info(f"{csv_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    dataframes[csv_file.stem] = df
                    
                except Exception as e:
                    logger.error(f"Failed to load {csv_file.name}: {e}")
            
            return dataframes
            
        except Exception as e:
            logger.error(f"Error loading CSV files from {dataset_path}: {e}")
            return dataframes
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """
        Get information about a downloaded dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        try:
            dataset_path = self.config.RAW_DATA_PATH / dataset_name
            
            if not dataset_path.exists():
                logger.warning(f"Dataset '{dataset_name}' not found at {dataset_path}")
                return None
            
            # Get all files in dataset
            all_files = list(dataset_path.rglob("*"))
            csv_files = list(dataset_path.glob("*.csv"))
            
            info = {
                'name': dataset_name,
                'path': str(dataset_path),
                'total_files': len(all_files),
                'csv_files': len(csv_files),
                'csv_file_names': [f.name for f in csv_files],
                'total_size_mb': sum(f.stat().st_size for f in all_files if f.is_file()) / (1024 * 1024)
            }
            
            # Get basic info about CSV files
            if csv_files:
                csv_info = []
                for csv_file in csv_files[:3]:  # Limit to first 3 files for quick info
                    try:
                        df = pd.read_csv(csv_file, nrows=0)  # Just get columns
                        csv_info.append({
                            'filename': csv_file.name,
                            'columns': list(df.columns),
                            'column_count': len(df.columns)
                        })
                    except Exception as e:
                        logger.warning(f"Could not read {csv_file.name}: {e}")
                
                info['csv_preview'] = csv_info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info for '{dataset_name}': {e}")
            return None
    
    def list_available_datasets(self) -> List[str]:
        """List all configured datasets"""
        return list(self.config.DATASETS.keys())
    
    def list_downloaded_datasets(self) -> List[str]:
        """List all downloaded datasets"""
        if not self.config.RAW_DATA_PATH.exists():
            return []
        
        return [d.name for d in self.config.RAW_DATA_PATH.iterdir() if d.is_dir()]