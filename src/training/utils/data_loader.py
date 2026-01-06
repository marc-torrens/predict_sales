"""
Data Loader
===========

Functions for loading preprocessed data.
"""

import pandas as pd
import pickle
from pathlib import Path


class DataLoader:
    """Class for loading preprocessed data"""
    
    def __init__(self, processed_data_dir=None):
        """
        Args:
            processed_data_dir: Directory with processed data. If None, resolved as <project_root>/data/processed
        """
        if processed_data_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
            self.processed_data_dir = base_dir / 'data' / 'processed'
        else:
            self.processed_data_dir = Path(processed_data_dir)
    
    def load_train_data(self):
        """Load preprocessed training data"""
        print("Loading preprocessed training data...")
        
        parquet_path = self.processed_data_dir / 'train_processed.parquet'
        
        if parquet_path.exists():
            df_train = pd.read_parquet(parquet_path)
            print(f"Loaded from Parquet: {df_train.shape}")
            return df_train
        else:
            raise FileNotFoundError(
                f"No processed training data found at {parquet_path}\n"
                "Please run preprocessing first, for example:\n"
                "  python src/data_processing/preprocess_data.py"
            )
    
    def load_test_data(self):
        """Load preprocessed test data"""
        print("Loading preprocessed test data...")
        
        parquet_path = self.processed_data_dir / 'test_processed.parquet'
        
        if parquet_path.exists():
            df_test = pd.read_parquet(parquet_path)
            print(f"Loaded from Parquet: {df_test.shape}")
            return df_test
        else:
            raise FileNotFoundError(
                f"No processed test data found at {parquet_path}\n"
                f"Please run preprocessing first: python src/data_processing/preprocess_data.py"
            )
    
    def load_encoders(self):
        """Load label encoders"""
        encoder_path = self.processed_data_dir / 'label_encoders.pkl'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            print("Label encoders loaded")
            return encoders
        return None


