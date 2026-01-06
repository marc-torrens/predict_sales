"""
Data Splitting Utilities
=========================

Functions for splitting data and preparing features.
"""

import pandas as pd
import numpy as np


def create_time_split(df, validation_days=30):
    """
    Create time-based train/validation split
    
    Args:
        df: Training dataframe
        validation_days: Number of days to use for validation (last N days)
    
    Returns:
        df_train, df_val: Training and validation dataframes
    """
    print(f"\nCreating time-based split (last {validation_days} days for validation)...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get date range
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=validation_days)
    
    # Split
    train_mask = df['date'] < split_date
    val_mask = df['date'] >= split_date
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    
    print(f"Training period: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    print(f"Validation period: {df_val['date'].min().date()} to {df_val['date'].max().date()}")
    print(f"Training samples: {len(df_train):,}")
    print(f"Validation samples: {len(df_val):,}")
    
    return df_train, df_val


def prepare_features(df):
    """
    Prepare features and target for training
    
    Args:
        df: Dataframe with features and target
    
    Returns:
        X, y, feature_cols: Features, target, and feature column names
    """
    print("\nPreparing features...")
    
    # Exclude metadata columns
    exclude_cols = ['id', 'date', 'store_nbr', 'family', 'sales', 
                   'type', 'city', 'state', 'cluster']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols].copy()
    y = df['sales'].copy()
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    return X, y, feature_cols



