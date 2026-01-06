"""
Iterative One-Step Prediction Script
====================================

This script performs iterative one-step ahead prediction for the test set.
For each day in the test period:
1. Computes lag features using previous predictions + historical training data
2. Computes rolling features using previous predictions + historical training data
3. Makes prediction for that day
4. Uses prediction to compute features for next day
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class IterativePredictor:
    """Class for iterative one-step ahead prediction"""
    
    def __init__(self, processed_data_dir='../../data/processed', train_processed=None, test_processed=None):
        self.processed_data_dir = Path(processed_data_dir)
        
        # Load processed data if not provided
        if train_processed is None:
            self.train_processed = pd.read_parquet(self.processed_data_dir / 'train_processed.parquet')
        else:
            self.train_processed = train_processed
            
        if test_processed is None:
            self.test_processed = pd.read_parquet(self.processed_data_dir / 'test_processed.parquet')
        else:
            self.test_processed = test_processed
        
        # Lag and window sizes (should match training)
        self.lags = [1, 7, 14, 30]
        self.windows = [7, 14, 30]
        
    def get_lag_features(self, date, historical_data, predictions_dict=None):
        """
        Get lag features for a given date
        
        Args:
            date: Target date to predict
            historical_data: DataFrame with historical sales (training + previous predictions)
            predictions_dict: Dict of {date: {store_family: sales}} for previous predictions
        """
        lag_features = {}
        
        for lag in self.lags:
            lag_date = date - pd.Timedelta(days=lag)
            
            # Check if we have a prediction for this date
            if predictions_dict and lag_date in predictions_dict:
                # Use prediction if available
                lag_sales = []
                for (store, family), sales in predictions_dict[lag_date].items():
                    lag_sales.append({
                        'store_nbr': store,
                        'family': family,
                        f'sales_lag_{lag}': sales
                    })
                if lag_sales:
                    lag_df = pd.DataFrame(lag_sales)
                else:
                    lag_df = pd.DataFrame(columns=['store_nbr', 'family', f'sales_lag_{lag}'])
            else:
                # Use historical training data
                lag_df = historical_data[
                    historical_data['date'] == lag_date
                ][['store_nbr', 'family', 'sales']].rename(columns={'sales': f'sales_lag_{lag}'})
            
            lag_features[f'sales_lag_{lag}'] = lag_df
        
        return lag_features
    
    def get_rolling_features(self, date, historical_data, predictions_dict=None):
        """
        Get rolling features for a given date
        
        Args:
            date: Target date to predict
            historical_data: DataFrame with historical sales (training + previous predictions)
            predictions_dict: Dict of {date: {store_family: sales}} for previous predictions
        """
        rolling_features = {}
        
        for window in self.windows:
            window_start = date - pd.Timedelta(days=window)
            
            # Combine historical data with predictions
            window_data = historical_data[
                (historical_data['date'] >= window_start) &
                (historical_data['date'] < date)
            ].copy()
            
            # Add predictions if available
            if predictions_dict:
                for pred_date in pd.date_range(window_start, date - pd.Timedelta(days=1), freq='D'):
                    if pred_date in predictions_dict:
                        pred_rows = []
                        for (store, family), sales in predictions_dict[pred_date].items():
                            pred_rows.append({
                                'date': pred_date,
                                'store_nbr': store,
                                'family': family,
                                'sales': sales
                            })
                        if pred_rows:
                            pred_df = pd.DataFrame(pred_rows)
                            window_data = pd.concat([window_data, pred_df], ignore_index=True)
            
            # Compute rolling statistics
            rolling_stats = window_data.groupby(['store_nbr', 'family'])['sales'].agg([
                ('mean', 'mean'),
                ('std', 'std')
            ]).reset_index()
            rolling_stats.columns = ['store_nbr', 'family', 
                                   f'sales_rolling_mean_{window}',
                                   f'sales_rolling_std_{window}']
            
            rolling_features[f'sales_rolling_mean_{window}'] = rolling_stats
            rolling_features[f'sales_rolling_std_{window}'] = rolling_stats
        
        return rolling_features
    
    def prepare_features_for_date(self, date, test_df, historical_data, predictions_dict=None):
        """
        Prepare all features for predicting a specific date
        
        Args:
            date: Date to predict
            test_df: Test dataframe with base features
            historical_data: Historical training data
            predictions_dict: Previous predictions
        """
        # Get test rows for this date
        test_rows = test_df[test_df['date'] == date].copy()
        
        if len(test_rows) == 0:
            return None
        
        # Get lag features
        lag_features = self.get_lag_features(date, historical_data, predictions_dict)
        
        # Merge lag features
        for lag in self.lags:
            lag_df = lag_features[f'sales_lag_{lag}']
            if len(lag_df) > 0:
                test_rows = test_rows.merge(
                    lag_df[['store_nbr', 'family', f'sales_lag_{lag}']],
                    on=['store_nbr', 'family'],
                    how='left'
                )
            else:
                test_rows[f'sales_lag_{lag}'] = np.nan
        
        # Get rolling features
        rolling_features = self.get_rolling_features(date, historical_data, predictions_dict)
        
        # Merge rolling features
        for window in self.windows:
            rolling_df = rolling_features[f'sales_rolling_mean_{window}']
            if len(rolling_df) > 0:
                test_rows = test_rows.merge(
                    rolling_df[['store_nbr', 'family', 
                              f'sales_rolling_mean_{window}',
                              f'sales_rolling_std_{window}']],
                    on=['store_nbr', 'family'],
                    how='left'
                )
            else:
                test_rows[f'sales_rolling_mean_{window}'] = np.nan
                test_rows[f'sales_rolling_std_{window}'] = np.nan
        
        # Fill NaN with 0
        test_rows = test_rows.fillna(0)
        
        return test_rows
    
    def predict_iteratively(self, model, test_dates=None):
        """
        Perform iterative one-step ahead prediction
        
        Args:
            model: Trained model with predict() method
            test_dates: List of dates to predict (default: all test dates)
        """
        if test_dates is None:
            test_dates = sorted(self.test_processed['date'].unique())
        
        # Get historical training data (last 30 days before test)
        test_start = min(test_dates)
        historical_start = test_start - pd.Timedelta(days=30)
        historical_data = self.train_processed[
            (self.train_processed['date'] >= historical_start) &
            (self.train_processed['date'] < test_start)
        ][['store_nbr', 'family', 'date', 'sales']].copy()
        
        # Store predictions: {date: {(store, family): sales}}
        predictions_dict = {}
        all_predictions = []
        
        print(f"\nStarting iterative prediction for {len(test_dates)} days...")
        print("="*50)
        
        for i, date in enumerate(test_dates):
            print(f"\nPredicting day {i+1}/{len(test_dates)}: {date.date()}")
            
            # Prepare features for this date
            features_df = self.prepare_features_for_date(
                date, 
                self.test_processed, 
                historical_data,
                predictions_dict
            )
            
            if features_df is None or len(features_df) == 0:
                print(f"  No data for date {date.date()}")
                continue
            
            # Select feature columns (exclude target and metadata)
            feature_cols = [col for col in features_df.columns 
                          if col not in ['id', 'date', 'store_nbr', 'family', 'sales', 
                                        'type', 'city', 'state', 'cluster']]
            
            # Make predictions
            X = features_df[feature_cols]
            predictions = model.predict(X)
            
            # Store predictions
            date_predictions = {}
            for idx, row in features_df.iterrows():
                store = row['store_nbr']
                family = row['family']
                pred = predictions[features_df.index.get_loc(idx)]
                date_predictions[(store, family)] = max(0, pred)  # Ensure non-negative
            
            predictions_dict[date] = date_predictions
            
            # Store for output
            for idx, row in features_df.iterrows():
                all_predictions.append({
                    'id': row['id'],
                    'date': date,
                    'store_nbr': row['store_nbr'],
                    'family': row['family'],
                    'sales': predictions[features_df.index.get_loc(idx)]
                })
            
            print(f"  Predicted {len(date_predictions)} store-family combinations")
            print(f"  Mean prediction: {np.mean(list(date_predictions.values())):.2f}")
        
        # Create submission dataframe
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values('id')
        
        print("\n" + "="*50)
        print("Iterative prediction complete!")
        print(f"Total predictions: {len(predictions_df)}")
        print("="*50)
        
        return predictions_df


def main():
    """Example usage"""
    predictor = IterativePredictor(processed_data_dir='../../data/processed')
    
    # Load model (you'll need to train and save your model first)
    # model = load_model('models/trained_model.pkl')
    
    # Make predictions
    # predictions = predictor.predict_iteratively(model)
    
    # Save predictions
    # predictions[['id', 'sales']].to_csv('predictions.csv', index=False)
    
    print("Iterative predictor ready!")
    print("To use: train a model, then call predictor.predict_iteratively(model)")


if __name__ == '__main__':
    main()

