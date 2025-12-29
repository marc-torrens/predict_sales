"""
Data Preprocessing Script for Sales Prediction
==============================================

This script preprocesses all datasets for sales prediction modeling:
- Handles missing values
- Processes outliers using IQR (bounds computed from training data)
- Handles missing periods in time series
- Merges all datasets
- Creates features (time, lag, rolling)
- Encodes categorical variables
- Saves processed data
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Main class for data preprocessing"""
    
    def __init__(self, data_dir='../../data', output_dir='../../data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load all raw datasets"""
        print("Loading datasets...")
        
        self.df_train = pd.read_csv(self.data_dir / 'train.csv')
        self.df_test = pd.read_csv(self.data_dir / 'test.csv')
        self.df_stores = pd.read_csv(self.data_dir / 'stores.csv')
        self.df_oil = pd.read_csv(self.data_dir / 'oil.csv')
        self.df_holidays = pd.read_csv(self.data_dir / 'holidays_events.csv')
        self.df_transactions = pd.read_csv(self.data_dir / 'transactions.csv')
        
        print(f"Train shape: {self.df_train.shape}")
        print(f"Test shape: {self.df_test.shape}")
        print(f"Stores shape: {self.df_stores.shape}")
        print(f"Oil shape: {self.df_oil.shape}")
        print(f"Holidays shape: {self.df_holidays.shape}")
        print(f"Transactions shape: {self.df_transactions.shape}")
        
    def process_dates(self):
        """Convert date columns to datetime"""
        print("\nProcessing dates...")
        
        self.df_train['date'] = pd.to_datetime(self.df_train['date'])
        self.df_test['date'] = pd.to_datetime(self.df_test['date'])
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])
        self.df_holidays['date'] = pd.to_datetime(self.df_holidays['date'])
        self.df_transactions['date'] = pd.to_datetime(self.df_transactions['date'])
        
        self.df_train['family'] = self.df_train['family'].astype(str)
        self.df_test['family'] = self.df_test['family'].astype(str)
        
    def process_oil(self):
        """Process oil price data - handle missing values"""
        print("\nProcessing oil prices...")
        
        self.df_oil = self.df_oil.sort_values('date')
        self.df_oil['dcoilwtico'] = self.df_oil['dcoilwtico'].ffill().bfill()
        
        missing_pct = (self.df_oil['dcoilwtico'].isna().sum() / len(self.df_oil)) * 100
        print(f"Oil price missing values after filling: {missing_pct:.2f}%")
        
    def process_holidays(self):
        """Process holidays data - create holiday flags"""
        print("\nProcessing holidays...")
        
        # Filter out transferred holidays
        self.df_holidays_active = self.df_holidays[
            self.df_holidays['type'] != 'Transfer'
        ].copy()
        
        # Create separate datasets for different holiday types
        self.df_holidays_national = self.df_holidays_active[
            self.df_holidays_active['locale'] == 'National'
        ].copy()
        
        self.df_holidays_regional = self.df_holidays_active[
            self.df_holidays_active['locale'] == 'Regional'
        ].copy()
        
        self.df_holidays_local = self.df_holidays_active[
            self.df_holidays_active['locale'] == 'Local'
        ].copy()
        
        print(f"National holidays: {len(self.df_holidays_national)}")
        print(f"Regional holidays: {len(self.df_holidays_regional)}")
        print(f"Local holidays: {len(self.df_holidays_local)}")
        
    def detect_outliers_iqr(self, data):
        """Detect outliers using IQR method
        Args:
            data: pandas Series of sales values
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound
    
    def compute_outlier_bounds_from_train(self, df_train, iqr_multiplier=3.0, only_upper=True):
        """
        Compute IQR bounds from training data for each store-family combination
        
        Args:
            df_train: Training dataframe
            iqr_multiplier: Multiplier for IQR (default 4.0, very conservative - fewer outliers)
            only_upper: If True, only detect upper outliers (not lower)
        """
        print("\nComputing outlier bounds from training data...")
        print(f"  IQR multiplier: {iqr_multiplier} (very conservative - fewer outliers detected)")
        print(f"  Only upper outliers: {only_upper}")
        print("  Action: Remove outliers (not cap)")
        
        # Sales statistics
        total_rows = len(df_train)
        zero_sales = (df_train['sales'] == 0).sum()
        positive_sales = (df_train['sales'] > 0).sum()
        
        print("\nSales distribution:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Zero sales: {zero_sales:,} ({zero_sales/total_rows*100:.2f}%)")
        print(f"  Positive sales: {positive_sales:,} ({positive_sales/total_rows*100:.2f}%)")
        
        outlier_bounds = {}
        
        for (store, family), group_df in df_train.groupby(['store_nbr', 'family']):
            sales_positive = group_df[group_df['sales'] > 0]['sales']
            if len(sales_positive) > 0:
                Q1 = sales_positive.quantile(0.25)
                Q3 = sales_positive.quantile(0.75)
                IQR = Q3 - Q1
                
                # Very conservative bounds (fewer outliers)
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outlier_bounds[(store, family)] = {
                    'lower_bound': lower_bound if not only_upper else -np.inf,
                    'upper_bound': upper_bound
                }
        
        print(f"\nComputed bounds for {len(outlier_bounds)} store-family combinations")
        return outlier_bounds
    
    def apply_outlier_bounds(self, df, outlier_bounds):
        """
        Remove outliers from data (drop rows instead of capping)
        
        Args:
            df: Dataframe to process
            outlier_bounds: Dictionary of outlier bounds
        """
        print("\nRemoving outliers from data...")
        
        df_processed = df.copy()
        outlier_indices = set()
        upper_outlier_count = 0
        lower_outlier_count = 0
        total_rows = len(df_processed)
        total_sales_rows = len(df_processed[df_processed['sales'] > 0])
        
        for (store, family), group_df in df_processed.groupby(['store_nbr', 'family']):
            if (store, family) in outlier_bounds:
                bounds = outlier_bounds[(store, family)]
                lower_bound = bounds['lower_bound']
                upper_bound = bounds['upper_bound']
                
                # Detect outliers
                upper_outliers = group_df['sales'] > upper_bound
                lower_outliers = group_df['sales'] < lower_bound
                
                # Collect outlier indices to remove
                if upper_outliers.any():
                    outlier_indices.update(group_df[upper_outliers].index.tolist())
                    upper_outlier_count += upper_outliers.sum()
                
                # Handle lower outliers (only if enabled)
                if lower_outliers.any() and lower_bound > -np.inf:
                    outlier_indices.update(group_df[lower_outliers].index.tolist())
                    lower_outlier_count += lower_outliers.sum()
        
        # Remove outliers
        df_processed = df_processed.drop(index=outlier_indices)
        
        total_outliers = upper_outlier_count + lower_outlier_count
        rows_after = len(df_processed)
        rows_removed = total_rows - rows_after
        
        outlier_pct_total = (total_outliers / total_rows * 100) if total_rows > 0 else 0
        outlier_pct_sales = (total_outliers / total_sales_rows * 100) if total_sales_rows > 0 else 0
        
        print(f"Outliers removed: {total_outliers:,} rows")
        print(f"  - Upper outliers: {upper_outlier_count:,}")
        print(f"  - Lower outliers: {lower_outlier_count:,}")
        print(f"  - Rows before: {total_rows:,}")
        print(f"  - Rows after: {rows_after:,}")
        print(f"  - Rows removed: {rows_removed:,} ({outlier_pct_total:.2f}% of total)")
        print(f"  - Percentage of rows with sales > 0 removed: {outlier_pct_sales:.2f}%")
        
        return df_processed
    
    def handle_missing_periods(self, df):
        """Handle missing periods in time series"""
        print("\nHandling missing periods...")
        
        # Create complete date range
        date_range = pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='D'
        )
        
        # Create MultiIndex for all combinations
        stores = df['store_nbr'].unique()
        families = df['family'].unique()
        
        # Create complete index
        complete_index = pd.MultiIndex.from_product(
            [stores, families, date_range],
            names=['store_nbr', 'family', 'date']
        )
        
        # Set index
        df_indexed = df.set_index(['store_nbr', 'family', 'date'])
        
        # Reindex to complete index
        df_complete = df_indexed.reindex(complete_index)
        
        # Reset index
        df_complete = df_complete.reset_index()
        
        # Sort
        df_complete = df_complete.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
        
        # Interpolate sales for each group
        df_complete['sales'] = (
            df_complete.groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.interpolate(method='linear').fillna(0))
        )
        
        # Fill promotion with 0 where missing
        df_complete['onpromotion'] = df_complete['onpromotion'].fillna(0)
        
        # Fill other columns that might be missing
        for col in df_complete.columns:
            if col not in ['store_nbr', 'family', 'date', 'sales', 'onpromotion']:
                df_complete[col] = df_complete[col].fillna(0)
        
        missing_count = df_complete['sales'].isna().sum()
        print(f"Missing periods filled: {missing_count}")
        
        return df_complete
    
    def create_time_features(self, df):
        """Create time-based features"""
        print("\nCreating time features...")
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        return df
    
    def merge_datasets(self, df):
        """Merge all datasets"""
        print("\nMerging datasets...")
        
        # Merge with stores
        df = df.merge(self.df_stores, on='store_nbr', how='left')
        
        # Merge with oil
        df = df.merge(
            self.df_oil[['date', 'dcoilwtico']], 
            on='date', 
            how='left'
        )
        
        # Merge with transactions
        df = df.merge(
            self.df_transactions, 
            on=['date', 'store_nbr'], 
            how='left'
        )
        
        # Merge national holidays
        df = df.merge(
            self.df_holidays_national[['date']].drop_duplicates(),
            on='date',
            how='left',
            indicator='is_national_holiday'
        )
        df['is_national_holiday'] = (df['is_national_holiday'] == 'both').astype(int)
        
        # Merge regional holidays
        df = df.merge(
            self.df_holidays_regional[['date', 'locale_name']].drop_duplicates(),
            left_on=['date', 'state'],
            right_on=['date', 'locale_name'],
            how='left',
            indicator='is_regional_holiday'
        )
        df['is_regional_holiday'] = (df['is_regional_holiday'] == 'both').astype(int)
        df = df.drop('locale_name', axis=1, errors='ignore')
        
        # Merge local holidays
        df = df.merge(
            self.df_holidays_local[['date', 'locale_name']].drop_duplicates(),
            left_on=['date', 'city'],
            right_on=['date', 'locale_name'],
            how='left',
            indicator='is_local_holiday'
        )
        df['is_local_holiday'] = (df['is_local_holiday'] == 'both').astype(int)
        df = df.drop('locale_name', axis=1, errors='ignore')
        
        # Create combined holiday flag
        df['is_any_holiday'] = (
            df['is_national_holiday'] | 
            df['is_regional_holiday'] | 
            df['is_local_holiday']
        ).astype(int)
        
        print(f"Merged data shape: {df.shape}")
        return df
    
    def create_lag_features(self, df, lags=[1, 7, 14, 30]):
        """
        Create lag features for time series
        Includes lags up to 30 days (1 month)
        For test set, historical training data is used for lags > 15 days
        """
        print(f"\nCreating lag features (lags: {lags})...")
        
        df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
        
        for lag in lags:
            df[f'sales_lag_{lag}'] = (
                df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
            )
        
        return df
    
    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """
        Create rolling window features
        Includes windows up to 30 days (1 month)
        For test set, historical training data is used for windows > 15 days
        """
        print(f"\nCreating rolling features (windows: {windows})...")
        
        df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
        df_index = df.index.copy()
        
        for window in windows:
            rolling_mean = (
                df.groupby(['store_nbr', 'family'])['sales']
                .rolling(window=window, min_periods=1)
                .mean()
            )
            rolling_std = (
                df.groupby(['store_nbr', 'family'])['sales']
                .rolling(window=window, min_periods=1)
                .std()
            )
            
            # Reset index to get back to original index
            rolling_mean = rolling_mean.reset_index(level=[0, 1], drop=True)
            rolling_std = rolling_std.reset_index(level=[0, 1], drop=True)
            
            # Align with dataframe index
            rolling_mean = rolling_mean.reindex(df_index)
            rolling_std = rolling_std.reindex(df_index).fillna(0)
            
            df[f'sales_rolling_mean_{window}'] = rolling_mean.values
            df[f'sales_rolling_std_{window}'] = rolling_std.values
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        # Store type encoding
        type_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        df['type_encoded'] = df['type'].map(type_mapping)
        
        # Family encoding (label encoding)
        le_family = LabelEncoder()
        df['family_encoded'] = le_family.fit_transform(df['family'])
        
        # State and city encoding
        le_state = LabelEncoder()
        le_city = LabelEncoder()
        df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
        df['city_encoded'] = le_city.fit_transform(df['city'].astype(str))
        
        # Save encoders for later use
        self.label_encoders = {
            'family': le_family,
            'state': le_state,
            'city': le_city
        }
        
        return df
    
    def process_train(self):
        """Process training data"""
        print("\n" + "="*50)
        print("PROCESSING TRAINING DATA")
        print("="*50)
        
        df = self.df_train.copy()
        
        # Compute outlier bounds from training data (will be used for test set too)
        # Using very conservative IQR multiplier (4.0) to detect fewer outliers
        # Only upper outliers (not lower) - removes extreme high values
        self.outlier_bounds = self.compute_outlier_bounds_from_train(
            df, 
            iqr_multiplier=3.0, 
            only_upper=True
        )
        
        # Remove outliers (drop rows instead of capping)
        df = self.apply_outlier_bounds(df, self.outlier_bounds)
        
        # Handle missing periods
        df = self.handle_missing_periods(df)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Merge datasets
        df = self.merge_datasets(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        # Fill any remaining NaN
        df = df.fillna(0)
        
        self.df_train_processed = df
        print(f"\nFinal training data shape: {df.shape}")
        return df
    
    def process_test(self):
        """
        Process test data for one-step iterative prediction.
        Only processes known features (no lag/rolling features).
        Lag features will be computed iteratively during prediction using previous predictions.
        """
        print("\n" + "="*50)
        print("PROCESSING TEST DATA (for one-step prediction)")
        print("="*50)
        
        df = self.df_test.copy()
        print(f"Initial test data shape: {df.shape}")
        
        # Note: Outlier bounds available but test data doesn't have sales to apply them to
        if hasattr(self, 'outlier_bounds'):
            print(f"Outlier bounds available for {len(self.outlier_bounds)} store-family combinations")
        
        # Create time features (known ahead of time)
        print("\nCreating time features for test data...")
        df = self.create_time_features(df)
        print(f"After time features: {df.shape}")
        
        # Merge datasets (known ahead of time: stores, oil, holidays, transactions)
        print("\nMerging datasets for test data...")
        df = self.merge_datasets(df)
        print(f"After merging: {df.shape}")
        
        # Note: Lag and rolling features will be computed iteratively during prediction
        # using previous predictions and historical training data
        print("\nNote: Lag and rolling features will be computed during iterative prediction")
        print("  - Day 1: Uses historical training data for lags")
        print("  - Day 2+: Uses previous day predictions + historical data for lags")
        
        # Encode categorical using saved encoders
        print("\nEncoding categorical variables for test data...")
        if hasattr(self, 'label_encoders'):
            df['family_encoded'] = self.label_encoders['family'].transform(df['family'].astype(str))
            df['state_encoded'] = self.label_encoders['state'].transform(df['state'].astype(str))
            df['city_encoded'] = self.label_encoders['city'].transform(df['city'].astype(str))
            
            type_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            df['type_encoded'] = df['type'].map(type_mapping)
            print("Categorical encoding complete")
        else:
            print("Warning: Label encoders not available")
        
        # Sort by date for iterative prediction
        df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
        
        # Fill any remaining NaN
        print("\nFilling remaining NaN values...")
        nan_before = df.isna().sum().sum()
        df = df.fillna(0)
        nan_after = df.isna().sum().sum()
        print(f"NaN values filled: {nan_before:,} -> {nan_after:,}")
        
        self.df_test_processed = df
        print(f"\nFinal test data shape: {df.shape}")
        print(f"Test data columns: {list(df.columns)}")
        print("\nTest data ready for iterative one-step prediction!")
        return df
    
    def save_processed_data(self):
        """Save processed data"""
        print("\n" + "="*50)
        print("SAVING PROCESSED DATA")
        print("="*50)
        
        # Save as Parquet (entire dataset, efficient format)
        print("Saving Parquet files (entire dataset)...")
        self.df_train_processed.to_parquet(
            self.output_dir / 'train_processed.parquet', 
            index=False,
            engine='pyarrow'
        )
        self.df_test_processed.to_parquet(
            self.output_dir / 'test_processed.parquet', 
            index=False,
            engine='pyarrow'
        )
        
        # Save CSV samples (first 1000 rows for inspection)
        print("Saving CSV samples (first 1000 rows)...")
        sample_size = 1000
        self.df_train_processed.head(sample_size).to_csv(
            self.output_dir / 'train_processed_sample.csv', 
            index=False
        )
        self.df_test_processed.head(sample_size).to_csv(
            self.output_dir / 'test_processed_sample.csv', 
            index=False
        )
        
        # Save label encoders as pickle (needed for encoding new data)
        if hasattr(self, 'label_encoders'):
            with open(self.output_dir / 'label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
        
        # Save outlier bounds as pickle (needed for applying to new data)
        if hasattr(self, 'outlier_bounds'):
            with open(self.output_dir / 'outlier_bounds.pkl', 'wb') as f:
                pickle.dump(self.outlier_bounds, f)
        
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("- train_processed.parquet (entire dataset)")
        print("- test_processed.parquet (entire dataset)")
        print("- train_processed_sample.csv (sample)")
        print("- test_processed_sample.csv (sample)")
        print("- label_encoders.pkl (for encoding new data)")
        print("- outlier_bounds.pkl (for applying to new data)")
    
    def run(self):
        """Run complete data preprocessing pipeline"""
        print("="*50)
        print("DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Process dates
        self.process_dates()
        
        # Process oil
        self.process_oil()
        
        # Process holidays
        self.process_holidays()
        
        # Process training data first (needed for test lag features)
        self.process_train()
        
        # Process test data (uses training data for lag features)
        self.process_test()
        
        # Save processed data
        self.save_processed_data()
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING COMPLETE!")
        print("="*50)


def main():
    """Main function"""
    preprocessor = DataPreprocessor(
        data_dir='data',
        output_dir='data/processed'
    )
    preprocessor.run()


if __name__ == '__main__':
    main()

