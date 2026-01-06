# Data Preprocessing Script

## Overview

The `preprocess_data.py` script performs comprehensive data preprocessing to prepare the sales prediction dataset for modeling.

## Features

### Data Loading
- Loads all datasets: train, test, stores, oil, holidays, transactions

### Data Cleaning
- **Missing Values**: Handles missing oil prices (~3.5%) using forward/backward fill
- **Outliers**: Computes IQR bounds from training data and applies them to both train and test
- **Missing Periods**: Fills missing time periods in time series using interpolation

### Feature Engineering
- **Time Features**: year, month, day, dayofweek, dayofyear, week, quarter, is_weekend
- **Lag Features**: Sales lagged by 1, 7, 14 days (max 14 days for 15-day test set compatibility)
- **Rolling Features**: Rolling mean and std for windows of 7, 14 days
- **Holiday Features**: Flags for national, regional, and local holidays
- **Categorical Encoding**: Label encoding for family, state, city, store type

### Data Merging
- Merges stores information
- Merges oil prices
- Merges transactions
- Merges holiday information (national, regional, local)

## Usage

```bash
cd src
python preprocess_data.py
```

## Output

The script saves processed data to `../../data/processed/`:
- `train_processed.pkl` - Processed training data (pickle format)
- `test_processed.pkl` - Processed test data (pickle format)
- `train_processed.csv` - Processed training data (CSV format, for inspection)
- `test_processed.csv` - Processed test data (CSV format, for inspection)
- `label_encoders.pkl` - Saved label encoders for categorical variables
- `outlier_bounds.pkl` - Saved IQR bounds for outlier detection

## Processing Steps

1. **Load Data**: Load all raw datasets
2. **Process Dates**: Convert date columns to datetime
3. **Process Oil**: Handle missing oil prices
4. **Process Holidays**: Create holiday flags
5. **Process Training Data**:
   - Compute outlier bounds (IQR method)
   - Apply outlier bounds
   - Handle missing periods
   - Create time features
   - Merge datasets
   - Create lag features (1, 7, 14 days)
   - Create rolling features (7, 14 days)
   - Encode categorical variables
6. **Process Test Data**:
   - Create time features
   - Merge datasets
   - Extract lag/rolling features from training data
   - Encode categorical variables using saved encoders
7. **Save Data**: Save processed datasets and encoders

## Key Design Decisions

- **Max Lag/Window = 14 days**: Ensures compatibility with 15-day test set
- **Outlier bounds from training**: Prevents data leakage, ensures consistency
- **Lag features for test**: Extracted directly from training data (test has no sales)
- **Rolling features for test**: Computed from training data window before test period

## Notes

- The script handles large datasets efficiently using vectorized operations
- Missing period handling creates complete time series for each store-family combination
- All NaN values are filled with 0 after processing
- Outlier bounds are saved for potential future use on new data

## Customization

You can customize the preprocessing by modifying the `DataPreprocessor` class:
- Adjust outlier detection thresholds
- Change lag periods (max 14 days recommended)
- Modify rolling window sizes (max 14 days recommended)
- Add additional features


