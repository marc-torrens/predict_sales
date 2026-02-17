# Sales Prediction Project

This repository contains my end-to-end solution for the Kaggle competition
**Store Sales - Time Series Forecasting**:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

It is also my first personal repository built with Cursor, collaborating with
GPT-5.3 Codex as an AI coding assistant throughout the workflow.

The objective is to forecast daily sales for multiple stores and product families
in Ecuador using time-series and tabular features. The project compares three
training strategies:

- one **single global model** for all series,
- one model **per store-family series**,
- and **clustered models** that group similar series.

During experimentation, I found that large-scale per-series training with
XGBoost + hyperparameter tuning is very resource-intensive on low-memory
machines. Because of that, the codebase includes memory-aware options
(`--no-shap`, lower CV folds/iterations, and lighter training configurations).

This project also includes explainability (SHAP) to understand which features
drive predictions.

I had problems running the scripts because of the extense of the data. For me worked lightgbm without hyperparameter finetuning and without final training over all data

## Project Highlights

- End-to-end pipeline: preprocessing -> training -> iterative prediction
- Multiple model families: XGBoost, LightGBM, Random Forest, ARIMA variants
- Three scalable training strategies (global, per-series, clustered)
- Time-based validation to reduce data leakage risk
- Model artifacts saved with metadata and evaluation summaries

## Project Structure

```
.
├── data/                  # Raw and processed data
├── models/                # Saved trained models
├── src/                   # Source code
│   ├── data_processing/   # Data preprocessing scripts
│   │   ├── preprocess_data.py
│   │   └── README_preprocessing.md
│   ├── training/          # Model training scripts
│   │   ├── train_single_model.py    # Single global model
│   │   ├── train_per_series.py      # Per store-family model
│   │   ├── train_clustered.py       # Clustered models
│   │   ├── models/                  # Model definitions
│   │   ├── evaluate/                # Evaluation functions
│   │   └── utils/                   # Utility functions
│   ├── prediction/       # Prediction scripts
│   │   └── predict_iterative.py
│   └── Data_analysis/    # Exploratory data analysis notebooks
│       ├── EDA.ipynb
│       └── Store_clustering.ipynb
└── venv/                 # (Optional) Python virtual environment
```

## Quick Start

All commands below are run from the **project root**.

### 1. Preprocess Data

```bash
cd src/data_processing
python preprocess_data.py
```

### 2. Train Model

Choose one of three training approaches:

**Single Global Model** (recommended for baseline):

```bash
cd ../training
python train_single_model.py --model xgboost
```

**Per Series Model** (separate model for each store-family):

```bash
python train_per_series.py --model xgboost --min-samples 100
```

**Clustered Models** (grouped by cluster/family/type/state):

```bash
python train_clustered.py --group-by cluster --model xgboost
```

### 3. Make Predictions

```bash
cd ../prediction
python predict_iterative.py
```

## Workflow

1. **Data Preprocessing** (`src/data_processing/`)
   - Cleans and processes raw data
   - Creates features (time, lag, rolling)
   - Handles outliers and missing values
   - Outputs: `data/processed/train_processed.parquet`, `data/processed/test_processed.parquet`

2. **Model Training** (`src/training/`)
   - Trains one-step ahead prediction models
   - Supports multiple approaches: single model, per-series, or clustered
   - Supports XGBoost, LightGBM, Random Forest, and statistical models (ARIMA, AutoARIMA)
   - Optional hyperparameter tuning with time-series cross-validation
   - Time-based train/validation split
   - Outputs: `models/{model_type}/{timestamp}/` with organized structure

3. **Prediction** (`src/prediction/`)
   - Iterative one-step ahead prediction
   - Uses previous predictions for lag features
   - Generates predictions for all test dates
   - Outputs: `predictions.csv`

## Training Approaches

### Single Global Model (`src/training/train_single_model.py`)

- **Approach**: One model for all store-family combinations  
- **Pros**: Simple, learns shared patterns, fast training  
- **Cons**: May not capture series-specific patterns  
- **Use case**: Baseline, when series share similar patterns  

**Usage:**

```bash
cd src/training

# Basic usage
python train_single_model.py --model xgboost

# With hyperparameter tuning
python train_single_model.py --model xgboost --tune-hyperparams --cv-folds 3

# Memory-efficient (disable SHAP)
python train_single_model.py --model xgboost --no-shap

# Statistical model
python train_single_model.py --model auto_arima
```

### Per Series Model (`src/training/train_per_series.py`)

- **Approach**: Separate model for each store-family combination (up to 1,782 models)  
- **Pros**: Captures unique patterns per series  
- **Cons**: Many models, may overfit on small series, memory-intensive  
- **Use case**: When each series has distinct behavior and sufficient data  

**Usage:**

```bash
cd src/training
python train_per_series.py --model xgboost --min-samples 100
```

### Clustered Models (`src/training/train_clustered.py`)

- **Approach**: Models grouped by cluster, family, store type, or state  
- **Pros**: Balance between single and per-series approaches  
- **Cons**: Need to choose grouping wisely  
- **Use case**: When series can be meaningfully grouped  

**Usage:**

```bash
cd src/training

# By cluster
python train_clustered.py --group-by cluster --model xgboost

# By family
python train_clustered.py --group-by family --model xgboost

# By store type
python train_clustered.py --group-by store_type --model xgboost

# By state
python train_clustered.py --group-by state --model xgboost
```

## Model Types

All training scripts support:

- **XGBoost**: Gradient boosting (default)  
- **LightGBM**: Fast gradient boosting  
- **Random Forest**: Ensemble of decision trees  
- **ARIMA/ARMA/SARIMA**: Statistical time series models  
- **AutoARIMA**: Automatic ARIMA parameter selection  

## Requirements

Install dependencies (from project root, ideally inside a virtualenv like `venv/`):

```bash
# Core dependencies
pip install pandas numpy scikit-learn pyarrow

# For XGBoost
pip install xgboost

# For LightGBM
pip install lightgbm

# For statistical models
pip install statsmodels pmdarima

# Optional: for hyperparameter tuning
pip install scikit-optimize

# Optional: for SHAP values
pip install shap
```

## Data Flow

```
Raw Data (data/)
    ↓
Preprocessing (src/data_processing/)
    ↓
Processed Data (data/processed/)
    ↓
Training (src/training/)
    ↓
Trained Model (models/{model_type}/{timestamp}/)
    ↓
Prediction (src/prediction/)
    ↓
Predictions (predictions.csv)
```

## Output Structure

Models are saved in organized directories:

```
models/
└── {model_type}/
    └── {YYYYMMDD_HHMMSS}/
        ├── model.pkl (or .txt for LightGBM)
        ├── metadata.json
        ├── train_evaluation.json
        ├── validation_evaluation.json
        ├── evaluation_summary.json
        ├── feature_importance.csv
        └── shap_values/ (if SHAP was calculated)
```

## Which Training Approach to Use?

- **Single Model**: Start here for baseline. Good if series share patterns.  
- **Per Series**: Use if each series has unique characteristics and enough data (100+ samples per series).  
- **Clustered**: Good middle ground. Use cluster/family grouping if EDA shows patterns.  

## Notes

- All models use **time-based train/validation split** to avoid data leakage  
- Models are trained for **one-step ahead prediction** (predict next day)  
- Final models are trained on **all data** (train + validation) for best test performance  
- Hyperparameter tuning can be memory-intensive; use `--no-shap` and reduce `--cv-folds` if needed  



