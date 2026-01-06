"""
Evaluation Metrics
==================

Functions for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error) for non-zero actuals
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None
    }
    
    return metrics


def evaluate_model(model, X, y, model_type='xgboost'):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X: Features
        y: True target values
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'arima', 'arma', 'sarima', 'auto_arima')
    
    Returns:
        Dictionary of metrics and predictions
    """
    # Make predictions
    if model_type == 'lightgbm':
        y_pred = model.predict(X)
    elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
        # Statistical models predict n_periods ahead
        n_periods = len(X) if hasattr(X, '__len__') else 1
        y_pred = model.predict(X, n_periods=n_periods)
        # Ensure same length as y
        if len(y_pred) != len(y):
            if len(y_pred) > len(y):
                y_pred = y_pred[:len(y)]
            else:
                # Repeat last prediction if needed
                last_pred = y_pred[-1] if len(y_pred) > 0 else 0
                y_pred = np.concatenate([y_pred, np.full(len(y) - len(y_pred), last_pred)])
    else:
        y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred)
    
    return metrics, y_pred


