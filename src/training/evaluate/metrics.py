"""
Evaluation Metrics
==================

Functions for evaluating model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def calculate_grouped_metrics(y_true, y_pred, group_values, group_name):
    """
    Calculate error metrics by group (e.g., by store_nbr or family).

    Args:
        y_true: True target values
        y_pred: Predicted target values
        group_values: Group labels aligned with y_true/y_pred
        group_name: Name of grouping column for output

    Returns:
        DataFrame with grouped metrics and sample counts
    """
    y_pred = np.maximum(y_pred, 0)

    eval_df = pd.DataFrame(
        {
            group_name: group_values,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )

    rows = []
    for group, grp in eval_df.groupby(group_name):
        y_t = grp["y_true"].to_numpy()
        y_p = grp["y_pred"].to_numpy()
        non_zero_mask = y_t > 0

        mape = None
        if non_zero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100
            )

        rows.append(
            {
                group_name: group,
                "n_samples": int(len(grp)),
                "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
                "mae": float(mean_absolute_error(y_t, y_p)),
                "mape": mape,
                "mean_actual": float(np.mean(y_t)),
                "mean_pred": float(np.mean(y_p)),
                "bias": float(np.mean(y_p - y_t)),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


