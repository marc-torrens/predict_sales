"""
Feature Importance
==================

Functions for extracting and analyzing feature importance.
"""

import pandas as pd


def get_feature_importance(model, feature_cols, model_type='xgboost'):
    """
    Get feature importance from model
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        model_type: Type of model
    
    Returns:
        DataFrame with feature importance
    """
    if model_type == 'xgboost':
        importance = model.feature_importances_
    elif model_type == 'lightgbm':
        importance = model.feature_importance(importance_type='gain')
    elif model_type == 'random_forest':
        importance = model.feature_importances_
    elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
        # Statistical models don't have feature importance in the same way
        # Return None or create a placeholder
        return None
    else:
        return None
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance


