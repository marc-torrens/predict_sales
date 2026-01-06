"""
Evaluation Module
=================

Contains evaluation metrics and model evaluation functions.
"""

from .metrics import evaluate_model, calculate_metrics
from .feature_importance import get_feature_importance

__all__ = ['evaluate_model', 'calculate_metrics', 'get_feature_importance']






