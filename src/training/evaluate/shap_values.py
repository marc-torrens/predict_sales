"""
SHAP Values Analysis
====================

Functions for calculating and analyzing SHAP (SHapley Additive exPlanations) values.
SHAP values explain the output of machine learning models by showing feature contributions.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP is not installed. SHAP values will not be available.")
    print("Install with: pip install shap")


def calculate_shap_values(model, X, model_type='xgboost', sample_size=None, max_samples=1000):
    """
    Calculate SHAP values for a model
    
    Args:
        model: Trained model
        X: Feature data (pandas DataFrame or numpy array)
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest')
        sample_size: Number of samples to use for SHAP calculation (None = use all)
        max_samples: Maximum samples to use (for performance)
    
    Returns:
        shap_values: SHAP values array
        shap_explainer: SHAP explainer object
        feature_names: List of feature names
    """
    if not SHAP_AVAILABLE:
        return None, None, None
    
    # Statistical models don't support SHAP
    if model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
        print("SHAP values are not available for statistical time series models.")
        return None, None, None
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_values = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_values = X
    
    # Sample data if too large (for performance)
    if sample_size is None:
        sample_size = min(len(X_values), max_samples)
    
    if len(X_values) > sample_size:
        print(f"Sampling {sample_size} rows for SHAP calculation (out of {len(X_values)} total)")
        indices = np.random.choice(len(X_values), size=sample_size, replace=False)
        X_sample = X_values[indices]
    else:
        X_sample = X_values
        indices = np.arange(len(X_values))
    
    try:
        # Create appropriate explainer based on model type
        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == 'lightgbm':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == 'random_forest':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            print(f"SHAP not supported for model type: {model_type}")
            return None, None, None
        
        # Handle multi-output case (shouldn't happen for regression, but just in case)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values, explainer, feature_names
    
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None, None


def get_shap_summary(shap_values, feature_names, X_sample):
    """
    Get summary statistics of SHAP values
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        X_sample: Sample feature data used for SHAP
    
    Returns:
        DataFrame with SHAP summary statistics
    """
    if shap_values is None:
        return None
    
    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Calculate mean SHAP values (can be positive or negative)
    mean_shap = shap_values.mean(axis=0)
    
    # Calculate standard deviation of SHAP values
    std_shap = shap_values.std(axis=0)
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'std_shap': std_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    return summary


def save_shap_values(shap_values, feature_names, output_path, X_sample=None):
    """
    Save SHAP values to files
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        output_path: Base path for output files
        X_sample: Sample feature data (optional, for summary plots)
    """
    if shap_values is None:
        return
    
    output_path = Path(output_path)
    
    # Save SHAP values as CSV
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_csv_path = output_path.with_suffix('.csv')
    shap_df.to_csv(shap_csv_path, index=False)
    print(f"SHAP values saved to: {shap_csv_path}")
    
    # Save summary statistics
    summary = get_shap_summary(shap_values, feature_names, X_sample)
    if summary is not None:
        summary_path = output_path.with_name(output_path.stem + '_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"SHAP summary saved to: {summary_path}")


def plot_shap_summary(shap_values, feature_names, X_sample, output_path=None):
    """
    Create SHAP summary plots (if matplotlib is available)
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        X_sample: Sample feature data
        output_path: Path to save plots (optional)
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        if output_path:
            output_path = Path(output_path)
            plot_path = output_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {plot_path}")
            plt.close()
        else:
            plt.show()
            plt.close()
    
    except ImportError:
        print("matplotlib not available. Skipping SHAP plots.")
    except Exception as e:
        print(f"Error creating SHAP plots: {e}")


def analyze_shap_values(model, X, y, model_type='xgboost', output_dir=None, 
                       sample_size=1000, create_plots=False):
    """
    Complete SHAP analysis: calculate, summarize, and save SHAP values
    
    Args:
        model: Trained model
        X: Feature data
        y: Target values (optional, for context)
        model_type: Type of model
        output_dir: Directory to save SHAP outputs
        sample_size: Number of samples for SHAP calculation
        create_plots: Whether to create SHAP plots
    
    Returns:
        Dictionary with SHAP analysis results
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not available. Install with: pip install shap")
        return None
    
    print("\n" + "="*50)
    print("CALCULATING SHAP VALUES")
    print("="*50)
    
    # Calculate SHAP values
    shap_values, explainer, feature_names = calculate_shap_values(
        model, X, model_type=model_type, sample_size=sample_size
    )
    
    if shap_values is None:
        return None
    
    # Get sample data used for SHAP
    if isinstance(X, pd.DataFrame):
        X_sample = X.iloc[:len(shap_values)] if len(X) > len(shap_values) else X
    else:
        X_sample = X[:len(shap_values)] if len(X) > len(shap_values) else X
    
    # Get summary
    summary = get_shap_summary(shap_values, feature_names, X_sample)
    
    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_path = output_dir / "shap_values"
        save_shap_values(shap_values, feature_names, base_path, X_sample)
        
        if create_plots:
            plot_shap_summary(shap_values, feature_names, X_sample, base_path)
    
    # Print top features
    if summary is not None:
        print("\nTop 10 features by mean absolute SHAP value:")
        print(summary.head(10).to_string(index=False))
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': feature_names,
        'summary': summary,
        'X_sample': X_sample
    }

