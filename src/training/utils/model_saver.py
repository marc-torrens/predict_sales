"""
Model Saver Utility
===================

Utilities for saving models, evaluations, and metadata in an organized structure.
Organizes outputs by model type and date.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime


class ModelSaver:
    """Handles saving models and evaluation results in an organized structure"""
    
    def __init__(self, base_model_dir):
        """
        Args:
            base_model_dir: Base directory for saving models
        """
        self.base_model_dir = Path(base_model_dir)
        self.base_model_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model_directory(self, model_type, timestamp=None):
        """
        Create directory structure: base_dir/model_type/YYYY-MM-DD_HHMMSS/
        
        Args:
            model_type: Type of model (e.g., 'xgboost', 'random_forest')
            timestamp: Optional timestamp string. If None, generates current timestamp
        
        Returns:
            Path to model directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create structure: model_type/YYYY-MM-DD_HHMMSS/
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_dir = self.base_model_dir / model_type / f"{date_str}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        return model_dir
    
    def save_model(self, model, model_type, model_dir, model_name=None):
        """
        Save model to file
        
        Args:
            model: Trained model object
            model_type: Type of model
            model_dir: Directory to save model
            model_name: Optional custom name (default: 'model')
        
        Returns:
            Path to saved model file
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            model_name = "model"
        
        if model_type == 'lightgbm':
            import lightgbm as lgb
            model_path = model_dir / f"{model_name}.txt"
            if hasattr(model, 'save_model'):
                model.save_model(str(model_path))
            else:
                # If it's a Booster object
                model.save_model(str(model_path))
        else:
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        return model_path
    
    def save_metadata(self, metadata, model_dir, filename="metadata.json"):
        """
        Save model metadata
        
        Args:
            metadata: Dictionary of metadata
            model_dir: Directory to save metadata
            filename: Name of metadata file
        
        Returns:
            Path to saved metadata file
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = model_dir / filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata_path
    
    def save_evaluation(self, metrics, model_dir, split_name="validation", filename=None):
        """
        Save evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
            model_dir: Directory to save evaluation
            split_name: Name of data split (e.g., 'validation', 'train')
            filename: Optional custom filename
        
        Returns:
            Path to saved evaluation file
        """
        model_dir = Path(model_dir)
        eval_dir = model_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{split_name}_metrics.json"
        
        eval_path = eval_dir / filename
        with open(eval_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return eval_path
    
    def save_feature_importance(self, feature_importance, model_dir, filename="feature_importance.csv"):
        """
        Save feature importance
        
        Args:
            feature_importance: DataFrame with feature importance
            model_dir: Directory to save feature importance
            filename: Name of feature importance file
        
        Returns:
            Path to saved feature importance file, or None if feature_importance is None
        """
        if feature_importance is None:
            return None
        
        model_dir = Path(model_dir)
        eval_dir = model_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        importance_path = eval_dir / filename
        feature_importance.to_csv(importance_path, index=False)
        
        return importance_path
    
    def save_shap_values(self, shap_analysis, model_dir):
        """
        Save SHAP values and summary
        
        Args:
            shap_analysis: Dictionary from analyze_shap_values
            model_dir: Directory to save SHAP values
        
        Returns:
            Path to SHAP directory, or None if shap_analysis is None
        """
        if shap_analysis is None:
            return None
        
        model_dir = Path(model_dir)
        shap_dir = model_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Save SHAP values if available
        if shap_analysis.get('shap_values') is not None:
            from evaluate.shap_values import save_shap_values
            
            save_shap_values(
                shap_analysis['shap_values'],
                shap_analysis['feature_names'],
                shap_dir / "shap_values",
                shap_analysis.get('X_sample')
            )
        
        # Save summary
        if shap_analysis.get('summary') is not None:
            summary_path = shap_dir / "shap_summary.csv"
            shap_analysis['summary'].to_csv(summary_path, index=False)
        
        return shap_dir
    
    def create_evaluation_summary(self, train_metrics, val_metrics, model_dir):
        """
        Create a summary of all evaluations
        
        Args:
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            model_dir: Directory to save summary
        
        Returns:
            Path to summary file
        """
        model_dir = Path(model_dir)
        eval_dir = model_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'train': train_metrics,
            'validation': val_metrics,
            'comparison': {
                'rmse_diff': val_metrics['rmse'] - train_metrics['rmse'],
                'mae_diff': val_metrics['mae'] - train_metrics['mae'],
                'r2_diff': val_metrics['r2'] - train_metrics['r2'],
                'overfitting_ratio': val_metrics['rmse'] / train_metrics['rmse'] if train_metrics['rmse'] > 0 else None
            }
        }
        
        summary_path = eval_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_path

