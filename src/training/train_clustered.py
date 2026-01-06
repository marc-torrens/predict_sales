"""
Train Clustered Models
=======================

Trains models grouped by cluster or family.
This is a middle ground between single model and per-series models.
"""

from datetime import datetime
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from evaluate.metrics import evaluate_model
from models.model_factory import create_model
from utils.data_loader import DataLoader
from utils.data_splitter import create_time_split, prepare_features

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple iterator wrapper
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from models.statistical_models import STATSMODELS_AVAILABLE
except ImportError:
    STATSMODELS_AVAILABLE = False


class ClusteredModelTrainer:
    """Train models grouped by cluster, family, or store type"""
    
    def __init__(self, processed_data_dir=None, model_dir=None):
        """
        Args:
            processed_data_dir: Directory with processed train/test data.
                                If None, resolved as <project_root>/data/processed
            model_dir: Directory to save models. If None, resolved as <project_root>/models
        """
        base_dir = Path(__file__).resolve().parents[2]
        
        if processed_data_dir is None:
            self.processed_data_dir = base_dir / 'data' / 'processed'
        else:
            self.processed_data_dir = Path(processed_data_dir)
        
        if model_dir is None:
            self.model_dir = base_dir / 'models'
        else:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metrics = {}
        
    def train_model(self, group_by='cluster', model_type='xgboost', validation_days=30, min_samples=500):
        """
        Train models grouped by cluster, family, or store type
        
        Args:
            group_by: 'cluster', 'family', 'store_type', or 'state'
            model_type: Type of model to train
            validation_days: Days for validation
            min_samples: Minimum samples required to train a model
        """
        print("="*50)
        print(f"TRAINING CLUSTERED MODELS (grouped by {group_by})")
        print("="*50)
        
        # Load data
        loader = DataLoader(self.processed_data_dir)
        df_train = loader.load_train_data()
        
        # Get groups
        if group_by == 'cluster':
            groups = df_train.groupby('cluster')
            print("\nApproach: One model per cluster")
        elif group_by == 'family':
            groups = df_train.groupby('family')
            print("\nApproach: One model per product family")
        elif group_by == 'store_type':
            groups = df_train.groupby('type')
            print("\nApproach: One model per store type")
        elif group_by == 'state':
            groups = df_train.groupby('state')
            print("\nApproach: One model per state")
        else:
            raise ValueError(f"Unknown group_by: {group_by}. Choose from: cluster, family, store_type, state")
        
        print(f"Total groups: {groups.ngroups}\n")
        
        all_train_metrics = []
        all_val_metrics = []
        
        # Train model for each group
        for group_name, group_data in tqdm(groups, desc=f"Training {group_by} models"):
            if len(group_data) < min_samples:
                print(f"Skipping {group_by}={group_name} (only {len(group_data)} samples)")
                continue
            
            # Create time split
            group_train, group_val = create_time_split(group_data, validation_days=validation_days)
            
            if len(group_train) == 0 or len(group_val) == 0:
                continue
            
            # Prepare features
            X_train, y_train, feature_cols = prepare_features(group_train)
            X_val, y_val, _ = prepare_features(group_val)
            
            # Train model for validation evaluation
            try:
                if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    params = create_model(model_type)
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=300,
                        valid_sets=[train_data, val_data],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=30),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
                    model = create_model(model_type)
                    model.fit(X_train, y_train)
                else:
                    model = create_model(model_type)
                    
                    if model_type == 'xgboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train)
                
                # Evaluate on validation set
                train_metrics, _ = evaluate_model(model, X_train, y_train, model_type)
                val_metrics, _ = evaluate_model(model, X_val, y_val, model_type)
                
                # Retrain on ALL data (train + validation) for final model
                X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
                y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
                
                if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    params = create_model(model_type)
                    all_data = lgb.Dataset(X_all, label=y_all)
                    final_model = lgb.train(
                        params,
                        all_data,
                        num_boost_round=300,
                        callbacks=[lgb.log_evaluation(period=0)]
                    )
                elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
                    final_model = create_model(model_type)
                    final_model.fit(X_all, y_all)
                else:
                    final_model = create_model(model_type)
                    if model_type == 'xgboost':
                        final_model.fit(X_all, y_all, verbose=False)
                    else:
                        final_model.fit(X_all, y_all)
                
                # Store final model (trained on all data) and validation metrics
                self.models[group_name] = final_model
                self.metrics[group_name] = {
                    'train': train_metrics,
                    'validation': val_metrics,
                    'n_train': len(group_train),
                    'n_val': len(group_val),
                    'n_all': len(X_all)
                }
                
                all_train_metrics.append(train_metrics)
                all_val_metrics.append(val_metrics)
                
            except Exception as e:
                print(f"\nError training model for {group_by}={group_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate aggregate metrics
        print("\n" + "="*50)
        print("AGGREGATE METRICS")
        print("="*50)
        
        avg_train_rmse = np.mean([m['rmse'] for m in all_train_metrics])
        avg_val_rmse = np.mean([m['rmse'] for m in all_val_metrics])
        avg_train_mae = np.mean([m['mae'] for m in all_train_metrics])
        avg_val_mae = np.mean([m['mae'] for m in all_val_metrics])
        
        print(f"\nModels trained: {len(self.models)}")
        print("\nAverage Training Metrics:")
        print(f"  RMSE: {avg_train_rmse:.2f}")
        print(f"  MAE: {avg_train_mae:.2f}")
        print("\nAverage Validation Metrics:")
        print(f"  RMSE: {avg_val_rmse:.2f}")
        print(f"  MAE: {avg_val_mae:.2f}")
        
        # Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"clustered_{group_by}_{model_type}_{timestamp}"
        
        models_dir = self.model_dir / model_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for group_name, model in tqdm(self.models.items(), desc="Saving models"):
            if model_type == 'lightgbm':
                model_path = models_dir / f"model_{group_by}_{group_name}.txt"
                model.save_model(str(model_path))
            else:
                model_path = models_dir / f"model_{group_by}_{group_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'approach': f'clustered_by_{group_by}',
            'timestamp': timestamp,
            'n_models': len(self.models),
            'aggregate_metrics': {
                'train': {
                    'rmse': float(avg_train_rmse),
                    'mae': float(avg_train_mae)
                },
                'validation': {
                    'rmse': float(avg_val_rmse),
                    'mae': float(avg_val_mae)
                }
            },
            'per_group_metrics': {
                str(group_name): metrics
                for group_name, metrics in self.metrics.items()
            }
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nModels saved to: {models_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return self.models, models_dir, metadata_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train clustered models')
    parser.add_argument('--group-by', type=str, default='cluster',
                       choices=['cluster', 'family', 'store_type', 'state'],
                       help='Group models by this attribute')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'random_forest', 'arima', 'arma', 'sarima', 'auto_arima'],
                       help='Model type to train')
    parser.add_argument('--validation-days', type=int, default=30)
    parser.add_argument('--min-samples', type=int, default=500)
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Processed data directory (default: <project_root>/data/processed)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model directory (default: <project_root>/models)')
    
    args = parser.parse_args()
    
    trainer = ClusteredModelTrainer(
        processed_data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    trainer.train_model(
        group_by=args.group_by,
        model_type=args.model,
        validation_days=args.validation_days,
        min_samples=args.min_samples
    )


if __name__ == '__main__':
    main()

