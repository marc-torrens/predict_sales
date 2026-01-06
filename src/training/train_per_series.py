"""
Train Per Series Model
=======================

Trains separate models for each store-family combination.
This approach recognizes that each series may have unique patterns.
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


class PerSeriesModelTrainer:
    """Train separate models for each store-family combination"""
    
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
        
    def train_model(self, model_type='xgboost', validation_days=30, min_samples=100):
        """
        Train separate models for each store-family combination
        
        Args:
            model_type: Type of model to train
            validation_days: Days for validation
            min_samples: Minimum samples required to train a model
        """
        print("="*50)
        print("TRAINING PER SERIES MODELS")
        print("="*50)
        print("\nApproach: Separate model for each store-family combination")
        print("Total combinations: 54 stores × 33 families = 1,782 models\n")
        
        # Load data
        loader = DataLoader(self.processed_data_dir)
        df_train = loader.load_train_data()
        
        # Get all store-family combinations
        combinations = df_train.groupby(['store_nbr', 'family']).size().reset_index(name='count')
        combinations = combinations[combinations['count'] >= min_samples]
        
        print(f"Training models for {len(combinations)} store-family combinations")
        print(f"(Skipping {1782 - len(combinations)} with < {min_samples} samples)\n")
        
        all_train_metrics = []
        all_val_metrics = []
        
        # Train model for each combination
        for idx, (_, row) in enumerate(tqdm(combinations.iterrows(), total=len(combinations), desc="Training models")):
            store = row['store_nbr']
            family = row['family']
            
            # Get data for this combination
            series_data = df_train[
                (df_train['store_nbr'] == store) & 
                (df_train['family'] == family)
            ].copy()
            
            if len(series_data) < min_samples:
                continue
            
            # Create time split
            series_train, series_val = create_time_split(series_data, validation_days=validation_days)
            
            if len(series_train) == 0 or len(series_val) == 0:
                continue
            
            # Prepare features
            X_train, y_train, feature_cols = prepare_features(series_train)
            X_val, y_val, _ = prepare_features(series_val)
            
            # Train model for validation evaluation
            try:
                if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    params = create_model(model_type)
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,  # Fewer rounds for smaller datasets
                        valid_sets=[train_data, val_data],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20),
                            lgb.log_evaluation(period=0)  # Silent
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
                        num_boost_round=200,
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
                self.models[(store, family)] = final_model
                self.metrics[(store, family)] = {
                    'train': train_metrics,
                    'validation': val_metrics,
                    'n_train': len(series_train),
                    'n_val': len(series_val),
                    'n_all': len(X_all)
                }
                
                all_train_metrics.append(train_metrics)
                all_val_metrics.append(val_metrics)
                
            except Exception as e:
                print(f"\nError training model for store {store}, family {family}: {e}")
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
        model_name = f"per_series_{model_type}_{timestamp}"
        
        models_dir = self.model_dir / model_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for (store, family), model in tqdm(self.models.items(), desc="Saving models"):
            if model_type == 'lightgbm':
                model_path = models_dir / f"model_store_{store}_family_{family}.txt"
                model.save_model(str(model_path))
            else:
                model_path = models_dir / f"model_store_{store}_family_{family}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'approach': 'per_series',
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
            'per_series_metrics': {
                f"store_{store}_family_{family}": metrics
                for (store, family), metrics in self.metrics.items()
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
    
    parser = argparse.ArgumentParser(description='Train per-series models')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'random_forest', 'arima', 'arma', 'sarima', 'auto_arima'],
                       help='Model type to train')
    parser.add_argument('--validation-days', type=int, default=30)
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum samples required to train a model')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Processed data directory (default: <project_root>/data/processed)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model directory (default: <project_root>/models)')
    
    args = parser.parse_args()
    
    trainer = PerSeriesModelTrainer(
        processed_data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    trainer.train_model(
        model_type=args.model,
        validation_days=args.validation_days,
        min_samples=args.min_samples
    )


if __name__ == '__main__':
    main()

