"""
Train Single Global Model
==========================

Trains a single model for all store-family combinations.
This is a baseline approach - all series share the same model.
"""

from datetime import datetime
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from evaluate.feature_importance import get_feature_importance
from evaluate.metrics import calculate_grouped_metrics, evaluate_model
from models.model_factory import create_model
from utils.data_loader import DataLoader
from utils.data_splitter import create_time_split, prepare_features
from utils.model_saver import ModelSaver

warnings.filterwarnings('ignore')

# Add training directory to path for imports when running as script
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


# Import SHAP analysis
try:
    from evaluate.shap_values import SHAP_AVAILABLE, analyze_shap_values
except ImportError:
    SHAP_AVAILABLE = False
    analyze_shap_values = None

# Check for ML library availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost is not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("WARNING: LightGBM is not installed. Install with: pip install lightgbm")

# Optional Bayesian optimization (scikit-optimize)
try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("INFO: scikit-optimize is not installed. Bayesian hyperparameter tuning will not be available.")
    print("      Install with: pip install scikit-optimize")


class SingleModelTrainer:
    """Train a single model for all store-family combinations"""
    
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
        
        # Initialize model saver for organized saving
        self.model_saver = ModelSaver(self.model_dir)

    def tune_hyperparameters(self, df_train, model_type='xgboost', cv_folds=3, n_iter=8, tune_sample_size=None):
        """
        Run time-series cross-validation to tune hyperparameters.
        Uses Bayesian optimization (BayesSearchCV) when available.
        Supports XGBoost, Random Forest, and statistical models.
        
        Args:
            df_train: Training dataframe
            model_type: Type of model to tune
            cv_folds: Number of CV folds
            n_iter: Number of iterations for Bayesian optimization
            tune_sample_size: If set, uses only the most recent N rows for tuning
        """
        if cv_folds <= 1:
            print("\nCV folds <= 1, skipping hyperparameter tuning.")
            return None

        # Statistical models use auto_arima for automatic tuning
        if model_type in ['arima', 'arma', 'sarima']:
            print(f"\nUsing AutoARIMA for {model_type} - automatic parameter selection.")
            return {'use_auto_arima': True}

        if model_type == 'auto_arima':
            print("\nAutoARIMA already performs automatic parameter selection.")
            return None

        if model_type not in ['xgboost', 'random_forest', 'lightgbm']:
            print(f"\nHyperparameter tuning not implemented for '{model_type}'.")
            print(f"Model type '{model_type}' will use default parameters.")
            return None

        print(f"\nRunning time-series CV with {cv_folds} folds for hyperparameter tuning ({model_type})...")

        # Sort data by date to preserve temporal order
        df_sorted = df_train.sort_values('date').reset_index(drop=True)
        X, y, feature_cols = prepare_features(df_sorted)

        # Optional downsampling for low-resource hyperparameter tuning.
        # We keep the most recent rows to preserve temporal relevance.
        if tune_sample_size is not None and tune_sample_size > 0 and len(X) > tune_sample_size:
            print(f"\nUsing tune sample size: {tune_sample_size:,} (from {len(X):,} rows)")
            X = X.tail(tune_sample_size).reset_index(drop=True)
            y = y.tail(tune_sample_size).reset_index(drop=True)
            print(f"Tuning data shape after sampling: X={X.shape}, y={y.shape}")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # LightGBM in this project uses `lgb.train` with params dict from create_model,
        # so it is not directly compatible with BayesSearchCV estimator API.
        # Use manual time-series CV grid search for LightGBM.
        if model_type == 'lightgbm' or not SKOPT_AVAILABLE:
            if model_type == 'lightgbm':
                print("\nUsing manual grid search for LightGBM.")
                print("BayesSearchCV is not compatible with current LightGBM factory in this project.\n")
            else:
                print("\nscikit-optimize (skopt) is not installed.")
                print("Install it with: pip install scikit-optimize")
                print("Falling back to a small manual grid search.\n")
            
            # Small fallback grid based on model type
            if model_type == 'xgboost':
                param_grid = [
                    {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300},
                    {'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 500},
                    {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 800},
                ]
            elif model_type == 'random_forest':
                param_grid = [
                    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
                    {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5},
                    {'n_estimators': 400, 'max_depth': 25, 'min_samples_split': 2},
                ]
            elif model_type == 'lightgbm':
                param_grid = [
                    {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 120},
                    {'num_leaves': 50, 'learning_rate': 0.05, 'n_estimators': 180},
                ]
            else:
                print(f"Fallback grid search not implemented for {model_type}")
                return None

            best_params = None
            best_score = float('inf')

            for i, params in enumerate(param_grid, 1):
                print(f"\nConfig {i}/{len(param_grid)}: {params}")
                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                    model_cv = create_model(model_type, **params)
                    
                    if model_type == 'xgboost':
                        model_cv.fit(
                            X_train_cv,
                            y_train_cv,
                            eval_set=[(X_val_cv, y_val_cv)],
                            verbose=False,
                        )
                    elif model_type == 'lightgbm':
                        params_dict = model_cv  # LightGBM returns params dict
                        train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                        val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                        model_cv = lgb.train(
                            params_dict,
                            train_data,
                            num_boost_round=params.get('n_estimators', 500),
                            valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
                        )
                    else:
                        model_cv.fit(X_train_cv, y_train_cv)

                    metrics_cv, _ = evaluate_model(model_cv, X_val_cv, y_val_cv, model_type)
                    fold_scores.append(metrics_cv['rmse'])
                    print(f"  Fold {fold}: RMSE={metrics_cv['rmse']:.2f}")

                mean_rmse = float(np.mean(fold_scores))
                print(f"  Mean CV RMSE: {mean_rmse:.2f}")

                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = params

            print(f"\nBest params from grid search: {best_params} with RMSE={best_score:.2f}")
            return best_params

        # Bayesian optimization using BayesSearchCV
        print("Using Bayesian optimization (BayesSearchCV) for hyperparameter tuning.")

        # Base estimator from our factory (will be overridden by search space params)
        base_estimator = create_model(model_type)

        # Define search spaces based on model type
        if model_type == 'xgboost':
            search_spaces = {
                'max_depth': Integer(4, 12),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(200, 1000),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'min_child_weight': Integer(1, 10),
            }
        elif model_type == 'random_forest':
            search_spaces = {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': ['sqrt', 'log2', None],
            }
        elif model_type == 'lightgbm':
            search_spaces = {
                'num_leaves': Integer(20, 100),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'feature_fraction': Real(0.6, 1.0),
                'bagging_fraction': Real(0.6, 1.0),
                'min_child_samples': Integer(10, 50),
            }
        else:
            print(f"Unknown model type for tuning: {model_type}")
            return None

        print(f"\nStarting Bayesian optimization with {cv_folds} CV folds and {n_iter} iterations...")
        print("="*70)
        print("WARNING: This can be memory-intensive. If process gets killed, try:")
        print("  - Reducing --cv-folds (e.g., --cv-folds 2)")
        print("  - Reducing --n-iter (e.g., --n-iter 5)")
        print("  - Or disable SHAP calculation if not needed")
        
        opt = BayesSearchCV(
            estimator=base_estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,  # Reduced to prevent crashes and memory issues
            verbose=1,  # Show progress
            refit=False,
            random_state=42,
        )

        # Fit with memory-efficient approach
        # Clear any cached data before fitting
        import gc
        gc.collect()
        
        opt.fit(X, y)
        
        # Clean up after fitting
        gc.collect()

        best_params = opt.best_params_
        best_rmse = -opt.best_score_
        print(f"\nBest params from Bayesian CV: {best_params} with RMSE={best_rmse:.2f}")
        return best_params

    def evaluate_model_performance(self, model, X_train, y_train, X_val, y_val,
                                   model_type='xgboost', feature_cols=None, calculate_shap=True,
                                   train_meta_df=None, val_meta_df=None):
        """
        Evaluate model performance on training and validation sets.
        Also calculates SHAP values on validation set.
        
        Args:
            model: Trained model (trained only on training set)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model
            feature_cols: List of feature column names
        
        Returns:
            Dictionary with:
                - train_metrics: Training set metrics
                - val_metrics: Validation set metrics
                - shap_analysis: SHAP analysis results (if calculated)
        """
        # Evaluate on validation set
        print("\n" + "="*50)
        print("VALIDATION SET EVALUATION")
        print("="*50)
        val_metrics, y_val_pred = evaluate_model(model, X_val, y_val, model_type)
        print(f"RMSE: {val_metrics['rmse']:.2f}")
        print(f"MAE: {val_metrics['mae']:.2f}")
        print(f"R²: {val_metrics['r2']:.4f}")
        if val_metrics.get('mape') is not None:
            print(f"MAPE: {val_metrics['mape']:.2f}%")

        # Grouped validation metrics (store and family)
        val_store_metrics = None
        val_family_metrics = None
        if val_meta_df is not None:
            val_store_metrics = calculate_grouped_metrics(
                y_val, y_val_pred, val_meta_df['store_nbr'].values, 'store_nbr'
            )
            val_family_metrics = calculate_grouped_metrics(
                y_val, y_val_pred, val_meta_df['family'].values, 'family'
            )
            print("\nTop 5 worst stores by validation RMSE:")
            print(val_store_metrics[['store_nbr', 'n_samples', 'rmse', 'mae']].head(5).to_string(index=False))
            print("\nTop 5 worst families by validation RMSE:")
            print(val_family_metrics[['family', 'n_samples', 'rmse', 'mae']].head(5).to_string(index=False))
        
        # Calculate SHAP values on validation set using model trained only on training set
        # This is the correct approach: validation set is truly unseen by this model
        # SHAP calculation can be memory-intensive, so it can be disabled if needed
        shap_analysis = None
        if calculate_shap and SHAP_AVAILABLE and analyze_shap_values is not None and model_type not in ['arima', 'arma', 'sarima', 'auto_arima']:
            print("\n" + "="*50)
            print("CALCULATING SHAP VALUES (on validation set)")
            print("="*50)
            print("Note: Using model trained only on training set.")
            print("      Validation set is truly unseen by this model.")
            
            # Calculate SHAP (will save later in organized structure)
            # Reduced sample_size to prevent OOM kills on systems with limited RAM
            # SHAP TreeExplainer can be memory-intensive
            shap_analysis = analyze_shap_values(
                model, X_val, y_val, model_type=model_type,
                output_dir=None,  # Will save in organized structure later
                sample_size=min(500, len(X_val)),  # Reduced from 1000 to 500 for memory efficiency
                create_plots=False
            )
            if shap_analysis:
                print("SHAP values calculated successfully.")
        
        # Evaluate on training set
        print("\n" + "="*50)
        print("TRAINING SET EVALUATION")
        print("="*50)
        train_metrics, y_train_pred = evaluate_model(model, X_train, y_train, model_type)
        print(f"RMSE: {train_metrics['rmse']:.2f}")
        print(f"MAE: {train_metrics['mae']:.2f}")
        print(f"R²: {train_metrics['r2']:.4f}")
        if train_metrics.get('mape') is not None:
            print(f"MAPE: {train_metrics['mape']:.2f}%")

        # Grouped training metrics (store and family)
        train_store_metrics = None
        train_family_metrics = None
        if train_meta_df is not None:
            train_store_metrics = calculate_grouped_metrics(
                y_train, y_train_pred, train_meta_df['store_nbr'].values, 'store_nbr'
            )
            train_family_metrics = calculate_grouped_metrics(
                y_train, y_train_pred, train_meta_df['family'].values, 'family'
            )
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'shap_analysis': shap_analysis,
            'grouped_metrics': {
                'train': {
                    'by_store': train_store_metrics,
                    'by_family': train_family_metrics,
                },
                'validation': {
                    'by_store': val_store_metrics,
                    'by_family': val_family_metrics,
                },
            },
        }

    def train_model_on_train_set(self, model_type='xgboost', X_train=None, y_train=None,
                                 X_val=None, y_val=None, extra_params=None):
        """
        Train model on training set only (for evaluation purposes)
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping/evaluation during training)
            y_val: Validation targets
            extra_params: Additional model parameters from hyperparameter tuning
        
        Returns:
            Trained model (trained only on training set)
        """
        if extra_params is None:
            extra_params = {}
        
        print(f"\nTraining {model_type} model on training set...")
        
        # Check library availability
        if model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
            try:
                from models.statistical_models import STATSMODELS_AVAILABLE
                if not STATSMODELS_AVAILABLE:
                    raise ImportError("statsmodels not available. Install with: pip install statsmodels")
            except ImportError:
                raise ImportError("statsmodels not available. Install with: pip install statsmodels")
        
        if model_type == 'lightgbm':
            params = create_model(model_type, **extra_params)
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if X_val is not None else None
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=extra_params.get('n_estimators', 500),
                valid_sets=[train_data, val_data] if val_data else [train_data],
                valid_names=['train', 'val'] if val_data else ['train'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50) if val_data else lgb.log_evaluation(period=100),
                    lgb.log_evaluation(period=100)
                ]
            )
        elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
            # Statistical models work directly with time series
            model = create_model(model_type, **extra_params)
            model.fit(X_train, y_train)
        else:
            model = create_model(model_type, **extra_params)
            
            if model_type == 'xgboost':
                # XGBoost 3.x API - early stopping is set via model parameters
                if X_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=100
                    )
                else:
                    model.fit(X_train, y_train, verbose=100)
            else:
                model.fit(X_train, y_train)
        
        return model

    def train_model_on_all_data(self, model_type='xgboost', X_all=None, y_all=None,
                                extra_params=None):
        """
        Train final model on all data (train + validation)
        
        Args:
            model_type: Type of model to train
            X_all: All features (train + validation combined)
            y_all: All targets (train + validation combined)
            extra_params: Additional model parameters from hyperparameter tuning
        
        Returns:
            Trained model (trained on all data)
        """
        if extra_params is None:
            extra_params = {}
        
        print(f"\nTraining final {model_type} model on all data (train + validation)...")
        
        # Check library availability
        if model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
            try:
                from models.statistical_models import STATSMODELS_AVAILABLE
                if not STATSMODELS_AVAILABLE:
                    raise ImportError("statsmodels not available. Install with: pip install statsmodels")
            except ImportError:
                raise ImportError("statsmodels not available. Install with: pip install statsmodels")
        
        if model_type == 'lightgbm':
            params = create_model(model_type, **extra_params)
            all_data = lgb.Dataset(X_all, label=y_all)
            
            final_model = lgb.train(
                params,
                all_data,
                num_boost_round=extra_params.get('n_estimators', 500),
                callbacks=[
                    lgb.log_evaluation(period=100)
                ]
            )
        elif model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
            # Statistical models work directly with time series
            final_model = create_model(model_type, **extra_params)
            final_model.fit(X_all, y_all)
        else:
            final_model = create_model(model_type, **extra_params)
            if model_type == 'xgboost':
                final_model.fit(
                    X_all, y_all,
                    verbose=100
                )
            else:
                final_model.fit(X_all, y_all)
        
        return final_model

    def train_model(self, model_type='xgboost', validation_days=30,
                    cv_folds=0, tune_hyperparams=False, n_iter=8,
                    calculate_shap=True, skip_final_retrain=False,
                    tune_sample_size=None):
        """Train a single global model"""
        print("="*50)
        print("TRAINING SINGLE GLOBAL MODEL")
        print("="*50)
        print("\nApproach: One model for all store-family combinations")
        print("The model learns patterns across all series simultaneously.\n")
        
        # Load data
        loader = DataLoader(self.processed_data_dir)
        df_train = loader.load_train_data()
        
        # Memory management: clear any cached data
        import gc
        gc.collect()

        # Optional hyperparameter tuning with time-series CV
        best_params = None
        if tune_hyperparams:
            best_params = self.tune_hyperparameters(
                df_train, 
                model_type=model_type, 
                cv_folds=cv_folds,
                n_iter=n_iter,
                tune_sample_size=tune_sample_size
            )
            if best_params and best_params.get('use_auto_arima'):
                model_type = 'auto_arima'
                best_params = None

        # Create time-based split for final train/validation evaluation
        df_train_split, df_val = create_time_split(df_train, validation_days=validation_days)
        
        # Prepare features
        X_train, y_train, feature_cols = prepare_features(df_train_split)
        X_val, y_val, _ = prepare_features(df_val)
        
        extra_params = best_params or {} if best_params else {}

        # Train model on training set only (for evaluation)
        model = self.train_model_on_train_set(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            extra_params=extra_params
        )
        
        # Evaluate model performance (including SHAP calculation)
        evaluation_results = self.evaluate_model_performance(
            model, X_train, y_train, X_val, y_val, model_type, feature_cols, calculate_shap,
            train_meta_df=df_train_split[['store_nbr', 'family']],
            val_meta_df=df_val[['store_nbr', 'family']],
        )
        train_metrics = evaluation_results['train_metrics']
        val_metrics = evaluation_results['val_metrics']
        shap_analysis = evaluation_results['shap_analysis']
        grouped_metrics = evaluation_results['grouped_metrics']
        
        # Retrain final model on ALL data (train + validation) unless disabled.
        # On low-RAM systems, concatenating + retraining can trigger OOM kills.
        if skip_final_retrain:
            print("\n" + "="*50)
            print("SKIPPING FINAL RETRAIN (LOW-RESOURCE MODE)")
            print("="*50)
            print("Keeping model trained on training split only.")
            final_model = model
        else:
            print("\n" + "="*50)
            print("RETRAINING FINAL MODEL ON ALL DATA")
            print("="*50)
            print("Combining training and validation sets for final model...")
            
            # Combine train and validation data
            X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
            y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
            
            print(f"Final training set size: {len(X_all):,} samples")
            
            # Train final model on all data
            final_model = self.train_model_on_all_data(
                model_type=model_type,
                X_all=X_all,
                y_all=y_all,
                extra_params=extra_params
            )
            
            print("Final model trained on all data.")
        
        # Create organized directory structure: model_type/YYYY-MM-DD_HHMMSS/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.model_saver.create_model_directory(model_type, timestamp)
        
        print(f"\nSaving model to organized directory: {model_dir}")
        
        # Feature importance from final model
        feature_importance = get_feature_importance(final_model, feature_cols, model_type)
        
        # Save model
        model_path = self.model_saver.save_model(final_model, model_type, model_dir, "model")
        
        # Save evaluations
        train_eval_path = self.model_saver.save_evaluation(train_metrics, model_dir, "train")
        val_eval_path = self.model_saver.save_evaluation(val_metrics, model_dir, "validation")
        summary_path = self.model_saver.create_evaluation_summary(train_metrics, val_metrics, model_dir)

        # Save grouped metrics by store/family for train/validation
        eval_dir = model_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        grouped_metric_paths = {
            'train_by_store': None,
            'train_by_family': None,
            'validation_by_store': None,
            'validation_by_family': None,
        }
        if grouped_metrics.get('train', {}).get('by_store') is not None:
            p = eval_dir / "train_metrics_by_store.csv"
            grouped_metrics['train']['by_store'].to_csv(p, index=False)
            grouped_metric_paths['train_by_store'] = str(p)
        if grouped_metrics.get('train', {}).get('by_family') is not None:
            p = eval_dir / "train_metrics_by_family.csv"
            grouped_metrics['train']['by_family'].to_csv(p, index=False)
            grouped_metric_paths['train_by_family'] = str(p)
        if grouped_metrics.get('validation', {}).get('by_store') is not None:
            p = eval_dir / "validation_metrics_by_store.csv"
            grouped_metrics['validation']['by_store'].to_csv(p, index=False)
            grouped_metric_paths['validation_by_store'] = str(p)
        if grouped_metrics.get('validation', {}).get('by_family') is not None:
            p = eval_dir / "validation_metrics_by_family.csv"
            grouped_metrics['validation']['by_family'].to_csv(p, index=False)
            grouped_metric_paths['validation_by_family'] = str(p)
        
        # Save feature importance
        importance_path = self.model_saver.save_feature_importance(feature_importance, model_dir)
        
        # Save SHAP values if calculated
        shap_dir = None
        shap_summary = None
        if shap_analysis is not None:
            shap_dir = self.model_saver.save_shap_values(shap_analysis, model_dir)
            if shap_analysis.get('summary') is not None:
                shap_summary = shap_analysis['summary'].to_dict('records')
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'model_path': str(model_path),
            'model_directory': str(model_dir),
            'timestamp': timestamp,
            'approach': 'single_global_model',
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'evaluation_files': {
                'train': str(train_eval_path),
                'validation': str(val_eval_path),
                'summary': str(summary_path),
                'feature_importance': str(importance_path) if importance_path else None,
                'train_by_store': grouped_metric_paths['train_by_store'],
                'train_by_family': grouped_metric_paths['train_by_family'],
                'validation_by_store': grouped_metric_paths['validation_by_store'],
                'validation_by_family': grouped_metric_paths['validation_by_family'],
            },
            'shap_directory': str(shap_dir) if shap_dir else None,
            'shap_summary': shap_summary
        }
        
        metadata_path = self.model_saver.save_metadata(metadata, model_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL SAVED SUCCESSFULLY")
        print("="*50)
        print(f"Model directory: {model_dir}")
        print(f"Model file: {model_path}")
        print("\nEvaluation Metrics:")
        print(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        print(f"  Validation MAE: {val_metrics['mae']:.2f}")
        print(f"  Validation R²: {val_metrics['r2']:.4f}")
        print("\nFiles saved:")
        print(f"  - Model: {model_path.name}")
        print(f"  - Metadata: {metadata_path.name}")
        print(f"  - Train metrics: {train_eval_path.name}")
        print(f"  - Validation metrics: {val_eval_path.name}")
        print(f"  - Evaluation summary: {summary_path.name}")
        if importance_path:
            print(f"  - Feature importance: {importance_path.name}")
        if shap_dir:
            print(f"  - SHAP values: {shap_dir.name}/")
        if skip_final_retrain:
            print("\nNote: The saved model was trained on TRAINING split only.")
            print("      (Final retrain on all data was skipped for low-resource mode.)")
        else:
            print("\nNote: The saved model was trained on ALL data (train + validation).")
        if shap_dir:
            print("      SHAP values were calculated on validation set (unseen by training model).")
        
        return final_model, model_path, metadata_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train single global model')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'random_forest', 'arima', 'arma', 'sarima', 'auto_arima'],
                       help='Model type to train')
    parser.add_argument('--validation-days', type=int, default=30,
                       help='Days for final time-based validation holdout')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Run hyperparameter tuning with time-series CV before final training')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of folds for time-series CV when tuning hyperparameters (only used with --tune-hyperparams, default: 3)')
    parser.add_argument('--n-iter', type=int, default=8,
                       help='Number of iterations for Bayesian hyperparameter optimization (only used with --tune-hyperparams, default: 8)')
    parser.add_argument('--tune-sample-size', type=int, default=None,
                       help='Rows to use for hyperparameter tuning (recent rows only, reduces RAM usage)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Processed data directory (default: <project_root>/data/processed)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model directory (default: <project_root>/models)')
    parser.add_argument('--no-shap', action='store_true',
                       help='Disable SHAP value calculation (reduces memory usage)')
    parser.add_argument('--skip-final-retrain', action='store_true',
                       help='Skip retraining on train+validation (faster, lower RAM)')
    
    args = parser.parse_args()
    
    trainer = SingleModelTrainer(
        processed_data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    trainer.train_model(
        model_type=args.model,
        validation_days=args.validation_days,
        cv_folds=args.cv_folds,
        tune_hyperparams=args.tune_hyperparams,
        n_iter=args.n_iter,
        calculate_shap=not args.no_shap,
        skip_final_retrain=args.skip_final_retrain,
        tune_sample_size=args.tune_sample_size
    )


if __name__ == '__main__':
    main()


