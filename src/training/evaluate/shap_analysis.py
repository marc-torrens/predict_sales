"""
SHAP Analysis Script
====================

Standalone script for calculating and analyzing SHAP values for trained models.
Can be run independently or called from training scripts.
"""

import sys
import argparse
from pathlib import Path
import pickle
import json
import pandas as pd

# Add training directory to path for imports
script_dir = Path(__file__).resolve().parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from evaluate.shap_values import analyze_shap_values, SHAP_AVAILABLE
from utils.data_loader import DataLoader
from utils.data_splitter import prepare_features


def load_model(model_path, model_type):
    """Load a trained model from file"""
    model_path = Path(model_path)
    
    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    return model


def analyze_model_shap(model_path, metadata_path=None, data_dir=None, 
                       sample_size=1000, output_dir=None, use_test_set=False):
    """
    Analyze SHAP values for a trained model
    
    Args:
        model_path: Path to trained model file
        metadata_path: Path to model metadata JSON (optional)
        data_dir: Directory with processed data
        sample_size: Number of samples to use for SHAP calculation
        output_dir: Directory to save SHAP outputs
        use_test_set: If True, use test set (if available). If False, use validation set.
                     Note: The final model was trained on train+validation, so validation
                     set is technically in the training data. For truly independent SHAP
                     analysis, use test set or a separate holdout set.
    """
    if not SHAP_AVAILABLE:
        print("ERROR: SHAP is not installed. Install with: pip install shap")
        return
    
    print("="*50)
    print("SHAP ANALYSIS")
    print("="*50)
    
    # Load metadata if available
    model_type = None
    feature_cols = None
    
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        model_type = metadata.get('model_type', 'xgboost')
        feature_cols = metadata.get('feature_columns', None)
        print(f"Loaded metadata: model_type={model_type}")
    else:
        # Try to infer from filename
        model_path_str = str(model_path)
        if 'xgboost' in model_path_str.lower():
            model_type = 'xgboost'
        elif 'lightgbm' in model_path_str.lower() or 'lgb' in model_path_str.lower():
            model_type = 'lightgbm'
        elif 'random_forest' in model_path_str.lower() or 'rf' in model_path_str.lower():
            model_type = 'random_forest'
        else:
            model_type = 'xgboost'  # Default
        print(f"Inferred model_type={model_type} from filename")
    
    # Check if model type supports SHAP
    if model_type in ['arima', 'arma', 'sarima', 'auto_arima']:
        print("SHAP values are not available for statistical time series models.")
        return
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, model_type)
    print("Model loaded successfully.")
    
    # Load data for SHAP calculation
    if data_dir is None:
        base_dir = Path(__file__).resolve().parents[3]
        data_dir = base_dir / 'data' / 'processed'
    else:
        data_dir = Path(data_dir)
    
    loader = DataLoader(data_dir)
    
    if use_test_set:
        # Try to load test set (if it has target values, which is unlikely)
        print(f"\nAttempting to load test set from: {data_dir}")
        try:
            df_test = loader.load_test_data()
            # Test set typically doesn't have 'sales' column, so we can't use it for SHAP
            # that requires target values for context
            if 'sales' not in df_test.columns:
                print("WARNING: Test set does not have target values. Falling back to validation set.")
                use_test_set = False
            else:
                df_shap = df_test.copy()
                data_source = "test set"
        except Exception as e:
            print(f"Could not load test set: {e}")
            print("Falling back to validation set.")
            use_test_set = False
    
    if not use_test_set:
        # Use validation set (last 30 days of training data)
        # NOTE: This data was used in final model training, so SHAP values are
        # for interpretability purposes, not for truly independent evaluation
        print(f"\nLoading training data from: {data_dir}")
        print("NOTE: Using validation set for SHAP calculation.")
        print("      The final model was trained on train+validation, so this data")
        print("      was seen during training. For truly independent SHAP analysis,")
        print("      use a separate test set with --use-test-set flag.")
        
        df_train = loader.load_train_data()
        
        # Use last 30 days as validation set (same as training)
        df_train = df_train.sort_values('date').reset_index(drop=True)
        max_date = df_train['date'].max()
        split_date = max_date - pd.Timedelta(days=30)
        df_shap = df_train[df_train['date'] >= split_date].copy()
        data_source = "validation set (last 30 days of training data)"
    
    print(f"\nUsing {data_source} for SHAP calculation")
    print(f"Data: {len(df_shap):,} samples")
    print(f"Date range: {df_shap['date'].min().date()} to {df_shap['date'].max().date()}")
    
    # Prepare features
    X_shap, y_shap, feature_cols_loaded = prepare_features(df_shap)
    
    if feature_cols is None:
        feature_cols = feature_cols_loaded
    
    print(f"Features: {len(feature_cols)}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(model_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    shap_output_dir = output_dir / f"{Path(model_path).stem}_shap"
    
    # Calculate SHAP values
    print("\n" + "="*50)
    print("CALCULATING SHAP VALUES")
    print("="*50)
    
    shap_analysis = analyze_shap_values(
        model, X_shap, y_shap, model_type=model_type,
        output_dir=shap_output_dir,
        sample_size=sample_size,
        create_plots=False
    )
    
    if shap_analysis is None:
        print("Failed to calculate SHAP values.")
        return
    
    # Save SHAP summary to metadata if metadata exists
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if shap_analysis.get('summary') is not None:
            metadata['shap_summary'] = shap_analysis['summary'].to_dict('records')
            metadata['shap_analysis_date'] = pd.Timestamp.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"\nUpdated metadata with SHAP summary: {metadata_path}")
    
    print("\n" + "="*50)
    print("SHAP ANALYSIS COMPLETE")
    print("="*50)
    print(f"SHAP values saved to: {shap_output_dir}")
    
    return shap_analysis


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Calculate SHAP values for a trained model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file (.pkl or .txt)')
    parser.add_argument('--metadata-path', type=str, default=None,
                       help='Path to model metadata JSON file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory with processed data (default: <project_root>/data/processed)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save SHAP outputs (default: same as model directory)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of samples to use for SHAP calculation (default: 1000)')
    parser.add_argument('--use-test-set', action='store_true',
                       help='Use test set for SHAP (if available). Default: use validation set.')
    
    args = parser.parse_args()
    
    analyze_model_shap(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        use_test_set=args.use_test_set
    )


if __name__ == '__main__':
    main()

