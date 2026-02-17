"""
Generate Kaggle Submission
==========================

Loads a trained model, runs iterative prediction on the processed test set,
and writes a Kaggle-ready submission CSV with columns: id, sales.
"""

from datetime import datetime
from pathlib import Path
import argparse
import pickle

import pandas as pd

from predict_iterative import IterativePredictor


def load_model(model_path: Path, model_type: str):
    """Load model based on model type."""
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(model_path))

    # xgboost / random_forest and others saved as pickle in this project.
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission from a trained model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model (e.g., models/lightgbm/<run>/model.txt)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "random_forest", "arima", "arma", "sarima", "auto_arima"],
        help="Model type used for loading the model file.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default=None,
        help="Directory containing train_processed.parquet and test_processed.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV path. If not set, uses submissions/submission_<model>_<timestamp>.csv",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    model_path = Path(args.model_path).expanduser().resolve()
    processed_data_dir = (
        Path(args.processed_data_dir).expanduser().resolve()
        if args.processed_data_dir
        else project_root / "data" / "processed"
    )

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
    else:
        out_dir = project_root / "submissions"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"submission_{args.model_type}_{ts}.csv"

    print(f"Loading model: {model_path}")
    model = load_model(model_path, args.model_type)

    print(f"Using processed data from: {processed_data_dir}")
    predictor = IterativePredictor(processed_data_dir=processed_data_dir)
    preds = predictor.predict_iteratively(model)

    submission = preds[["id", "sales"]].copy()
    submission["id"] = submission["id"].astype(int)
    submission["sales"] = submission["sales"].clip(lower=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print("\nSubmission generated successfully")
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(submission)}")
    print("\nPreview:")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()


