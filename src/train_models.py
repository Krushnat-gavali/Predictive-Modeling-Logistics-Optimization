"""Train, evaluate, compare, and serialize delivery-time regression models.

Run from the project root with:
    py src\train_models.py

All paths are resolved relative to this file, not the current working directory.
Imputation and scaling are fitted only on the training split.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_NAME = "Predictive Modeling and Optimization in Logistics Systems Using Python"
MODULE_NAME = "src/train_models.py"
TARGET_COLUMN = "Time_taken(min)"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Identifiers and raw coordinates are excluded. The engineered distance feature
# is retained; invalid distances must be represented as missing, not zero.
EXCLUDED_NON_PREDICTORS = {
    "ID",
    "Delivery_person_ID",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    TARGET_COLUMN,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ModelTraining")


def get_project_paths() -> dict[str, Path]:
    """Resolve project input and output paths from this script's location."""
    root = Path(__file__).resolve().parent.parent
    evaluation_dir = root / "outputs" / "evaluation"
    models_dir = root / "models"
    return {
        "project_root": root,
        "input_csv": root / "data" / "processed" / "feature_engineered_delivery_data.csv",
        "models_dir": models_dir,
        "evaluation_dir": evaluation_dir,
        "comparison_csv": evaluation_dir / "model_comparison.csv",
        "predictions_csv": evaluation_dir / "model_predictions.csv",
        "summary_json": evaluation_dir / "model_training_summary.json",
        "best_model": models_dir / "best_delivery_time_model.joblib",
    }


def sanitize_for_json(value: Any) -> Any:
    """Convert common NumPy/pandas values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (pd.Series, pd.Index)):
        return sanitize_for_json(value.tolist())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.str_,)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the engineered CSV and validate its basic schema."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {path}\n"
            "Run src/data_preprocessing.py and src/feature_engineering.py first."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read input CSV {path}: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names detected: {duplicates}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Required target '{TARGET_COLUMN}' is missing. "
            f"Available columns: {frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError("Input dataset contains no records.")
    logger.info("Loaded dataset: %s rows x %s columns", f"{len(frame):,}", len(frame.columns))
    return frame


def prepare_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], int]:
    """Filter invalid targets and create a numeric, leakage-conscious feature matrix."""
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    target_values = target.to_numpy(dtype=float, na_value=np.nan)
    valid_target = np.isfinite(target_values) & (target_values > 0)
    removed_rows = int((~valid_target).sum())

    usable = frame.loc[valid_target].copy()
    y = pd.Series(target_values[valid_target], index=usable.index, name=TARGET_COLUMN, dtype=float)
    excluded = [
        column
        for column in usable.columns
        if column in EXCLUDED_NON_PREDICTORS and column != TARGET_COLUMN
    ]
    feature_columns = [
        column for column in usable.columns if column not in EXCLUDED_NON_PREDICTORS
    ]
    if not feature_columns:
        raise ValueError("No predictor columns remain after excluding identifiers and target.")
    if len(usable) < 50:
        raise ValueError(f"Only {len(usable)} valid target rows remain; at least 50 are required.")

    X = usable[feature_columns].copy()
    non_numeric = [
        column for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        raise ValueError(
            "Feature engineering should output numeric predictors only. "
            f"Non-numeric columns found: {non_numeric}"
        )

    # Normalize infinities to NaN so the training-only imputer handles them.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Prevent an all-missing feature from being silently dropped by versions
    # of SimpleImputer that do not preserve empty features. Such a feature has
    # no usable training information and is excluded explicitly and audibly.
    all_missing_features = [column for column in X.columns if X[column].isna().all()]
    if all_missing_features:
        logger.warning(
            "Excluding %d predictor(s) that are entirely missing: %s",
            len(all_missing_features),
            all_missing_features,
        )
        X = X.drop(columns=all_missing_features)
        feature_columns = [column for column in feature_columns if column not in all_missing_features]
        excluded.extend(all_missing_features)

    if X.shape[1] == 0:
        raise ValueError("No usable predictor features remain after validation.")
    logger.info(
        "Prepared %s observations and %s predictor features; excluded %s invalid target rows.",
        f"{len(X):,}",
        len(feature_columns),
        removed_rows,
    )
    return X, y, feature_columns, excluded, removed_rows


def build_model_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Create models with imputers/scaling inside each pipeline."""
    return {
        "Dummy Regressor (Mean Baseline)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", DummyRegressor(strategy="mean")),
            ]
        ),
        "Linear Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=16,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate unrounded held-out regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae_minutes": mae, "rmse_minutes": rmse, "r2_score": r2}


def slugify(name: str) -> str:
    """Convert a display name to a stable filesystem-safe filename stem."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def verify_artifacts(paths: list[Path]) -> None:
    """Ensure every expected output exists and is non-empty."""
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Expected output file was not created: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Output file is empty: {path}")


def run_training_pipeline() -> None:
    """Run data preparation, train/test split, model evaluation, and export."""
    logger.info("=" * 78)
    logger.info("LOGISTICS DELIVERY-TIME MODEL TRAINING PIPELINE")
    logger.info("=" * 78)

    paths = get_project_paths()
    root = paths["project_root"]
    input_path = paths["input_csv"]
    models_dir = paths["models_dir"]
    evaluation_dir = paths["evaluation_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", root)
    logger.info("Input dataset: %s", input_path)

    frame = load_dataset(input_path)
    original_rows, original_columns = frame.shape
    X, y, feature_names, excluded_columns, removed_target_rows = prepare_data(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    logger.info("Training rows: %s | Test rows: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    pipelines = build_model_pipelines()
    records: list[dict[str, Any]] = []
    predictions = pd.DataFrame(
        {
            "source_row_index": y_test.index,
            "actual_time_min": y_test.to_numpy(),
        }
    ).reset_index(drop=True)

    best_name = ""
    best_rmse = math.inf
    best_pipeline: Pipeline | None = None
    model_paths: dict[str, str] = {}

    for model_name, pipeline in pipelines.items():
        logger.info("Training %s ...", model_name)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        duration = time.perf_counter() - start

        predicted = np.asarray(pipeline.predict(X_test), dtype=float)
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"{model_name} produced non-finite predictions.")
        metrics = regression_metrics(y_test, predicted)

        model_path = models_dir / f"{slugify(model_name)}_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        model_paths[model_name] = str(model_path)

        records.append(
            {
                "model_name": model_name,
                **metrics,
                "training_time_seconds": float(duration),
                "model_path": str(model_path),
            }
        )
        predictions[f"pred_{slugify(model_name)}"] = predicted
        predictions[f"absolute_error_{slugify(model_name)}"] = np.abs(y_test.to_numpy() - predicted)

        logger.info(
            "%s | MAE %.4f min | RMSE %.4f min | R2 %.4f | train %.2fs",
            model_name,
            metrics["mae_minutes"],
            metrics["rmse_minutes"],
            metrics["r2_score"],
            duration,
        )

        if metrics["rmse_minutes"] < best_rmse:
            best_rmse = metrics["rmse_minutes"]
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    # Persist best estimator independently at the requested stable path.
    joblib.dump(best_pipeline, paths["best_model"])

    comparison = pd.DataFrame(records).sort_values("rmse_minutes", ascending=True)
    comparison.to_csv(paths["comparison_csv"], index=False)
    predictions.to_csv(paths["predictions_csv"], index=False)

    summary = {
        "project_name": PROJECT_NAME,
        "module_name": MODULE_NAME,
        "training_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_dataset_path": str(input_path),
        "target_variable": TARGET_COLUMN,
        "target_units": "minutes",
        "random_seed": RANDOM_STATE,
        "test_split_fraction": TEST_SIZE,
        "dataset_dimensions": {
            "original_row_count": int(original_rows),
            "original_column_count": int(original_columns),
            "removed_invalid_target_rows": int(removed_target_rows),
            "usable_row_count": int(len(X)),
            "training_row_count": int(len(X_train)),
            "testing_row_count": int(len(X_test)),
            "predictor_feature_count": int(len(feature_names)),
        },
        "features_used": feature_names,
        "excluded_columns": excluded_columns,
        "preprocessing": {
            "numeric_missing_and_infinite_values": "Converted to missing values; median imputation fitted within each training pipeline.",
            "scaling": "StandardScaler used in the Linear Regression pipeline only; fitted on training data.",
            "train_test_split": "80/20 with random_state=42.",
        },
        "model_selection_policy": (
            "Select the model with the lowest unrounded RMSE on the held-out 20% test split. "
            "This is an initial comparison; repeated tuning should use cross-validation on training data."
        ),
        "best_model": {
            "model_name": best_name,
            "selection_metric": "lowest_unrounded_test_rmse",
            "test_rmse_minutes": float(best_rmse),
            "artifact_path": str(paths["best_model"]),
        },
        "model_evaluations": records,
        "saved_artifacts": {
            "comparison_csv": str(paths["comparison_csv"]),
            "predictions_csv": str(paths["predictions_csv"]),
            "best_model_joblib": str(paths["best_model"]),
            "individual_models": model_paths,
        },
        "data_limitations_and_domain_notes": [
            "The input is the supplied Kaggle food-delivery dataset; its records are not described here as synthetic.",
            "Geographic distance is a straight-line geodesic estimate, not road-network distance.",
            "Real-time traffic, weather conditions, dispatch latency, and dynamic shipping costs are not available as measured fields in this dataset.",
            "The held-out test metrics describe performance on this dataset and do not guarantee performance in other locations or future conditions.",
        ],
    }
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(summary), handle, indent=2, allow_nan=False)

    expected_files = [
        paths["comparison_csv"],
        paths["predictions_csv"],
        paths["summary_json"],
        paths["best_model"],
        *[Path(value) for value in model_paths.values()],
    ]
    verify_artifacts(expected_files)

    # Read back tables and JSON; reload serialized pipelines to validate artifacts.
    pd.read_csv(paths["comparison_csv"])
    predictions_check = pd.read_csv(paths["predictions_csv"])
    if len(predictions_check) != len(y_test):
        raise RuntimeError("Prediction CSV read-back row count does not match test-set size.")
    with paths["summary_json"].open("r", encoding="utf-8") as handle:
        json.load(handle)
    for model_path in model_paths.values():
        joblib.load(model_path)
    joblib.load(paths["best_model"])

    print("\n" + "=" * 78)
    print("MODEL TRAINING AND EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Project root:             {root}")
    print(f"Input dataset:            {input_path}")
    print(f"Usable records:           {len(X):,}")
    print(f"Training / test rows:     {len(X_train):,} / {len(X_test):,}")
    print(f"Predictor count:          {len(feature_names)}")
    print("\nModel comparison (sorted by RMSE):")
    for record in comparison.to_dict(orient="records"):
        print(
            f"  {record['model_name']:<34} "
            f"MAE={record['mae_minutes']:.4f} min | "
            f"RMSE={record['rmse_minutes']:.4f} min | "
            f"R2={record['r2_score']:.4f}"
        )
    print("-" * 78)
    print(f"Best model:               {best_name}")
    print(f"Best test RMSE:           {best_rmse:.4f} minutes")
    print(f"Best model artifact:      {paths['best_model']}")
    print(f"Comparison CSV:           {paths['comparison_csv']}")
    print(f"Predictions CSV:          {paths['predictions_csv']}")
    print(f"Summary JSON:             {paths['summary_json']}")
    print(f"Artifacts verified:       {len(expected_files)} files")
    print("Pipeline status:          SUCCESS")
    print("=" * 78)


def main() -> None:
    """CLI entry point with a clear failure message."""
    try:
        run_training_pipeline()
    except Exception as exc:
        logger.critical("MODEL TRAINING PIPELINE FAILED: %s", exc, exc_info=True)
        print("\nMODEL TRAINING PIPELINE FAILED")
        print(f"Reason: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()