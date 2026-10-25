"""
Predictive Modeling and Optimization in Logistics Systems Using Python
======================================================================
Module: src/hyperparameter_tuning.py
Project Root: C:\Predictive_Logistics_Optimization (resolved dynamically)

Description:
    Production-grade hyperparameter optimization pipeline for food delivery time
    prediction using scikit-learn's RandomizedSearchCV and KFold cross-validation:
      - Dynamically resolves project root and prepares output directories before logging.
      - Dual logging configured to console and outputs/evaluation/hyperparameter_tuning.log.
      - Loads feature-engineered data from data/processed/feature_engineered_delivery_data.csv.
      - Validates schema, strips column whitespace, and detects duplicate column names.
      - Safely filters non-numeric, missing, and non-positive target (Time_taken(min)) values.
      - Excludes non-predictive identifiers and raw coordinates from predictors.
      - Drops entirely missing predictor columns without filling with arbitrary zeros.
      - Replaces positive and negative infinities in predictors with NaN.
      - Splits data into 80% training and 20% held-out test partitions (random_state=42).
      - Preserves original dataset row indices in the test predictions table.
      - Encapsulates median SimpleImputer inside a Pipeline with RandomForestRegressor,
        ensuring imputer statistics are fitted strictly on training folds.
      - Configures n_jobs=1 in RandomForestRegressor and n_jobs=-1 in RandomizedSearchCV
        to eliminate nested parallelism overhead.
      - Evaluates 12 hyperparameter candidates across 5-fold cross-validation (60 fits total)
        using negative root mean squared error.
      - Accurately converts negative scikit-learn scoring metrics into positive RMSE values:
          * mean test RMSE = -mean_test_score
          * std test RMSE = std_test_score
          * mean train RMSE = -mean_train_score
          * std train RMSE = std_train_score
      - Evaluates the best pipeline on the uncompromised held-out test set once.
      - Calculates MAE, RMSE, and R2 against a training-target median baseline.
      - Exports tuned model, results table, predictions, and strict JSON summaries (allow_nan=False).
      - Robust sanitize_for_strict_json() safely handles containers, pd.NA, pd.NaT, NaN, Inf.
      - Executes thorough multi-step post-save verification including dynamic sample size checks.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# Module Logger Instance
# -----------------------------------------------------------------------------
logger = logging.getLogger("HyperparameterTuning")

# -----------------------------------------------------------------------------
# Pipeline Constants & Hyperparameter Search Configuration
# -----------------------------------------------------------------------------
PROJECT_NAME: str = "Predictive Modeling and Optimization in Logistics Systems Using Python"
MODULE_NAME: str = "src/hyperparameter_tuning.py"
TARGET_COLUMN: str = "Time_taken(min)"

# Reference benchmark from initial baseline training
REFERENCE_RF_TEST_MAE: float = 5.8585
REFERENCE_RF_TEST_RMSE: float = 7.4693
REFERENCE_RF_TEST_R2: float = 0.3637

# Partition and reproducibility configurations
TEST_SPLIT_SIZE: float = 0.20
RANDOM_STATE_SEED: int = 42
CV_FOLDS_COUNT: int = 5
N_ITER_SEARCH: int = 12
TOTAL_CV_FITS: int = N_ITER_SEARCH * CV_FOLDS_COUNT

# Non-predictive identifier and raw coordinate columns
EXCLUDED_NON_PREDICTORS: list[str] = [
    "ID",
    "Delivery_person_ID",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

# Random Forest hyperparameter distributions
PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "regressor__n_estimators": [100, 150, 200, 250],
    "regressor__max_depth": [None, 10, 14, 16, 20, 24],
    "regressor__min_samples_split": [2, 4, 6, 8, 10],
    "regressor__min_samples_leaf": [1, 2, 3, 4],
    "regressor__max_features": [1.0, "sqrt", 0.8],
}


# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
def get_project_paths() -> dict[str, Path]:
    """
    Resolve all directory and artifact paths dynamically using pathlib.Path.

    Ensures standalone execution regardless of current working directory.

    Returns:
        dict[str, Path]: Absolute path mapping for project inputs and outputs.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    input_csv_path = project_root / "data" / "processed" / "feature_engineered_delivery_data.csv"
    models_dir = project_root / "models"
    eval_dir = project_root / "outputs" / "evaluation"

    tuned_model_path = models_dir / "tuned_random_forest_delivery_model.joblib"
    tuning_results_csv_path = eval_dir / "hyperparameter_tuning_results.csv"
    best_params_json_path = eval_dir / "best_hyperparameters.json"
    predictions_csv_path = eval_dir / "tuned_model_predictions.csv"
    summary_json_path = eval_dir / "hyperparameter_tuning_summary.json"
    log_file_path = eval_dir / "hyperparameter_tuning.log"

    return {
        "project_root": project_root,
        "input_csv_path": input_csv_path,
        "models_dir": models_dir,
        "eval_dir": eval_dir,
        "tuned_model_path": tuned_model_path,
        "tuning_results_csv_path": tuning_results_csv_path,
        "best_params_json_path": best_params_json_path,
        "predictions_csv_path": predictions_csv_path,
        "summary_json_path": summary_json_path,
        "log_file_path": log_file_path,
    }


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
def configure_logging(log_file_path: Path) -> None:
    """
    Configure dual logging to console (stdout) and the execution log file.

    Guarantees directories exist prior to opening the file handler.

    Args:
        log_file_path: Destination path for the log file.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# -----------------------------------------------------------------------------
# Strict JSON Serialization Sanitation
# -----------------------------------------------------------------------------
def sanitize_for_strict_json(data: Any) -> Any:
    """
    Recursively sanitize arbitrary data structures for strict JSON compliance.

    Handles:
      - None, pd.NA, pd.NaT -> None (null in JSON)
      - Python and NumPy integers -> int
      - Python and NumPy floats -> float (NaN, PosInf, NegInf -> None)
      - Python and NumPy booleans -> bool
      - NumPy ndarrays, Pandas Series, and Index objects -> lists of sanitized elements
      - Nested dictionaries and sequences (list, tuple, set)

    Evaluates container types before scalar evaluation to prevent ambiguous
    truth value errors on multi-element arrays or Series.

    Args:
        data: Arbitrary data structure.

    Returns:
        Any: JSON-compliant data structure compatible with json.dump(..., allow_nan=False).
    """
    if data is None:
        return None

    # Handle dictionaries
    if isinstance(data, dict):
        return {str(k): sanitize_for_strict_json(v) for k, v in data.items()}

    # Handle standard Python sequences
    if isinstance(data, (list, tuple, set)):
        return [sanitize_for_strict_json(item) for item in data]

    # Handle NumPy ndarrays
    if isinstance(data, np.ndarray):
        return [sanitize_for_strict_json(item) for item in data.tolist()]

    # Handle Pandas Series and Index objects
    if isinstance(data, (pd.Series, pd.Index)):
        return [sanitize_for_strict_json(item) for item in data.tolist()]

    # Handle pandas missing value sentinels safely
    if data is pd.NA or data is pd.NaT:
        return None

    # Handle booleans (must precede integer check since bool subclasses int)
    if isinstance(data, (bool, np.bool_)):
        return bool(data)

    # Handle integers
    if isinstance(data, (int, np.integer)):
        return int(data)

    # Handle floating point numbers
    if isinstance(data, (float, np.floating)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)

    # Safely test for pd.isna on remaining scalars
    try:
        if pd.isna(data):
            return None
    except Exception:
        pass

    return data


# -----------------------------------------------------------------------------
# Data Ingestion and Validation
# -----------------------------------------------------------------------------
def load_and_validate_dataset(input_csv_path: Path) -> pd.DataFrame:
    """
    Load feature-engineered dataset and verify schema readiness for tuning.

    Validation steps:
      1. Verify file exists, is a regular file, and is non-empty.
      2. Parse CSV using pandas.
      3. Strip whitespace from column names.
      4. Detect and reject duplicate column names.
      5. Verify the target column Time_taken(min) exists.

    Args:
        input_csv_path: Path to input CSV.

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    Raises:
        FileNotFoundError: If input file is missing.
        ValueError: If file is empty, malformed, or missing required target.
    """
    if not input_csv_path.exists():
        err_msg = (
            f"Input dataset file not found at: {input_csv_path}\n"
            "Please ensure 'src/feature_engineering.py' has executed successfully."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    if not input_csv_path.is_file():
        err_msg = f"Target path is not a regular file: {input_csv_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if input_csv_path.stat().st_size == 0:
        err_msg = f"Input dataset is empty (0 bytes): {input_csv_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("Loading feature-engineered dataset from %s ...", input_csv_path)
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as exc:
        err_msg = f"Failed to parse CSV file at {input_csv_path}: {exc}"
        logger.error(err_msg)
        raise ValueError(err_msg) from exc

    # Clean whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Detect duplicate column names
    col_list = df.columns.tolist()
    if len(col_list) != len(set(col_list)):
        duplicates = sorted({col for col in col_list if col_list.count(col) > 1})
        err_msg = f"Duplicate column names detected in input dataset: {duplicates}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Check target presence
    if TARGET_COLUMN not in df.columns:
        err_msg = (
            f"Required target column '{TARGET_COLUMN}' is missing from dataset.\n"
            f"Available columns: {list(df.columns)}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(
        "Successfully loaded dataset: %d rows x %d columns.",
        df.shape[0],
        df.shape[1],
    )
    return df


# -----------------------------------------------------------------------------
# Target and Predictor Separation (Leakage-Free)
# -----------------------------------------------------------------------------
def prepare_predictors_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], int, list[str]]:
    """
    Isolate predictor matrix X and target vector y with strict leakage prevention.

    Operations:
      - Convert target column to numeric safely (errors="coerce").
      - Exclude rows whose target is missing, non-finite, or non-positive.
      - Report original row count and count of excluded invalid target rows.
      - Exclude non-predictive identifiers and raw latitude/longitude columns.
      - Exclude target column from predictors.
      - Check that all remaining predictors are numeric; reject non-numeric features with
        a helpful error directing the user to the feature-engineering module.
      - Replace positive and negative infinity in predictors with missing values (np.nan).
      - Drop and report any predictor columns that are entirely missing.
      - Preserve source row indices throughout.

    Args:
        df: Input DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.Series, list[str], list[str], int, list[str]]:
            (X, y, predictor_names, excluded_columns, removed_target_rows, dropped_empty_columns)

    Raises:
        ValueError: If insufficient valid rows remain or if non-numeric predictors exist.
    """
    initial_rows = len(df)
    target_numeric = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # Positive, non-null, finite target criteria
    valid_target_mask = target_numeric.notna() & np.isfinite(target_numeric) & (target_numeric > 0)
    removed_target_rows = int((~valid_target_mask).sum())

    if removed_target_rows > 0:
        logger.warning(
            "Excluding %d rows with missing, non-numeric, non-positive, or infinite target values.",
            removed_target_rows,
        )
    else:
        logger.info("All %d rows contain valid, positive target values.", initial_rows)

    df_valid = df[valid_target_mask].copy()
    y = target_numeric[valid_target_mask].astype(float)

    if len(df_valid) < 100:
        err_msg = f"Insufficient valid records ({len(df_valid)}) for cross-validation and evaluation."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Exclude non-predictive columns and target
    drop_candidates = set(EXCLUDED_NON_PREDICTORS + [TARGET_COLUMN])
    actual_excluded = [col for col in df_valid.columns if col in drop_candidates and col != TARGET_COLUMN]

    candidate_predictors = [col for col in df_valid.columns if col not in drop_candidates]
    if not candidate_predictors:
        err_msg = "No predictor columns remain after filtering non-predictive columns."
        logger.error(err_msg)
        raise ValueError(err_msg)

    X = df_valid[candidate_predictors].copy()

    # Check for unexpected non-numeric (categorical/object/string) predictors
    non_numeric_cols = [
        col for col in candidate_predictors if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric_cols:
        err_msg = (
            f"Unexpected non-numeric predictor columns found: {non_numeric_cols}. "
            "All categorical features must be encoded numerically during the feature-engineering "
            "stage (e.g., via one-hot encoding in 'src/feature_engineering.py') before model tuning."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Convert positive and negative infinity in predictors to NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Inspect for entirely missing predictor columns and drop them cleanly
    dropped_empty_columns: list[str] = []
    for col in list(X.columns):
        if X[col].isna().all():
            dropped_empty_columns.append(col)
            X.drop(columns=[col], inplace=True)
            logger.warning("Dropped entirely missing predictor column: '%s'", col)

    final_predictor_names = list(X.columns)
    if not final_predictor_names:
        err_msg = "All predictor columns were empty and dropped. No usable predictors remain."
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(
        "Prepared %d predictor features across %d usable records (retained %d of %d original rows).",
        len(final_predictor_names),
        len(X),
        len(X),
        initial_rows,
    )
    if actual_excluded:
        logger.info("Excluded non-predictive columns: %s", actual_excluded)
    if dropped_empty_columns:
        logger.info("Dropped entirely missing predictor columns: %s", dropped_empty_columns)

    return X, y, final_predictor_names, actual_excluded, removed_target_rows, dropped_empty_columns


# -----------------------------------------------------------------------------
# Tuning Pipeline Construction
# -----------------------------------------------------------------------------
def build_tuning_pipeline(random_state: int = 42) -> Pipeline:
    """
    Construct base Pipeline combining median SimpleImputer and RandomForestRegressor.

    Design details:
      1. SimpleImputer(strategy='median') is encapsulated inside the Pipeline so
         imputation statistics are fitted strictly on each cross-validation training fold,
         preventing data leakage across folds and test sets.
      2. RandomForestRegressor uses n_jobs=1 during tuning to avoid nested parallelism
         with RandomizedSearchCV (which executes candidate folds in parallel with n_jobs=-1).
      3. The saved pipeline retains the fitted imputer, enabling direct inference on new data.

    Args:
        random_state: Reproducibility random seed.

    Returns:
        Pipeline: Scikit-learn Pipeline ready for hyperparameter search.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                random_state=random_state,
                n_jobs=1,  # n_jobs=1 prevents CPU resource thrashing with outer parallel CV
            ),
        ),
    ])
    return pipeline


# -----------------------------------------------------------------------------
# Evaluation Metrics Calculation
# -----------------------------------------------------------------------------
def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R2.

    Args:
        y_true: Ground truth target values (minutes).
        y_pred: Predicted target values (minutes).

    Returns:
        dict[str, float]: Performance metrics dictionary rounded to 4 decimals.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


# -----------------------------------------------------------------------------
# Main Hyperparameter Tuning Execution
# -----------------------------------------------------------------------------
def run_hyperparameter_tuning() -> None:
    """
    Orchestrate full hyperparameter tuning, evaluation, serialization, and verification.
    """
    # 1. Resolve Paths and Ensure Output Directories Exist
    paths = get_project_paths()
    project_root = paths["project_root"]
    input_csv_path = paths["input_csv_path"]
    models_dir = paths["models_dir"]
    eval_dir = paths["eval_dir"]
    log_file_path = paths["log_file_path"]

    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Configure dual logging before commencing operations
    configure_logging(log_file_path)

    logger.info("=" * 78)
    logger.info("LOGISTICS PREDICTIVE MODELING: HYPERPARAMETER TUNING PIPELINE")
    logger.info("=" * 78)
    logger.info("Project Name:             %s", PROJECT_NAME)
    logger.info("Project Root:             %s", project_root)
    logger.info("Input Dataset Path:       %s", input_csv_path)
    logger.info("Models Directory:         %s", models_dir)
    logger.info("Evaluation Directory:     %s", eval_dir)
    logger.info("Log File Destination:     %s", log_file_path)

    # 2. Ingestion and Data Validation
    df_raw = load_and_validate_dataset(input_csv_path)
    original_row_count, original_col_count = df_raw.shape

    (
        X,
        y,
        predictor_names,
        excluded_columns,
        removed_target_rows,
        dropped_empty_columns,
    ) = prepare_predictors_and_target(df_raw)

    # 3. Partition Data into Train and Held-Out Test Splits (80/20, seed=42)
    # The held-out test set is kept strictly separate from hyperparameter selection.
    logger.info(
        "Partitioning data into train/test splits (test_size=%.2f, random_state=%d)...",
        TEST_SPLIT_SIZE,
        RANDOM_STATE_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE_SEED,
    )

    train_row_count = len(X_train)
    test_row_count = len(X_test)
    logger.info(
        "Data partitioned: %d training samples (80%%) | %d held-out test samples (20%%).",
        train_row_count,
        test_row_count,
    )

    # 4. Construct Pipeline and Configure RandomizedSearchCV
    base_pipeline = build_tuning_pipeline(random_state=RANDOM_STATE_SEED)
    kfold = KFold(n_splits=CV_FOLDS_COUNT, shuffle=True, random_state=RANDOM_STATE_SEED)

    logger.info(
        "Configuring RandomizedSearchCV with %d iterations across %d KFold cross-validation folds.",
        N_ITER_SEARCH,
        CV_FOLDS_COUNT,
    )
    logger.info(
        "The search will perform %d candidates * %d folds = %d cross-validation fits, "
        "followed by a final refit on the training data.",
        N_ITER_SEARCH,
        CV_FOLDS_COUNT,
        TOTAL_CV_FITS,
    )
    logger.info("Hyperparameter search distributions: %s", PARAM_DISTRIBUTIONS)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=kfold,
        random_state=RANDOM_STATE_SEED,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    # 5. Execute RandomizedSearchCV Strictly on Training Data
    logger.info("Initiating RandomizedSearchCV on X_train (%d samples)...", train_row_count)
    t_start = time.perf_counter()
    search.fit(X_train, y_train)
    t_end = time.perf_counter()
    elapsed_tuning_time = round(t_end - t_start, 2)
    logger.info("Hyperparameter search completed in %.2f seconds.", elapsed_tuning_time)

    # 6. Extract Best Estimator and Cross-Validation Metrics
    best_pipeline: Pipeline = search.best_estimator_
    best_params_raw: dict[str, Any] = search.best_params_
    best_cv_neg_rmse: float = float(search.best_score_)
    best_cv_rmse: float = round(-best_cv_neg_rmse, 4)

    logger.info("Best Cross-Validation RMSE: %.4f minutes", best_cv_rmse)
    logger.info("Optimal Raw Hyperparameters: %s", best_params_raw)

    best_params_clean = {
        k.replace("regressor__", ""): v for k, v in best_params_raw.items()
    }

    # 7. Model Evaluation on Held-Out Test Set
    logger.info("Evaluating tuned model on uncompromised held-out test set (%d samples)...", test_row_count)
    y_test_pred = best_pipeline.predict(X_test)

    if len(y_test_pred) != test_row_count:
        err_msg = f"Prediction count ({len(y_test_pred)}) does not match test set size ({test_row_count})."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    if not np.all(np.isfinite(y_test_pred)):
        err_msg = "Model test predictions contain NaN or non-finite values."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    tuned_test_metrics = compute_regression_metrics(y_test, y_test_pred)
    logger.info(
        "Tuned Model Held-Out Test Metrics -> MAE: %.4f min | RMSE: %.4f min | R2: %.4f",
        tuned_test_metrics["MAE"],
        tuned_test_metrics["RMSE"],
        tuned_test_metrics["R2"],
    )

    # 8. Compute Training Target Median Baseline for Comparison
    median_train_target = float(np.median(y_train))
    y_baseline_pred = np.full(shape=test_row_count, fill_value=median_train_target, dtype=float)
    baseline_test_metrics = compute_regression_metrics(y_test, y_baseline_pred)
    logger.info(
        "Median Baseline Held-Out Test Metrics -> MAE: %.4f min | RMSE: %.4f min | R2: %.4f (train median = %.2f min)",
        baseline_test_metrics["MAE"],
        baseline_test_metrics["RMSE"],
        baseline_test_metrics["R2"],
        median_train_target,
    )

    logger.info(
        "Reference Initial Random Forest Baseline -> MAE: %.4f min | RMSE: %.4f min | R2: %.4f",
        REFERENCE_RF_TEST_MAE,
        REFERENCE_RF_TEST_RMSE,
        REFERENCE_RF_TEST_R2,
    )

    # 9. Save Tuned Model Pipeline
    tuned_model_path = paths["tuned_model_path"]
    logger.info("Saving tuned model pipeline to %s ...", tuned_model_path)
    joblib.dump(best_pipeline, tuned_model_path)

    # 10. Process and Save Cross-Validation Results Table
    cv_results_df = pd.DataFrame(search.cv_results_)

    # Correct scikit-learn negative RMSE score transformation:
    # mean test RMSE = -mean_test_score
    # std test RMSE = std_test_score (non-negative standard deviation)
    # mean train RMSE = -mean_train_score
    # std train RMSE = std_train_score
    cv_results_df["mean_test_rmse_minutes"] = (-cv_results_df["mean_test_score"]).round(4)
    cv_results_df["std_test_rmse_minutes"] = cv_results_df["std_test_score"].round(4)

    if "mean_train_score" in cv_results_df.columns:
        cv_results_df["mean_train_rmse_minutes"] = (-cv_results_df["mean_train_score"]).round(4)
        cv_results_df["std_train_rmse_minutes"] = cv_results_df["std_train_score"].round(4)

    # Sort candidates by mean test RMSE in ascending order (best candidate first)
    cv_results_df.sort_values(by="mean_test_rmse_minutes", ascending=True, inplace=True)
    cv_results_df.reset_index(drop=True, inplace=True)

    tuning_results_csv_path = paths["tuning_results_csv_path"]
    cv_results_df.to_csv(tuning_results_csv_path, index=False)
    logger.info("Saved hyperparameter tuning results table (%d candidates) to: %s", len(cv_results_df), tuning_results_csv_path)

    # 11. Save Best Hyperparameters JSON
    best_params_payload = {
        "best_hyperparameters": best_params_clean,
        "best_hyperparameters_raw": best_params_raw,
        "best_cross_validation_rmse_minutes": best_cv_rmse,
        "scoring_method": "neg_root_mean_squared_error",
        "cross_validation_folds": CV_FOLDS_COUNT,
        "randomized_search_iterations": N_ITER_SEARCH,
        "total_cross_validation_fits": TOTAL_CV_FITS,
        "random_state": RANDOM_STATE_SEED,
    }
    best_params_json_path = paths["best_params_json_path"]
    with open(best_params_json_path, "w", encoding="utf-8") as f_json:
        json.dump(sanitize_for_strict_json(best_params_payload), f_json, indent=4, allow_nan=False)
    logger.info("Saved best hyperparameters configuration to: %s", best_params_json_path)

    # 12. Save Held-Out Test Predictions CSV (Preserving source row indices)
    predictions_df = pd.DataFrame({
        "source_row_index": y_test.index,
        "actual_delivery_time_min": np.round(y_test.values, 4),
        "predicted_delivery_time_min": np.round(y_test_pred, 4),
        "absolute_error_min": np.round(np.abs(y_test.values - y_test_pred), 4),
    })
    predictions_csv_path = paths["predictions_csv_path"]
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.info("Saved held-out test predictions to: %s (%d rows)", predictions_csv_path, len(predictions_df))

    # 13. Save Comprehensive Hyperparameter Tuning Summary JSON
    summary_json_path = paths["summary_json_path"]
    summary_report = {
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_name": PROJECT_NAME,
        "module_name": MODULE_NAME,
        "input_dataset_path": str(input_csv_path),
        "target_column": TARGET_COLUMN,
        "target_units": "minutes",
        "dataset_summary": {
            "original_row_count": original_row_count,
            "original_column_count": original_col_count,
            "invalid_target_rows_excluded": removed_target_rows,
            "retained_row_count": len(X),
            "training_row_count": train_row_count,
            "testing_row_count": test_row_count,
            "predictor_feature_count": len(predictor_names),
        },
        "predictor_columns_used": predictor_names,
        "excluded_identifier_and_coordinate_columns": excluded_columns,
        "entirely_missing_predictor_columns_dropped": dropped_empty_columns,
        "tuning_configuration": {
            "search_strategy": "RandomizedSearchCV",
            "candidate_iterations": N_ITER_SEARCH,
            "cross_validation_folds": CV_FOLDS_COUNT,
            "total_cv_fits": TOTAL_CV_FITS,
            "scoring_metric": "neg_root_mean_squared_error",
            "parameter_distributions": PARAM_DISTRIBUTIONS,
            "random_state": RANDOM_STATE_SEED,
            "test_split_fraction": TEST_SPLIT_SIZE,
            "elapsed_tuning_time_seconds": elapsed_tuning_time,
        },
        "best_hyperparameters": best_params_clean,
        "best_cross_validation_rmse_minutes": best_cv_rmse,
        "tuned_model_held_out_test_metrics": {
            "mae_minutes": tuned_test_metrics["MAE"],
            "rmse_minutes": tuned_test_metrics["RMSE"],
            "r2_score": tuned_test_metrics["R2"],
        },
        "median_baseline_test_metrics": {
            "training_median_value_minutes": round(median_train_target, 4),
            "mae_minutes": baseline_test_metrics["MAE"],
            "rmse_minutes": baseline_test_metrics["RMSE"],
            "r2_score": baseline_test_metrics["R2"],
        },
        "reference_baseline_comparison": {
            "reference_initial_rf_mae_minutes": REFERENCE_RF_TEST_MAE,
            "reference_initial_rf_rmse_minutes": REFERENCE_RF_TEST_RMSE,
            "reference_initial_rf_r2": REFERENCE_RF_TEST_R2,
        },
        "output_artifact_paths": {
            "tuned_model_joblib": str(tuned_model_path),
            "hyperparameter_tuning_results_csv": str(tuning_results_csv_path),
            "best_hyperparameters_json": str(best_params_json_path),
            "tuned_model_predictions_csv": str(predictions_csv_path),
            "hyperparameter_tuning_summary_json": str(summary_json_path),
            "execution_log": str(log_file_path),
        },
        "dataset_limitations": [
            "Dataset represents historical delivery logs from the Kaggle food delivery dataset; it is empirical and not synthetic.",
            "delivery_distance_km represents straight-line geodesic distance computed via the Haversine formula, not road driving distance.",
            "Real-time traffic congestion, inclement weather conditions, kitchen preparation delays, and courier dispatch wait times are not captured as measured features.",
            "Missing feature values are imputed using median values computed strictly within training folds to avoid data leakage.",
            "Hyperparameter tuning was conducted strictly on training data folds without referencing the held-out test set.",
        ],
    }

    with open(summary_json_path, "w", encoding="utf-8") as f_json:
        json.dump(sanitize_for_strict_json(summary_report), f_json, indent=4, allow_nan=False)
    logger.info("Saved hyperparameter tuning summary JSON to: %s", summary_json_path)

    # 14. Output Verification (Comprehensive Validation Checks)
    logger.info("Executing comprehensive output artifact verification...")

    # Verification 1: Confirm all 5 primary artifacts and log file exist and are non-empty
    primary_artifacts = [
        tuned_model_path,
        tuning_results_csv_path,
        best_params_json_path,
        summary_json_path,
        predictions_csv_path,
        log_file_path,
    ]
    for art_path in primary_artifacts:
        if not art_path.exists():
            err_msg = f"Output verification check failed: File does not exist: {art_path}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        if art_path.stat().st_size == 0:
            err_msg = f"Output verification check failed: File is empty (0 bytes): {art_path}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    # Verification 2: Reload model with joblib.load and confirm it is a scikit-learn Pipeline
    reloaded_pipeline = joblib.load(tuned_model_path)
    if not isinstance(reloaded_pipeline, Pipeline):
        err_msg = f"Output verification check failed: Reloaded object is not a scikit-learn Pipeline ({type(reloaded_pipeline)})."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Verification 3: Read back tuning results CSV using pandas and verify row count and expected columns
    readback_results_df = pd.read_csv(tuning_results_csv_path)
    if len(readback_results_df) != N_ITER_SEARCH:
        err_msg = (
            f"Output verification check failed: Tuning results table has {len(readback_results_df)} "
            f"rows; expected {N_ITER_SEARCH} candidates."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    expected_results_cols = {"mean_test_rmse_minutes", "std_test_rmse_minutes", "params"}
    if not expected_results_cols.issubset(set(readback_results_df.columns)):
        err_msg = f"Output verification check failed: Results CSV missing expected columns: {expected_results_cols - set(readback_results_df.columns)}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Verification 4: Read back predictions CSV using pandas and verify row count and expected columns
    readback_pred_df = pd.read_csv(predictions_csv_path)
    if len(readback_pred_df) != test_row_count:
        err_msg = (
            f"Output verification check failed: Predictions table has {len(readback_pred_df)} "
            f"rows; expected exactly {test_row_count} test examples."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    expected_pred_cols = {"source_row_index", "actual_delivery_time_min", "predicted_delivery_time_min", "absolute_error_min"}
    if not expected_pred_cols.issubset(set(readback_pred_df.columns)):
        err_msg = f"Output verification check failed: Predictions CSV missing expected columns: {expected_pred_cols - set(readback_pred_df.columns)}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Verification 5: Parse both JSON files with json.load and check required keys
    with open(best_params_json_path, "r", encoding="utf-8") as f_in:
        parsed_best_params = json.load(f_in)
        required_best_params_keys = {"best_hyperparameters", "best_cross_validation_rmse_minutes", "scoring_method"}
        if not required_best_params_keys.issubset(set(parsed_best_params.keys())):
            err_msg = f"Output verification check failed: best_hyperparameters.json missing keys: {required_best_params_keys - set(parsed_best_params.keys())}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    with open(summary_json_path, "r", encoding="utf-8") as f_in:
        parsed_summary = json.load(f_in)
        required_summary_keys = {"tuned_model_held_out_test_metrics", "median_baseline_test_metrics", "tuning_configuration"}
        if not required_summary_keys.issubset(set(parsed_summary.keys())):
            err_msg = f"Output verification check failed: hyperparameter_tuning_summary.json missing keys: {required_summary_keys - set(parsed_summary.keys())}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    # Verification 6: Predict on dynamic sample size min(5, len(X_test)) and verify finite values
    sample_size = min(5, len(X_test))
    sample_test_features = X_test.iloc[:sample_size]
    sample_preds = reloaded_pipeline.predict(sample_test_features)
    if len(sample_preds) != sample_size or not np.all(np.isfinite(sample_preds)):
        err_msg = f"Output verification check failed: Reloaded model failed to predict finite values for {sample_size} test samples."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    logger.info("All verification checks passed successfully. Output artifacts verified and fully operational.")

    # 15. Terminal Execution Summary
    print("\n" + "=" * 78)
    print("HYPERPARAMETER TUNING PIPELINE: EXECUTION SUMMARY")
    print("=" * 78)
    print(f"Project Name:                {PROJECT_NAME}")
    print(f"Project Root:                {project_root}")
    print(f"Input Dataset:               {input_csv_path}")
    print(f"Dataset Records:             {len(X):,} retained rows ({removed_target_rows} invalid target rows excluded)")
    print(f"Predictor Features:          {len(predictor_names)} numerical columns")
    print(f"Data Partition:              {train_row_count:,} train rows | {test_row_count:,} test rows (80/20)")
    print(f"Search Method:               RandomizedSearchCV ({N_ITER_SEARCH} candidates, {CV_FOLDS_COUNT}-fold CV)")
    print(f"Total Fits Executed:         {TOTAL_CV_FITS} cross-validation fits + final training refit")
    print(f"Tuning Elapsed Time:         {elapsed_tuning_time:.2f} seconds")
    print("-" * 78)
    print("CROSS-VALIDATION & HELD-OUT TEST EVALUATION:")
    print(f"  - Best Cross-Validation RMSE: {best_cv_rmse:.4f} minutes")
    print(f"  - Tuned Model Test MAE:       {tuned_test_metrics['MAE']:.4f} minutes")
    print(f"  - Tuned Model Test RMSE:      {tuned_test_metrics['RMSE']:.4f} minutes")
    print(f"  - Tuned Model Test R2:        {tuned_test_metrics['R2']:.4f}")
    print(f"  - Median Baseline Test MAE:   {baseline_test_metrics['MAE']:.4f} minutes")
    print(f"  - Median Baseline Test RMSE:  {baseline_test_metrics['RMSE']:.4f} minutes")
    print(f"  - Median Baseline Test R2:    {baseline_test_metrics['R2']:.4f}")
    print("-" * 78)
    print("BEST HYPERPARAMETERS SELECTED:")
    for param_name, param_val in best_params_clean.items():
        print(f"  - {param_name:<26}: {param_val}")
    print("-" * 78)
    print(f"1. Tuned Model Artifact:      {tuned_model_path}")
    print(f"2. Tuning Results CSV:        {tuning_results_csv_path}")
    print(f"3. Best Parameters JSON:      {best_params_json_path}")
    print(f"4. Test Predictions CSV:      {predictions_csv_path}")
    print(f"5. Summary Report JSON:       {summary_json_path}")
    print(f"6. Execution Log File:        {log_file_path}")
    print("Artifact Verification:        ALL 6 ARTIFACTS VERIFIED NON-EMPTY & VALID")
    print("Pipeline Execution:           SUCCESS")
    print("=" * 78 + "\n")


def main() -> None:
    """Entry point for standalone command-line execution."""
    try:
        run_hyperparameter_tuning()
    except Exception as exc:
        logger.critical("HYPERPARAMETER TUNING PIPELINE FAILED: %s", exc, exc_info=True)
        print("\n" + "=" * 78)
        print("HYPERPARAMETER TUNING PIPELINE FAILED")
        print("=" * 78)
        print(f"Reason: {exc}")
        print("=" * 78 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()