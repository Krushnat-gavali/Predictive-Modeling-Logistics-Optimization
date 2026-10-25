"""
Logistics Delivery-Time Optimization and Sensitivity Analysis
=============================================================

Project:
    Predictive Modeling and Optimization in Logistics Systems Using Python

Purpose:
    Evaluate illustrative delivery-distance reduction scenarios and examine
    how the trained delivery-time prediction model responds to selected
    predictor values.

Important scientific limitations:
    - delivery_distance_km represents straight-line geographic distance,
      not verified road-network distance.
    - Scenario predictions are model-based estimates, not guaranteed outcomes.
    - Sensitivity analysis measures model response; it does not establish
      causation or represent realistic operational interventions.
    - The dataset does not directly establish the effects of traffic, weather,
      food preparation, or other unobserved operational factors.
    - Courier age and ratings are examined only for model sensitivity.
      They are not operational optimization targets.

Run from the project root:
    py src\\optimization_analysis.py

The project root is also resolved dynamically from this file's location.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Project paths and configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_delivery_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "tuned_random_forest_delivery_model.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optimization"

SCENARIO_CSV = OUTPUT_DIR / "optimization_scenario_summary.csv"
SENSITIVITY_CSV = OUTPUT_DIR / "feature_sensitivity_analysis.csv"
RECOMMENDATIONS_FILE = OUTPUT_DIR / "optimization_recommendations.txt"
SUMMARY_JSON = OUTPUT_DIR / "optimization_analysis_summary.json"
LOG_FILE = OUTPUT_DIR / "optimization_analysis.log"

SCENARIO_PLOT = OUTPUT_DIR / "predicted_delivery_time_scenarios.png"
SENSITIVITY_PLOT = OUTPUT_DIR / "feature_sensitivity_analysis.png"

TARGET_COLUMN = "Time_taken(min)"

PREDICTOR_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "is_geo_valid",
    "delivery_distance_km",
    "log1p_delivery_distance_km",
    "Type_of_order__Buffet",
    "Type_of_order__Drinks",
    "Type_of_order__Meal",
    "Type_of_order__Snack",
    "Type_of_vehicle__bicycle",
    "Type_of_vehicle__electric_scooter",
    "Type_of_vehicle__motorcycle",
    "Type_of_vehicle__scooter",
]

EXPECTED_DATA_COLUMNS = PREDICTOR_COLUMNS + [TARGET_COLUMN]

SENSITIVITY_FEATURES = [
    "delivery_distance_km",
    "Delivery_person_Ratings",
    "Delivery_person_Age",
]

DISTANCE_REDUCTION_PERCENTAGES = [5, 10, 15]

MAX_SAMPLE_SIZE = 5000
RANDOM_SEED = 42

SENSITIVITY_PERCENTILES = [
    ("Minimum", 0.00),
    ("25th percentile", 0.25),
    ("Median", 0.50),
    ("75th percentile", 0.75),
    ("Maximum", 1.00),
]

SCENARIO_COLUMNS = [
    "scenario",
    "distance_reduction_percent",
    "sample_size",
    "eligible_records",
    "ineligible_records",
    "baseline_mean_predicted_delivery_time_min",
    "scenario_mean_predicted_delivery_time_min",
    "mean_predicted_change_min",
    "mean_predicted_change_percent",
    "baseline_median_predicted_delivery_time_min",
    "scenario_median_predicted_delivery_time_min",
    "median_predicted_change_min",
    "minimum_predicted_change_min",
    "maximum_predicted_change_min",
    "interpretation",
]

SENSITIVITY_COLUMNS = [
    "feature",
    "feature_description",
    "test_type",
    "percentile_label",
    "percentile",
    "tested_value",
    "sample_size",
    "baseline_mean_predicted_delivery_time_min",
    "scenario_mean_predicted_delivery_time_min",
    "mean_predicted_change_min",
    "baseline_median_predicted_delivery_time_min",
    "scenario_median_predicted_delivery_time_min",
    "median_predicted_change_min",
    "interpretation",
]

EXPECTED_OUTPUT_ARTIFACTS = [
    SCENARIO_CSV,
    SENSITIVITY_CSV,
    RECOMMENDATIONS_FILE,
    SUMMARY_JSON,
    LOG_FILE,
    SCENARIO_PLOT,
    SENSITIVITY_PLOT,
]


# ============================================================================
# Logging and general utilities
# ============================================================================

def configure_logging() -> logging.Logger:
    """Configure file and console logging for this analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("optimization_analysis")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def require(condition: bool, message: str) -> None:
    """Raise an informative exception when a critical check fails."""
    if not condition:
        raise RuntimeError(message)


def safe_float(value: Any) -> float | None:
    """Convert a value to a finite float, returning None if invalid."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(converted):
        return None

    return converted


def json_safe(value: Any) -> Any:
    """Convert common Python and NumPy values to strict JSON-safe values."""
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        converted = float(value)
        return converted if math.isfinite(converted) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if pd.isna(value):
        return None

    return value


def write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without permitting NaN or Infinity values."""
    safe_payload = json_safe(payload)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            safe_payload,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")


def verify_json_readback(path: Path) -> dict[str, Any]:
    """Read a JSON artifact back and ensure it contains a JSON object."""
    require(path.is_file(), f"JSON output was not created: {path}")

    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    require(
        isinstance(result, dict),
        f"Expected a JSON object in {path.name}.",
    )

    return result


def verify_csv_readback(
    path: Path,
    expected_columns: list[str],
    expected_rows: int,
) -> pd.DataFrame:
    """
    Verify CSV existence, headers, and row count.

    A CSV with headers and zero records is valid when an analysis legitimately
    produces no eligible scenarios.
    """
    require(path.is_file(), f"CSV output was not created: {path}")

    result = pd.read_csv(path)

    missing_columns = [
        column
        for column in expected_columns
        if column not in result.columns
    ]

    require(
        not missing_columns,
        f"{path.name} is missing expected columns: {missing_columns}",
    )

    require(
        list(result.columns) == expected_columns,
        (
            f"{path.name} columns do not match the expected order. "
            f"Expected {expected_columns}; got {list(result.columns)}."
        ),
    )

    require(
        len(result) == expected_rows,
        (
            f"{path.name} row-count mismatch. "
            f"Expected {expected_rows}; got {len(result)}."
        ),
    )

    return result


def save_csv_with_headers(
    records: list[dict[str, Any]],
    path: Path,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Save records to CSV while preserving headers for empty results."""
    result = pd.DataFrame(records, columns=expected_columns)

    result.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    return verify_csv_readback(
        path=path,
        expected_columns=expected_columns,
        expected_rows=len(records),
    )


# ============================================================================
# Input and model validation
# ============================================================================

def load_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load and validate the engineered dataset."""
    require(
        DATA_FILE.is_file(),
        (
            f"Required dataset was not found: {DATA_FILE}. "
            "Run the preprocessing and feature-engineering stages first."
        ),
    )

    logger.info("Loading engineered dataset: %s", DATA_FILE)

    data = pd.read_csv(DATA_FILE)

    require(not data.empty, "The engineered dataset contains no records.")

    missing_columns = [
        column
        for column in EXPECTED_DATA_COLUMNS
        if column not in data.columns
    ]

    require(
        not missing_columns,
        (
            "The engineered dataset is missing required columns: "
            f"{missing_columns}"
        ),
    )

    # Select only the expected model inputs and the target. This prevents
    # unrelated columns from accidentally entering prediction inputs.
    data = data.loc[:, EXPECTED_DATA_COLUMNS].copy()

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    valid_target = (
        data[TARGET_COLUMN].notna()
        & np.isfinite(data[TARGET_COLUMN])
        & (data[TARGET_COLUMN] > 0)
    )

    invalid_target_count = int((~valid_target).sum())

    if invalid_target_count:
        logger.warning(
            "Excluding %s records with invalid delivery-time targets.",
            invalid_target_count,
        )

    data = data.loc[valid_target].copy().reset_index(drop=True)

    require(
        not data.empty,
        "No records remain after validating the delivery-time target.",
    )

    for column in PREDICTOR_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    logger.info(
        "Validated dataset: %s rows and %s predictor columns.",
        len(data),
        len(PREDICTOR_COLUMNS),
    )

    logger.info(
        "Invalid target records excluded: %s",
        invalid_target_count,
    )

    return data


def load_trained_model(logger: logging.Logger) -> Any:
    """Load only the required tuned Random Forest pipeline."""
    require(
        MODEL_FILE.is_file(),
        (
            f"Required tuned model was not found: {MODEL_FILE}. "
            "Run src/hyperparameter_tuning.py first."
        ),
    )

    logger.info("Loading tuned model: %s", MODEL_FILE)

    model = joblib.load(MODEL_FILE)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

        require(
            model_features == PREDICTOR_COLUMNS,
            (
                "The loaded model's feature names or order do not match "
                "the required 13 predictors. "
                f"Expected {PREDICTOR_COLUMNS}; got {model_features}."
            ),
        )

    return model


def build_predictor_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return the exact 13 model predictors in the trained order."""
    missing_columns = [
        column
        for column in PREDICTOR_COLUMNS
        if column not in data.columns
    ]

    require(
        not missing_columns,
        f"Cannot build predictor matrix; missing columns: {missing_columns}",
    )

    require(
        TARGET_COLUMN not in PREDICTOR_COLUMNS,
        f"The target {TARGET_COLUMN!r} must not be a predictor.",
    )

    predictors = data.loc[:, PREDICTOR_COLUMNS].copy()

    require(
        TARGET_COLUMN not in predictors.columns,
        f"Target leakage detected: {TARGET_COLUMN!r} entered predictors.",
    )

    require(
        list(predictors.columns) == PREDICTOR_COLUMNS,
        "Predictor columns do not match the trained feature order.",
    )

    return predictors


def predict_delivery_times(
    model: Any,
    predictors: pd.DataFrame,
    logger: logging.Logger,
) -> np.ndarray:
    """Generate and validate predictions for a predictor matrix."""
    require(
        TARGET_COLUMN not in predictors.columns,
        f"Target leakage detected: {TARGET_COLUMN!r} entered predictions.",
    )

    require(
        list(predictors.columns) == PREDICTOR_COLUMNS,
        "Prediction inputs do not match the expected predictor order.",
    )

    try:
        raw_predictions = model.predict(predictors)
    except Exception as exc:
        logger.exception("Model prediction failed.")
        raise RuntimeError(
            f"Model prediction failed: {exc}"
        ) from exc

    predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)

    require(
        len(predictions) == len(predictors),
        (
            "Prediction count does not match input count. "
            f"Inputs: {len(predictors)}; predictions: {len(predictions)}."
        ),
    )

    require(
        np.isfinite(predictions).all(),
        "Model predictions contain NaN or infinite values.",
    )

    return predictions


# ============================================================================
# Geographic eligibility and scenario construction
# ============================================================================

def valid_geographic_flags(data: pd.DataFrame) -> pd.Series:
    """
    Identify records whose geographic-validity flag explicitly means true.

    Accepted values are Boolean True or numeric 1. Missing values, strings,
    unexpected numbers, and other values are treated as invalid.
    """
    if "is_geo_valid" not in data.columns:
        return pd.Series(False, index=data.index, dtype=bool)

    flags = data["is_geo_valid"]

    def is_explicitly_true(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if isinstance(value, (int, np.integer)):
            return int(value) == 1

        if isinstance(value, (float, np.floating)):
            return math.isfinite(float(value)) and float(value) == 1.0

        return False

    return flags.map(is_explicitly_true).astype(bool)


def geographic_eligibility(data: pd.DataFrame) -> pd.Series:
    """
    Require a finite, positive straight-line distance and an explicit
    valid-coordinate flag.
    """
    distance = pd.to_numeric(
        data["delivery_distance_km"],
        errors="coerce",
    )

    finite_positive_distance = (
        distance.notna()
        & np.isfinite(distance)
        & (distance > 0)
    )

    return (
        finite_positive_distance
        & valid_geographic_flags(data)
    ).astype(bool)


def synchronize_distance_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep distance and its log-transformed feature synchronized."""
    result = data.copy()

    distance = pd.to_numeric(
        result["delivery_distance_km"],
        errors="coerce",
    )

    result["delivery_distance_km"] = distance

    # Invalid or non-positive distances are represented as missing for the
    # model's existing median imputer. They are never scenario-eligible.
    valid_distance = (
        distance.notna()
        & np.isfinite(distance)
        & (distance > 0)
    )

    result.loc[
        valid_distance,
        "log1p_delivery_distance_km",
    ] = np.log1p(distance.loc[valid_distance])

    result.loc[
        ~valid_distance,
        "log1p_delivery_distance_km",
    ] = np.nan

    return result


def select_reproducible_sample(
    data: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Select a reproducible sample from the available dataset.

    When the dataset exceeds MAX_SAMPLE_SIZE, the sample is selected using
    RANDOM_SEED. Otherwise, all available records are used.
    """
    if len(data) > MAX_SAMPLE_SIZE:
        sample = data.sample(
            n=MAX_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        ).copy()
    else:
        sample = data.copy()

    sample = sample.reset_index(drop=True)

    logger.info(
        "Reproducible evaluation sample: %s of %s available records.",
        len(sample),
        len(data),
    )

    return sample


# ============================================================================
# Distance-reduction scenario analysis
# ============================================================================

def analyze_distance_scenarios(
    model: Any,
    sample: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compare baseline predictions with illustrative distance-reduction probes.

    Only records with finite positive distances and explicitly valid
    geographic flags are eligible. Scenario distances are hypothetical
    modifications to straight-line distance, not verified road routes.
    """
    sample = synchronize_distance_features(sample)

    eligibility = geographic_eligibility(sample)
    eligible_sample = sample.loc[eligibility].copy()

    total_count = len(sample)
    eligible_count = len(eligible_sample)
    ineligible_count = total_count - eligible_count

    sample_counts = {
        "total_sample_records": int(total_count),
        "eligible_records": int(eligible_count),
        "ineligible_records": int(ineligible_count),
    }

    logger.info(
        "Geographic eligibility: %s eligible; %s ineligible; %s total.",
        eligible_count,
        ineligible_count,
        total_count,
    )

    scenario_records: list[dict[str, Any]] = []

    if eligible_sample.empty:
        logger.warning(
            "No records qualify for distance scenarios. "
            "Header-only scenario output will be generated."
        )

        return scenario_records, {
            **sample_counts,
            "scenario_count": 0,
            "substantive_scenario_results_available": False,
            "scenario_note": (
                "No records met the geographic eligibility criteria. "
                "No distance-reduction estimates were generated."
            ),
        }

    baseline_predictors = build_predictor_matrix(eligible_sample)
    baseline_predictions = predict_delivery_times(
        model,
        baseline_predictors,
        logger,
    )

    for reduction_percent in DISTANCE_REDUCTION_PERCENTAGES:
        scenario_name = f"Distance reduction {reduction_percent}%"

        scenario_data = eligible_sample.copy()

        original_distance = scenario_data[
            "delivery_distance_km"
        ].to_numpy(dtype=float)

        modified_distance = original_distance * (
            1.0 - reduction_percent / 100.0
        )

        valid_modified_distance = (
            np.isfinite(modified_distance)
            & (modified_distance > 0)
        )

        # Eligibility was established before applying the scenario.
        # Retain only rows that remain valid after modification.
        if not valid_modified_distance.all():
            scenario_data = scenario_data.loc[
                valid_modified_distance
            ].copy()

            baseline_for_scenario = baseline_predictions[
                valid_modified_distance
            ]

            modified_distance = modified_distance[
                valid_modified_distance
            ]
        else:
            baseline_for_scenario = baseline_predictions

        scenario_data["delivery_distance_km"] = modified_distance
        scenario_data = synchronize_distance_features(scenario_data)

        scenario_predictors = build_predictor_matrix(scenario_data)
        scenario_predictions = predict_delivery_times(
            model,
            scenario_predictors,
            logger,
        )

        scenario_total = len(sample)
        scenario_eligible = len(scenario_data)
        scenario_ineligible = scenario_total - scenario_eligible

        if scenario_eligible == 0:
            logger.warning(
                "%s has no valid records after scenario checks.",
                scenario_name,
            )
            continue

        prediction_changes = (
            scenario_predictions - baseline_for_scenario
        )

        baseline_mean = float(np.mean(baseline_for_scenario))
        scenario_mean = float(np.mean(scenario_predictions))
        baseline_median = float(np.median(baseline_for_scenario))
        scenario_median = float(np.median(scenario_predictions))
        mean_change = scenario_mean - baseline_mean

        mean_change_percent = (
            (mean_change / baseline_mean) * 100.0
            if baseline_mean != 0
            else None
        )

        scenario_records.append(
            {
                "scenario": scenario_name,
                "distance_reduction_percent": reduction_percent,
                "sample_size": scenario_total,
                "eligible_records": scenario_eligible,
                "ineligible_records": scenario_ineligible,
                "baseline_mean_predicted_delivery_time_min": baseline_mean,
                "scenario_mean_predicted_delivery_time_min": scenario_mean,
                "mean_predicted_change_min": mean_change,
                "mean_predicted_change_percent": mean_change_percent,
                "baseline_median_predicted_delivery_time_min": baseline_median,
                "scenario_median_predicted_delivery_time_min": scenario_median,
                "median_predicted_change_min": (
                    scenario_median - baseline_median
                ),
                "minimum_predicted_change_min": float(
                    np.min(prediction_changes)
                ),
                "maximum_predicted_change_min": float(
                    np.max(prediction_changes)
                ),
                "interpretation": (
                    "Illustrative model sensitivity scenario only. "
                    "Distance refers to straight-line geographic distance, "
                    "not verified road-network distance. This estimate does "
                    "not establish causation or guarantee an operational "
                    "delivery-time reduction."
                ),
            }
        )

        logger.info(
            "%s: %s eligible records; baseline mean %.3f min; "
            "scenario mean %.3f min; predicted mean change %.3f min.",
            scenario_name,
            scenario_eligible,
            baseline_mean,
            scenario_mean,
            mean_change,
        )

    result_metadata = {
        **sample_counts,
        "scenario_count": len(scenario_records),
        "substantive_scenario_results_available": bool(scenario_records),
        "scenario_note": (
            "Scenario estimates describe the trained model's predictions "
            "under hypothetical changes to straight-line distance. They "
            "are not causal estimates, road-route simulations, guarantees, "
            "or measured operational outcomes."
        ),
    }

    return scenario_records, result_metadata


# ============================================================================
# Feature sensitivity analysis
# ============================================================================

def describe_sensitivity_feature(feature: str) -> str:
    """Provide a neutral description for each sensitivity feature."""
    descriptions = {
        "delivery_distance_km": (
            "Straight-line geographic distance in kilometres."
        ),
        "Delivery_person_Ratings": (
            "Delivery-person rating as represented in the dataset."
        ),
        "Delivery_person_Age": (
            "Delivery-person age as represented in the dataset."
        ),
    }

    return descriptions.get(feature, feature)


def sensitivity_interpretation(feature: str) -> str:
    """Return a scientific caveat appropriate to a sensitivity probe."""
    base = (
        "Model sensitivity analysis only; this is not a causal experiment "
        "or a realistic operational intervention. The tested value is "
        "applied to the sampled records while other predictor values, "
        "including one-hot categorical columns, remain unchanged."
    )

    if feature == "delivery_distance_km":
        return (
            base
            + " Distance is straight-line geographic distance, not "
            "verified road-network distance."
        )

    if feature in {
        "Delivery_person_Ratings",
        "Delivery_person_Age",
    }:
        return (
            base
            + " This feature is examined only to understand model response. "
            "The analysis does not recommend changing courier age or ratings."
        )

    return base


def analyze_feature_sensitivity(
    model: Any,
    sample: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Evaluate predictions at observed feature percentiles.

    These tests measure model response and do not imply that changing a
    feature would produce the predicted outcome in the real world.
    """
    base_data = synchronize_distance_features(sample)
    base_predictors = build_predictor_matrix(base_data)

    baseline_predictions = predict_delivery_times(
        model,
        base_predictors,
        logger,
    )

    baseline_mean = float(np.mean(baseline_predictions))
    baseline_median = float(np.median(baseline_predictions))

    sensitivity_records: list[dict[str, Any]] = []
    feature_summary: dict[str, Any] = {}

    for feature in SENSITIVITY_FEATURES:
        if feature not in base_data.columns:
            logger.warning(
                "Skipping sensitivity feature not available: %s",
                feature,
            )
            continue

        observed_values = pd.to_numeric(
            base_data[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_values = observed_values[np.isfinite(observed_values)]

        if len(finite_values) == 0:
            logger.warning(
                "Skipping sensitivity feature with no finite values: %s",
                feature,
            )
            feature_summary[feature] = {
                "status": "skipped",
                "reason": "No finite observed values were available.",
                "result_count": 0,
            }
            continue

        percentile_values = np.quantile(
            finite_values,
            [percentile for _, percentile in SENSITIVITY_PERCENTILES],
        )

        feature_record_count = 0

        for (label, percentile), tested_value in zip(
            SENSITIVITY_PERCENTILES,
            percentile_values,
        ):
            tested_value = float(tested_value)

            modified_data = base_data.copy()
            modified_data[feature] = tested_value

            if feature == "delivery_distance_km":
                modified_data = synchronize_distance_features(
                    modified_data
                )

            modified_predictors = build_predictor_matrix(modified_data)

            # Explicitly retain the trained predictor order and ensure the
            # target is not present in any prediction matrix.
            require(
                TARGET_COLUMN not in modified_predictors.columns,
                "Target leakage detected during sensitivity analysis.",
            )

            predictions = predict_delivery_times(
                model,
                modified_predictors,
                logger,
            )

            predicted_mean = float(np.mean(predictions))
            predicted_median = float(np.median(predictions))

            sensitivity_records.append(
                {
                    "feature": feature,
                    "feature_description": describe_sensitivity_feature(
                        feature
                    ),
                    "test_type": "Model sensitivity analysis",
                    "percentile_label": label,
                    "percentile": percentile,
                    "tested_value": tested_value,
                    "sample_size": len(modified_data),
                    "baseline_mean_predicted_delivery_time_min": baseline_mean,
                    "scenario_mean_predicted_delivery_time_min": predicted_mean,
                    "mean_predicted_change_min": (
                        predicted_mean - baseline_mean
                    ),
                    "baseline_median_predicted_delivery_time_min": (
                        baseline_median
                    ),
                    "scenario_median_predicted_delivery_time_min": (
                        predicted_median
                    ),
                    "median_predicted_change_min": (
                        predicted_median - baseline_median
                    ),
                    "interpretation": sensitivity_interpretation(feature),
                }
            )

            feature_record_count += 1

        feature_summary[feature] = {
            "status": (
                "completed"
                if feature_record_count > 0
                else "no_results"
            ),
            "result_count": feature_record_count,
            "finite_observed_value_count": int(len(finite_values)),
            "percentiles_tested": [
                {
                    "label": label,
                    "percentile": percentile,
                }
                for label, percentile in SENSITIVITY_PERCENTILES
            ],
            "interpretation": sensitivity_interpretation(feature),
        }

        logger.info(
            "Sensitivity analysis completed for %s: %s probes.",
            feature,
            feature_record_count,
        )

    metadata = {
        "sensitivity_record_count": len(sensitivity_records),
        "features_requested": SENSITIVITY_FEATURES,
        "features_analyzed": [
            feature
            for feature, details in feature_summary.items()
            if details.get("status") == "completed"
        ],
        "feature_results": feature_summary,
        "analysis_type": (
            "Model sensitivity analysis, not causal experiments or "
            "realistic interventions."
        ),
    }

    return sensitivity_records, metadata


# ============================================================================
# Visualizations and image verification
# ============================================================================

def verify_image_decoding(path: Path) -> bool:
    """Check whether a saved PNG can be decoded as an image."""
    if not path.is_file():
        return False

    try:
        image_data = mpimg.imread(path)

        return (
            isinstance(image_data, np.ndarray)
            and image_data.size > 0
            and image_data.ndim in (2, 3)
        )
    except Exception:
        return False


def create_placeholder_chart(
    path: Path,
    title: str,
    explanation: str,
) -> dict[str, Any]:
    """Create a valid PNG when no substantive chart can be drawn."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=18)
    ax.text(
        0.5,
        0.5,
        explanation,
        ha="center",
        va="center",
        wrap=True,
        transform=ax.transAxes,
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    decoded = verify_image_decoding(path)

    return {
        "path": str(path),
        "status": "placeholder",
        "is_placeholder": True,
        "image_verified": bool(decoded),
        "reason": explanation,
    }


def create_scenario_chart(
    scenario_data: pd.DataFrame,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Plot baseline and scenario mean predictions when results exist."""
    if scenario_data.empty:
        return create_placeholder_chart(
            SCENARIO_PLOT,
            "Predicted Delivery-Time Scenarios",
            (
                "No substantive distance-scenario results are available. "
                "No records met the required geographic eligibility criteria."
            ),
        )

    required = [
        "scenario",
        "baseline_mean_predicted_delivery_time_min",
        "scenario_mean_predicted_delivery_time_min",
    ]

    if not all(column in scenario_data.columns for column in required):
        return create_placeholder_chart(
            SCENARIO_PLOT,
            "Predicted Delivery-Time Scenarios",
            "Required scenario chart columns are unavailable.",
        )

    plot_data = scenario_data.dropna(subset=required).copy()

    if plot_data.empty:
        return create_placeholder_chart(
            SCENARIO_PLOT,
            "Predicted Delivery-Time Scenarios",
            "No finite scenario values are available to plot.",
        )

    positions = np.arange(len(plot_data))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        positions - width / 2,
        plot_data["baseline_mean_predicted_delivery_time_min"],
        width,
        label="Baseline mean prediction",
    )

    ax.bar(
        positions + width / 2,
        plot_data["scenario_mean_predicted_delivery_time_min"],
        width,
        label="Scenario mean prediction",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_data["scenario"], rotation=15, ha="right")
    ax.set_ylabel("Predicted delivery time (minutes)")
    ax.set_title("Model Predictions Under Illustrative Distance Scenarios")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.text(
        0.01,
        0.01,
        (
            "Model sensitivity estimates only; straight-line distance is not "
            "road distance. Results do not establish causation or guarantee "
            "operational improvements."
        ),
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(SCENARIO_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)

    decoded = verify_image_decoding(SCENARIO_PLOT)

    logger.info(
        "Scenario chart saved; image decoding verification: %s.",
        decoded,
    )

    return {
        "path": str(SCENARIO_PLOT),
        "status": (
            "substantive_analytical"
            if decoded
            else "created_but_decode_failed"
        ),
        "is_placeholder": False,
        "image_verified": bool(decoded),
        "reason": (
            "Plots baseline and scenario model predictions."
            if decoded
            else "PNG file was saved but could not be decoded."
        ),
    }


def create_sensitivity_chart(
    sensitivity_data: pd.DataFrame,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Plot mean predicted delivery-time changes from sensitivity probes."""
    if sensitivity_data.empty:
        return create_placeholder_chart(
            SENSITIVITY_PLOT,
            "Feature Sensitivity Analysis",
            "No substantive feature-sensitivity results are available.",
        )

    required = [
        "feature",
        "percentile_label",
        "mean_predicted_change_min",
    ]

    if not all(column in sensitivity_data.columns for column in required):
        return create_placeholder_chart(
            SENSITIVITY_PLOT,
            "Feature Sensitivity Analysis",
            "Required sensitivity chart columns are unavailable.",
        )

    plot_data = sensitivity_data.dropna(
        subset=["mean_predicted_change_min"]
    ).copy()

    if plot_data.empty:
        return create_placeholder_chart(
            SENSITIVITY_PLOT,
            "Feature Sensitivity Analysis",
            "No finite sensitivity values are available to plot.",
        )

    features = list(dict.fromkeys(plot_data["feature"].tolist()))

    fig, axes = plt.subplots(
        nrows=len(features),
        ncols=1,
        figsize=(11, max(4, 3.8 * len(features))),
        squeeze=False,
    )

    for row_index, feature in enumerate(features):
        ax = axes[row_index, 0]

        feature_data = plot_data.loc[
            plot_data["feature"] == feature
        ].copy()

        x_positions = np.arange(len(feature_data))

        ax.plot(
            x_positions,
            feature_data["mean_predicted_change_min"],
            marker="o",
        )

        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            feature_data["percentile_label"],
            rotation=15,
            ha="right",
        )
        ax.set_ylabel("Change in mean predicted minutes")
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Model Sensitivity Analysis at Observed Feature Percentiles",
        fontsize=14,
    )

    fig.text(
        0.01,
        0.005,
        (
            "Sensitivity probes are not causal experiments or realistic "
            "interventions. Courier age and ratings are not optimization "
            "targets. Distance is straight-line geographic distance."
        ),
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(SENSITIVITY_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)

    decoded = verify_image_decoding(SENSITIVITY_PLOT)

    logger.info(
        "Sensitivity chart saved; image decoding verification: %s.",
        decoded,
    )

    return {
        "path": str(SENSITIVITY_PLOT),
        "status": (
            "substantive_analytical"
            if decoded
            else "created_but_decode_failed"
        ),
        "is_placeholder": False,
        "image_verified": bool(decoded),
        "reason": (
            "Plots changes in model predictions across observed "
            "feature percentiles."
            if decoded
            else "PNG file was saved but could not be decoded."
        ),
    }


# ============================================================================
# Recommendations report
# ============================================================================

def generate_recommendations_report(
    scenario_data: pd.DataFrame,
    sensitivity_data: pd.DataFrame,
    sample_metadata: dict[str, Any],
    sensitivity_metadata: dict[str, Any],
    chart_status: dict[str, dict[str, Any]],
) -> None:
    """Write a transparent report of the analysis and its limitations."""
    lines: list[str] = []

    lines.append("LOGISTICS DELIVERY-TIME OPTIMIZATION ANALYSIS")
    lines.append("=" * 58)
    lines.append("")
    lines.append(
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}"
    )
    lines.append("")
    lines.append("1. ANALYSIS SCOPE")
    lines.append("-" * 58)
    lines.append(
        "This report summarizes model-based distance scenarios and "
        "feature-sensitivity probes using the tuned delivery-time model."
    )
    lines.append(
        "Evaluation records are a reproducible sample from the available "
        "engineered dataset. The sample is capped at the configured maximum "
        "when the dataset is larger."
    )
    lines.append("")
    lines.append("2. GEOGRAPHIC ELIGIBILITY")
    lines.append("-" * 58)
    lines.append(
        f"Total evaluation records: "
        f"{sample_metadata.get('total_sample_records', 0)}"
    )
    lines.append(
        f"Eligible records: {sample_metadata.get('eligible_records', 0)}"
    )
    lines.append(
        f"Ineligible records: {sample_metadata.get('ineligible_records', 0)}"
    )
    lines.append(
        "Eligibility requires a finite positive distance and an explicit "
        "valid geographic-coordinate flag."
    )
    lines.append("")
    lines.append("3. DISTANCE SCENARIO RESULTS")
    lines.append("-" * 58)

    if scenario_data.empty:
        lines.append(
            "No substantive distance scenario results were available. "
            "The scenario CSV retains its expected column headers."
        )
    else:
        for _, row in scenario_data.iterrows():
            lines.append(
                f"{row['scenario']}: "
                f"{row['eligible_records']} eligible records; "
                f"baseline mean prediction "
                f"{row['baseline_mean_predicted_delivery_time_min']:.3f} "
                f"minutes; scenario mean prediction "
                f"{row['scenario_mean_predicted_delivery_time_min']:.3f} "
                f"minutes; predicted mean change "
                f"{row['mean_predicted_change_min']:.3f} minutes."
            )

        lines.append("")
        lines.append(
            "These values describe predictions under hypothetical changes "
            "to straight-line geographic distance. They do not establish "
            "causation, guaranteed delivery-time reductions, financial "
            "savings, or proven operational improvements."
        )

    lines.append("")
    lines.append("4. FEATURE SENSITIVITY ANALYSIS")
    lines.append("-" * 58)

    if sensitivity_data.empty:
        lines.append(
            "No substantive feature-sensitivity results were available. "
            "The sensitivity CSV retains its expected column headers."
        )
    else:
        lines.append(
            f"Sensitivity probes generated: {len(sensitivity_data)}"
        )
        lines.append(
            "The analysis examines model response at observed percentiles "
            "while other predictors, including one-hot categorical columns, "
            "remain unchanged."
        )
        lines.append(
            "These probes are not causal experiments or realistic "
            "operational interventions."
        )

    lines.append("")
    lines.append("5. INTERPRETATION AND LIMITATIONS")
    lines.append("-" * 58)
    lines.append(
        "- delivery_distance_km represents straight-line geographic "
        "distance, not verified road-network distance."
    )
    lines.append(
        "- Model predictions are estimates and are not guaranteed outcomes."
    )
    lines.append(
        "- The available dataset does not directly establish effects from "
        "traffic, weather, food preparation, or other unobserved factors."
    )
    lines.append(
        "- Courier age and ratings are analyzed only for model sensitivity. "
        "Changing either is not recommended as an optimization action."
    )
    lines.append(
        "- The results should not be interpreted as causal effects, "
        "financial savings, or proof of operational improvements."
    )
    lines.append("")
    lines.append("6. CHART ARTIFACT STATUS")
    lines.append("-" * 58)

    for chart_name, details in chart_status.items():
        lines.append(
            f"{chart_name}: status={details.get('status')}; "
            f"placeholder={details.get('is_placeholder')}; "
            f"image_verified={details.get('image_verified')}."
        )

    lines.append("")
    lines.append("7. APPROPRIATE NEXT STEPS")
    lines.append("-" * 58)
    lines.append(
        "Treat the scenario results as exploratory model diagnostics. "
        "Any operational use would require suitable road-network distances, "
        "relevant operational measurements, prospective validation, and "
        "evaluation against observed outcomes."
    )
    lines.append("")

    RECOMMENDATIONS_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# Summary creation and final artifact verification
# ============================================================================

def verify_chart_status(
    chart_name: str,
    details: dict[str, Any],
) -> None:
    """Validate required status metadata for a chart."""
    required_fields = {
        "path",
        "status",
        "is_placeholder",
        "image_verified",
        "reason",
    }

    missing_fields = required_fields - set(details.keys())

    require(
        not missing_fields,
        (
            f"Chart status for {chart_name} is missing fields: "
            f"{sorted(missing_fields)}"
        ),
    )

    require(
        isinstance(details["is_placeholder"], bool),
        f"Chart {chart_name} has an invalid is_placeholder value.",
    )

    require(
        isinstance(details["image_verified"], bool),
        f"Chart {chart_name} has an invalid image_verified value.",
    )

    chart_path = Path(details["path"])

    require(
        chart_path.is_file(),
        f"Chart artifact is missing: {chart_path}",
    )

    require(
        details["image_verified"],
        f"Chart image decoding verification failed: {chart_name}",
    )


def verify_summary_contents(
    summary: dict[str, Any],
    expected_sample_size: int,
    expected_scenario_count: int,
    expected_sensitivity_count: int,
) -> None:
    """Verify essential summary fields after reading the JSON back."""
    required_fields = [
        "project_root",
        "dataset_path",
        "model_path",
        "target_column",
        "predictor_columns",
        "dataset_records_available",
        "evaluation_sample_size",
        "sample_counts",
        "scenario_count",
        "sensitivity_record_count",
        "substantive_scenario_results_available",
        "substantive_sensitivity_results_available",
        "chart_status",
        "artifact_verification",
        "scientific_limitations",
    ]

    missing = [
        field
        for field in required_fields
        if field not in summary
    ]

    require(
        not missing,
        f"Summary JSON is missing required fields: {missing}",
    )

    require(
        summary["target_column"] == TARGET_COLUMN,
        "Summary JSON target column is incorrect.",
    )

    require(
        summary["predictor_columns"] == PREDICTOR_COLUMNS,
        "Summary JSON predictor names or order are incorrect.",
    )

    require(
        TARGET_COLUMN not in summary["predictor_columns"],
        "Target leakage detected in the summary predictor list.",
    )

    require(
        summary["evaluation_sample_size"] == expected_sample_size,
        "Summary JSON evaluation sample size is incorrect.",
    )

    require(
        summary["scenario_count"] == expected_scenario_count,
        "Summary JSON scenario count is incorrect.",
    )

    require(
        summary["sensitivity_record_count"] == expected_sensitivity_count,
        "Summary JSON sensitivity record count is incorrect.",
    )

    sample_counts = summary["sample_counts"]

    for field in (
        "total_sample_records",
        "eligible_records",
        "ineligible_records",
    ):
        require(
            field in sample_counts,
            f"Summary JSON sample counts are missing {field!r}.",
        )

    require(
        sample_counts["total_sample_records"] == expected_sample_size,
        "Summary JSON total sample count is incorrect.",
    )

    require(
        sample_counts["eligible_records"]
        + sample_counts["ineligible_records"]
        == expected_sample_size,
        "Eligible and ineligible sample counts do not add up correctly.",
    )

    charts = summary["chart_status"]

    for chart_name in (
        "predicted_delivery_time_scenarios",
        "feature_sensitivity_analysis",
    ):
        require(
            chart_name in charts,
            f"Summary JSON is missing chart status: {chart_name}.",
        )

        verify_chart_status(chart_name, charts[chart_name])

    artifacts = summary["artifact_verification"]

    require(
        isinstance(artifacts, dict),
        "Summary JSON artifact_verification must be an object.",
    )


def verify_all_artifacts(
    logger: logging.Logger,
    scenario_count: int,
    sensitivity_count: int,
) -> dict[str, Any]:
    """Verify the seven required output artifacts before success is reported."""
    artifact_results: dict[str, Any] = {}

    for path in EXPECTED_OUTPUT_ARTIFACTS:
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0

        artifact_results[path.name] = {
            "exists": bool(exists),
            "size_bytes": int(size_bytes),
        }

        require(
            exists,
            f"Required output artifact was not created: {path}",
        )

        require(
            size_bytes > 0,
            f"Required output artifact is empty: {path}",
        )

    # Read the two CSVs back and validate their headers and record counts.
    verify_csv_readback(
        SCENARIO_CSV,
        SCENARIO_COLUMNS,
        scenario_count,
    )

    verify_csv_readback(
        SENSITIVITY_CSV,
        SENSITIVITY_COLUMNS,
        sensitivity_count,
    )

    # Confirm both PNGs can be decoded.
    for chart_path in (SCENARIO_PLOT, SENSITIVITY_PLOT):
        require(
            verify_image_decoding(chart_path),
            f"Saved chart could not be decoded: {chart_path}",
        )

    logger.info(
        "All %s required output artifacts passed existence and content checks.",
        len(EXPECTED_OUTPUT_ARTIFACTS),
    )

    return artifact_results


# ============================================================================
# Main pipeline
# ============================================================================

def main() -> int:
    """Run optimization scenarios, sensitivity analysis, and verification."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = configure_logging()

    start_time = datetime.now(timezone.utc)

    logger.info("=" * 76)
    logger.info("LOGISTICS DELIVERY-TIME OPTIMIZATION ANALYSIS")
    logger.info("=" * 76)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Dataset path: %s", DATA_FILE)
    logger.info("Model path: %s", MODEL_FILE)
    logger.info("Output directory: %s", OUTPUT_DIR)

    try:
        data = load_dataset(logger)
        model = load_trained_model(logger)

        sample = select_reproducible_sample(data, logger)

        # Check that target data remains separate from every model input.
        predictors = build_predictor_matrix(sample)

        require(
            TARGET_COLUMN not in predictors.columns,
            "Target leakage detected before the analysis began.",
        )

        logger.info(
            "Validated %s predictors in the exact trained order.",
            len(predictors.columns),
        )

        scenario_records, scenario_metadata = analyze_distance_scenarios(
            model,
            sample,
            logger,
        )

        sensitivity_records, sensitivity_metadata = (
            analyze_feature_sensitivity(
                model,
                sample,
                logger,
            )
        )

        scenario_data = save_csv_with_headers(
            scenario_records,
            SCENARIO_CSV,
            SCENARIO_COLUMNS,
        )

        sensitivity_data = save_csv_with_headers(
            sensitivity_records,
            SENSITIVITY_CSV,
            SENSITIVITY_COLUMNS,
        )

        scenario_chart_status = create_scenario_chart(
            scenario_data,
            logger,
        )

        sensitivity_chart_status = create_sensitivity_chart(
            sensitivity_data,
            logger,
        )

        chart_status = {
            "predicted_delivery_time_scenarios": scenario_chart_status,
            "feature_sensitivity_analysis": sensitivity_chart_status,
        }

        for chart_name, details in chart_status.items():
            verify_chart_status(chart_name, details)

        generate_recommendations_report(
            scenario_data=scenario_data,
            sensitivity_data=sensitivity_data,
            sample_metadata=scenario_metadata,
            sensitivity_metadata=sensitivity_metadata,
            chart_status=chart_status,
        )

        end_time = datetime.now(timezone.utc)

        summary: dict[str, Any] = {
            "project_root": str(PROJECT_ROOT),
            "dataset_path": str(DATA_FILE),
            "model_path": str(MODEL_FILE),
            "generated_at_utc": end_time.isoformat(),
            "duration_seconds": (
                end_time - start_time
            ).total_seconds(),
            "target_column": TARGET_COLUMN,
            "predictor_columns": PREDICTOR_COLUMNS,
            "dataset_records_available": int(len(data)),
            "evaluation_sample_size": int(len(sample)),
            "sample_selection": {
                "method": (
                    "All records when dataset size is within the cap; "
                    "otherwise random sampling without replacement."
                ),
                "maximum_sample_size": MAX_SAMPLE_SIZE,
                "random_seed": RANDOM_SEED,
                "description": (
                    "A reproducible sample from the available dataset."
                ),
            },
            "sample_counts": {
                "total_sample_records": scenario_metadata[
                    "total_sample_records"
                ],
                "eligible_records": scenario_metadata[
                    "eligible_records"
                ],
                "ineligible_records": scenario_metadata[
                    "ineligible_records"
                ],
            },
            "scenario_count": len(scenario_records),
            "scenario_results": scenario_records,
            "scenario_metadata": scenario_metadata,
            "sensitivity_record_count": len(sensitivity_records),
            "sensitivity_metadata": sensitivity_metadata,
            "substantive_scenario_results_available": bool(
                scenario_records
            ),
            "substantive_sensitivity_results_available": bool(
                sensitivity_records
            ),
            "chart_status": chart_status,
            "artifact_verification": {},
            "scientific_limitations": [
                (
                    "delivery_distance_km is straight-line geographic "
                    "distance, not verified road-network distance."
                ),
                (
                    "Distance-reduction scenarios are hypothetical model "
                    "probes and do not guarantee delivery-time reductions."
                ),
                (
                    "Sensitivity results are not causal estimates or "
                    "realistic operational interventions."
                ),
                (
                    "Courier age and ratings are not recommended as "
                    "operational optimization targets."
                ),
                (
                    "The available dataset does not directly establish "
                    "effects from traffic, weather, or food preparation."
                ),
                (
                    "No financial savings or proven operational improvements "
                    "are inferred from the model predictions."
                ),
            ],
            "analysis_status": (
                "Artifacts are pending final verification."
            ),
        }

        # Write an initial summary so it can be verified as one of the
        # required artifacts. Its verification results are then written back.
        write_strict_json(SUMMARY_JSON, summary)

        # First verify all artifacts other than the final JSON content.
        artifact_results = verify_all_artifacts(
            logger=logger,
            scenario_count=len(scenario_records),
            sensitivity_count=len(sensitivity_records),
        )

        summary["artifact_verification"] = {
            "all_required_artifacts_exist": True,
            "all_required_artifacts_nonempty": True,
            "csv_headers_and_row_counts_verified": True,
            "chart_image_decoding_verified": True,
            "required_artifact_count": len(EXPECTED_OUTPUT_ARTIFACTS),
            "artifacts": artifact_results,
        }

        summary["analysis_status"] = (
            "ARTIFACTS_VERIFIED_SUBSTANTIVE_RESULTS_AVAILABLE"
            if scenario_records or sensitivity_records
            else "ARTIFACTS_VERIFIED_NO_SUBSTANTIVE_RESULTS_AVAILABLE"
        )

        write_strict_json(SUMMARY_JSON, summary)

        # Verify the final JSON by reading it back, including chart statuses,
        # sample counts, and scenario/sensitivity counts.
        saved_summary = verify_json_readback(SUMMARY_JSON)

        verify_summary_contents(
            summary=saved_summary,
            expected_sample_size=len(sample),
            expected_scenario_count=len(scenario_records),
            expected_sensitivity_count=len(sensitivity_records),
        )

        # Recheck the final JSON after its final write, and recheck the
        # required artifacts before printing the success message.
        require(
            SUMMARY_JSON.is_file() and SUMMARY_JSON.stat().st_size > 0,
            "Final JSON summary failed its existence or size check.",
        )

        final_artifacts = verify_all_artifacts(
            logger=logger,
            scenario_count=len(scenario_records),
            sensitivity_count=len(sensitivity_records),
        )

        logger.info("-" * 76)
        logger.info("FINAL ARTIFACT VERIFICATION")
        logger.info("-" * 76)

        for artifact_name, details in final_artifacts.items():
            logger.info(
                "%s | verified=%s | size=%s bytes",
                artifact_name,
                details["exists"] and details["size_bytes"] > 0,
                details["size_bytes"],
            )

        logger.info(
            "Scenario chart: %s; placeholder=%s; image_verified=%s",
            scenario_chart_status["status"],
            scenario_chart_status["is_placeholder"],
            scenario_chart_status["image_verified"],
        )

        logger.info(
            "Sensitivity chart: %s; placeholder=%s; image_verified=%s",
            sensitivity_chart_status["status"],
            sensitivity_chart_status["is_placeholder"],
            sensitivity_chart_status["image_verified"],
        )

        logger.info(
            "Substantive scenario results available: %s",
            bool(scenario_records),
        )

        logger.info(
            "Substantive sensitivity results available: %s",
            bool(sensitivity_records),
        )

        logger.info("=" * 76)
        logger.info(
            "SUCCESS: All seven output artifacts have been verified."
        )

        if scenario_records or sensitivity_records:
            logger.info(
                "Substantive analytical results are available; interpret "
                "them within the documented scientific limitations."
            )
        else:
            logger.warning(
                "Artifact verification succeeded, but no substantive "
                "scenario or sensitivity results were available."
            )

        logger.info("Summary JSON: %s", SUMMARY_JSON)
        logger.info("Recommendations report: %s", RECOMMENDATIONS_FILE)
        logger.info("Output directory: %s", OUTPUT_DIR)
        logger.info("=" * 76)

        return 0

    except Exception as exc:
        logger.error("Optimization analysis failed: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        logger.error(
            "No final success message was issued because verification "
            "did not complete."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())