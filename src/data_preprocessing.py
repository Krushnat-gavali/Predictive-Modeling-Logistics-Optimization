"""
Predictive Modeling and Optimization in Logistics Systems Using Python
======================================================================

Module: src/data_preprocessing.py

Description:
    Data ingestion, validation, preprocessing, and quality auditing for the
    Kaggle Food Delivery Time dataset.

    This module:
    - Validates the dataset schema.
    - Audits missing values, duplicate rows, and repeated IDs.
    - Validates delivery time, age, ratings, and geographical coordinates.
    - Flags suspicious coordinates without discarding otherwise useful rows.
    - Calculates straight-line geographical distance using the Haversine formula.
    - Saves the cleaned dataset and a machine-readable JSON quality report.

Important:
    delivery_distance_km represents straight-line distance, not road distance.
    Age and rating limits are assumptions used for data-quality screening.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("DataPreprocessing")


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPECTED_COLUMNS = [
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Type_of_order",
    "Type_of_vehicle",
    "Time_taken(min)",
]

TARGET_COLUMN = "Time_taken(min)"

# These are screening assumptions, not proof that a record is incorrect.
AGE_MIN = 18.0
AGE_MAX = 70.0

RATING_MIN = 1.0
RATING_MAX = 5.0

LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0

LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0

# Coordinates close to (0, 0) are flagged for investigation.
NEAR_ZERO_EPSILON = 0.05

# Destination placeholder detection near (0.01, 0.01).
PLACEHOLDER_LATITUDE = 0.01
PLACEHOLDER_LONGITUDE = 0.01
PLACEHOLDER_EPSILON = 0.005

EARTH_RADIUS_KM = 6371.0

CATEGORICAL_COLUMNS = [
    "Type_of_order",
    "Type_of_vehicle",
]

NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    TARGET_COLUMN,
]

GEO_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


# ============================================================================
# PROJECT PATHS
# ============================================================================

def get_project_paths() -> tuple[Path, Path, Path, Path]:
    """
    Resolve project paths from this script's location.

    Returns:
        A tuple containing:
        project_root, raw_csv_path, cleaned_csv_path, quality_json_path.
    """
    project_root = Path(__file__).resolve().parent.parent

    raw_csv_path = (
        project_root / "data" / "raw" / "deliverytime.csv"
    )

    cleaned_csv_path = (
        project_root / "data" / "processed" / "cleaned_delivery_data.csv"
    )

    quality_json_path = (
        project_root / "outputs" / "evaluation" / "data_quality_summary.json"
    )

    return (
        project_root,
        raw_csv_path,
        cleaned_csv_path,
        quality_json_path,
    )


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def sanitize_for_json(value: Any) -> Any:
    """
    Convert NumPy values and non-finite numbers into JSON-compatible values.

    NaN and infinity are converted to None, which JSON serialization writes
    as null.
    """
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())

    return value


# ============================================================================
# NUMERIC CONVERSION
# ============================================================================

def convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a Series to numeric values without imputing missing data.

    Common missing-value representations are treated as missing.
    Unparseable values become NaN.
    """
    text = series.astype("string").str.strip()

    missing_tokens = [
        "",
        "nan",
        "null",
        "none",
        "?",
        "na",
        "n/a",
    ]

    text = text.mask(
        text.str.lower().isin(missing_tokens),
        pd.NA,
    )

    return pd.to_numeric(text, errors="coerce")


# ============================================================================
# DATA LOADING AND SCHEMA VALIDATION
# ============================================================================

def load_and_validate_raw_csv(raw_csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV and verify that all expected columns exist.

    The raw file is read but never modified by this function.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_csv_path}\n"
            "Place deliverytime.csv inside the project's data/raw directory."
        )

    if not raw_csv_path.is_file():
        raise ValueError(
            f"The input path is not a file: {raw_csv_path}"
        )

    if raw_csv_path.stat().st_size == 0:
        raise ValueError(
            f"The input CSV is empty: {raw_csv_path}"
        )

    logger.info("Loading dataset: %s", raw_csv_path)

    try:
        df = pd.read_csv(raw_csv_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(
            f"Could not read the dataset as a valid CSV: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The CSV contains column headers but no data records."
        )

    # Remove accidental whitespace around column headers.
    original_columns = list(df.columns)
    normalized_columns = [
        str(column).strip()
        for column in original_columns
    ]

    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError(
            "Column normalization created duplicate column names. "
            "Please inspect the CSV headers."
        )

    df.columns = normalized_columns

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset schema validation failed.\n"
            f"Missing columns: {missing_columns}\n"
            f"Columns found: {list(df.columns)}"
        )

    logger.info(
        "Schema validation successful: %s rows and %s columns.",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


# ============================================================================
# HAVERSINE DISTANCE
# ============================================================================

def calculate_haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calculate straight-line great-circle distance in kilometers.

    All input coordinates must be valid decimal-degree coordinates.
    This distance is not equivalent to actual road travel distance.
    """
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lon2_rad = np.radians(np.asarray(lon2, dtype=float))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2.0) ** 2
    )

    # Avoid floating-point errors outside the arcsine domain.
    a = np.clip(a, 0.0, 1.0)

    central_angle = 2.0 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * central_angle


# ============================================================================
# QUALITY AUDIT HELPERS
# ============================================================================

def build_missing_value_audit(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Calculate missing-value counts and percentages for every column."""
    row_count = len(df)
    audit = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        missing_percentage = (
            missing_count / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        audit[column] = {
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 4),
        }

    return audit


def build_numeric_summary(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Calculate descriptive statistics for selected numeric columns."""
    summary = {}

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            summary[column] = {
                "count": 0,
                "minimum": None,
                "mean": None,
                "median": None,
                "maximum": None,
                "standard_deviation": None,
            }
            continue

        summary[column] = {
            "count": int(values.count()),
            "minimum": round(float(values.min()), 4),
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "maximum": round(float(values.max()), 4),
            "standard_deviation": round(
                float(values.std()) if len(values) > 1 else 0.0,
                4,
            ),
        }

    return summary


# ============================================================================
# DATA CLEANING AND QUALITY AUDITING
# ============================================================================

def clean_and_audit_delivery_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean and audit the delivery dataset.

    Exact duplicate rows are removed if found. Repeated IDs alone are not
    grounds for deleting records. Rows with invalid targets are excluded.
    Invalid ages, ratings, and coordinates are handled separately.
    """
    original_row_count = len(raw_df)
    original_column_count = len(raw_df.columns)

    df = raw_df.copy()

    quality_summary: dict[str, Any] = {
        "dataset_overview": {
            "raw_row_count": int(original_row_count),
            "raw_column_count": int(original_column_count),
            "columns": list(raw_df.columns),
        },
        "raw_missing_values": build_missing_value_audit(raw_df),
    }

    # ------------------------------------------------------------------
    # 1. Duplicate rows and identifier audit
    # ------------------------------------------------------------------

    exact_duplicate_count = int(df.duplicated().sum())

    unique_id_count = int(df["ID"].nunique(dropna=False))
    repeated_id_rows = int(df["ID"].duplicated().sum())

    unique_delivery_person_count = int(
        df["Delivery_person_ID"].nunique(dropna=True)
    )

    quality_summary["identifier_and_duplicate_audit"] = {
        "exact_duplicate_rows_found": exact_duplicate_count,
        "unique_id_values": unique_id_count,
        "repeated_id_rows": repeated_id_rows,
        "unique_delivery_person_ids": unique_delivery_person_count,
        "handling_policy": (
            "Exact duplicate rows are removed when present. Repeated ID values "
            "are reported separately and are not automatically treated as "
            "duplicate orders."
        ),
    }

    if exact_duplicate_count > 0:
        df = df.drop_duplicates().copy()

    quality_summary[
        "identifier_and_duplicate_audit"
    ]["exact_duplicate_rows_removed"] = exact_duplicate_count

    # ------------------------------------------------------------------
    # 2. Normalize categorical text
    # ------------------------------------------------------------------

    categorical_audit = {}

    for column in CATEGORICAL_COLUMNS:
        # StringDtype preserves missing values instead of turning them
        # into the literal text "nan".
        values = df[column].astype("string").str.strip()

        missing_tokens = [
            "",
            "nan",
            "null",
            "none",
            "?",
            "na",
            "n/a",
        ]

        values = values.mask(
            values.str.lower().isin(missing_tokens),
            pd.NA,
        )

        df[column] = values

        category_counts = df[column].value_counts(
            dropna=False
        )

        categorical_audit[column] = {
            "unique_nonmissing_categories": int(
                df[column].nunique(dropna=True)
            ),
            "missing_count": int(df[column].isna().sum()),
            "frequency_distribution": {
                str(category): int(count)
                for category, count in category_counts.items()
            },
        }

    quality_summary["categorical_audit"] = categorical_audit

    # ------------------------------------------------------------------
    # 3. Convert numeric fields
    # ------------------------------------------------------------------

    numeric_conversion_audit = {}

    for column in NUMERIC_COLUMNS:
        original_values = df[column].copy()
        converted_values = convert_to_numeric(original_values)

        numeric_conversion_audit[column] = {
            "missing_before_conversion": int(
                original_values.isna().sum()
            ),
            "missing_after_conversion": int(
                converted_values.isna().sum()
            ),
            "unparseable_or_newly_missing": int(
                (
                    original_values.notna()
                    & converted_values.isna()
                ).sum()
            ),
        }

        df[column] = converted_values

    for column in GEO_COLUMNS:
        original_values = df[column].copy()
        converted_values = convert_to_numeric(original_values)

        numeric_conversion_audit[column] = {
            "missing_before_conversion": int(
                original_values.isna().sum()
            ),
            "missing_after_conversion": int(
                converted_values.isna().sum()
            ),
            "unparseable_or_newly_missing": int(
                (
                    original_values.notna()
                    & converted_values.isna()
                ).sum()
            ),
        }

        df[column] = converted_values

    quality_summary["numeric_conversion_audit"] = (
        numeric_conversion_audit
    )

    # ------------------------------------------------------------------
    # 4. Validate delivery-time target
    # ------------------------------------------------------------------

    target_values = df[TARGET_COLUMN]

    target_missing_count = int(target_values.isna().sum())
    target_nonpositive_count = int(
        (target_values.notna() & (target_values <= 0)).sum()
    )

    valid_target_mask = (
        target_values.notna()
        & np.isfinite(target_values)
        & (target_values > 0)
    )

    invalid_target_count = int((~valid_target_mask).sum())

    quality_summary["target_audit"] = {
        "target_column": TARGET_COLUMN,
        "missing_or_nonnumeric_count": target_missing_count,
        "zero_or_negative_count": target_nonpositive_count,
        "invalid_target_rows_removed": invalid_target_count,
        "valid_target_rows_retained": int(valid_target_mask.sum()),
        "handling_policy": (
            "Missing, nonnumeric, zero, negative, and non-finite target "
            "values are excluded. Target values are never imputed."
        ),
    }

    df = df.loc[valid_target_mask].copy()

    # ------------------------------------------------------------------
    # 5. Validate delivery-person ages
    # ------------------------------------------------------------------

    age_values = df["Delivery_person_Age"]

    age_out_of_range_mask = (
        (age_values < AGE_MIN)
        | (age_values > AGE_MAX)
    )

    age_out_of_range_count = int(
        age_out_of_range_mask.sum()
    )

    age_non_finite_mask = (
        age_values.notna()
        & ~np.isfinite(age_values)
    )

    age_non_finite_count = int(age_non_finite_mask.sum())

    df["Delivery_person_Age"] = age_values.mask(
        age_out_of_range_mask | age_non_finite_mask,
        np.nan,
    )

    quality_summary["age_audit"] = {
        "assumed_minimum_age": AGE_MIN,
        "assumed_maximum_age": AGE_MAX,
        "out_of_range_count": age_out_of_range_count,
        "non_finite_count": age_non_finite_count,
        "missing_after_cleaning": int(
            df["Delivery_person_Age"].isna().sum()
        ),
        "handling_policy": (
            "The 18–70 age range is a screening assumption. Values outside "
            "this range and non-finite values are set to missing; their rows "
            "are retained."
        ),
    }

    # ------------------------------------------------------------------
    # 6. Validate delivery-person ratings
    # ------------------------------------------------------------------

    rating_values = df["Delivery_person_Ratings"]

    rating_out_of_range_mask = (
        (rating_values < RATING_MIN)
        | (rating_values > RATING_MAX)
    )

    rating_out_of_range_count = int(
        rating_out_of_range_mask.sum()
    )

    ratings_equal_six_count = int(
        (rating_values == 6.0).sum()
    )

    rating_non_finite_mask = (
        rating_values.notna()
        & ~np.isfinite(rating_values)
    )

    rating_non_finite_count = int(
        rating_non_finite_mask.sum()
    )

    df["Delivery_person_Ratings"] = rating_values.mask(
        rating_out_of_range_mask | rating_non_finite_mask,
        np.nan,
    )

    quality_summary["ratings_audit"] = {
        "assumed_minimum_rating": RATING_MIN,
        "assumed_maximum_rating": RATING_MAX,
        "out_of_range_count": rating_out_of_range_count,
        "ratings_equal_to_6": ratings_equal_six_count,
        "non_finite_count": rating_non_finite_count,
        "missing_after_cleaning": int(
            df["Delivery_person_Ratings"].isna().sum()
        ),
        "handling_policy": (
            "Ratings outside the assumed 1–5 range are set to missing. "
            "Values of 6.0 are not replaced with 5.0. Affected rows are retained."
        ),
    }

    # ------------------------------------------------------------------
    # 7. Validate coordinates and identify suspicious placeholders
    # ------------------------------------------------------------------

    r_lat = df["Restaurant_latitude"]
    r_lon = df["Restaurant_longitude"]
    d_lat = df["Delivery_location_latitude"]
    d_lon = df["Delivery_location_longitude"]

    restaurant_bounds_valid = (
        r_lat.between(LATITUDE_MIN, LATITUDE_MAX)
        & r_lon.between(LONGITUDE_MIN, LONGITUDE_MAX)
    )

    delivery_bounds_valid = (
        d_lat.between(LATITUDE_MIN, LATITUDE_MAX)
        & d_lon.between(LONGITUDE_MIN, LONGITUDE_MAX)
    )

    coordinates_in_bounds = (
        restaurant_bounds_valid
        & delivery_bounds_valid
    )

    # Restaurant coordinates close to (0, 0).
    restaurant_near_zero = (
        r_lat.abs().le(NEAR_ZERO_EPSILON)
        & r_lon.abs().le(NEAR_ZERO_EPSILON)
    )

    # Delivery coordinates close to (0, 0).
    delivery_near_zero = (
        d_lat.abs().le(NEAR_ZERO_EPSILON)
        & d_lon.abs().le(NEAR_ZERO_EPSILON)
    )

    # Delivery coordinates close to (0.01, 0.01).
    delivery_near_point_zero_one = (
        (d_lat - PLACEHOLDER_LATITUDE).abs().le(
            PLACEHOLDER_EPSILON
        )
        & (d_lon - PLACEHOLDER_LONGITUDE).abs().le(
            PLACEHOLDER_EPSILON
        )
    )

    suspicious_coordinates = (
        restaurant_near_zero
        | delivery_near_zero
        | delivery_near_point_zero_one
    )

    identical_origin_destination = (
        r_lat.eq(d_lat)
        & r_lon.eq(d_lon)
        & r_lat.notna()
        & r_lon.notna()
    )

    # A coordinate is usable only if all four coordinates are present,
    # physically valid, and not flagged as suspicious placeholders.
    is_geo_valid = (
        r_lat.notna()
        & r_lon.notna()
        & d_lat.notna()
        & d_lon.notna()
        & coordinates_in_bounds
        & ~suspicious_coordinates
        & np.isfinite(r_lat)
        & np.isfinite(r_lon)
        & np.isfinite(d_lat)
        & np.isfinite(d_lon)
    )

    df["is_geo_valid"] = is_geo_valid.astype(bool)

    # Distances remain missing when coordinates are not valid.
    df["delivery_distance_km"] = np.nan

    valid_geo_indices = df.index[df["is_geo_valid"]]

    if len(valid_geo_indices) > 0:
        distances = calculate_haversine_distance_km(
            lat1=df.loc[
                valid_geo_indices, "Restaurant_latitude"
            ].to_numpy(),
            lon1=df.loc[
                valid_geo_indices, "Restaurant_longitude"
            ].to_numpy(),
            lat2=df.loc[
                valid_geo_indices, "Delivery_location_latitude"
            ].to_numpy(),
            lon2=df.loc[
                valid_geo_indices, "Delivery_location_longitude"
            ].to_numpy(),
        )

        df.loc[
            valid_geo_indices, "delivery_distance_km"
        ] = distances

    quality_summary["geographical_audit"] = {
        "restaurant_coordinates_near_zero": int(
            restaurant_near_zero.sum()
        ),
        "delivery_coordinates_near_zero": int(
            delivery_near_zero.sum()
        ),
        "delivery_coordinates_near_0_01_0_01": int(
            delivery_near_point_zero_one.sum()
        ),
        "rows_with_suspicious_placeholder_coordinates": int(
            suspicious_coordinates.sum()
        ),
        "rows_with_out_of_bounds_coordinates": int(
            (~coordinates_in_bounds).sum()
        ),
        "rows_with_identical_origin_destination": int(
            identical_origin_destination.sum()
        ),
        "valid_coordinates_for_distance": int(
            df["is_geo_valid"].sum()
        ),
        "invalid_coordinates_for_distance": int(
            (~df["is_geo_valid"]).sum()
        ),
        "distance_feature": {
            "column": "delivery_distance_km",
            "method": "Haversine great-circle distance",
            "unit": "kilometers",
            "represents_road_distance": False,
        },
        "handling_policy": (
            "Out-of-bounds, missing, non-finite, and suspicious near-zero "
            "coordinates are excluded from distance calculation. Records are "
            "retained for analysis using other valid features. Negative "
            "coordinates are accepted when within physical bounds."
        ),
    }

    # ------------------------------------------------------------------
    # 8. Final missing-value audit and numeric statistics
    # ------------------------------------------------------------------

    quality_summary["cleaned_missing_values"] = (
        build_missing_value_audit(df)
    )

    quality_summary["numeric_summary"] = build_numeric_summary(
        df,
        [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            *GEO_COLUMNS,
            "delivery_distance_km",
            TARGET_COLUMN,
        ],
    )

    # ------------------------------------------------------------------
    # 9. Pipeline metrics
    # ------------------------------------------------------------------

    removed_rows = original_row_count - len(df)

    quality_summary["pipeline_execution_metrics"] = {
        "raw_record_count": int(original_row_count),
        "cleaned_record_count": int(len(df)),
        "total_removed_rows": int(removed_rows),
        "exact_duplicates_removed": int(exact_duplicate_count),
        "invalid_target_rows_removed": int(invalid_target_count),
    }

    logger.info(
        "Preprocessing completed: %s rows retained from %s raw rows.",
        f"{len(df):,}",
        f"{original_row_count:,}",
    )

    return df, quality_summary


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_quality_summary(
    quality_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the quality audit as valid JSON."""
    sanitized_summary = sanitize_for_json(quality_summary)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            sanitized_summary,
            json_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Run the complete preprocessing and quality-audit pipeline."""
    logger.info("=" * 75)
    logger.info("LOGISTICS PREDICTIVE MODELING: DATA PREPROCESSING")
    logger.info("=" * 75)

    (
        project_root,
        raw_csv_path,
        cleaned_csv_path,
        quality_json_path,
    ) = get_project_paths()

    logger.info("Project root: %s", project_root)
    logger.info("Raw input: %s", raw_csv_path)
    logger.info("Cleaned output: %s", cleaned_csv_path)
    logger.info("Quality report: %s", quality_json_path)

    # Create output directories.
    cleaned_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load and validate the input.
    try:
        raw_df = load_and_validate_raw_csv(raw_csv_path)

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input validation failed: %s", exc)
        sys.exit(1)

    # Run preprocessing.
    try:
        cleaned_df, quality_summary = clean_and_audit_delivery_data(
            raw_df
        )

    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        sys.exit(1)

    # Save the cleaned CSV.
    try:
        cleaned_df.to_csv(
            cleaned_csv_path,
            index=False,
            encoding="utf-8",
        )

    except OSError as exc:
        logger.error(
            "Could not save cleaned dataset: %s",
            exc,
        )
        sys.exit(1)

    # Save the quality report.
    try:
        save_quality_summary(
            quality_summary,
            quality_json_path,
        )

    except (OSError, TypeError, ValueError) as exc:
        logger.error(
            "Could not save quality report: %s",
            exc,
        )
        sys.exit(1)

    # Print execution summary.
    execution = quality_summary[
        "pipeline_execution_metrics"
    ]

    target_audit = quality_summary["target_audit"]

    geo_audit = quality_summary["geographical_audit"]

    print("\n" + "=" * 75)
    print("DATA PREPROCESSING EXECUTION SUMMARY")
    print("=" * 75)

    print(f"Project root:              {project_root}")
    print(f"Input dataset:             {raw_csv_path}")
    print(f"Cleaned dataset:           {cleaned_csv_path}")
    print(f"Quality report:            {quality_json_path}")

    print("-" * 75)

    print(
        f"Original rows:             "
        f"{execution['raw_record_count']:,}"
    )

    print(
        f"Cleaned rows:              "
        f"{execution['cleaned_record_count']:,}"
    )

    print(
        f"Rows removed:              "
        f"{execution['total_removed_rows']:,}"
    )

    print(
        f"Exact duplicates removed:  "
        f"{execution['exact_duplicates_removed']:,}"
    )

    print(
        f"Invalid targets removed:   "
        f"{execution['invalid_target_rows_removed']:,}"
    )

    print(
        f"Valid target records:      "
        f"{target_audit['valid_target_rows_retained']:,}"
    )

    print(
        f"Suspicious coordinate rows: "
        f"{geo_audit['rows_with_suspicious_placeholder_coordinates']:,}"
    )

    print(
        f"Valid coordinate records:  "
        f"{geo_audit['valid_coordinates_for_distance']:,}"
    )

    print("=" * 75)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()