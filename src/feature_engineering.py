"""
Feature engineering for the food-delivery time prediction project.

Input:
  data/processed/cleaned_delivery_data.csv
Outputs:
  data/processed/feature_engineered_delivery_data.csv
  outputs/evaluation/feature_engineering_summary.json

Haversine distance is straight-line distance, not road distance. The source data
does not directly measure traffic, weather, dispatch delays, or shipping costs.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FeatureEngineering")

PROJECT_NAME = "Predictive Modeling and Optimization in Logistics Systems Using Python"
MODULE_NAME = "src/feature_engineering.py"
TARGET = "Time_taken(min)"
IDS = ["ID", "Delivery_person_ID"]
COORDS = [
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
]
CATEGORICAL = ["Type_of_order", "Type_of_vehicle"]
BASE_NUMERIC = ["Delivery_person_Age", "Delivery_person_Ratings"]
EARTH_RADIUS_KM = 6371.0088
NULL_ISLAND_EPSILON = 0.05


def get_project_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parent.parent
    return (
        root,
        root / "data" / "processed" / "cleaned_delivery_data.csv",
        root / "data" / "processed" / "feature_engineered_delivery_data.csv",
        root / "outputs" / "evaluation" / "feature_engineering_summary.json",
    )


def sanitize_for_strict_json(value: Any) -> Any:
    """Convert common Python/NumPy/pandas objects to strict JSON-safe values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): sanitize_for_strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize_for_strict_json(v) for v in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [sanitize_for_strict_json(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def calculate_haversine_distance_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate great-circle (straight-line) distance in kilometres."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_and_validate_cleaned_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {path}. Run src/data_preprocessing.py first."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Input dataset is empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Unable to read input CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate column names found after trimming whitespace.")
    required = [TARGET, *CATEGORICAL, *BASE_NUMERIC]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")
    if "delivery_distance_km" not in df.columns and not all(c in df.columns for c in COORDS):
        raise ValueError("Input must contain delivery_distance_km or all four coordinate columns.")
    logger.info("Loaded dataset: %s rows x %s columns", *df.shape)
    return df


def validate_and_filter_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    target = pd.to_numeric(df[TARGET], errors="coerce")
    valid = target.notna() & np.isfinite(target) & (target > 0)
    removed = int((~valid).sum())
    out = df.loc[valid].copy()
    out[TARGET] = target.loc[valid].astype(float)
    if out.empty:
        raise ValueError("No valid positive target values remain.")
    return out, removed


def get_coordinate_validity(df: pd.DataFrame) -> pd.Series | None:
    if not all(c in df.columns for c in COORDS):
        return None
    x = {c: pd.to_numeric(df[c], errors="coerce") for c in COORDS}
    rlat, rlon = x["Restaurant_latitude"], x["Restaurant_longitude"]
    dlat, dlon = x["Delivery_location_latitude"], x["Delivery_location_longitude"]
    valid = (
        rlat.notna() & rlon.notna() & dlat.notna() & dlon.notna()
        & np.isfinite(rlat) & np.isfinite(rlon) & np.isfinite(dlat) & np.isfinite(dlon)
        & rlat.between(-90, 90) & dlat.between(-90, 90)
        & rlon.between(-180, 180) & dlon.between(-180, 180)
    )
    restaurant_null_island = rlat.abs().le(NULL_ISLAND_EPSILON) & rlon.abs().le(NULL_ISLAND_EPSILON)
    delivery_null_island = dlat.abs().le(NULL_ISLAND_EPSILON) & dlon.abs().le(NULL_ISLAND_EPSILON)
    return valid & ~(restaurant_null_island | delivery_null_island)


def normalize_geo_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    values = series.astype("string").str.strip().str.lower()
    return values.isin(["1", "1.0", "true", "yes"])


def engineer_geographic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    coord_mask = get_coordinate_validity(work)
    coords_available = coord_mask is not None
    if coords_available:
        for col in COORDS:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    flag_present = "is_geo_valid" in work.columns
    flag_valid = normalize_geo_flag(work["is_geo_valid"]) if flag_present else None
    distance_col = "delivery_distance_km"

    if distance_col in work.columns:
        method = "existing_distance_reused"
        distance = pd.to_numeric(work[distance_col], errors="coerce")
        valid_distance = distance.notna() & np.isfinite(distance) & (distance >= 0)
        # Keep a valid existing distance even if coordinates are missing/invalid.
        # Geo quality is represented independently in is_geo_valid and the audit.
        work[distance_col] = distance.where(valid_distance).astype(float)
    elif coords_available:
        method = "haversine_computed"
        work[distance_col] = np.nan
        calculation_mask = coord_mask.copy()
        if flag_valid is not None:
            calculation_mask &= flag_valid
        indices = work.index[calculation_mask]
        if len(indices):
            computed = calculate_haversine_distance_km(
                work.loc[indices, "Restaurant_latitude"].to_numpy(dtype=float),
                work.loc[indices, "Restaurant_longitude"].to_numpy(dtype=float),
                work.loc[indices, "Delivery_location_latitude"].to_numpy(dtype=float),
                work.loc[indices, "Delivery_location_longitude"].to_numpy(dtype=float),
            )
            good = np.isfinite(computed) & (computed >= 0)
            work.loc[indices[good], distance_col] = computed[good]
    else:
        method = "unavailable"
        work[distance_col] = np.nan

    # Geographic validity concerns the coordinates/flag, not distance availability.
    if coord_mask is not None:
        geo_valid = coord_mask.copy()
        if flag_valid is not None:
            geo_valid &= flag_valid
    elif flag_valid is not None:
        geo_valid = flag_valid.copy()
    else:
        # Coordinates unavailable and no trustworthy flag: represent as not verified.
        geo_valid = pd.Series(False, index=work.index)

    work["is_geo_valid"] = geo_valid.astype("int64")
    distance = pd.to_numeric(work[distance_col], errors="coerce")
    valid_distance = distance.notna() & np.isfinite(distance) & (distance >= 0)
    work[distance_col] = distance.where(valid_distance).astype(float)
    log_col = "log1p_delivery_distance_km"
    work[log_col] = np.nan
    work.loc[valid_distance, log_col] = np.log1p(work.loc[valid_distance, distance_col])

    return work, {
        "distance_calculation_method": method,
        "valid_distance_count": int(valid_distance.sum()),
        "missing_or_invalid_distance_count": int((~valid_distance).sum()),
        "valid_coordinates_count": int(coord_mask.sum()) if coord_mask is not None else None,
        "invalid_coordinates_count": int((~coord_mask).sum()) if coord_mask is not None else None,
        "is_geo_valid_true_count": int(geo_valid.sum()),
        "is_geo_valid_false_count": int((~geo_valid).sum()),
        "existing_geo_flag_present": flag_present,
        "distance_note": (
            "Haversine is straight-line distance, not road distance. If an existing distance "
            "is reused, its provenance depends on the preprocessing stage."
        ),
    }


def engineer_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    audit: dict[str, Any] = {}
    ranges = {"Delivery_person_Age": (18.0, 100.0), "Delivery_person_Ratings": (1.0, 5.0)}
    for col, (low, high) in ranges.items():
        values = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work[col] = values
        observed = values.dropna()
        audit[col] = {
            "missing_count": int(values.isna().sum()),
            "anomalies_outside_range": int(((observed < low) | (observed > high)).sum()),
            "audit_range": [low, high],
        }
    return work, audit


def engineer_categorical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    work = df.copy()
    dummy_columns: list[str] = []
    audit: dict[str, Any] = {}
    for col in CATEGORICAL:
        raw = work[col].astype("string").str.strip()
        missing = work[col].isna() | raw.str.lower().isin(["", "nan", "null", "none", "?", "<na>"])
        normalized = raw.mask(missing, "__MISSING__").fillna("__MISSING__").astype(str)
        dummies = pd.get_dummies(normalized, prefix=col, prefix_sep="__", dtype="int64")
        for dummy in dummies.columns:
            work[dummy] = dummies[dummy].astype("int64")
            dummy_columns.append(dummy)
        audit[col] = {
            "categories": sorted(normalized.unique().tolist()),
            "missing_count": int(missing.sum()),
            "generated_columns": list(dummies.columns),
        }
        work.drop(columns=[col], inplace=True)
    return work, dummy_columns, audit


def prevent_target_leakage_and_select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    work = df.copy()
    excluded_ids = [c for c in IDS if c in work.columns]
    excluded_coords = [c for c in COORDS if c in work.columns]
    work.drop(columns=excluded_ids + excluded_coords, inplace=True, errors="ignore")
    if TARGET not in work.columns:
        raise ValueError(f"Target '{TARGET}' is missing.")
    predictors = [c for c in work.columns if c != TARGET]
    final = work[predictors + [TARGET]].copy()
    return final, predictors, {
        "excluded_identifiers": excluded_ids,
        "excluded_raw_coordinates": excluded_coords,
        "target_excluded_from_predictors": TARGET not in predictors,
    }


def execute_exact_18_validations(
    df: pd.DataFrame,
    original_rows: int,
    removed_targets: int,
    predictors: list[str],
    dummy_columns: list[str],
) -> list[str]:
    """Execute exactly 18 real assertions before saving."""
    checks: list[str] = []
    # 1
    assert TARGET in df.columns, "Validation 1 failed: target missing."
    checks.append("Check 1 passed: target exists.")
    # 2
    assert df[TARGET].notna().all(), "Validation 2 failed: target has missing values."
    checks.append("Check 2 passed: target has no missing values.")
    # 3
    assert np.isfinite(df[TARGET].to_numpy(dtype=float)).all(), "Validation 3 failed: target is non-finite."
    checks.append("Check 3 passed: target is finite.")
    # 4
    assert (df[TARGET] > 0).all(), "Validation 4 failed: target is not positive."
    checks.append("Check 4 passed: target is positive.")
    # 5
    assert len(df) == original_rows - removed_targets, "Validation 5 failed: row count mismatch."
    checks.append("Check 5 passed: row count matches target filtering.")
    # 6
    assert "ID" not in df.columns, "Validation 6 failed: ID remains."
    checks.append("Check 6 passed: ID excluded.")
    # 7
    assert "Delivery_person_ID" not in df.columns, "Validation 7 failed: Delivery_person_ID remains."
    checks.append("Check 7 passed: Delivery_person_ID excluded.")
    # 8
    assert not any(c in df.columns for c in COORDS), "Validation 8 failed: raw coordinates remain."
    checks.append("Check 8 passed: raw coordinates excluded.")
    # 9
    assert "delivery_distance_km" in df and pd.api.types.is_numeric_dtype(df["delivery_distance_km"]), "Validation 9 failed: distance missing/non-numeric."
    checks.append("Check 9 passed: distance exists and is numeric.")
    # 10
    d = df["delivery_distance_km"].dropna().to_numpy(dtype=float)
    assert np.isfinite(d).all() and (d >= 0).all(), "Validation 10 failed: invalid distance."
    checks.append("Check 10 passed: available distances are finite and non-negative.")
    # 11
    assert "is_geo_valid" in df, "Validation 11 failed: is_geo_valid missing."
    geo = df["is_geo_valid"]
    assert pd.api.types.is_integer_dtype(geo.dtype) and set(geo.unique()).issubset({0, 1}), "Validation 11 failed: geo flag must be integer 0/1."
    checks.append("Check 11 passed: geo flag contains integer 0/1.")
    # 12
    assert "log1p_delivery_distance_km" in df and pd.api.types.is_numeric_dtype(df["log1p_delivery_distance_km"]), "Validation 12 failed: log distance missing/non-numeric."
    checks.append("Check 12 passed: log-distance feature exists and is numeric.")
    # 13
    valid = df["delivery_distance_km"].notna()
    log_values = df["log1p_delivery_distance_km"]
    assert log_values.loc[~valid].isna().all(), "Validation 13 failed: missing distance has log value."
    if valid.any():
        actual = log_values.loc[valid].to_numpy(dtype=float)
        expected = np.log1p(df.loc[valid, "delivery_distance_km"].to_numpy(dtype=float))
        assert np.isfinite(actual).all() and (actual >= 0).all() and np.allclose(actual, expected, rtol=1e-5, atol=1e-8), "Validation 13 failed: log transform inconsistent."
    checks.append("Check 13 passed: log transform and missingness are consistent.")
    # 14
    for col in predictors:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Validation 14 failed: predictor {col!r} is not numeric."
    checks.append("Check 14 passed: all predictors are numeric.")
    # 15
    for col in dummy_columns:
        assert col in df.columns, f"Validation 15 failed: dummy {col!r} missing."
        assert pd.api.types.is_integer_dtype(df[col].dtype) and set(df[col].unique()).issubset({0, 1}), f"Validation 15 failed: dummy {col!r} must be integer 0/1."
    checks.append("Check 15 passed: dummy features contain integer 0/1.")
    # 16
    assert TARGET not in predictors, "Validation 16 failed: target leakage in predictors."
    checks.append("Check 16 passed: target excluded from predictors.")
    # 17
    assert df.columns[-1] == TARGET and list(df.columns[:-1]) == predictors, "Validation 17 failed: target/order incorrect."
    checks.append("Check 17 passed: target is the final column.")
    # 18
    assert list(df.columns) == predictors + [TARGET], "Validation 18 failed: schema mismatch."
    checks.append("Check 18 passed: schema matches predictor list.")
    logger.info("All exactly 18 validation checks passed.")
    return checks


def persist_and_verify_csv(
    df: pd.DataFrame,
    path: Path,
    expected_rows: int,
    expected_cols: int,
    expected_order: list[str],
    dummy_columns: list[str],
) -> None:
    """Save CSV and verify the persisted data by reloading it."""
    try:
        df.to_csv(path, index=False)
        saved = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not save/reload output CSV: {exc}") from exc
    if saved.shape != (expected_rows, expected_cols):
        raise RuntimeError(f"Saved CSV shape mismatch: expected {(expected_rows, expected_cols)}, got {saved.shape}.")
    if list(saved.columns) != expected_order:
        raise RuntimeError("Saved CSV column names/order do not match.")
    for col in expected_order:
        if col != TARGET and not pd.api.types.is_numeric_dtype(saved[col]):
            raise RuntimeError(f"Saved predictor {col!r} is not numeric.")
    for col in dummy_columns:
        if not pd.api.types.is_integer_dtype(saved[col].dtype) or not set(saved[col].unique()).issubset({0, 1}):
            raise RuntimeError(f"Saved dummy {col!r} does not contain integer 0/1 values.")
    target = pd.to_numeric(saved[TARGET], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all() or not (target > 0).all():
        raise RuntimeError("Saved target contains missing, non-finite, or non-positive values.")
    distance = pd.to_numeric(saved["delivery_distance_km"], errors="coerce")
    d = distance.dropna().to_numpy(dtype=float)
    if not np.isfinite(d).all() or (d < 0).any():
        raise RuntimeError("Saved distance contains invalid values.")
    log_distance = pd.to_numeric(saved["log1p_delivery_distance_km"], errors="coerce")
    valid = distance.notna()
    if not log_distance.loc[~valid].isna().all():
        raise RuntimeError("Saved missing distances do not have missing log-distance values.")
    if valid.any():
        actual = log_distance.loc[valid].to_numpy(dtype=float)
        expected = np.log1p(distance.loc[valid].to_numpy(dtype=float))
        if not np.isfinite(actual).all() or (actual < 0).any() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-8):
            raise RuntimeError("Saved log-distance values are inconsistent.")
    geo = saved["is_geo_valid"]
    if not pd.api.types.is_integer_dtype(geo.dtype) or not set(geo.unique()).issubset({0, 1}):
        raise RuntimeError("Saved is_geo_valid must contain integer 0/1 values.")
    logger.info("CSV read-back verification passed.")


def run_feature_engineering_pipeline() -> None:
    root, input_path, output_path, summary_path = get_project_paths()
    logger.info("=" * 72)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("Project root: %s", root)
    logger.info("Input: %s", input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_and_validate_cleaned_input(input_path)
    original_rows, original_cols = raw.shape
    duplicate_count = int(raw.duplicated().sum())
    target_valid, removed = validate_and_filter_target(raw)
    geo_df, geo_audit = engineer_geographic_features(target_valid)
    numeric_df, numeric_audit = engineer_numeric_features(geo_df)
    encoded, dummy_columns, categorical_audit = engineer_categorical_features(numeric_df)
    final, predictors, exclusion_audit = prevent_target_leakage_and_select_features(encoded)

    checks = execute_exact_18_validations(final, original_rows, removed, predictors, dummy_columns)
    persist_and_verify_csv(final, output_path, len(final), final.shape[1], list(final.columns), dummy_columns)

    numeric_predictors = [
        c for c in predictors
        if c in BASE_NUMERIC or c in ["delivery_distance_km", "log1p_delivery_distance_km", "is_geo_valid"]
    ]
    summary = {
        "project_name": PROJECT_NAME,
        "module_name": MODULE_NAME,
        "generation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_dataset_path": str(input_path),
        "output_dataset_path": str(output_path),
        "summary_json_path": str(summary_path),
        "original_row_count": original_rows,
        "original_column_count": original_cols,
        "final_row_count": len(final),
        "final_column_count": final.shape[1],
        "rows_removed_due_to_invalid_target": removed,
        "duplicate_rows_in_original_dataset": duplicate_count,
        "target_variable": TARGET,
        "target_units": "minutes",
        "predictor_feature_names": predictors,
        "numeric_feature_names": numeric_predictors,
        "one_hot_encoded_feature_names": dummy_columns,
        "categorical_feature_audit": categorical_audit,
        "excluded_identifier_columns": exclusion_audit["excluded_identifiers"],
        "excluded_raw_coordinate_columns": exclusion_audit["excluded_raw_coordinates"],
        "missing_values_per_final_column": {c: int(final[c].isna().sum()) for c in final.columns},
        "geographic_audit": geo_audit,
        "numeric_feature_audit": numeric_audit,
        "validation_count_executed": len(checks),
        "validation_status": "EXACTLY_18_VALIDATIONS_PASSED",
        "validation_check_details": checks,
        "target_leakage_prevention": {
            "target_excluded_from_predictors": TARGET not in predictors,
            "target_placement": "final_column",
            "leakage_verification": "CONFIRMED_CLEAN",
        },
        "data_limitations_and_domain_notes": [
            "The target is food delivery time in minutes.",
            "Haversine distance is straight-line distance, not actual road-network distance.",
            "The source dataset does not directly provide measured traffic congestion, weather conditions, dispatch delays, or shipping costs.",
        ],
    }
    try:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(sanitize_for_strict_json(summary), handle, indent=4, allow_nan=False)
    except Exception as exc:
        raise RuntimeError(f"Could not write strict JSON summary: {exc}") from exc

    print("\n" + "=" * 72)
    print("FEATURE ENGINEERING PIPELINE COMPLETED")
    print("=" * 72)
    print(f"Project root:              {root}")
    print(f"Original shape:            {original_rows:,} rows x {original_cols:,} columns")
    print(f"Invalid target rows removed:{removed:>7,}")
    print(f"Final shape:               {len(final):,} rows x {final.shape[1]:,} columns")
    print(f"Predictor count:            {len(predictors)}")
    print(f"One-hot feature count:      {len(dummy_columns)}")
    print(f"Valid distances:            {geo_audit['valid_distance_count']:,}")
    print(f"Validation status:          EXACTLY 18 CHECKS PASSED")
    print(f"Output CSV:                 {output_path}")
    print(f"Summary JSON:               {summary_path}")
    print("=" * 72)


def main() -> None:
    try:
        run_feature_engineering_pipeline()
    except Exception as exc:
        logger.critical("PIPELINE EXECUTION FAILED: %s", exc, exc_info=True)
        print(f"\nPIPELINE EXECUTION FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()