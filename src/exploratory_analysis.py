"""
Exploratory Data Analysis for the Food Delivery Time Prediction Project.

Project:
    Predictive Modeling and Optimization in Logistics Systems Using Python

Input:
    data/processed/cleaned_delivery_data.csv

Outputs:
    outputs/evaluation/

This script generates dataset summaries, missing-value analysis,
numeric and categorical statistics, delivery-time analysis,
correlations, outlier summaries, data-quality checks, and a text report.

The input dataset is never modified by this script.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_delivery_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

TARGET_COLUMN = "Time_taken(min)"

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
    TARGET_COLUMN,
]

IDENTIFIER_COLUMNS = [
    "ID",
    "Delivery_person_ID",
]

COORDINATE_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


# ============================================================
# 2. LOGGING
# ============================================================

def configure_logging() -> None:
    """Configure console and file logging."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_file = OUTPUT_DIR / "exploratory_analysis.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file,
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


# ============================================================
# 3. DATA LOADING AND VALIDATION
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset and validate its required columns."""

    logging.info("Loading cleaned delivery dataset...")
    logging.info("Input file: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run src/data_preprocessing.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if df.empty:
        raise ValueError("The input dataset contains no rows.")

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicate_columns}"
        )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    logging.info(
        "Dataset loaded successfully: %s rows, %s columns.",
        f"{len(df):,}",
        len(df.columns),
    )

    logging.info(
        "Columns available for analysis: %s",
        ", ".join(df.columns.astype(str)),
    )

    return df


# ============================================================
# 4. COMMON HELPER FUNCTIONS
# ============================================================

def save_csv(
    data: pd.DataFrame,
    filename: str,
    index: bool = False,
) -> Path:
    """Save a DataFrame to the evaluation directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    data.to_csv(
        output_path,
        index=index,
        encoding="utf-8",
    )

    logging.info("Saved: %s", output_path)

    return output_path


def get_numeric_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a separate DataFrame containing numeric columns.

    Boolean columns and categorical identifiers are excluded.
    Numeric-looking text columns are converted when possible.
    Infinite values are treated as missing.
    """

    numeric_data: dict[str, pd.Series] = {}

    for column in df.columns:

        if pd.api.types.is_bool_dtype(df[column].dtype):
            continue

        if column in IDENTIFIER_COLUMNS:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if (
            pd.api.types.is_numeric_dtype(df[column].dtype)
            or converted.notna().any()
        ):
            numeric_data[column] = converted.replace(
                [np.inf, -np.inf],
                np.nan,
            )

    if not numeric_data:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(
        numeric_data,
        index=df.index,
    )


def get_safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Convert a column to numeric values safely."""

    if column not in df.columns:
        return pd.Series(
            index=df.index,
            dtype="float64",
            name=column,
        )

    if pd.api.types.is_bool_dtype(df[column].dtype):
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
            name=column,
        )

    series = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    return series.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def safe_float(value: Any) -> float:
    """Return a finite float or NaN."""

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError, OverflowError):
        pass

    return float("nan")


def format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format a numeric value for the text report."""

    number = safe_float(value)

    if not np.isfinite(number):
        return "Not available"

    return f"{number:,.{decimals}f}"


# ============================================================
# 5. DATASET OVERVIEW
# ============================================================

def analyze_dataset_overview(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate general dataset statistics."""

    logging.info("Generating dataset overview...")

    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())

    overview = pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Total Cells",
                "Missing Cells",
                "Missing Cells Percentage",
                "Exact Duplicate Rows",
                "Unique Column Names",
                "Memory Usage (MB)",
            ],
            "Value": [
                len(df),
                len(df.columns),
                total_cells,
                missing_cells,
                (
                    missing_cells / total_cells * 100
                    if total_cells
                    else 0.0
                ),
                int(df.duplicated().sum()),
                int(df.columns.nunique()),
                float(
                    df.memory_usage(deep=True).sum()
                    / (1024 ** 2)
                ),
            ],
        }
    )

    column_information = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Data_Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non_Missing_Count": [
                int(df[column].notna().sum())
                for column in df.columns
            ],
            "Missing_Count": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Unique_Values": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    save_csv(
        overview,
        "dataset_overview.csv",
    )

    save_csv(
        column_information,
        "column_information.csv",
    )

    return overview


# ============================================================
# 6. MISSING-VALUE ANALYSIS
# ============================================================

def analyze_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate missing-value counts and percentages."""

    logging.info("Analyzing missing values...")

    missing_counts = df.isna().sum()

    missing_percentages = (
        missing_counts / len(df) * 100
        if len(df)
        else missing_counts.astype(float)
    )

    summary = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Missing_Count": missing_counts.values,
            "Missing_Percentage": missing_percentages.values,
            "Non_Missing_Count": df.notna().sum().values,
        }
    )

    summary = summary.sort_values(
        by=["Missing_Count", "Column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    save_csv(
        summary,
        "missing_values_summary.csv",
    )

    logging.info(
        "Total missing cells: %s",
        f"{int(missing_counts.sum()):,}",
    )

    return summary


# ============================================================
# 7. NUMERIC DESCRIPTIVE STATISTICS
# ============================================================

def analyze_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric measurements."""

    logging.info("Calculating numeric descriptive statistics...")

    numeric_df = get_numeric_dataframe(df)

    statistic_columns = [
        "Column",
        "Count",
        "Missing_Count",
        "Mean",
        "Standard_Deviation",
        "Minimum",
        "25th_Percentile",
        "Median",
        "75th_Percentile",
        "Maximum",
        "Skewness",
    ]

    records: list[dict[str, Any]] = []

    for column in numeric_df.columns:
        series = numeric_df[column]

        values = series.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        values = values[np.isfinite(values)]

        missing_count = int(series.isna().sum())

        if len(values) == 0:
            records.append(
                {
                    "Column": column,
                    "Count": 0,
                    "Missing_Count": missing_count,
                    "Mean": np.nan,
                    "Standard_Deviation": np.nan,
                    "Minimum": np.nan,
                    "25th_Percentile": np.nan,
                    "Median": np.nan,
                    "75th_Percentile": np.nan,
                    "Maximum": np.nan,
                    "Skewness": np.nan,
                }
            )
            continue

        clean_series = pd.Series(
            values,
            dtype="float64",
        )

        records.append(
            {
                "Column": column,
                "Count": int(len(clean_series)),
                "Missing_Count": missing_count,
                "Mean": float(clean_series.mean()),
                "Standard_Deviation": (
                    float(clean_series.std(ddof=1))
                    if len(clean_series) > 1
                    else np.nan
                ),
                "Minimum": float(clean_series.min()),
                "25th_Percentile": float(
                    clean_series.quantile(0.25)
                ),
                "Median": float(
                    clean_series.quantile(0.50)
                ),
                "75th_Percentile": float(
                    clean_series.quantile(0.75)
                ),
                "Maximum": float(clean_series.max()),
                "Skewness": safe_float(
                    clean_series.skew()
                ),
            }
        )

    summary = pd.DataFrame(
        records,
        columns=statistic_columns,
    )

    save_csv(
        summary,
        "numeric_summary.csv",
    )

    logging.info(
        "Numeric descriptive statistics calculated for %s columns.",
        len(summary),
    )

    return summary


# ============================================================
# 8. CATEGORICAL FREQUENCY ANALYSIS
# ============================================================

def analyze_categorical_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize categorical columns.

    Identifier columns are excluded to avoid producing frequency
    tables containing thousands of individual identifiers.
    """

    logging.info("Analyzing categorical columns...")

    categorical_columns = [
        column
        for column in df.columns
        if (
            column not in IDENTIFIER_COLUMNS
            and not pd.api.types.is_numeric_dtype(df[column].dtype)
            and not pd.api.types.is_bool_dtype(df[column].dtype)
        )
    ]

    records: list[dict[str, Any]] = []

    for column in categorical_columns:
        logging.info(
            "Analyzing categorical column: %s",
            column,
        )

        frequencies = df[column].value_counts(
            dropna=False,
            sort=True,
        )

        unique_count = int(
            df[column].nunique(dropna=True)
        )

        for value, count in frequencies.items():
            display_value = (
                "<MISSING>"
                if pd.isna(value)
                else str(value)
            )

            records.append(
                {
                    "Column": column,
                    "Value": display_value,
                    "Count": int(count),
                    "Percentage": (
                        int(count) / len(df) * 100
                        if len(df)
                        else 0.0
                    ),
                    "Unique_Values_In_Column": unique_count,
                }
            )

    summary = pd.DataFrame(
        records,
        columns=[
            "Column",
            "Value",
            "Count",
            "Percentage",
            "Unique_Values_In_Column",
        ],
    )

    save_csv(
        summary,
        "categorical_summary.csv",
    )

    logging.info(
        "Categorical frequency analysis completed for %s columns.",
        len(categorical_columns),
    )

    return summary


# ============================================================
# 9. DELIVERY-TIME ANALYSIS
# ============================================================

def analyze_delivery_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze the delivery-time target in minutes."""

    logging.info("Analyzing delivery-time distribution...")

    target = get_safe_numeric_series(
        df,
        TARGET_COLUMN,
    )

    values = target.dropna().to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(
            f"No valid numeric observations were found in "
            f"'{TARGET_COLUMN}'."
        )

    target = pd.Series(
        values,
        dtype="float64",
    )

    q25 = float(target.quantile(0.25))
    median = float(target.quantile(0.50))
    q75 = float(target.quantile(0.75))

    summary = pd.DataFrame(
        [
            {
                "Target_Column": TARGET_COLUMN,
                "Total_Rows": len(df),
                "Valid_Observations": len(target),
                "Missing_Or_Invalid_Observations": (
                    len(df) - len(target)
                ),
                "Mean_Minutes": float(target.mean()),
                "Median_Minutes": median,
                "Standard_Deviation_Minutes": (
                    float(target.std(ddof=1))
                    if len(target) > 1
                    else np.nan
                ),
                "Minimum_Minutes": float(target.min()),
                "25th_Percentile_Minutes": q25,
                "75th_Percentile_Minutes": q75,
                "Maximum_Minutes": float(target.max()),
                "Range_Minutes": float(
                    target.max() - target.min()
                ),
                "Interquartile_Range_Minutes": float(
                    q75 - q25
                ),
                "Skewness": safe_float(target.skew()),
                "Negative_Or_Zero_Times": int(
                    (target <= 0).sum()
                ),
            }
        ]
    )

    save_csv(
        summary,
        "delivery_time_summary.csv",
    )

    logging.info(
        "Valid delivery-time observations: %s",
        f"{len(target):,}",
    )

    logging.info(
        "Mean delivery time: %s minutes.",
        format_number(target.mean()),
    )

    logging.info(
        "Median delivery time: %s minutes.",
        format_number(target.median()),
    )

    return summary


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson correlations between numeric measurements."""

    logging.info("Calculating numeric correlations...")

    numeric_df = get_numeric_dataframe(df)

    if numeric_df.shape[1] < 2:
        correlation_matrix = pd.DataFrame()
        target_summary = pd.DataFrame(
            columns=[
                "Feature",
                "Pearson_Correlation_With_Target",
                "Absolute_Correlation",
            ]
        )

    else:
        correlation_matrix = numeric_df.corr(
            method="pearson",
            min_periods=2,
        )

        if TARGET_COLUMN in correlation_matrix.columns:
            target_correlations = correlation_matrix[
                TARGET_COLUMN
            ].drop(
                labels=[TARGET_COLUMN],
                errors="ignore",
            )

            target_summary = pd.DataFrame(
                {
                    "Feature": target_correlations.index,
                    "Pearson_Correlation_With_Target": (
                        target_correlations.values
                    ),
                    "Absolute_Correlation": (
                        target_correlations.abs().values
                    ),
                }
            )

            target_summary = target_summary.sort_values(
                by="Absolute_Correlation",
                ascending=False,
                na_position="last",
            ).reset_index(drop=True)

        else:
            target_summary = pd.DataFrame(
                columns=[
                    "Feature",
                    "Pearson_Correlation_With_Target",
                    "Absolute_Correlation",
                ]
            )

    save_csv(
        correlation_matrix,
        "correlation_matrix.csv",
        index=True,
    )

    save_csv(
        target_summary,
        "target_correlation_summary.csv",
    )

    logging.info("Correlation analysis completed.")

    return correlation_matrix, target_summary


# ============================================================
# 11. IQR OUTLIER ANALYSIS
# ============================================================

def analyze_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag potential outliers using the interquartile range.

    No observations are removed.
    """

    logging.info("Detecting potential statistical outliers...")

    numeric_df = get_numeric_dataframe(df)

    records: list[dict[str, Any]] = []

    for column in numeric_df.columns:
        series = numeric_df[column]

        values = series.dropna().to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            records.append(
                {
                    "Column": column,
                    "Valid_Count": 0,
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "Lower_Bound": np.nan,
                    "Upper_Bound": np.nan,
                    "Outlier_Count": 0,
                    "Outlier_Percentage": 0.0,
                    "Minimum": np.nan,
                    "Maximum": np.nan,
                }
            )
            continue

        clean_series = pd.Series(
            values,
            dtype="float64",
        )

        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (clean_series < lower_bound)
            | (clean_series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        records.append(
            {
                "Column": column,
                "Valid_Count": len(clean_series),
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
                "Outlier_Count": outlier_count,
                "Outlier_Percentage": (
                    outlier_count / len(clean_series) * 100
                ),
                "Minimum": float(clean_series.min()),
                "Maximum": float(clean_series.max()),
            }
        )

    summary = pd.DataFrame(records)

    save_csv(
        summary,
        "outlier_summary.csv",
    )

    total_flags = (
        int(summary["Outlier_Count"].sum())
        if not summary.empty
        else 0
    )

    logging.info(
        "Potential outlier flags across columns: %s",
        f"{total_flags:,}",
    )

    logging.info(
        "Outlier flags are column-specific. "
        "No rows were removed."
    )

    return summary


# ============================================================
# 12. DOMAIN-SPECIFIC DATA QUALITY CHECKS
# ============================================================

def analyze_domain_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag selected data-quality concerns for further review."""

    logging.info("Checking selected data-quality conditions...")

    records: list[dict[str, Any]] = []

    def add_check(
        name: str,
        description: str,
        count: int,
    ) -> None:
        records.append(
            {
                "Check": name,
                "Description": description,
                "Flagged_Count": int(count),
                "Percentage_Of_Rows": (
                    int(count) / len(df) * 100
                    if len(df)
                    else 0.0
                ),
            }
        )

    add_check(
        "Exact duplicate rows",
        "Rows identical across every column.",
        int(df.duplicated().sum()),
    )

    if "ID" in df.columns:
        id_values = df["ID"].dropna()

        repeated_ids = (
            int(df["ID"].notna().sum())
            - int(id_values.nunique())
        )

        add_check(
            "Repeated non-missing IDs",
            "Repeated identifier values; these are not automatically "
            "treated as duplicate orders.",
            repeated_ids,
        )

    if "Delivery_person_Age" in df.columns:
        age = get_safe_numeric_series(
            df,
            "Delivery_person_Age",
        )

        invalid_age = age.notna() & (
            (age < 18) | (age > 100)
        )

        add_check(
            "Age outside review range",
            "Age values below 18 or above 100.",
            int(invalid_age.sum()),
        )

    if "Delivery_person_Ratings" in df.columns:
        rating = get_safe_numeric_series(
            df,
            "Delivery_person_Ratings",
        )

        invalid_rating = rating.notna() & (
            (rating < 0) | (rating > 5)
        )

        add_check(
            "Ratings outside 0 to 5",
            "Numeric ratings outside the expected 0-to-5 range.",
            int(invalid_rating.sum()),
        )

    target = get_safe_numeric_series(
        df,
        TARGET_COLUMN,
    )

    non_positive_target = target.notna() & (target <= 0)

    add_check(
        "Non-positive delivery times",
        "Delivery-time values that are zero or negative.",
        int(non_positive_target.sum()),
    )

    if all(
        column in df.columns
        for column in COORDINATE_COLUMNS
    ):
        coordinates = pd.DataFrame(
            {
                column: get_safe_numeric_series(df, column)
                for column in COORDINATE_COLUMNS
            },
            index=df.index,
        )

        restaurant_coordinates = coordinates[
            [
                "Restaurant_latitude",
                "Restaurant_longitude",
            ]
        ]

        delivery_coordinates = coordinates[
            [
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            ]
        ]

        restaurant_zero = (
            restaurant_coordinates.eq(0).all(axis=1)
            & restaurant_coordinates.notna().all(axis=1)
        )

        delivery_zero = (
            delivery_coordinates.eq(0).all(axis=1)
            & delivery_coordinates.notna().all(axis=1)
        )

        add_check(
            "Restaurant coordinates both zero",
            "Restaurant latitude and longitude are both zero.",
            int(restaurant_zero.sum()),
        )

        add_check(
            "Delivery coordinates both zero",
            "Delivery latitude and longitude are both zero.",
            int(delivery_zero.sum()),
        )

        all_coordinates_zero = (
            coordinates.eq(0).all(axis=1)
            & coordinates.notna().all(axis=1)
        )

        add_check(
            "All four coordinates zero",
            "All restaurant and delivery coordinate values are zero.",
            int(all_coordinates_zero.sum()),
        )

    summary = pd.DataFrame(
        records,
        columns=[
            "Check",
            "Description",
            "Flagged_Count",
            "Percentage_Of_Rows",
        ],
    )

    save_csv(
        summary,
        "domain_anomaly_summary.csv",
    )

    return summary


# ============================================================
# 13. TEXT REPORT
# ============================================================

def generate_text_report(
    df: pd.DataFrame,
    overview: pd.DataFrame,
    missing_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    delivery_summary: pd.DataFrame,
    target_correlations: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
) -> Path:
    """Generate a consolidated human-readable EDA report."""

    logging.info("Generating the EDA text report...")

    lines: list[str] = []

    def add_section(title: str) -> None:
        lines.extend(
            [
                "",
                "=" * 72,
                title,
                "=" * 72,
            ]
        )

    lines.extend(
        [
            "=" * 72,
            "FOOD DELIVERY TIME PREDICTION",
            "EXPLORATORY DATA ANALYSIS REPORT",
            "=" * 72,
            "",
            "PROJECT",
            (
                "Predictive Modeling and Optimization in Logistics "
                "Systems Using Python"
            ),
            "",
            "INPUT DATASET",
            str(INPUT_FILE),
        ]
    )

    add_section("1. DATASET OVERVIEW")

    for _, row in overview.iterrows():
        lines.append(
            f"{row['Metric']}: {row['Value']}"
        )

    add_section("2. MISSING-VALUE ANALYSIS")

    columns_with_missing = missing_summary[
        missing_summary["Missing_Count"] > 0
    ]

    if columns_with_missing.empty:
        lines.append("No missing values were detected.")

    else:
        for _, row in columns_with_missing.iterrows():
            lines.append(
                f"{row['Column']}: "
                f"{int(row['Missing_Count']):,} missing "
                f"({format_number(row['Missing_Percentage'])}%)."
            )

    add_section("3. DELIVERY-TIME SUMMARY")

    if not delivery_summary.empty:
        row = delivery_summary.iloc[0]

        report_metrics = [
            ("Valid observations", "Valid_Observations"),
            ("Mean (minutes)", "Mean_Minutes"),
            ("Median (minutes)", "Median_Minutes"),
            (
                "Standard deviation (minutes)",
                "Standard_Deviation_Minutes",
            ),
            ("Minimum (minutes)", "Minimum_Minutes"),
            ("Maximum (minutes)", "Maximum_Minutes"),
            (
                "Interquartile range (minutes)",
                "Interquartile_Range_Minutes",
            ),
        ]

        for label, column in report_metrics:
            value = row[column]

            if column == "Valid_Observations":
                formatted = f"{int(value):,}"
            else:
                formatted = format_number(value)

            lines.append(f"{label}: {formatted}")

    add_section("4. NUMERIC DESCRIPTIVE STATISTICS")

    if numeric_summary.empty:
        lines.append("No numeric measurement columns were available.")

    else:
        for _, row in numeric_summary.iterrows():
            lines.append(str(row["Column"]))
            lines.append(
                f"  Count: {int(row['Count']):,}"
            )
            lines.append(
                f"  Mean: {format_number(row['Mean'])}"
            )
            lines.append(
                f"  Median: {format_number(row['Median'])}"
            )
            lines.append(
                f"  Minimum: {format_number(row['Minimum'])}"
            )
            lines.append(
                f"  Maximum: {format_number(row['Maximum'])}"
            )

    add_section("5. CATEGORICAL FREQUENCIES")

    if categorical_summary.empty:
        lines.append("No categorical columns were available.")

    else:
        for column in categorical_summary["Column"].drop_duplicates():
            column_data = categorical_summary[
                categorical_summary["Column"] == column
            ]

            unique_count = int(
                column_data["Unique_Values_In_Column"].max()
            )

            lines.append(
                f"{column}: {unique_count} unique non-missing values"
            )

            for _, row in column_data.head(5).iterrows():
                lines.append(
                    f"  {row['Value']}: "
                    f"{int(row['Count']):,} "
                    f"({format_number(row['Percentage'])}%)"
                )

    add_section("6. CORRELATION WITH DELIVERY TIME")

    if target_correlations.empty:
        lines.append("No target correlations were available.")

    else:
        valid_correlations = target_correlations.dropna(
            subset=["Pearson_Correlation_With_Target"]
        )

        if valid_correlations.empty:
            lines.append("No valid target correlations were available.")

        else:
            for _, row in valid_correlations.head(10).iterrows():
                lines.append(
                    f"{row['Feature']}: "
                    f"{format_number(row['Pearson_Correlation_With_Target'], 4)}"
                )

    lines.append(
        "Note: Pearson correlation measures linear association, "
        "not causation."
    )

    add_section("7. POTENTIAL STATISTICAL OUTLIERS")

    flagged = (
        outlier_summary[
            outlier_summary["Outlier_Count"] > 0
        ].sort_values(
            by="Outlier_Count",
            ascending=False,
        )
        if not outlier_summary.empty
        else pd.DataFrame()
    )

    if flagged.empty:
        lines.append("No values were flagged by the IQR rule.")

    else:
        for _, row in flagged.iterrows():
            lines.append(
                f"{row['Column']}: "
                f"{int(row['Outlier_Count']):,} flagged "
                f"({format_number(row['Outlier_Percentage'])}%)."
            )

    lines.append(
        "IQR flags identify unusual values, not necessarily incorrect data."
    )
    lines.append("No rows were removed during outlier analysis.")

    add_section("8. DOMAIN-SPECIFIC DATA QUALITY CHECKS")

    for _, row in anomaly_summary.iterrows():
        lines.append(
            f"{row['Check']}: "
            f"{int(row['Flagged_Count']):,} flagged "
            f"({format_number(row['Percentage_Of_Rows'])}%)."
        )

        lines.append(
            f"  {row['Description']}"
        )

    add_section("9. GENERATED OUTPUTS")

    output_files = [
        "dataset_overview.csv",
        "column_information.csv",
        "missing_values_summary.csv",
        "numeric_summary.csv",
        "categorical_summary.csv",
        "delivery_time_summary.csv",
        "correlation_matrix.csv",
        "target_correlation_summary.csv",
        "outlier_summary.csv",
        "domain_anomaly_summary.csv",
        "eda_report.txt",
        "exploratory_analysis.log",
    ]

    for filename in output_files:
        lines.append(str(OUTPUT_DIR / filename))

    add_section("10. LIMITATIONS")

    lines.extend(
        [
            "- The report describes the supplied dataset.",
            "- Correlation does not establish causation.",
            "- Statistical outliers are not automatically data errors.",
            "- Coordinate checks identify records for review only.",
            "- This script does not train or evaluate prediction models.",
            "- The cleaned dataset is not modified by this script.",
        ]
    )

    lines.extend(
        [
            "",
            "=" * 72,
            "END OF REPORT",
            "=" * 72,
        ]
    )

    report_path = OUTPUT_DIR / "eda_report.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logging.info("Saved EDA report: %s", report_path)

    return report_path


# ============================================================
# 14. MAIN PIPELINE
# ============================================================

def main() -> int:
    """Run the complete exploratory analysis pipeline."""

    configure_logging()

    logging.info("=" * 72)
    logging.info("FOOD DELIVERY TIME - EXPLORATORY DATA ANALYSIS")
    logging.info("=" * 72)

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Output directory: %s", OUTPUT_DIR)

    try:
        df = load_dataset()

        overview = analyze_dataset_overview(df)

        missing_summary = analyze_missing_values(df)

        numeric_summary = analyze_numeric_columns(df)

        categorical_summary = analyze_categorical_columns(df)

        delivery_summary = analyze_delivery_time(df)

        _, target_correlations = analyze_correlations(df)

        outlier_summary = analyze_outliers(df)

        anomaly_summary = analyze_domain_anomalies(df)

        report_path = generate_text_report(
            df=df,
            overview=overview,
            missing_summary=missing_summary,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            delivery_summary=delivery_summary,
            target_correlations=target_correlations,
            outlier_summary=outlier_summary,
            anomaly_summary=anomaly_summary,
        )

        logging.info("=" * 72)
        logging.info("EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info("=" * 72)

        logging.info(
            "Dataset analyzed: %s rows x %s columns.",
            f"{len(df):,}",
            len(df.columns),
        )

        logging.info("Results directory: %s", OUTPUT_DIR)
        logging.info("EDA report: %s", report_path)

        return 0

    except Exception:
        logging.exception("EDA pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
