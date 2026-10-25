"""
Visualize Food Delivery Time Data
=================================

Project:
    YuvaIntern Week 4 - Predictive Modeling and Optimization
    in Logistics Systems Using Python

Purpose:
    Generate professional visualizations for the cleaned Kaggle
    food-delivery dataset.

Input:
    data/processed/cleaned_delivery_data.csv

Output:
    outputs/figures/
        01_delivery_time_distribution.png
        02_delivery_time_boxplot.png
        03_delivery_time_by_vehicle.png
        04_delivery_time_by_order_type.png
        05_delivery_distance_distribution.png
        06_distance_vs_delivery_time.png
        07_mean_delivery_time_by_vehicle.png
        08_mean_delivery_time_by_order_type.png
        09_correlation_heatmap.png

Notes:
    - Does not modify the input dataset.
    - Uses only columns available in the cleaned dataset.
    - Excludes invalid geographic records from distance analysis
      when the is_geo_valid column is available.
    - Saves figures at 300 DPI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend for reliable file generation.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

TARGET_COLUMN = "Time_taken(min)"
DISTANCE_COLUMN = "delivery_distance_km"
GEO_VALID_COLUMN = "is_geo_valid"

VEHICLE_COLUMN = "Type_of_vehicle"
ORDER_TYPE_COLUMN = "Type_of_order"

NUMERIC_COLUMNS = [
    TARGET_COLUMN,
    DISTANCE_COLUMN,
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

CORRELATION_CANDIDATES = [
    TARGET_COLUMN,
    DISTANCE_COLUMN,
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

DPI = 300

FIGURE_FILES = [
    "01_delivery_time_distribution.png",
    "02_delivery_time_boxplot.png",
    "03_delivery_time_by_vehicle.png",
    "04_delivery_time_by_order_type.png",
    "05_delivery_distance_distribution.png",
    "06_distance_vs_delivery_time.png",
    "07_mean_delivery_time_by_vehicle.png",
    "08_mean_delivery_time_by_order_type.png",
    "09_correlation_heatmap.png",
]


# ============================================================
# 2. LOGGING AND VISUALIZATION STYLE
# ============================================================

def configure_logging() -> None:
    """Configure console and file logging."""

    log_dir = PROJECT_ROOT / "outputs" / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "visualization.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
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


def configure_visual_style() -> None:
    """Apply a consistent style to all generated figures."""

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font_scale=1.0,
    )

    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.autolayout": False,
        }
    )


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load and validate the cleaned delivery dataset."""

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Input dataset: %s", INPUT_FILE)
    logging.info("Figure directory: %s", OUTPUT_DIR)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset was not found:\n{INPUT_FILE}\n\n"
            "Run src/data_preprocessing.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The cleaned dataset contains no rows.")

    # Handle accidental whitespace in column names.
    df.columns = df.columns.astype(str).str.strip()

    # Support a target column with surrounding whitespace.
    target_matches = [
        column
        for column in df.columns
        if column.strip() == TARGET_COLUMN
    ]

    if TARGET_COLUMN not in df.columns and target_matches:
        df = df.rename(
            columns={target_matches[0]: TARGET_COLUMN}
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert numeric columns without stripping signs or digits.
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Normalize categorical values.
    for column in [VEHICLE_COLUMN, ORDER_TYPE_COLUMN]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # Remove invalid target values from visualization data only.
    original_rows = len(df)

    df = df.loc[
        df[TARGET_COLUMN].notna()
        & np.isfinite(df[TARGET_COLUMN])
        & (df[TARGET_COLUMN] > 0)
    ].copy()

    removed_rows = original_rows - len(df)

    logging.info("Rows loaded: %s", original_rows)
    logging.info(
        "Rows excluded due to invalid delivery time: %s",
        removed_rows,
    )
    logging.info("Rows available for visualization: %s", len(df))

    if df.empty:
        raise ValueError(
            "No valid positive delivery-time records remain."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return df


# ============================================================
# 4. COMMON HELPER FUNCTIONS
# ============================================================

def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    """Save a figure and close it to release memory."""

    output_path = OUTPUT_DIR / filename

    try:
        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        logging.info("Saved figure: %s", output_path)

    finally:
        plt.close(fig)


def valid_category_data(
    df: pd.DataFrame,
    category_column: str,
) -> pd.DataFrame:
    """Return rows with a usable categorical value and valid target."""

    if category_column not in df.columns:
        raise KeyError(
            f"Required category column '{category_column}' is missing."
        )

    result = df.loc[
        df[category_column].notna()
        & df[TARGET_COLUMN].notna()
    ].copy()

    result[category_column] = (
        result[category_column]
        .astype(str)
        .str.strip()
    )

    result = result.loc[
        result[category_column] != ""
    ].copy()

    if result.empty:
        raise ValueError(
            f"No usable observations for '{category_column}'."
        )

    return result


def valid_distance_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return observations suitable for distance-based visualization.

    When is_geo_valid exists, only explicitly valid records are used.
    Otherwise, coordinate validity is checked where possible.

    Distances are straight-line Haversine estimates, not road distances.
    """

    if DISTANCE_COLUMN not in df.columns:
        raise KeyError(
            f"Required distance column '{DISTANCE_COLUMN}' is missing. "
            "Run src/data_preprocessing.py first."
        )

    result = df.copy()

    result[DISTANCE_COLUMN] = pd.to_numeric(
        result[DISTANCE_COLUMN],
        errors="coerce",
    )

    valid = (
        result[DISTANCE_COLUMN].notna()
        & np.isfinite(result[DISTANCE_COLUMN])
        & (result[DISTANCE_COLUMN] > 0)
        & (result[DISTANCE_COLUMN] < 1000)
    )

    # Prefer the preprocessing pipeline's geographic validity flag.
    if GEO_VALID_COLUMN in result.columns:
        geo_flag = result[GEO_VALID_COLUMN]

        if pd.api.types.is_bool_dtype(geo_flag):
            parsed_geo_flag = geo_flag
        else:
            parsed_geo_flag = (
                geo_flag.astype("string")
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                )
            )

        valid &= parsed_geo_flag.eq(True).fillna(False)

    else:
        # Fall back to checking geographic coordinates if the flag
        # is absent. Coordinates should be within valid latitude and
        # longitude ranges and should not all be zero.
        coordinate_columns = [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]

        if all(
            column in result.columns
            for column in coordinate_columns
        ):
            for column in coordinate_columns:
                result[column] = pd.to_numeric(
                    result[column],
                    errors="coerce",
                )

            valid &= (
                result["Restaurant_latitude"].between(-90, 90)
                & result["Delivery_location_latitude"].between(-90, 90)
                & result["Restaurant_longitude"].between(-180, 180)
                & result["Delivery_location_longitude"].between(-180, 180)
                & (result["Restaurant_latitude"] != 0)
                & (result["Restaurant_longitude"] != 0)
                & (result["Delivery_location_latitude"] != 0)
                & (result["Delivery_location_longitude"] != 0)
            )

    result = result.loc[valid].copy()

    logging.info(
        "Geographically eligible observations: %s",
        len(result),
    )

    if result.empty:
        raise ValueError(
            "No valid distance observations are available."
        )

    return result


# ============================================================
# 5. DELIVERY-TIME DISTRIBUTION
# ============================================================

def plot_delivery_time_distribution(
    df: pd.DataFrame,
) -> None:
    """Plot the distribution of delivery time."""

    values = df[TARGET_COLUMN].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        values,
        bins=35,
        kde=True,
        ax=ax,
    )

    mean_value = values.mean()
    median_value = values.median()

    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.2f} min",
    )

    ax.axvline(
        median_value,
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_value:.2f} min",
    )

    ax.set_title("Distribution of Food Delivery Time")
    ax.set_xlabel("Delivery time (minutes)")
    ax.set_ylabel("Number of deliveries")
    ax.legend()

    save_figure(
        fig,
        "01_delivery_time_distribution.png",
    )


def plot_delivery_time_boxplot(
    df: pd.DataFrame,
) -> None:
    """Plot a boxplot of delivery time."""

    values = df[TARGET_COLUMN].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        x=values,
        ax=ax,
    )

    ax.set_title("Delivery Time: Spread and Potential Outliers")
    ax.set_xlabel("Delivery time (minutes)")

    save_figure(
        fig,
        "02_delivery_time_boxplot.png",
    )


# ============================================================
# 6. DELIVERY TIME BY CATEGORY
# ============================================================

def plot_delivery_time_by_category(
    df: pd.DataFrame,
    category_column: str,
    filename: str,
    title: str,
    x_label: str,
) -> None:
    """Create a boxplot of delivery time for each category."""

    category_data = valid_category_data(
        df,
        category_column,
    )

    # Order categories by their median delivery time.
    category_order = (
        category_data.groupby(
            category_column,
            observed=True,
        )[TARGET_COLUMN]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=category_data,
        x=category_column,
        y=TARGET_COLUMN,
        order=category_order,
        hue=category_column,
        legend=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Delivery time (minutes)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    save_figure(fig, filename)


def plot_mean_delivery_time_by_category(
    df: pd.DataFrame,
    category_column: str,
    filename: str,
    title: str,
    x_label: str,
) -> None:
    """Compare mean delivery time and sample size by category."""

    category_data = valid_category_data(
        df,
        category_column,
    )

    summary = (
        category_data.groupby(
            category_column,
            observed=True,
        )[TARGET_COLUMN]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = sns.barplot(
        data=summary,
        x=category_column,
        y="mean",
        hue=category_column,
        legend=False,
        ax=ax,
    )

    # Include the sample count above each bar.
    for patch, (_, row) in zip(
        bars.patches,
        summary.iterrows(),
    ):
        height = patch.get_height()

        if np.isfinite(height):
            ax.annotate(
                f"n={int(row['count']):,}",
                (
                    patch.get_x() + patch.get_width() / 2,
                    height,
                ),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 4),
                textcoords="offset points",
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean delivery time (minutes)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    save_figure(fig, filename)


# ============================================================
# 7. DISTANCE ANALYSIS
# ============================================================

def plot_delivery_distance_distribution(
    df: pd.DataFrame,
) -> None:
    """Plot the distribution of valid straight-line distances."""

    distance_df = valid_distance_data(df)

    values = distance_df[DISTANCE_COLUMN]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        values,
        bins=35,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        "Distribution of Valid Straight-Line Delivery Distances"
    )
    ax.set_xlabel("Estimated straight-line distance (km)")
    ax.set_ylabel("Number of deliveries")

    ax.text(
        0.98,
        0.95,
        f"Valid records: {len(values):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    save_figure(
        fig,
        "05_delivery_distance_distribution.png",
    )


def plot_distance_vs_delivery_time(
    df: pd.DataFrame,
) -> None:
    """Plot the relationship between distance and delivery time."""

    distance_df = valid_distance_data(df)

    plot_df = distance_df.loc[
        distance_df[TARGET_COLUMN].notna(),
        [DISTANCE_COLUMN, TARGET_COLUMN],
    ].copy()

    if plot_df.empty:
        raise ValueError(
            "No valid distance and delivery-time pairs are available."
        )

    # A sample improves rendering speed for large datasets.
    # The underlying dataset is not modified.
    sample_size = min(8000, len(plot_df))

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(
            n=sample_size,
            random_state=42,
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(
        data=plot_df,
        x=DISTANCE_COLUMN,
        y=TARGET_COLUMN,
        scatter_kws={
            "alpha": 0.35,
            "s": 18,
        },
        line_kws={
            "linewidth": 2,
        },
        ax=ax,
    )

    ax.set_title(
        "Straight-Line Distance vs. Food Delivery Time"
    )
    ax.set_xlabel("Estimated straight-line distance (km)")
    ax.set_ylabel("Delivery time (minutes)")

    ax.text(
        0.02,
        0.98,
        (
            f"Plotted observations: {len(plot_df):,}\n"
            "Distance is not road-network distance."
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    save_figure(
        fig,
        "06_distance_vs_delivery_time.png",
    )


# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
) -> None:
    """Plot correlations among available numeric variables."""

    available_columns = [
        column
        for column in CORRELATION_CANDIDATES
        if column in df.columns
    ]

    if len(available_columns) < 2:
        raise ValueError(
            "At least two suitable numeric columns are required "
            "for a correlation heatmap."
        )

    numeric_df = df[available_columns].copy()

    for column in numeric_df.columns:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    # Correlations are calculated pairwise by pandas.
    # Do not imply that every correlation uses the same sample size.
    correlation = numeric_df.corr(
        method="pearson",
        min_periods=2,
    )

    if correlation.isna().all().all():
        raise ValueError(
            "Unable to calculate correlations from the available data."
        )

    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "label": "Pearson correlation coefficient",
        },
        ax=ax,
    )

    ax.set_title(
        "Correlation Heatmap of Delivery-Time Dataset"
    )

    save_figure(
        fig,
        "09_correlation_heatmap.png",
    )


# ============================================================
# 9. EXECUTION PIPELINE
# ============================================================

def main() -> int:
    """Execute the complete visualization pipeline."""

    configure_logging()
    configure_visual_style()

    logging.info("=" * 72)
    logging.info("FOOD DELIVERY DATA VISUALIZATION")
    logging.info("=" * 72)

    try:
        df = load_data()

    except Exception:
        logging.exception("Unable to load the cleaned dataset.")
        return 1

    tasks = [
        (
            "Delivery-time distribution",
            lambda: plot_delivery_time_distribution(df),
        ),
        (
            "Delivery-time boxplot",
            lambda: plot_delivery_time_boxplot(df),
        ),
        (
            "Delivery time by vehicle",
            lambda: plot_delivery_time_by_category(
                df,
                VEHICLE_COLUMN,
                "03_delivery_time_by_vehicle.png",
                "Delivery Time by Vehicle Type",
                "Vehicle type",
            ),
        ),
        (
            "Delivery time by order type",
            lambda: plot_delivery_time_by_category(
                df,
                ORDER_TYPE_COLUMN,
                "04_delivery_time_by_order_type.png",
                "Delivery Time by Order Type",
                "Order type",
            ),
        ),
        (
            "Delivery-distance distribution",
            lambda: plot_delivery_distance_distribution(df),
        ),
        (
            "Distance versus delivery time",
            lambda: plot_distance_vs_delivery_time(df),
        ),
        (
            "Mean delivery time by vehicle",
            lambda: plot_mean_delivery_time_by_category(
                df,
                VEHICLE_COLUMN,
                "07_mean_delivery_time_by_vehicle.png",
                "Mean Delivery Time by Vehicle Type",
                "Vehicle type",
            ),
        ),
        (
            "Mean delivery time by order type",
            lambda: plot_mean_delivery_time_by_category(
                df,
                ORDER_TYPE_COLUMN,
                "08_mean_delivery_time_by_order_type.png",
                "Mean Delivery Time by Order Type",
                "Order type",
            ),
        ),
        (
            "Correlation heatmap",
            lambda: plot_correlation_heatmap(df),
        ),
    ]

    successful = []
    failed = []

    for task_name, task_function in tasks:
        logging.info("-" * 60)
        logging.info("Generating: %s", task_name)

        try:
            task_function()
            successful.append(task_name)

        except Exception as exc:
            failed.append(
                {
                    "task": task_name,
                    "error": str(exc),
                }
            )

            logging.exception(
                "Failed to generate: %s",
                task_name,
            )

    logging.info("=" * 72)
    logging.info("VISUALIZATION SUMMARY")
    logging.info("=" * 72)
    logging.info("Successful charts: %s/%s", len(successful), len(tasks))
    logging.info("Failed charts: %s/%s", len(failed), len(tasks))
    logging.info("Output directory: %s", OUTPUT_DIR)

    for filename in FIGURE_FILES:
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            logging.info(
                "VERIFIED: %s (%s bytes)",
                filename,
                output_path.stat().st_size,
            )
        else:
            logging.warning(
                "NOT FOUND: %s",
                filename,
            )

    if failed:
        logging.error("The following charts could not be generated:")

        for item in failed:
            logging.error(
                "%s: %s",
                item["task"],
                item["error"],
            )

        return 1

    logging.info(
        "All %s visualizations were generated successfully.",
        len(successful),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())