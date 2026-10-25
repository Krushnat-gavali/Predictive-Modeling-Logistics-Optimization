# Predictive Modeling and Optimization in Logistics Systems Using Python

## Project Overview

This project applies data analysis, feature engineering, machine learning, and scenario-based optimization to food-delivery time prediction. It uses a real-world-style delivery dataset to investigate how delivery characteristics relate to the time taken to complete an order.

The project follows a modular Python workflow that includes data preprocessing, exploratory data analysis, geographic feature engineering, regression model development, hyperparameter tuning, model evaluation, and optimization scenario analysis.

A Random Forest regression model is trained to estimate delivery duration in minutes. Its performance is compared with a simple baseline and a Linear Regression model. Hyperparameter tuning is then used to investigate whether the Random Forest model can achieve better predictive performance.

The project also examines hypothetical delivery-distance reduction scenarios to understand how changing this model input affects predicted delivery time. These scenarios are analytical experiments, not guaranteed operational improvements.

## Project Objectives

The main objectives are to:

* Prepare the delivery dataset for machine learning.
* Identify missing values, unusual observations, and data-quality issues.
* Explore delivery-time distributions and relationships among available variables.
* Calculate approximate geographic distance using delivery coordinates.
* Develop and compare regression models for delivery-time prediction.
* Tune Random Forest hyperparameters using cross-validation.
* Evaluate models using standard regression metrics.
* Investigate hypothetical distance-reduction scenarios.
* Generate charts, evaluation tables, and machine-readable summaries.
* Organize the project into reusable and maintainable Python modules.

## Dataset

### Source

The project uses a food-delivery dataset obtained from Kaggle.

**Local raw dataset path:**

`data/raw/deliverytime.csv`

### Dataset Information

* **Number of records:** 45,593
* **Original columns:** 11
* **Prediction target:** `Time_taken(min)`
* **Problem type:** Supervised regression
* **Target unit:** Minutes

### Original Dataset Columns

| Column                        | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| `ID`                          | Order record identifier                         |
| `Delivery_person_ID`          | Delivery-person identifier                      |
| `Delivery_person_Age`         | Recorded age of the delivery person             |
| `Delivery_person_Ratings`     | Recorded delivery-person rating                 |
| `Restaurant_latitude`         | Restaurant latitude                             |
| `Restaurant_longitude`        | Restaurant longitude                            |
| `Delivery_location_latitude`  | Delivery destination latitude                   |
| `Delivery_location_longitude` | Delivery destination longitude                  |
| `Type_of_order`               | Order category                                  |
| `Type_of_vehicle`             | Vehicle category                                |
| `Time_taken(min)`             | Delivery duration in minutes; prediction target |

The preprocessing and feature-engineering stages add derived fields, including geographic-validity indicators and `delivery_distance_km`.

### Geographic Distance

The project calculates an approximate straight-line distance between the restaurant and delivery coordinates using the Haversine formula.

This measurement represents geographic distance rather than actual road distance. It does not account for road networks, traffic congestion, route restrictions, or navigation choices.

### Data Limitations

The supplied dataset does not directly include all operational factors that may influence delivery duration. In particular, it does not provide dedicated measurements for traffic conditions, weather conditions, kitchen preparation time, shipment weight, dispatch delay, or shipping cost.

Consequently, the project does not claim to measure these unavailable factors or establish their effects on delivery time.

## Technologies Used

* **Python** — Main programming language
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical calculations
* **Scikit-learn** — Machine learning, model evaluation, and hyperparameter tuning
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualizations
* **Joblib** — Saving and loading trained models
* **JSON and CSV** — Structured summaries and result storage
* **Git and GitHub** — Version control and project sharing

## Project Structure

```text
Predictive_Logistics_Optimization/
│
├── data/
│   ├── raw/
│   │   └── deliverytime.csv
│   └── processed/
│       ├── cleaned_delivery_data.csv
│       └── feature_engineered_delivery_data.csv
│
├── models/
│   ├── best_delivery_time_model.joblib
│   └── tuned_random_forest_delivery_model.joblib
│
├── outputs/
│   ├── evaluation/
│   ├── figures/
│   └── optimization/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── hyperparameter_tuning.py
│   ├── optimization_analysis.py
│   └── visualize_delivery_data.py
│
├── report/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The structure above illustrates the main project files and output locations. Some output files may be added as the workflow runs. The `report/` directory is reserved for the final project report.*

## Installation and Setup

### 1. Install Python

Install a compatible Python version and verify that the Python launcher is available.

Open PowerShell or the VS Code terminal and run:

```powershell
py --version
```

### 2. Open the Project Directory

```powershell
cd C:\Predictive_Logistics_Optimization
```

### 3. Install Dependencies

```powershell
py -m pip install -r requirements.txt
```

The dependencies are listed in `requirements.txt`.

### 4. Check the Dataset

Confirm that the raw CSV file exists at:

```text
data/raw/deliverytime.csv
```

The preprocessing stage must be able to read this file before the remaining workflow can run.

## Execution Workflow

Run the project modules from the project root directory in the following order.

### Step 1: Data Preprocessing

```powershell
py src\data_preprocessing.py
```

This stage prepares the raw dataset, checks data quality, handles applicable missing or invalid values, and creates a cleaned dataset.

Expected main outputs include:

* `data/processed/cleaned_delivery_data.csv`
* `outputs/evaluation/data_quality_summary.json`

### Step 2: Exploratory Data Analysis

```powershell
py src\exploratory_analysis.py
```

This stage summarizes the dataset, examines missing values and numerical distributions, and identifies potential outliers and unusual observations.

It generates CSV summaries and a text report under `outputs/`.

### Step 3: Feature Engineering

```powershell
py src\feature_engineering.py
```

This stage prepares model features, including the approximate geographic delivery distance, and transforms categorical variables into a format suitable for regression models.

Expected main output:

`data/processed/feature_engineered_delivery_data.csv`

### Step 4: Train and Compare Models

```powershell
py src\train_models.py
```

This stage trains and evaluates the baseline, Linear Regression, and Random Forest models.

The comparison results and prediction outputs are saved under `outputs/evaluation/`. The trained model artifacts are saved under `models/`.

### Step 5: Hyperparameter Tuning

```powershell
py src\hyperparameter_tuning.py
```

This stage evaluates selected Random Forest configurations using cross-validation and records the best configuration found during the search.

Expected outputs include tuning results, best hyperparameters, tuned-model predictions, and a saved tuned model.

### Step 6: Optimization Scenario Analysis

```powershell
py src\optimization_analysis.py
```

This stage evaluates hypothetical changes to delivery distance and examines their effect on model predictions.

It generates scenario summaries, feature-sensitivity results, charts, and recommendations under `outputs/optimization/`.

### Step 7: Generate Visualizations

```powershell
py src\visualize_delivery_data.py
```

This stage generates charts illustrating delivery-time distributions, delivery time by vehicle and order type, geographic distance, relationships between distance and delivery time, and numerical correlations.

The figures are saved under `outputs/figures/`.

**Important:** Run the scripts in the listed order when starting from the raw dataset. Each downstream stage depends on outputs created by earlier stages. If a script reports an error, resolve it before proceeding.

## Model Development

The project compares three regression approaches.

### 1. Dummy Regression Baseline

The baseline provides a simple reference prediction against which more sophisticated models can be compared.

### 2. Linear Regression

Linear Regression provides a relatively simple model for examining relationships between the available input features and delivery duration.

### 3. Random Forest Regression

Random Forest combines predictions from multiple decision trees. It can model nonlinear relationships and interactions that a basic linear model may not capture.

### Hyperparameter Tuning

The tuning stage searches selected Random Forest configurations using cross-validation. The best configuration is selected according to the search's evaluation criterion and then evaluated on a test set.

## Evaluation Metrics

The project uses the following metrics.

| Metric | Meaning                                                                                 | Preferred Direction |
| ------ | --------------------------------------------------------------------------------------- | ------------------- |
| MAE    | Mean Absolute Error; average absolute prediction error in minutes                       | Lower               |
| RMSE   | Root Mean Squared Error; penalizes larger errors more strongly                          | Lower               |
| R²     | Measures how much target variation is explained relative to a mean-prediction reference | Higher              |

MAE and RMSE are expressed in minutes because the target is delivery duration. R² is a dimensionless score and can be negative when a model performs worse than the relevant mean-prediction reference.

## Initial Model Results

The initial model-comparison run produced the following results.

| Model             | MAE (minutes) | RMSE (minutes) |                   R² |
| ----------------- | ------------: | -------------: | -------------------: |
| Dummy baseline    |        7.5790 |         9.3637 | Approximately 0.0000 |
| Linear Regression |        6.3559 |         8.0513 |               0.2607 |
| Random Forest     |        5.8585 |         7.4693 |               0.3637 |

In this run, Random Forest achieved the lowest MAE and RMSE and the highest R² among the three initial models.

### Tuned Random Forest Results

The hyperparameter-tuning stage reported:

* **Best cross-validation RMSE:** 7.3209 minutes
* **Test MAE:** 5.7337 minutes
* **Test RMSE:** 7.2987 minutes
* **Test R²:** 0.3924

The tuning process selected the following configuration:

| Parameter           | Selected Value |
| ------------------- | -------------- |
| `n_estimators`      | 200            |
| `min_samples_split` | 6              |
| `min_samples_leaf`  | 1              |
| `max_features`      | `sqrt`         |
| `max_depth`         | 10             |

These values describe the recorded project runs. Results may vary if the dataset, software environment, preprocessing logic, random seeds, or evaluation setup changes.

The initial and tuned results should not be treated as a perfectly controlled comparison unless both models are evaluated using the same test records and evaluation pipeline.

## Optimization Scenario Analysis

The optimization module examines hypothetical reductions in delivery distance by changing the distance feature and comparing model predictions.

The recorded scenario analysis reported these average predicted changes:

| Hypothetical Distance Reduction | Average Predicted Delivery-Time Change |
| ------------------------------- | -------------------------------------: |
| 5%                              |           Approximately −0.244 minutes |
| 10%                             |           Approximately −0.452 minutes |
| 15%                             |           Approximately −0.601 minutes |

These are model-based scenario results, not measured improvements from real delivery operations. For example, a predicted change of −0.601 minutes is approximately 36 seconds.

The analysis assumes that the model's predictions remain informative under the hypothetical feature changes. It does not demonstrate that a delivery company can achieve those distance reductions or that actual delivery times will improve by the predicted amounts.

The dataset contains straight-line geographic distance rather than actual route distance. Therefore, the scenario analysis should be interpreted as an exploratory sensitivity study, not a production route-optimization system.

## Output Files

### Data Quality and Analysis

The `outputs/` directory contains generated analytical summaries, including:

* Dataset overview and column information
* Missing-value summaries
* Numerical and categorical summaries
* Correlation summaries
* Outlier and domain-anomaly summaries
* Data-quality checks
* Exploratory analysis report

### Model Evaluation

The `outputs/evaluation/` directory includes generated files such as:

* `model_comparison.csv`
* `model_predictions.csv`
* `model_training_summary.json`
* `hyperparameter_tuning_results.csv`
* `best_hyperparameters.json`
* `tuned_model_predictions.csv`
* `hyperparameter_tuning_summary.json`

The exact files present depend on which scripts have been run successfully.

### Figures

The `outputs/figures/` directory contains nine visualizations:

1. `01_delivery_time_distribution.png`
2. `02_delivery_time_boxplot.png`
3. `03_delivery_time_by_vehicle.png`
4. `04_delivery_time_by_order_type.png`
5. `05_delivery_distance_distribution.png`
6. `06_distance_vs_delivery_time.png`
7. `07_mean_delivery_time_by_vehicle.png`
8. `08_mean_delivery_time_by_order_type.png`
9. `09_correlation_heatmap.png`

### Optimization Results

The `outputs/optimization/` directory contains scenario-analysis results, feature-sensitivity outputs, charts, logs, and written recommendations.

## Reproducibility

For consistent execution:

1. Keep the original dataset at the documented path.
2. Install dependencies from `requirements.txt`.
3. Run the scripts from the project root.
4. Execute the workflow in the documented order.
5. Preserve the generated evaluation results when comparing model runs.
6. Check the logs and summaries for errors or changes in output.

The reported metrics correspond to the completed runs described in this README. Re-running the workflow may produce different results if relevant settings or dependencies have changed.

## Limitations

The project has several important limitations:

* It depends on the quality and representativeness of the supplied dataset.
* Some observations contain missing values or questionable geographic coordinates.
* Straight-line distance does not represent real road distance.
* Available features do not capture every factor affecting delivery duration.
* Predictive associations do not establish causation.
* Hyperparameter-tuning results depend on the search space and evaluation design.
* Hypothetical scenario predictions are not proof of real-world operational improvements.
* The model has not been established as a production-ready real-time delivery service.

These limitations should be considered when interpreting the findings.

## Future Improvements

Potential extensions include:

* Testing additional regression algorithms and stronger validation strategies.
* Evaluating models on consistently defined training, validation, and test sets.
* Improving geographic data validation and distance calculations.
* Incorporating licensed road-network distance and travel-time data.
* Adding reliable traffic, weather, preparation-time, and dispatch-time information where available.
* Performing error analysis across different delivery types and distance ranges.
* Comparing predictions against new, independently collected observations.
* Developing a deployment interface only after further testing and validation.

## Conclusion

This project demonstrates a modular machine-learning workflow for food-delivery duration prediction. It combines data-quality assessment, exploratory analysis, geographic feature engineering, regression-model comparison, hyperparameter tuning, visualization, and hypothetical scenario analysis.

In the recorded runs, Random Forest outperformed the initial baseline and Linear Regression models according to MAE, RMSE, and R². The tuned model also produced promising evaluation results on its recorded test set. However, the findings remain dependent on the dataset and evaluation design, and the optimization scenarios should be treated as exploratory model-based estimates rather than guaranteed operational outcomes.

The project provides a foundation for further experimentation in predictive analytics and logistics decision support.

---

**Project:** Predictive Modeling and Optimization in Logistics Systems Using Python
**Program:** YuvaIntern — Week 4
**Language:** Python
**Problem Type:** Regression
**Target Variable:** `Time_taken(min)`