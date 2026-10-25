# Predictive Modeling and Optimization in Logistics Systems Using Python

A Python-based machine learning project for food-delivery time prediction, exploratory data analysis, feature engineering, hyperparameter tuning, and scenario-based logistics optimization.

## 1. Project Overview

Delivery time is an important performance indicator in logistics. Understanding the factors associated with delivery duration can help analysts investigate delivery patterns and evaluate potential operational improvements.

This project implements a modular Python workflow to analyze a food-delivery dataset and predict delivery duration in minutes. The workflow includes data preprocessing, exploratory data analysis (EDA), geographic feature engineering, regression modeling, hyperparameter tuning, model evaluation, visualization, and optimization scenario analysis.

Three regression approaches are compared: a mean-prediction baseline, Linear Regression, and Random Forest Regression. The Random Forest model is subsequently tuned using cross-validation to investigate whether its predictive performance can be improved.

The project also examines hypothetical delivery-distance reduction scenarios to understand how changes to an input feature affect model predictions. These scenarios are analytical experiments, not guarantees of real-world operational improvements.

## 2. Project Objectives

The main objectives are to:

* Prepare the dataset for analysis and machine learning.
* Identify missing values, unusual observations, and data-quality issues.
* Explore delivery-time distributions and relationships between available variables.
* Calculate approximate geographic distance from restaurant and delivery coordinates.
* Develop and compare regression models for delivery-time prediction.
* Tune Random Forest hyperparameters using cross-validation.
* Evaluate predictive performance using standard regression metrics.
* Investigate hypothetical delivery-distance reduction scenarios.
* Generate visualizations, evaluation tables, and structured reports.
* Organize the implementation into reusable Python modules.

## 3. Dataset

### 3.1 Dataset Source

The project uses a food-delivery dataset obtained from Kaggle.

**Raw dataset path:** `data/raw/deliverytime.csv`

### 3.2 Dataset Information

| Property                   | Description           |
| -------------------------- | --------------------- |
| Number of records          | 45,593                |
| Number of original columns | 11                    |
| Prediction target          | `Time_taken(min)`     |
| Machine learning task      | Supervised regression |
| Target unit                | Minutes               |

### 3.3 Original Dataset Columns

| Column                        | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `ID`                          | Order record identifier                            |
| `Delivery_person_ID`          | Delivery-person identifier                         |
| `Delivery_person_Age`         | Recorded age of the delivery person                |
| `Delivery_person_Ratings`     | Recorded delivery-person rating                    |
| `Restaurant_latitude`         | Restaurant latitude                                |
| `Restaurant_longitude`        | Restaurant longitude                               |
| `Delivery_location_latitude`  | Delivery destination latitude                      |
| `Delivery_location_longitude` | Delivery destination longitude                     |
| `Type_of_order`               | Order category                                     |
| `Type_of_vehicle`             | Vehicle category                                   |
| `Time_taken(min)`             | Delivery duration in minutes and prediction target |

The preprocessing and feature-engineering stages prepare the dataset and add derived information, including geographic-validity indicators and `delivery_distance_km`.

### 3.4 Geographic Distance Calculation

The project uses the Haversine formula to calculate the approximate straight-line distance between restaurant and delivery coordinates.

This measurement represents geographic distance rather than actual road distance. It does not account for road networks, traffic congestion, route restrictions, or navigation choices.

### 3.5 Dataset Limitations

The dataset does not directly provide dedicated measurements for every factor that may affect delivery duration. Examples of unavailable operational measurements include traffic conditions, weather conditions, kitchen preparation time, shipment weight, dispatch delay, and shipping cost.

Consequently, the project does not claim to measure the effects of these unavailable variables. Predictions and optimization scenarios are limited to the information represented by the dataset and its engineered features.

## 4. Technologies Used

| Technology     | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| Python         | Main programming language                                      |
| Pandas         | Data manipulation and analysis                                 |
| NumPy          | Numerical calculations                                         |
| Scikit-learn   | Regression models, model evaluation, and hyperparameter tuning |
| Matplotlib     | Data visualization                                             |
| Seaborn        | Statistical visualizations                                     |
| Joblib         | Saving and loading trained models                              |
| JSON and CSV   | Storing structured results and summaries                       |
| Git and GitHub | Version control and project sharing                            |

## 5. Project Structure

The project is organized into separate directories for datasets, trained models, generated outputs, and Python source code.

```text
Predictive_Logistics_Optimization/
|
|-- data/
|   |-- raw/
|   |   `-- deliverytime.csv
|   `-- processed/
|       |-- cleaned_delivery_data.csv
|       `-- feature_engineered_delivery_data.csv
|
|-- models/
|   |-- best_delivery_time_model.joblib
|   |-- dummy_regressor_mean_baseline_pipeline.joblib
|   |-- linear_regression_pipeline.joblib
|   |-- random_forest_regressor_pipeline.joblib
|   `-- tuned_random_forest_delivery_model.joblib
|
|-- outputs/
|   |-- evaluation/
|   |-- figures/
|   `-- optimization/
|
|-- src/
|   |-- data_preprocessing.py
|   |-- exploratory_analysis.py
|   |-- feature_engineering.py
|   |-- train_models.py
|   |-- hyperparameter_tuning.py
|   |-- optimization_analysis.py
|   |-- visualize_delivery_data.py
|   `-- __init__.py
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

### Directory Descriptions

* **`data/raw/`** — Contains the original dataset.
* **`data/processed/`** — Contains cleaned and feature-engineered datasets.
* **`models/`** — Contains saved machine-learning model artifacts.
* **`outputs/evaluation/`** — Contains data-quality summaries, EDA reports, model evaluations, predictions, and tuning results.
* **`outputs/figures/`** — Contains generated exploratory-analysis visualizations.
* **`outputs/optimization/`** — Contains optimization summaries, sensitivity results, recommendations, and related visualizations.
* **`src/`** — Contains the Python scripts that implement the project workflow.
* **`requirements.txt`** — Lists the Python package dependencies.
* **`README.md`** — Documents the project, setup, execution, results, and limitations.

## 6. Installation and Setup

### 6.1 Prerequisites

Before running the project, ensure that you have:

* Python installed on your computer.
* Visual Studio Code or another Python-compatible editor.
* Windows PowerShell or another terminal.
* The project dataset at `data/raw/deliverytime.csv`.

### 6.2 Open the Project Directory

Open PowerShell and navigate to the project root:

```powershell
cd C:\Predictive_Logistics_Optimization
```

Run the following command to verify the current directory:

```powershell
Get-Location
```

### 6.3 Install Dependencies

The project dependencies are listed in `requirements.txt`.

Install the required packages using:

```powershell
py -m pip install -r requirements.txt
```

The requirements include NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, and Python-Docx.

If your computer uses a different Python launcher, use the command configured for your installation.

## 7. Running the Project

Run the Python scripts from the project root directory in the following order. Allow each stage to finish successfully before starting the next one.

### Step 1: Data Preprocessing

```powershell
py src\data_preprocessing.py
```

This stage prepares the raw dataset for further analysis and saves the cleaned dataset under `data/processed/`.

### Step 2: Exploratory Data Analysis

```powershell
py src\exploratory_analysis.py
```

This stage generates descriptive statistics, data-quality summaries, correlation information, and exploratory-analysis reports under `outputs/evaluation/`.

### Step 3: Feature Engineering

```powershell
py src\feature_engineering.py
```

This stage creates engineered features, including approximate geographic delivery distance, and saves the resulting dataset under `data/processed/`.

### Step 4: Model Training and Comparison

```powershell
py src\train_models.py
```

This stage trains the baseline and regression models, evaluates their predictions, and saves model artifacts and evaluation results.

### Step 5: Hyperparameter Tuning

```powershell
py src\hyperparameter_tuning.py
```

This stage evaluates candidate Random Forest hyperparameter configurations using cross-validation and records the tuning results and selected parameters.

### Step 6: Optimization Analysis

```powershell
py src\optimization_analysis.py
```

This stage evaluates hypothetical feature-change scenarios and generates sensitivity summaries, scenario results, and optimization recommendations.

### Step 7: Visualization

```powershell
py src\visualize_delivery_data.py
```

This stage generates charts illustrating delivery-time distributions, delivery characteristics, geographic distance, and relationships among the available variables.

The resulting figures are saved under `outputs/figures/`.

**Execution note:** The stages are listed in the intended workflow order. Later stages may depend on datasets or models created by earlier stages. If a script reports an error, resolve it before continuing.

## 8. Machine Learning Methodology

### 8.1 Prediction Target

The target variable is `Time_taken(min)`, representing delivery duration in minutes.

The project treats delivery-time prediction as a supervised regression problem.

### 8.2 Models Evaluated

#### Mean-Prediction Baseline

A `DummyRegressor` provides a simple reference point against which the more sophisticated regression models can be compared.

#### Linear Regression

Linear Regression provides a relatively simple model for estimating relationships between the input features and delivery duration.

#### Random Forest Regression

Random Forest Regression combines predictions from multiple decision trees and can represent nonlinear relationships between input features and delivery duration.

#### Tuned Random Forest Regression

Hyperparameter tuning evaluates alternative Random Forest configurations through cross-validation. The selected configuration is then evaluated on the test data.

### 8.3 Evaluation Metrics

The project uses three standard regression metrics.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted delivery times. Lower values indicate smaller average absolute errors.

**Root Mean Squared Error (RMSE)**

RMSE measures prediction error while assigning greater weight to larger errors. Lower values indicate better performance under this metric.

**Coefficient of Determination (R²)**

R² compares model predictions with a mean-prediction reference and indicates how well the model accounts for variation in the target. Higher values generally indicate better performance.

MAE and RMSE are expressed in minutes when the target is measured in minutes.

## 9. Recorded Model Results

The following values are the results previously recorded during the project's model-training and hyperparameter-tuning workflow.

| Model                          | MAE (minutes) | RMSE (minutes) |                   R² |
| ------------------------------ | ------------: | -------------: | -------------------: |
| Mean baseline                  |        7.5790 |         9.3637 | Approximately 0.0000 |
| Linear Regression              |        6.3559 |         8.0513 |               0.2607 |
| Random Forest Regression       |        5.8585 |         7.4693 |               0.3637 |
| Tuned Random Forest Regression |        5.7337 |         7.2987 |               0.3924 |

### Results Interpretation

Among these four recorded results, the tuned Random Forest model achieved the lowest MAE and RMSE and the highest R².

Its recorded test MAE of approximately 5.73 minutes means that the average absolute prediction error on that evaluation set was about 5.73 minutes. This is an average, not a guarantee for any individual delivery.

The tuned model's R² of 0.3924 indicates that substantial variation in delivery duration remains unexplained. Its predictions should therefore be treated as estimates rather than exact delivery guarantees.

Model results can change if the dataset, preprocessing, features, train-test split, random seeds, or training configuration changes. The evaluation files under `outputs/evaluation/` provide supporting project records.

## 10. Hyperparameter Tuning

The recorded tuned Random Forest configuration is:

| Hyperparameter      | Selected value |
| ------------------- | -------------- |
| `n_estimators`      | 200            |
| `min_samples_split` | 6              |
| `min_samples_leaf`  | 1              |
| `max_features`      | `sqrt`         |
| `max_depth`         | 10             |

The recorded best cross-validation RMSE was approximately 7.3209 minutes.

The tuned model's recorded test RMSE was approximately 7.2987 minutes. Cross-validation and test-set results represent different evaluation procedures and should be interpreted separately.

These values represent the best configuration among the configurations evaluated by this project's tuning process. They are not necessarily globally optimal.

## 11. Optimization and Sensitivity Analysis

The optimization component investigates how hypothetical reductions in the engineered delivery-distance feature affect predicted delivery time.

The previously recorded scenario results were approximately:

| Hypothetical distance reduction | Change in predicted delivery time |
| ------------------------------- | --------------------------------: |
| 5%                              |               0.244 minutes lower |
| 10%                             |               0.452 minutes lower |
| 15%                             |               0.601 minutes lower |

For the 15% scenario, the predicted reduction is approximately 36 seconds.

These values describe model sensitivity under the scenarios evaluated. They are not measured operational savings and do not prove that reducing actual delivery routes by the same percentage will reduce delivery time by the predicted amount.

The calculated feature is straight-line geographic distance rather than actual road distance. Real-world outcomes may also depend on traffic, road networks, restaurant preparation, rider availability, weather, and other factors not directly represented in the dataset.

The optimization findings should therefore be treated as exploratory estimates that can inform further investigation, not as causal conclusions or guaranteed improvements.

## 12. Generated Outputs

The project generates evaluation files, visualizations, model artifacts, and optimization results.

### 12.1 Evaluation Outputs

The `outputs/evaluation/` directory contains dataset summaries, data-quality reports, exploratory-analysis results, model comparisons, prediction records, and hyperparameter-tuning results.

Important files include:

```text
outputs/evaluation/dataset_overview.csv
outputs/evaluation/data_quality_summary.json
outputs/evaluation/eda_report.txt
outputs/evaluation/model_comparison.csv
outputs/evaluation/model_predictions.csv
outputs/evaluation/best_hyperparameters.json
outputs/evaluation/hyperparameter_tuning_results.csv
outputs/evaluation/tuned_model_predictions.csv
```

Additional files provide missing-value summaries, numerical and categorical summaries, correlation information, outlier summaries, feature-engineering details, and execution logs.

### 12.2 Visualization Outputs

The `outputs/figures/` directory contains nine charts:

1. `01_delivery_time_distribution.png`
2. `02_delivery_time_boxplot.png`
3. `03_delivery_time_by_vehicle.png`
4. `04_delivery_time_by_order_type.png`
5. `05_delivery_distance_distribution.png`
6. `06_distance_vs_delivery_time.png`
7. `07_mean_delivery_time_by_vehicle.png`
8. `08_mean_delivery_time_by_order_type.png`
9. `09_correlation_heatmap.png`

These charts support the investigation of delivery-time distributions, vehicle and order categories, geographic distance, and relationships among numerical variables.

### 12.3 Optimization Outputs

The `outputs/optimization/` directory contains:

```text
feature_sensitivity_analysis.csv
feature_sensitivity_analysis.png
optimization_analysis.log
optimization_analysis_summary.json
optimization_recommendations.txt
optimization_scenario_summary.csv
predicted_delivery_time_scenarios.png
```

These files document the evaluated scenarios, sensitivity results, recommendations, and related visualizations.

## 13. Saved Models

The `models/` directory contains the saved machine-learning artifacts:

* `best_delivery_time_model.joblib`
* `dummy_regressor_mean_baseline_pipeline.joblib`
* `linear_regression_pipeline.joblib`
* `random_forest_regressor_pipeline.joblib`
* `tuned_random_forest_delivery_model.joblib`

These files store trained model objects or pipelines so they can be loaded without repeating the entire training process, provided that the loading code uses the correct artifact and compatible dependencies.

**Security note:** Only load serialized model files from trusted sources. Joblib files should not be treated as safe files when obtained from unknown origins.

## 14. Limitations and Responsible Interpretation

The project is an analytical and educational machine-learning implementation. Its results should be interpreted with the following limitations in mind:

1. **Dataset coverage:** The dataset does not contain every operational variable that can affect delivery duration.
2. **Geographic distance:** The Haversine distance is a straight-line estimate, not the distance travelled on roads.
3. **Prediction uncertainty:** A model prediction is an estimate and may differ from actual delivery time.
4. **Model performance:** The recorded R² indicates that substantial variation in delivery duration remains unexplained.
5. **Scenario assumptions:** The optimization analysis changes model inputs hypothetically and does not establish causality.
6. **Generalization:** Performance on other cities, delivery systems, time periods, or datasets may differ.
7. **Data quality:** Results depend on the quality and representativeness of the supplied data and the preprocessing workflow.

For operational use, the model would require further validation on representative, current delivery data and monitoring of prediction accuracy over time.

## 15. Future Improvements

Possible future extensions include:

* Integrating actual road-network distances and route information.
* Incorporating reliable traffic and weather measurements.
* Adding restaurant preparation and dispatch-time data where available.
* Testing additional regression algorithms and feature-selection methods.
* Evaluating performance across different locations and time periods.
* Improving model interpretability and uncertainty assessment.
* Validating proposed optimization scenarios against real operational outcomes.
* Developing a simple interface for entering delivery information and viewing predictions.

These are possible future enhancements, not features claimed to be implemented in the current project.

## 16. Conclusion

This project demonstrates a modular workflow for analyzing delivery data and building machine-learning models to estimate food-delivery duration.

It combines data preprocessing, exploratory analysis, geographic feature engineering, regression-model comparison, Random Forest hyperparameter tuning, visualization, and scenario-based sensitivity analysis.

Among the recorded experiments, the tuned Random Forest model achieved the best reported test metrics, with an MAE of approximately 5.73 minutes, an RMSE of approximately 7.30 minutes, and an R² of 0.3924.

The optimization component demonstrates how predictive models can be used to explore hypothetical changes to input features. However, these scenarios do not establish causal effects or guarantee real-world savings.

Overall, the project provides a practical foundation for further work in logistics analytics, delivery-time prediction, and data-driven operational investigation.