# Energy Demand Prediction Using Machine Learning

A Django-based web application for electricity demand forecasting using historical consumption, weather-related variables, and time-based features. The project compares machine learning regression models and provides an interactive interface for generating demand predictions and viewing evaluation visualizations.

## Project Overview

Accurate electricity demand forecasting supports generation planning, load management, grid reliability, and operational cost reduction. Demand varies with historical usage, weather conditions, time of day, and recurring weekly or seasonal patterns. This project explores machine learning models that learn relationships between these factors and electricity consumption.

The application combines a machine learning workflow with a Django web interface. Users can enter a timestamp and selected weather and consumption values to obtain a demand estimate and a low, medium, or high demand category.

## Objectives

- Forecast electricity demand from historical consumption and contextual features.
- Engineer time-based and historical consumption features.
- Train and compare regression models.
- Evaluate predictions using MAE, RMSE, and R².
- Provide an interactive Django interface for demand prediction.
- Visualize prediction behaviour, residuals, error distributions, and feature importance.
- Present demand-level recommendations as decision-support information.

## Models

The project report describes comparison of four primary models:

| Model | Role |
|---|---|
| Random Forest Regressor | Tree-ensemble regression |
| Gradient Boosting Regressor | Boosting-based regression |
| Support Vector Regression (SVR) | Kernel-based regression |
| Artificial Neural Network (ANN) | Neural-network regression |

The repository also contains an XGBoost model artifact (`model.pkl`). Its role in the current Django application is separate from the four-model comparison unless corresponding training and inference code is verified.

**Interactive prediction:** The inspected Django prediction view loads `gb.pkl` (Gradient Boosting). The presence of other saved model files does not mean the prediction page automatically uses them.

## Features and Methodology

### Input and engineered features

The inspected training workflow uses 12 model features:

- `Temperature`
- `Humidity`
- `Wind_Speed`
- `Avg_Past_Consumption`
- `Lag_1`
- `Lag_24`
- `Rolling_Mean_3`
- `Rolling_Mean_24`
- `Hour`
- `DayOfWeek`
- `Month`
- `Weekend`

The timestamp is converted into calendar features. Historical lag and rolling-average features represent recent and longer-term consumption patterns.

### Training workflow

1. Load the energy-demand CSV dataset.
2. Convert the timestamp field to datetime.
3. Derive hour, day-of-week, month, and weekend features.
4. Select input features and the electricity-consumption target.
5. Split observations into training and test subsets.
6. Apply feature scaling for models that require it, such as SVR and ANN.
7. Train the regression models.
8. Generate predictions and calculate evaluation metrics.
9. Save trained model artifacts and the scaler.
10. Generate diagnostic plots for model analysis.

The inspected Django training view uses a sequential 80/20 split. The academic report also discusses cross-validation; readers should distinguish that report-level description from the specific split implemented in the current Django view.

### Evaluation metrics

- **MAE (Mean Absolute Error):** average absolute prediction error.
- **RMSE (Root Mean Squared Error):** gives greater weight to larger errors.
- **R² (Coefficient of Determination):** measures the proportion of target variation explained by the model.

R² is not classification accuracy. A percentage display derived from `R² × 100` should be interpreted as a transformed R² value, not conventional model accuracy.

## Web Application

The Django application includes these main pages:

- **Home:** project introduction and navigation.
- **Train:** starts model training and displays model metrics and generated plots.
- **Predict:** accepts a timestamp, temperature, humidity, wind speed, and past-consumption value. The inspected prediction view uses the saved Gradient Boosting model.
- **Overview:** describes the project, features, workflow, and applications.

The prediction interface assigns a low, medium, or high demand category and displays a recommendation based on that category. These categories are application-level thresholds, not a separately trained classification model.

## Repository Structure

```text
Energy-Demand-Forecasting-ML/
├── EnergyDemand_Forecasting_Using_MachineLearning/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── users/
│   └── ... Django application code ...
├── templates/
│   ├── index.html
│   ├── overview.html
│   ├── predict.html
│   ├── result.html
│   └── train.html
├── media/
│   ├── ... datasets ...
│   ├── graphs/
│   └── models/
├── .gitignore
├── manage.py
├── requirements.txt
└── README.md
```

The exact contents of `media/` depend on which datasets, generated plots, and model artifacts are included. The large Random Forest artifact `rf.pkl`, local environment file, database file, and project documents are intentionally excluded from Git tracking.

## Datasets

The project folder contains several CSV files:

| File | Observed contents | Notes |
|---|---|---|
| `energy_demand_dataset.csv` | 29,952 rows, 10 columns | Main forecasting dataset used by the inspected training view |
| `energy_demand_dataset_30000_rows.csv` | 29,952 rows, 10 columns | Inspected as an exact duplicate of the main forecasting CSV |
| `smart_meter_data.csv` | 5,000 rows, 7 columns | Smart-meter data with an anomaly label |
| `smart_meter_data_enhanced.csv` | 4,976 rows, 11 columns | Includes lag and rolling features and an anomaly label |

The inspected Django training function reads `media/energy_demand_dataset.csv`. The two smart-meter datasets are additional project assets; their presence does not establish that they are used by the current forecasting training or prediction path.

The inspected CSV values were normalized. Consult the accompanying project documentation for dataset provenance and licensing or usage conditions before redistributing source data.

## Visualizations

The project generates or includes diagnostic plots such as:

- Actual vs. predicted demand
- Residual errors
- Error distribution
- Feature importance

These plots help inspect prediction trends and error behaviour. Visualizations are qualitative diagnostics and should be considered alongside numerical test-set metrics.

## Technology Stack

- **Language:** Python
- **Web framework:** Django
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn, TensorFlow/Keras
- **Additional model artifact:** XGBoost
- **Visualization:** Matplotlib
- **Model persistence:** Joblib and Keras HDF5

See `requirements.txt` for the project's listed Python dependencies.

## Setup and Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/tirunagirilokesh/Energy-Demand-Forecasting-ML.git
cd Energy-Demand-Forecasting-ML
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root. Do not commit this file.

```env
DJANGO_SECRET_KEY=replace-with-a-new-secret-key
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
```

Generate your own secret key and keep credentials out of source control. For deployment, set `DJANGO_DEBUG=False` and configure allowed hosts and other production settings appropriately.

### 5. Start the Django development server

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

The training page is available at `/train/`, and the prediction page is available at `/predict/`.

**Note:** In the inspected implementation, visiting the training route can initiate model training. Training may take time and requires the relevant dataset and dependencies.

## Model and Data Availability

Some model artifacts and datasets may not be included in the GitHub repository. In particular, the approximately 208 MB Random Forest file `media/models/rf.pkl` is excluded because it exceeds GitHub's standard 100 MB per-file limit. The `.env` file, local SQLite database, and project documentation folder are also excluded.

If a required model or dataset is absent after cloning, the corresponding feature may not run until the file is supplied locally or the model is regenerated using the training workflow. Do not commit private credentials or large model files directly to Git.

## Current Limitations and Reproducibility Notes

The following points were identified during a read-only review of the project code and artifacts:

- The interactive prediction view loads the Gradient Boosting model (`gb.pkl`); other saved models are not automatically selected by that view.
- The inspected training and prediction paths do not apply feature transformations consistently. In particular, the Gradient Boosting model is trained on raw features, while the prediction view constructs scaled values. This can affect prediction reliability.
- The prediction view reuses one past-consumption input for multiple lag and rolling features. These represent different historical calculations and should not be assumed to be interchangeable.
- The target is trained as `Electricity_Consumed`, while the prediction view applies an additional multiplication by a fixed past-consumption maximum. The validity of that conversion depends on the target's actual training scale.
- The application displays an `accuracy` value derived from R² multiplied by 100 in the training page. This is not classification accuracy.
- The project includes multiple CSV datasets, but the inspected training function reads the main energy-demand dataset.
- This is an academic demonstration and has not been established as a production-ready grid-control or real-time utility system.

These notes make the repository more transparent and reproducible. They do not alter the existing application.

## Future Improvements

- Align training and inference preprocessing through one shared, persisted preprocessing pipeline.
- Calculate lag and rolling features from the correct historical observations.
- Validate target scaling and output units end to end.
- Add time-series-aware validation and report test-set metrics for each model.
- Compare against statistical forecasting baselines.
- Add automated tests for input validation, feature construction, and prediction.
- Improve production deployment settings, static-file handling, and model/data management.
- Consider model versioning and a documented process for obtaining large artifacts.

## Academic Project

**Project title:** Energy Demand Prediction Using Advanced Machine Learning Techniques and Real-Time Weather Features

Developed as an undergraduate project in the Department of Computer Science and Engineering at TKR College of Engineering and Technology, Hyderabad, India.

The accompanying academic report contains the extended introduction, literature review, methodology, results, conclusion, and references. The report itself is not included in this repository.

---

If you use or build upon this project, describe the implementation and its limitations accurately and cite any external datasets or research sources that you reuse.
