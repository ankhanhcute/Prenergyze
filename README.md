# Prenergyze

**Prenergyze** is a machine learning-powered energy load forecasting system that predicts electricity grid demand based on weather data. The system uses an ensemble of multiple ML models to provide accurate load forecasts for energy grid operators.

## Features

- **Multi-Model Ensemble Forecasting**: Combines predictions from multiple machine learning models (Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, LSTM, SARIMAX) for improved accuracy
- **Weather-Based Predictions**: Uses comprehensive weather metrics including temperature, humidity, precipitation, cloud cover, and wind data
- **Cyclical Pattern Recognition**: Incorporates SARIMAX and time-series specific features to capture daily and weekly load cycles
- **Recursive Forecasting**: Supports multi-step forecasting (up to 7 days) using recursive prediction strategies
- **RESTful API**: FastAPI-based backend with automatic API documentation
- **Interactive Web Interface**: Modern React-based frontend with dynamic charts, weather-responsive backgrounds, and daily/hourly forecast views
- **Model Comparison**: Built-in model performance tracking and comparison with live metrics
- **Fast Inference**: Optimized ensemble selection based on accuracy and inference time

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Documentation](#api-documentation)
- [Development](#development)

## Architecture

The project consists of three main components:

1. **Backend API** (`backend/api/`): FastAPI application serving model predictions
2. **Frontend** (`frontend/`): HTML/JavaScript interface for interacting with the API
3. **ML Pipeline** (`backend/scripts/`): Data collection, preprocessing, model training, and inference scripts
4. **Reports** (`reports/`): Generated reports and training logs (e.g., CatBoost info)

### Data Flow

```
Weather Data (Open-Meteo) + Energy Data (EIA) 
    ↓
Data Preprocessing & Feature Engineering
    ↓
Model Training (Multiple ML Models)
    ↓
Ensemble Model Creation
    ↓
FastAPI Service
    ↓
Web Frontend
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) EIA API key for data collection

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Prenergyze
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv prenergyze_venv
   
   # On Windows
   prenergyze_venv\Scripts\activate
   
   # On macOS/Linux
   source prenergyze_venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (for data collection):
   Create a `.env` file in the project root:
   ```env
   EIA_API_KEY=your_eia_api_key_here
   ```

5. **Verify model files exist**:
   Ensure trained model files are present in `backend/models/`:
   - `linear_regression.pkl`
   - `random_forest.pkl`
   - `xgboost.pkl`
   - `lightgbm.pkl`
   - `lstm.pth`
   - `model_comparison.json`

## 🎯 Usage

### Starting the Backend API

From the project root:

```bash
# Option 1: Using Python directly
cd backend/api
python app.py

# Option 2: Using uvicorn
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

### Using the Frontend

1. **Start the backend API** (see above)

2. **Open the frontend**:
   - Simply open `frontend/index.html` in a web browser, or
   - Serve it with a simple HTTP server:
     ```bash
     # Python 3
     python -m http.server 8080
     # Then navigate to http://localhost:8080
     ```

3. **Configure and test**:
   - Set the API base URL (default: `http://localhost:8000`)
   - Click "Refresh Status" to check API health
   - Click "Load Sample Data" to fill in sample weather values
   - Click "Get Forecast" to generate a load prediction

### Making API Requests

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Model Information
```bash
curl http://localhost:8000/models
```

#### Make a Forecast
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "weather_data": [{
      "date": "2024-01-01T00:00:00",
      "temperature_2m": 25.5,
      "apparent_temperature": 26.0,
      "relative_humidity_2m": 65.0,
      "vapour_pressure_deficit": 1.2,
      "pressure_msl": 1013.25,
      "precipitation": 0.0,
      "cloud_cover": 30.0,
      "cloud_cover_low": 10.0,
      "cloud_cover_mid": 15.0,
      "cloud_cover_high": 5.0,
      "et0_fao_evapotranspiration": 3.5,
      "sunshine_duration": 8.0,
      "wind_speed_10m": 5.0,
      "wind_gusts_10m": 7.0
    }],
    "use_ensemble": true
  }'
```

## 📁 Project Structure

```
Prenergyze/
├── backend/
│   ├── api/                    # FastAPI application
│   │   ├── app.py             # Main API application
│   │   ├── schemas/          # Pydantic schemas
│   │   └── services/         # Business logic services
│   ├── scripts/
│   │   ├── data_collection/  # Data fetching scripts
│   │   ├── static_data_preprocessing/  # Data cleaning & feature engineering
│   │   ├── training/         # Model training scripts
│   │   └── inference/        # Model loading & prediction
│   ├── models/               # Trained model files
│   ├── data/
│   │   ├── raw/              # Raw data files
│   │   └── processed/        # Processed datasets
│   ├── notebooks/            # Jupyter notebooks for analysis
│   └── plots/                # Visualization outputs
├── frontend/
│   └── index.html            # Web interface
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🤖 Models

The system uses an ensemble of machine learning models:

| Model | Description |
|-------|-------------|
| **CatBoost** | Top-performing gradient boosting model that handles categorical features automatically. |
| **Random Forest** | Ensemble learning method using multiple decision trees. Robust to outliers. |
| **XGBoost** | Highly efficient gradient boosting algorithm known for high performance. |
| **LightGBM** | Fast gradient boosting framework capable of handling large datasets efficiently. |
| **LSTM** | Deep learning model (Long Short-Term Memory) for capturing long-term temporal dependencies. |
| **SARIMAX** | Statistical model explicitly designed to capture seasonality and external weather factors. |
| **Linear Regression** | Simple baseline model assuming linear relationship between features and load. |

### Ensemble Strategy

The system automatically selects the top-performing models based on:
- Cross-validation RMSE
- Inference time constraints
- Model availability
- **Force-inclusion logic**: SARIMAX is included to ensure cyclical patterns are captured even if its raw RMSE is higher.

By default, the ensemble uses the top 3 models plus SARIMAX, using inverse RMSE weighting.

## 📚 API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint. Returns:
- `status`: API health status
- `models_loaded`: List of loaded model names
- `ensemble_ready`: Whether ensemble is ready for predictions

#### `GET /models`
Get information about available models. Returns:
- `available_models`: List of all available models
- `ensemble_models`: Models used in the ensemble
- `model_metadata`: Performance metrics for each model

#### `POST /forecast`
Make a load forecast prediction.

**Request Body:**
```json
{
  "weather_data": [
    {
      "date": "2024-01-01T00:00:00",
      "temperature_2m": 25.5,
      "apparent_temperature": 26.0,
      "relative_humidity_2m": 65.0,
      "vapour_pressure_deficit": 1.2,
      "pressure_msl": 1013.25,
      "precipitation": 0.0,
      "cloud_cover": 30.0,
      "cloud_cover_low": 10.0,
      "cloud_cover_mid": 15.0,
      "cloud_cover_high": 5.0,
      "et0_fao_evapotranspiration": 3.5,
      "sunshine_duration": 8.0,
      "wind_speed_10m": 5.0,
      "wind_gusts_10m": 7.0,
      "wind_direction_10m": 180.0,
      "wind_dir_cos_10m": -1.0,
      "wind_dir_sin_10m": 0.0
    }
  ],
  "historical_load": [15000.0, 15200.0],  // Optional
  "use_ensemble": true,                     // Optional, default: true
  "selected_models": ["linear_regression", "xgboost"]  // Optional
}
```

**Response:**
```json
{
  "forecast": [15432.5],
  "individual_predictions": {
    "linear_regression": [15400.0],
    "xgboost": [15465.0]
  },
  "model_weights": {
    "linear_regression": 0.6,
    "xgboost": 0.4
  },
  "models_used": ["linear_regression", "xgboost"]
}
```

Interactive API documentation is available at `/docs` when the server is running.

## 🔧 Development

### Data Collection

Collect energy demand data from EIA:
```bash
python backend/scripts/data_collection/eia_fetcher.py
```

Collect weather data from Open-Meteo:
```bash
python backend/scripts/data_collection/meteo_fetch.py
```

### Data Preprocessing

1. **Clean and merge datasets**:
   ```bash
   python backend/scripts/static_data_preprocessing/clean_merge.py
   ```

2. **Feature engineering**:
   ```bash
   python backend/scripts/static_data_preprocessing/feature_engineering.py
   ```

### Model Training

Train all models:
```bash
python backend/scripts/training/train_all_models.py
```

Or train individual models:
```bash
python backend/scripts/training/linear_regression.py
python backend/scripts/training/random_forest.py
python backend/scripts/training/train_xgboost.py
python backend/scripts/training/train_lightgbm.py
python backend/scripts/training/lstm.py
```

### Testing

Run tests:
```bash
pytest backend/tests/
```

### Notebooks

Jupyter notebooks for exploratory data analysis and model development are available in `backend/notebooks/`.

## 📊 Data Sources

- **Energy Data**: U.S. Energy Information Administration (EIA) API
- **Weather Data**: Open-Meteo Historical Weather API

## 🛠️ Technologies

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Machine Learning**: 
  - scikit-learn (Linear Regression, Random Forest)
  - XGBoost
  - LightGBM
  - CatBoost
  - PyTorch (LSTM)
  - Statsmodels (SARIMAX)
- **Data Processing**: pandas, numpy
- **Visualization**: Recharts (Frontend), matplotlib/seaborn (Notebooks)

## 👥 Contributors

- Adrian Morton (Lead)
- Leonardo Herrera (Co-lead)
- Khanh Truong
- Khang Ho
- Rhode Sanchez
- Annette Garcia
- Julian Novak
- Gabriela Hernandez

## 🙏 Acknowledgments

- U.S. Energy Information Administration for energy demand data
- Open-Meteo for weather data
- All open-source libraries and frameworks used in this project
