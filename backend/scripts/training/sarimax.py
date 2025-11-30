"""
Train SARIMAX model for time series forecasting.
SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
is well-suited for load forecasting as it explicitly models seasonality and external factors (weather).
"""
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

# Add scripts to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from training.data_prep import load_and_prepare_data, get_train_test_split, prepare_features_target

# Define paths
MODELS_DIR = BASE_DIR / 'models'
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_sarimax():
    """Train SARIMAX model and save it."""
    print("Loading and preparing data...")
    
    # Load data
    if not DATA_PATH.exists():
        # Fallback path for running from root
        alt_path = Path('backend/data/processed/FEATURE_ENGINEERED_DATASET.csv')
        if alt_path.exists():
            data = load_and_prepare_data(str(alt_path))
        else:
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    else:
        data = load_and_prepare_data(str(DATA_PATH))
    
    # SARIMAX works best with raw time series and exogenous variables
    # We need to be careful about which features we use as exogenous variables
    # to avoid multicollinearity and overfitting
    
    # Select key exogenous features (weather)
    # We are adding explicit lag features as exogenous variables to help capture specific cyclical patterns.
    # Since we are using these strong lag predictors, we will simplify the internal ARIMA structure
    # to act as a Regression with ARIMA errors (ARIMAX).
    exog_features = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
        'precipitation', 'cloud_cover', 'wind_speed_10m',
        'hour', 'day_of_week', 'month',  # Time features
        'load_lag_24h', 'load_lag_48h', 'load_lag_72h', 'load_lag_168h', 'load_lag_336h'
    ]
    
    print(f"Using features: {exog_features}")
    
    # Filter data to keep only load and exogenous features
    # Also ensure we have a datetime index for statsmodels
    if 'date' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
         # data_prep.load_and_prepare_data drops the date column by default if we don't modify it
         # We might need to reload or just use the integer index
         pass

    # Split data
    train_data, test_data = get_train_test_split(data)
    
    # Prepare X (exogenous) and y (endogenous/target)
    X_train = train_data[exog_features]
    y_train = train_data['load']
    X_test = test_data[exog_features]
    y_test = test_data['load']
    
    # Scale exogenous features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target (endogenous) - Helpful for optimization convergence
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Convert back to DataFrame for convenience
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=exog_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=exog_features, index=X_test.index)
    
    print("Training SARIMAX model...")
    print("This may take a while...")
    
    start_time = time.time()
    
    # SARIMAX configuration
    # We use a simplified ARIMA structure (Regression with AR(1) errors) because the 
    # exogenous lag variables (24h, 168h, 336h) now handle the heavy lifting of seasonality.
    # This is more robust and aligns with how the other models (XGBoost, etc.) work.
    
    model = SARIMAX(
        endog=y_train_scaled,
        exog=X_train_scaled,
        order=(1, 0, 0),  # Simple AR(1) for residual correlation
        seasonal_order=(0, 0, 0, 0),  # Disable internal seasonality in favor of explicit lag features
        trend='c', # Explicitly include intercept
        enforce_stationarity=True, # Enforce stationarity for stability
        enforce_invertibility=True
    )
    
    # Fit model
    print("Fitting model...")
    sarimax_model = model.fit(disp=True)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate
    print("Evaluating model...")
    start_inference = time.time()
    
    # Predict on test set
    # ... logic for handling gap ...
    
    # Recover the gap data from the original dataset
    # train_data ends at index X
    # test_data starts at index X + gap
    train_end_idx = train_data.index[-1]
    test_start_idx = test_data.index[0]
    test_end_idx = test_data.index[-1]
    
    # Get exog for the gap + test period
    # We need to go back to the original 'data' dataframe
    full_exog_subset = data.loc[train_end_idx+1 : test_end_idx, exog_features]
    full_exog_scaled = scaler_X.transform(full_exog_subset)
    
    # Predict for the full range (gap + test)
    full_pred_scaled = sarimax_model.predict(
        start=len(train_data),
        end=len(train_data) + len(full_exog_subset) - 1,
        exog=full_exog_scaled
    )
    
    # Extract only the test part (the last 1800 points)
    y_pred_scaled = full_pred_scaled.iloc[-len(y_test):]
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.values.reshape(-1, 1)).flatten()
    
    inference_time = (time.time() - start_inference) * 1000 / len(y_test) # ms per sample
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save model and metadata
    print("Saving model...")
    
    # Save the model results wrapper (which contains the fitted parameters)
    model_path = MODELS_DIR / 'sarimax.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(sarimax_model, f)
        
    # Save scaler X
    scaler_path = MODELS_DIR / 'sarimax_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_X, f)

    # Save scaler y (NEW) - ForecastService needs to know about this!
    # Actually, ForecastService currently expects only 'sarimax_scaler.pkl' for X scaling?
    # We might need to update ForecastService if we introduce a Y scaler.
    # For now, let's overwrite 'sarimax_scaler.pkl' with scaler_X as before, 
    # and save scaler_y separately.
    scaler_y_path = MODELS_DIR / 'sarimax_scaler_y.pkl'
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Save metadata
    metadata = {
        'model_name': 'SARIMAX',
        'feature_names': exog_features,
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'inference_time_ms': inference_time,
        'train_time_seconds': train_time,
        'order': (1, 0, 0),
        'seasonal_order': (0, 0, 0, 0)
    }
    
    metadata_path = MODELS_DIR / 'sarimax_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Update comparison report
    update_model_comparison(metadata)
    
    print("SARIMAX training complete!")


def update_model_comparison(metadata):
    """Update the model comparison JSON file."""
    comparison_path = MODELS_DIR / 'model_comparison.json'
    
    if comparison_path.exists():
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
    else:
        comparison = {'models': {}, 'ranking': []}
    
    # Add/Update SARIMAX
    model_name = 'sarimax'
    comparison['models'][model_name] = {
        'test_rmse': metadata['test_rmse'],
        'test_mae': metadata['test_mae'],
        'test_r2': metadata['test_r2'],
        'inference_time_ms': metadata['inference_time_ms'],
        'status': 'success'
    }
    
    # Update ranking
    ranking = []
    for name, metrics in comparison['models'].items():
        if metrics.get('status') == 'success':
            ranking.append({
                'model': name,
                'test_rmse': metrics.get('test_rmse', float('inf')),
                'test_mae': metrics.get('test_mae', float('inf')),
                'test_r2': metrics.get('test_r2', -float('inf')),
                'inference_time_ms': metrics.get('inference_time_ms', 0)
            })
    
    # Sort by RMSE (lower is better)
    ranking.sort(key=lambda x: x['test_rmse'])
    
    # Add rank field
    for i, item in enumerate(ranking):
        item['rank'] = i + 1
    
    comparison['ranking'] = ranking
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    train_sarimax()

