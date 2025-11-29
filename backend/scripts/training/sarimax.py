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
    # Remove lag features of the target (load) because SARIMAX handles AR terms internally
    exog_features = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
        'precipitation', 'cloud_cover', 'wind_speed_10m',
        'hour', 'day_of_week', 'month'  # Time features are good exogenous variables
    ]
    
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for convenience (optional, but good for keeping track)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=exog_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=exog_features, index=X_test.index)
    
    print("Training SARIMAX model...")
    print("This may take a while...")
    
    start_time = time.time()
    
    # SARIMAX configuration
    # Order (p,d,q): (1, 0, 1) - simple ARMA model for the residuals
    # Seasonal Order (P,D,Q,s): (1, 1, 1, 24) - Daily seasonality (24 hours)
    # Note: Seasonal training can be very slow. We might simplify for this demo.
    # Simplified: (1, 0, 1) with no seasonality in the AR part, rely on exogenous 'hour' feature for seasonality
    # This is much faster and often sufficient when we have strong exogenous predictors like temperature and hour.
    
    # If we rely on 'hour' and 'day_of_week' as exogenous, we can often use a simpler ARIMA error structure.
    model = SARIMAX(
        endog=y_train,
        exog=X_train_scaled,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0), # Disable internal seasonality for speed, rely on exog features
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Fit model
    sarimax_model = model.fit(disp=False)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate
    print("Evaluating model...")
    start_inference = time.time()
    
    # Predict on test set
    # The gap logic in get_train_test_split leaves a gap between train and test
    # SARIMAX.predict(start, end) expects indices relative to the training data start
    # Or datetime values if index is datetime.
    
    # Statsmodels predict is tricky with gaps.
    # We will perform one-step ahead prediction for the test set duration,
    # but since we trained on data UP TO the test set (minus gap), 
    # we need to handle the gap or just predict for the test timestamps.
    
    # We will use the forecast method which is easier for out-of-sample
    # But first we need to make sure we are aligned.
    
    # Since we are using a simple split without retraining on the gap,
    # we should ideally re-fit or use 'append' but that's complex.
    # For this MVP, we will just predict for the test period using the provided exogenous vars.
    
    # Note: The error "Required (1968, 9), got (1800, 9)" suggests statsmodels expects
    # data for the gap period too if we predict from train_end to test_end.
    # 1968 = 1800 (test_size) + 168 (gap).
    
    # Solution: We must provide exog for the gap period as well to predict into the test set.
    # Or, we can just evaluate on the last N points if we only care about test performance.
    
    # Let's construct the full exog matrix from the end of training to the end of testing
    # This includes the gap.
    
    # Recover the gap data from the original dataset
    # train_data ends at index X
    # test_data starts at index X + gap
    train_end_idx = train_data.index[-1]
    test_start_idx = test_data.index[0]
    test_end_idx = test_data.index[-1]
    
    # Get exog for the gap + test period
    # We need to go back to the original 'data' dataframe
    full_exog_subset = data.loc[train_end_idx+1 : test_end_idx, exog_features]
    full_exog_scaled = scaler.transform(full_exog_subset)
    
    # Predict for the full range (gap + test)
    full_pred = sarimax_model.predict(
        start=len(train_data),
        end=len(train_data) + len(full_exog_subset) - 1,
        exog=full_exog_scaled
    )
    
    # Extract only the test part (the last 1800 points)
    y_pred = full_pred.iloc[-len(y_test):]
    
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
        
    # Save scaler
    scaler_path = MODELS_DIR / 'sarimax_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'model_name': 'SARIMAX',
        'feature_names': exog_features,
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'inference_time_ms': inference_time,
        'train_time_seconds': train_time,
        'order': (1, 0, 1),
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

