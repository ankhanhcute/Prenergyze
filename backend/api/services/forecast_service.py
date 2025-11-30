"""
Forecast service for handling model predictions.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add scripts to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # This gives 'backend/'
sys.path.insert(0, str(BASE_DIR / 'backend' / 'scripts' / 'inference'))

from ensemble import EnsembleModel, create_ensemble_from_comparison
from preprocess import prepare_features_for_inference, align_features
from load_models import load_model_comparison


class ForecastService:
    """Service for making load forecasts."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize forecast service with models."""
        self.device = device
        self.ensemble = None
        self.individual_models = {}
        self.model_metadata = {}
        self._load_models()
    
    def _load_models(self):
        """Load ensemble and individual models."""
        try:
            # Try to create ensemble from comparison
            self.ensemble = create_ensemble_from_comparison(
                top_n=3,
                max_inference_time_ms=200.0,
                device=self.device
            )
            print("Ensemble model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load ensemble: {e}")
            # Fallback to loading all available models
            try:
                self.ensemble = EnsembleModel(device=self.device)
                print("Loaded all available models as ensemble")
            except Exception as e2:
                print(f"Error: Failed to load any models: {e2}")
                self.ensemble = None
        
        # Load model metadata
        comparison = load_model_comparison()
        if comparison:
            self.model_metadata = comparison.get('models', {})
    
    def is_ready(self) -> bool:
        """Check if service is ready to make predictions."""
        return self.ensemble is not None
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        if self.ensemble:
            return self.ensemble.model_names
        return []
    
    def forecast(
        self,
        weather_data: List[Dict],
        historical_load: Optional[List[float]] = None,
        use_ensemble: bool = True,
        selected_models: Optional[List[str]] = None
    ) -> Dict:
        """
        Make load forecast using recursive multi-step strategy.
        
        Args:
            weather_data: List of weather data dictionaries (future)
            historical_load: Optional historical load values
            use_ensemble: Whether to use ensemble (True) or best single model (False)
            selected_models: Optional list of model names to use. If None, uses default ensemble.
            
        Returns:
            Dictionary with forecast and metadata
        """
        if not self.is_ready():
            raise RuntimeError("Forecast service is not ready. Models not loaded.")
        
        # Convert weather data to DataFrame
        future_df = pd.DataFrame(weather_data)
        
        # Impute missing features in future weather
        future_df = self._impute_missing_features(future_df)
        
        # Load recent historical data (weather + load) for context
        # We need this to calculate lag features correctly
        hist_df = pd.DataFrame()
        try:
            data_path = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
            
            
            if data_path.exists():
                # Load enough history to cover largest lag (168h) + buffer
                # Loading last 336 hours (2 weeks)
                full_hist_df = pd.read_csv(data_path, low_memory=False)
                full_hist_df['date'] = pd.to_datetime(full_hist_df['date'])
                full_hist_df = full_hist_df.sort_values('date')
                hist_df = full_hist_df.tail(336).copy().reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Failed to load historical context: {e}")
        
        # If historical_load provided in request, update/append it to our context
        # This allows the client to provide the most up-to-date load reading
        if historical_load and len(historical_load) > 0:
            # Create a simple series for the provided load
            client_load = np.array(historical_load)
            
            if not hist_df.empty and 'load' in hist_df.columns:
                # If we have history, try to align or just replace the tail
                # Assuming client load is the MOST recent data ending at "now"
                # We replace the last N values of hist_df['load'] with client_load
                # taking care not to exceed bounds
                
                # Calculate reasonable minimum load for validation later
                min_load = max(1000.0, np.percentile(client_load[client_load > 0], 10)) if len(client_load[client_load > 0]) > 0 else 1000.0
                
                # If client load is longer than history, we might need to extend history
                # But we lack weather data for the extension.
                # Best effort: replace the tail of history with client load
                overlap = min(len(hist_df), len(client_load))
                hist_df.iloc[-overlap:, hist_df.columns.get_loc('load')] = client_load[-overlap:]
            else:
                # Fallback if no history file: create dummy history from client load
                # We won't have weather history, so weather lags will be NaN (imputed to mean)
                min_load = max(1000.0, np.percentile(client_load[client_load > 0], 10)) if len(client_load[client_load > 0]) > 0 else 1000.0
                
                dates = pd.date_range(end=pd.Timestamp.now(), periods=len(client_load), freq='H')
                hist_df = pd.DataFrame({
                    'date': dates,
                    'load': client_load
                })
                # Fill missing weather cols with defaults
                hist_df = self._impute_missing_features(hist_df)
        else:
            # Default min_load if no client history
            if not hist_df.empty and 'load' in hist_df.columns:
                loads = hist_df['load'].values
                min_load = max(1000.0, np.percentile(loads[loads > 0], 10))
            else:
                min_load = 1000.0

        # Prepare ensemble
        ensemble_to_use = self.ensemble
        if use_ensemble and selected_models:
            try:
                ensemble_to_use = EnsembleModel(
                    model_names=selected_models,
                    device=self.device
                )
            except Exception as e:
                print(f"Warning: Failed to create custom ensemble: {e}")

        # Get feature names
        feature_names = None
        for model_name in ensemble_to_use.model_names:
            model_info = ensemble_to_use.models[model_name]
            metadata = model_info.get('metadata', {})
            if 'feature_names' in metadata:
                feature_names = metadata['feature_names']
                break
        
        # RECURSIVE FORECASTING LOOP
        # We predict one step at a time, appending the prediction to history
        # so it can be used as a lag feature for subsequent steps.
        
        predictions = []
        individual_predictions_ts = {name: [] for name in ensemble_to_use.model_names}
        
        # Working dataframe starts with history
        current_df = hist_df.copy()
        
        # Ensure columns match
        common_cols = list(set(current_df.columns) & set(future_df.columns))
        
        for i in range(len(future_df)):
            # Get the weather for this step (future_df row i)
            next_step_weather = future_df.iloc[[i]].copy()
            
            # We don't know load yet
            if 'load' in next_step_weather.columns:
                next_step_weather = next_step_weather.drop('load', axis=1)
            
            # Append to current_df (temporarily with NaN load)
            # We need to align columns
            # Ensure next_step_weather has all columns from current_df (except load)
            for col in current_df.columns:
                if col not in next_step_weather.columns and col != 'load':
                    next_step_weather[col] = np.nan # Should be filled by imputation if needed, or irrelevant
            
            # Append row
            temp_df = pd.concat([current_df, next_step_weather], ignore_index=True)
            
            # Engineer features on the whole sequence
            # This ensures lags are calculated correctly using history + previous preds
            try:
                # Pass ONLY the load series we have so far as history for the helper
                # But engineer_features uses the 'load' column in the DF if present.
                # In temp_df, the last row has NaN load.
                # engineer_features will produce features for the last row.
                # Since load is NaN, load_lag_1 will be taken from previous row (which has load).
                
                # We don't pass historical_load arg because it's already in temp_df
                feat_df = prepare_features_for_inference(temp_df, historical_load=None)
                
                # Get the last row (the one we want to predict)
                X_row_df = feat_df.iloc[[-1]].copy()
                
                # Align features
                if feature_names:
                    X_row_df = align_features(X_row_df, feature_names)
                
                # Handle NaNs (impute)
                X_row_df = X_row_df.ffill().bfill().fillna(0.0)
                
                X = X_row_df.values
                
                # Predict
                if use_ensemble:
                    pred, indiv_preds = ensemble_to_use.predict(X, feature_names)
                    # scalar prediction
                    val = float(pred[0])
                    indivs = {k: float(v[0]) for k, v in indiv_preds.items()}
                else:
                    # Single model logic (simplified)
                    # Use the same predict method but with single model ensemble logic
                    # Or just reuse the code block above which handles weights
                    pred, indiv_preds = ensemble_to_use.predict(X, feature_names)
                    val = float(pred[0])
                    indivs = {k: float(v[0]) for k, v in indiv_preds.items()}
                
                # Enforce constraints
                val = max(val, min_load)
                val = min(val, 50000.0) # Cap at 50GW
                
                predictions.append(val)
                for k, v in indivs.items():
                    individual_predictions_ts[k].append(max(v, min_load))
                
                # UPDATE HISTORY with prediction
                # Add the predicted load to the weather row we just used
                next_step_weather['load'] = val
                
                # Align columns to match current_df before concat
                # Filter to common columns to avoid schema drift
                next_step_weather = next_step_weather.reindex(columns=current_df.columns)
                
                current_df = pd.concat([current_df, next_step_weather], ignore_index=True)
                
                # Optimization: Keep dataframe size manageable
                # We only need enough history for max lag (168h)
                if len(current_df) > 500:
                    current_df = current_df.iloc[-336:].reset_index(drop=True)
                    
            except Exception as e:
                print(f"Error in forecast step {i}: {e}")
                # Fallback: use previous prediction or min_load
                val = predictions[-1] if predictions else min_load
                predictions.append(val)
                # Add dummy row to keep loop going
                next_step_weather['load'] = val
                current_df = pd.concat([current_df, next_step_weather], ignore_index=True)
        
        # Format output
        return {
            'forecast': predictions,
            'individual_predictions': individual_predictions_ts,
            'model_weights': ensemble_to_use.weights,
            'models_used': ensemble_to_use.model_names
        }
    
    def _impute_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing weather features using reasonable defaults and forward fill.
        
        Args:
            df: DataFrame with weather data (may have missing values)
            
        Returns:
            DataFrame with imputed values
        """
        # Load recent historical data for imputation baselines
        try:
            data_path = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
            if data_path.exists():
                hist_df = pd.read_csv(data_path, low_memory=False, nrows=1000)  # Last 1000 rows for efficiency
                hist_df['date'] = pd.to_datetime(hist_df['date'], errors='coerce')
                hist_df = hist_df.dropna(subset=['date'])
                
                # Calculate means from recent historical data for imputation
                imputation_means = {
                    'temperature_2m': hist_df['temperature_2m'].mean() if 'temperature_2m' in hist_df.columns else 25.0,
                    'apparent_temperature': hist_df['apparent_temperature'].mean() if 'apparent_temperature' in hist_df.columns else 27.0,
                    'relative_humidity_2m': hist_df['relative_humidity_2m'].mean() if 'relative_humidity_2m' in hist_df.columns else 70.0,
                    'pressure_msl': hist_df['pressure_msl'].mean() if 'pressure_msl' in hist_df.columns else 1013.25,
                    'vapour_pressure_deficit': hist_df['vapour_pressure_deficit'].mean() if 'vapour_pressure_deficit' in hist_df.columns else 1.0,
                }
            else:
                # Fallback to reasonable defaults if no historical data
                imputation_means = {
                    'temperature_2m': 25.0,
                    'apparent_temperature': 27.0,
                    'relative_humidity_2m': 70.0,
                    'pressure_msl': 1013.25,
                    'vapour_pressure_deficit': 1.0,
                }
        except Exception:
            # Fallback defaults
            imputation_means = {
                'temperature_2m': 25.0,
                'apparent_temperature': 27.0,
                'relative_humidity_2m': 70.0,
                'pressure_msl': 1013.25,
                'vapour_pressure_deficit': 1.0,
            }
        
        # Impute critical features with means (if completely missing)
        for col, default_value in imputation_means.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_value)
        
        # Forward fill for time-series continuity (if some values exist)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                # Forward fill, then backward fill, then default
                df[col] = df[col].ffill().bfill()
                if df[col].isnull().any():
                    # If still missing, use column mean or default
                    default = imputation_means.get(col, 0.0)
                    df[col] = df[col].fillna(default)
        
        # Fill non-critical features with defaults
        df = df.fillna({
            'precipitation': 0.0,
            'cloud_cover': 0.0,
            'cloud_cover_low': 0.0,
            'cloud_cover_mid': 0.0,
            'cloud_cover_high': 0.0,
            'wind_speed_10m': 0.0,
            'wind_gusts_10m': 0.0,
            'sunshine_duration': 0.0,
            'et0_fao_evapotranspiration': 0.0,
            'wind_direction_10m': 180.0,  # Default wind direction
        })
        
        # Ensure wind direction sin/cos are calculated if missing
        if 'wind_direction_10m' in df.columns:
            if 'wind_dir_cos_10m' not in df.columns or df['wind_dir_cos_10m'].isnull().any():
                wind_dir_rad = np.radians(df['wind_direction_10m'])
                df['wind_dir_cos_10m'] = np.cos(wind_dir_rad)
            if 'wind_dir_sin_10m' not in df.columns or df['wind_dir_sin_10m'].isnull().any():
                wind_dir_rad = np.radians(df['wind_direction_10m'])
                df['wind_dir_sin_10m'] = np.sin(wind_dir_rad)
        
        return df
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        try:
            available_models = self.get_available_models()
            ensemble_models = self.ensemble.model_names if self.ensemble and hasattr(self.ensemble, 'model_names') else []
            return {
                'available_models': available_models,
                'ensemble_models': ensemble_models,
                'model_metadata': self.model_metadata
            }
        except Exception as e:
            # Return safe defaults if there's an error
            return {
                'available_models': [],
                'ensemble_models': [],
                'model_metadata': {}
            }
