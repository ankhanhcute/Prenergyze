"""
Forecast service for handling model predictions.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add scripts to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'inference'))

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
        Make load forecast.
        
        Args:
            weather_data: List of weather data dictionaries
            historical_load: Optional historical load values
            use_ensemble: Whether to use ensemble (True) or best single model (False)
            selected_models: Optional list of model names to use. If None, uses default ensemble.
            
        Returns:
            Dictionary with forecast and metadata
        """
        if not self.is_ready():
            raise RuntimeError("Forecast service is not ready. Models not loaded.")
        
        # Convert weather data to DataFrame
        df = pd.DataFrame(weather_data)
        
        # Check for missing critical values
        critical_cols = ['temperature_2m', 'relative_humidity_2m', 'pressure_msl']
        missing_critical = df[critical_cols].isnull().any(axis=1)
        if missing_critical.any():
            num_missing = missing_critical.sum()
            raise ValueError(
                f"Found {num_missing} weather data points with missing critical values. "
                f"Please ensure all weather data has temperature, humidity, and pressure values."
            )
        
        # Fill other missing values with reasonable defaults
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
        })
        
        # Prepare features
        historical_load_series = None
        if historical_load:
            historical_load_series = pd.Series(historical_load)
        
        try:
            features_df = prepare_features_for_inference(df, historical_load_series)
        except Exception as e:
            raise ValueError(f"Failed to prepare features: {e}")
        
        # Determine which ensemble to use
        ensemble_to_use = self.ensemble
        if use_ensemble and selected_models:
            # Create a custom ensemble with selected models
            try:
                ensemble_to_use = EnsembleModel(
                    model_names=selected_models,
                    device=self.device
                )
            except Exception as e:
                raise ValueError(f"Failed to create ensemble with selected models: {e}. Available models: {self.get_available_models()}")
        
        # Get feature names from ensemble
        feature_names = None
        for model_name in ensemble_to_use.model_names:
            model_info = ensemble_to_use.models[model_name]
            metadata = model_info.get('metadata', {})
            if 'feature_names' in metadata:
                feature_names = metadata['feature_names']
                break
        
        # Align features
        if feature_names:
            features_df = align_features(features_df, feature_names)
        
        # Convert to numpy array
        X = features_df.values
        
        # Handle NaN values (fill with 0 for now)
        X = np.nan_to_num(X, nan=0.0)
        
        # Make prediction
        if use_ensemble:
            ensemble_pred, individual_preds = ensemble_to_use.predict(X, feature_names)
            
            # Validate predictions - ensure non-negative (energy load can't be negative)
            ensemble_pred = np.maximum(ensemble_pred, 0.0)
            individual_preds = {
                name: np.maximum(pred, 0.0) for name, pred in individual_preds.items()
            }
            
            return {
                'forecast': ensemble_pred.tolist(),
                'individual_predictions': {
                    name: pred.tolist() for name, pred in individual_preds.items()
                },
                'model_weights': ensemble_to_use.weights,
                'models_used': ensemble_to_use.model_names
            }
        else:
            # Use best single model (first in ensemble)
            best_model_name = ensemble_to_use.model_names[0]
            model_info = ensemble_to_use.models[best_model_name]
            
            # Predict using best model
            if best_model_name == 'linear_regression':
                from load_models import load_linear_regression
                model, scaler, _ = load_linear_regression()
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
            elif best_model_name == 'lstm':
                pred, _ = self.ensemble._predict_lstm(X, model_info)
            else:
                pred = self.ensemble._predict_tree_model(X, model_info)
            
            return {
                'forecast': pred.tolist() if isinstance(pred, np.ndarray) else [float(pred)],
                'individual_predictions': {best_model_name: pred.tolist() if isinstance(pred, np.ndarray) else [float(pred)]},
                'model_weights': {best_model_name: 1.0},
                'models_used': [best_model_name]
            }
    
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

