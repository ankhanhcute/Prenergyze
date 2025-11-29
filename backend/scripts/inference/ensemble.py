"""
Ensemble model for combining predictions from multiple models.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from load_models import load_all_available_models
from preprocess import align_features


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models."""
    
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        weighting_method: str = 'inverse_rmse',
        device: str = 'cpu'
    ):
        """
        Initialize ensemble model.
        
        Args:
            model_names: List of model names to include in ensemble. If None, uses all available.
            weights: Dictionary of model names to weights. If None, weights are calculated automatically.
            weighting_method: Method for calculating weights ('inverse_rmse', 'equal', 'custom')
            device: Device for LSTM model ('cpu' or 'cuda')
        """
        self.device = device
        self.models = load_all_available_models(device)
        
        # Filter models if specific names provided
        if model_names is not None:
            self.models = {name: self.models[name] for name in model_names if name in self.models}
        
        if not self.models:
            raise ValueError("No models available for ensemble")
        
        self.model_names = list(self.models.keys())
        self.weighting_method = weighting_method
        self.weights = weights or self._calculate_weights()
        
        print(f"Ensemble initialized with models: {self.model_names}")
        print(f"Weights: {self.weights}")
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate weights for ensemble based on validation performance."""
        if self.weighting_method == 'equal':
            # Equal weights
            weight = 1.0 / len(self.model_names)
            return {name: weight for name in self.model_names}
        
        elif self.weighting_method == 'inverse_rmse':
            # Inverse RMSE weighting (better models get higher weight)
            rmse_values = {}
            for name in self.model_names:
                metadata = self.models[name].get('metadata', {})
                cv_rmse = metadata.get('cv_rmse')
                if cv_rmse is not None and cv_rmse > 0:
                    rmse_values[name] = cv_rmse
                else:
                    # Fallback to test RMSE or equal weight
                    test_rmse = metadata.get('test_rmse')
                    if test_rmse is not None and test_rmse > 0:
                        rmse_values[name] = test_rmse
                    else:
                        # Use a large default RMSE for equal weighting
                        rmse_values[name] = 1000.0
            
            # Calculate inverse RMSE weights
            inverse_rmse = {name: 1.0 / rmse for name, rmse in rmse_values.items()}
            total_inverse = sum(inverse_rmse.values())
            weights = {name: inv_rmse / total_inverse for name, inv_rmse in inverse_rmse.items()}
            
            return weights
        
        elif self.weighting_method == 'custom':
            # Custom weights must be provided
            raise ValueError("Custom weighting requires weights parameter")
        
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
    
    def _predict_linear_regression(self, X: np.ndarray, model_info: Dict) -> np.ndarray:
        """Predict using Linear Regression model."""
        model = model_info['model']
        scaler = model_info['scaler']
        
        if scaler is None:
            raise ValueError("Linear Regression requires scaler")
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return predictions
    
    def _predict_tree_model(self, X: np.ndarray, model_info: Dict) -> np.ndarray:
        """Predict using tree-based models (RF, XGBoost, LightGBM)."""
        model = model_info['model']
        predictions = model.predict(X)
        return predictions
    
    def _predict_lstm(self, X: np.ndarray, model_info: Dict, sequence_length: int = 24) -> np.ndarray:
        """Predict using LSTM model."""
        model = model_info['model']
        scaler_X = model_info.get('scaler_X')
        scaler_y = model_info.get('scaler_y')
        metadata = model_info.get('metadata', {})
        
        seq_len = metadata.get('sequence_length', sequence_length)
        
        if scaler_X is None or scaler_y is None:
            raise ValueError("LSTM requires scalers")
        
        # Scale features
        X_scaled = scaler_X.transform(X)
        
        # Prepare sequences
        if len(X_scaled) < seq_len:
            # Pad with zeros if not enough data
            padding = np.zeros((seq_len - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        # Take last sequence_length samples
        X_seq = X_scaled[-seq_len:].reshape(1, seq_len, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy().flatten()
        
        # Inverse transform
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def _predict_sarimax(self, X: np.ndarray, model_info: Dict) -> np.ndarray:
        """Predict using SARIMAX model."""
        model = model_info['model']
        scaler = model_info['scaler']
        
        # SARIMAX expects scaled exogenous variables
        # We only need the last row (current prediction step) because
        # statsmodels forecast methods handle the time index relative to the end of training data.
        # However, since we are in a recursive loop predicting one step at a time,
        # and we don't want to re-fit the model or manipulate the internal state complexly,
        # we can use the fitted model's forecast with the new exog.
        
        # IMPORTANT: The loaded 'model' is a SARIMAXResultsWrapper from statsmodels.
        # Its .forecast() or .get_forecast() methods typically project from the end of the *training* data.
        # This is problematic for real-time inference on new data far in the future from training time.
        
        # Ideally, SARIMAX needs to be updated with recent observations (Kalman Filter update).
        # model.append(new_obs) or model.extend(new_obs) could work but might be slow.
        
        # For this implementation, we'll assume we are predicting for the NEXT step
        # relative to the model's internal state.
        # If the model is stale (trained long ago), this will be inaccurate (it will predict for trained_date + 1).
        # BUT, if we just retrained it (which we did), it's fine.
        
        # However, in a recursive loop (steps 1, 2, 3...), we are moving away from the training end.
        # We need to tell the model "predict 1 step ahead, given this exog".
        # Actually, simply calling .forecast(steps=1, exog=...) usually predicts
        # for the *next* step after the model's end index.
        
        # If we are in a recursive loop in forecast_service, we are calling this method multiple times.
        # Each time, we pass 1 row of X.
        # We can't easily update the model state inside this pure prediction method without side effects.
        
        # Hack/Workaround for MVP:
        # We use the model's predict function but we rely on the fact that we just want *a* prediction
        # given the exogenous variables. The AR/MA components will decay to 0 (or mean) quickly 
        # if we project far out, leaving mostly the exogenous influence (weather).
        # This is actually exactly what we want for long-term "wave pattern" from weather.
        
        # So we will use the .predict() method with exog.
        # Note: X shape is (1, n_features) usually in the loop.
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        # Predict 1 step. 
        # We use start=model.nobs, end=model.nobs + len(X) - 1
        # effectively predicting "next steps"
        
        # To avoid "value error" about index, we use integers.
        # Note: We are not updating the AR history here, so the AR component is static
        # (based on end of training). This is a limitation of static SARIMAX inference.
        # But the exogenous (weather) part will drive the wave pattern.
        
        start_idx = model.model.nobs
        end_idx = start_idx + len(X) - 1
        
        pred = model.predict(start=start_idx, end=end_idx, exog=X_scaled)
        
        return pred.values
    
    def predict(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make ensemble prediction.
        
        Args:
            X: Feature array (n_samples, n_features)
            feature_names: List of feature names (for alignment)
            
        Returns:
            Tuple of (ensemble_predictions, individual_predictions_dict)
        """
        individual_predictions = {}
        
        # Get feature names from first available model
        if feature_names is None:
            for model_info in self.models.values():
                metadata = model_info.get('metadata', {})
                if 'feature_names' in metadata:
                    feature_names = metadata['feature_names']
                    break
        
        # Make predictions from each model
        for name in self.model_names:
            try:
                model_info = self.models[name]
                
                # Align features if needed
                if feature_names is not None:
                    import pandas as pd
                    X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]] if X.shape[1] <= len(feature_names) else None)
                    X_aligned = align_features(X_df, feature_names).values
                else:
                    X_aligned = X
                
                # Predict based on model type
                if name == 'linear_regression':
                    pred = self._predict_linear_regression(X_aligned, model_info)
                elif name == 'lstm':
                    pred = self._predict_lstm(X_aligned, model_info)
                elif name == 'sarimax':
                    pred = self._predict_sarimax(X_aligned, model_info)
                else:
                    # Tree-based models
                    pred = self._predict_tree_model(X_aligned, model_info)
                
                individual_predictions[name] = pred
                
            except Exception as e:
                print(f"Warning: Failed to get prediction from {name}: {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(list(individual_predictions.values())[0]))
        total_weight = 0
        
        for name, pred in individual_predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred, individual_predictions
    
    def predict_single(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> float:
        """
        Make single prediction (for single sample).
        
        Args:
            X: Feature array (1, n_features) or (n_features,)
            feature_names: List of feature names
            
        Returns:
            Single prediction value
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        ensemble_pred, _ = self.predict(X, feature_names)
        return float(ensemble_pred[0])


def create_ensemble_from_comparison(
    comparison_path: Optional[str] = None,
    top_n: int = 3,
    max_inference_time_ms: float = 200.0,
    device: str = 'cpu'
) -> EnsembleModel:
    """
    Create ensemble from model comparison, selecting top N models.
    
    Args:
        comparison_path: Path to model_comparison.json. If None, uses default location.
        top_n: Number of top models to include
        max_inference_time_ms: Maximum inference time in milliseconds
        device: Device for LSTM model
        
    Returns:
        EnsembleModel instance
    """
    import json
    from pathlib import Path
    
    if comparison_path is None:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
        comparison_path = BASE_DIR / 'models' / 'model_comparison.json'
    
    if not Path(comparison_path).exists():
        print("Warning: Model comparison not found. Using all available models.")
        return EnsembleModel(device=device)
    
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)
    
    # Get top N models that meet inference time requirement
    ranking = comparison.get('ranking', [])
    selected_models = [
        item['model'] for item in ranking
        if item.get('inference_time_ms', float('inf')) <= max_inference_time_ms
    ][:top_n]
    
    if not selected_models:
        print("Warning: No models meet criteria. Using all available models.")
        return EnsembleModel(device=device)
    
    print(f"Selected models for ensemble: {selected_models}")
    return EnsembleModel(model_names=selected_models, device=device)

