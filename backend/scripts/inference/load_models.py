"""
Utilities for loading trained models.
"""
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn

import sys
from pathlib import Path
training_dir = Path(__file__).resolve().parent.parent / 'training'
sys.path.insert(0, str(training_dir))

try:
    from lstm import LSTMModel
except ImportError:
    # Define LSTMModel here if import fails
    class LSTMModel(nn.Module):
        """LSTM model for time series forecasting."""
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            last_output = self.dropout(last_output)
            output = self.fc(last_output)
            return output


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
MODELS_DIR = BASE_DIR / 'models'


def load_linear_regression():
    """Load Linear Regression model, scaler, and metadata."""
    model_path = MODELS_DIR / 'linear_regression.pkl'
    scaler_path = MODELS_DIR / 'linear_regression_scaler.pkl'
    metadata_path = MODELS_DIR / 'linear_regression_metadata.pkl'
    
    if not model_path.exists():
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    return model, scaler, metadata


def load_random_forest():
    """Load Random Forest model and metadata."""
    model_path = MODELS_DIR / 'random_forest.pkl'
    metadata_path = MODELS_DIR / 'random_forest_metadata.pkl'
    
    if not model_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    return model, metadata


def load_xgboost():
    """Load XGBoost model and metadata."""
    model_path = MODELS_DIR / 'xgboost.pkl'
    metadata_path = MODELS_DIR / 'xgboost_metadata.pkl'
    
    if not model_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    return model, metadata


def load_lightgbm():
    """Load LightGBM model and metadata."""
    model_path = MODELS_DIR / 'lightgbm.pkl'
    metadata_path = MODELS_DIR / 'lightgbm_metadata.pkl'
    
    if not model_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    return model, metadata


def load_lstm(device: str = 'cpu'):
    """Load LSTM model, scalers, and metadata."""
    model_path = MODELS_DIR / 'lstm.pth'
    scaler_X_path = MODELS_DIR / 'lstm_scaler_X.pkl'
    scaler_y_path = MODELS_DIR / 'lstm_scaler_y.pkl'
    metadata_path = MODELS_DIR / 'lstm_metadata.pkl'
    
    if not model_path.exists():
        return None, None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    scaler_X = None
    if scaler_X_path.exists():
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
    
    scaler_y = None
    if scaler_y_path.exists():
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    return model, scaler_X, scaler_y, metadata


def load_sarimax():
    """Load SARIMAX model, scaler, and metadata."""
    model_path = MODELS_DIR / 'sarimax.pkl'
    scaler_path = MODELS_DIR / 'sarimax_scaler.pkl'
    metadata_path = MODELS_DIR / 'sarimax_metadata.pkl'
    
    if not model_path.exists():
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
    return model, scaler, metadata


def load_model_comparison() -> Optional[Dict]:
    """Load model comparison report."""
    comparison_path = MODELS_DIR / 'model_comparison.json'
    
    if not comparison_path.exists():
        return None
    
    with open(comparison_path, 'r') as f:
        return json.load(f)


def load_all_available_models(device: str = 'cpu') -> Dict[str, Any]:
    """
    Load all available models.
    
    Returns:
        Dictionary with model names as keys and (model, scaler, metadata) tuples as values
    """
    models = {}
    
    # Linear Regression
    lr_model, lr_scaler, lr_metadata = load_linear_regression()
    if lr_model is not None:
        models['linear_regression'] = {
            'model': lr_model,
            'scaler': lr_scaler,
            'metadata': lr_metadata
        }
    
    # Random Forest
    rf_model, rf_metadata = load_random_forest()
    if rf_model is not None:
        models['random_forest'] = {
            'model': rf_model,
            'scaler': None,
            'metadata': rf_metadata
        }
    
    # XGBoost
    xgb_model, xgb_metadata = load_xgboost()
    if xgb_model is not None:
        models['xgboost'] = {
            'model': xgb_model,
            'scaler': None,
            'metadata': xgb_metadata
        }
    
    # LightGBM
    lgb_model, lgb_metadata = load_lightgbm()
    if lgb_model is not None:
        models['lightgbm'] = {
            'model': lgb_model,
            'scaler': None,
            'metadata': lgb_metadata
        }
    
    # LSTM
    lstm_model, lstm_scaler_X, lstm_scaler_y, lstm_metadata = load_lstm(device)
    if lstm_model is not None:
        models['lstm'] = {
            'model': lstm_model,
            'scaler_X': lstm_scaler_X,
            'scaler_y': lstm_scaler_y,
            'metadata': lstm_metadata
        }

    # SARIMAX
    sarimax_model, sarimax_scaler, sarimax_metadata = load_sarimax()
    if sarimax_model is not None:
        models['sarimax'] = {
            'model': sarimax_model,
            'scaler': sarimax_scaler,
            'metadata': sarimax_metadata
        }
    
    return models
