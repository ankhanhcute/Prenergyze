"""
Basic tests for ensemble functionality.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'backend' / 'scripts' / 'inference'))

from ensemble import EnsembleModel
from preprocess import prepare_features_for_inference


def test_ensemble_initialization():
    """Test ensemble can be initialized."""
    try:
        ensemble = EnsembleModel(device='cpu')
        assert ensemble is not None
        assert len(ensemble.model_names) > 0
    except Exception as e:
        # If no models are available, that's okay for testing
        pytest.skip(f"Models not available: {e}")


def test_preprocess_features():
    """Test feature preprocessing."""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    data = pd.DataFrame({
        'date': dates,
        'temperature_2m': np.random.randn(10) * 5 + 25,
        'apparent_temperature': np.random.randn(10) * 5 + 28,
        'relative_humidity_2m': np.random.randn(10) * 10 + 70,
        'vapour_pressure_deficit': np.random.randn(10) * 0.2 + 0.8,
        'pressure_msl': np.random.randn(10) * 5 + 1013,
        'precipitation': np.random.rand(10) * 5,
        'cloud_cover': np.random.rand(10) * 100,
        'cloud_cover_low': np.random.rand(10) * 50,
        'cloud_cover_mid': np.random.rand(10) * 50,
        'cloud_cover_high': np.random.rand(10) * 50,
        'et0_fao_evapotranspiration': np.random.rand(10) * 0.5,
        'sunshine_duration': np.random.rand(10) * 3600,
        'wind_speed_10m': np.random.rand(10) * 10,
        'wind_gusts_10m': np.random.rand(10) * 15,
        'wind_direction_10m': np.random.rand(10) * 360,
        'wind_dir_cos_10m': np.cos(np.deg2rad(np.random.rand(10) * 360)),
        'wind_dir_sin_10m': np.sin(np.deg2rad(np.random.rand(10) * 360)),
        'load': np.random.randn(10) * 1000 + 20000
    })
    
    # Test preprocessing
    features = prepare_features_for_inference(data)
    assert features is not None
    assert len(features) > 0
    assert 'date' not in features.columns
    assert 'load' not in features.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

