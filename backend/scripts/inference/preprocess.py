"""
Inference preprocessing pipeline.
Uses shared feature engineering from data_prep.py for consistency.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

# Import shared feature engineering from training
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from data_prep import engineer_features


def prepare_features_for_inference(
    data: pd.DataFrame,
    historical_load: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Prepare features for inference, using shared feature engineering from training.
    
    Args:
        data: DataFrame with weather features and optional load column
        historical_load: Historical load values if available (for lag features)
        
    Returns:
        DataFrame with engineered features ready for model prediction
    """
    # Use shared feature engineering function
    df = engineer_features(data, historical_load=historical_load, keep_date=False)
    
    # Drop target column if present (for inference)
    if 'load' in df.columns:
        df = df.drop(['load'], axis=1)
    
    return df


def align_features(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Align DataFrame columns with expected feature list.
    Adds missing columns with NaN and removes extra columns.
    
    Args:
        df: DataFrame with features
        expected_features: List of expected feature names
        
    Returns:
        DataFrame with aligned columns
    """
    # Add missing columns with NaN
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = np.nan
    
    # Reorder and select only expected features
    df = df[expected_features]
    
    return df

