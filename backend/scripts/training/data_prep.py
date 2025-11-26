"""
Shared data preparation utilities for model training and inference.
Handles feature engineering and data splitting consistently across all models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional


def past_mean(series, window):
    """Calculate rolling mean with shift to avoid data leakage."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def past_std(series, window):
    """Calculate rolling std with shift to avoid data leakage."""
    return series.shift(1).rolling(window, min_periods=1).std()


def past_sum(series, window):
    """Calculate rolling sum with shift to avoid data leakage."""
    return series.shift(1).rolling(window, min_periods=1).sum()


def engineer_features(
    data: pd.DataFrame,
    historical_load: Optional[pd.Series] = None,
    keep_date: bool = False
) -> pd.DataFrame:
    """
    Apply feature engineering to data.
    This is the core feature engineering function used by both training and inference.
    
    Args:
        data: DataFrame with weather features and load column
        historical_load: Optional historical load values (for inference when current data doesn't have load)
        keep_date: Whether to keep the date column (default False, drops it)
        
    Returns:
        DataFrame with engineered features
    """
    df = data.copy()
    
    # Ensure date column exists and is datetime
    if 'date' not in df.columns:
        raise ValueError("Data must include 'date' column")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Use historical load if provided, otherwise use load from data
    if historical_load is not None:
        # Combine historical and current load
        load_series = pd.concat([historical_load, df['load'] if 'load' in df.columns else pd.Series()])
        load_series = load_series.reset_index(drop=True)
    elif 'load' in df.columns:
        load_series = df['load']
    else:
        # If no load data, we can't create load-based features
        load_series = None
    
    # Load-based features (if load data available)
    if load_series is not None:
        df['load_roll_std_3h'] = past_std(load_series, 3)
        df['load_roll_std_24h'] = past_std(load_series, 24)
        df['load_roll_std_168h'] = past_std(load_series, 168)
        df['load_roll_mean_24h'] = past_mean(load_series, 24)
        df['load_roll_mean_168h'] = past_mean(load_series, 168)
        
        # Load lag features
        df['load_lag_1h'] = load_series.shift(1)
        df['load_lag_2h'] = load_series.shift(2)
        df['load_lag_3h'] = load_series.shift(3)
        df['load_lag_24h'] = load_series.shift(24)
        df['load_lag_25h'] = load_series.shift(25)
        df['load_lag_168h'] = load_series.shift(168)
    
    # Wind direction features
    if 'wind_dir_cos_10m' in df.columns:
        df['wind_dir_cos_10m_roll_mean_3h'] = past_mean(df['wind_dir_cos_10m'], 3)
    if 'wind_dir_sin_10m' in df.columns:
        df['wind_dir_sin_10m_roll_mean_3h'] = past_mean(df['wind_dir_sin_10m'], 3)
    
    # Pressure & temperature features
    if 'pressure_msl' in df.columns:
        df['pressure_msl_roll_mean_3h'] = past_mean(df['pressure_msl'], 3)
        df['pressure_msl_roll_mean_24h'] = past_mean(df['pressure_msl'], 24)
        df['pressure_msl_lag_24h'] = df['pressure_msl'].shift(24)
    
    if 'temperature_2m' in df.columns:
        df['temperature_2m_roll_mean_3h'] = past_mean(df['temperature_2m'], 3)
        df['temperature_2m_roll_mean_24h'] = past_mean(df['temperature_2m'], 24)
        df['temperature_2m_lag_24h'] = df['temperature_2m'].shift(24)
    
    # Apparent temperature
    if 'apparent_temperature' in df.columns:
        df['apparent_temperature_roll_mean_3h'] = past_mean(df['apparent_temperature'], 3)
        df['apparent_temperature_lag_24h'] = df['apparent_temperature'].shift(24)
    
    # Humidity
    if 'relative_humidity_2m' in df.columns:
        df['relative_humidity_2m_roll_mean_3h'] = past_mean(df['relative_humidity_2m'], 3)
        df['relative_humidity_2m_roll_mean_24h'] = past_mean(df['relative_humidity_2m'], 24)
        df['relative_humidity_2m_lag_1h'] = df['relative_humidity_2m'].shift(1)
        df['relative_humidity_2m_lag_24h'] = df['relative_humidity_2m'].shift(24)
    
    # VPD
    if 'vapour_pressure_deficit' in df.columns:
        df['vapour_pressure_deficit_roll_mean_3h'] = past_mean(df['vapour_pressure_deficit'], 3)
        df['vapour_pressure_deficit_roll_mean_24h'] = past_mean(df['vapour_pressure_deficit'], 24)
        df['vapour_pressure_deficit_lag_1h'] = df['vapour_pressure_deficit'].shift(1)
        df['vapour_pressure_deficit_lag_24h'] = df['vapour_pressure_deficit'].shift(24)
    
    # Cloud cover features
    for cloud_col in ['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']:
        if cloud_col in df.columns:
            df[f'{cloud_col}_roll_mean_3h'] = past_mean(df[cloud_col], 3)
    
    # Wind features
    if 'wind_gusts_10m' in df.columns:
        df['wind_gusts_10m_roll_mean_3h'] = past_mean(df['wind_gusts_10m'], 3)
    if 'wind_speed_10m' in df.columns:
        df['wind_speed_10m_roll_mean_3h'] = past_mean(df['wind_speed_10m'], 3)
    
    # Sunshine and evapotranspiration
    if 'sunshine_duration' in df.columns:
        df['sunshine_duration_roll_mean_3h'] = past_mean(df['sunshine_duration'], 3)
    if 'et0_fao_evapotranspiration' in df.columns:
        df['et0_fao_evapotranspiration_roll_mean_3h'] = past_mean(df['et0_fao_evapotranspiration'], 3)
    
    # Precipitation
    if 'precipitation' in df.columns:
        df['precipitation_roll_sum_3h'] = past_sum(df['precipitation'], 3)
    
    # Drop date column if not keeping it
    if not keep_date and 'date' in df.columns:
        df = df.drop(['date'], axis=1)
    
    return df


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Load and prepare data with feature engineering.
    
    Args:
        data_path: Path to FEATURE_ENGINEERED_DATASET.csv
        
    Returns:
        DataFrame with features and target ready for training
    """
    data = pd.read_csv(data_path)
    
    # Parse date and create time features
    # Use shared feature engineering function
    data = engineer_features(data, keep_date=False)
    
    # Drop rows with NaN (from shifting)
    data.dropna(inplace=True)
    
    # Remove first 168 rows to ensure rolling windows are properly initialized
    data = data.iloc[168:].reset_index(drop=True)
    
    return data


def get_train_test_split(data: pd.DataFrame, test_size: int = 1800, gap: int = 168) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets using time series split.
    
    Args:
        data: Prepared DataFrame
        test_size: Number of hours for test set
        gap: Gap between train and test to prevent leakage
        
    Returns:
        Tuple of (train_data, test_data)
    """
    # Use last test_size hours for test, with gap
    cutoff_idx = len(data) - test_size - gap
    train_data = data.iloc[:cutoff_idx].copy()
    test_data = data.iloc[cutoff_idx + gap:].copy()
    
    return train_data, test_data


def get_cv_splits(data: pd.DataFrame, n_splits: int = 4, test_size: int = 1800, gap: int = 168):
    """
    Get TimeSeriesSplit cross-validation splits.
    
    Args:
        data: Prepared DataFrame
        n_splits: Number of CV folds
        test_size: Size of test set for each fold
        gap: Gap between train and test
        
    Returns:
        TimeSeriesSplit object
    """
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)


def prepare_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from prepared data.
    
    Args:
        data: Prepared DataFrame
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    X = data.drop(['load'], axis=1)
    y = data['load']
    return X, y

