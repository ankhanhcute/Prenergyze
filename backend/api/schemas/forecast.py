"""
Pydantic schemas for forecast API requests and responses.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class WeatherDataPoint(BaseModel):
    """Single weather data point."""
    date: datetime
    temperature_2m: float
    apparent_temperature: float
    relative_humidity_2m: float
    vapour_pressure_deficit: float
    pressure_msl: float
    precipitation: float
    cloud_cover: float
    cloud_cover_low: float
    cloud_cover_mid: float
    cloud_cover_high: float
    et0_fao_evapotranspiration: float
    sunshine_duration: float
    wind_speed_10m: float
    wind_gusts_10m: float
    wind_direction_10m: Optional[float] = None
    wind_dir_cos_10m: Optional[float] = None
    wind_dir_sin_10m: Optional[float] = None


class ForecastRequest(BaseModel):
    """Request schema for forecast endpoint."""
    weather_data: List[WeatherDataPoint] = Field(..., description="Weather data for forecasting")
    historical_load: Optional[List[float]] = Field(None, description="Historical load values (optional)")
    use_ensemble: bool = Field(True, description="Whether to use ensemble or single best model")
    selected_models: Optional[List[str]] = Field(None, description="List of model names to use in ensemble. If None, uses optimal default ensemble.")


class ForecastResponse(BaseModel):
    """Response schema for forecast endpoint."""
    forecast: List[float] = Field(..., description="Forecasted load values")
    individual_predictions: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Individual model predictions (if ensemble used)"
    )
    model_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Weights used in ensemble (if ensemble used)"
    )
    models_used: List[str] = Field(..., description="List of models used for prediction")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    ensemble_ready: bool


class ModelInfoResponse(BaseModel):
    """Model information response."""
    available_models: List[str]
    ensemble_models: List[str]
    model_metadata: Dict[str, Dict]

