"""
FastAPI application for energy load forecasting.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts to path
# __file__ is backend/api/app.py
# parent.parent gives backend/
BASE_DIR = Path(__file__).resolve().parent.parent  # This gives 'backend/'
sys.path.insert(0, str(BASE_DIR / 'api'))

from schemas.forecast import ForecastRequest, ForecastResponse, HealthResponse, ModelInfoResponse
from services.forecast_service import ForecastService

app = FastAPI(
    title="Prenergyze Energy Load Forecasting API",
    description="API for predicting energy grid load based on weather metrics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecast service
forecast_service = ForecastService(device='cpu')


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Starting up forecast service...")
    if not forecast_service.is_ready():
        print("Warning: Forecast service not ready. Some endpoints may not work.")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Prenergyze Energy Load Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    models_loaded = forecast_service.get_available_models()
    return HealthResponse(
        status="healthy" if forecast_service.is_ready() else "degraded",
        models_loaded=models_loaded,
        ensemble_ready=forecast_service.is_ready()
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["Models"])
async def get_models():
    """Get information about available models."""
    try:
        model_info = forecast_service.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(request: ForecastRequest):
    """
    Make energy load forecast based on weather data.
    
    Args:
        request: Forecast request with weather data and optional historical load
        
    Returns:
        Forecast response with predicted load values
    """
    try:
        # Convert Pydantic models to dictionaries
        weather_data = [item.dict() for item in request.weather_data]
        
        result = forecast_service.forecast(
            weather_data=weather_data,
            historical_load=request.historical_load,
            use_ensemble=request.use_ensemble,
            selected_models=request.selected_models
        )
        
        return ForecastResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/data/historical", tags=["Data"])
async def get_historical_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 1000
):
    """
    Get historical energy load and weather data.
    
    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        limit: Maximum number of records to return (default: 1000)
        
    Returns:
        JSON array of historical data records
    """
    try:
        # BASE_DIR is 'backend/', so we need to go up one level or use correct path
        # Check if we're in backend/api or backend/
        data_path = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
        
        # If that doesn't exist, try alternative path
        if not data_path.exists():
            # Try from backend/ directory
            alt_path = BASE_DIR.parent / 'backend' / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
            if alt_path.exists():
                data_path = alt_path
            else:
                # Try relative to current file
                current_file = Path(__file__).resolve()
                data_path = current_file.parent.parent.parent / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
        
        if not data_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Historical data file not found. Searched: {data_path}"
            )
        
        df = pd.read_csv(data_path, low_memory=False)
        
        # Check if 'date' column exists
        if 'date' not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"CSV file missing 'date' column. Available columns: {list(df.columns[:10])}"
            )
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Filter by date range if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Limit results
        if limit and limit > 0:
            df = df.tail(min(limit, len(df)))
        
        # Select key columns for visualization
        key_columns = [
            'date', 'load', 'temperature_2m', 'apparent_temperature',
            'relative_humidity_2m', 'pressure_msl', 'precipitation',
            'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in key_columns if col in df.columns]
        
        if not available_columns:
            raise HTTPException(
                status_code=500,
                detail=f"No matching columns found. Available columns: {list(df.columns[:10])}"
            )
        
        df_subset = df[available_columns].copy()
        
        # Convert to JSON-serializable format
        df_subset['date'] = df_subset['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Replace NaN with None for JSON serialization
        result = df_subset.where(pd.notnull(df_subset), None).to_dict('records')
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error loading historical data: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/data/correlation", tags=["Data"])
async def get_correlation():
    """
    Get correlation matrix between weather variables and load.
    
    Returns:
        JSON object with correlation matrix
    """
    try:
        data_path = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Historical data file not found")
        
        df = pd.read_csv(data_path)
        
        # Select numeric columns for correlation
        numeric_cols = [
            'load', 'temperature_2m', 'apparent_temperature',
            'relative_humidity_2m', 'vapour_pressure_deficit', 'pressure_msl',
            'precipitation', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
            'cloud_cover_high', 'et0_fao_evapotranspiration', 'sunshine_duration',
            'wind_speed_10m', 'wind_gusts_10m'
        ]
        
        # Only include columns that exist
        available_cols = [col for col in numeric_cols if col in df.columns]
        df_numeric = df[available_cols].select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = df_numeric.corr()
        
        # Convert to JSON-serializable format
        return {
            'columns': corr_matrix.columns.tolist(),
            'matrix': corr_matrix.values.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating correlation: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

