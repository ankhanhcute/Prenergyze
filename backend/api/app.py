"""
FastAPI application for energy load forecasting.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import sys
from pathlib import Path

# Add scripts to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
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
    model_info = forecast_service.get_model_info()
    return ModelInfoResponse(**model_info)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

