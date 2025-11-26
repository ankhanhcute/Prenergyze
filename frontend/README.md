# Prenergyze Frontend

Simple HTML/JavaScript frontend for testing the Energy Load Forecasting API.

## Features

- **API Health Check**: Monitor API status and loaded models
- **Model Information**: View available models and ensemble configuration
- **Forecast Interface**: Submit weather data and get load forecasts
- **Sample Data**: Quick-load button for testing with sample data
- **Responsive Design**: Works on desktop and mobile devices

## Usage

1. **Start the Backend API**:
   ```bash
   cd backend/api
   python app.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
   ```

2. **Open the Frontend**:
   - Simply open `frontend/index.html` in a web browser
   - Or serve it with a simple HTTP server:
     ```bash
     # Python 3
     python -m http.server 8080
     
     # Then navigate to http://localhost:8080
     ```

3. **Configure API URL** (if needed):
   - Default is `http://localhost:8000`
   - Change in the "API Base URL" field if your API runs on a different port/host

4. **Test the API**:
   - Click "Refresh Status" to check API health
   - Click "Load Sample Data" to fill in sample weather values
   - Click "Get Forecast" to generate a load prediction
   - View model information in the right panel

## API Endpoints Used

- `GET /health` - Check API health and loaded models
- `GET /models` - Get information about available models
- `POST /forecast` - Submit weather data and get load forecast

## Notes

- The frontend uses vanilla JavaScript (no dependencies)
- CORS is enabled on the backend for all origins (development only)
- The forecast returns load in MW (Megawatts)

