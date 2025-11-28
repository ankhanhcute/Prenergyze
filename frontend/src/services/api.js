import axios from 'axios';

// Use proxy path in development, or direct URL if specified
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Check API health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(`Health check failed: ${error.message}`);
  }
};

/**
 * Get model information
 */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch model info: ${error.message}`);
  }
};

/**
 * Get forecast prediction
 * @param {Array} weatherData - Array of weather data objects
 * @param {Object} options - Forecast options
 * @param {Array} options.historicalLoad - Optional historical load values
 * @param {boolean} options.useEnsemble - Whether to use ensemble
 * @param {Array} options.selectedModels - Optional array of selected model names
 */
export const getForecast = async (weatherData, options = {}) => {
  try {
    const requestData = {
      weather_data: weatherData,
      historical_load: options.historicalLoad || null,
      use_ensemble: options.useEnsemble !== undefined ? options.useEnsemble : true,
      selected_models: options.selectedModels || null,
    };

    const response = await api.post('/forecast', requestData);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || `Forecast failed: ${error.message}`);
    }
    throw new Error(`Forecast failed: ${error.message}`);
  }
};

/**
 * Get historical data
 */
export const getHistoricalData = async () => {
  try {
    const response = await api.get('/data/historical');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch historical data: ${error.message}`);
  }
};

/**
 * Get correlation matrix
 */
export const getCorrelation = async () => {
  try {
    const response = await api.get('/data/correlation');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch correlation data: ${error.message}`);
  }
};

export default api;

