import { useState, useEffect, useCallback } from 'react';
import { fetchWeatherForecast } from '../../services/meteoApi';

const WeatherForecastSelector = ({ onWeatherDataFetched, autoFetch = false }) => {
  const [lat, setLat] = useState('28.084358');
  const [lon, setLon] = useState('-82.372894');
  const [forecastHours, setForecastHours] = useState(24);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [weatherData, setWeatherData] = useState(null);

  const handleFetchWeather = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      // Fetch future weather forecast from current time
      const data = await fetchWeatherForecast(
        parseFloat(lat),
        parseFloat(lon),
        forecastHours,
        null // null means use current time
      );
      setWeatherData(data);
      if (onWeatherDataFetched) {
        onWeatherDataFetched(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [lat, lon, forecastHours, onWeatherDataFetched]);

  // Auto-fetch on mount if enabled
  useEffect(() => {
    if (autoFetch) {
      handleFetchWeather();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoFetch]); // Run only when autoFetch changes (effectively once on mount if true)

  return (
    <div className="card">
      <h2>Weather Forecast Selector</h2>
      <p style={{ marginBottom: '20px', color: '#9ca3af', fontSize: '14px' }}>
        Fetch forecasted weather data from Open-Meteo API to use for load predictions.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
        <div className="form-group">
          <label>Latitude:</label>
          <input
            type="number"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            step="0.000001"
          />
        </div>
        <div className="form-group">
          <label>Longitude:</label>
          <input
            type="number"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            step="0.000001"
          />
        </div>
      </div>

      <div className="form-group" style={{ marginBottom: '15px' }}>
        <label>Forecast Horizon (hours):</label>
        <select
          value={forecastHours}
          onChange={(e) => setForecastHours(parseInt(e.target.value))}
        >
          <option value={6}>6 hours</option>
          <option value={12}>12 hours</option>
          <option value={24}>24 hours (1 day)</option>
          <option value={48}>48 hours (2 days)</option>
          <option value={72}>72 hours (3 days)</option>
        </select>
        <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '8px' }}>
          Forecast will start from current time and predict the next {forecastHours} hours
        </p>
      </div>

      <button
        className="btn"
        onClick={handleFetchWeather}
        disabled={loading}
      >
        {loading ? 'Fetching...' : 'Fetch Weather Forecast'}
      </button>

      {error && (
        <div className="error" style={{ marginTop: '15px' }}>
          {error}
        </div>
      )}

      {weatherData && (
        <div className="success" style={{ marginTop: '15px' }}>
          <p>Successfully fetched {weatherData.length} weather data points</p>
          <p style={{ fontSize: '12px', marginTop: '5px' }}>
            Date range: {weatherData[0]?.date} to {weatherData[weatherData.length - 1]?.date}
          </p>
        </div>
      )}
    </div>
  );
};

export default WeatherForecastSelector;

