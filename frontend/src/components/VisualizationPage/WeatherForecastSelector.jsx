import { useState, useEffect, useCallback } from 'react';
import { fetchWeatherForecast } from '../../services/meteoApi';

const LOCATIONS = [
  { name: 'Miami-Dade', lat: '25.5516', lon: '-80.6327' },
  { name: 'Broward', lat: '26.1901', lon: '-80.3659' },
  { name: 'Palm Beach', lat: '26.7056', lon: '-80.0364' }
];

const WeatherForecastSelector = ({ onWeatherDataFetched, autoFetch = false }) => {
  const [selectedLocation, setSelectedLocation] = useState(LOCATIONS[0]);
  const [lat, setLat] = useState(LOCATIONS[0].lat);
  const [lon, setLon] = useState(LOCATIONS[0].lon);
  const [forecastHours, setForecastHours] = useState(24);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [weatherData, setWeatherData] = useState(null);

  const handleLocationSelect = (location) => {
    setSelectedLocation(location);
    setLat(location.lat);
    setLon(location.lon);
  };

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
        Select a county to fetch forecasted weather data from Open-Meteo API for load predictions.
      </p>

      <div className="form-group" style={{ marginBottom: '20px' }}>
        <label style={{ marginBottom: '10px', display: 'block' }}>Select County:</label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
          {LOCATIONS.map((location) => (
            <button
              key={location.name}
              onClick={() => handleLocationSelect(location)}
              className="btn"
              style={{
                background: selectedLocation.name === location.name 
                  ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' 
                  : 'rgba(31, 41, 55, 0.5)',
                border: selectedLocation.name === location.name 
                  ? 'none' 
                  : '1px solid rgba(255, 255, 255, 0.1)',
                color: '#e5e7eb',
                opacity: 1,
                transform: 'none',
                boxShadow: selectedLocation.name === location.name 
                  ? '0 4px 12px rgba(245, 158, 11, 0.3)' 
                  : 'none'
              }}
            >
              {location.name}
            </button>
          ))}
        </div>
        <div style={{ marginTop: '10px', fontSize: '12px', color: '#9ca3af' }}>
          Selected Coordinates: {lat}° N, {lon}° W
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
          <option value={168}>168 hours (7 days)</option>
        </select>
        <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '8px' }}>
          Forecast will start from current time and predict the next {forecastHours} hours
        </p>
      </div>

      <button
        className="btn"
        onClick={handleFetchWeather}
        disabled={loading}
        style={{ width: '100%', marginTop: '10px' }}
      >
        {loading ? 'Fetching Forecast...' : `Get Forecast for ${selectedLocation.name}`}
      </button>

      {error && (
        <div className="error" style={{ marginTop: '15px' }}>
          {error}
        </div>
      )}

      {weatherData && (
        <div className="success" style={{ marginTop: '15px' }}>
          <p>Successfully fetched {weatherData.length} weather data points for {selectedLocation.name}</p>
          <p style={{ fontSize: '12px', marginTop: '5px' }}>
            Date range: {weatherData[0]?.date} to {weatherData[weatherData.length - 1]?.date}
          </p>
        </div>
      )}
    </div>
  );
};

export default WeatherForecastSelector;
