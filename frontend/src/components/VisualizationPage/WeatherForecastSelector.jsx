import { useState, useEffect } from 'react';
import { fetchWeatherForecast } from '../../services/meteoApi';
import { formatDateForAPI } from '../../utils/dataProcessing';

const WeatherForecastSelector = ({ onWeatherDataFetched }) => {
  const [lat, setLat] = useState('28.084358');
  const [lon, setLon] = useState('-82.372894');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [weatherData, setWeatherData] = useState(null);

  // Set default dates (today and 7 days from now)
  useEffect(() => {
    const today = new Date();
    const weekFromNow = new Date();
    weekFromNow.setDate(weekFromNow.getDate() + 7);
    setStartDate(formatDateForAPI(today));
    setEndDate(formatDateForAPI(weekFromNow));
  }, []);

  const handleFetchWeather = async () => {
    if (!startDate || !endDate) {
      setError('Please select start and end dates');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await fetchWeatherForecast(
        parseFloat(lat),
        parseFloat(lon),
        startDate,
        endDate
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
  };

  return (
    <div className="card">
      <h2>Weather Forecast Selector</h2>
      <p style={{ marginBottom: '20px', color: '#666', fontSize: '14px' }}>
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
        <div className="form-group">
          <label>Start Date:</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>End Date:</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
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

