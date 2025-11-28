import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getHistoricalData, getForecast } from '../../services/api';
import { processHistoricalData, prepareForecastChartData, formatDateForAPI } from '../../utils/dataProcessing';

const ForecastChart = ({ weatherData }) => {
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        setLoading(true);
        const data = await getHistoricalData();
        const processed = processHistoricalData(data);
        setHistoricalData(processed);
        setChartData(processed.map(item => ({ ...item, type: 'historical' })));
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHistorical();
  }, []);

  useEffect(() => {
    if (weatherData && weatherData.length > 0 && historicalData.length > 0) {
      generateForecast();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weatherData, historicalData]);

  const generateForecast = async () => {
    if (!weatherData || weatherData.length === 0) {
      return;
    }

    try {
      setForecastLoading(true);
      setError(null);

      // Get last few historical load values for context
      const lastHistoricalLoads = historicalData
        .slice(-24)
        .map(item => item.load)
        .filter(val => val > 0);

      // Filter out weather data points with missing critical values
      const validWeatherData = weatherData.filter(item => {
        // Check for critical missing values
        return item.temperature_2m !== null && 
               item.temperature_2m !== undefined &&
               !isNaN(item.temperature_2m) &&
               item.relative_humidity_2m !== null &&
               item.relative_humidity_2m !== undefined;
      });

      if (validWeatherData.length === 0) {
        throw new Error('Weather data contains too many missing values. Please fetch weather data again.');
      }

      const forecast = await getForecast(validWeatherData, {
        useEnsemble: true,
        historicalLoad: lastHistoricalLoads.length > 0 ? lastHistoricalLoads : null
      });

      // Validate forecast values - filter out unrealistic predictions
      if (forecast.forecast) {
        const validatedForecast = forecast.forecast.map(val => {
          const numVal = parseFloat(val);
          // Energy load should be positive and reasonable (between 0 and 50000 MW)
          if (numVal < 0) {
            console.warn(`Negative forecast value detected: ${numVal}. Clamping to 0.`);
            return 0;
          }
          if (numVal > 50000) {
            console.warn(`Unrealistically high forecast value: ${numVal}. Clamping to 50000.`);
            return 50000;
          }
          return numVal;
        });
        forecast.forecast = validatedForecast;
      }

      setForecastData(forecast);

      // Combine historical and forecast data
      const combined = prepareForecastChartData(historicalData, forecast, weatherData);
      setChartData(combined);
    } catch (err) {
      setError(err.message);
    } finally {
      setForecastLoading(false);
    }
  };

  const formatXAxis = (tickItem) => {
    if (typeof tickItem === 'string') {
      const date = new Date(tickItem);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
    const date = new Date(tickItem);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  if (loading) {
    return (
      <div className="card">
        <h2>Load Forecast</h2>
        <div className="loading">
          <div className="spinner"></div>
          Loading historical data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h2>Load Forecast</h2>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  // Prepare data with separate keys for historical and forecast
  const chartDataWithTypes = chartData.map(item => ({
    date: item.date,
    historicalLoad: item.type === 'historical' ? item.load : null,
    forecastLoad: item.type === 'forecast' ? item.load : null,
  }));

  return (
    <div className="card">
      <h2>Load Forecast</h2>
      <p style={{ marginBottom: '20px', color: '#666', fontSize: '14px' }}>
        Historical load data combined with ML model predictions based on weather forecasts.
      </p>

      {forecastLoading && (
        <div className="loading" style={{ marginBottom: '20px' }}>
          <div className="spinner"></div>
          Generating forecast...
        </div>
      )}

      {forecastData && (
        <div className="success" style={{ marginBottom: '20px' }}>
          <p><strong>Forecast Generated</strong></p>
          {forecastData.forecast && forecastData.forecast.length > 0 && (
            <>
              <p>
                Forecast Range: {Math.min(...forecastData.forecast).toFixed(2)} - {Math.max(...forecastData.forecast).toFixed(2)} MW
              </p>
              <p>
                Average Forecast: {(forecastData.forecast.reduce((a, b) => a + b, 0) / forecastData.forecast.length).toFixed(2)} MW
              </p>
              <p style={{ fontSize: '12px', marginTop: '5px' }}>
                {forecastData.forecast.length} data points forecasted
              </p>
            </>
          )}
          {forecastData.models_used && (
            <p style={{ fontSize: '12px', marginTop: '5px' }}>
              Models: {forecastData.models_used.join(', ')}
            </p>
          )}
        </div>
      )}

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartDataWithTypes}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            labelFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleString();
            }}
            formatter={(value, name) => {
              if (value === null) return null;
              return [`${value.toFixed(2)} MW`, name];
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="historicalLoad" 
            stroke="#667eea" 
            strokeWidth={2}
            dot={false}
            name="Historical Load"
            connectNulls={false}
          />
          <Line 
            type="monotone" 
            dataKey="forecastLoad" 
            stroke="#10b981" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Forecasted Load"
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {!weatherData && (
        <div style={{ marginTop: '15px', padding: '15px', background: '#fef3c7', borderRadius: '5px' }}>
          <p style={{ margin: 0, fontSize: '14px', color: '#92400e' }}>
            <strong>Note:</strong> Fetch weather forecast data using the selector above to generate predictions.
          </p>
        </div>
      )}
    </div>
  );
};

export default ForecastChart;

