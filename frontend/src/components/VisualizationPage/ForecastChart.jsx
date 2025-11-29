import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { getRecentHistoricalData, getForecast } from '../../services/api';
import { processHistoricalData, prepareForecastChartData, getCurrentTime } from '../../utils/dataProcessing';

const ForecastChart = ({ weatherData }) => {
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedWeather, setSelectedWeather] = useState('none');

  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        setLoading(true);
        // Try to fetch recent historical data (last 168 hours = 7 days)
        let data;
        try {
          data = await getRecentHistoricalData(168);
        } catch (err) {
          // Fallback: use regular historical endpoint with date filter
          console.warn('Recent historical endpoint not available, using fallback:', err.message);
          const now = new Date();
          const sevenDaysAgo = new Date(now);
          sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
          
          const { getHistoricalData } = await import('../../services/api');
          const allData = await getHistoricalData();
          
          // Filter to last 7 days
          data = allData.filter(item => {
            const itemDate = new Date(item.date);
            return itemDate >= sevenDaysAgo && itemDate <= now;
          });
        }
        
        const processed = processHistoricalData(data);
        setHistoricalData(processed);
        // Filter to only show data up to current time
        const now = new Date();
        const recentData = processed
          .filter(item => new Date(item.date) <= now)
          .map(item => ({ ...item, type: 'historical' }));
        setChartData(recentData);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHistorical();
  }, []);

  const generateForecast = async () => {
    if (!weatherData || weatherData.length === 0) {
      return;
    }

    try {
      setForecastLoading(true);
      setError(null);

      // Filter out weather data points with missing critical values FIRST
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

      // Get recent historical load values for context (last 24-168 hours)
      // Models need sufficient context for lag features
      // CRITICAL: Ensure continuity - get data up to the forecast start time
      const now = new Date();
      
      // Get the first forecast date to ensure continuity
      const firstForecastDate = validWeatherData[0]?.date 
        ? new Date(validWeatherData[0].date) 
        : now;
      
      // Get historical data up to the forecast start (or current time, whichever is earlier)
      const cutoffTime = firstForecastDate < now ? firstForecastDate : now;
      
      const recentHistorical = historicalData
        .filter(item => {
          const itemDate = new Date(item.date);
          return itemDate <= cutoffTime;
        })
        .sort((a, b) => new Date(a.date) - new Date(b.date))
        .slice(-168); // Last 168 hours (7 days) for context
      
      // Ensure we have recent data (within last 24 hours) for continuity
      const mostRecentDate = recentHistorical.length > 0 
        ? new Date(recentHistorical[recentHistorical.length - 1].date)
        : null;
      
      const hoursSinceLastData = mostRecentDate 
        ? (cutoffTime - mostRecentDate) / (1000 * 60 * 60)
        : Infinity;
      
      if (hoursSinceLastData > 2) {
        console.warn(`Warning: Gap of ${hoursSinceLastData.toFixed(1)} hours between historical data and forecast start. This may cause forecast discontinuity.`);
      }
      
      const lastHistoricalLoads = recentHistorical
        .map(item => item.load)
        .filter(val => val > 0 && !isNaN(val));
      
      // If we have a small gap, repeat the last value to maintain continuity
      if (hoursSinceLastData > 0 && hoursSinceLastData <= 2 && lastHistoricalLoads.length > 0) {
        const lastValue = lastHistoricalLoads[lastHistoricalLoads.length - 1];
        const gapHours = Math.ceil(hoursSinceLastData);
        for (let i = 0; i < gapHours && i < 2; i++) {
          lastHistoricalLoads.push(lastValue);
        }
      }

      const forecast = await getForecast(validWeatherData, {
        useEnsemble: true,
        historicalLoad: lastHistoricalLoads.length > 0 ? lastHistoricalLoads : null
      });

      // Validate forecast values - filter out unrealistic predictions
      if (forecast.forecast) {
        // Get recent historical load for smoothing transition
        const recentLoads = lastHistoricalLoads.slice(-24); // Last 24 hours
        const avgRecentLoad = recentLoads.length > 0 
          ? recentLoads.reduce((a, b) => a + b, 0) / recentLoads.length 
          : null;
        const lastLoad = lastHistoricalLoads.length > 0 
          ? lastHistoricalLoads[lastHistoricalLoads.length - 1] 
          : null;
        
        // Calculate minimum load from historical data (10th percentile to prevent zeros)
        const historicalLoadsForMin = lastHistoricalLoads.filter(val => val > 0);
        let minLoad = 5000; // Default minimum
        if (historicalLoadsForMin.length > 0) {
          // Calculate 10th percentile
          const sorted = [...historicalLoadsForMin].sort((a, b) => a - b);
          const percentileIndex = Math.floor(sorted.length * 0.1);
          const percentile10 = sorted[percentileIndex] || sorted[0];
          minLoad = Math.max(1000, percentile10); // At least 1000 MW minimum
        }
        
        // First pass: validate and apply minimum/maximum constraints
        const initialForecast = forecast.forecast.map((val) => {
          let numVal = parseFloat(val);
          
          // Ensure minimum load (no zeros)
          if (numVal < minLoad) {
            console.warn(`Forecast value ${numVal} below minimum ${minLoad}. Setting to minimum.`);
            numVal = minLoad;
          }
          
          // Cap at reasonable maximum
          if (numVal > 50000) {
            console.warn(`Unrealistically high forecast value: ${numVal}. Clamping to 50000.`);
            numVal = 50000;
          }
          
          return numVal;
        });
        
        // Second pass: apply smoothing to maintain wave pattern
        // Using a loop instead of map to safely access previous values
        const validatedForecast = [];
        for (let index = 0; index < initialForecast.length; index++) {
          let numVal = initialForecast[index];
          
          // Apply smoothing to maintain wave pattern
          // For first value: smooth transition from historical
          if (index === 0 && lastLoad !== null && avgRecentLoad !== null) {
            const forecastDiff = Math.abs(numVal - lastLoad);
            
            // If forecast is very different from last load, smooth the transition
            if (forecastDiff > avgRecentLoad * 0.15) {
              // Blend with last load to smooth transition
              numVal = numVal * 0.6 + lastLoad * 0.4;
            }
          }
          // For subsequent values: apply exponential smoothing to maintain wave pattern
          else if (index > 0) {
            const prevVal = validatedForecast[index - 1];
            // Exponential smoothing: 70% current prediction, 30% previous value
            // This maintains continuity and wave pattern
            numVal = numVal * 0.7 + prevVal * 0.3;
          }
          
          validatedForecast.push(numVal);
        }
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

  useEffect(() => {
    if (weatherData && weatherData.length > 0 && historicalData.length > 0) {
      generateForecast();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weatherData, historicalData]);

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
    date: item.date.getTime(), // Convert to timestamp for chart
    dateObj: item.date, // Keep original for tooltip
    historicalLoad: item.type === 'historical' ? item.load : null,
    forecastLoad: item.type === 'forecast' ? item.load : null,
    temperature_2m: item.temperature_2m,
    relative_humidity_2m: item.relative_humidity_2m
  }));

  const now = new Date().getTime();

  // Check for weekend transition in forecast to provide insights
  const hasWeekendTransition = chartData.some((item, index) => {
    if (index === 0 || item.type !== 'forecast') return false;
    const prevDate = new Date(chartData[index-1].date);
    const currDate = new Date(item.date);
    // Check if moving from Friday (5) to Saturday (6)
    return prevDate.getDay() === 5 && currDate.getDay() === 6;
  });

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '15px' }}>
        <h2 style={{ margin: 0 }}>Load Forecast</h2>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <select 
            value={selectedWeather} 
            onChange={(e) => setSelectedWeather(e.target.value)}
            style={{ 
              padding: '8px 12px', 
              fontSize: '12px', 
              width: 'auto',
              background: 'rgba(31, 41, 55, 0.5)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '8px',
              color: '#e5e7eb',
              cursor: 'pointer'
            }}
          >
            <option value="none">Hide Weather</option>
            <option value="temperature_2m">Temperature (°C)</option>
            <option value="relative_humidity_2m">Humidity (%)</option>
          </select>
          <button
            className="btn btn-secondary"
            onClick={() => {
              if (weatherData && weatherData.length > 0) {
                generateForecast();
              }
            }}
            disabled={!weatherData || weatherData.length === 0 || forecastLoading}
            style={{ fontSize: '12px', padding: '8px 16px' }}
          >
            {forecastLoading ? 'Generating...' : 'Refresh Forecast'}
          </button>
        </div>
      </div>
      <p style={{ marginBottom: '20px', color: '#9ca3af', fontSize: '14px' }}>
        Recent historical load data (last 7 days) combined with ML model predictions for the next few hours based on weather forecasts.
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
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis 
            dataKey="date" 
            tickFormatter={(value) => {
              const date = new Date(value);
              return formatXAxis(date);
            }}
            angle={-45}
            textAnchor="end"
            height={80}
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
          />
          <YAxis 
            yAxisId="load"
            label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
          />
          {selectedWeather !== 'none' && (
            <YAxis 
              yAxisId="weather"
              orientation="right"
              label={{ 
                value: selectedWeather === 'temperature_2m' ? 'Temperature (°C)' : 'Humidity (%)', 
                angle: 90, 
                position: 'insideRight', 
                fill: '#f59e0b' 
              }}
              stroke="#f59e0b"
              tick={{ fill: '#f59e0b' }}
            />
          )}
          <Tooltip 
            contentStyle={{
              backgroundColor: 'rgba(17, 17, 19, 0.95)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb'
            }}
            labelFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleString();
            }}
            formatter={(value, name) => {
              if (value === null) return null;
              if (name === 'Temperature') return [`${value.toFixed(1)} °C`, name];
              if (name === 'Humidity') return [`${value.toFixed(1)} %`, name];
              return [`${value.toFixed(2)} MW`, name];
            }}
          />
          <Legend wrapperStyle={{ color: '#d1d5db' }} />
          <ReferenceLine 
            x={now} 
            yAxisId="load"
            stroke="#a78bfa" 
            strokeWidth={2}
            strokeDasharray="3 3"
            label={{ value: "Now", position: "topRight", fill: "#a78bfa" }}
          />
          <Line 
            yAxisId="load"
            type="monotone" 
            dataKey="historicalLoad" 
            stroke="#8b5cf6" 
            strokeWidth={2}
            dot={false}
            name="Historical Load"
            connectNulls={false}
          />
          <Line 
            yAxisId="load"
            type="monotone" 
            dataKey="forecastLoad" 
            stroke="#22c55e" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Forecasted Load"
            connectNulls={false}
          />
          {selectedWeather !== 'none' && (
            <Line 
              yAxisId="weather"
              type="monotone" 
              dataKey={selectedWeather} 
              stroke="#f59e0b" 
              strokeWidth={2}
              dot={false}
              name={selectedWeather === 'temperature_2m' ? 'Temperature' : 'Humidity'}
              connectNulls={true}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {!weatherData && (
        <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)', borderRadius: '12px' }}>
          <p style={{ margin: 0, fontSize: '14px', color: '#a78bfa' }}>
            <strong>Note:</strong> Fetch weather forecast data using the selector above to generate predictions for the next few hours.
          </p>
        </div>
      )}

      <div style={{ marginTop: '20px', fontSize: '12px', color: '#9ca3af' }}>
        <p style={{ margin: '8px 0' }}>
          <strong style={{ color: '#d1d5db' }}>Current Time:</strong> {new Date().toLocaleString()}
        </p>
        {weatherData && weatherData.length > 0 && (
          <p style={{ margin: '8px 0' }}>
            <strong style={{ color: '#d1d5db' }}>Forecast Horizon:</strong> Next {weatherData.length} hours
          </p>
        )}
        {hasWeekendTransition && (
          <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(59, 130, 246, 0.1)', borderLeft: '3px solid #3b82f6', borderRadius: '0 4px 4px 0' }}>
            <p style={{ margin: 0, color: '#93c5fd' }}>
              <strong>Insight:</strong> The forecast covers a transition to the weekend. Lower load predictions are expected as industrial and commercial energy demand typically decreases on weekends.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ForecastChart;

