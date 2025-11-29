/**
 * Data processing utilities for handling historical data, correlations, and chart formatting
 */

/**
 * Format date string for display
 */
export const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

/**
 * Format date for API requests (YYYY-MM-DD)
 */
export const formatDateForAPI = (date) => {
  if (typeof date === 'string') {
    return date.split('T')[0];
  }
  const d = new Date(date);
  return d.toISOString().split('T')[0];
};

/**
 * Process historical data for chart display
 */
export const processHistoricalData = (data) => {
  if (!data || !Array.isArray(data)) {
    return [];
  }

  return data.map(item => ({
    ...item,
    date: new Date(item.date),
    load: parseFloat(item.load) || 0,
    temperature_2m: parseFloat(item.temperature_2m) || 0,
    relative_humidity_2m: parseFloat(item.relative_humidity_2m) || 0,
  })).sort((a, b) => a.date - b.date);
};

/**
 * Process correlation matrix for heatmap display
 */
export const processCorrelationMatrix = (correlationData) => {
  if (!correlationData || !correlationData.matrix || !correlationData.columns) {
    return { data: [], columns: [] };
  }

  const { matrix, columns } = correlationData;
  const heatmapData = [];

  // Convert matrix to array of objects for Recharts heatmap
  for (let i = 0; i < columns.length; i++) {
    for (let j = 0; j < columns.length; j++) {
      heatmapData.push({
        x: columns[j],
        y: columns[i],
        value: matrix[i][j] || 0
      });
    }
  }

  return {
    data: heatmapData,
    columns: columns
  };
};

/**
 * Prepare forecast data for chart display
 * Uses actual weather forecast dates to align forecast properly
 */
export const prepareForecastChartData = (historicalData, forecastData, weatherForecast) => {
  const chartData = [];
  const now = new Date();

  // Add historical data (up to current time)
  if (historicalData && Array.isArray(historicalData)) {
    historicalData.forEach(item => {
      const itemDate = new Date(item.date);
      // Only include historical data up to current time
      if (itemDate <= now) {
        const loadValue = parseFloat(item.load);
        // Only add if load is a valid positive number
        if (!isNaN(loadValue) && loadValue > 0) {
          chartData.push({
            date: itemDate,
            load: loadValue,
            type: 'historical',
            temperature_2m: parseFloat(item.temperature_2m) || null,
            relative_humidity_2m: parseFloat(item.relative_humidity_2m) || null
          });
        }
      }
    });
  }

  // Add forecast data using weather forecast dates
  if (forecastData && Array.isArray(forecastData.forecast) && weatherForecast && Array.isArray(weatherForecast)) {
    forecastData.forecast.forEach((value, index) => {
      let forecastDate;
      
      // Use weather forecast date if available and it's in the future
      if (weatherForecast[index] && weatherForecast[index].date) {
        forecastDate = new Date(weatherForecast[index].date);
        // Only add if it's in the future
        if (forecastDate > now) {
          const loadValue = parseFloat(value);
          // Only add if load is a valid positive number
          if (!isNaN(loadValue) && loadValue > 0) {
            chartData.push({
              date: forecastDate,
              load: loadValue,
              type: 'forecast',
              temperature_2m: weatherForecast[index] ? (parseFloat(weatherForecast[index].temperature_2m) || null) : null,
              relative_humidity_2m: weatherForecast[index] ? (parseFloat(weatherForecast[index].relative_humidity_2m) || null) : null
            });
          }
        }
      } else {
        // Fallback: calculate from current time
        forecastDate = new Date(now);
        forecastDate.setHours(forecastDate.getHours() + index + 1);
        const loadValue = parseFloat(value);
        // Only add if load is a valid positive number
        if (!isNaN(loadValue) && loadValue > 0) {
          chartData.push({
            date: forecastDate,
            load: loadValue,
            type: 'forecast',
            temperature_2m: weatherForecast[index] ? (parseFloat(weatherForecast[index].temperature_2m) || null) : null,
            relative_humidity_2m: weatherForecast[index] ? (parseFloat(weatherForecast[index].relative_humidity_2m) || null) : null
          });
        }
      }
    });
  }

  return chartData.sort((a, b) => a.date - b.date);
};

/**
 * Get current time in ISO format
 */
export const getCurrentTime = () => {
  return new Date().toISOString();
};

/**
 * Get date N hours from now
 */
export const getDateNHoursFromNow = (hours) => {
  const date = new Date();
  date.setHours(date.getHours() + hours);
  return date;
};

/**
 * Calculate date range for filtering
 */
export const getDateRange = (data, days = 30) => {
  if (!data || data.length === 0) {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - days);
    return { start, end };
  }

  const dates = data.map(item => new Date(item.date)).sort((a, b) => b - a);
  const end = dates[0];
  const start = new Date(end);
  start.setDate(start.getDate() - days);

  return { start, end };
};

/**
 * Filter data by date range
 */
export const filterByDateRange = (data, startDate, endDate) => {
  if (!data || !Array.isArray(data)) {
    return [];
  }

  const start = new Date(startDate);
  const end = new Date(endDate);

  return data.filter(item => {
    const itemDate = new Date(item.date);
    return itemDate >= start && itemDate <= end;
  });
};

/**
 * Format number with commas
 */
export const formatNumber = (num) => {
  if (num === null || num === undefined) return 'N/A';
  return parseFloat(num).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
};

/**
 * Get color for correlation value
 */
export const getCorrelationColor = (value) => {
  const absValue = Math.abs(value);
  if (absValue >= 0.7) return '#ef4444'; // Strong correlation (red)
  if (absValue >= 0.4) return '#f59e0b'; // Moderate correlation (orange)
  if (absValue >= 0.2) return '#eab308'; // Weak correlation (yellow)
  return '#d1d5db'; // No correlation (gray)
};

