import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getHistoricalData } from '../../services/api';
import { processHistoricalData, formatDateForAPI, filterByDateRange } from '../../utils/dataProcessing';

const HistoricalDataChart = () => {
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const historicalData = await getHistoricalData({ limit: 50000 });
        const processed = processHistoricalData(historicalData);
        setData(processed);
        
        // Set default date range to the full available range initially
        // The user can then zoom in/filter
        if (processed.length > 0) {
          const end = new Date(processed[processed.length - 1].date);
          const start = new Date(processed[0].date); // Start from the beginning
          
          setStartDate(formatDateForAPI(start));
          setEndDate(formatDateForAPI(end));
          setFilteredData(processed); // Show all data by default or maybe last year?
          
          // Let's show the full range but maybe we should downsample for performance if too large?
          // Recharts can handle a few thousand points, but 17k might be heavy.
          // But user asked to "go back to the beginning".
          // Let's default to showing everything if it's not crazy large, or let them filter.
        }
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    if (startDate && endDate && data.length > 0) {
      const filtered = filterByDateRange(data, startDate, endDate);
      // Downsample if too many points to keep chart responsive
      if (filtered.length > 2000) {
        const rate = Math.ceil(filtered.length / 2000);
        setFilteredData(filtered.filter((_, i) => i % rate === 0));
      } else {
        setFilteredData(filtered);
      }
    } else {
      setFilteredData(data);
    }
  }, [startDate, endDate, data]);

  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  if (loading) {
    return (
      <div className="card">
        <h2>Historical Load Data</h2>
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
        <h2>Historical Load Data</h2>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Historical Load Data</h2>
      
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <div className="form-group" style={{ flex: '1', minWidth: '150px' }}>
          <label>Start Date:</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="form-group" style={{ flex: '1', minWidth: '150px' }}>
          <label>End Date:</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={filteredData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis}
            angle={-45}
            textAnchor="end"
            height={80}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
          />
          <YAxis 
            label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: 'rgba(17, 17, 19, 0.95)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb'
            }}
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value) => [`${value.toFixed(2)} MW`, 'Load']}
          />
          <Legend wrapperStyle={{ color: '#d1d5db' }} />
          <Line 
            type="monotone" 
            dataKey="load" 
            stroke="#8b5cf6" 
            strokeWidth={2}
            dot={false}
            name="Energy Load (MW)"
          />
        </LineChart>
      </ResponsiveContainer>
      
      <div style={{ marginTop: '20px', fontSize: '14px', color: '#9ca3af' }}>
        Showing {filteredData.length} data points
      </div>
    </div>
  );
};

export default HistoricalDataChart;

