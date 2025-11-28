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
        const historicalData = await getHistoricalData();
        const processed = processHistoricalData(historicalData);
        setData(processed);
        setFilteredData(processed);
        
        // Set default date range (last 30 days)
        if (processed.length > 0) {
          const end = new Date(processed[processed.length - 1].date);
          const start = new Date(end);
          start.setDate(start.getDate() - 30);
          setStartDate(formatDateForAPI(start));
          setEndDate(formatDateForAPI(end));
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
      setFilteredData(filtered);
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
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value) => [`${value.toFixed(2)} MW`, 'Load']}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="load" 
            stroke="#667eea" 
            strokeWidth={2}
            dot={false}
            name="Energy Load (MW)"
          />
        </LineChart>
      </ResponsiveContainer>
      
      <div style={{ marginTop: '15px', fontSize: '14px', color: '#666' }}>
        Showing {filteredData.length} data points
      </div>
    </div>
  );
};

export default HistoricalDataChart;

