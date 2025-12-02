import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

const DailyForecastChart = ({ forecastData }) => {
  const dailyData = useMemo(() => {
    if (!forecastData || !forecastData.forecast || !forecastData.forecast.length) {
      return [];
    }
    
    const now = new Date();
    const startDate = new Date(now);
    startDate.setHours(startDate.getHours() + 1, 0, 0, 0);

    const groupedData = {};

    forecastData.forecast.forEach((value, index) => {
      const date = new Date(startDate);
      date.setHours(startDate.getHours() + index);
      
      const dayKey = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
      
      if (!groupedData[dayKey]) {
        groupedData[dayKey] = {
          sum: 0,
          count: 0,
          min: Infinity,
          max: -Infinity,
          dateObj: date
        };
      }
      
      groupedData[dayKey].sum += value;
      groupedData[dayKey].count += 1;
      groupedData[dayKey].min = Math.min(groupedData[dayKey].min, value);
      groupedData[dayKey].max = Math.max(groupedData[dayKey].max, value);
    });

    return Object.keys(groupedData).map(key => {
      const item = groupedData[key];
      const avg = item.sum / item.count;
      
      return {
        day: key,
        averageLoad: parseFloat(avg.toFixed(2)),
        minLoad: parseFloat(item.min.toFixed(2)),
        maxLoad: parseFloat(item.max.toFixed(2)),
        fullDate: item.dateObj
      };
    });
  }, [forecastData]);

  if (!dailyData.length) {
    return null;
  }

  const allValues = dailyData.map(d => d.averageLoad);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  const yDomain = [Math.floor(minVal * 0.9), Math.ceil(maxVal * 1.05)];

  return (
    <div className="card">
      <h2 style={{ marginBottom: '20px' }}>7-Day Daily Average Forecast</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={dailyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.1)" vertical={false} />
          <XAxis 
            dataKey="day" 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af' }} 
          />
          <YAxis 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af' }} 
            domain={['auto', 'auto']} // Let Recharts handle the scaling dynamically
            label={{ value: 'Avg Load (MW)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
          />
          <Tooltip 
            cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
            contentStyle={{
              backgroundColor: 'rgba(17, 17, 19, 0.95)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '8px',
              color: '#e5e7eb'
            }}
            formatter={(value, name) => {
              if (name === 'averageLoad') return [`${value} MW`, 'Average Load'];
              if (name === 'minLoad') return [`${value} MW`, 'Min Load'];
              if (name === 'maxLoad') return [`${value} MW`, 'Max Load'];
              return [value, name];
            }}
          />
          <Legend wrapperStyle={{ color: '#d1d5db' }} />
          <Bar dataKey="averageLoad" name="Average Daily Load" radius={[4, 4, 0, 0]} barSize={40} fill="#FDE047">
            {dailyData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={`url(#colorGradient${index})`} />
            ))}
          </Bar>
          {/**/}
          <defs>
            {dailyData.map((entry, index) => (
              <linearGradient id={`colorGradient${index}`} x1="0" y1="0" x2="0" y2="1" key={`grad-${index}`}>
                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.9} />
                <stop offset="100%" stopColor="#d97706" stopOpacity={0.6} />
              </linearGradient>
            ))}
          </defs>
        </BarChart>
      </ResponsiveContainer>
      
      <div style={{ marginTop: '15px', display: 'flex', gap: '15px', overflowX: 'auto', paddingBottom: '10px', justifyContent: 'center' }}>
        {dailyData.map((day, idx) => (
          <div key={idx} style={{ 
            background: 'rgba(255, 255, 255, 0.05)', 
            padding: '10px', 
            borderRadius: '8px', 
            minWidth: '100px',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <div style={{ fontWeight: 'bold', color: '#f59e0b', marginBottom: '5px' }}>
              {day.fullDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            </div>
            <div style={{ fontSize: '12px', color: '#9ca3af' }}>Avg: <span style={{ color: '#e5e7eb' }}>{day.averageLoad}</span></div>
            <div style={{ fontSize: '12px', color: '#9ca3af' }}>Pk: <span style={{ color: '#e5e7eb' }}>{day.maxLoad}</span></div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DailyForecastChart;

