import { useState, useEffect } from 'react';
import { getCorrelation } from '../../services/api';
import { getCorrelationColor } from '../../utils/dataProcessing';

const CorrelationHeatmap = () => {
  const [correlationData, setCorrelationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchCorrelation = async () => {
      try {
        setLoading(true);
        const data = await getCorrelation();
        setCorrelationData(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCorrelation();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <h2>Correlation Heatmap</h2>
        <div className="loading">
          <div className="spinner"></div>
          Loading correlation data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h2>Correlation Heatmap</h2>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  if (!correlationData || !correlationData.matrix || !correlationData.columns) {
    return (
      <div className="card">
        <h2>Correlation Heatmap</h2>
        <div className="error">No correlation data available</div>
      </div>
    );
  }

  const { matrix, columns } = correlationData;

  return (
    <div className="card">
      <h2>Correlation Heatmap</h2>
      <p style={{ marginBottom: '20px', color: '#666', fontSize: '14px' }}>
        Correlation coefficients between weather variables and energy load. 
        Values range from -1 (strong negative correlation) to +1 (strong positive correlation).
      </p>
      
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
          <thead>
            <tr>
              <th style={{ padding: '10px', textAlign: 'left', border: '1px solid #ddd', background: '#f3f4f6' }}>
                Variable
              </th>
              {columns.map((col) => (
                <th 
                  key={col}
                  style={{ 
                    padding: '10px', 
                    textAlign: 'center', 
                    border: '1px solid #ddd', 
                    background: '#f3f4f6',
                    minWidth: '80px'
                  }}
                >
                  {col.length > 15 ? col.substring(0, 15) + '...' : col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowCol, rowIndex) => (
              <tr key={rowCol}>
                <td 
                  style={{ 
                    padding: '10px', 
                    fontWeight: 'bold', 
                    border: '1px solid #ddd',
                    background: '#f9fafb'
                  }}
                >
                  {rowCol}
                </td>
                {columns.map((colCol, colIndex) => {
                  const value = matrix[rowIndex][colIndex];
                  const color = getCorrelationColor(value);
                  return (
                    <td
                      key={colCol}
                      style={{
                        padding: '10px',
                        textAlign: 'center',
                        border: '1px solid #ddd',
                        background: color,
                        color: Math.abs(value) > 0.5 ? 'white' : '#333',
                        fontWeight: Math.abs(value) > 0.7 ? 'bold' : 'normal'
                      }}
                      title={`${rowCol} vs ${colCol}: ${value.toFixed(3)}`}
                    >
                      {value.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <div style={{ width: '20px', height: '20px', background: '#ef4444', border: '1px solid #ddd' }}></div>
          <span style={{ fontSize: '12px' }}>Strong (|r| ≥ 0.7)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <div style={{ width: '20px', height: '20px', background: '#f59e0b', border: '1px solid #ddd' }}></div>
          <span style={{ fontSize: '12px' }}>Moderate (0.4 ≤ |r| &lt; 0.7)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <div style={{ width: '20px', height: '20px', background: '#eab308', border: '1px solid #ddd' }}></div>
          <span style={{ fontSize: '12px' }}>Weak (0.2 ≤ |r| &lt; 0.4)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <div style={{ width: '20px', height: '20px', background: '#d1d5db', border: '1px solid #ddd' }}></div>
          <span style={{ fontSize: '12px' }}>None (|r| &lt; 0.2)</span>
        </div>
      </div>
    </div>
  );
};

export default CorrelationHeatmap;

