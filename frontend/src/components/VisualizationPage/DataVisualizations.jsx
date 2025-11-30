import { useState, useEffect, useMemo } from 'react';
import { getHistoricalData, getCorrelation } from '../../services/api';
import { getCorrelationColor } from '../../utils/dataProcessing';
import { 
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Legend, ZAxis
} from 'recharts';

const DataVisualizations = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [historicalData, setHistoricalData] = useState([]);
  const [correlationData, setCorrelationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dynamicVisualizations, setDynamicVisualizations] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Request a large limit to cover the full range
        const [histData, corrData] = await Promise.all([
          getHistoricalData({ limit: 50000 }),
          getCorrelation()
        ]);
        
        // Downsample historical data for scatter plots to improve performance
        const downsampleRate = Math.ceil(histData.length / 2000); 
        
        const downsampled = histData.filter((_, index) => index % downsampleRate === 0).map(item => {
          const newItem = { ...item, dateObj: new Date(item.date) };
          // Parse floats
          Object.keys(item).forEach(key => {
            if (key !== 'date') {
              newItem[key] = parseFloat(item[key]) || 0;
            }
          });
          return newItem;
        });
        
        setHistoricalData(downsampled);
        setCorrelationData(corrData);
        setError(null);
      } catch (err) {
        console.error("Error fetching data:", err);
        setError("Failed to load visualization data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Dynamically generate visualizations based on data columns
  useEffect(() => {
    if (!historicalData.length || !correlationData) return;

    const columns = Object.keys(historicalData[0]);
    
    // Filter features: exclude metadata, lags, rolling means, and target
    const featureColumns = columns.filter(col => {
      if (['date', 'dateObj', 'load', 'time_idx', 'group_id'].includes(col)) return false;
      if (col.includes('_lag_')) return false;
      if (col.includes('_roll_')) return false;
      // Optional: exclude sin/cos time features if they are too technical/boring
      if (col.includes('_sin') || col.includes('_cos')) return false; 
      return true;
    });

    // Helper to get correlation value from the matrix if available
    const getCorrValue = (feature) => {
      if (!correlationData.columns || !correlationData.matrix) return null;
      const loadIdx = correlationData.columns.indexOf('load');
      const featureIdx = correlationData.columns.indexOf(feature);
      
      if (loadIdx !== -1 && featureIdx !== -1) {
        return correlationData.matrix[loadIdx][featureIdx];
      }
      return null; // Or calculate client-side if needed
    };

    const generatedViz = [
      // 1. Correlation Matrix (First Slide)
      {
        id: 'corr_matrix',
        title: "Feature Correlation Heatmap",
        insight: "Overview of correlations between all features and Energy Load.",
        type: 'matrix'
      }
    ];

    // 2. Add visualization for each feature
    featureColumns.forEach(feature => {
      const corr = getCorrValue(feature);
      const corrText = corr !== null ? `Correlation with Load: ${corr.toFixed(3)}` : "Correlation not available";
      
      let insightText = `Visualizing relationship between ${feature} and Energy Load. ${corrText}.`;
      
      // Add simple interpretation based on correlation
      if (corr !== null) {
        if (Math.abs(corr) > 0.7) insightText += " Strong relationship observed.";
        else if (Math.abs(corr) > 0.3) insightText += " Moderate relationship observed.";
        else insightText += " Weak or non-linear relationship.";
      }

      generatedViz.push({
        id: `viz_${feature}`,
        title: `${formatFeatureName(feature)} vs Load`,
        insight: insightText,
        feature: feature,
        type: 'scatter',
        corr: corr
      });
    });

    setDynamicVisualizations(generatedViz);

  }, [historicalData, correlationData]);


  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .replace('2m', '')
      .replace('10m', '')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  // Render Correlation Heatmap Table
  const renderCorrelationMatrix = () => {
    if (!correlationData || !correlationData.matrix || !correlationData.columns) {
      return <div className="error">No correlation data available</div>;
    }

    const { matrix, columns } = correlationData;
    // Filter for key variables to keep table readable or show all valid features
    // Let's show features that are in our dynamic list + load
    const validFeatures = dynamicVisualizations
      .filter(v => v.feature)
      .map(v => v.feature);
    
    // Also include load
    const indices = ['load', ...validFeatures]
      .map(v => columns.indexOf(v))
      .filter(i => i !== -1)
      .slice(0, 15); // Limit to 15 columns to prevent overflow if too many

    return (
      <div style={{ overflowX: 'auto', width: '100%', height: '100%' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
          <thead>
            <tr>
              <th style={{ padding: '4px', textAlign: 'left', background: 'rgba(31, 41, 55, 0.5)' }}>Var</th>
              {indices.map(i => (
                <th key={columns[i]} style={{ padding: '4px', textAlign: 'center', background: 'rgba(31, 41, 55, 0.5)' }}>
                  {formatFeatureName(columns[i])}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {indices.map(rowIndex => (
              <tr key={columns[rowIndex]}>
                <td style={{ padding: '4px', fontWeight: 'bold', background: 'rgba(31, 41, 55, 0.5)' }}>
                  {formatFeatureName(columns[rowIndex])}
                </td>
                {indices.map(colIndex => {
                  const value = matrix[rowIndex][colIndex];
                  return (
                    <td
                      key={columns[colIndex]}
                      style={{
                        padding: '4px',
                        textAlign: 'center',
                        background: getCorrelationColor(value),
                        color: Math.abs(value) > 0.5 ? 'white' : '#374151',
                        fontWeight: Math.abs(value) > 0.7 ? 'bold' : 'normal',
                        border: '1px solid rgba(255,255,255,0.1)'
                      }}
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
    );
  };

  const renderScatterPlot = (feature) => (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
        <XAxis 
          type="number" 
          dataKey={feature} 
          name={formatFeatureName(feature)} 
          stroke="#9ca3af" 
          label={{ value: formatFeatureName(feature), position: 'insideBottom', offset: -5, fill: '#9ca3af' }} 
        />
        <YAxis 
          type="number" 
          dataKey="load" 
          name="Load" 
          unit="MW" 
          stroke="#9ca3af" 
          label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} 
        />
        <Tooltip 
          cursor={{ strokeDasharray: '3 3' }} 
          contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} 
        />
        <Scatter name={`${formatFeatureName(feature)} vs Load`} data={historicalData} fill="#f59e0b" fillOpacity={0.6} />
      </ScatterChart>
    </ResponsiveContainer>
  );

  const nextSlide = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % dynamicVisualizations.length);
  };

  const prevSlide = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + dynamicVisualizations.length) % dynamicVisualizations.length);
  };

  if (loading) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="loading">
          <div className="spinner"></div>
          Loading visualizations data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!dynamicVisualizations.length) {
    return null;
  }

  const currentViz = dynamicVisualizations[currentIndex];

  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: '500px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h2 style={{ margin: 0 }}>Data Insights</h2>
        <span style={{ color: '#9ca3af', fontSize: '14px' }}>
          {currentIndex + 1} / {dynamicVisualizations.length}
        </span>
      </div>

      <div style={{ 
        flex: 1, 
        position: 'relative', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        overflow: 'hidden', 
        borderRadius: '12px', 
        background: 'rgba(17, 24, 39, 0.5)',
        padding: '20px',
        border: '1px solid rgba(255, 255, 255, 0.05)'
      }}>
        {/* Render current visualization component */}
        <div style={{ width: '100%', height: '300px' }}>
          {currentViz.type === 'matrix' ? renderCorrelationMatrix() : renderScatterPlot(currentViz.feature)}
        </div>
        
        <button 
          onClick={prevSlide}
          style={{
            position: 'absolute',
            left: '10px',
            top: '50%',
            transform: 'translateY(-50%)',
            background: 'rgba(0,0,0,0.6)',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10
          }}
        >
          &#8592;
        </button>
        
        <button 
          onClick={nextSlide}
          style={{
            position: 'absolute',
            right: '10px',
            top: '50%',
            transform: 'translateY(-50%)',
            background: 'rgba(0,0,0,0.6)',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10
          }}
        >
          &#8594;
        </button>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3 style={{ fontSize: '18px', marginBottom: '8px', color: '#e5e7eb' }}>{currentViz.title}</h3>
        <div style={{ 
          padding: '12px', 
          background: 'rgba(245, 158, 11, 0.1)', 
          borderLeft: '3px solid #f59e0b', 
          borderRadius: '0 8px 8px 0' 
        }}>
          <p style={{ margin: 0, fontSize: '14px', color: '#d1d5db', lineHeight: '1.5' }}>
            <strong>Insight:</strong> {currentViz.insight}
          </p>
        </div>
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '15px', flexWrap: 'wrap' }}>
        {dynamicVisualizations.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              border: 'none',
              background: index === currentIndex ? '#f59e0b' : 'rgba(255, 255, 255, 0.2)',
              cursor: 'pointer',
              padding: 0
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default DataVisualizations;
